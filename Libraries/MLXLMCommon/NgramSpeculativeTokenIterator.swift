// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Draft-model-free speculative decoder (lever ①).
///
/// Proposes the next tokens with a CPU ``PromptLookupSpeculator`` (prompt-lookup /
/// n-gram), verifies them in ONE multi-query forward over the shared KV cache, and
/// accepts the matching prefix — rolling back the rejected tail with
/// ``trimPromptCache(_:numTokens:)``. One forward reads the weight stream once and
/// emits `accepted + 1` tokens, so it amortizes the dominant decode cost on
/// repetitive / structured / long-context workloads, with **no draft model, no
/// training, and bit-exact output** (the verify makes proposals correctness-neutral).
///
/// An acceptance EMA self-disables speculation when recent rounds accept little, so
/// free-form generation degrades to ordinary single-token decode (no regression).
///
/// Speculation runs ONLY for exact greedy decode with no logit processor, where the
/// multi-query verify makes the deterministic n-gram draft correctness-neutral, so the
/// output is byte-identical to a greedy ``TokenIterator``. For `temperature > 0` or an
/// active logit processor it falls back to exact single-token decode (no speculation):
/// distribution-correct speculative sampling for a point-mass draft requires resampling
/// the *residual* `(p - q)+` and carrying processor state across accepted tokens, which
/// is not yet validated, so those paths fail closed to the exact decoder.
public struct NgramSpeculativeTokenIterator: TokenIteratorProtocol {

    var y: LMInput.Text
    let model: any LanguageModel
    var cache: [KVCache]
    var state: LMOutput.State?

    var processor: LogitProcessor?
    let sampler: LogitSampler
    let parameters: GenerateParameters

    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0

    private var speculator: PromptLookupSpeculator
    private let maxProposalTokens: Int

    // Acceptance EMA self-disable: skip proposing when recent acceptance is poor.
    private var acceptanceEMA: Double = 1.0
    private let emaAlpha: Double
    private let emaFloor: Double
    private let emaWarmupRounds: Int
    private var roundsCompleted = 0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var runtimeError: Error?
    private var requiresSynchronousGenerationEval = false

    let quantizeKVCache: (inout [KVCache]) -> Void

    // Metrics (acceptance gate + speed proxy).
    public private(set) var acceptedDraftTokens = 0
    public private(set) var totalDraftTokens = 0
    public private(set) var speculativeRounds = 0
    public private(set) var modelForwards = 0
    public var lastRuntimeError: Error? { runtimeError }
    public var acceptanceRate: Double {
        totalDraftTokens > 0 ? Double(acceptedDraftTokens) / Double(totalDraftTokens) : 0
    }

    public init(
        input: LMInput,
        model: any LanguageModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        ngram: Int = 3,
        maxProposalTokens: Int = 4,
        emaAlpha: Double = 0.2,
        // Self-disable floor on accepted/proposed. The verify forward costs more
        // than a single decode (qL>1), so speculation only pays above a break-even
        // acceptance. That break-even is lower on larger / longer-context (weight-
        // dominated) models and higher on tiny models — an on-device tuning constant.
        // 0.5 protects free-form text from regressing on small models while keeping
        // the wins on structured/repetitive workloads.
        emaFloor: Double = 0.5,
        emaWarmupRounds: Int = 8
    ) throws {
        self.model = model
        self.y = input.text
        let runtime = try resolvedGenerationParameters(for: model, parameters: parameters)
        self.cache = cache ?? model.newCache(parameters: runtime)
        self.sampler = runtime.sampler()
        self.processor = runtime.processor()
        self.parameters = runtime
        self.maxTokens = runtime.maxTokens
        self.maxProposalTokens = Swift.max(1, maxProposalTokens)
        self.speculator = PromptLookupSpeculator(ngram: ngram, maxProposalTokens: self.maxProposalTokens)
        self.emaAlpha = emaAlpha
        self.emaFloor = emaFloor
        self.emaWarmupRounds = emaWarmupRounds

        let resolved = runtime
        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache, kvBits: resolved.kvBits, kvGroupSize: resolved.kvGroupSize,
                quantizedKVStart: resolved.quantizedKVStart, kvCacheStrategy: resolved.kvCacheStrategy,
                kvCodec: resolved.kvCodec, turboQuantPreset: resolved.turboQuantPreset,
                turboQuantBackend: resolved.turboQuantBackend,
                turboQuantOptimizationPolicy: resolved.turboQuantOptimizationPolicy,
                turboQuantFallbackPolicy: resolved.turboQuantFallbackPolicy,
                turboQuantSeed: resolved.turboQuantSeed,
                turboQuantValueBits: resolved.turboQuantValueBits,
                turboQuantPrecisionPolicy: resolved.effectiveTurboQuantPrecisionPolicy,
                turboQuantValueGroupSize: resolved.turboQuantValueGroupSize,
                turboQuantSparseValuePolicy: resolved.effectiveTurboQuantSparseValuePolicy,
                turboQuantSparseValueSelection: resolved.effectiveTurboQuantSparseValueSelection,
                turboQuantResidentBudgetBytes: resolved.turboQuantPerCacheResidentBudgetBytes,
                spillMemoryWatermarkBytes: resolved.spillMemoryWatermarkBytes,
                kvLayerPolicy: resolved.kvLayerPolicy)
        }

        guard canTrimPromptCache(self.cache) else {
            throw KVCacheError(message: "n-gram speculative decoding requires trimmable KV caches.")
        }
        let prefillStart = Date.timeIntervalSinceReferenceDate
        try prepare(input: input, windowSize: runtime.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    private mutating func forward(_ tokens: MLXArray) -> LMOutput? {
        let input = LMInput.Text(tokens: tokens)
        let result: LMOutput
        if let throwingModel = model as? any ThrowingLanguageModel {
            do {
                result = try throwingModel.callAsFunctionThrowing(
                    input[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
            } catch {
                runtimeError = error
                return nil
            }
        } else {
            result = model(input[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        }
        state = result.state
        return result
    }

    private mutating func commit(_ token: Int) {
        pendingTokens.append(token)
        speculator.append(token)
    }

    private mutating func syncRecurrentIfNeeded() {
        if materializeRecurrentKVCacheState(cache) { requiresSynchronousGenerationEval = true }
    }

    private mutating func prepare(input: LMInput, windowSize: Int?) throws {
        processor?.prompt(input.text.tokens)
        // Seed the speculator with the full prompt token stream.
        speculator.append(contentsOf: input.text.tokens.asArray(Int.self))

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let remaining):
            // Chunked prefill remainder: process it, producing the first token.
            y = remaining
            guard let result = forward(remaining.tokens) else { return }
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            commit(token.item(Int.self))
            y = .init(tokens: token)
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            state = result.state
            commit(token.item(Int.self))
            y = .init(tokens: token)
        }
        quantizeKVCache(&cache)
        syncRecurrentIfNeeded()
    }

    private mutating func singleTokenStep() {
        guard let result = forward(y.tokens) else { return }
        modelForwards += 1
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        commit(token.item(Int.self))
        y = .init(tokens: token)
        quantizeKVCache(&cache)
        syncRecurrentIfNeeded()
    }

    mutating func speculateRound() {
        guard runtimeError == nil else { return }
        let remaining = maxTokens.map { $0 - tokenCount } ?? maxProposalTokens
        guard remaining > 0 else { return }

        // Speculation is exact ONLY for greedy decode with no logit processor: the
        // multi-query verify makes the deterministic n-gram draft correctness-neutral
        // via argmax. For temperature > 0 or an active processor, distribution-correct
        // speculative sampling (residual when the draft is a point mass) and processor-
        // state carry are not yet validated, so fall back to exact single-token decode.
        let speculationExact = parameters.temperature == 0 && processor == nil
        // Acceptance-gated proposal.
        var draft = [Int]()
        let gateOpen = speculationExact
            && (roundsCompleted < emaWarmupRounds || acceptanceEMA >= emaFloor)
        if gateOpen {
            let proposed = speculator.propose()
            let budget = Swift.min(maxProposalTokens, Swift.max(0, remaining - 1))
            if !proposed.isEmpty && budget > 0 {
                draft = Array(proposed.prefix(budget))
            }
        }

        if draft.isEmpty {
            singleTokenStep()
            return
        }

        // One multi-query verify forward over [y] + draft.
        let draftArray = MLXArray(draft.map { Int32($0) })
        let verifyTokens = concatenated([y.tokens, draftArray])
        guard let result = forward(verifyTokens) else { return }
        modelForwards += 1
        let positions = draft.count + 1
        let verifyStart = verifyTokens.dim(0) - positions  // == 0

        // Greedy fast path only — the gate guarantees we reach here exclusively for
        // temperature == 0 with no logit processor, where the deterministic n-gram draft
        // is correctness-neutral. Argmax all positions in ONE eval/sync (vs qL+1).
        let slice = result.logits[0, verifyStart..., 0...]  // [positions, vocab]
        let mainTokens = slice.argMax(axis: -1).asArray(Int.self)

        var accepted = 0
        while accepted < draft.count && mainTokens[accepted] == draft[accepted] {
            commit(mainTokens[accepted])
            accepted += 1
        }
        commit(mainTokens[accepted])  // correction / bonus token

        // Roll back rejected drafts from the cache.
        let rejected = draft.count - accepted
        if rejected > 0 { _ = trimPromptCache(cache, numTokens: rejected) }

        // The last committed token becomes the next round's input seed.
        if let last = pendingTokens.last { y = .init(tokens: MLXArray([Int32(last)])) }

        acceptedDraftTokens += accepted
        totalDraftTokens += draft.count
        speculativeRounds += 1
        let ratio = draft.isEmpty ? 1.0 : Double(accepted) / Double(draft.count)
        acceptanceEMA = emaAlpha * ratio + (1 - emaAlpha) * acceptanceEMA
        roundsCompleted += 1

        quantizeKVCache(&cache)
        syncRecurrentIfNeeded()
    }

    mutating public func next() -> Int? {
        guard runtimeError == nil else { return nil }
        if let maxTokens, tokenCount >= maxTokens { return nil }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()
        guard runtimeError == nil, !pendingTokens.isEmpty else { return nil }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}
