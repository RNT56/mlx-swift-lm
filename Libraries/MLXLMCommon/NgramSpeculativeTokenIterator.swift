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
/// Output is identical to a greedy ``TokenIterator`` for `temperature == 0`; for
/// `temperature > 0` it uses the standard speculative-sampling acceptance rule
/// (distribution-preserving).
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

        // Acceptance-gated proposal.
        var draft = [Int]()
        let gateOpen = roundsCompleted < emaWarmupRounds || acceptanceEMA >= emaFloor
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
        let mainLogits = result.logits
        let positions = draft.count + 1
        let verifyStart = verifyTokens.dim(0) - positions  // == 0

        // Sample the model's token at each verify position.
        var mainTokens = [Int]()
        var mainProcessed = [MLXArray]()
        if parameters.temperature == 0.0 && processor == nil {
            // Greedy fast path: argmax all positions in ONE eval/sync (vs qL+1).
            let slice = mainLogits[0, verifyStart..., 0...]  // [positions, vocab]
            mainTokens = slice.argMax(axis: -1).asArray(Int.self)
        } else if var verifyProcessor = processor {
            for i in 0 ..< positions {
                var logits = mainLogits[0..., verifyStart + i, 0...]
                logits = verifyProcessor.process(logits: logits)
                let token = sampler.sample(logits: logits)
                verifyProcessor.didSample(token: token)
                mainTokens.append(token.item(Int.self))
                mainProcessed.append(logits)
            }
        } else {
            for i in 0 ..< positions {
                let logits = mainLogits[0..., verifyStart + i, 0...]
                mainTokens.append(sampler.sample(logits: logits).item(Int.self))
                mainProcessed.append(logits)
            }
        }

        var accepted = 0
        if parameters.temperature == 0.0 {
            while accepted < draft.count && mainTokens[accepted] == draft[accepted] {
                commit(mainTokens[accepted])
                accepted += 1
            }
            commit(mainTokens[accepted])  // correction / bonus token
        } else {
            let temp = parameters.temperature
            var corrected = false
            while accepted < draft.count {
                let x = draft[accepted]
                let pTarget = MLX.softmax(mainProcessed[accepted] / temp, axis: -1)
                let pTargetX = (pTarget.ndim == 2 ? pTarget[0, x] : pTarget[x]).item(Float.self)
                // Draft is deterministic (prompt-lookup) -> q(x)=1, accept prob = pTarget(x).
                if Float.random(in: 0 ..< 1) < pTargetX {
                    commit(x)
                    accepted += 1
                } else {
                    commit(mainTokens[accepted])  // resample from target (q is a point mass)
                    corrected = true
                    break
                }
            }
            if !corrected && accepted == draft.count { commit(mainTokens[accepted]) }
        }

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
