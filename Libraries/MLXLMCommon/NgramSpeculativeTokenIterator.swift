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
    private let adaptiveK: Bool
    private let cheapProposalTokens: Int

    /// Per-round proposal width. With adaptive-k off, the fixed `maxProposalTokens`.
    /// With it on: explore at full width during EMA warmup, then scale the width by
    /// the acceptance EMA between `cheapProposalTokens` (below the lm_head verify
    /// cliff) and `maxProposalTokens` — high acceptance earns a wider (costlier)
    /// verify; marginal acceptance stays at the near-free cheap width.
    private func currentProposalCap() -> Int {
        guard adaptiveK, roundsCompleted >= emaWarmupRounds else { return maxProposalTokens }
        // Full width when acceptance is high (wide rounds pay off — and at the long,
        // weight-dominated contexts where the lm_head verify cliff bites, acceptance
        // IS high so we want full width); shrink to the cheap width (below the cliff)
        // only when acceptance is genuinely low and a wide verify would be wasted.
        return acceptanceEMA >= 0.7 ? maxProposalTokens : cheapProposalTokens
    }

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

    // N7 optimistic prefetch (opt-in; default off keeps the validated synchronous path).
    // While the CPU reads + accepts round R, the GPU is already computing round R+1
    // (issued on a full-acceptance assumption: R+1's seed = R's argmax-at-last kept as a
    // LAZY MLXArray, R+1's draft proposed from history+R's drafts). On a misprediction the
    // R+1 forward ran on a wrong cache state, so its KV appends + R's rejects are trimmed
    // and the speculator is truncated back. Greedy / no-processor only.
    private let enablePrefetch: Bool
    private struct PendingVerify {
        var logits: MLXArray  // [1, draft.count + 1, vocab], lazy/asyncEval'd
        var draft: [Int]
        var specMark: Int  // speculator.count BEFORE this round's draft was optimistically appended
    }
    private var inflight: PendingVerify?

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
        emaWarmupRounds: Int = 8,
        // N7: opt-in async prefetch pipeline. Default off — the synchronous path is the
        // validated one; prefetch is a long-context (≥~16K) throughput lever.
        enablePrefetch: Bool = false,
        // Adaptive-k (free lever): the §1 qmm probe showed the lm_head verify cost
        // cliffs sharply above the cheap width (T(2)≈0.93–1.1× vs T(4)≈2.5× on the
        // 194 MB head), so a verify forward at the cheap width is nearly free today
        // while wider rounds pay the head fall-off. When enabled, the per-round
        // proposal width tracks the acceptance EMA — default to `cheapProposalTokens`
        // (below the cliff) and widen toward `maxProposalTokens` only when recent
        // acceptance is high enough to justify the wider (more expensive) verify.
        // Bit-exact: only the proposal WIDTH changes; the greedy-argmax verify is
        // unchanged, so output stays byte-identical. A zero-kernel baseline that
        // §5a / §1-batched-qmv must beat. Default off (validated path untouched).
        adaptiveK: Bool = false,
        cheapProposalTokens: Int = 2
    ) throws {
        self.model = model
        self.y = input.text
        self.enablePrefetch = enablePrefetch
        self.adaptiveK = adaptiveK
        self.cheapProposalTokens = Swift.max(1, Swift.min(cheapProposalTokens, maxProposalTokens))
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
                kvLayerPolicy: resolved.kvLayerPolicy,
                kvScheme: resolved.kvScheme)
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
            let budget = Swift.min(currentProposalCap(), Swift.max(0, remaining - 1))
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

    // ----- N7 optimistic prefetch (opt-in) -----

    private mutating func recordAccept(accepted: Int, drafted: Int) {
        acceptedDraftTokens += accepted
        totalDraftTokens += drafted
        speculativeRounds += 1
        let ratio = drafted > 0 ? Double(accepted) / Double(drafted) : 1.0
        acceptanceEMA = emaAlpha * ratio + (1 - emaAlpha) * acceptanceEMA
        roundsCompleted += 1
    }

    /// Build + issue (asyncEval) one verify forward of `[seed] + draft`, where the draft
    /// is chosen from the speculator under the EMA gate and `remaining` budget. Mutates
    /// the cache by `1 + draft.count`. Returns nil if no draft is proposed (caller decodes
    /// a single token instead). Does NOT touch the speculator (the draft is tentative).
    private mutating func issueVerify(seed: MLXArray, remaining: Int) -> PendingVerify? {
        let gateOpen = roundsCompleted < emaWarmupRounds || acceptanceEMA >= emaFloor
        guard gateOpen else { return nil }
        let proposed = speculator.propose()
        let budget = Swift.min(currentProposalCap(), Swift.max(0, remaining - 1))
        guard !proposed.isEmpty, budget > 0 else { return nil }
        let draft = Array(proposed.prefix(budget))
        let verifyTokens = concatenated([seed, MLXArray(draft.map { Int32($0) })])
        guard let result = forward(verifyTokens) else { return nil }
        modelForwards += 1
        asyncEval(result.logits)
        return PendingVerify(logits: result.logits, draft: draft, specMark: speculator.count)
    }

    /// One pipelined speculative round: consume the in-flight verify (round R) while the
    /// GPU computes round R+1 (issued optimistically on a full-acceptance assumption).
    /// Bit-exact for greedy / no-processor (gated by the caller); rolls back R+1 + R's
    /// rejects + the speculator on a misprediction.
    mutating func prefetchRound() {
        guard runtimeError == nil else { return }
        // Recurrent caches require synchronous eval — fall back to the safe path.
        if requiresSynchronousGenerationEval { speculateRound(); return }

        let remaining = maxTokens.map { $0 - tokenCount } ?? maxProposalTokens
        guard remaining > 0 else { return }

        if inflight == nil {
            guard let f = issueVerify(seed: y.tokens, remaining: remaining) else {
                singleTokenStep()
                return
            }
            inflight = f
        }
        let cur = inflight!
        let curLast = cur.draft.count  // bonus position (argmax at the last verify slot)

        // Optimistically pre-issue round R+1 assuming cur fully accepts: its seed is cur's
        // argmax-at-last kept LAZY (no readback), its draft proposed from history+cur.draft.
        var nextInflight: PendingVerify?
        let remainingAfterCur = remaining - (cur.draft.count + 1)
        if remainingAfterCur > 0 {
            let nextSeedLazy = cur.logits[0, curLast, 0...].argMax(axis: -1).reshaped([1])
            let specMarkNext = speculator.count
            speculator.append(contentsOf: cur.draft)  // optimistic (full-accept); rolled back on mispredict
            let gateOpen = roundsCompleted < emaWarmupRounds || acceptanceEMA >= emaFloor
            var nextDraft = [Int]()
            if gateOpen {
                // propose() over history+cur.draft returns [predicted_bonus, e1, e2, ...].
                // R+1's seed is the MODEL's bonus (nextSeedLazy), so R+1's draft must start
                // AFTER the bonus — drop the proposal's first token (the predicted bonus).
                let proposed = speculator.propose().dropFirst()
                let budget = Swift.min(maxProposalTokens, Swift.max(0, remainingAfterCur - 1))
                if !proposed.isEmpty, budget > 0 { nextDraft = Array(proposed.prefix(budget)) }
            }
            if !nextDraft.isEmpty {
                let verifyTokens = concatenated([nextSeedLazy, MLXArray(nextDraft.map { Int32($0) })])
                if let result = forward(verifyTokens) {
                    modelForwards += 1
                    asyncEval(result.logits)
                    nextInflight = PendingVerify(
                        logits: result.logits, draft: nextDraft, specMark: specMarkNext)
                } else {
                    speculator.truncate(to: specMarkNext)  // undo optimistic append on error
                }
            } else {
                speculator.truncate(to: specMarkNext)  // not issuing next: undo optimistic append
            }
        }

        // Read cur's argmax (blocks; the GPU is busy on R+1 if issued).
        let positions = cur.draft.count + 1
        let mainTokens = cur.logits[0, 0 ..< positions, 0...].argMax(axis: -1).asArray(Int.self)
        var accepted = 0
        while accepted < cur.draft.count && mainTokens[accepted] == cur.draft[accepted] {
            accepted += 1
        }
        let bonus = mainTokens[accepted]

        if accepted == cur.draft.count, let next = nextInflight {
            // FULL accept and R+1 was issued on the correct assumption — keep both.
            for i in 0 ..< accepted { pendingTokens.append(mainTokens[i]) }
            pendingTokens.append(bonus)
            speculator.append(bonus)  // speculator already holds cur.draft (optimistic append)
            inflight = next  // bonus == next's lazy seed
            y = .init(tokens: MLXArray([Int32(bonus)]))
            recordAccept(accepted: accepted, drafted: cur.draft.count)
        } else {
            // Misprediction (or no R+1 issued): discard R+1 and commit cur synchronously.
            if let next = nextInflight {
                _ = trimPromptCache(cache, numTokens: next.draft.count + 1)  // R+1's KV appends
                speculator.truncate(to: next.specMark)  // remove the optimistic cur.draft append
            }
            for i in 0 ..< accepted { commit(mainTokens[i]) }
            commit(bonus)  // correction / bonus
            let rejected = cur.draft.count - accepted
            if rejected > 0 { _ = trimPromptCache(cache, numTokens: rejected) }
            inflight = nil
            y = .init(tokens: MLXArray([Int32(bonus)]))
            recordAccept(accepted: accepted, drafted: cur.draft.count)
        }

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
        // Prefetch is exact only for greedy / no-processor (same gate as speculation).
        if enablePrefetch && parameters.temperature == 0 && processor == nil {
            prefetchRound()
        } else {
            speculateRound()
        }
        guard runtimeError == nil, !pendingTokens.isEmpty else { return nil }
        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}
