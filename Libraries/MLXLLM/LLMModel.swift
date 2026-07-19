// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon

private func turboQuantPrefillSynchronizationInterval(
    promptLength: Int,
    stepSize: Int
) -> Int? {
    let environment = ProcessInfo.processInfo.environment
    let explicit = environment["TQ_PREFILL_SYNC_INTERVAL"]
        ?? environment["TURBOQUANT_PREFILL_SYNC_INTERVAL"]
    if let explicit, let value = Int(explicit.trimmingCharacters(in: .whitespaces)) {
        return value > 0 ? value : nil
    }

    guard promptLength >= 65_536 else { return nil }
    let targetTokensPerCommandBuffer = 8_192
    return max(1, targetTokensPerCommandBuffer / max(1, stepSize))
}

private func turboQuantFlushPrefillCommandBuffer(clearCache: Bool = false) {
    Stream.gpu.synchronize()
    if clearCache {
        Memory.clearCache()
    }
}

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text
        let prefillSyncInterval = turboQuantPrefillSynchronizationInterval(
            promptLength: y.tokens.size,
            stepSize: prefillStepSize
        )
        var chunksSinceSync = 0

        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            // Prepare the prompt in chunks if larger than the prefill size.
            // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
            // chunk N.
            var state: LMOutput.State?
            while y.tokens.size > prefillStepSize {
                let input = y[.newAxis, ..<prefillStepSize]
                let output: LMOutput
                if let throwingModel = self as? any ThrowingLanguageModel {
                    output = try throwingModel.callAsFunctionThrowing(
                        input,
                        cache: cache.isEmpty ? nil : cache,
                        state: state
                    )
                } else {
                    output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
                }
                state = output.state
                asyncEval(cache)
                chunksSinceSync += 1
                if let prefillSyncInterval,
                    chunksSinceSync % prefillSyncInterval == 0
                {
                    turboQuantFlushPrefillCommandBuffer(clearCache: chunksSinceSync > 0)
                }
                y = y[prefillStepSize...]
            }

            // Single sync after the loop to flush any remaining async work.
            eval(cache)
            if prefillSyncInterval != nil {
                Stream.gpu.synchronize()
            }
        }

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
