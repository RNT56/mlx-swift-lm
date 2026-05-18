// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX

/// Experimental model contract for DFlash speculative decoding.
///
/// No existing model conforms to this protocol in Wave 4. It is intentionally isolated so future
/// target-model adapters can opt in with small, reviewable changes.
public protocol DFlashTargetModel: LanguageModel {
    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray

    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray

    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray])

    var dflashIsHybridGDN: Bool { get }
    var dflashUseTapeRollback: Bool { get }
}

public extension DFlashTargetModel {
    var dflashUseTapeRollback: Bool { true }
}

public enum DFlashEvent: Sendable {
    case prefill(promptTokenCount: Int, prefillUs: Double)
    case prefillProgress(tokensProcessed: Int, tokensTotal: Int)
    case token(tokenID: Int, generatedTokens: Int, acceptanceRatio: Double, cyclesCompleted: Int)
    case summary(DFlashSummary)
}

public struct DFlashSummary: Sendable {
    public struct PhaseTimings: Sendable {
        public let prefill: Double
        public let draft: Double
        public let verify: Double
        public let replay: Double

        public init(prefill: Double, draft: Double, verify: Double, replay: Double) {
            self.prefill = prefill
            self.draft = draft
            self.verify = verify
            self.replay = replay
        }
    }

    public let elapsedUs: Double
    public let promptTokenCount: Int
    public let generatedTokenIDs: [Int]
    public let acceptedFromDraft: Int
    public let acceptanceRatio: Double
    public let blockTokens: Int
    public let cyclesCompleted: Int
    public let phaseTimingsUs: PhaseTimings

    public init(
        elapsedUs: Double,
        promptTokenCount: Int,
        generatedTokenIDs: [Int],
        acceptedFromDraft: Int,
        acceptanceRatio: Double,
        blockTokens: Int,
        cyclesCompleted: Int,
        phaseTimingsUs: PhaseTimings
    ) {
        self.elapsedUs = elapsedUs
        self.promptTokenCount = promptTokenCount
        self.generatedTokenIDs = generatedTokenIDs
        self.acceptedFromDraft = acceptedFromDraft
        self.acceptanceRatio = acceptanceRatio
        self.blockTokens = blockTokens
        self.cyclesCompleted = cyclesCompleted
        self.phaseTimingsUs = phaseTimingsUs
    }

    public var generationTokens: Int { generatedTokenIDs.count }

    public var tokensPerSecond: Double {
        let generationUs = elapsedUs - phaseTimingsUs.prefill
        return generationUs > 0 ? Double(generationTokens) / (generationUs / 1_000_000.0) : 0
    }
}

public protocol DFlashEngine: Sendable {
    func armRollback(targetCache: [KVCache], prefixLen: Int)

    func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int
}

public final class FullAttentionDFlashEngine: DFlashEngine, @unchecked Sendable {
    public init() {}

    public func armRollback(targetCache: [KVCache], prefixLen: Int) {}

    public func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        DFlashRuntime.restoreTargetCacheAfterAcceptance(
            targetCache,
            targetLen: targetLen,
            acceptanceLength: acceptanceLength,
            draftedTokens: draftedTokens
        )
    }
}

public final class HybridGDNDFlashEngine: DFlashEngine, @unchecked Sendable {
    public init() {}

    public func armRollback(targetCache: [KVCache], prefixLen: Int) {
        DFlashRuntime.armTargetRollback(targetCache: targetCache, prefixLen: prefixLen)
    }

    public func rollback(
        targetCache: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        DFlashRuntime.restoreTargetCacheAfterAcceptance(
            targetCache,
            targetLen: targetLen,
            acceptanceLength: acceptanceLength,
            draftedTokens: draftedTokens
        )
    }
}

/// Disabled-by-default DFlash runtime helpers.
///
/// This enum exposes core utilities only. It does not register models, alter default generation,
/// or provide CLI/server entry points.
public enum DFlashRuntime {
    public static let isDefaultGenerationIntegrated = false

    public static func buildSuppressTokenMask(
        vocabSize: Int,
        suppressTokenIDs: [Int]?
    ) -> MLXArray? {
        let ids = Set((suppressTokenIDs ?? []).filter { $0 >= 0 && $0 < vocabSize })
        guard !ids.isEmpty else { return nil }

        var mask = [Bool](repeating: false, count: vocabSize)
        for id in ids {
            mask[id] = true
        }
        return MLXArray(mask)
    }

    public static func greedyTokensWithMask(
        logits: MLXArray,
        suppressTokenMask: MLXArray? = nil
    ) -> MLXArray {
        guard let suppressTokenMask else {
            return argMax(logits, axis: -1).asType(.uint32)
        }

        let floor = MLXArray(-1e9, dtype: logits.dtype)
        let maskedLogits = MLX.where(suppressTokenMask, floor, logits)
        return argMax(maskedLogits, axis: -1).asType(.uint32)
    }

    public static func matchAcceptanceLength(
        draftedTokens: MLXArray,
        posteriorTokens: MLXArray
    ) -> MLXArray {
        let count = draftedTokens.dim(0)
        guard count > 0 else { return MLXArray(0, dtype: .int32) }

        let matches = (draftedTokens .== posteriorTokens).asType(.int32)
        return cumprod(matches, axis: 0).sum(axis: 0, keepDims: false)
    }

    public static func makeTargetCache(targetModel: any DFlashTargetModel) -> [KVCache] {
        var cache = targetModel.newCache(parameters: nil)
        if targetModel.dflashIsHybridGDN {
            for index in cache.indices where cache[index] is MambaCache {
                cache[index] = targetModel.dflashUseTapeRollback
                    ? RecurrentRollbackCache()
                    : MambaSnapshotCache()
            }
        }
        return cache
    }

    public static func armTargetRollback(targetCache: [KVCache], prefixLen: Int) {
        for cache in targetCache {
            (cache as? DFlashRollbackCache)?.armRollback(prefixLen: prefixLen)
        }
    }

    @discardableResult
    public static func restoreTargetCacheAfterAcceptance(
        _ cacheEntries: [KVCache],
        targetLen: Int,
        acceptanceLength: Int,
        draftedTokens: Int
    ) -> Int {
        let fullyAccepted = draftedTokens > 0 && acceptanceLength == draftedTokens
        var replayNs = 0

        for cache in cacheEntries {
            if let rollbackCache = cache as? DFlashRollbackCache {
                if fullyAccepted {
                    rollbackCache.clearTransients()
                    continue
                }

                let startNs = Int(DispatchTime.now().uptimeNanoseconds)
                rollbackCache.rollback(nAccepted: acceptanceLength)
                replayNs += Int(DispatchTime.now().uptimeNanoseconds) - startNs
            } else if let mambaCache = cache as? MambaCache {
                mambaCache.offset = targetLen
            } else if cache.isTrimmable {
                let offset = cache.offset
                if offset > targetLen {
                    let startNs = Int(DispatchTime.now().uptimeNanoseconds)
                    cache.trim(offset - targetLen)
                    replayNs += Int(DispatchTime.now().uptimeNanoseconds) - startNs
                }
            }
        }

        return replayNs
    }
}
