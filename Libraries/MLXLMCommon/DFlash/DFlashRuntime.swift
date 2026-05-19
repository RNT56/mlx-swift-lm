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

extension DFlashTargetModel {
    public var dflashUseTapeRollback: Bool { true }
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
                cache[index] =
                    targetModel.dflashUseTapeRollback
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

    public static func generate(
        targetModel: any DFlashTargetModel,
        draftModel: DFlashDraftModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        blockTokens: Int? = nil,
        stopTokenIDs: [Int] = [],
        suppressTokenIDs: [Int]? = nil,
        draftSinkSize: Int = 64,
        draftWindowSize: Int = 1024,
        prefillStepSize: Int = 2048
    ) -> AsyncStream<DFlashEvent> {
        AsyncStream(bufferingPolicy: .unbounded) { continuation in
            let targetModelBox = SendableBox(targetModel)
            let draftModelBox = SendableBox(draftModel)
            let task = Task {
                let targetModel = targetModelBox.consume()
                let draftModel = draftModelBox.consume()
                generateStreaming(
                    targetModel: targetModel,
                    draftModel: draftModel,
                    promptTokens: promptTokens,
                    maxNewTokens: maxNewTokens,
                    blockTokens: blockTokens,
                    stopTokenIDs: stopTokenIDs,
                    suppressTokenIDs: suppressTokenIDs,
                    draftSinkSize: draftSinkSize,
                    draftWindowSize: draftWindowSize,
                    prefillStepSize: prefillStepSize
                ) { event in
                    guard !Task.isCancelled else { return }
                    continuation.yield(event)
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public static func generateSync(
        targetModel: any DFlashTargetModel,
        draftModel: DFlashDraftModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        blockTokens: Int? = nil,
        stopTokenIDs: [Int] = [],
        suppressTokenIDs: [Int]? = nil,
        draftSinkSize: Int = 64,
        draftWindowSize: Int = 1024,
        prefillStepSize: Int = 2048
    ) -> [DFlashEvent] {
        var events = [DFlashEvent]()
        generateStreaming(
            targetModel: targetModel,
            draftModel: draftModel,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            blockTokens: blockTokens,
            stopTokenIDs: stopTokenIDs,
            suppressTokenIDs: suppressTokenIDs,
            draftSinkSize: draftSinkSize,
            draftWindowSize: draftWindowSize,
            prefillStepSize: prefillStepSize
        ) { event in
            events.append(event)
        }
        return events
    }

    private static func generateStreaming(
        targetModel: any DFlashTargetModel,
        draftModel: DFlashDraftModel,
        promptTokens: [Int],
        maxNewTokens: Int,
        blockTokens: Int?,
        stopTokenIDs: [Int],
        suppressTokenIDs: [Int]?,
        draftSinkSize: Int,
        draftWindowSize: Int,
        prefillStepSize: Int,
        yield: (DFlashEvent) -> Void
    ) {
        let promptLength = promptTokens.count
        guard promptLength > 0, maxNewTokens > 0 else { return }

        let promptArray = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, -1)
            .asType(.uint32)
        let engine: any DFlashEngine =
            targetModel.dflashIsHybridGDN ? HybridGDNDFlashEngine() : FullAttentionDFlashEngine()
        let draftBackend = DFlashDraftBackend()
        let targetCache = makeTargetCache(targetModel: targetModel)
        let draftCache = draftBackend.makeCache(
            draftModel: draftModel,
            sinkSize: draftSinkSize,
            windowSize: draftWindowSize
        )

        let targetLayerIDs = draftModel.targetLayerIDs
        let captureLayerIDs = Set(targetLayerIDs.map { $0 + 1 })
        let maskTokenID = draftModel.maskTokenID
        let startNs = Int(DispatchTime.now().uptimeNanoseconds)

        var targetHidden: MLXArray?
        var prefillLogits: MLXArray?
        let stepSize = max(1, prefillStepSize)

        for chunkStart in stride(from: 0, to: promptLength, by: stepSize) {
            let chunkEnd = min(chunkStart + stepSize, promptLength)
            let chunkIDs = promptArray[0..., chunkStart ..< chunkEnd]
            let (chunkLogits, captured) = targetModel.dflashForwardWithCapture(
                inputIDs: chunkIDs,
                cache: targetCache,
                captureLayerIDs: captureLayerIDs
            )
            guard let feature = contextFeature(captured: captured, targetLayerIDs: targetLayerIDs)
            else {
                return
            }

            asyncEval(chunkLogits)
            for value in captured.values {
                asyncEval(value)
            }

            if targetHidden == nil {
                targetHidden = MLXArray.zeros(
                    [feature.dim(0), promptLength, feature.dim(-1)],
                    dtype: feature.dtype
                )
            }
            targetHidden![0..., chunkStart ..< chunkEnd, 0...] = feature
            eval(targetHidden!)
            prefillLogits = chunkLogits

            yield(.prefillProgress(tokensProcessed: chunkEnd, tokensTotal: promptLength))
        }

        guard let targetHidden, let prefillLogits else { return }

        MLX.Memory.clearCache()
        let prefillNs = Int(DispatchTime.now().uptimeNanoseconds) - startNs
        let suppressTokenMask = buildSuppressTokenMask(
            vocabSize: prefillLogits.dim(-1),
            suppressTokenIDs: suppressTokenIDs
        )
        var stagedFirst = greedyTokensWithMask(
            logits: prefillLogits[0..., -1, 0...],
            suppressTokenMask: suppressTokenMask
        ).reshaped(-1)

        yield(.prefill(promptTokenCount: promptLength, prefillUs: Double(prefillNs) / 1000.0))

        let firstTokenID = stagedFirst.item(Int.self)
        var generatedTokenIDs = [firstTokenID]
        yield(
            .token(
                tokenID: firstTokenID,
                generatedTokens: 1,
                acceptanceRatio: 0,
                cyclesCompleted: 0
            )
        )

        let requestedBlockTokens = blockTokens ?? draftModel.blockSize
        let effectiveBlockTokens = max(1, min(requestedBlockTokens, draftModel.blockSize))
        let maskTokenTail = MLXArray(
            Array(repeating: Int32(maskTokenID), count: max(0, effectiveBlockTokens - 1))
        ).asType(.uint32)
        let stopTokenSet = Set(stopTokenIDs)

        var currentTargetHidden = targetHidden
        var acceptedFromDraft = 0
        var cyclesCompleted = 0
        var targetLength = promptLength
        var firstTokenAlreadyYielded = true
        var verifyNsTotal = 0
        var draftNsTotal = 0
        var replayNsTotal = 0
        var prefetchedDraft: MLXArray?
        var prefetchedBlockLength: Int?

        while generatedTokenIDs.count < maxNewTokens {
            let remaining = maxNewTokens - generatedTokenIDs.count
            let blockLength = max(1, min(effectiveBlockTokens, remaining))
            let stagedFirstForCycle = stagedFirst

            let drafted: MLXArray?
            if blockLength > 1 {
                if let readyDraft = prefetchedDraft, prefetchedBlockLength == blockLength {
                    drafted = readyDraft
                    prefetchedDraft = nil
                    prefetchedBlockLength = nil
                } else {
                    let draftStart = Int(DispatchTime.now().uptimeNanoseconds)
                    drafted = draftBackend.draftGreedy(
                        targetModel: targetModel,
                        draftModel: draftModel,
                        draftCache: draftCache,
                        stagedFirst: stagedFirst,
                        targetHidden: currentTargetHidden,
                        blockLen: blockLength,
                        maskTokenTail: maskTokenTail,
                        suppressTokenMask: suppressTokenMask
                    )
                    draftNsTotal += Int(DispatchTime.now().uptimeNanoseconds) - draftStart
                }
            } else {
                drafted = nil
            }

            let verifyTokenIDs: MLXArray
            if blockLength <= 1 {
                verifyTokenIDs = stagedFirstForCycle[..<1]
            } else if let drafted {
                verifyTokenIDs = concatenated(
                    [stagedFirstForCycle[..<1], drafted[..<(blockLength - 1)]],
                    axis: 0
                )
            } else {
                verifyTokenIDs = stagedFirstForCycle[..<1]
            }

            engine.armRollback(targetCache: targetCache, prefixLen: targetLength)
            let verifyStart = Int(DispatchTime.now().uptimeNanoseconds)
            let (verifyLogits, verifyHiddenStates) = targetModel.dflashForwardWithCapture(
                inputIDs: verifyTokenIDs[.newAxis],
                cache: targetCache,
                captureLayerIDs: captureLayerIDs
            )
            guard
                let verifyFeature = contextFeature(
                    captured: verifyHiddenStates,
                    targetLayerIDs: targetLayerIDs
                )
            else {
                return
            }
            asyncEval(verifyLogits)
            for value in verifyHiddenStates.values {
                asyncEval(value)
            }
            verifyNsTotal += Int(DispatchTime.now().uptimeNanoseconds) - verifyStart

            let posterior = greedyTokensWithMask(
                logits: verifyLogits[0],
                suppressTokenMask: suppressTokenMask
            )
            let acceptanceLength: Int
            if verifyTokenIDs.dim(0) > 1 {
                acceptanceLength = matchAcceptanceLength(
                    draftedTokens: verifyTokenIDs[1...],
                    posteriorTokens: posterior[..<(verifyTokenIDs.dim(0) - 1)]
                ).item(Int.self)
            } else {
                acceptanceLength = 0
            }

            let commitCount = 1 + acceptanceLength
            let committedHidden = verifyFeature[0..., ..<commitCount, 0...]
            asyncEval(committedHidden)
            let committedSegment = verifyTokenIDs[..<commitCount]
            let stagedFirstNext = posterior[acceptanceLength ..< (acceptanceLength + 1)]

            let nextRemaining = maxNewTokens - generatedTokenIDs.count - commitCount
            let nextBlockLength = max(1, min(effectiveBlockTokens, nextRemaining))
            if nextBlockLength > 1, generatedTokenIDs.count + commitCount < maxNewTokens {
                prefetchedDraft = draftBackend.draftGreedy(
                    targetModel: targetModel,
                    draftModel: draftModel,
                    draftCache: draftCache,
                    stagedFirst: stagedFirstNext,
                    targetHidden: committedHidden,
                    blockLen: nextBlockLength,
                    maskTokenTail: maskTokenTail,
                    suppressTokenMask: suppressTokenMask
                )
                prefetchedBlockLength = nextBlockLength
                asyncEval(prefetchedDraft!)
            } else {
                prefetchedDraft = nil
                prefetchedBlockLength = nil
            }

            targetLength += commitCount
            currentTargetHidden = committedHidden
            replayNsTotal += engine.rollback(
                targetCache: targetCache,
                targetLen: targetLength,
                acceptanceLength: acceptanceLength,
                draftedTokens: blockLength - 1
            )
            cyclesCompleted += 1
            acceptedFromDraft += acceptanceLength

            let committedIDs = committedSegment.asArray(Int.self)
            for tokenID in committedIDs {
                guard generatedTokenIDs.count < maxNewTokens else { break }
                if firstTokenAlreadyYielded {
                    firstTokenAlreadyYielded = false
                    continue
                }

                generatedTokenIDs.append(tokenID)
                let acceptanceRatio =
                    Double(acceptedFromDraft) / Double(max(1, generatedTokenIDs.count))
                yield(
                    .token(
                        tokenID: tokenID,
                        generatedTokens: generatedTokenIDs.count,
                        acceptanceRatio: acceptanceRatio,
                        cyclesCompleted: cyclesCompleted
                    )
                )
            }

            if committedIDs.contains(where: { stopTokenSet.contains($0) }) {
                break
            }
            stagedFirst = stagedFirstNext
        }

        let elapsedNs = Int(DispatchTime.now().uptimeNanoseconds) - startNs
        let acceptanceRatio = Double(acceptedFromDraft) / Double(max(1, generatedTokenIDs.count))
        yield(
            .summary(
                DFlashSummary(
                    elapsedUs: Double(elapsedNs) / 1000.0,
                    promptTokenCount: promptLength,
                    generatedTokenIDs: generatedTokenIDs,
                    acceptedFromDraft: acceptedFromDraft,
                    acceptanceRatio: acceptanceRatio,
                    blockTokens: effectiveBlockTokens,
                    cyclesCompleted: cyclesCompleted,
                    phaseTimingsUs: .init(
                        prefill: Double(prefillNs) / 1000.0,
                        draft: Double(draftNsTotal) / 1000.0,
                        verify: Double(verifyNsTotal) / 1000.0,
                        replay: Double(replayNsTotal) / 1000.0
                    )
                )
            )
        )
    }

    private static func contextFeature(
        captured: [Int: MLXArray],
        targetLayerIDs: [Int]
    ) -> MLXArray? {
        var selected = [MLXArray]()
        selected.reserveCapacity(targetLayerIDs.count)
        for layerID in targetLayerIDs {
            guard let hidden = captured[layerID + 1] else {
                return nil
            }
            selected.append(hidden)
        }
        return concatenated(selected, axis: -1)
    }

}
