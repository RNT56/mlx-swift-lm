// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX

public final class DFlashDraftBackend: @unchecked Sendable {
    public init() {}

    public func makeCache(
        draftModel: DFlashDraftModel,
        sinkSize: Int = 64,
        windowSize: Int = 1024
    ) -> [ContextOnlyDraftKVCache] {
        (0 ..< draftModel.layers.count).map { _ in
            ContextOnlyDraftKVCache(sinkSize: sinkSize, windowSize: windowSize)
        }
    }

    public func draftGreedy(
        targetModel: any DFlashTargetModel,
        draftModel: DFlashDraftModel,
        draftCache: [ContextOnlyDraftKVCache],
        stagedFirst: MLXArray,
        targetHidden: MLXArray,
        blockLen: Int,
        maskTokenTail: MLXArray,
        suppressTokenMask: MLXArray? = nil
    ) -> MLXArray {
        precondition(blockLen > 1, "draftGreedy requires blockLen > 1")

        let blockTokenIDs = concatenated(
            [stagedFirst[..<1], maskTokenTail[..<(blockLen - 1)]],
            axis: 0
        )

        let noiseEmbedding = targetModel.dflashEmbedTokens(blockTokenIDs[.newAxis])
        let draftHidden = draftModel(
            noiseEmbedding: noiseEmbedding,
            targetHidden: targetHidden,
            cache: draftCache
        )
        let draftLogits = targetModel.dflashLmHeadLogits(draftHidden[.ellipsis, 1..., 0...])
        let drafted = DFlashRuntime.greedyTokensWithMask(
            logits: draftLogits,
            suppressTokenMask: suppressTokenMask
        ).squeezed(axis: 0)

        asyncEval(drafted)
        return drafted
    }
}
