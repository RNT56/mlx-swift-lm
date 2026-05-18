// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file

import MLX
import MLXLMCommon
import MLXNN

extension Qwen3Model: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? model.embedTokens.asLinear(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let (hiddenStates, captured) = model.callCapturing(
            inputIDs,
            cache: cache.map { $0 },
            captureLayerIDs: captureLayerIDs
        )
        return (dflashLmHeadLogits(hiddenStates), captured)
    }

    public var dflashIsHybridGDN: Bool { false }
}

extension Qwen3MoEModel: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? model.embedTokens.asLinear(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let (hiddenStates, captured) = model.callCapturing(
            inputIDs,
            cache: cache.map { $0 },
            captureLayerIDs: captureLayerIDs
        )
        return (dflashLmHeadLogits(hiddenStates), captured)
    }

    public var dflashIsHybridGDN: Bool { false }
}

extension Qwen35TextModel: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? model.embedTokens.asLinear(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let (hiddenStates, captured) = model.callCapturing(
            inputIDs,
            cache: cache.map { $0 },
            captureLayerIDs: captureLayerIDs
        )
        return (dflashLmHeadLogits(hiddenStates), captured)
    }

    public var dflashIsHybridGDN: Bool { false }
}

extension Qwen35Model: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        languageModel.dflashEmbedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        languageModel.dflashLmHeadLogits(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        languageModel.dflashForwardWithCapture(
            inputIDs: inputIDs,
            cache: cache,
            captureLayerIDs: captureLayerIDs
        )
    }

    public var dflashIsHybridGDN: Bool { languageModel.dflashIsHybridGDN }
}

extension LlamaModel: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? model.embedTokens.asLinear(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let (hiddenStates, captured) = model.callCapturing(
            inputIDs,
            cache: cache.map { $0 },
            captureLayerIDs: captureLayerIDs
        )
        return (dflashLmHeadLogits(hiddenStates), captured)
    }

    public var dflashIsHybridGDN: Bool { false }
}

extension Qwen3NextModel: DFlashTargetModel {
    public func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? model.embedTokens.asLinear(hiddenStates)
    }

    public func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let (hiddenStates, captured) = model.callCapturing(
            inputIDs,
            cache: cache.map { $0 },
            captureLayerIDs: captureLayerIDs
        )
        return (dflashLmHeadLogits(hiddenStates), captured)
    }

    public var dflashIsHybridGDN: Bool { false }
}
