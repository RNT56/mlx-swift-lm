// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXNN

final class DFlashGLUMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public struct DFlashDraftConfiguration: Codable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var numKeyValueHeads: Int
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var headDim: Int
    public var tieWordEmbeddings: Bool
    public var numTargetLayers: Int
    public var blockSize: Int
    public var attentionBias: Bool
    public var attentionDropout: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var layerTypes: [String]
    public var dflashConfig: DFlashConfig?

    public struct DFlashConfig: Codable, Sendable {
        public var targetLayerIds: [Int]?
        public var maskTokenId: Int?

        public init(targetLayerIds: [Int]? = nil, maskTokenId: Int? = nil) {
            self.targetLayerIds = targetLayerIds
            self.maskTokenId = maskTokenId
        }

        enum CodingKeys: String, CodingKey {
            case targetLayerIds = "target_layer_ids"
            case maskTokenId = "mask_token_id"
        }
    }

    public init(
        modelType: String = "dflash_qwen3",
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 4,
        intermediateSize: Int = 2816,
        numAttentionHeads: Int = 16,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 151_936,
        numKeyValueHeads: Int = 8,
        maxPositionEmbeddings: Int = 131_072,
        ropeTheta: Float = 1_000_000.0,
        headDim: Int = 128,
        tieWordEmbeddings: Bool = false,
        numTargetLayers: Int = 36,
        blockSize: Int = 16,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        ropeScaling: [String: StringOrNumber]? = nil,
        layerTypes: [String] = [],
        dflashConfig: DFlashConfig? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.numKeyValueHeads = numKeyValueHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.tieWordEmbeddings = tieWordEmbeddings
        self.numTargetLayers = numTargetLayers
        self.blockSize = blockSize
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.ropeScaling = ropeScaling
        self.layerTypes = layerTypes
        self.dflashConfig = dflashConfig
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case numKeyValueHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numTargetLayers = "num_target_layers"
        case blockSize = "block_size"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case ropeScaling = "rope_scaling"
        case layerTypes = "layer_types"
        case dflashConfig = "dflash_config"
    }
}

public func buildDFlashTargetLayerIDs(numTargetLayers: Int, numDraftLayers: Int) -> [Int] {
    if numDraftLayers <= 1 {
        return [numTargetLayers / 2]
    }

    let start = 1
    let end = numTargetLayers - 3
    let span = end - start
    return (0 ..< numDraftLayers).map { index in
        Int(round(Double(start) + Double(index) * Double(span) / Double(numDraftLayers - 1)))
    }
}

public final class ContextOnlyDraftKVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset = 0
    public let sinkSize: Int
    public let windowSize: Int

    public init(sinkSize: Int = 64, windowSize: Int = 1024) {
        self.sinkSize = sinkSize
        self.windowSize = windowSize
    }

    public func appendContext(contextKeys: MLXArray, contextValues: MLXArray, numPositions: Int) {
        guard numPositions > 0 else { return }

        if keys == nil {
            keys = contextKeys
            values = contextValues
        } else {
            keys = concatenated([keys!, contextKeys], axis: 2)
            values = concatenated([values!, contextValues], axis: 2)
        }
        offset += numPositions
        applyWindow()
    }

    public func fetch() -> (MLXArray?, MLXArray?) {
        (keys, values)
    }

    public var cacheLength: Int {
        keys?.dim(2) ?? 0
    }

    private func applyWindow() {
        guard let keys, let values else { return }
        let cacheLength = keys.dim(2)
        let maxLength = sinkSize + windowSize
        guard cacheLength > maxLength else { return }

        let sinkKeys = keys[.ellipsis, ..<sinkSize, 0...]
        let sinkValues = values[.ellipsis, ..<sinkSize, 0...]
        let windowKeys = keys[.ellipsis, (-windowSize)..., 0...]
        let windowValues = values[.ellipsis, (-windowSize)..., 0...]
        self.keys = concatenated([sinkKeys, windowKeys], axis: 2)
        self.values = concatenated([sinkValues, windowValues], axis: 2)
    }
}

final class DFlashAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ args: DFlashDraftConfiguration) {
        nHeads = args.numAttentionHeads
        nKVHeads = args.numKeyValueHeads
        headDim = args.headDim
        scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(args.hiddenSize, nHeads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(args.hiddenSize, nKVHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(nHeads * headDim, args.hiddenSize, bias: args.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        rope = initializeRope(
            dims: headDim,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        targetHidden: MLXArray,
        cache: ContextOnlyDraftKVCache? = nil
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let blockLength = hiddenStates.dim(1)
        let contextLength = targetHidden.dim(1)

        var queries = qNorm(qProj(hiddenStates).reshaped(batchSize, blockLength, nHeads, headDim))
            .transposed(0, 2, 1, 3)
        var contextKeys = kNorm(
            kProj(targetHidden).reshaped(batchSize, contextLength, nKVHeads, headDim)
        ).transposed(0, 2, 1, 3)
        let contextValues = vProj(targetHidden).reshaped(batchSize, contextLength, nKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        var noiseKeys = kNorm(
            kProj(hiddenStates).reshaped(batchSize, blockLength, nKVHeads, headDim)
        ).transposed(0, 2, 1, 3)
        let noiseValues = vProj(hiddenStates).reshaped(batchSize, blockLength, nKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        if let cache {
            let cacheOffset = cache.offset
            let queryOffset = cacheOffset + contextLength

            queries = rope(queries, offset: queryOffset)
            contextKeys = rope(contextKeys, offset: cacheOffset)
            noiseKeys = rope(noiseKeys, offset: queryOffset)

            cache.appendContext(
                contextKeys: contextKeys,
                contextValues: contextValues,
                numPositions: contextLength
            )
            let (cachedKeys, cachedValues) = cache.fetch()
            let keys = concatenated([cachedKeys!, noiseKeys], axis: 2)
            let values = concatenated([cachedValues!, noiseValues], axis: 2)

            let output = DFlashKernels.sdpaFallback(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale
            )
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batchSize, blockLength, -1))
        }

        queries = rope(queries, offset: contextLength)
        contextKeys = rope(contextKeys, offset: 0)
        noiseKeys = rope(noiseKeys, offset: contextLength)

        let keys = concatenated([contextKeys, noiseKeys], axis: 2)
        let values = concatenated([contextValues, noiseValues], axis: 2)
        let output = DFlashKernels.sdpaFallback(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batchSize, blockLength, -1))
    }
}

final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: DFlashGLUMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: DFlashDraftConfiguration) {
        _selfAttention.wrappedValue = DFlashAttention(args)
        _mlp.wrappedValue = DFlashGLUMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        targetHidden: MLXArray,
        cache: ContextOnlyDraftKVCache? = nil
    ) -> MLXArray {
        let residual = hiddenStates
        var h = inputLayerNorm(hiddenStates)
        h = selfAttention(h, targetHidden: targetHidden, cache: cache)
        h = residual + h

        let secondResidual = h
        h = postAttentionLayerNorm(h)
        h = mlp(h)
        return secondResidual + h
    }
}

public final class DFlashDraftModel: Module {
    let args: DFlashDraftConfiguration
    let layers: [DFlashDecoderLayer]

    public let modelType: String
    public let targetLayerIDs: [Int]
    public let blockSize: Int
    public let maskTokenID: Int

    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm

    public init(_ args: DFlashDraftConfiguration) {
        self.args = args
        modelType = args.modelType
        layers = (0 ..< args.numHiddenLayers).map { _ in DFlashDecoderLayer(args) }
        targetLayerIDs = args.dflashConfig?.targetLayerIds
            ?? buildDFlashTargetLayerIDs(
                numTargetLayers: args.numTargetLayers,
                numDraftLayers: args.numHiddenLayers
            )
        blockSize = args.blockSize
        maskTokenID = args.dflashConfig?.maskTokenId ?? 0

        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(
            targetLayerIDs.count * args.hiddenSize,
            args.hiddenSize,
            bias: false
        )
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        super.init()
    }

    public func projectTargetHidden(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    public func callAsFunction(
        noiseEmbedding: MLXArray,
        targetHidden: MLXArray,
        cache: [ContextOnlyDraftKVCache]? = nil
    ) -> MLXArray {
        var hiddenStates = noiseEmbedding
        let projectedHidden = projectTargetHidden(targetHidden)
        let draftCache = cache ?? layers.map { _ in ContextOnlyDraftKVCache() }

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                targetHidden: projectedHidden,
                cache: index < draftCache.count ? draftCache[index] : nil
            )
        }

        return norm(hiddenStates)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

public func extractDFlashContextFeature(
    hiddenStates: [MLXArray],
    layerIDs: [Int]
) -> MLXArray {
    let selected = layerIDs.map { hiddenStates[$0 + 1] }
    return concatenated(selected, axis: -1)
}

public func extractDFlashContextFeature(
    capturedHiddenStates: [Int: MLXArray],
    targetLayerIDs: [Int]
) -> MLXArray {
    let selected = targetLayerIDs.map { capturedHiddenStates[$0 + 1]! }
    return concatenated(selected, axis: -1)
}
