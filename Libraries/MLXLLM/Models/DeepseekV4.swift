// Copyright (c) 2025 Apple Inc.

// Port of DeepSeek-V4 inference code
// Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    enum AttentionLayerType: String, Codable, Sendable {
        case sliding = "sliding_attention"
        case compressedSparse = "compressed_sparse_attention"
        case heavilyCompressed = "heavily_compressed_attention"
    }

    enum MLPLayerType: String, Codable, Sendable {
        case hashMoE = "hash_moe"
        case moe
    }

    // Core architecture
    var vocabSize: Int
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var headDim: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var rmsNormEps: Float

    // Output projection grouping
    var oGroups: Int
    var oLoraRank: Int

    // Attention / compression (per layer)
    var slidingWindow: Int
    var compressRatios: [Int]
    var compressRates: [String: Int]?
    var compressRateCSA: Int
    var compressRateHCA: Int
    var compressRopeTheta: Float
    var layerTypes: [AttentionLayerType]

    // MoE
    var nRoutedExperts: Int
    var nSharedExperts: Int
    var numExpertsPerTok: Int
    var scoringFunc: String
    var routedScalingFactor: Float
    var swiguLimit: Float
    var numHashLayers: Int
    var numNextnPredictLayers: Int
    var normTopkProb: Bool
    var mlpLayerTypes: [MLPLayerType]

    // Hyper-Connections (mHC)
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float

    // RoPE
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int

    // Compressed sparse attention indexer
    var indexHeads: Int
    var indexHeadDim: Int
    var indexTopK: Int

    // Output
    var tieWordEmbeddings: Bool

    // Nope head dim (derived)
    var nopeHeadDim: Int { headDim - qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case compressRates = "compress_rates"
        case compressRateCSA = "compress_rate_csa"
        case compressRateHCA = "compress_rate_hca"
        case compressRopeTheta = "compress_rope_theta"
        case layerTypes = "layer_types"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case swiguLimit = "swiglu_limit"
        case numHashLayers = "num_hash_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case normTopkProb = "norm_topk_prob"
        case mlpLayerTypes = "mlp_layer_types"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129_280
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4_096
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2_048
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 43
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 64
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 512
        qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1_024
        qkRopeHeadDim = try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        oGroups = try container.decodeIfPresent(Int.self, forKey: .oGroups) ?? 8
        oLoraRank = try container.decodeIfPresent(Int.self, forKey: .oLoraRank) ?? 1_024
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128

        compressRates = try container.decodeIfPresent([String: Int].self, forKey: .compressRates)
        compressRateCSA =
            try container.decodeIfPresent(Int.self, forKey: .compressRateCSA)
            ?? compressRates?["compressed_sparse_attention"] ?? 4
        compressRateHCA =
            try container.decodeIfPresent(Int.self, forKey: .compressRateHCA)
            ?? compressRates?["heavily_compressed_attention"] ?? 128
        compressRatios = try container.decodeIfPresent([Int].self, forKey: .compressRatios) ?? []
        compressRopeTheta =
            try container.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160_000

        nRoutedExperts = try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 256
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        scoringFunc =
            try container.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sqrtsoftplus"
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.5
        swiguLimit = try container.decodeIfPresent(Float.self, forKey: .swiguLimit) ?? 10
        numHashLayers = try container.decodeIfPresent(Int.self, forKey: .numHashLayers) ?? 3
        numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        hcMult = try container.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcSinkhornIters = try container.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        hcEps = try container.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1.0e-6

        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576

        indexHeads = try container.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 64
        indexHeadDim = try container.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopK = try container.decodeIfPresent(Int.self, forKey: .indexTopK) ?? 512
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        if let explicitLayerTypes = try container.decodeIfPresent(
            [AttentionLayerType].self, forKey: .layerTypes)
        {
            layerTypes = Array(explicitLayerTypes.prefix(numHiddenLayers))
        } else {
            let csaRate = compressRateCSA
            let ratios = compressRatios.isEmpty
                ? Self.defaultCompressRatios(
                    count: numHiddenLayers,
                    compressedSparseRate: csaRate,
                    heavilyCompressedRate: compressRateHCA)
                : compressRatios
            let resolvedLayerTypes = ratios.prefix(numHiddenLayers).map {
                switch $0 {
                case 0:
                    return DeepseekV4Configuration.AttentionLayerType.sliding
                case csaRate:
                    return DeepseekV4Configuration.AttentionLayerType.compressedSparse
                default:
                    return DeepseekV4Configuration.AttentionLayerType.heavilyCompressed
                }
            }
            layerTypes = resolvedLayerTypes
        }
        if layerTypes.count < numHiddenLayers {
            layerTypes += Array(
                repeating: .heavilyCompressed,
                count: numHiddenLayers - layerTypes.count)
        }

        if let explicitMLPTypes = try container.decodeIfPresent(
            [MLPLayerType].self, forKey: .mlpLayerTypes)
        {
            mlpLayerTypes = Array(explicitMLPTypes.prefix(numHiddenLayers))
        } else {
            mlpLayerTypes =
                Array(repeating: .hashMoE, count: min(numHiddenLayers, numHashLayers))
                + Array(repeating: .moe, count: max(0, numHiddenLayers - numHashLayers))
        }
        if mlpLayerTypes.count < numHiddenLayers {
            mlpLayerTypes += Array(repeating: .moe, count: numHiddenLayers - mlpLayerTypes.count)
        }
    }

    private static func defaultCompressRatios(
        count: Int,
        compressedSparseRate: Int,
        heavilyCompressedRate: Int
    ) -> [Int] {
        guard count > 0 else { return [] }
        if count == 1 { return [0] }
        return (0 ..< count).map { index in
            if index == 0 { return 0 }
            return index.isMultiple(of: 2) ? heavilyCompressedRate : compressedSparseRate
        }
    }
}

// MARK: - Helper Functions

/// sqrtsoftplus activation: sqrt(softplus(x)) = sqrt(log(1 + e^x))
/// Uses numerically stable form to avoid exp overflow for large positive x.
private func sqrtSoftplus(_ x: MLXArray) -> MLXArray {
    let sp = MLX.maximum(x, MLXArray(0)) + MLX.log1p(MLX.exp(-MLX.abs(x)))
    return MLX.sqrt(sp)
}

/// Apply per-head RMS normalization (without learnable scale)
private func headRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
}

// MARK: - Sinkhorn-based Hyper-Connection helpers

/// Split mixes into (pre, post, comb) with Sinkhorn normalization.
/// mixes: [B, S, mix_hc] where mix_hc = (2+hc)*hc
/// Returns pre [B,S,hc], post [B,S,hc], comb [B,S,hc,hc]
private func hcSplitSinkhorn(
    _ mixes: MLXArray,
    hcScale: MLXArray,  // [3]
    hcBase: MLXArray,  // [mix_hc]
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let hc = hcMult
    let B = mixes.dim(0)
    let S = mixes.dim(1)

    // Split mixes into 3 parts
    let preMix = mixes[.ellipsis, ..<hc]  // [B, S, hc]
    let postMix = mixes[.ellipsis, hc ..< 2 * hc]  // [B, S, hc]
    let combMix = mixes[.ellipsis, (2 * hc)...]  // [B, S, hc*hc]

    // Per-part scale, per-element base
    let preBase = hcBase[..<hc]
    let postBase = hcBase[hc ..< 2 * hc]
    let combBase = hcBase[(2 * hc)...]

    // Apply scale, add bias, then sigmoid + eps
    var pre = sigmoid(preMix * hcScale[0] + preBase) + eps  // [B, S, hc]
    let post = sigmoid(postMix * hcScale[1] + postBase) + eps  // [B, S, hc]
    var comb = (sigmoid(combMix * hcScale[2] + combBase) + eps)
        .reshaped(B, S, hc, hc)  // [B, S, hc, hc]

    // Normalize pre so it sums to 1 across hc copies
    pre = pre / pre.sum(axis: -1, keepDims: true)

    // Sinkhorn normalization for comb (alternating column/row normalize)
    for _ in 0 ..< sinkhornIters {
        comb = comb / comb.sum(axis: -2, keepDims: true)  // column normalize
        comb = comb / comb.sum(axis: -1, keepDims: true)  // row normalize
    }

    return (pre, post, comb)
}

/// Hyper-Connection pre-step: reduce [B,S,hc,D] -> [B,S,D] with Sinkhorn weights.
/// Returns (reduced_x, post_weights, comb_matrix).
private func hcPre(
    x: MLXArray,  // [B, S, hc, D]
    hcFn: MLXArray,  // [mix_hc, hc*D]
    hcScale: MLXArray,  // [3]
    hcBase: MLXArray,  // [mix_hc]
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let dtype = x.dtype
    let B = x.dim(0)
    let S = x.dim(1)
    let hc = x.dim(2)
    let D = x.dim(3)

    // Flatten: [B, S, hc*D]
    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)

    // RMS-style normalization scale
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)

    // Linear projection: [B, S, mix_hc]
    let mixes = matmul(xFlat, hcFn.T) * normScale

    let (pre, post, comb) = hcSplitSinkhorn(
        mixes, hcScale: hcScale, hcBase: hcBase,
        hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)

    // Weighted sum of hc copies: [B, S, D]
    let y = (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2)

    return (y, post, comb)
}

/// Hyper-Connection post-step: expand sublayer output back to [B,S,hc,D].
/// y[b,s,j,:] = post[b,s,j]*x[b,s,:] + sum_i(comb[b,s,i,j]*residual[b,s,i,:])
private func hcPost(
    x: MLXArray,  // [B, S, D] - sublayer output
    residual: MLXArray,  // [B, S, hc, D] - input to this block
    post: MLXArray,  // [B, S, hc]
    comb: MLXArray  // [B, S, hc, hc]
) -> MLXArray {
    // term1: post[b,s,j] * x[b,s,:] -> broadcast to [B,S,hc,D]
    let term1 = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2)

    // term2: sum_i(comb[b,s,i,j] * residual[b,s,i,:])
    // comb.unsqueeze(-1): [B,S,hc_i,hc_j,1]
    // residual.unsqueeze(-2): [B,S,hc_i,1,D]
    // product: [B,S,hc_i,hc_j,D] -> sum over dim 2 -> [B,S,hc_j,D]
    let combExp = comb.expandedDimensions(axis: -1)  // [B,S,hc,hc,1]
    let residualExp = residual.expandedDimensions(axis: -2)  // [B,S,hc,1,D]
    let term2 = (combExp * residualExp).sum(axis: 2)  // [B,S,hc,D]

    return (term1 + term2).asType(x.dtype)
}

// MARK: - HCParams Module

/// Lightweight Module to hold the three Hyper-Connection tensors loaded from checkpoint.
/// Key names (fn, base, scale) match the `hc_attn.*` / `hc_ffn.*` / `hc_head.*` paths.
class HCParams: Module {
    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(fn: MLXArray, base: MLXArray, scale: MLXArray) {
        self._fn.wrappedValue = fn
        self._base.wrappedValue = base
        self._scale.wrappedValue = scale
    }
}

/// Final HC head: reduce [B,S,hc,D] -> [B,S,D] for lm_head.
/// No Sinkhorn - just sigmoid + eps weighted sum.
private func hcHead(
    x: MLXArray,  // [B, S, hc, D]
    hcFn: MLXArray,  // [hc, hc*D]
    hcScale: MLXArray,  // [1]
    hcBase: MLXArray,  // [hc]
    eps: Float
) -> MLXArray {
    let dtype = x.dtype
    let B = x.dim(0)
    let S = x.dim(1)
    let hc = x.dim(2)
    let D = x.dim(3)

    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale  // [B, S, hc]
    let pre = sigmoid(mixes * hcScale + hcBase) + eps  // [B, S, hc]

    // Weighted sum: [B, S, D]
    let y = (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2)
    return y.asType(dtype)
}

// MARK: - Attention

/// Attention with cache update that optionally applies per-head sink bias.
/// Mirrors `attentionWithCacheUpdateAndSinks` from MiMoV2Flash but uses the public API.
private func deepseekAttentionWithSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask, sinks: sinks)
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (qk, qv) = quantizedKVCache.updateQuantized(keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: qk, quantizedValues: qv,
            scale: scale, mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode)
    }
    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: cachedKeys, values: cachedValues,
        scale: scale, mask: mask, sinks: sinks)
}

final class DeepseekV4KVCache: KVCache, CustomDebugStringConvertible {
    let layerType: DeepseekV4Configuration.AttentionLayerType
    let slidingWindow: Int
    var offset: Int = 0

    private var keys: MLXArray?
    private var values: MLXArray?
    private var compressionKV = [String: MLXArray]()
    private var compressionGate = [String: MLXArray]()
    private var compressionEntries = [String: Int]()
    private var compressed = [String: MLXArray]()

    init(layerType: DeepseekV4Configuration.AttentionLayerType, slidingWindow: Int) {
        self.layerType = layerType
        self.slidingWindow = slidingWindow
    }

    var maxSize: Int? {
        layerType == .sliding ? slidingWindow : nil
    }

    func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let nextKeys = keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        let nextValues = values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        let keep = layerType == .sliding ? max(slidingWindow, 1) : Int.max
        if nextKeys.dim(2) > keep {
            keys = nextKeys[.ellipsis, (nextKeys.dim(2) - keep)..., 0...]
            values = nextValues[.ellipsis, (nextValues.dim(2) - keep)..., 0...]
        } else {
            keys = nextKeys
            values = nextValues
        }
        offset += newKeys.dim(2)
        return (keys ?? newKeys, values ?? newValues)
    }

    var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [keys, values]
        }
        set {
            guard newValue.count == 2 else {
                keys = nil
                values = nil
                return
            }
            keys = newValue[0]
            values = newValue[1]
        }
    }

    var metaState: [String] {
        get { ["deepseek_v4", layerType.rawValue, "\(offset)", "\(slidingWindow)"] }
        set {
            if newValue.count >= 3, let parsedOffset = Int(newValue[2]) {
                offset = parsedOffset
            }
        }
    }

    var isTrimmable: Bool { true }

    @discardableResult
    func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        if trimmed > 0 {
            keys = nil
            values = nil
        }
        return trimmed
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 { return .none }
        let effectiveWindow = windowSize ?? maxSize
        if returnArray || (effectiveWindow != nil && n > effectiveWindow!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: effectiveWindow))
        }
        return .causal
    }

    func innerState() -> [MLXArray] {
        state
    }

    func copy() -> any KVCache {
        let copy = DeepseekV4KVCache(layerType: layerType, slidingWindow: slidingWindow)
        copy.offset = offset
        copy.keys = keys
        copy.values = values
        copy.compressionKV = compressionKV
        copy.compressionGate = compressionGate
        copy.compressionEntries = compressionEntries
        copy.compressed = compressed
        return copy
    }

    func storeCompressionWeights(
        name: String,
        kv: MLXArray,
        gate: MLXArray,
        compressRate: Int
    ) -> (MLXArray, MLXArray, Int) {
        let firstWindowPosition = (compressionEntries[name] ?? 0) * compressRate
        var kv = kv
        var gate = gate
        if let priorKV = compressionKV[name], let priorGate = compressionGate[name], priorKV.dim(1) > 0 {
            kv = concatenated([priorKV, kv], axis: 1)
            gate = concatenated([priorGate, gate], axis: 1)
        }

        let usable = (kv.dim(1) / compressRate) * compressRate
        if usable < kv.dim(1) {
            compressionKV[name] = kv[0..., usable..., 0...]
            compressionGate[name] = gate[0..., usable..., 0...]
        } else {
            compressionKV.removeValue(forKey: name)
            compressionGate.removeValue(forKey: name)
        }

        return (kv[0..., ..<usable, 0...], gate[0..., ..<usable, 0...], firstWindowPosition)
    }

    func updateCompressedState(name: String, compressed newCompressed: MLXArray) -> MLXArray {
        if let previous = compressed[name], newCompressed.dim(1) > 0 {
            compressed[name] = concatenated([previous, newCompressed], axis: 1)
        } else if compressed[name] == nil {
            compressed[name] = newCompressed
        }
        compressionEntries[name, default: 0] += newCompressed.dim(1)
        return compressed[name] ?? newCompressed
    }

    var debugDescription: String {
        "DeepseekV4KVCache(type: \(layerType.rawValue), offset: \(offset), keys: \(keys?.shape.description ?? "-"))"
    }
}

private func deepseekCompressionPositions(
    batch: Int,
    windows: Int,
    firstWindowPosition: Int,
    rate: Int
) -> MLXArray {
    let positions = MLXArray(Int32(0) ..< Int32(windows)) * Int32(rate) + Int32(firstWindowPosition)
    return tiled(positions[.newAxis, 0...], repetitions: [batch, 1])
}

final class DeepseekV4IndexerCompressor: Module {
    let config: DeepseekV4Configuration

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._wkv.wrappedValue = Linear(config.hiddenSize, 2 * config.indexHeadDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, 2 * config.indexHeadDim, bias: false)
        self._ape.wrappedValue = zeros([config.compressRateCSA, 2 * config.indexHeadDim])
        self._norm.wrappedValue = RMSNorm(dimensions: config.indexHeadDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: DeepseekV4KVCache?) -> MLXArray {
        let batch = x.dim(0)
        let rate = config.compressRateCSA
        var kv = wkv(x)
        var gate = wgate(x)
        if let cache {
            (kv, gate, _) = cache.storeCompressionWeights(
                name: "indexer", kv: kv, gate: gate, compressRate: rate)
        } else {
            let usable = (kv.dim(1) / rate) * rate
            kv = kv[0..., ..<usable, 0...]
            gate = gate[0..., ..<usable, 0...]
        }
        guard kv.dim(1) > 0 else {
            return zeros([batch, 0, config.indexHeadDim], dtype: x.dtype)
        }

        let windows = kv.dim(1) / rate
        kv = kv.reshaped(batch, windows, rate, -1)
        gate = gate.reshaped(batch, windows, rate, -1) + ape.asType(gate.dtype)
        var compressed = (kv * softmax(gate.asType(.float32), axis: 2, precise: true).asType(kv.dtype))
            .sum(axis: 2)
        if compressed.dim(-1) > config.indexHeadDim {
            compressed = compressed[.ellipsis, ..<config.indexHeadDim]
        }
        compressed = norm(compressed)
        if let cache {
            return cache.updateCompressedState(name: "indexer", compressed: compressed)
        }
        return compressed
    }
}

final class DeepseekV4Indexer: Module {
    let config: DeepseekV4Configuration
    let scale: Float
    let weightsScale: Float

    @ModuleInfo(key: "compressor") var compressor: DeepseekV4IndexerCompressor
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    init(config: DeepseekV4Configuration) {
        self.config = config
        self.scale = pow(Float(config.indexHeadDim), -0.5)
        self.weightsScale = pow(Float(config.indexHeads), -0.5)
        self._compressor.wrappedValue = DeepseekV4IndexerCompressor(config: config)
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.indexHeads * config.indexHeadDim, bias: false)
        self._weightsProj.wrappedValue = Linear(config.hiddenSize, config.indexHeads, bias: false)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        qResidual: MLXArray,
        cache: DeepseekV4KVCache?
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let length = hiddenStates.dim(1)
        let compressedKV = compressor(hiddenStates, cache: cache)
        let compressedLength = compressedKV.dim(1)
        guard compressedLength > 0 else {
            return zeros([batch, length, 0], dtype: .int32)
        }

        let queries = wqB(qResidual)
            .reshaped(batch, length, config.indexHeads, config.indexHeadDim)
        var scores = matmul(
            queries.asType(.float32),
            compressedKV.asType(.float32).transposed(0, 2, 1)[0..., .newAxis, 0..., 0...]
        )
        scores = relu(scores) * scale
        let weights = weightsProj(hiddenStates).asType(.float32) * weightsScale
        var indexScores = (scores * weights[0..., 0..., 0..., .newAxis]).sum(axis: 2)

        let cacheOffset = cache?.offset ?? 0
        let positions = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + length))
            .reshaped(1, length)
        let causalThreshold = (positions + 1).floorDivide(config.compressRateCSA)
        let entryIndices = MLXArray(Int32(0) ..< Int32(compressedLength))
        let futureMask = entryIndices.reshaped(1, 1, -1) .>= causalThreshold[0..., 0..., .newAxis]
        indexScores = MLX.where(futureMask, MLXArray(-1.0e9).asType(indexScores.dtype), indexScores)

        let topK = min(config.indexTopK, compressedLength)
        let selected = argPartition(-indexScores, kth: max(topK - 1, 0), axis: -1)[0..., 0..., ..<topK]
        let invalid = selected .>= causalThreshold[0..., 0..., .newAxis]
        return MLX.where(invalid, MLXArray(Int32(-1)), selected)
    }
}

final class DeepseekV4AttentionCompressor: Module {
    let config: DeepseekV4Configuration
    let layerType: DeepseekV4Configuration.AttentionLayerType
    let compressRate: Int

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ParameterInfo(key: "ape") var ape: MLXArray
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?

    init(config: DeepseekV4Configuration, layerType: DeepseekV4Configuration.AttentionLayerType) {
        self.config = config
        self.layerType = layerType
        self.compressRate =
            layerType == .compressedSparse ? config.compressRateCSA : config.compressRateHCA
        let projectionDim = layerType == .compressedSparse ? 2 * config.headDim : config.headDim
        self._wkv.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        self._ape.wrappedValue = zeros([compressRate, projectionDim])
        self._norm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        if layerType == .compressedSparse {
            self._indexer.wrappedValue = DeepseekV4Indexer(config: config)
        }
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        qResidual: MLXArray,
        cache: DeepseekV4KVCache?
    ) -> (kv: MLXArray, bias: MLXArray?) {
        let batch = hiddenStates.dim(0)
        let sequenceLength = hiddenStates.dim(1)
        var kv = wkv(hiddenStates)
        var gate = wgate(hiddenStates)
        let firstWindowPosition: Int
        if let cache {
            (kv, gate, firstWindowPosition) = cache.storeCompressionWeights(
                name: "compressor", kv: kv, gate: gate, compressRate: compressRate)
        } else {
            let usable = (kv.dim(1) / compressRate) * compressRate
            kv = kv[0..., ..<usable, 0...]
            gate = gate[0..., ..<usable, 0...]
            firstWindowPosition = 0
        }

        var compressed: MLXArray
        if kv.dim(1) > 0 {
            let windows = kv.dim(1) / compressRate
            kv = kv.reshaped(batch, windows, compressRate, -1)
            gate = gate.reshaped(batch, windows, compressRate, -1) + ape.asType(gate.dtype)
            compressed = (kv * softmax(gate.asType(.float32), axis: 2, precise: true).asType(kv.dtype))
                .sum(axis: 2)
            if compressed.dim(-1) > config.headDim {
                compressed = compressed[.ellipsis, ..<config.headDim]
            }
            _ = deepseekCompressionPositions(
                batch: batch,
                windows: windows,
                firstWindowPosition: firstWindowPosition,
                rate: compressRate)
            compressed = norm(compressed)
        } else {
            compressed = zeros([batch, 0, config.headDim], dtype: hiddenStates.dtype)
        }

        if let cache {
            compressed = cache.updateCompressedState(name: "compressor", compressed: compressed)
        }
        let compressedKV = compressed[0..., .newAxis, 0..., 0...]
        guard compressedKV.dim(2) > 0 else {
            return (compressedKV, nil)
        }

        if layerType == .heavilyCompressed {
            if sequenceLength == 1 {
                return (compressedKV, nil)
            }
            let entryIndices = MLXArray(Int32(0) ..< Int32(compressedKV.dim(2)))
            let cacheOffset = cache?.offset ?? 0
            let positions = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + sequenceLength))
                .reshaped(1, sequenceLength)
            let causalThreshold = (positions + 1).floorDivide(compressRate)
            let future =
                entryIndices.reshaped(1, 1, 1, -1) .>= causalThreshold[0..., .newAxis, 0..., .newAxis]
            let bias = MLX.where(
                future,
                MLXArray(-1.0e9).asType(hiddenStates.dtype),
                MLXArray(0.0).asType(hiddenStates.dtype)
            )
            return (compressedKV, bias)
        }

        guard let indexer else { return (compressedKV, nil) }
        let indices = indexer(hiddenStates, qResidual: qResidual, cache: cache)
        let compressedLength = compressedKV.dim(2)
        let entryIndices = MLXArray(Int32(0) ..< Int32(compressedLength)).reshaped(1, 1, 1, -1)
        let valid = indices .>= 0
        let safeIndices = MLX.where(valid, indices, MLXArray(Int32(compressedLength)))
        let hits = (safeIndices[0..., 0..., 0..., .newAxis] .== entryIndices).any(axis: -2)
        let bias = MLX.where(
            hits,
            MLXArray(0.0).asType(hiddenStates.dtype),
            MLXArray(-1.0e9).asType(hiddenStates.dtype)
        )[0..., .newAxis, 0..., 0...]
        return (compressedKV, bias)
    }
}

class DeepseekV4Attention: Module {
    let config: DeepseekV4Configuration
    let layerType: DeepseekV4Configuration.AttentionLayerType
    let numHeads: Int
    let headDim: Int
    let nopeHeadDim: Int
    let ropeHeadDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let nHeadsPerGroup: Int
    let scale: Float
    let eps: Float

    let rope: RoPELayer

    // Q low-rank projections
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear

    // Unified KV projection (K and V share the same projection)
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    // Grouped output projection
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear

    // Attention sink bias (per head, no .weight suffix)
    // Stored via update(parameters:) using the key "attn_sink"
    @ParameterInfo(key: "attn_sink") var attn_sink: MLXArray
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4AttentionCompressor?

    init(config: DeepseekV4Configuration, layerIndex: Int) {
        self.config = config
        self.layerType = config.layerTypes[layerIndex]
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.nopeHeadDim = config.nopeHeadDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.nHeadsPerGroup = config.numAttentionHeads / config.oGroups
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        // Q projections
        self._wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)

        // Unified KV: single head, headDim dimensional
        self._wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        // Grouped output projection
        // wo_a: Linear(nHeadsPerGroup * headDim, oGroups * oLoraRank) per group -> stored as [oGroups*oLoraRank, nHeadsPerGroup*headDim]
        self._woA.wrappedValue = Linear(
            nHeadsPerGroup * config.headDim, config.oGroups * config.oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(
            config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)

        // Attention sink: per-head bias [numAttentionHeads], applied to attention logits before softmax.
        // Shape matches numAttentionHeads (== qkRopeHeadDim in this architecture).
        self._attn_sink.wrappedValue = zeros([config.numAttentionHeads])
        if layerType != .sliding {
            self._compressor.wrappedValue = DeepseekV4AttentionCompressor(
                config: config,
                layerType: layerType)
        }

        // RoPE using compress_rope_theta (used for most layers with compress_ratio != 0)
        // We use a single rope config as a simplification
        let ropeBase = config.compressRopeTheta
        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: ropeBase,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    /// Grouped output projection matching the reference Python implementation.
    /// For QuantizedLinear wo_a: slices weight rows per group, calls quantizedMM.
    /// For plain Linear wo_a: uses batched matmul after weight reshape.
    /// Input:  [B, L, n_heads, head_dim]
    /// Output: [B, L, oGroups * oLoraRank]
    private func groupedOutputProjection(_ out: MLXArray) -> MLXArray {
        let B = out.dim(0)
        let L = out.dim(1)
        let groupFeat = numHeads * headDim / oGroups  // = nHeadsPerGroup * headDim

        // Flatten to [B, L, n_heads * head_dim] for easy group slicing
        let outFlat = out.reshaped(B, L, numHeads * headDim)

        if let qLinear = woA as? QuantizedLinear {
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat
                let gEnd = (g + 1) * groupFeat
                let rStart = g * oLoraRank
                let rEnd = (g + 1) * oLoraRank

                // Per-group input: [B, L, groupFeat]
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]
                // Slice weight rows for this group
                let wRows = qLinear.weight[rStart ..< rEnd]
                let sRows = qLinear.scales[rStart ..< rEnd]
                let bRows = qLinear.biases.map { $0[rStart ..< rEnd] }

                // quantizedMM: [B, L, groupFeat] @ dequant(wRows)^T -> [B, L, oLoraRank]
                let y = quantizedMM(
                    groupInput,
                    wRows,
                    scales: sRows,
                    biases: bRows,
                    transpose: true,
                    groupSize: qLinear.groupSize,
                    bits: qLinear.bits,
                    mode: qLinear.mode
                )
                pieces.append(y)
            }
            return concatenated(pieces, axis: -1)  // [B, L, oGroups * oLoraRank]
        } else {
            // Non-quantized fallback: per-group matmul (same structure as quantized path).
            // A single batched matmul would broadcast batch dims [B,L] against [oGroups],
            // which fails when L != oGroups, so we loop instead.
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat
                let gEnd = (g + 1) * groupFeat
                let rStart = g * oLoraRank
                let rEnd = (g + 1) * oLoraRank
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]  // [B, L, groupFeat]
                let wa_g = woA.weight[rStart ..< rEnd]  // [oLoraRank, groupFeat]
                pieces.append(matmul(groupInput, wa_g.T))  // [B, L, oLoraRank]
            }
            return concatenated(pieces, axis: -1)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // --- Query ---
        // Low-rank Q: wq_a -> q_norm -> wq_b
        let qResidual = qNorm(wqA(x))
        var q = wqB(qResidual)  // [B, L, n_heads * head_dim]
        q = q.reshaped(B, L, numHeads, headDim)
            .transposed(0, 2, 1, 3)  // [B, n_heads, L, head_dim]
        // Per-head RMS normalization (no learnable scale)
        q = headRmsNorm(q, eps: eps)

        // Split Q into nope and rope parts
        let qNope = q[.ellipsis, ..<nopeHeadDim]  // [B, n_heads, L, nope_head_dim]
        var qRope = q[.ellipsis, nopeHeadDim...]  // [B, n_heads, L, rope_head_dim]
        qRope = applyRotaryPosition(rope, to: qRope, offset: cache?.ropeOffset)
        let queries = concatenated([qNope, qRope], axis: -1)  // [B, n_heads, L, head_dim]

        // --- KV: k = v (reference: k = v = concat([k_nope, k_pe_roped])) ---
        let kv = kvNorm(wkv(x))  // [B, L, head_dim]
        let kvNope = kv[.ellipsis, ..<nopeHeadDim]
            .reshaped(B, L, 1, nopeHeadDim)
            .transposed(0, 2, 1, 3)  // [B, 1, L, nope_head_dim]
        var kvRope = kv[.ellipsis, nopeHeadDim...]
            .reshaped(B, L, 1, ropeHeadDim)
            .transposed(0, 2, 1, 3)  // [B, 1, L, rope_head_dim]
        kvRope = applyRotaryPosition(rope, to: kvRope, offset: cache?.ropeOffset)
        let kFull = concatenated([kvNope, kvRope], axis: -1)  // [B, 1, L, head_dim]
        // In reference k = v = kFull: both K and V have rope applied to their rope dims.
        // attentionWithCacheUpdate handles the KV cache update internally.
        let deepseekCache = cache as? DeepseekV4KVCache

        // --- Attention ---
        // Pass kFull as both keys and values; cache update happens inside.
        // Apply attn_sink (per-head bias) to attention logits when non-zero.
        // Cast to queries.dtype (bfloat16) - attn_sink may be loaded as float32 from the
        // checkpoint, but MLXFast.scaledDotProductAttention requires sinks to promote to
        // the output dtype (bfloat16); float32 does not satisfy this constraint.
        let sinksToUse: MLXArray? =
            attn_sink.sum().item(Float.self) != 0
            ? attn_sink.asType(queries.dtype)
            : nil
        var output: MLXArray
        if let compressor {
            let (cachedKeys, cachedValues) =
                cache?.update(keys: kFull, values: kFull) ?? (kFull, kFull)
            let compressed = compressor(x, qResidual: qResidual, cache: deepseekCache)
            let keys = concatenated([cachedKeys, compressed.kv.asType(cachedKeys.dtype)], axis: 2)
            let values = concatenated([cachedValues, compressed.kv.asType(cachedValues.dtype)], axis: 2)
            let compressedMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let bias = compressed.bias {
                let localKeyLength = cachedKeys.dim(2)
                let localMask =
                    switch mask {
                    case .array(let maskArray):
                        maskArray
                    default:
                        createCausalMask(n: L, offset: max(localKeyLength - L, 0))
                    }
                let localMask4D =
                    localMask.ndim == 2
                    ? localMask[.newAxis, .newAxis, 0..., 0...]
                    : localMask
                let joinedMask = concatenated(
                    [
                        MLX.where(
                            localMask4D,
                            MLXArray(0.0).asType(queries.dtype),
                            MLXArray(-1.0e9).asType(queries.dtype)),
                        bias.asType(queries.dtype),
                    ],
                    axis: -1)
                compressedMask = .array(joinedMask)
            } else {
                compressedMask = mask
            }
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: compressedMask, sinks: sinksToUse)
        } else {
            output = deepseekAttentionWithSinks(
                queries: queries,
                keys: kFull,
                values: kFull,
                cache: cache,
                scale: scale,
                mask: mask,
                sinks: sinksToUse
            )
        }
        output = output
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads, headDim)  // [B, L, n_heads, head_dim]

        // --- Grouped output projection ---
        let oLora = groupedOutputProjection(output)  // [B, L, oGroups * oLoraRank]
        return woB(oLora)
    }
}

// MARK: - MoE Components

/// Single FFN expert: SwiGLU with optional activation clamping.
class DeepseekV4Expert: Module, UnaryLayer {
    let swiguLimit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, swiguLimit: Float) {
        self.swiguLimit = swiguLimit
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = gateProj(x)
        var up = upProj(x)
        if swiguLimit > 0 {
            gate = clip(gate, min: -swiguLimit, max: swiguLimit)
            up = clip(up, min: -swiguLimit, max: swiguLimit)
        }
        return downProj(silu(gate) * up)
    }
}

/// MoE routing gate with score-based routing or token-hash routing via tid2eid.
class DeepseekV4Gate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoringFunc: String
    let isHash: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray  // [n_routed_experts, hidden_size]
    @ParameterInfo(key: "e_score_correction_bias") var e_score_correction_bias: MLXArray  // [n_routed_experts]
    @ParameterInfo(key: "tid2eid") var tid2eid: MLXArray  // [vocab_size, top_k]

    init(config: DeepseekV4Configuration, isHash: Bool) {
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.scoringFunc = config.scoringFunc
        self.isHash = isHash
        self._weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
        self._e_score_correction_bias.wrappedValue = zeros([config.nRoutedExperts])
        self._tid2eid.wrappedValue = zeros([config.vocabSize, config.numExpertsPerTok], dtype: .int32)
    }

    /// Hash-routing layers can load with only `tid2eid`; dense router weights are not required.
    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "e_score_correction_bias" || (isHash && parameter == "weight") {
            return  // keep zero-initialized default
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> (MLXArray, MLXArray) {
        // Compute expert scores
        let logits = x.matmul(weight.T)  // [B, S, n_experts]
        var scores: MLXArray
        switch scoringFunc {
        case "softmax":
            scores = softmax(logits, axis: -1)
        case "sigmoid":
            scores = sigmoid(logits)
        default:
            // sqrtsoftplus: sqrt(softplus(x)) = sqrt(log(1 + e^x))
            scores = sqrtSoftplus(logits)
        }

        let inds: MLXArray
        if isHash, let inputIds {
            let routed = tid2eid[inputIds.asType(.int32)].asType(.int32)
            inds = routed.dim(-1) > topK ? routed[.ellipsis, ..<topK] : routed
        } else {
            // Bias-shifted scores for top-k selection (bias not applied to routing weights)
            let scoresForChoice = scores + e_score_correction_bias
            inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        }

        // Gather weights using original (non-biased) scores
        var selectedScores = takeAlong(scores, inds, axis: -1)

        if topK > 1 && normTopkProb {
            let denominator = selectedScores.sum(axis: -1, keepDims: true) + 1e-20
            selectedScores = selectedScores / denominator
        }
        selectedScores = selectedScores * routedScalingFactor

        return (inds, selectedScores)
    }
}

/// Mixture-of-Experts layer with shared expert.
class DeepseekV4MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4Expert

    init(config: DeepseekV4Configuration, layerIndex: Int) {
        self.numExpertsPerTok = config.numExpertsPerTok

        // Routed experts (stacked via SwitchGLU, same as V3)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: { x in
                // SwiGLU with limit
                if config.swiguLimit > 0 {
                    let g = clip(x, min: -config.swiguLimit, max: config.swiguLimit)
                    return silu(g)
                }
                return silu(x)
            }
        )
        self.gate = DeepseekV4Gate(
            config: config,
            isHash: config.mlpLayerTypes[layerIndex] == .hashMoE)

        // Shared expert (1 expert, same intermediate size)
        self._sharedExperts.wrappedValue = DeepseekV4Expert(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize,
            swiguLimit: config.swiguLimit
        )
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray?) -> MLXArray {
        let (indices, scores) = gate(x, inputIds: inputIds)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        // Add shared expert output
        y = y + sharedExperts(x)
        return y
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, inputIds: nil)
    }
}

// MARK: - Decoder Block (with mHC Hyper-Connections)

public class DeepseekV4Block: Module {
    let config: DeepseekV4Configuration
    let layerIndex: Int

    // Key "attn" matches checkpoint path `layers.{l}.attn.*`
    @ModuleInfo(key: "attn") var selfAttn: DeepseekV4Attention
    // Plain var: property name "ffn" matches checkpoint path `layers.{l}.ffn.*`
    var ffn: DeepseekV4MoE
    // Key names match checkpoint: `attn_norm`, `ffn_norm`
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    // Hyper-Connection parameter bundles.
    // Underscore names match checkpoint paths: `hc_attn.fn/base/scale`, `hc_ffn.fn/base/scale`
    var hc_attn: HCParams
    var hc_ffn: HCParams

    init(config: DeepseekV4Configuration, layerIndex: Int) {
        self.config = config
        self.layerIndex = layerIndex

        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config, layerIndex: layerIndex)
        self.ffn = DeepseekV4MoE(config: config, layerIndex: layerIndex)

        self._attnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Initialize HC parameters (will be overwritten by weight loading)
        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let hcDim = hc * config.hiddenSize
        self.hc_attn = HCParams(
            fn: zeros([mixHc, hcDim]),
            base: zeros([mixHc]),
            scale: ones([3]))
        self.hc_ffn = HCParams(
            fn: zeros([mixHc, hcDim]),
            base: zeros([mixHc]),
            scale: ones([3]))
    }

    func callAsFunction(
        _ x: MLXArray,
        inputIds: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        // x: [B, S, hc, D]
        let residualAttn = x

        // HC pre for attention: [B,S,hc,D] -> [B,S,D]
        let (xAttn, postAttn, combAttn) = hcPre(
            x: residualAttn,
            hcFn: hc_attn.fn,
            hcScale: hc_attn.scale,
            hcBase: hc_attn.base,
            hcMult: config.hcMult,
            sinkhornIters: config.hcSinkhornIters,
            eps: config.hcEps
        )

        // Attention sublayer: [B,S,D] -> [B,S,D]
        let attnOut = selfAttn(attnNorm(xAttn), mask: mask, cache: cache)

        // HC post for attention: [B,S,D] -> [B,S,hc,D]
        let residualFfn = hcPost(x: attnOut, residual: residualAttn, post: postAttn, comb: combAttn)

        // HC pre for FFN: [B,S,hc,D] -> [B,S,D]
        let (xFfn, postFfn, combFfn) = hcPre(
            x: residualFfn,
            hcFn: hc_ffn.fn,
            hcScale: hc_ffn.scale,
            hcBase: hc_ffn.base,
            hcMult: config.hcMult,
            sinkhornIters: config.hcSinkhornIters,
            eps: config.hcEps
        )

        // FFN sublayer: [B,S,D] -> [B,S,D]
        let ffnOut = ffn(ffnNorm(xFfn), inputIds: inputIds)

        // HC post for FFN: [B,S,D] -> [B,S,hc,D]
        return hcPost(x: ffnOut, residual: residualFfn, post: postFfn, comb: combFfn)
    }
}

// MARK: - Inner Model

public class DeepseekV4ModelInner: Module {
    var config: DeepseekV4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    // HC head parameter bundle for final reduction [B,S,hc,D] -> [B,S,D]
    // Underscore name matches checkpoint path `model.hc_head.fn/base/scale`
    var hc_head: HCParams

    public var totalLayerCount: Int {
        layers.count - (MTPConfig.retainMTPWeights ? config.numNextnPredictLayers : 0)
    }

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        let retainMTP = MTPConfig.retainMTPWeights && config.numNextnPredictLayers > 0
        let totalCount = config.numHiddenLayers - (retainMTP ? 0 : config.numNextnPredictLayers)
        self.layers = (0 ..< totalCount).map {
            DeepseekV4Block(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // HC head parameters (will be overwritten by weight loading)
        let hc = config.hcMult
        self.hc_head = HCParams(
            fn: zeros([hc, hc * config.hiddenSize]),
            base: zeros([hc]),
            scale: ones([1]))
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        // x: [B, S] token IDs
        let B = x.dim(0)
        let S = x.dim(1)
        let hc = config.hcMult

        // Embed tokens: [B, S, D]
        var h = embedTokens(x)

        // Expand to hc copies: [B, S, hc, D]
        // Repeat along new hc dimension
        h = h.expandedDimensions(axis: 2)  // [B, S, 1, D]
        h = repeated(h, count: hc, axis: 2)  // [B, S, hc, D]

        // Create causal attention mask; reshape to 3D so dim(1)==S
        let hForMask = h.reshaped([B, S, hc * config.hiddenSize])  // [B, S, hc*D]
        let attentionMask = createAttentionMask(h: hForMask, cache: cache?.first)

        for (i, layer) in layers.prefix(totalLayerCount).enumerated() {
            h = layer(h, inputIds: x, mask: attentionMask, cache: cache?[i])
        }

        // HC head: [B, S, hc, D] -> [B, S, D]
        h = hcHead(
            x: h, hcFn: hc_head.fn, hcScale: hc_head.scale,
            hcBase: hc_head.base, eps: config.hcEps)

        return norm(h)
    }
}

// MARK: - Top-level Model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    /// One KV head per layer (unified KV, single head)
    public var kvHeads: [Int]

    var args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: 1, count: args.numHiddenLayers - args.numNextnPredictLayers)
        self.model = DeepseekV4ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if args.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        }
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        // 1. Dequantize FP8 weights (weight_scale_inv pattern, same as V3)
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var padded = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        // 2. Stack per-expert weights into SwitchGLU format (for non-pre-stacked checkpoints)
        // MLX quantized checkpoints already have stacked weights; this is a no-op for them.
        let instantiatedLayerCount = model.layers.count
        let stackLayerCount = min(args.numHiddenLayers, instantiatedLayerCount)
        for l in 0 ..< stackLayerCount {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).ffn.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let stacked = (0 ..< args.nRoutedExperts).map {
                            // Prefer dequantized value from newWeights (FP8 dequant), fall back to original
                            newWeights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]
                                ?? weights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).ffn.switch_mlp.\(projName).\(key)"] = MLX.stacked(
                            stacked)
                        for j in 0 ..< args.nRoutedExperts {
                            newWeights.removeValue(
                                forKey: "\(prefix).ffn.experts.\(j).\(projName).\(key)")
                        }
                    }
                }
            }
        }

        // 3. Filter out MTP (multi-token prediction) layers and rotary_emb keys.
        var finalWeights = [String: MLXArray]()
        for (key, value) in newWeights {
            // Drop rotary embedding precomputed frequencies
            if key.contains("rotary_emb.inv_freq") { continue }

            if key.starts(with: "model.layers.") {
                let parts = key.split(separator: ".")
                if parts.count >= 3, let layerIdx = Int(parts[2]) {
                    if layerIdx >= instantiatedLayerCount {
                        continue
                    }
                }
            }
            finalWeights[key] = value
        }
        return finalWeights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let mainLayerCount = args.numHiddenLayers - args.numNextnPredictLayers
        return (0 ..< mainLayerCount).map {
            DeepseekV4KVCache(layerType: args.layerTypes[$0], slidingWindow: args.slidingWindow)
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - MTPLanguageModel Conformance for DeepseekV4Model

/// DeepSeek V4 uses a different MTP scheme: the MTP layers are the last
/// `numNextnPredictLayers` standard transformer blocks (`model.layers[numMainLayers...]`).
/// They share the same architecture as the main blocks but operate on the final hidden state.
/// The main `lm_head` is reused for all MTP depth projections.
extension DeepseekV4Model: MTPLanguageModel {
    public func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?)
        -> [MLXArray]
    {
        let mtpLayers = model.layers.suffix(args.numNextnPredictLayers)
        guard MTPConfig.retainMTPWeights, !mtpLayers.isEmpty else {
            return [callAsFunction(inputs, cache: cache)]
        }

        // Run the main model body (excludes MTP layers \u2014 DeepseekV4ModelInner only
        // instantiates `numMain` blocks, so this is the standard forward pass)
        let mainHidden = model(inputs, cache: cache)
        let mainLogits =
            args.tieWordEmbeddings ? model.embedTokens.asLinear(mainHidden) : lmHead(mainHidden)
        var result = [mainLogits]

        // Chain MTP blocks stored in `model.mtpLayers`
        var prevHidden = mainHidden
        let B = prevHidden.dim(0)
        let S = prevHidden.dim(1)
        let hc = args.hcMult
        for (i, mtpLayer) in mtpLayers.enumerated() {
            let mtpCache = mtpCaches?[i]
            // Expand [B, S, D] -> [B, S, hc, D]
            var h = prevHidden.expandedDimensions(axis: 2)
            h = repeated(h, count: hc, axis: 2)

            let hForMask = h.reshaped([B, S, hc * args.hiddenSize])
            let attentionMask = createAttentionMask(h: hForMask, cache: mtpCache?.first)

            h = mtpLayer(h, inputIds: inputs, mask: attentionMask, cache: mtpCache?.first)

            // Reduce back to [B, S, D]
            prevHidden = hcHead(
                x: h, hcFn: model.hc_head.fn, hcScale: model.hc_head.scale,
                hcBase: model.hc_head.base, eps: args.hcEps)

            let mtpHidden = model.norm(prevHidden)
            let mtpLogits =
                args.tieWordEmbeddings ? model.embedTokens.asLinear(mtpHidden) : lmHead(mtpHidden)
            result.append(mtpLogits)
        }

        return result
    }

    public func makeMTPCaches(parameters: GenerateParameters?) -> [[KVCache]] {
        let mainLayerCount = args.numHiddenLayers - args.numNextnPredictLayers
        return (0 ..< args.numNextnPredictLayers).map { index in
            let layerIndex = mainLayerCount + index
            let layerType = args.layerTypes[min(layerIndex, args.layerTypes.count - 1)]
            return [DeepseekV4KVCache(layerType: layerType, slidingWindow: args.slidingWindow)]
        }
    }
}
