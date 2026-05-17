import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
///   - sinks: Optional attention sinks. Compressed TurboQuant attention does not support sinks yet,
///     so sink-enabled calls use the packed/dequantized fallback paths.
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
    if sinks != nil, let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol {
        turboQuantCache.recordCompressedAttentionFailure("attention sinks use packed fallback")
    }
    if sinks == nil,
        var turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        turboQuantCache.supportsCompressedAttention(
            queries: queries,
            keys: keys,
            values: values,
            mask: mask
        )
    {
        let previousOffset = turboQuantCache.offset
        let previousState = turboQuantCache.state.map { $0[.ellipsis] }
        let previousMetaState = turboQuantCache.metaState
        do {
            let (compressedKeys, compressedValues) = try turboQuantCache.updateCompressed(
                keys: keys,
                values: values
            )
            return try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                scale: scale,
                mask: mask,
                preferOnlineFused: turboQuantCache.prefersOnlineFusedAttention,
                kernelProfile: turboQuantCache.attentionDiagnostics.selectedKernelProfile
            )
        } catch {
            turboQuantCache.metaState = previousMetaState
            turboQuantCache.state = previousState
            if let baseCache = turboQuantCache as? BaseKVCache {
                baseCache.offset = previousOffset
            }
            turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
            // Fall through to the packed quantized cache path. The compressed
            // path is opportunistic and must never make generation fail.
        }
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        if let sinks {
            let decodedKeys = dequantized(
                quantizedKeys.0,
                scales: quantizedKeys.1,
                biases: quantizedKeys.2,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
            let decodedValues = dequantized(
                quantizedValues.0,
                scales: quantizedValues.1,
                biases: quantizedValues.2,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: decodedKeys,
                values: decodedValues,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        }
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
}
