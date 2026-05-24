import Foundation
import MLX

public typealias QuantizedKVStorage = (MLXArray, MLXArray, MLXArray?)

/// Cached attention state that can be passed between model layers without assuming raw KV arrays.
public enum AttentionKVState {
    case raw(keys: MLXArray, values: MLXArray)
    case quantized(
        keys: QuantizedKVStorage,
        values: QuantizedKVStorage,
        cache: any QuantizedKVCacheProtocol
    )
    case turboQuant(
        keys: TurboQuantAttentionCode,
        values: TurboQuantAttentionCode,
        cache: any TurboQuantCompressedKVCacheProtocol
    )

    public var keyLength: Int {
        switch self {
        case .raw(let keys, _):
            keys.dim(2)
        case .quantized(let keys, _, _):
            keys.0.dim(-2)
        case .turboQuant(let keys, _, _):
            keys.layout.logicalLength
        }
    }
}

public func attentionKeyLengthAfterUpdate(cache: KVCache?, keys: MLXArray) -> Int {
    let updatedLength = (cache?.offset ?? 0) + keys.dim(2)
    if let maxSize = cache?.maxSize {
        return min(updatedLength, maxSize)
    }
    return updatedLength
}

public func adjustedAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode?,
    keyLength: Int,
    dtype: DType? = nil
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard let mask else { return .none }

    func adjusted(_ maskArray: MLXArray) -> MLXArray {
        let sliced =
            maskArray.dim(-1) == keyLength
            ? maskArray
            : maskArray[.ellipsis, 0 ..< keyLength]
        if let dtype, sliced.dtype != .bool {
            return sliced.asType(dtype)
        }
        return sliced
    }

    switch mask {
    case .array(let maskArray):
        return .array(adjusted(maskArray))
    case .arrays(let maskArrays):
        guard let firstMask = maskArrays.first else { return .none }
        return .array(adjusted(firstMask))
    case .causal, .none:
        return mask
    }
}

public func withTurboQuantCompressedCacheUpdate<T>(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    _ body: (
        TurboQuantAttentionCode, TurboQuantAttentionCode, any TurboQuantCompressedKVCacheProtocol
    )
        throws -> T
) -> T? {
    guard var turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        turboQuantCache.supportsCompressedAttention(
            queries: queries,
            keys: keys,
            values: values,
            mask: mask
        )
    else {
        return nil
    }

    let previousOffset = turboQuantCache.offset
    let previousState = turboQuantCache.state.map { $0[.ellipsis] }
    let previousMetaState = turboQuantCache.metaState
    func restorePreviousState() {
        turboQuantCache.metaState = previousMetaState
        turboQuantCache.state = previousState
        if let baseCache = turboQuantCache as? BaseKVCache {
            baseCache.offset = previousOffset
        }
    }
    do {
        let (compressedKeys, compressedValues) = try turboQuantCache.updateCompressed(
            keys: keys,
            values: values
        )
        return try body(compressedKeys, compressedValues, turboQuantCache)
    } catch {
        restorePreviousState()
        turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
        return nil
    }
}

func packedQuantizedAttentionFallback(
    queries: MLXArray,
    cache: any TurboQuantCompressedKVCacheProtocol,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray? {
    guard let quantizedCache = cache as? any QuantizedKVCacheProtocol,
        let (quantizedKeys, quantizedValues) = quantizedCache.getQuantizedState()
    else {
        return nil
    }

    return quantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        scale: scale,
        mask: mask,
        sinks: sinks,
        groupSize: quantizedCache.groupSize,
        bits: quantizedCache.bits,
        mode: quantizedCache.mode
    )
}

private func turboQuantAttentionStorageArrays(_ code: TurboQuantAttentionCode) -> [MLXArray] {
    [
        code.packedMagnitudes,
        code.signs,
        code.highPrecisionMask,
        code.residualSigns,
        code.scales,
    ]
}

private func asyncEvalTurboQuantAttentionStorage(
    keys: TurboQuantAttentionCode,
    values: TurboQuantAttentionCode
) {
    asyncEval(turboQuantAttentionStorageArrays(keys) + turboQuantAttentionStorageArrays(values))
}

/// Install compressed KV during prefill while honoring the cache optimization policy.
private func turboQuantCompressedPrefillAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> (output: MLXArray, state: AttentionKVState)? {
    guard queries.dim(2) > 1, queries.dim(2) == keys.dim(2) else {
        return nil
    }
    guard var turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        turboQuantCache.supportsCompressedAttention(
            queries: queries,
            keys: keys,
            values: values,
            mask: mask
        )
    else {
        return nil
    }

    let previousOffset = turboQuantCache.offset
    let previousState = turboQuantCache.state.map { $0[.ellipsis] }
    let previousMetaState = turboQuantCache.metaState
    func restorePreviousState() {
        turboQuantCache.metaState = previousMetaState
        turboQuantCache.state = previousState
        if let baseCache = turboQuantCache as? BaseKVCache {
            baseCache.offset = previousOffset
        }
    }
    do {
        let (compressedKeys, compressedValues) = try turboQuantCache.updateCompressed(
            keys: keys,
            values: values
        )
        let state = AttentionKVState.turboQuant(
            keys: compressedKeys,
            values: compressedValues,
            cache: turboQuantCache
        )

        if previousOffset == 0, turboQuantCache.prefersExactInitialPrefill {
            // The exact raw prefill output does not consume the compressed writes, so
            // schedule them explicitly to avoid carrying raw prompt tensors into decode.
            asyncEvalTurboQuantAttentionStorage(keys: compressedKeys, values: compressedValues)
            return (
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: mask,
                    sinks: sinks
                ),
                state
            )
        }

        if let output = try? turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: compressedKeys,
            valueCode: compressedValues,
            scale: scale,
            mask: adjustedAttentionMask(
                mask,
                keyLength: compressedKeys.layout.logicalLength
            ),
            sinks: sinks,
            preferOnlineFused: turboQuantCache.prefersOnlineFusedAttention,
            kernelProfile: turboQuantCache.attentionDiagnostics.selectedKernelProfile
        ) {
            return (output, state)
        }

        if let decodedKeys = try? turboQuantMetalDecodeAttention(
            compressedKeys, outputDType: queries.dtype),
            let decodedValues = try? turboQuantMetalDecodeAttention(
                compressedValues, outputDType: queries.dtype)
        {
            return (
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: decodedKeys,
                    values: decodedValues,
                    scale: scale,
                    mask: adjustedAttentionMask(
                        mask,
                        keyLength: compressedKeys.layout.logicalLength
                    ),
                    sinks: sinks
                ),
                state
            )
        }

        restorePreviousState()
        turboQuantCache.recordCompressedAttentionFailure(
            "TurboQuant compressed prefill produced no runnable attention path"
        )
        return nil
    } catch {
        restorePreviousState()
        turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
        return nil
    }
}

public func attentionWithKVState(
    queries: MLXArray,
    state: AttentionKVState,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    switch state {
    case .raw(let keys, let values):
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )

    case .quantized(let keys, let values, let cache):
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: keys,
            quantizedValues: values,
            scale: scale,
            mask: mask,
            sinks: sinks,
            groupSize: cache.groupSize,
            bits: cache.bits,
            mode: cache.mode
        )

    case .turboQuant(let keys, let values, let cache):
        do {
            return try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: keys,
                valueCode: values,
                scale: scale,
                mask: mask,
                sinks: sinks,
                preferOnlineFused: cache.prefersOnlineFusedAttention,
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
            )
        } catch {
            cache.recordCompressedAttentionFailure(String(describing: error))
            if let output = packedQuantizedAttentionFallback(
                queries: queries,
                cache: cache,
                scale: scale,
                mask: mask,
                sinks: sinks
            ) {
                return output
            }
            if let decodedKeys = try? turboQuantMetalDecodeAttention(
                keys, outputDType: queries.dtype),
                let decodedValues = try? turboQuantMetalDecodeAttention(
                    values, outputDType: queries.dtype)
            {
                return MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: decodedKeys,
                    values: decodedValues,
                    scale: scale,
                    mask: mask,
                    sinks: sinks
                )
            }
            fatalError(
                "TurboQuant compressed attention failed and compressed state could not be decoded: \(error)"
            )
        }
    }
}

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
///   - sinks: Optional attention sinks. Compressed TurboQuant applies sinks through the
///     two-stage path and uses the packed/dequantized fallback when compressed kernels
///     are unavailable.
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
    attentionWithCacheUpdateReturningState(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: scale,
        mask: mask,
        sinks: sinks
    ).output
}

public func attentionWithCacheUpdateReturningState(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> (output: MLXArray, state: AttentionKVState) {
    guard let cache else {
        let state = AttentionKVState.raw(keys: keys, values: values)
        return (
            attentionWithKVState(
                queries: queries,
                state: state,
                scale: scale,
                mask: mask,
                sinks: sinks
            ),
            state
        )
    }
    if let prefill = turboQuantCompressedPrefillAttention(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: scale,
        mask: mask,
        sinks: sinks
    ) {
        return prefill
    }
    if let result = withTurboQuantCompressedCacheUpdate(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        mask: mask,
        { compressedKeys, compressedValues, turboQuantCache in
            let output = try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                scale: scale,
                mask: mask,
                sinks: sinks,
                preferOnlineFused: turboQuantCache.prefersOnlineFusedAttention,
                kernelProfile: turboQuantCache.attentionDiagnostics.selectedKernelProfile
            )
            return (
                output,
                AttentionKVState.turboQuant(
                    keys: compressedKeys,
                    values: compressedValues,
                    cache: turboQuantCache
                )
            )
        }
    ) {
        return result
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        let state = AttentionKVState.quantized(
            keys: quantizedKeys,
            values: quantizedValues,
            cache: quantizedKVCache
        )
        return (
            attentionWithKVState(
                queries: queries,
                state: state,
                scale: scale,
                mask: mask,
                sinks: sinks
            ),
            state
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        let state = AttentionKVState.raw(keys: cachedKeys, values: cachedValues)
        return (
            attentionWithKVState(
                queries: queries,
                state: state,
                scale: scale,
                mask: mask,
                sinks: sinks
            ),
            state
        )
    }
}
