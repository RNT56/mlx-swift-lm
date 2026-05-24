import Foundation
import MLX

public typealias QuantizedKVStorage = (MLXArray, MLXArray, MLXArray?)

public enum TurboQuantAttentionStateError: Error, CustomStringConvertible, Equatable {
    case compressedAttentionUnavailable(String)
    case noSemanticallyCorrectFallback(String)

    public var description: String {
        switch self {
        case .compressedAttentionUnavailable(let message):
            "TurboQuant compressed attention unavailable: \(message)"
        case .noSemanticallyCorrectFallback(let message):
            "TurboQuant attention has no semantically correct fallback: \(message)"
        }
    }
}

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

public func withTurboQuantCompressedCacheUpdateThrowing<T>(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    _ body: (
        TurboQuantAttentionCode, TurboQuantAttentionCode, any TurboQuantCompressedKVCacheProtocol
    )
        throws -> T
) throws -> T? {
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
        try turboQuantCache.validateCompressedState(context: "compressed cache update")
        return try body(compressedKeys, compressedValues, turboQuantCache)
    } catch {
        restorePreviousState()
        turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
        throw error
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
    do {
        return try withTurboQuantCompressedCacheUpdateThrowing(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            mask: mask,
            body
        )
    } catch {
        if let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol {
            let reason = "compressed attention failed: \(String(describing: error))"
            turboQuantCache.recordFallback(
                TurboQuantFallbackResult(
                    fromPath: turboQuantCache.attentionDiagnostics.activeAttentionPath,
                    toPath: nil,
                    policy: .fatalOnFailure,
                    reason: reason,
                    isSemanticallyExact: false
                )
            )
        }
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

private func recordTurboQuantFallback(
    cache: any TurboQuantCompressedKVCacheProtocol,
    path: TurboQuantFallbackPath,
    reason: String
) {
    let toPath: TurboQuantAttentionPath? =
        switch path {
        case .onlineFusedCompressed:
            .onlineFused
        case .tiledOnlineFused:
            .tiledOnlineFused
        case .twoStageQKAV:
            .twoStageCompressed
        case .packedQuantizedSDPA:
            .mlxPackedFallback
        case .decodedCompressedSDPA, .rawExactSDPA:
            .baseline
        case .typedFailure:
            nil
        }
    let policy: TurboQuantFallbackPolicy =
        switch path {
        case .packedQuantizedSDPA:
            .packedAllowed
        case .decodedCompressedSDPA:
            .compressedDecodeAllowed
        case .rawExactSDPA, .onlineFusedCompressed, .tiledOnlineFused, .twoStageQKAV:
            .exactRequired
        case .typedFailure:
            .fatalOnFailure
        }
    let result = TurboQuantFallbackResult(
        fromPath: cache.attentionDiagnostics.activeAttentionPath,
        toPath: toPath,
        policy: policy,
        reason: reason,
        isSemanticallyExact: path == .rawExactSDPA || path == .decodedCompressedSDPA
    )
    cache.recordFallback(result)
    turboQuantTrace(
        "fallback from \(result.fromPath.rawValue) to \(result.toPath?.rawValue ?? "none"): \(result.reason)"
    )
}

private func turboQuantTrace(_ message: String) {
    guard TurboQuantRuntimeControl.enabled("TURBOQUANT_TRACE") else { return }
    FileHandle.standardError.write(Data("TurboQuant: \(message)\n".utf8))
}

private func exactScaledDotProductAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: keys,
        values: values,
        scale: scale,
        mask: mask,
        sinks: sinks
    )
}

private func turboQuantAttentionFallbackLadder(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    cache: any TurboQuantCompressedKVCacheProtocol,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    rawExactKeys: MLXArray? = nil,
    rawExactValues: MLXArray? = nil,
    rawExactReason: String? = nil
) throws -> MLXArray {
    let adjustedMask = adjustedAttentionMask(
        mask,
        keyLength: keyCode.layout.logicalLength
    )
    try cache.validateCompressedState(context: "attention fallback ladder")

    var failures: [String] = []
    let canUseOnline =
        sinks == nil && cache.prefersOnlineFusedAttention
        && keyCode.layout.headDimension == valueCode.layout.headDimension
        && MLX.turboQuantMetalSupportsOnlineFusedAttention(
            queries: queries,
            keyCode: keyCode,
            mask: adjustedMask
        )

    if canUseOnline {
        do {
            return try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                scale: scale,
                mask: adjustedMask,
                sinks: sinks,
                preferOnlineFused: true,
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
            )
        } catch {
            let reason = "online fused compressed attention failed: \(error)"
            failures.append(reason)
        }
    } else {
        failures.append("online fused compressed attention unsupported for this query/cache/mask")
    }

    do {
        let output = try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: keyCode,
            valueCode: valueCode,
            scale: scale,
            mask: adjustedMask,
            sinks: sinks,
            preferOnlineFused: false,
            kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
        )
        if !failures.isEmpty {
            recordTurboQuantFallback(
                cache: cache,
                path: .twoStageQKAV,
                reason: failures.joined(separator: "; ")
            )
        }
        return output
    } catch {
        let reason = "two-stage compressed QK/AV attention failed: \(error)"
        failures.append(reason)
    }

    if let output = packedQuantizedAttentionFallback(
        queries: queries,
        cache: cache,
        scale: scale,
        mask: adjustedMask,
        sinks: sinks
    ) {
        recordTurboQuantFallback(
            cache: cache,
            path: .packedQuantizedSDPA,
            reason: failures.joined(separator: "; ")
        )
        return output
    }
    failures.append("packed quantized SDPA fallback unavailable")

    do {
        let (decodedKeys, decodedValues) = try cache.decodedCompressedState(
            outputDType: queries.dtype)
        let output = exactScaledDotProductAttention(
            queries: queries,
            keys: decodedKeys,
            values: decodedValues,
            scale: scale,
            mask: adjustedMask,
            sinks: sinks
        )
        recordTurboQuantFallback(
            cache: cache,
            path: .decodedCompressedSDPA,
            reason: failures.joined(separator: "; ")
        )
        return output
    } catch {
        failures.append("decode compressed K/V fallback failed: \(error)")
    }

    if let rawExactKeys, let rawExactValues {
        let output = exactScaledDotProductAttention(
            queries: queries,
            keys: rawExactKeys,
            values: rawExactValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
        recordTurboQuantFallback(
            cache: cache,
            path: .rawExactSDPA,
            reason: rawExactReason ?? failures.joined(separator: "; ")
        )
        return output
    }

    let reason = failures.joined(separator: "; ")
    recordTurboQuantFallback(cache: cache, path: .typedFailure, reason: reason)
    throw TurboQuantAttentionStateError.compressedAttentionUnavailable(reason)
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
    values: TurboQuantAttentionCode,
    output: MLXArray? = nil
) {
    var arrays = turboQuantAttentionStorageArrays(keys) + turboQuantAttentionStorageArrays(values)
    if let output {
        arrays.append(output)
    }
    asyncEval(arrays)
}

/// Install compressed KV during prefill while honoring the cache optimization policy.
private func turboQuantCompressedPrefillAttentionThrowing(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) throws -> (output: MLXArray, state: AttentionKVState)? {
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
        try turboQuantCache.validateCompressedState(context: "compressed prefill append")
        let state = AttentionKVState.turboQuant(
            keys: compressedKeys,
            values: compressedValues,
            cache: turboQuantCache
        )

        if previousOffset == 0, turboQuantCache.prefersExactInitialPrefill {
            // The exact raw prefill output does not consume the compressed writes, so
            // schedule them explicitly to avoid carrying raw prompt tensors into decode.
            let output = exactScaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
            asyncEvalTurboQuantAttentionStorage(
                keys: compressedKeys,
                values: compressedValues,
                output: output
            )
            recordTurboQuantFallback(
                cache: turboQuantCache,
                path: .rawExactSDPA,
                reason: "preserving exact initial prefill logits while committing compressed cache"
            )
            return (
                output,
                state
            )
        }

        let output = try turboQuantAttentionFallbackLadder(
            queries: queries,
            keyCode: compressedKeys,
            valueCode: compressedValues,
            cache: turboQuantCache,
            scale: scale,
            mask: mask,
            sinks: sinks,
            rawExactKeys: keys,
            rawExactValues: values,
            rawExactReason: "compressed prefill failed; using exact raw chunk output"
        )
        return (output, state)
    } catch {
        restorePreviousState()
        turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
        throw error
    }
}

private func turboQuantCompressedPrefillAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> (output: MLXArray, state: AttentionKVState)? {
    try? turboQuantCompressedPrefillAttentionThrowing(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: scale,
        mask: mask,
        sinks: sinks
    )
}

public func attentionWithKVStateThrowing(
    queries: MLXArray,
    state: AttentionKVState,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) throws -> MLXArray {
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
            return try turboQuantAttentionFallbackLadder(
                queries: queries,
                keyCode: keys,
                valueCode: values,
                cache: cache,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        } catch {
            cache.recordCompressedAttentionFailure(String(describing: error))
            throw TurboQuantAttentionStateError.compressedAttentionUnavailable(
                "compressed attention failed and compressed state could not be decoded: \(error)"
            )
        }
    }
}

@available(
    *, deprecated,
    message: "Use attentionWithKVStateThrowing so TurboQuant failures remain recoverable."
)
public func attentionWithKVState(
    queries: MLXArray,
    state: AttentionKVState,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> MLXArray {
    do {
        return try attentionWithKVStateThrowing(
            queries: queries,
            state: state,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    } catch {
        if case .turboQuant(_, _, let cache) = state {
            cache.recordCompressedAttentionFailure(String(describing: error))
        }
        let message = String(describing: error)
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FATAL_FALLBACK") {
            fatalError(message)
        }
        fatalError("No semantically correct non-throwing attention fallback: \(message)")
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
    do {
        return try attentionWithCacheUpdateReturningStateThrowing(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: sinks
        ).output
    } catch {
        fatalError("No semantically correct non-throwing attention fallback: \(error)")
    }
}

public func attentionWithCacheUpdateReturningStateThrowing(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) throws -> (output: MLXArray, state: AttentionKVState) {
    guard let cache else {
        let state = AttentionKVState.raw(keys: keys, values: values)
        return (
            try attentionWithKVStateThrowing(
                queries: queries,
                state: state,
                scale: scale,
                mask: mask,
                sinks: sinks
            ),
            state
        )
    }
    if let prefill = try turboQuantCompressedPrefillAttentionThrowing(
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
    if var turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
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
            try turboQuantCache.validateCompressedState(context: "decode compressed update")
            let state = AttentionKVState.turboQuant(
                keys: compressedKeys,
                values: compressedValues,
                cache: turboQuantCache
            )
            let output = try turboQuantAttentionFallbackLadder(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                cache: turboQuantCache,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
            return (output, state)
        } catch {
            restorePreviousState()
            turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
            throw error
        }
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
            try attentionWithKVStateThrowing(
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
            try attentionWithKVStateThrowing(
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

@available(
    *, deprecated,
    message:
        "Use attentionWithCacheUpdateReturningStateThrowing so TurboQuant failures remain recoverable."
)
public func attentionWithCacheUpdateReturningState(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil
) -> (output: MLXArray, state: AttentionKVState) {
    do {
        return try attentionWithCacheUpdateReturningStateThrowing(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    } catch {
        if let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol {
            let reason = "compressed attention failed: \(String(describing: error))"
            turboQuantCache.recordFallback(
                TurboQuantFallbackResult(
                    fromPath: turboQuantCache.attentionDiagnostics.activeAttentionPath,
                    toPath: nil,
                    policy: .fatalOnFailure,
                    reason: reason,
                    isSemanticallyExact: false
                )
            )
        }
        let message = String(describing: error)
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FATAL_FALLBACK") {
            fatalError(message)
        }
        fatalError("No semantically correct non-throwing attention fallback: \(message)")
    }
}
