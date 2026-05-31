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

private func turboQuantRuntimeFailure(_ error: Error) -> TurboQuantRuntimeFailure {
    TurboQuantRuntimeFailure(error)
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
    case hybridTurboQuant(
        keys: MLXArray,
        values: MLXArray,
        selection: TurboQuantColdSelection,
        cache: HybridTurboQuantKVCache
    )

    public var keyLength: Int {
        switch self {
        case .raw(let keys, _):
            return keys.dim(2)
        case .quantized(let keys, _, _):
            return keys.0.dim(-2)
        case .turboQuant(let keys, _, _):
            return keys.layout.logicalLength
        case .hybridTurboQuant(let keys, _, let selection, let cache):
            let keyLength = keys.dim(2)
            if keyLength == cache.rawHotLength {
                return keyLength + selection.selectedTokenCount
            }
            return keyLength
        }
    }
}

private struct TurboQuantAttentionInputs {
    var queries: MLXArray
    var keys: MLXArray
    var values: MLXArray
}

private func canonicalTurboQuantInputs(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray
) -> TurboQuantAttentionInputs {
    TurboQuantAttentionInputs(
        queries: queries,
        keys: keys,
        values: values
    )
}

private func supportsPackedQuantizedAttention(
    keys: MLXArray,
    values: MLXArray,
    cache: any QuantizedKVCacheProtocol
) -> Bool {
    cache.groupSize > 0
        && keys.ndim == 4
        && values.ndim == 4
        && keys.dim(3).isMultiple(of: cache.groupSize)
        && values.dim(3).isMultiple(of: cache.groupSize)
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
    let canonical = canonicalTurboQuantInputs(queries: queries, keys: keys, values: values)
    guard let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        turboQuantCache.supportsCompressedAttention(
            queries: canonical.queries,
            keys: canonical.keys,
            values: canonical.values,
            mask: mask
        )
    else {
        return nil
    }

    let checkpoint = turboQuantCache.makeCompressedUpdateCheckpoint(
        appendingTokenCount: canonical.keys.dim(2))
    do {
        let (compressedKeys, compressedValues) = try turboQuantCache.updateCompressed(
            keys: canonical.keys,
            values: canonical.values
        )
        return try body(compressedKeys, compressedValues, turboQuantCache)
    } catch {
        turboQuantCache.restoreCompressedUpdateCheckpoint(checkpoint)
        turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
        throw turboQuantRuntimeFailure(error)
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

private func turboQuantPackedUpdateFallbackAfterFailure(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: any TurboQuantCompressedKVCacheProtocol,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    failure: Error
) -> (output: MLXArray, state: AttentionKVState)? {
    guard cache.fallbackPolicy == .packedAllowed || cache.fallbackPolicy == .compressedDecodeAllowed,
        let quantizedCache = cache as? any QuantizedKVCacheProtocol,
        supportsPackedQuantizedAttention(keys: keys, values: values, cache: quantizedCache)
    else {
        return nil
    }

    let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(keys: keys, values: values)
    let reason = "compressed cache update failed; using packed fallback: \(failure)"
    cache.recordFallback(
        TurboQuantFallbackResult(
            fromPath: cache.attentionDiagnostics.activeAttentionPath,
            toPath: .mlxPackedFallback,
            policy: .packedAllowed,
            reason: reason,
            isSemanticallyExact: false
        )
    )
    let output = quantizedScaledDotProductAttention(
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
    return (
        output,
        .quantized(keys: quantizedKeys, values: quantizedValues, cache: quantizedCache)
    )
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
        case .nativeMLXCompressed:
            .nativeMLXCompressed
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
        case .rawExactSDPA, .nativeMLXCompressed, .onlineFusedCompressed, .tiledOnlineFused, .twoStageQKAV:
            .exactRequired
        case .typedFailure:
            .fatalOnFailure
        }
    let result = TurboQuantFallbackResult(
        fromPath: cache.attentionDiagnostics.activeAttentionPath,
        toPath: toPath,
        policy: policy,
        reason: reason,
        isSemanticallyExact: path == .rawExactSDPA
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
    rawExactReason: String? = nil,
    stateAlreadyValidated: Bool = false
) throws -> MLXArray {
    let adjustedMask = adjustedAttentionMask(
        mask,
        keyLength: keyCode.layout.logicalLength
    )
    if !stateAlreadyValidated {
        try cache.validateCompressedState(context: "attention fallback ladder")
    }

    func rawExactOutput(reason: String) -> MLXArray? {
        if let exact = cache.exactRawStateIfComplete() {
            let output = exactScaledDotProductAttention(
                queries: queries,
                keys: exact.keys,
                values: exact.values,
                scale: scale,
                mask: adjustedMask,
                sinks: sinks
            )
            recordTurboQuantFallback(
                cache: cache,
                path: .rawExactSDPA,
                reason: reason
            )
            return output
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
                reason: reason
            )
            return output
        }
        return nil
    }

    var failures: [String] = []
    if cache.optimizationPolicy == .conservative,
        let output = rawExactOutput(
            reason: "conservative TurboQuant policy requires exact raw attention while the raw shadow is complete"
        )
    {
        return output
    }

    let cachedPath = cache.attentionDiagnostics.activeAttentionPath
    let cachedOnlineAdmission =
        stateAlreadyValidated
        && (cachedPath == .onlineFused || cachedPath == .tiledOnlineFused)
    let sparseVThreshold = cache.sparseValuePolicy.resolvedThreshold(
        runtimeMode: cache.resolvedRuntimeMode,
        contextLength: keyCode.layout.logicalLength
    )
    let nativeCapabilities = TurboQuantKernelAvailability.current.attentionCapabilities
    let nativeMaskSupported: Bool
    switch adjustedMask {
    case .none, .causal:
        nativeMaskSupported = true
    case .array, .arrays:
        nativeMaskSupported = false
    }
    let canUseNative =
        nativeCapabilities.nativeCompressedAttention == true
        && (sparseVThreshold == nil || nativeCapabilities.nativeSparseVSupport == true)
        && sinks == nil
        && queries.dim(2) <= 8
        && keyCode.layout.headDimension == valueCode.layout.headDimension
        && nativeMaskSupported

    if canUseNative {
        do {
            let result = try MLX.turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: TurboQuantNativeAttentionOptions(
                    scale: scale,
                    causal: adjustedMask.isCausal,
                    sparseVThreshold: sparseVThreshold ?? 0,
                    diagnostics: nativeCapabilities.nativeDiagnosticsSupport == true,
                    backendVersion: nativeCapabilities.nativeBackendVersion
                        ?? TurboQuantNativeAttentionOptions.backendVersion
                )
            )
            return result.output
        } catch {
            let reason = "native MLX compressed attention failed: \(error)"
            failures.append(reason)
            recordTurboQuantFallback(
                cache: cache,
                path: .nativeMLXCompressed,
                reason: reason
            )
        }
    } else {
        failures.append(
            nativeCapabilities.nativeFallbackReason
                ?? "native MLX compressed attention unsupported for this query/cache/mask"
        )
    }

    let canUseOnline =
        sinks == nil && cache.prefersOnlineFusedAttention
        && keyCode.layout.headDimension == valueCode.layout.headDimension
        && (cachedOnlineAdmission
            || MLX.turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: adjustedMask
            ))

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
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile,
                sparseVThreshold: sparseVThreshold
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
            kernelProfile: cache.attentionDiagnostics.selectedKernelProfile,
            sparseVThreshold: sparseVThreshold
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

    let failureReason = failures.joined(separator: "; ")

    switch cache.fallbackPolicy {
    case .fatalOnFailure:
        recordTurboQuantFallback(cache: cache, path: .typedFailure, reason: failureReason)
        throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(failureReason)
    case .exactRequired:
        if let output = rawExactOutput(reason: rawExactReason ?? failureReason) {
            return output
        }
        recordTurboQuantFallback(cache: cache, path: .typedFailure, reason: failureReason)
        throw TurboQuantRuntimeFailure.decodedFallbackUnavailable(failureReason)
    case .packedAllowed, .compressedDecodeAllowed:
        break
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

    if cache.fallbackPolicy == .packedAllowed {
        let reason = failures.joined(separator: "; ")
        if let output = rawExactOutput(reason: rawExactReason ?? reason) {
            return output
        }
        recordTurboQuantFallback(cache: cache, path: .typedFailure, reason: reason)
        throw TurboQuantRuntimeFailure.decodedFallbackUnavailable(reason)
    }

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

    if let output = rawExactOutput(reason: rawExactReason ?? failures.joined(separator: "; ")) {
        return output
    }

    let reason = failures.joined(separator: "; ")
    recordTurboQuantFallback(cache: cache, path: .typedFailure, reason: reason)
    throw TurboQuantRuntimeFailure.decodedFallbackUnavailable(reason)
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
    let canonical = canonicalTurboQuantInputs(queries: queries, keys: keys, values: values)
    guard let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        turboQuantCache.supportsCompressedAttention(
            queries: canonical.queries,
            keys: canonical.keys,
            values: canonical.values,
            mask: mask
        )
    else {
        return nil
    }

    let previousOffset = turboQuantCache.offset
    let checkpoint = turboQuantCache.makeCompressedUpdateCheckpoint(
        appendingTokenCount: canonical.keys.dim(2))
    do {
        let (compressedKeys, compressedValues) = try turboQuantCache.updateCompressed(
            keys: canonical.keys,
            values: canonical.values
        )
        let state = AttentionKVState.turboQuant(
            keys: compressedKeys,
            values: compressedValues,
            cache: turboQuantCache
        )

        if previousOffset == 0, turboQuantCache.prefersExactInitialPrefill {
            // The exact raw prefill output does not consume the compressed writes, so
            // schedule them explicitly to avoid carrying raw prompt tensors into decode.
            let output = exactScaledDotProductAttention(
                queries: canonical.queries,
                keys: canonical.keys,
                values: canonical.values,
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
            queries: canonical.queries,
            keyCode: compressedKeys,
            valueCode: compressedValues,
            cache: turboQuantCache,
            scale: scale,
            mask: mask,
            sinks: sinks,
            rawExactKeys: canonical.keys,
            rawExactValues: canonical.values,
            rawExactReason: "compressed prefill failed; using exact raw chunk output",
            stateAlreadyValidated: true
        )
        return (output, state)
    } catch {
        turboQuantCache.restoreCompressedUpdateCheckpoint(checkpoint)
        if let fallback = turboQuantPackedUpdateFallbackAfterFailure(
            queries: canonical.queries,
            keys: canonical.keys,
            values: canonical.values,
            cache: turboQuantCache,
            scale: scale,
            mask: mask,
            sinks: sinks,
            failure: turboQuantRuntimeFailure(error)
        ) {
            return fallback
        }
        turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
        throw turboQuantRuntimeFailure(error)
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
        if let affineCache = cache as? any NativeAffineInt4KVCacheProtocol {
            return try affineInt4NativeScaledDotProductAttention(
                queries: queries,
                quantizedKeys: keys,
                quantizedValues: values,
                scale: scale,
                mask: mask,
                sinks: sinks,
                groupSize: affineCache.groupSize
            )
        }
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
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "compressed attention failed and compressed state could not be decoded: \(error)"
            )
        }

    case .hybridTurboQuant(let keys, let values, let selection, let cache):
        let keysAreRawHot = keys.dim(2) == cache.rawHotLength
        var segmentedFallbackReason: String?
        if queries.dim(2) == 1, sinks == nil, keysAreRawHot {
            let coldSegments = cache.selectedColdCompressedSegments(selection: selection)
            if !coldSegments.isEmpty {
                do {
                    let output = try MLX.turboQuantMetalSegmentedScaledDotProductAttention(
                        queries: queries,
                        rawKeys: keys,
                        rawValues: values,
                        coldSegments: coldSegments,
                        scale: scale,
                        outputDType: queries.dtype,
                        sparseVThreshold: cache.sparseValuePolicy.resolvedThreshold(
                            runtimeMode: .capacityTurboQuant,
                            contextLength: cache.offset
                        )
                    )
                    cache.recordFallbackReason(nil)
                    cache.recordAttentionRoute(
                        cache.segmentedAttentionRoute(selection: selection),
                        selection: selection
                    )
                    return output
                } catch {
                    segmentedFallbackReason = "selected_segmented_attention_state_fallback:\(error)"
                    cache.recordFallbackReason(segmentedFallbackReason)
                }
            }
        }

        let attentionKeys: MLXArray
        let attentionValues: MLXArray
        if keysAreRawHot,
            let selectedCold = try cache.selectedColdState(
                selection: selection,
                outputDType: keys.dtype
            )
        {
            attentionKeys = concatenated([selectedCold.keys, keys], axis: 2)
            attentionValues = concatenated([selectedCold.values, values], axis: 2)
        } else {
            attentionKeys = keys
            attentionValues = values
        }

        let adjustedMask = turboQuantHybridMask(
            original: mask,
            queryLength: queries.dim(2),
            keyLength: attentionKeys.dim(2)
        )
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: attentionKeys,
            values: attentionValues,
            scale: scale,
            mask: adjustedMask,
            sinks: sinks
        )
        if segmentedFallbackReason == nil {
            cache.recordFallbackReason(nil)
        }
        cache.recordAttentionRoute(
            cache.decodedAttentionRoute(selection: selection),
            selection: selection
        )
        return output
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
        fatalError(
            "Debug-only non-throwing TurboQuant attention wrapper has no recoverable output: \(message)"
        )
    }
}

private func legacyTurboQuantPackedFallbackAfterFailure(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: any TurboQuantCompressedKVCacheProtocol,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    failure: Error
) -> (output: MLXArray, state: AttentionKVState)? {
    guard let quantizedCache = cache as? any QuantizedKVCacheProtocol else {
        return nil
    }

    _ = quantizedCache.getQuantizedState()
    let (quantizedKeys, quantizedValues) = quantizedCache.updateQuantized(
        keys: keys,
        values: values
    )
    let reason = "legacy non-throwing TurboQuant wrapper used packed fallback after: \(failure)"
    cache.recordFallback(
        TurboQuantFallbackResult(
            fromPath: cache.attentionDiagnostics.activeAttentionPath,
            toPath: .mlxPackedFallback,
            policy: .packedAllowed,
            reason: reason,
            isSemanticallyExact: false
        )
    )
    let output = quantizedScaledDotProductAttention(
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
    return (
        output,
        .quantized(keys: quantizedKeys, values: quantizedValues, cache: quantizedCache)
    )
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
        if let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
            let fallback = legacyTurboQuantPackedFallbackAfterFailure(
                queries: queries,
                keys: keys,
                values: values,
                cache: turboQuantCache,
                scale: scale,
                mask: mask,
                sinks: sinks,
                failure: turboQuantRuntimeFailure(error)
            )
        {
            return fallback.output
        }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FATAL_FALLBACK") {
            fatalError(String(describing: error))
        }
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
    if let throughputCache = cache as? ThroughputTurboQuantKVCache {
        let (cachedKeys, cachedValues) = try throughputCache.updateThroughput(
            keys: keys,
            values: values
        )
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
    if let hybridCache = cache as? HybridTurboQuantKVCache {
        return try turboQuantHybridAttentionThrowing(
            queries: queries,
            keys: keys,
            values: values,
            cache: hybridCache,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
    if queries.dim(2) > 1, queries.dim(2) == keys.dim(2),
        let prefill = try turboQuantCompressedPrefillAttentionThrowing(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    {
        return prefill
    }
    let useTurboQuantInputs = cache is TurboQuantCompressedKVCacheProtocol
    let turboQuantInputs =
        useTurboQuantInputs
        ? canonicalTurboQuantInputs(queries: queries, keys: keys, values: values)
        : TurboQuantAttentionInputs(queries: queries, keys: keys, values: values)
    if let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        turboQuantCache.supportsCompressedAttention(
            queries: turboQuantInputs.queries,
            keys: turboQuantInputs.keys,
            values: turboQuantInputs.values,
            mask: mask
        )
    {
        let checkpoint = turboQuantCache.makeCompressedUpdateCheckpoint(
            appendingTokenCount: turboQuantInputs.keys.dim(2))
        do {
            let (compressedKeys, compressedValues) = try turboQuantCache.updateCompressed(
                keys: turboQuantInputs.keys,
                values: turboQuantInputs.values
            )
            let state = AttentionKVState.turboQuant(
                keys: compressedKeys,
                values: compressedValues,
                cache: turboQuantCache
            )
            let output = try turboQuantAttentionFallbackLadder(
                queries: turboQuantInputs.queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                cache: turboQuantCache,
                scale: scale,
                mask: mask,
                sinks: sinks,
                stateAlreadyValidated: true
            )
            return (output, state)
        } catch {
            turboQuantCache.restoreCompressedUpdateCheckpoint(checkpoint)
            if let fallback = turboQuantPackedUpdateFallbackAfterFailure(
                queries: turboQuantInputs.queries,
                keys: turboQuantInputs.keys,
                values: turboQuantInputs.values,
                cache: turboQuantCache,
                scale: scale,
                mask: mask,
                sinks: sinks,
                failure: turboQuantRuntimeFailure(error)
            ) {
                return fallback
            }
            turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
            throw turboQuantRuntimeFailure(error)
        }
    }
    if let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        let quantizedKVCache = cache as? QuantizedKVCacheProtocol,
        !supportsPackedQuantizedAttention(
            keys: turboQuantInputs.keys,
            values: turboQuantInputs.values,
            cache: quantizedKVCache
        )
    {
        if cache is RotatingTurboQuantKVCache {
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
        let reason =
            "TurboQuant compressed attention unavailable and packed fallback does not support head dimensions k=\(turboQuantInputs.keys.dim(3)) v=\(turboQuantInputs.values.dim(3)) with group size \(quantizedKVCache.groupSize)"
        turboQuantCache.recordCompressedAttentionFailure(reason)
        throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(reason)
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let keys = useTurboQuantInputs ? turboQuantInputs.keys : keys
        let values = useTurboQuantInputs ? turboQuantInputs.values : values
        let queries = useTurboQuantInputs ? turboQuantInputs.queries : queries
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
            if let fallback = legacyTurboQuantPackedFallbackAfterFailure(
                queries: queries,
                keys: keys,
                values: values,
                cache: turboQuantCache,
                scale: scale,
                mask: mask,
                sinks: sinks,
                failure: turboQuantRuntimeFailure(error)
            ) {
                return fallback
            }
        }
        let message = String(describing: error)
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FATAL_FALLBACK") {
            fatalError(message)
        }
        fatalError("No semantically correct non-throwing attention fallback: \(message)")
    }
}
