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

private func turboQuantContiguousStorage(_ storage: QuantizedKVStorage) -> QuantizedKVStorage {
    (
        storage.0.contiguous(stream: .gpu),
        storage.1.contiguous(stream: .gpu),
        storage.2.map { $0.contiguous(stream: .gpu) }
    )
}

private func turboQuantPrefixStorage(
    _ storage: QuantizedKVStorage,
    tokenCount: Int
) -> QuantizedKVStorage {
    guard tokenCount >= 0, storage.0.ndim >= 3, storage.0.dim(-2) != tokenCount else {
        return storage
    }
    let tokens = 0 ..< min(tokenCount, storage.0.dim(-2))
    return (
        storage.0[.ellipsis, tokens, 0...],
        storage.1[.ellipsis, tokens, 0...],
        storage.2.map { $0[.ellipsis, tokens, 0...] }
    )
}

private func turboQuantContiguousPolarWHTCode(
    _ code: TurboQuantPolarWHTAttentionValueCode
) -> TurboQuantPolarWHTAttentionValueCode {
    var contiguous = code
    contiguous.packedIndices = contiguous.packedIndices.contiguous(stream: .gpu)
    contiguous.norms = contiguous.norms.contiguous(stream: .gpu)
    return contiguous
}

private func turboQuantUsesFailClosedHybridPolarWHTUpdate(
    _ cache: any TurboQuantCompressedKVCacheProtocol
) -> Bool {
    cache.kvCodec == .polarWHT
        && cache.activeBackend == .metalPolarWHT
        && cache.precisionPolicy.key.isHighPrecision
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

private let polarWHTReferenceHybridPullOutMaxWorkItems =
    ProcessInfo.processInfo.environment["TQ_POLARWHT_REFERENCE_PULLOUT_MAX_WORK_ITEMS"]
    .flatMap(Int.init) ?? 8_388_608

private let polarWHTReferenceHybridMetalAVEnabled: Bool = {
    guard let value = ProcessInfo.processInfo.environment["TQ_POLARWHT_REFERENCE_METAL_AV"]?
        .lowercased()
    else {
        return true
    }
    return !["0", "false", "no", "off", "disabled"].contains(value)
}()

private func polarWHTReferencePullOutValueAttention(
    weights: MLXArray,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    batchSize: Int,
    queryHeadCount: Int,
    queryLength: Int,
    kvHeadCount: Int,
    repeats: Int
) throws -> (output: MLXArray, usedMetal: Bool)? {
    let outputVectors = batchSize * queryHeadCount * queryLength
    let workItems = outputVectors * valueCode.layout.logicalLength
    if polarWHTReferenceHybridMetalAVEnabled {
        do {
            let metalWeights =
                repeats > 1
                ? weights.reshaped([
                    batchSize, queryHeadCount, queryLength, valueCode.layout.logicalLength,
                ])
                : weights
            let metalOutput = try turboQuantMetalPolarWHTAV(
                attentionWeights: metalWeights.contiguous(stream: .gpu),
                valueCode: valueCode,
                outputDType: .float32
            )
            return (metalOutput, true)
        } catch {
            // Keep the reference path portable: failed experimental JIT AV falls through to
            // the CPU pull-out budget and then the decoded-value fallback.
        }
    }

    guard workItems <= polarWHTReferenceHybridPullOutMaxWorkItems else {
        return nil
    }

    let keyLength = valueCode.layout.logicalLength
    let headDimension = valueCode.layout.headDimension
    let weightStorage = weights.asType(.float32).asArray(Float.self)
    var output = [Float]()
    output.reserveCapacity(outputVectors * headDimension)
    for batch in 0 ..< batchSize {
        for queryHead in 0 ..< queryHeadCount {
            let kvHead = queryHead / repeats
            let repeatIndex = queryHead % repeats
            for queryToken in 0 ..< queryLength {
                var tokenWeights = [Float](repeating: 0, count: keyLength)
                for keyToken in 0 ..< keyLength {
                    let sourceIndex: Int
                    if repeats > 1 {
                        sourceIndex =
                            ((((batch * kvHeadCount + kvHead) * repeats + repeatIndex)
                                * queryLength + queryToken) * keyLength) + keyToken
                    } else {
                        sourceIndex =
                            (((batch * kvHeadCount + kvHead) * queryLength + queryToken)
                                * keyLength) + keyToken
                    }
                    tokenWeights[keyToken] = weightStorage[sourceIndex]
                }
                output += try turboQuantPolarWHTReferenceAccumulateAttentionValue(
                    weights: tokenWeights,
                    code: valueCode,
                    batchIndex: batch,
                    kvHeadIndex: kvHead
                )
            }
        }
    }
    return (MLXArray(output, [batchSize, queryHeadCount, queryLength, headDimension]), false)
}

private func turboQuantMaskedAffineScores(
    _ scores: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXArray {
    var scores = scores
    switch mask {
    case .causal:
        let (qLength, kLength) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qLength) + MLXArray(kLength - qLength)
        let kIndices = MLXArray(0 ..< kLength)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1),
            expandedDimensions(kIndices, axis: -2)
        )
        scores = MLX.where(
            causalMask,
            scores,
            MLXArray(-Float.greatestFiniteMagnitude)
        )

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(
                maskArray,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude)
            )
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(
                    maskArray,
                    scores,
                    MLXArray(-Float.greatestFiniteMagnitude)
                )
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }
    return scores
}

private func turboQuantThresholdedAttentionWeights(
    _ weights: MLXArray,
    threshold: Float?
) -> MLXArray {
    guard let threshold, threshold > 0 else { return weights }
    return MLX.where(
        weights .>= threshold,
        weights,
        MLXArray.zeros(like: weights)
    )
}

private enum HybridAffineKPolarWHTValueRoute {
    case decodedNative
    case decodedNativeContiguousRetry
    case decodedGeneric
    case segmentedPackedNative
    case packedNative
    case packedNativeContiguousRetry
    case packedGeneric
}

private func hybridAffineKPolarWHTValueAttention(
    queries: MLXArray,
    quantizedKeys: QuantizedKVStorage,
    tailQuantizedKeys: QuantizedKVStorage? = nil,
    quantizedCache: any QuantizedKVCacheProtocol,
    valueCode: TurboQuantPolarWHTAttentionValueCode,
    tailValueCode: TurboQuantPolarWHTAttentionValueCode? = nil,
    decodedValues: MLXArray?,
    decodedValueLayout: MLX.TurboQuantAttentionLayout?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sparseVThreshold: Float?
) throws -> (
    output: MLXArray, usedDecodedValueBuffer: Bool, route: HybridAffineKPolarWHTValueRoute
) {
    let batchSize = queries.dim(0)
    let queryHeadCount = queries.dim(1)
    let queryLength = queries.dim(2)
    let headDimension = queries.dim(3)
    let kvHeadCount = valueCode.layout.kvHeadCount
    let keyLength = valueCode.layout.logicalLength
    let disableFusedHybrid =
        TurboQuantRuntimeControl.enabled("TURBOQUANT_DISABLE_HYBRID_POLARWHT_FUSED")
    guard quantizedKeys.0.ndim >= 4,
        quantizedKeys.0.dim(-3) == kvHeadCount,
        quantizedKeys.0.dim(-2) >= keyLength,
        queryHeadCount % kvHeadCount == 0
    else {
        throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
            "hybrid affine K scoring requires key sidecar shape to match PolarWHT value layout"
        )
    }

    if tailQuantizedKeys != nil || tailValueCode != nil {
        guard let tailQuantizedKeys, let tailValueCode else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "segmented hybrid tail requires both affine key and PolarWHT value tail sidecars"
            )
        }
        guard sparseVThreshold == nil || sparseVThreshold == 0,
            quantizedCache.mode == .affine,
            quantizedCache.bits == 8,
            let baseKeyBiases = quantizedKeys.2,
            let tailKeyBiases = tailQuantizedKeys.2,
            tailValueCode.layout.batchSize == batchSize,
            tailValueCode.layout.kvHeadCount == kvHeadCount,
            tailValueCode.layout.headDimension == headDimension,
            tailValueCode.layout.logicalLength > 0
        else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "segmented hybrid tail sidecars are present but shape or precision gates failed"
            )
        }
        let compactBaseForSegmentedHybrid =
            TurboQuantRuntimeControl.enabled("TURBOQUANT_COMPACT_SEGMENTED_HYBRID_BASE")
        let baseKeyWeight =
            compactBaseForSegmentedHybrid ? quantizedKeys.0.contiguous(stream: .gpu) : quantizedKeys.0
        let baseKeyScales =
            compactBaseForSegmentedHybrid ? quantizedKeys.1.contiguous(stream: .gpu) : quantizedKeys.1
        let segmentedBaseKeyBiases =
            compactBaseForSegmentedHybrid ? baseKeyBiases.contiguous(stream: .gpu) : baseKeyBiases
        let contiguousTailKeyWeight = tailQuantizedKeys.0.contiguous(stream: .gpu)
        let contiguousTailKeyScales = tailQuantizedKeys.1.contiguous(stream: .gpu)
        let contiguousTailKeyBiases = tailKeyBiases.contiguous(stream: .gpu)
        let baseValueCode =
            compactBaseForSegmentedHybrid ? turboQuantContiguousPolarWHTCode(valueCode) : valueCode
        let contiguousTailValueCode = turboQuantContiguousPolarWHTCode(tailValueCode)
        guard let nativeOutput = try MLX
            .turboQuantMetalSegmentedHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
                queries: queries,
                baseKeyWeight: baseKeyWeight,
                baseKeyScales: baseKeyScales,
                baseKeyBiases: segmentedBaseKeyBiases,
                tailKeyWeight: contiguousTailKeyWeight,
                tailKeyScales: contiguousTailKeyScales,
                tailKeyBiases: contiguousTailKeyBiases,
                keyGroupSize: quantizedCache.groupSize,
                baseValueCode: baseValueCode,
                tailValueCode: contiguousTailValueCode,
                scale: scale,
                mask: mask,
                sinks: nil,
                sparseVThreshold: sparseVThreshold,
                outputDType: .float32
            )
        else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "segmented hybrid native attention was not admitted; q=\(queries.shape) baseKey=\(quantizedKeys.0.shape) baseValueLength=\(valueCode.layout.logicalLength) tailKey=\(tailQuantizedKeys.0.shape) tailValueLength=\(tailValueCode.layout.logicalLength)"
            )
        }
        return (nativeOutput, false, .segmentedPackedNative)
    }

    if let decodedValues,
        (sparseVThreshold == nil || sparseVThreshold == 0),
        decodedValues.ndim == 4,
        decodedValues.dim(0) == batchSize,
        decodedValues.dim(1) == kvHeadCount,
        decodedValues.dim(2) >= keyLength,
        decodedValues.dim(3) == headDimension
    {
        let decodedLayout = decodedValueLayout
        let decodedLogicalLength = decodedLayout?.logicalLength ?? decodedValues.dim(2)
        let decodedRingOffset = decodedLayout?.ringOffset ?? 0
        let decodedPinnedPrefixLength = decodedLayout?.pinnedPrefixLength ?? 0
        let decodedValuesAreLogical =
            decodedLogicalLength == keyLength
            && decodedValues.dim(2) == keyLength
            && decodedRingOffset == 0
            && decodedPinnedPrefixLength == 0
        guard decodedLogicalLength == keyLength else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "hybrid decoded value buffer length \(decodedLogicalLength) does not match PolarWHT value length \(keyLength)"
            )
        }
        if !disableFusedHybrid, quantizedCache.mode == .affine, quantizedCache.bits == 8,
            let keyBiases = quantizedKeys.2
        {
            func nativeDecodedValueOutput(
                keys: QuantizedKVStorage,
                decodedValues candidateDecodedValues: MLXArray
            ) throws -> MLXArray? {
                guard let candidateKeyBiases = keys.2 else { return nil }
                return try MLX
                    .turboQuantMetalHybridAffineK8DecodedValueScaledDotProductAttentionIfSupported(
                        queries: queries,
                        keyWeight: keys.0,
                        keyScales: keys.1,
                        keyBiases: candidateKeyBiases,
                        keyGroupSize: quantizedCache.groupSize,
                        decodedValues: candidateDecodedValues,
                        decodedValueLogicalLength: decodedLogicalLength,
                        decodedValueRingOffset: decodedRingOffset,
                        decodedValuePinnedPrefixLength: decodedPinnedPrefixLength,
                        scale: scale,
                        mask: mask,
                        sinks: nil,
                        sparseVThreshold: sparseVThreshold,
                        outputDType: .float32
                    )
            }

            if let nativeOutput = try nativeDecodedValueOutput(
                keys: quantizedKeys,
                decodedValues: decodedValues
            ) {
                return (nativeOutput, true, .decodedNative)
            }
            let contiguousKeys = turboQuantContiguousStorage((
                quantizedKeys.0,
                quantizedKeys.1,
                keyBiases
            ))
            if let nativeOutput = try nativeDecodedValueOutput(
                keys: contiguousKeys,
                decodedValues: decodedValues.contiguous(stream: .gpu)
            ) {
                return (nativeOutput, true, .decodedNativeContiguousRetry)
            }
        }

        guard decodedValuesAreLogical else {
            return try hybridAffineKPolarWHTValueAttention(
                queries: queries,
                quantizedKeys: quantizedKeys,
                quantizedCache: quantizedCache,
                valueCode: valueCode,
                decodedValues: nil,
                decodedValueLayout: nil,
                scale: scale,
                mask: mask,
                sparseVThreshold: sparseVThreshold
            )
        }

        let repeats = queryHeadCount / kvHeadCount
        var scaledQueries = queries * scale
        var qKeys = turboQuantPrefixStorage(quantizedKeys, tokenCount: keyLength)
        var groupedScores = false
        if repeats > 1 {
            scaledQueries = scaledQueries.reshaped([
                batchSize, kvHeadCount, repeats, queryLength, headDimension,
            ])
            qKeys = (
                expandedDimensions(qKeys.0, axis: -3),
                expandedDimensions(qKeys.1, axis: -3),
                qKeys.2.map { expandedDimensions($0, axis: -3) }
            )
            groupedScores = true
        }

        var scores = quantizedMM(
            scaledQueries,
            qKeys.0,
            scales: qKeys.1,
            biases: qKeys.2,
            transpose: true,
            groupSize: quantizedCache.groupSize,
            bits: quantizedCache.bits,
            mode: quantizedCache.mode
        )
        scores = turboQuantMaskedAffineScores(scores, mask: mask)
        let weights = softmax(scores.asType(.float32), axis: -1)
        let output: MLXArray
        if groupedScores {
            output = matmul(
                weights.asType(decodedValues.dtype),
                expandedDimensions(decodedValues, axis: 2)
            )
            .reshaped([batchSize, queryHeadCount, queryLength, headDimension])
        } else {
            output = matmul(weights.asType(decodedValues.dtype), decodedValues)
        }
        return (output.asType(.float32), true, .decodedGeneric)
    }

    if !disableFusedHybrid, quantizedCache.mode == .affine, quantizedCache.bits == 8,
        let keyBiases = quantizedKeys.2
    {
        func nativePackedValueOutput(
            keys: QuantizedKVStorage,
            valueCode candidateValueCode: TurboQuantPolarWHTAttentionValueCode
        ) throws -> MLXArray? {
            guard let candidateKeyBiases = keys.2 else { return nil }
            return try MLX
                .turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
                    queries: queries,
                    keyWeight: keys.0,
                    keyScales: keys.1,
                    keyBiases: candidateKeyBiases,
                    keyGroupSize: quantizedCache.groupSize,
                    valueCode: candidateValueCode,
                    scale: scale,
                    mask: mask,
                    sinks: nil,
                    sparseVThreshold: sparseVThreshold,
                    outputDType: .float32
                )
        }

        if let fusedOutput = try nativePackedValueOutput(
            keys: quantizedKeys,
            valueCode: valueCode
        ) {
            return (fusedOutput, false, .packedNative)
        }
        if let fusedOutput = try nativePackedValueOutput(
            keys: turboQuantContiguousStorage((quantizedKeys.0, quantizedKeys.1, keyBiases)),
            valueCode: turboQuantContiguousPolarWHTCode(valueCode)
        ) {
            return (fusedOutput, false, .packedNativeContiguousRetry)
        }
    }

    let repeats = queryHeadCount / kvHeadCount
    var scaledQueries = queries * scale
    var qKeys = turboQuantPrefixStorage(quantizedKeys, tokenCount: keyLength)
    if repeats > 1 {
        scaledQueries = scaledQueries.reshaped([
            batchSize, kvHeadCount, repeats, queryLength, headDimension,
        ])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2.map { expandedDimensions($0, axis: -3) }
        )
    }

    var scores = quantizedMM(
        scaledQueries,
        qKeys.0,
        scales: qKeys.1,
        biases: qKeys.2,
        transpose: true,
        groupSize: quantizedCache.groupSize,
        bits: quantizedCache.bits,
        mode: quantizedCache.mode
    )
    scores = turboQuantMaskedAffineScores(scores, mask: mask)

    var weights = softmax(scores.asType(.float32), axis: -1)
    if repeats > 1 {
        weights = weights.reshaped([
            batchSize, queryHeadCount, queryLength, keyLength,
        ])
    }
    weights = turboQuantThresholdedAttentionWeights(weights, threshold: sparseVThreshold)
    let output = try turboQuantMetalPolarWHTAV(
        attentionWeights: weights.contiguous(stream: .gpu),
        valueCode: valueCode,
        outputDType: .float32
    )
    return (output, false, .packedGeneric)
}

private func turboQuantPolarWHTAttentionDiagnostics(
    from sparseDiagnostics: TurboQuantSparseValueDiagnostics?,
    path: TurboQuantAttentionPath
) -> TurboQuantNativeAttentionDiagnostics? {
    guard let sparseDiagnostics else { return nil }
    let kernelKind: Int32 =
        path == .metalHybridK8PolarWHTValue ? 42 : 41
    let considered = min(sparseDiagnostics.considered, Int(Int32.max))
    let skipped = min(sparseDiagnostics.skipped, Int(Int32.max))
    let retained = max(0, considered - skipped)
    return TurboQuantNativeAttentionDiagnostics(values: [
        Int32(TurboQuantNativeAttentionOptions.backendVersion),
        kernelKind,
        0,
        0,
        Int32(skipped),
        Int32(considered),
        0,
        sparseDiagnostics.enabled ? 1 : 0,
        0,
        0,
        0,
        0,
        Int32(considered),
        Int32(retained),
    ])
}

private func metalPolarWHTScaledDotProductAttention(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    turboQuantCache: any TurboQuantCompressedKVCacheProtocol,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) throws -> MLXArray? {
    let availability = TurboQuantKernelAvailability.current
    let requiresHybridValueKernel = turboQuantCache.precisionPolicy.key.isHighPrecision
    let polarWHTAttentionAvailable =
        requiresHybridValueKernel
            ? availability.attentionCapabilities.hybridK8PolarWHTValueAttention
            : availability.supportsMetalPolarWHTAttention
    guard turboQuantCache.kvCodec == .polarWHT,
        turboQuantCache.activeBackend == .metalPolarWHT,
        polarWHTAttentionAvailable
    else {
        return nil
    }
    guard queries.ndim == 4, queries.dim(2) == 1 else {
        return nil
    }
    guard let valueCode = turboQuantCache.polarWHTValueStateForAttention else {
        throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
            "metalPolarWHT attention requested but PolarWHT value sidecar is unavailable"
        )
    }
    let sparseSelection = turboQuantCache.sparseValueSelection.resolved(
        runtimeMode: turboQuantCache.resolvedRuntimeMode,
        contextLength: valueCode.layout.logicalLength,
        policy: turboQuantCache.sparseValuePolicy
    )
    let threshold = sparseSelection.resolvedThreshold
    let output: MLXArray
    let reason: String
    let attentionPath: TurboQuantAttentionPath
    let sparseDiagnostics: TurboQuantSparseValueDiagnostics?
    let canUsePolarWHTQK = !turboQuantCache.precisionPolicy.key.isHighPrecision
    if canUsePolarWHTQK,
        var sidecarKeyCode = turboQuantCache.polarWHTKeyStateForAttention,
        sidecarKeyCode.layout.batchSize == valueCode.layout.batchSize,
        sidecarKeyCode.layout.kvHeadCount == valueCode.layout.kvHeadCount,
        sidecarKeyCode.layout.logicalLength == valueCode.layout.logicalLength,
        sidecarKeyCode.layout.headDimension == valueCode.layout.headDimension
    {
        sidecarKeyCode = turboQuantContiguousPolarWHTCode(sidecarKeyCode)
        let result = try MLX.turboQuantMetalPolarWHTScaledDotProductAttentionWithDiagnostics(
            queries: queries.contiguous(stream: .gpu),
            keyCode: sidecarKeyCode,
            valueCode: turboQuantContiguousPolarWHTCode(valueCode),
            scale: scale,
            mask: mask,
            sinks: sinks,
            sparseVThreshold: threshold,
            outputDType: queries.dtype
        )
        output = result.output
        attentionPath = .metalPolarWHTHybrid
        sparseDiagnostics = result.sparseValueDiagnostics
        reason =
            threshold.map {
                "metalPolarWHT used sidecar QK plus WHT-pulled V with sparse threshold \($0)"
            }
            ?? "metalPolarWHT used sidecar QK plus WHT-pulled V"
    } else if requiresHybridValueKernel {
        guard let quantizedCache = turboQuantCache as? any QuantizedKVCacheProtocol else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "metalHybridK8PolarWHTValue requires a QuantizedKVCache-compatible affine key cache"
            )
        }
        guard let affineKeyState = turboQuantCache.hybridAffineKeyStateForAttention else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "metalHybridK8PolarWHTValue requires an affine K8 key sidecar"
            )
        }
        let decodedValueBufferEnabled =
            !TurboQuantRuntimeControl.enabled("TURBOQUANT_DISABLE_POLARWHT_DECODED_VALUE_BUFFER")
            && (
                turboQuantCache.resolvedRuntimeMode == .throughputTurboQuant
                    || TurboQuantRuntimeControl.enabled(
                        "TURBOQUANT_ENABLE_POLARWHT_DECODED_VALUE_BUFFER")
            )
        let hybridResult = try hybridAffineKPolarWHTValueAttention(
            queries: queries.contiguous(stream: .gpu),
            quantizedKeys: affineKeyState,
            tailQuantizedKeys: turboQuantCache.hybridAffineKeyTailStateForAttention,
            quantizedCache: quantizedCache,
            valueCode: valueCode,
            tailValueCode: turboQuantCache.polarWHTValueTailStateForAttention,
            decodedValues: decodedValueBufferEnabled
                ? turboQuantCache.polarWHTDecodedValueState : nil,
            decodedValueLayout: decodedValueBufferEnabled
                ? turboQuantCache.polarWHTDecodedValueLayout : nil,
            scale: scale,
            mask: mask,
            sparseVThreshold: threshold
        )
        output = hybridResult.output
        attentionPath = .metalHybridK8PolarWHTValue
        sparseDiagnostics = nil
        switch hybridResult.route {
        case .decodedNative:
            reason =
                "metalHybridK8PolarWHTValue used fused resident affine K8 scoring plus buffered PolarWHT-decoded V"
        case .decodedNativeContiguousRetry:
            reason =
                "metalHybridK8PolarWHTValue used fused contiguous-retry affine K8 scoring plus buffered PolarWHT-decoded V"
        case .decodedGeneric:
            reason =
                "metalHybridK8PolarWHTValue used separate affine K8 scoring plus buffered PolarWHT-decoded V"
        case .segmentedPackedNative:
            reason =
                "metalHybridK8PolarWHTValue used segmented affine K8 scoring plus WHT-pulled V tail"
        case .packedNative:
            reason =
                threshold.map {
                    "metalHybridK8PolarWHTValue used fused resident affine K8 scoring plus WHT-pulled V with sparse threshold \($0)"
                }
                ?? "metalHybridK8PolarWHTValue used fused resident affine K8 scoring plus WHT-pulled V"
        case .packedNativeContiguousRetry:
            reason =
                threshold.map {
                    "metalHybridK8PolarWHTValue used fused contiguous-retry affine K8 scoring plus WHT-pulled V with sparse threshold \($0)"
                }
                ?? "metalHybridK8PolarWHTValue used fused contiguous-retry affine K8 scoring plus WHT-pulled V"
        case .packedGeneric:
            reason =
                threshold.map {
                    "metalHybridK8PolarWHTValue used separate affine K8 scoring plus WHT-pulled V with sparse threshold \($0)"
                }
                ?? "metalHybridK8PolarWHTValue used separate affine K8 scoring plus WHT-pulled V"
        }
    } else {
        guard keyCode.layout.batchSize == valueCode.layout.batchSize,
            keyCode.layout.kvHeadCount == valueCode.layout.kvHeadCount,
            keyCode.layout.logicalLength == valueCode.layout.logicalLength,
            keyCode.layout.headDimension == valueCode.layout.headDimension
        else {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "metalPolarWHT hybrid requires aligned compressed key and PolarWHT value layouts"
            )
        }
        let result = try MLX.turboQuantMetalHybridPolarWHTValueScaledDotProductAttentionWithDiagnostics(
            queries: queries.contiguous(stream: .gpu),
            keyCode: keyCode,
            valueCode: turboQuantContiguousPolarWHTCode(valueCode),
            scale: scale,
            mask: mask,
            sinks: sinks,
            sparseVThreshold: threshold,
            outputDType: queries.dtype
        )
        output = result.output
        attentionPath =
            turboQuantCache.precisionPolicy.key.isHighPrecision
                ? .metalHybridK8PolarWHTValue : .metalPolarWHTHybrid
        sparseDiagnostics = result.sparseValueDiagnostics
        let routeName =
            attentionPath == .metalHybridK8PolarWHTValue
                ? "metalHybridK8PolarWHTValue"
                : "metalPolarWHT"
        reason =
            threshold.map {
                "\(routeName) used compressed key scoring plus WHT-pulled V with sparse threshold \($0)"
            }
            ?? "\(routeName) used compressed key scoring plus WHT-pulled V"
    }
    let lastRecordedPath = turboQuantCache.fallbackResults.last?.toPath
    let lastRecordedPolicy = turboQuantCache.fallbackResults.last?.policy
    let lastRecordedReason = turboQuantCache.fallbackResults.last?.reason
    if lastRecordedPath != attentionPath || lastRecordedPolicy != .exactRequired
        || lastRecordedReason != reason
    {
        turboQuantCache.recordFallback(
            TurboQuantFallbackResult(
                fromPath: turboQuantCache.attentionDiagnostics.activeAttentionPath,
                toPath: attentionPath,
                policy: .exactRequired,
                reason: reason,
                isSemanticallyExact: false
            )
        )
    }
    turboQuantCache.recordPolarWHTAttentionDiagnostics(
        turboQuantPolarWHTAttentionDiagnostics(from: sparseDiagnostics, path: attentionPath),
        path: attentionPath
    )
    return output.asType(queries.dtype)
}

private func polarWHTReferenceHybridScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: QuantizedKVStorage,
    cache: any QuantizedKVCacheProtocol,
    turboQuantCache: any TurboQuantCompressedKVCacheProtocol,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) throws -> MLXArray? {
    guard turboQuantCache.kvCodec == .polarWHT,
        turboQuantCache.requestedBackend == .polarWHTReference
            || turboQuantCache.activeBackend == .polarWHTReference
    else {
        return nil
    }
    guard sinks == nil else {
        return nil
    }
    guard let valueCode = turboQuantCache.polarWHTValueState else {
        return nil
    }
    guard queries.ndim == 4, quantizedKeys.0.ndim == 4 else {
        return nil
    }

    let batchSize = queries.dim(0)
    let queryHeadCount = queries.dim(1)
    let queryLength = queries.dim(2)
    let headDimension = queries.dim(3)
    let kvHeadCount = quantizedKeys.0.dim(-3)
    let keyLength = quantizedKeys.0.dim(-2)
    guard batchSize == valueCode.layout.batchSize,
        kvHeadCount == valueCode.layout.kvHeadCount,
        keyLength == valueCode.layout.logicalLength,
        headDimension == valueCode.layout.headDimension,
        kvHeadCount > 0,
        queryHeadCount % kvHeadCount == 0
    else {
        return nil
    }
    guard queryLength == 1 else {
        return nil
    }

    let repeats = queryHeadCount / kvHeadCount
    var scores: MLXArray?
    var usedPolarWHTQK = false
    let canUsePolarWHTQK =
        queryLength == 1 && !turboQuantCache.precisionPolicy.key.isHighPrecision
    if canUsePolarWHTQK,
        var keyCode = turboQuantCache.polarWHTKeyState,
        keyCode.layout.batchSize == batchSize,
        keyCode.layout.kvHeadCount == kvHeadCount,
        keyCode.layout.logicalLength == keyLength,
        keyCode.layout.headDimension == headDimension
    {
        do {
            keyCode.packedIndices = keyCode.packedIndices.contiguous(stream: .gpu)
            keyCode.norms = keyCode.norms.contiguous(stream: .gpu)
            var polarScores = try turboQuantMetalPolarWHTQK(
                queries: queries.contiguous(stream: .gpu),
                keyCode: keyCode,
                scale: scale,
                mask: mask
            )
            if repeats > 1 {
                polarScores = polarScores.reshaped([
                    batchSize, kvHeadCount, repeats, queryLength, keyLength,
                ])
            }
            scores = polarScores
            usedPolarWHTQK = true
        } catch {
            usedPolarWHTQK = false
        }
    }

    if !usedPolarWHTQK {
        var scaledQueries = queries * scale
        var qKeys = quantizedKeys
        if repeats > 1 {
            scaledQueries = scaledQueries.reshaped([
                batchSize, kvHeadCount, repeats, queryLength, headDimension,
            ])
            qKeys = (
                expandedDimensions(qKeys.0, axis: -3),
                expandedDimensions(qKeys.1, axis: -3),
                qKeys.2.map { expandedDimensions($0, axis: -3) }
            )
        }

        var affineScores = quantizedMM(
            scaledQueries,
            qKeys.0,
            scales: qKeys.1,
            biases: qKeys.2,
            transpose: true,
            groupSize: cache.groupSize,
            bits: cache.bits,
            mode: cache.mode
        )

        switch mask {
        case .causal:
            let (qLength, kLength) = (affineScores.dim(-2), affineScores.dim(-1))
            let qIndices = MLXArray(0 ..< qLength) + MLXArray(kLength - qLength)
            let kIndices = MLXArray(0 ..< kLength)
            let causalMask = greaterEqual(
                expandedDimensions(qIndices, axis: -1),
                expandedDimensions(kIndices, axis: -2)
            )
            affineScores = MLX.where(
                causalMask,
                affineScores,
                MLXArray(-Float.greatestFiniteMagnitude)
            )

        case .array(let maskArray):
            if maskArray.dtype == .bool {
                affineScores = MLX.where(
                    maskArray,
                    affineScores,
                    MLXArray(-Float.greatestFiniteMagnitude)
                )
            } else {
                affineScores = affineScores + maskArray
            }

        case .arrays(let maskArrays):
            if let maskArray = maskArrays.first {
                if maskArray.dtype == .bool {
                    affineScores = MLX.where(
                        maskArray,
                        affineScores,
                        MLXArray(-Float.greatestFiniteMagnitude)
                    )
                } else {
                    affineScores = affineScores + maskArray
                }
            }

        case .none:
            break
        }
        scores = affineScores
    }

    guard let scores else { return nil }
    let weights = softmax(scores, axis: -1)
    let output: MLXArray
    let valuePathReason: String
    let scoringPathReason =
        usedPolarWHTQK ? "Metal PolarWHT QK scoring" : "affine K scoring"
    if let pullOutResult = try polarWHTReferencePullOutValueAttention(
        weights: weights,
        valueCode: valueCode,
        batchSize: batchSize,
        queryHeadCount: queryHeadCount,
        queryLength: queryLength,
        kvHeadCount: kvHeadCount,
        repeats: repeats
    ) {
        output = pullOutResult.output
        valuePathReason =
            pullOutResult.usedMetal
            ? "PolarWHT reference hybrid used \(scoringPathReason) with Metal WHT-pulled value accumulation"
            : "PolarWHT reference hybrid used \(scoringPathReason) with CPU WHT-pulled value accumulation"
    } else {
        var decodedValues = try turboQuantPolarWHTReferenceDecodeAttentionValues(valueCode)
            .asType(queries.dtype)
        if repeats > 1 {
            decodedValues = expandedDimensions(decodedValues, axis: -3)
        }
        var decodedOutput = matmul(weights.asType(decodedValues.dtype), decodedValues)
        if repeats > 1 {
            decodedOutput = decodedOutput.reshaped([
                batchSize, queryHeadCount, queryLength, decodedOutput.dim(-1),
            ])
        }
        output = decodedOutput
        valuePathReason =
            "PolarWHT reference hybrid used \(scoringPathReason) with decoded PolarWHT values"
    }

    turboQuantCache.recordFallback(
        TurboQuantFallbackResult(
            fromPath: turboQuantCache.attentionDiagnostics.activeAttentionPath,
            toPath: .polarWHTReferenceHybrid,
            policy: .packedAllowed,
            reason: valuePathReason,
            isSemanticallyExact: false
        )
    )
    return output.asType(queries.dtype)
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

private func prepareTurboQuantSparseAuxiliaryState(
    cache: any TurboQuantCompressedKVCacheProtocol,
    selection: TurboQuantSparseValueSelection
) {
    if selection.mode == .pageTopK {
        cache.ensureKeyPageSummary()
    }
    if selection.mode == .candidateSparse
        || TurboQuantRuntimeControl.enabled("TURBOQUANT_PREPARE_CANDIDATE_SPARSE_SKETCH")
    {
        cache.ensureKeyCandidateSketch()
    }
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
    let skipQJLCompressedPaths = cache.kvCodec == .polarWHT
    if skipQJLCompressedPaths {
        do {
            if let output = try metalPolarWHTScaledDotProductAttention(
                queries: queries,
                keyCode: keyCode,
                turboQuantCache: cache,
                scale: scale,
                mask: adjustedMask,
                sinks: sinks
            ) {
                return output
            }
            failures.append("metalPolarWHT attention was not admitted for this cache")
        } catch {
            let reason = "metalPolarWHT attention failed: \(error)"
            failures.append(reason)
            cache.recordFallback(
                TurboQuantFallbackResult(
                    fromPath: cache.attentionDiagnostics.activeAttentionPath,
                    toPath: cache.precisionPolicy.key.isHighPrecision
                        ? .metalHybridK8PolarWHTValue : .metalPolarWHTHybrid,
                    policy: .compressedDecodeAllowed,
                    reason: reason,
                    isSemanticallyExact: false
                )
            )
        }
    }
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
    let sparseSelection = cache.sparseValueSelection.resolved(
        runtimeMode: cache.resolvedRuntimeMode,
        contextLength: keyCode.layout.logicalLength,
        policy: cache.sparseValuePolicy
    )
    let sparseVRequested = sparseSelection.isEnabled
    let sparseNativeMaxContext =
        TurboQuantRuntimeControl.intValue("TURBOQUANT_SPARSE_V_NATIVE_MAX_CONTEXT")
        ?? 16_384
    let sparseNativeAllowedForContext =
        !sparseVRequested || sparseNativeMaxContext <= 0
        || keyCode.layout.logicalLength <= sparseNativeMaxContext
    let nativeCapabilities = TurboQuantKernelAvailability.current.attentionCapabilities
    let nativeBackend =
        nativeCapabilities.nativeSegmentedAttentionBackend ?? .unavailable
    let nativeMaskSupported: Bool
    switch adjustedMask {
    case .none, .causal:
        nativeMaskSupported = true
    case .array, .arrays:
        nativeMaskSupported = false
    }
    var nativeRejectionReasons: [String] = []
    if nativeCapabilities.nativeCompressedAttention != true {
        nativeRejectionReasons.append(
            nativeCapabilities.nativeFallbackReason
                ?? "native MLX compressed attention backend is unavailable"
        )
    }
    if sparseVRequested && sparseNativeAllowedForContext
        && nativeCapabilities.nativeSparseVSupport != true
    {
        nativeRejectionReasons.append("native MLX compressed attention lacks sparse-V support")
    }
    if !sparseNativeAllowedForContext {
        nativeRejectionReasons.append(
            "Sparse-V native long-context guard disabled sparse selection at context \(keyCode.layout.logicalLength); set TURBOQUANT_SPARSE_V_NATIVE_MAX_CONTEXT=0 to force"
        )
    }
    if sinks != nil {
        nativeRejectionReasons.append("native MLX compressed attention does not support sinks")
    }
    if queries.dim(2) > 8 {
        nativeRejectionReasons.append(
            "native MLX compressed attention supports qLen <= 8; received \(queries.dim(2))"
        )
    }
    if keyCode.layout.headDimension != valueCode.layout.headDimension {
        nativeRejectionReasons.append(
            "native MLX compressed attention requires matching K/V head dimensions"
        )
    }
    if !nativeMaskSupported {
        nativeRejectionReasons.append(
            "native MLX compressed attention supports only none/causal masks"
        )
    }
    if skipQJLCompressedPaths {
        nativeRejectionReasons.append("QJL native compressed attention bypassed for PolarWHT storage")
    }
    let canUseNative =
        !skipQJLCompressedPaths && nativeRejectionReasons.isEmpty

    if canUseNative {
        do {
            let nativeQueries = queries.dtype == .bfloat16 ? queries.asType(.float16) : queries
            prepareTurboQuantSparseAuxiliaryState(cache: cache, selection: sparseSelection)
            let keyPageSummary =
                sparseSelection.mode == .pageTopK ? cache.keyPageSummary : nil
            let keyCandidateSketch =
                sparseSelection.mode == .candidateSparse ? cache.keyCandidateSketch : nil
            let result = try MLX.turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: nativeQueries,
                keyCode: keyCode,
                valueCode: valueCode,
                options: sparseSelection.nativeOptions(
                    scale: scale,
                    causal: adjustedMask.isCausal,
                    diagnostics: sparseVRequested
                        || nativeCapabilities.nativeDiagnosticsSupport == true,
                    backendVersion: nativeCapabilities.nativeBackendVersion
                        ?? TurboQuantNativeAttentionOptions.backendVersion
                ),
                keyPageSummary: keyPageSummary,
                keyCandidateSketch: keyCandidateSketch
            )
            cache.recordNativeAttentionDiagnostics(result.diagnostics, selection: sparseSelection)
            return queries.dtype == .bfloat16 ? result.output.asType(.bfloat16) : result.output
        } catch {
            let reason =
                "native MLX compressed attention failed via \(nativeBackend): \(error)"
            failures.append(reason)
            recordTurboQuantFallback(
                cache: cache,
                path: .nativeMLXCompressed,
                reason: reason
            )
        }
    } else {
        failures.append(
            "native MLX compressed attention bypassed via \(nativeBackend): "
                + nativeRejectionReasons.joined(separator: ", ")
        )
    }
    if sparseVRequested {
        failures.append(
            "Sparse-V \(sparseSelection.mode.rawValue) requested but native compressed attention was unavailable; using dense compressed fallback"
        )
    }

    let canUseOnline =
        !skipQJLCompressedPaths && sinks == nil && cache.prefersOnlineFusedAttention
        && (!sparseVRequested || !sparseNativeAllowedForContext)
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
                sparseVThreshold: nil
            )
        } catch {
            let reason = "online fused compressed attention failed: \(error)"
            failures.append(reason)
        }
    } else {
        failures.append("online fused compressed attention unsupported for this query/cache/mask")
    }

    if skipQJLCompressedPaths {
        failures.append("QJL two-stage attention bypassed for PolarWHT value storage")
    } else {
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
                sparseVThreshold: nil
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
    cache: (any TurboQuantCompressedKVCacheProtocol)? = nil,
    output: MLXArray? = nil
) {
    var arrays = turboQuantAttentionStorageArrays(keys) + turboQuantAttentionStorageArrays(values)
    if let keySidecar = cache?.polarWHTKeyState {
        arrays.append(keySidecar.packedIndices)
        arrays.append(keySidecar.norms)
    }
    if let valueSidecar = cache?.polarWHTValueState {
        arrays.append(valueSidecar.packedIndices)
        arrays.append(valueSidecar.norms)
    }
    if let output {
        arrays.append(output)
    }
    asyncEval(arrays)
}

private func supportsTurboQuantCompressedPrefillCommit(
    cache: any TurboQuantCompressedKVCacheProtocol,
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> Bool {
    if cache.supportsCompressedAttention(
        queries: queries,
        keys: keys,
        values: values,
        mask: mask
    ) {
        return true
    }
    let availability = TurboQuantKernelAvailability.current
    let requiresHybridValueKernel = cache.precisionPolicy.key.isHighPrecision
    let polarWHTDecodeAvailable =
        requiresHybridValueKernel
            ? availability.attentionCapabilities.hybridK8PolarWHTValueAttention
            : availability.supports(.metalPolarWHT)
    guard cache.kvCodec == .polarWHT,
        cache.activeBackend == .metalPolarWHT,
        polarWHTDecodeAvailable
    else {
        return false
    }
    guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else {
        return false
    }
    guard queries.dim(2) > 1, queries.dim(2) == keys.dim(2), keys.dim(2) == values.dim(2) else {
        return false
    }
    guard queries.dim(0) == keys.dim(0), queries.dim(0) == values.dim(0),
        keys.dim(1) == values.dim(1)
    else {
        return false
    }
    guard queries.dim(3) == keys.dim(3),
        keys.dim(3) == values.dim(3),
        queries.dim(3) > 0,
        queries.dim(3) <= 256,
        (queries.dim(3) & (queries.dim(3) - 1)) == 0
    else {
        return false
    }
    guard queries.dim(1) % keys.dim(1) == 0 else {
        return false
    }
    return true
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
        supportsTurboQuantCompressedPrefillCommit(
            cache: turboQuantCache,
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
        let (compressedKeys, compressedValues) = try TurboQuantTiming.measure(
            .compressedCacheUpdate
        ) {
            try turboQuantCache.updateCompressed(
                keys: canonical.keys,
                values: canonical.values
            )
        }
        let state = AttentionKVState.turboQuant(
            keys: compressedKeys,
            values: compressedValues,
            cache: turboQuantCache
        )

        if previousOffset == 0, turboQuantCache.prefersExactInitialPrefill {
            // The exact raw prefill output does not consume the compressed writes, so
            // schedule them explicitly to avoid carrying raw prompt tensors into decode.
            let output = TurboQuantTiming.measure(.exactPrefillAttention) {
                exactScaledDotProductAttention(
                    queries: canonical.queries,
                    keys: canonical.keys,
                    values: canonical.values,
                    scale: scale,
                    mask: mask,
                    sinks: sinks
                )
            }
            asyncEvalTurboQuantAttentionStorage(
                keys: compressedKeys,
                values: compressedValues,
                cache: turboQuantCache,
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

        let output = try TurboQuantTiming.measure(.compressedAttention) {
            try turboQuantAttentionFallbackLadder(
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
        }
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
        if let mixedCache = cache as? any NativeAffineK8V4KVCacheProtocol {
            if let rawState = (mixedCache as? any TurboQuantCompressedKVCacheProtocol)?
                .exactRawStateIfComplete(),
                rawState.keys.dim(-1) == queries.dim(-1),
                rawState.values.dim(-1) == queries.dim(-1)
            {
                mixedCache.recordNativeAffineK8V4AttentionPath(
                    .mlxPackedFallback,
                    failureReason: "affine K8/Vx quantization unsupported for head dimension \(queries.dim(-1)); used exact raw fallback"
                )
                return MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: rawState.keys,
                    values: rawState.values,
                    scale: scale,
                    mask: mask,
                    sinks: sinks
                )
            }
            let nativeSupported = supportsNativeAffineK8V4ScaledDotProductAttention(
                queries: queries,
                quantizedKeys: keys,
                quantizedValues: values,
                mask: mask,
                sinks: sinks,
                keyGroupSize: mixedCache.keyGroupSize,
                keyBits: mixedCache.keyBits,
                valueGroupSize: mixedCache.valueGroupSize,
                valueBits: mixedCache.valueBits
            )
            let sparseVThreshold = mixedCache.sparseValuePolicy.resolvedThreshold(
                runtimeMode: mixedCache.sparseValueRuntimeMode,
                contextLength: keys.0.dim(-2)
            )
            do {
                let residualState = mixedCache.getValueResidualState()
                let output =
                    if let residualState {
                        try mixedAffineK8VxResidualScaledDotProductAttention(
                            queries: queries,
                            quantizedKeys: keys,
                            quantizedValues: values,
                            residualState: residualState,
                            scale: scale,
                            mask: mask,
                            sinks: sinks,
                            keyGroupSize: mixedCache.keyGroupSize,
                            keyBits: mixedCache.keyBits,
                            valueGroupSize: mixedCache.valueGroupSize,
                            valueBits: mixedCache.valueBits,
                            sparseVThreshold: sparseVThreshold
                        )
                    } else {
                        try mixedAffineK8V4ScaledDotProductAttention(
                            queries: queries,
                            quantizedKeys: keys,
                            quantizedValues: values,
                            scale: scale,
                            mask: mask,
                            sinks: sinks,
                            keyGroupSize: mixedCache.keyGroupSize,
                            keyBits: mixedCache.keyBits,
                            valueGroupSize: mixedCache.valueGroupSize,
                            valueBits: mixedCache.valueBits,
                            sparseVThreshold: sparseVThreshold
                        )
                    }
                let nativePath: TurboQuantAttentionPath = (
                    mixedCache.valueBits == TurboQuantKVCodec.affineK8V4ValueBits
                        ? .affineK8V4Native
                        : residualState != nil ? .affineK8VxResidual : .affineK8VxNative
                )
                mixedCache.recordNativeAffineK8V4AttentionPath(
                    (nativeSupported || residualState != nil) ? nativePath : .mlxPackedFallback,
                    failureReason: nativeSupported
                        ? nil
                        : residualState != nil
                            ? "affine K8/Vx residual correction used quantizedMM residual pass"
                            : "affine K8/Vx native attention unsupported; used quantizedMM fallback"
                )
                return output
            } catch {
                mixedCache.recordNativeAffineK8V4AttentionPath(
                    .unavailable,
                    failureReason: String(describing: error)
                )
                throw error
            }
        }
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
        if let turboQuantCache = cache as? any TurboQuantCompressedKVCacheProtocol,
            let output = try polarWHTReferenceHybridScaledDotProductAttention(
                queries: queries,
                quantizedKeys: keys,
                cache: cache,
                turboQuantCache: turboQuantCache,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        {
            return output
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
    let timingStart = TurboQuantTiming.start()
    defer { TurboQuantTiming.record(.attentionWithCacheUpdate, startedAt: timingStart) }

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
        let checkpoint =
            turboQuantUsesFailClosedHybridPolarWHTUpdate(turboQuantCache)
            ? nil
            : turboQuantCache.makeCompressedUpdateCheckpoint(
                appendingTokenCount: turboQuantInputs.keys.dim(2))
        do {
            let (compressedKeys, compressedValues) = try TurboQuantTiming.measure(
                .compressedCacheUpdate
            ) {
                try turboQuantCache.updateCompressed(
                    keys: turboQuantInputs.keys,
                    values: turboQuantInputs.values
                )
            }
            let state = AttentionKVState.turboQuant(
                keys: compressedKeys,
                values: compressedValues,
                cache: turboQuantCache
            )
            let output = try TurboQuantTiming.measure(.compressedAttention) {
                try turboQuantAttentionFallbackLadder(
                    queries: turboQuantInputs.queries,
                    keyCode: compressedKeys,
                    valueCode: compressedValues,
                    cache: turboQuantCache,
                    scale: scale,
                    mask: mask,
                    sinks: sinks,
                    stateAlreadyValidated: true
                )
            }
            return (output, state)
        } catch {
            if let checkpoint {
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
            }
            turboQuantCache.recordCompressedAttentionFailure(String(describing: error))
            throw turboQuantRuntimeFailure(error)
        }
    }
    if let turboQuantCache = cache as? TurboQuantCompressedKVCacheProtocol,
        let quantizedKVCache = cache as? QuantizedKVCacheProtocol,
        !(cache is any NativeAffineK8V4KVCacheProtocol),
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
