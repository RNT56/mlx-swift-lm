// Copyright © 2026 RNT56.

import Foundation
import MLX

// MARK: - 4.1 fallback-materialization OOM guard
//
// The compressed→raw fallback decode materializes the *full* context at once, on the path that
// fires precisely because the compressed kernel failed under memory pressure. Decoding to fp16
// (done at the call sites) halves the spike; this guard turns a remaining hard MLX allocation
// abort (process crash / jetsam) into a recoverable error so generation can degrade cleanly.

/// Available process memory: `os_proc_available_memory()` on iOS is the true headroom before jetsam;
/// macOS falls back to total system memory (a loose bound — the guard mainly protects iOS).
func turboQuantAvailableProcessMemoryBytes() -> Int {
    #if os(iOS) || os(tvOS) || os(visionOS)
        if #available(iOS 13.0, tvOS 13.0, visionOS 1.0, *) {
            let value = UInt64(os_proc_available_memory())
            return value > UInt64(Int.max) ? Int.max : Int(value)
        }
    #endif
    return ModelFitPlanner.currentSystemMemoryBytes()
}

/// Fraction of available memory kept free during a fallback decode. **On-device tuning constant** —
/// raise on thermally/jetsam-constrained devices; this is the single knob for the guard.
private let turboQuantFallbackReserveFraction = 0.15

/// Estimated bytes a full fallback decode of these codes will allocate at `dtype`.
private func turboQuantDecodedFallbackBytes(
    _ keys: TurboQuantAttentionCode, _ values: TurboQuantAttentionCode, dtype: DType
) -> Int {
    let perElement = dtype == .float32 ? 4 : 2
    let keyElems = keys.layout.logicalShape.reduce(1, *)
    let valueElems = values.layout.logicalShape.reduce(1, *)
    return (keyElems + valueElems) * perElement
}

/// Throw a recoverable error before a full-context fallback decode that would not safely fit.
private func turboQuantGuardFallbackMaterialization(
    keys: TurboQuantAttentionCode, values: TurboQuantAttentionCode, dtype: DType
) throws {
    let needed = turboQuantDecodedFallbackBytes(keys, values, dtype: dtype)
    let available = turboQuantAvailableProcessMemoryBytes()
    let usable = available - Int(Double(available) * turboQuantFallbackReserveFraction)
    guard usable >= needed else {
        throw TurboQuantRuntimeFailure.decodedFallbackUnavailable(
            "fallback decode needs ~\(needed / 1_048_576) MB but only ~\(max(0, usable) / 1_048_576) MB is safely available")
    }
}

public typealias TurboQuantPreset = MLX.TurboQuantPreset
public typealias TurboQuantBackend = MLX.TurboQuantBackend
public typealias TurboQuantKernelAvailability = MLX.TurboQuantKernelAvailability
public typealias TurboQuantAttentionCode = MLX.TurboQuantAttentionCode
public typealias TurboQuantPolarWHTAttentionValueCode = MLX.TurboQuantPolarWHTAttentionValueCode
public typealias TurboQuantAttentionPath = MLX.TurboQuantAttentionPath
public typealias TurboQuantKernelProfile = MLX.TurboQuantKernelProfile
public typealias TurboQuantDeviceCapabilities = MLX.TurboQuantDeviceCapabilities
public typealias TurboQuantRuntimeProbeResult = MLX.TurboQuantRuntimeProbeResult
public typealias TurboQuantRuntimeSelfTestStatus = MLX.TurboQuantRuntimeSelfTestStatus

let defaultTurboQuantSeed: UInt64 = 0x9E37_79B9_7F4A_7C15
private let turboQuantValueSeedSalt: UInt64 = 0xD1B5_4A32_D192_ED03

private func copiedTurboQuantPolarWHTCode(
    _ code: TurboQuantPolarWHTAttentionValueCode?
) -> TurboQuantPolarWHTAttentionValueCode? {
    guard var copy = code else { return nil }
    copy.packedIndices = copy.packedIndices[.ellipsis]
    copy.norms = copy.norms[.ellipsis]
    return copy
}

private func copiedTurboQuantPolarWHTValueCode(
    _ code: TurboQuantPolarWHTAttentionValueCode?
) -> TurboQuantPolarWHTAttentionValueCode? {
    copiedTurboQuantPolarWHTCode(code)
}

private func copiedTurboQuantPackedTensor(
    _ packed: TurboQuantPackedTensor?
) -> TurboQuantPackedTensor? {
    guard let packed else { return nil }
    return (
        weight: packed.weight[.ellipsis],
        scales: packed.scales[.ellipsis],
        biases: packed.biases.map { $0[.ellipsis] }
    )
}

fileprivate struct TurboQuantAffineKeySidecar {
    var packed: TurboQuantPackedTensor
    var logicalLength: Int

    var capacity: Int {
        packed.weight.dim(2)
    }
}

private func copiedTurboQuantAffineKeySidecar(
    _ sidecar: TurboQuantAffineKeySidecar?
) -> TurboQuantAffineKeySidecar? {
    guard let sidecar else { return nil }
    guard let copiedPacked = copiedTurboQuantPackedTensor(sidecar.packed) else { return nil }
    return TurboQuantAffineKeySidecar(
        packed: copiedPacked,
        logicalLength: sidecar.logicalLength
    )
}

private func turboQuantPackedStorage(
    _ sidecar: TurboQuantAffineKeySidecar?
) -> QuantizedKVStorage? {
    guard let sidecar else { return nil }
    let activeLength = min(max(0, sidecar.logicalLength), sidecar.capacity)
    guard activeLength > 0 else { return nil }
    return turboQuantPackedStorage(
        turboQuantSlicePackedTensor(sidecar.packed, tokens: 0 ..< activeLength)
    )
}

private func turboQuantAffineKeySidecarBytes(
    _ sidecar: TurboQuantAffineKeySidecar?
) -> Int {
    guard let sidecar else { return 0 }
    return turboQuantArrayBytes(
        [sidecar.packed.weight, sidecar.packed.scales, sidecar.packed.biases].compactMap { $0 }
    )
}

private func turboQuantPackedStorage(
    _ packed: TurboQuantPackedTensor?
) -> QuantizedKVStorage? {
    guard let packed else { return nil }
    return (packed.weight, packed.scales, packed.biases)
}

private func turboQuantSlicePackedTensor(
    _ packed: TurboQuantPackedTensor,
    tokens range: Range<Int>
) -> TurboQuantPackedTensor {
    (
        weight: packed.weight[.ellipsis, range, 0...],
        scales: packed.scales[.ellipsis, range, 0...],
        biases: packed.biases.map { $0[.ellipsis, range, 0...] }
    )
}

private func expandedHybridAffineKeyStorage(
    from packed: TurboQuantPackedTensor,
    logicalLength: Int,
    capacity requestedCapacity: Int
) -> TurboQuantPackedTensor {
    let capacity = max(0, requestedCapacity)
    let sourceCapacity = packed.weight.dim(2)
    let activeLength = min(max(0, logicalLength), sourceCapacity, capacity)
    if capacity <= sourceCapacity {
        return turboQuantSlicePackedTensor(packed, tokens: 0 ..< capacity)
    }

    var weightShape = packed.weight.shape
    var scaleShape = packed.scales.shape
    weightShape[2] = capacity
    scaleShape[2] = capacity
    let expanded = (
        weight: MLXArray.zeros(weightShape, dtype: packed.weight.dtype),
        scales: MLXArray.zeros(scaleShape, dtype: packed.scales.dtype),
        biases: packed.biases.map { biases -> MLXArray in
            var biasShape = biases.shape
            biasShape[2] = capacity
            return MLXArray.zeros(biasShape, dtype: biases.dtype)
        }
    )
    guard activeLength > 0 else { return expanded }
    let active = 0 ..< activeLength
    expanded.weight[.ellipsis, active, 0...] = packed.weight[.ellipsis, active, 0...]
    expanded.scales[.ellipsis, active, 0...] = packed.scales[.ellipsis, active, 0...]
    if let biases = packed.biases {
        expanded.biases?[.ellipsis, active, 0...] = biases[.ellipsis, active, 0...]
    }
    return expanded
}

private func turboQuantConcatPackedTensors(
    _ tensors: [TurboQuantPackedTensor]
) -> TurboQuantPackedTensor? {
    guard let first = tensors.first else { return nil }
    guard tensors.count > 1 else { return first }
    let hasBiases = first.biases != nil
    let biasParts = tensors.compactMap(\.biases)
    return (
        weight: concatenated(tensors.map(\.weight), axis: 2),
        scales: concatenated(tensors.map(\.scales), axis: 2),
        biases: hasBiases && biasParts.count == tensors.count
            ? concatenated(biasParts, axis: 2) : nil
    )
}

private func turboQuantPolarWHTAttentionUnavailableReason(
    backendFallbackReason: String?,
    valueBytes: Int,
    payloadAllocated: Bool
) -> String {
    let payload =
        payloadAllocated
        ? "value payload present (\(valueBytes) bytes)"
        : "value payload unavailable"
    let backend = backendFallbackReason.map { "; backend fallback: \($0)" } ?? ""
    return "PolarWHT compressed attention requires native PolarWHT kernels; \(payload)\(backend)"
}

private func turboQuantEncodePolarWHTAttentionValues(
    _ array: MLXArray,
    bits: Int,
    seed: UInt64,
    capacity: Int,
    logicalLength: Int,
    ringOffset: Int = 0,
    pinnedPrefixLength: Int = 0,
    normStorage: DType = .float32
) throws -> TurboQuantPolarWHTAttentionValueCode {
    if TurboQuantKernelAvailability.current.supportsMetalPolarWHTCodec {
        do {
            return try MLX.turboQuantMetalPolarWHTEncodeAttentionValues(
                array,
                bits: bits,
                seed: seed,
                capacity: capacity,
                logicalLength: logicalLength,
                ringOffset: ringOffset,
                pinnedPrefixLength: pinnedPrefixLength,
                normStorage: normStorage,
                stream: .gpu
            )
        } catch {
            // Keep the additive path portable: sidecar creation can still fall back to the
            // deterministic reference encoder while attention admission remains fail-closed.
        }
    }
    return try MLX.turboQuantPolarWHTReferenceEncodeAttentionValues(
        array,
        bits: bits,
        seed: seed,
        capacity: capacity,
        logicalLength: logicalLength,
        ringOffset: ringOffset,
        pinnedPrefixLength: pinnedPrefixLength,
        normStorage: normStorage
    )
}

func turboQuantCompressedKVCodec(
    requested: TurboQuantKVCodec? = nil,
    backend: TurboQuantBackend
) -> TurboQuantKVCodec {
    if requested == .polarWHT {
        return .polarWHT
    }
    switch backend {
    case .polarWHTReference, .metalPolarWHT:
        return .polarWHT
    default:
        return .polarQJL
    }
}

func turboQuantDefaultValueBits(
    preset: TurboQuantPreset,
    kvCodec: TurboQuantKVCodec,
    requestedValueBits: Int?
) -> Int {
    requestedValueBits
        ?? (kvCodec == .polarWHT
            ? TurboQuantKVCodec.polarWHTDefaultValueBits
            : preset.defaultValueBits)
}

func turboQuantMetalCodecAvailable(
    kvCodec: TurboQuantKVCodec,
    availability: TurboQuantKernelAvailability
) -> Bool {
    kvCodec == .polarWHT
        ? availability.supportsMetalPolarWHTCodec
        : availability.supportsMetalPolarQJLCodec
}

func turboQuantMetalAttentionAvailable(
    kvCodec: TurboQuantKVCodec,
    availability: TurboQuantKernelAvailability
) -> Bool {
    kvCodec == .polarWHT
        ? availability.supportsMetalPolarWHTAttention
        : availability.supportsMetalPolarQJLAttention
}

private func turboQuantIsMetalCompressedBackend(_ backend: TurboQuantBackend) -> Bool {
    backend == .metalPolarQJL || backend == .metalPolarWHT
}

public enum TurboQuantKVCodec: String, Codable, Sendable, CaseIterable {
    case polarQJL = "polar_qjl"
    case polarWHT = "polar_wht"
    case affineK8V4 = "affine_k8_v4"
    case affineK8Vx = "affine_k8_vx"
    case affineInt4 = "affine_int4"

    public static let polarWHTDefaultValueBits = 3
    public static let affineK8V4KeyBits = 8
    public static let affineK8V4ValueBits = 4
    public static let affineK8V4KeyGroupSize = 64
    public static let affineK8V4ValueGroupSize = 32
    public static let affineK8VxSupportedValueBits: Set<Int> = [2, 3, 4]
    public static let affineInt4Bits = 4
    public static let affineInt4DefaultGroupSize = 32

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        switch value {
        case Self.polarQJL.rawValue, "polarQJL", "polar-qjl":
            self = .polarQJL
        case Self.polarWHT.rawValue, "polarWHT", "polar-wht", "wht", "polar_wht_v3":
            self = .polarWHT
        case Self.affineK8V4.rawValue, "affineK8V4", "affine-k8-v4", "k8v4":
            self = .affineK8V4
        case Self.affineK8Vx.rawValue, "affineK8Vx", "affine-k8-vx", "k8vx":
            self = .affineK8Vx
        case Self.affineInt4.rawValue, "affineInt4", "affine-int4":
            self = .affineInt4
        default:
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported KV codec '\(value)'"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

public enum TurboQuantFallbackPath: String, Equatable, Codable, Sendable, CaseIterable {
    case nativeMLXCompressed
    case onlineFusedCompressed
    case tiledOnlineFused
    case twoStageQKAV
    case packedQuantizedSDPA
    case decodedCompressedSDPA
    case rawExactSDPA
    case typedFailure
}

public struct TurboQuantRuntimeCacheFootprint: Equatable, Codable, Sendable {
    public var logicalLength: Int
    public var capacity: Int
    public var compressedBytes: Int
    public var packedFallbackBytes: Int
    public var rawShadowBytes: Int
    public var decodedTransientBytes: Int
    public var lifecycle: TurboQuantCacheLifecycle

    public init(
        logicalLength: Int,
        capacity: Int,
        compressedBytes: Int,
        packedFallbackBytes: Int = 0,
        rawShadowBytes: Int = 0,
        decodedTransientBytes: Int = 0,
        lifecycle: TurboQuantCacheLifecycle
    ) {
        self.logicalLength = logicalLength
        self.capacity = capacity
        self.compressedBytes = compressedBytes
        self.packedFallbackBytes = packedFallbackBytes
        self.rawShadowBytes = rawShadowBytes
        self.decodedTransientBytes = decodedTransientBytes
        self.lifecycle = lifecycle
    }

    public var residentBytes: Int {
        compressedBytes + packedFallbackBytes + rawShadowBytes
    }

    public var totalBytesIncludingTransient: Int {
        residentBytes + decodedTransientBytes
    }
}

public enum KVCacheStrategy: String, Codable, Sendable, CaseIterable {
    case none
    case mlxAffine
    case affineK8V4
    case affineK8Vx
    case affineInt4
    case adaptiveTurboQuant
    case hybridTurboQuant
    case turboQuant
}

extension KVCacheStrategy {
    public var canUseTurboQuant: Bool {
        self == .adaptiveTurboQuant || self == .hybridTurboQuant || self == .turboQuant
    }

    public var createsTurboQuantCacheImmediately: Bool {
        self == .hybridTurboQuant || self == .turboQuant
    }

    public var createsAffineInt4CacheImmediately: Bool {
        self == .affineInt4
    }

    public var createsAffineK8V4CacheImmediately: Bool {
        self == .affineK8V4
    }

    public var createsAffineK8VxCacheImmediately: Bool {
        self == .affineK8V4 || self == .affineK8Vx
    }
}

public enum TurboQuantOptimizationPolicy: String, Codable, Sendable, CaseIterable {
    case auto
    case conservative
    case preferMemory
    case preferThroughput
}

public struct TurboQuantAttentionDiagnostics: Equatable, Codable, Sendable {
    public var layerIndex: Int? = nil
    public var metalAttentionAvailable: Bool
    public var activeAttentionPath: TurboQuantAttentionPath
    public var nativeBackend: String? = nil
    public var nativeBackendVersion: Int? = nil
    public var nativeFallbackReason: String? = nil
    public var nativeKernelKind: Int? = nil
    public var nativeSparseVSkipRatio: Double? = nil
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var selfTestStatus: TurboQuantRuntimeSelfTestStatus
    public var selfTestFailureReason: String?
    public var optimizationPolicy: TurboQuantOptimizationPolicy
    public var fallbackReason: String?
    public var lastUnsupportedShape: String?
    public var rawFallbackAllocated: Bool
    public var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public var lastFallback: TurboQuantFallbackResult?
    public var sparseVEnabled: Bool = false
    public var sparseVThreshold: Float?
    public var sparseVSelectionMode: TurboQuantSparseValueSelectionMode? = nil
    public var sparseVTopK: Int? = nil
    public var sparseVCumulativeMass: Float? = nil
    public var sparseVMaxTopK: Int? = nil
    public var sparseVRecentTokenCount: Int? = nil
    public var sparseVOlderTokenCount: Int? = nil
    public var sparseVPageCandidateCount: Int? = nil
    public var sparseVSkippedTokens: Int? = nil
    public var sparseVTotalTokens: Int? = nil
    /// True only when native sparse-V diagnostics were emitted for the last attention attempt.
    public var sparseVActive: Bool? = nil
    public var sparseVSkipRatio: Double?
    public var sparseVRetainedMass: Double? = nil
    public var boundaryProtectedLayerCount: Int = 0
    public var boundaryProtectionReason: String?
    public var keyBits: Int? = nil
    public var valueBits: Int? = nil
    public var keyGroupSize: Int? = nil
    public var valueGroupSize: Int? = nil
    public var keyPageSummaryAvailable: Bool? = nil
    public var keyPageSummaryShape: [Int]? = nil
    public var keyPageSummaryUnavailableReason: String? = nil
    public var keyCandidateSketchAvailable: Bool? = nil
    public var keyCandidateSketchShape: [Int]? = nil
    public var keyCandidateSketchUnavailableReason: String? = nil
    public var polarWHTKeyBytes: Int = 0
    public var polarWHTKeyPayloadAllocated: Bool = false
    public var polarWHTValueBytes: Int = 0
    public var polarWHTValuePayloadAllocated: Bool = false
    /// Count of rows appended through the fused quantize-append (P1-1) kernel.
    /// Nil when the fused path is not applicable (e.g. non-affine caches).
    public var fusedAppendCount: Int? = nil
    /// Count of rows that fell back to the unfused ladder while the fused flag
    /// was enabled because a guard did not hold.
    public var fusedAppendFallbackCount: Int? = nil
    /// First guard that blocked the fused path while it was enabled, if any.
    public var fusedAppendFallbackReason: String? = nil
}

public struct TurboQuantKVCacheDiagnostics: Equatable, Codable, Sendable {
    public var layerIndex: Int? = nil
    public var kvCodec: TurboQuantKVCodec = .polarQJL
    public var preset: TurboQuantPreset
    public var requestedBackend: TurboQuantBackend
    public var activeBackend: TurboQuantBackend
    public var fallbackReason: String?
    public var metalCodecAvailable: Bool
    public var metalAttentionAvailable: Bool
    public var activeAttentionPath: TurboQuantAttentionPath
    public var nativeBackend: String? = nil
    public var nativeBackendVersion: Int? = nil
    public var nativeFallbackReason: String? = nil
    public var nativeKernelKind: Int? = nil
    public var nativeSparseVSkipRatio: Double? = nil
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var selfTestStatus: TurboQuantRuntimeSelfTestStatus
    public var selfTestFailureReason: String?
    public var optimizationPolicy: TurboQuantOptimizationPolicy
    public var lastUnsupportedShape: String?
    public var groupSize: Int
    public var bits: Int
    public var valueBits: Int
    public var maxSize: Int?
    public var rawFallbackAllocated: Bool
    public var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public var lastFallback: TurboQuantFallbackResult?
    public var footprint: TurboQuantRuntimeCacheFootprint?
    public var polarWHTKeyBytes: Int = 0
    public var polarWHTKeyPayloadAllocated: Bool = false
    public var polarWHTValueBytes: Int = 0
    public var polarWHTValuePayloadAllocated: Bool = false
    public var sparseVEnabled: Bool = false
    public var sparseVThreshold: Float?
    public var sparseVSelectionMode: TurboQuantSparseValueSelectionMode? = nil
    public var sparseVTopK: Int? = nil
    public var sparseVCumulativeMass: Float? = nil
    public var sparseVMaxTopK: Int? = nil
    public var sparseVRecentTokenCount: Int? = nil
    public var sparseVOlderTokenCount: Int? = nil
    public var sparseVPageCandidateCount: Int? = nil
    public var sparseVSkippedTokens: Int? = nil
    public var sparseVTotalTokens: Int? = nil
    /// True only when native sparse-V diagnostics were emitted for the last attention attempt.
    public var sparseVActive: Bool? = nil
    public var sparseVSkipRatio: Double?
    public var sparseVRetainedMass: Double? = nil
    public var boundaryProtectedLayerCount: Int = 0
    public var boundaryProtectionReason: String?
    public var keyCandidateSketchAvailable: Bool? = nil
    public var keyCandidateSketchShape: [Int]? = nil
    public var keyCandidateSketchUnavailableReason: String? = nil
}

private func turboQuantSparseVActive(
    _ diagnostics: TurboQuantNativeAttentionDiagnostics?
) -> Bool {
    (diagnostics?.sparseTotalTokens ?? 0) > 0
}

private func turboQuantSparseVInactiveReason(
    enabled: Bool,
    kvCodec: TurboQuantKVCodec,
    activeBackend: TurboQuantBackend,
    nativeDiagnostics: TurboQuantNativeAttentionDiagnostics?,
    fallbackReason: String?
) -> String? {
    if let fallbackReason { return fallbackReason }
    guard enabled, !turboQuantSparseVActive(nativeDiagnostics) else { return nil }
    if kvCodec == .polarWHT, activeBackend == .polarWHTReference {
        return "Sparse-V is not implemented for PolarWHT reference hybrid; dense PolarWHT value accumulation used"
    }
    return nil
}

let turboQuantKeyCandidateSketchWidth = 64
let turboQuantKeyCandidateSketchProjectionCount = turboQuantKeyCandidateSketchWidth / 2

func turboQuantKeyCandidateSketchProjectionSign(
    projectionIndex: Int,
    dimension: Int
) -> Float {
    let projectionTerm = UInt32(projectionIndex + 1) &* 747_796_405
    let dimensionTerm = UInt32(dimension + 1) &* 2_891_336_453
    let hash = projectionTerm ^ dimensionTerm
    return (hash & 1) == 0 ? 1 : -1
}

private func turboQuantKeyCandidateSketchProjectionSigns(
    headDimension: Int,
    projectionIndex: Int
) -> [Float] {
    guard headDimension > 0 else { return [] }
    return (0 ..< headDimension).map { dimension in
        turboQuantKeyCandidateSketchProjectionSign(
            projectionIndex: projectionIndex,
            dimension: dimension
        )
    }
}

private func turboQuantKeyCandidateSketchUnavailableReason(
    activeBackend: TurboQuantBackend,
    compressedKeys: TurboQuantAttentionCode?,
    artifactName: String
) -> String? {
    guard activeBackend == .metalPolarQJL else {
        return "\(artifactName) require metalPolarQJL backend; active backend is \(activeBackend.rawValue)"
    }
    guard let compressedKeys else {
        return "no compressed key state is available"
    }
    guard compressedKeys.layout.ringOffset == 0 else {
        return "ring offset \(compressedKeys.layout.ringOffset) makes \(artifactName) unsafe"
    }
    guard compressedKeys.layout.logicalLength > 0 else {
        return "compressed key state is empty"
    }
    return nil
}

private func turboQuantBuildKeyCandidateSketch(
    keyCode: TurboQuantAttentionCode,
    pageSize: Int = MLX.turboQuantKeyPageSummaryPageSize
) throws -> MLXArray {
    guard pageSize > 0 else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "key candidate sketch page size must be positive"
        )
    }
    guard keyCode.layout.ringOffset == 0 else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "key candidate sketch requires non-rotated compressed key layout"
        )
    }

    let layout = keyCode.layout
    let pageCapacity = (layout.capacity + pageSize - 1) / pageSize
    let sketch = MLXArray.zeros(
        [
            layout.batchSize,
            layout.kvHeadCount,
            pageCapacity,
            turboQuantKeyCandidateSketchWidth,
        ],
        dtype: .float32
    )
    guard layout.logicalLength > 0 else { return sketch }

    let decodedKeys = try MLX.turboQuantMetalDecodeAttention(
        keyCode,
        outputDType: .float32
    )
    for pageIndex in 0 ..< pageCapacity {
        let tokenStart = pageIndex * pageSize
        guard tokenStart < layout.logicalLength else { break }
        let tokenEnd = min(tokenStart + pageSize, layout.logicalLength)
        let pageKeys = decodedKeys[0..., 0..., tokenStart ..< tokenEnd, 0...]
        var minima: [MLXArray] = []
        var maxima: [MLXArray] = []
        minima.reserveCapacity(turboQuantKeyCandidateSketchProjectionCount)
        maxima.reserveCapacity(turboQuantKeyCandidateSketchProjectionCount)
        for projectionIndex in 0 ..< turboQuantKeyCandidateSketchProjectionCount {
            let signs = MLXArray(
                turboQuantKeyCandidateSketchProjectionSigns(
                    headDimension: layout.headDimension,
                    projectionIndex: projectionIndex
                ),
                [1, 1, 1, layout.headDimension]
            )
            let projected = (pageKeys * signs).sum(axis: 3, keepDims: true)
            minima.append(projected.min(axis: 2))
            maxima.append(projected.max(axis: 2))
        }
        let pageSketch = concatenated(minima + maxima, axis: 2)
        sketch[0..., 0..., pageIndex ..< (pageIndex + 1), 0...] =
            pageSketch.expandedDimensions(axis: 2)
    }
    return sketch
}

private func turboQuantUpdateKeyCandidateSketchPage(
    existingSketch: MLXArray,
    encodedKeys: TurboQuantAttentionCode,
    pageIndex: Int
) throws -> MLXArray {
    guard pageIndex >= 0, pageIndex < existingSketch.dim(2) else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "key candidate sketch page index \(pageIndex) is outside capacity \(existingSketch.dim(2))"
        )
    }
    let tokenSketch = try turboQuantBuildKeyCandidateSketch(keyCode: encodedKeys)
    let updatedSketch = existingSketch
    let pageRange = pageIndex ..< (pageIndex + 1)
    let projectionRange = 0 ..< turboQuantKeyCandidateSketchProjectionCount
    let maximumRange =
        turboQuantKeyCandidateSketchProjectionCount ..< turboQuantKeyCandidateSketchWidth
    updatedSketch[0..., 0..., pageRange, projectionRange] = MLX.minimum(
        updatedSketch[0..., 0..., pageRange, projectionRange],
        tokenSketch[0..., 0..., 0 ..< 1, projectionRange]
    )
    updatedSketch[0..., 0..., pageRange, maximumRange] = MLX.maximum(
        updatedSketch[0..., 0..., pageRange, maximumRange],
        tokenSketch[0..., 0..., 0 ..< 1, maximumRange]
    )
    return updatedSketch
}

public protocol TurboQuantCompressedKVCacheProtocol: KVCache, AnyObject {
    var kvCodec: TurboQuantKVCodec { get }
    var preset: TurboQuantPreset { get }
    var requestedBackend: TurboQuantBackend { get }
    var activeBackend: TurboQuantBackend { get }
    var optimizationPolicy: TurboQuantOptimizationPolicy { get }
    var fallbackPolicy: TurboQuantFallbackPolicy { get }
    var precisionPolicy: TurboQuantKVPrecisionPolicy { get }
    var attentionDiagnostics: TurboQuantAttentionDiagnostics { get }
    var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? { get }
    var hybridAffineKeyState: QuantizedKVStorage? { get }
    var hybridAffineKeyStateForAttention: QuantizedKVStorage? { get }
    var hybridAffineKeyTailStateForAttention: QuantizedKVStorage? { get }
    var polarWHTKeyState: TurboQuantPolarWHTAttentionValueCode? { get }
    var polarWHTValueState: TurboQuantPolarWHTAttentionValueCode? { get }
    var polarWHTKeyStateForAttention: TurboQuantPolarWHTAttentionValueCode? { get }
    var polarWHTValueStateForAttention: TurboQuantPolarWHTAttentionValueCode? { get }
    var polarWHTValueTailStateForAttention: TurboQuantPolarWHTAttentionValueCode? { get }
    var polarWHTDecodedValueState: MLXArray? { get }
    var polarWHTDecodedValueLayout: MLX.TurboQuantAttentionLayout? { get }
    var cacheLifecycle: TurboQuantCacheLifecycle { get }
    var fallbackResults: [TurboQuantFallbackResult] { get }
    var cacheFootprint: TurboQuantRuntimeCacheFootprint { get }
    var keyPageSummary: MLXArray? { get }
    var keyCandidateSketch: MLXArray? { get }
    var sparseValuePolicy: TurboQuantSparseValuePolicy { get }
    var sparseValueSelection: TurboQuantSparseValueSelection { get }
    var resolvedRuntimeMode: TurboQuantRuntimeMode { get }
    var layerIndex: Int? { get }

    func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot
    func exportSnapshot(
        identity: TurboQuantKVSnapshotIdentity,
        conversationID: UUID,
        snapshotID: UUID,
        encryptionKeyID: String,
        createdAt: Date
    ) throws -> TurboQuantKVSnapshotPayload
    func importSnapshot(
        _ payload: TurboQuantKVSnapshotPayload,
        expectedIdentity: TurboQuantKVSnapshotIdentity
    ) throws

    func supportsCompressedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool

    func updateCompressed(keys: MLXArray, values: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    )
    func makeCompressedUpdateCheckpoint(appendingTokenCount tokenCount: Int)
        -> TurboQuantCompressedUpdateCheckpoint
    func restoreCompressedUpdateCheckpoint(_ checkpoint: TurboQuantCompressedUpdateCheckpoint)

    func ensureKeyPageSummary()
    func ensureKeyCandidateSketch()
    func recordCompressedAttentionFailure(_ message: String)
    func recordFallback(_ result: TurboQuantFallbackResult)
    func recordNativeAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        selection: TurboQuantSparseValueSelection
    )
    func recordPolarWHTAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        path: TurboQuantAttentionPath
    )
    func validateCompressedState(context: String) throws
    func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray)
    func releaseRawShadow()
    func exactRawStateIfComplete() -> (keys: MLXArray, values: MLXArray)?
}

public struct TurboQuantCompressedUpdateCheckpoint {
    fileprivate var payload: TurboQuantCompressedUpdateCheckpointPayload
}

extension TurboQuantCompressedKVCacheProtocol {
    public var sparseValueSelection: TurboQuantSparseValueSelection { .off }
    public var layerIndex: Int? { nil }
    public var polarWHTKeyState: TurboQuantPolarWHTAttentionValueCode? { nil }
    public var polarWHTValueState: TurboQuantPolarWHTAttentionValueCode? { nil }
    public var hybridAffineKeyStateForAttention: QuantizedKVStorage? { hybridAffineKeyState }
    public var hybridAffineKeyTailStateForAttention: QuantizedKVStorage? { nil }
    public var polarWHTValueTailStateForAttention: TurboQuantPolarWHTAttentionValueCode? { nil }
    public var polarWHTDecodedValueState: MLXArray? { nil }
    public var polarWHTDecodedValueLayout: MLX.TurboQuantAttentionLayout? { nil }

    public func ensureKeyPageSummary() {}

    public func recordNativeAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        selection: TurboQuantSparseValueSelection
    ) {}

    public func recordPolarWHTAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        path: TurboQuantAttentionPath
    ) {}
}

private enum TurboQuantCompressedUpdateCheckpointPayload {
    case fullState(
        offset: Int,
        metaState: [String],
        state: [MLXArray],
        hybridAffineKeyState: TurboQuantAffineKeySidecar?,
        polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?,
        polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?
    )
    case rotatingFullState(
        offset: Int,
        writeIndex: Int,
        metaState: [String],
        state: [MLXArray],
        rawFallbackState: [MLXArray]?,
        rawFallbackMetaState: [String]?,
        packedFallbackState: [MLXArray]?,
        packedFallbackMetaState: [String]?,
        packedKeys: TurboQuantPackedTensor?,
        packedValues: TurboQuantPackedTensor?,
        polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?,
        polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?,
        fallbackResultCount: Int,
        lifecycle: TurboQuantCacheLifecycle,
        lastAttentionPath: TurboQuantAttentionPath,
        lastUnsupportedShape: String?,
        lastDecodedTransientBytes: Int
    )
    case linearCompressed(
        offset: Int,
        compressedKeys: TurboQuantAttentionCode,
        compressedValues: TurboQuantAttentionCode,
        hybridAffineKeyState: TurboQuantAffineKeySidecar?,
        polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?,
        polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?,
        packedFallbackState: [MLXArray],
        fallbackResultCount: Int,
        lifecycle: TurboQuantCacheLifecycle,
        lastAttentionPath: TurboQuantAttentionPath,
        lastUnsupportedShape: String?,
        lastDecodedTransientBytes: Int
    )
    case rotatingCompressed(
        offset: Int,
        writeIndex: Int,
        compressedKeys: TurboQuantAttentionCode,
        compressedValues: TurboQuantAttentionCode,
        rawFallbackState: [MLXArray]?,
        rawFallbackMetaState: [String]?,
        packedFallbackState: [MLXArray]?,
        packedFallbackMetaState: [String]?,
        packedKeys: TurboQuantPackedTensor?,
        packedValues: TurboQuantPackedTensor?,
        polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?,
        polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?,
        fallbackResultCount: Int,
        lifecycle: TurboQuantCacheLifecycle,
        lastAttentionPath: TurboQuantAttentionPath,
        lastUnsupportedShape: String?,
        lastDecodedTransientBytes: Int
    )
}

extension TurboQuantCompressedKVCacheProtocol {
    public var keyPageSummary: MLXArray? { nil }
    public var keyCandidateSketch: MLXArray? { nil }
    public var hybridAffineKeyState: QuantizedKVStorage? { nil }
    public var polarWHTKeyStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTKeyState
    }
    public var polarWHTValueStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTValueState
    }

    public func ensureKeyCandidateSketch() {}

    public var prefersOnlineFusedAttention: Bool {
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FORCE_FUSED") {
            return true
        }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_FORCE_TWO_STAGE") {
            return false
        }
        switch optimizationPolicy {
        case .auto, .preferMemory, .preferThroughput:
            return true
        case .conservative:
            return false
        }
    }

    public var prefersExactInitialPrefill: Bool {
        switch optimizationPolicy {
        case .auto, .conservative, .preferThroughput:
            true
        case .preferMemory:
            false
        }
    }

    public func exactRawStateIfComplete() -> (keys: MLXArray, values: MLXArray)? {
        nil
    }

    public func exportSnapshot(
        identity: TurboQuantKVSnapshotIdentity,
        conversationID: UUID,
        encryptionKeyID: String = "lm-local-unencrypted",
        createdAt: Date = Date()
    ) throws -> TurboQuantKVSnapshotPayload {
        try exportSnapshot(
            identity: identity,
            conversationID: conversationID,
            snapshotID: UUID(),
            encryptionKeyID: encryptionKeyID,
            createdAt: createdAt
        )
    }

}

extension TurboQuantCompressedKVCacheProtocol where Self: NativeAffineK8V4KVCacheProtocol {
    public var kvCodec: TurboQuantKVCodec { .affineK8V4 }
    public var preset: TurboQuantPreset { .turbo8 }
    public var requestedBackend: TurboQuantBackend { .mlxPacked }
    public var activeBackend: TurboQuantBackend { .mlxPacked }
    public var optimizationPolicy: TurboQuantOptimizationPolicy { .preferThroughput }
    public var fallbackPolicy: TurboQuantFallbackPolicy { .fatalOnFailure }
    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? { nil }
    public var fallbackResults: [TurboQuantFallbackResult] { [] }
    public var requestedRuntimeMode: TurboQuantRuntimeMode { sparseValueRuntimeMode }
    public var resolvedRuntimeMode: TurboQuantRuntimeMode { sparseValueRuntimeMode }

    public var cacheLifecycle: TurboQuantCacheLifecycle {
        offset > 0
            ? .compressedCommitted(
                logicalLength: runtimeSnapshot().logicalLength,
                capacity: runtimeSnapshot().capacity
            )
            : .empty
    }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let snapshot = runtimeSnapshot()
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: snapshot.logicalLength,
            capacity: snapshot.capacity,
            compressedBytes: snapshot.keyBytes + snapshot.valueBytes,
            lifecycle: cacheLifecycle
        )
    }

    public func supportsCompressedAttention(
        queries _: MLXArray,
        keys _: MLXArray,
        values _: MLXArray,
        mask _: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        false
    }

    public func updateCompressed(keys _: MLXArray, values _: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    ) {
        throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
            "Affine K8/Vx cache uses native quantized attention, not Polar/QJL compressed attention"
        )
    }

    public func makeCompressedUpdateCheckpoint(appendingTokenCount _: Int)
        -> TurboQuantCompressedUpdateCheckpoint
    {
        TurboQuantCompressedUpdateCheckpoint(
            payload: .fullState(
                offset: offset,
                metaState: metaState,
                state: state.map { $0[.ellipsis] },
                hybridAffineKeyState: nil,
                polarWHTKeyCode: nil,
                polarWHTValueCode: nil
            )
        )
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        recordNativeAffineK8V4AttentionPath(
            attentionDiagnostics.activeAttentionPath,
            failureReason: message
        )
    }

    public func recordFallback(_ result: TurboQuantFallbackResult) {
        recordNativeAffineK8V4AttentionPath(
            result.toPath ?? attentionDiagnostics.activeAttentionPath,
            failureReason: result.reason
        )
    }

    public func validateCompressedState(context _: String) throws {}

    public func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray) {
        guard let current = getQuantizedState() else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "Affine K8/Vx cache has no quantized key/value state to decode"
            )
        }
        let keys = dequantized(
            current.0.0,
            scales: current.0.1,
            biases: current.0.2,
            groupSize: keyGroupSize,
            bits: keyBits,
            mode: mode,
            dtype: outputDType
        )
        let values = dequantized(
            current.1.0,
            scales: current.1.1,
            biases: current.1.2,
            groupSize: valueGroupSize,
            bits: valueBits,
            mode: mode,
            dtype: outputDType
        )
        return (keys, values)
    }

    public func releaseRawShadow() {}

    public func exactRawStateIfComplete() -> (keys: MLXArray, values: MLXArray)? {
        nil
    }

    public func exportSnapshot(
        identity _: TurboQuantKVSnapshotIdentity,
        conversationID _: UUID,
        snapshotID _: UUID,
        encryptionKeyID _: String,
        createdAt _: Date
    ) throws -> TurboQuantKVSnapshotPayload {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "Affine K8/Vx cache snapshots are not represented as TurboQuant Polar/QJL blobs"
        )
    }

    public func importSnapshot(
        _ payload: TurboQuantKVSnapshotPayload,
        expectedIdentity _: TurboQuantKVSnapshotIdentity
    ) throws {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "Cannot import TurboQuant Polar/QJL snapshot \(payload.manifest.snapshotID) into Affine K8/Vx cache"
        )
    }
}

extension AffineK8V4KVCache {
    public func restoreCompressedUpdateCheckpoint(
        _ checkpoint: TurboQuantCompressedUpdateCheckpoint
    ) {
        guard case .fullState(_, let metaState, let state, _, _, _) = checkpoint.payload else {
            return
        }
        self.state = state
        self.metaState = metaState
    }
}

public func turboQuantCacheFootprints(
    _ cache: [KVCache]
) -> [TurboQuantRuntimeCacheFootprint] {
    cache.compactMap { ($0 as? TurboQuantCompressedKVCacheProtocol)?.cacheFootprint }
}

public func turboQuantAggregateCacheFootprint(
    _ cache: [KVCache]
) -> TurboQuantRuntimeCacheFootprint {
    let footprints = turboQuantCacheFootprints(cache)
    return footprints.reduce(
        TurboQuantRuntimeCacheFootprint(
            logicalLength: 0,
            capacity: 0,
            compressedBytes: 0,
            lifecycle: .empty
        )
    ) { partial, item in
        TurboQuantRuntimeCacheFootprint(
            logicalLength: partial.logicalLength + item.logicalLength,
            capacity: partial.capacity + item.capacity,
            compressedBytes: partial.compressedBytes + item.compressedBytes,
            packedFallbackBytes: partial.packedFallbackBytes + item.packedFallbackBytes,
            rawShadowBytes: partial.rawShadowBytes + item.rawShadowBytes,
            decodedTransientBytes: partial.decodedTransientBytes + item.decodedTransientBytes,
            lifecycle: item.lifecycle
        )
    }
}

private struct RestoredAttentionLayoutMetadata {
    var capacity: Int?
    var logicalLength: Int
    var ringOffset: Int
    var pinnedPrefixLength: Int
    var keyHeadDimension: Int?
    var valueHeadDimension: Int?
    var kvHeadCount: Int?

    init(
        capacity: Int?,
        logicalLength: Int,
        ringOffset: Int,
        pinnedPrefixLength: Int,
        headDimension: Int?,
        valueHeadDimension: Int? = nil,
        kvHeadCount: Int?
    ) {
        self.capacity = capacity
        self.logicalLength = logicalLength
        self.ringOffset = ringOffset
        self.pinnedPrefixLength = pinnedPrefixLength
        self.keyHeadDimension = headDimension
        self.valueHeadDimension = valueHeadDimension ?? headDimension
        self.kvHeadCount = kvHeadCount
    }
}

private func turboQuantMetaInt(_ meta: [String], key: String) -> Int? {
    let prefix = "\(key)="
    guard let value = meta.first(where: { $0.hasPrefix(prefix) })?.dropFirst(prefix.count) else {
        return nil
    }
    return Int(value)
}

private func turboQuantSupportsAttentionDimension(_ dimension: Int) -> Bool {
    dimension > 0 && dimension <= 512
}

private func turboQuantSupportsPolarWHTAttentionDimension(_ dimension: Int) -> Bool {
    dimension > 0 && dimension <= 256 && (dimension & (dimension - 1)) == 0
}

private func turboQuantNativeSupportsMask(_ mask: MLXFast.ScaledDotProductAttentionMaskMode) -> Bool {
    switch mask {
    case .none, .causal:
        true
    case .array, .arrays:
        false
    }
}

private func turboQuantSupportsPackedFallback(keys: MLXArray, values: MLXArray, groupSize: Int)
    -> Bool
{
    guard groupSize > 0, keys.ndim == 4, values.ndim == 4 else { return false }
    return keys.dim(3).isMultiple(of: groupSize) && values.dim(3).isMultiple(of: groupSize)
}

private func turboQuantSupportsPackedFallback(
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    groupSize: Int
) -> Bool {
    guard groupSize > 0 else { return false }
    guard !turboQuantIsCompactValuePlaceholder(valueCode) else { return false }
    return keyCode.layout.headDimension.isMultiple(of: groupSize)
        && valueCode.layout.headDimension.isMultiple(of: groupSize)
}

private func turboQuantUsesPolarWHTValueOnlyStorage(
    kvCodec: TurboQuantKVCodec,
    precisionPolicy: TurboQuantKVPrecisionPolicy
) -> Bool {
    kvCodec == .polarWHT && precisionPolicy.key.isHighPrecision
}

private func turboQuantIsCompactValuePlaceholder(_ code: TurboQuantAttentionCode) -> Bool {
    code.role == .value
        && code.scalesPerGroup == 0
        && code.packedMagnitudes.shape == [1]
        && code.signs.shape == [1]
        && code.highPrecisionMask.shape == [1]
        && code.residualSigns.shape == [1]
        && code.scales.shape == [1]
}

private func turboQuantCompactValuePlaceholderCode(
    layout: MLX.TurboQuantAttentionLayout,
    preset: TurboQuantPreset,
    groupSize: Int,
    seed: UInt64,
    valueBits: Int
) -> TurboQuantAttentionCode {
    TurboQuantAttentionCode(
        layout: layout,
        preset: preset,
        role: .value,
        groupSize: groupSize,
        seed: seed,
        valueBits: valueBits,
        scalesPerGroup: 0,
        packedMagnitudes: MLXArray.zeros([1], dtype: .uint32),
        signs: MLXArray.zeros([1], dtype: .uint32),
        highPrecisionMask: MLXArray.zeros([1], dtype: .uint32),
        residualSigns: MLXArray.zeros([1], dtype: .uint32),
        scales: MLXArray.zeros([1], dtype: .float32)
    )
}

private func turboQuantRestoredValueLayout(
    valuePacked: MLXArray,
    keyLayout: MLX.TurboQuantAttentionLayout,
    valueHeadDimension: Int,
    groupSize: Int
) -> MLX.TurboQuantAttentionLayout {
    let isPlaceholder = valuePacked.shape == [1]
    return MLX.TurboQuantAttentionLayout(
        layoutVersion: keyLayout.layoutVersion,
        batchSize: isPlaceholder ? keyLayout.batchSize : valuePacked.dim(0),
        kvHeadCount: isPlaceholder ? keyLayout.kvHeadCount : valuePacked.dim(1),
        capacity: keyLayout.capacity,
        logicalLength: keyLayout.logicalLength,
        ringOffset: keyLayout.ringOffset,
        pinnedPrefixLength: keyLayout.pinnedPrefixLength,
        headDimension: valueHeadDimension,
        groupsPerVector: isPlaceholder
            ? (valueHeadDimension + groupSize - 1) / groupSize
            : valuePacked.dim(3),
        magnitudeWordsPerGroup: isPlaceholder ? 1 : valuePacked.dim(4),
        bitsetWordsPerGroup: keyLayout.bitsetWordsPerGroup
    )
}

private func turboQuantRestoredScalesPerGroup(_ scales: MLXArray) -> Int {
    scales.shape == [1] ? 0 : scales.dim(4)
}

private enum TurboQuantCacheError: Error, CustomStringConvertible {
    case compressedBackfillUnavailable(String)
    case compressedStorageInvalid(String)
    case cacheLifecycleInvalid(String)
    case residentBudgetExceeded(residentBytes: Int, budgetBytes: Int)

    var description: String {
        switch self {
        case .compressedBackfillUnavailable(let message):
            "TurboQuant compressed cache backfill unavailable: \(message)"
        case .compressedStorageInvalid(let message):
            "TurboQuant compressed cache storage invalid: \(message)"
        case .cacheLifecycleInvalid(let message):
            "TurboQuant cache lifecycle invalid: \(message)"
        case .residentBudgetExceeded(let residentBytes, let budgetBytes):
            "TurboQuant compressed cache resident bytes \(residentBytes) exceed admitted budget \(budgetBytes)"
        }
    }
}

private func turboQuantArrayBytes(_ arrays: [MLXArray]) -> Int {
    arrays.reduce(0) { $0 + $1.nbytes }
}

private func turboQuantCodeBytes(_ code: TurboQuantAttentionCode) -> Int {
    turboQuantArrayBytes([
        code.packedMagnitudes,
        code.signs,
        code.highPrecisionMask,
        code.residualSigns,
        code.scales,
    ])
}

private func turboQuantKeyValueBytes(_ arrays: [MLXArray]) -> (keyBytes: Int, valueBytes: Int) {
    switch arrays.count {
    case 0:
        return (0, 0)
    case 1:
        return (arrays[0].nbytes, 0)
    case 2:
        return (arrays[0].nbytes, arrays[1].nbytes)
    case 4:
        return (
            turboQuantArrayBytes(Array(arrays.prefix(2))),
            turboQuantArrayBytes(Array(arrays.dropFirst(2)))
        )
    case 6:
        return (
            turboQuantArrayBytes(Array(arrays.prefix(3))),
            turboQuantArrayBytes(Array(arrays.dropFirst(3)))
        )
    default:
        let midpoint = arrays.count / 2
        return (
            turboQuantArrayBytes(Array(arrays.prefix(midpoint))),
            turboQuantArrayBytes(Array(arrays.dropFirst(midpoint)))
        )
    }
}

private func turboQuantStorageTokenCapacity(_ arrays: [MLXArray]) -> Int {
    guard let firstTemporalArray = arrays.first(where: { $0.ndim >= 3 }) else { return 0 }
    return max(0, firstTemporalArray.dim(2))
}

private func validateTurboQuantUpdateInputs(
    keys: MLXArray,
    values: MLXArray,
    context: String
) throws {
    guard keys.ndim == 4, values.ndim == 4 else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): keys and values must be rank 4"
        )
    }
    guard keys.dim(0) == values.dim(0), keys.dim(1) == values.dim(1) else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): key/value batch or head counts differ"
        )
    }
    guard keys.dim(2) == values.dim(2) else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): key/value token counts differ"
        )
    }
}

private func turboQuantStorageShape(
    _ code: TurboQuantAttentionCode,
    wordsPerGroup: Int
) -> [Int] {
    [
        code.layout.batchSize,
        code.layout.kvHeadCount,
        code.layout.capacity,
        code.layout.groupsPerVector,
        wordsPerGroup,
    ]
}

private func turboQuantCompactOrStorageShape(
    _ array: MLXArray,
    code: TurboQuantAttentionCode
) -> Bool {
    array.shape == [1]
        || array.shape
            == turboQuantStorageShape(
                code,
                wordsPerGroup: code.layout.bitsetWordsPerGroup
            )
}

private func validateTurboQuantCode(_ code: TurboQuantAttentionCode, context: String) throws {
    let layout = code.layout
    guard TurboQuantAttentionLayout.supportedVersions.contains(layout.layoutVersion) else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): unsupported layout version \(layout.layoutVersion)"
        )
    }
    guard layout.batchSize > 0, layout.kvHeadCount > 0, layout.capacity > 0,
        layout.logicalLength >= 0, layout.logicalLength <= layout.capacity,
        layout.headDimension > 0, code.groupSize > 0
    else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): invalid layout shape \(layout)"
        )
    }
    guard layout.ringOffset >= 0, layout.ringOffset < layout.capacity else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): ring offset \(layout.ringOffset) is outside capacity \(layout.capacity)"
        )
    }
    guard layout.pinnedPrefixLength >= 0, layout.pinnedPrefixLength <= layout.capacity else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): pinned prefix \(layout.pinnedPrefixLength) is outside capacity \(layout.capacity)"
        )
    }
    let ringCapacity = layout.capacity - layout.pinnedPrefixLength
    if ringCapacity == 0 {
        guard layout.ringOffset == 0 else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): ring offset must be zero when pinned prefix consumes capacity"
            )
        }
    } else {
        guard layout.ringOffset < ringCapacity else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): ring offset \(layout.ringOffset) is outside rotating region \(ringCapacity)"
            )
        }
    }
    guard layout.groupsPerVector == (layout.headDimension + code.groupSize - 1) / code.groupSize
    else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): groups per vector does not match head dimension and group size"
        )
    }

    if turboQuantIsCompactValuePlaceholder(code) {
        guard code.packedMagnitudes.dtype == .uint32,
            code.signs.dtype == .uint32,
            code.highPrecisionMask.dtype == .uint32,
            code.residualSigns.dtype == .uint32,
            code.scales.dtype == .float32
        else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): compact value placeholder dtype mismatch"
            )
        }
        return
    }

    let packedShape = turboQuantStorageShape(code, wordsPerGroup: layout.magnitudeWordsPerGroup)
    let bitsetShape = turboQuantStorageShape(code, wordsPerGroup: layout.bitsetWordsPerGroup)
    let scalesShape = turboQuantStorageShape(code, wordsPerGroup: code.scalesPerGroup)

    guard code.packedMagnitudes.dtype == .uint32, code.packedMagnitudes.shape == packedShape else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): packed magnitudes shape/dtype mismatch"
        )
    }
    guard code.scales.dtype == .float32, code.scales.shape == scalesShape else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): scales shape/dtype mismatch"
        )
    }
    if code.role == .key {
        guard code.signs.dtype == .uint32, code.signs.shape == bitsetShape,
            code.highPrecisionMask.dtype == .uint32,
            turboQuantCompactOrStorageShape(code.highPrecisionMask, code: code),
            code.residualSigns.dtype == .uint32,
            turboQuantCompactOrStorageShape(code.residualSigns, code: code)
        else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): key bitset storage shape/dtype mismatch"
            )
        }
    } else {
        guard code.signs.dtype == .uint32,
            turboQuantCompactOrStorageShape(code.signs, code: code),
            code.highPrecisionMask.dtype == .uint32,
            turboQuantCompactOrStorageShape(code.highPrecisionMask, code: code),
            code.residualSigns.dtype == .uint32,
            turboQuantCompactOrStorageShape(code.residualSigns, code: code)
        else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): value bitset storage shape/dtype mismatch"
            )
        }
    }
}

private func validateTurboQuantPair(
    keys: TurboQuantAttentionCode,
    values: TurboQuantAttentionCode,
    context: String
) throws {
    try validateTurboQuantCode(keys, context: "\(context) key")
    try validateTurboQuantCode(values, context: "\(context) value")
    guard keys.role == .key, values.role == .value else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): compressed state must contain key and value codes"
        )
    }
    guard keys.layout.layoutVersion == values.layout.layoutVersion,
        keys.layout.batchSize == values.layout.batchSize,
        keys.layout.kvHeadCount == values.layout.kvHeadCount,
        keys.layout.capacity == values.layout.capacity,
        keys.layout.logicalLength == values.layout.logicalLength,
        keys.layout.ringOffset == values.layout.ringOffset,
        keys.layout.pinnedPrefixLength == values.layout.pinnedPrefixLength,
        keys.preset == values.preset,
        keys.groupSize == values.groupSize
    else {
        throw TurboQuantCacheError.compressedStorageInvalid(
            "\(context): key and value compressed metadata differ"
        )
    }
}

private func turboQuantSnapshotArrays(from state: [MLXArray]) throws -> [String: MLXArray] {
    guard state.count == TurboQuantKVSnapshotArrayName.ordered.count else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot requires \(TurboQuantKVSnapshotArrayName.ordered.count) compressed arrays, found \(state.count)"
        )
    }
    return Dictionary(
        uniqueKeysWithValues: zip(TurboQuantKVSnapshotArrayName.ordered, state.map { $0[.ellipsis] })
    )
}

private func turboQuantSnapshotArrays(
    from state: [MLXArray],
    polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?,
    polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?
) throws -> [String: MLXArray] {
    var arrays = try turboQuantSnapshotArrays(from: state)
    if let polarWHTKeyCode {
        arrays[TurboQuantKVSnapshotArrayName.polarWHTKeyPackedIndices] =
            polarWHTKeyCode.packedIndices[.ellipsis]
        arrays[TurboQuantKVSnapshotArrayName.polarWHTKeyNorms] =
            polarWHTKeyCode.norms[.ellipsis]
    }
    if let polarWHTValueCode {
        arrays[TurboQuantKVSnapshotArrayName.polarWHTValuePackedIndices] =
            polarWHTValueCode.packedIndices[.ellipsis]
        arrays[TurboQuantKVSnapshotArrayName.polarWHTValueNorms] =
            polarWHTValueCode.norms[.ellipsis]
    }
    return arrays
}

private func turboQuantSnapshotOrderedArrays(
    _ arrays: [String: MLXArray]
) throws -> [MLXArray] {
    let required = Set(TurboQuantKVSnapshotArrayName.ordered)
    let known = Set(TurboQuantKVSnapshotArrayName.allKnown)
    let actual = Set(arrays.keys)
    guard required.isSubset(of: actual), actual.isSubset(of: known) else {
        let missing = required.subtracting(actual).sorted().joined(separator: ",")
        let extra = actual.subtracting(known).sorted().joined(separator: ",")
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot payload arrays mismatch; missing [\(missing)] extra [\(extra)]"
        )
    }
    let optionalKey = Set(TurboQuantKVSnapshotArrayName.polarWHTKeyOrdered)
    let optionalValue = Set(TurboQuantKVSnapshotArrayName.polarWHTValueOrdered)
    let keySidecarNames = actual.intersection(optionalKey)
    guard keySidecarNames.isEmpty || keySidecarNames == optionalKey else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT key payload arrays must be all-or-none"
        )
    }
    let valueSidecarNames = actual.intersection(optionalValue)
    guard valueSidecarNames.isEmpty || valueSidecarNames == optionalValue else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT value payload arrays must be all-or-none"
        )
    }
    return try TurboQuantKVSnapshotArrayName.ordered.map { name in
        guard let array = arrays[name] else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot is missing compressed array \(name)"
            )
        }
        return array
    }
}

private func turboQuantSnapshotArrayDescriptors(
    _ arrays: [String: MLXArray]
) -> [TurboQuantKVSnapshotArrayDescriptor] {
    TurboQuantKVSnapshotArrayName.allKnown.compactMap { name in
        arrays[name].map { TurboQuantKVSnapshotArrayDescriptor(name: name, array: $0) }
    }
}

private func turboQuantSnapshotPolarWHTKeyCode(
    manifest: TurboQuantKVSnapshotManifest,
    arrays: [String: MLXArray]
) throws -> TurboQuantPolarWHTAttentionValueCode? {
    let packedName = TurboQuantKVSnapshotArrayName.polarWHTKeyPackedIndices
    let normsName = TurboQuantKVSnapshotArrayName.polarWHTKeyNorms
    let packed = arrays[packedName]
    let norms = arrays[normsName]
    guard packed != nil || norms != nil || manifest.polarWHTKeyPayloadAllocated else {
        return nil
    }
    guard manifest.kvCodec == .polarWHT else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot includes PolarWHT key payload for non-PolarWHT codec"
        )
    }
    guard let packed, let norms else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT key payload arrays must be all-or-none"
        )
    }
    guard let bits = manifest.polarWHTKeyBits,
        let seed = manifest.polarWHTKeySeed,
        let packedWordsPerVector = manifest.polarWHTKeyPackedWordsPerVector
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT key payload metadata is incomplete"
        )
    }
    let layout = TurboQuantAttentionLayout(
        batchSize: manifest.batchSize,
        kvHeadCount: manifest.kvHeadCount,
        capacity: manifest.capacity,
        logicalLength: manifest.logicalLength,
        ringOffset: manifest.ringOffset,
        pinnedPrefixLength: manifest.pinnedPrefixLength,
        headDimension: manifest.keyHeadDimension,
        groupsPerVector: 1,
        magnitudeWordsPerGroup: packedWordsPerVector,
        bitsetWordsPerGroup: 0
    )
    let code = TurboQuantPolarWHTAttentionValueCode(
        layout: layout,
        bits: bits,
        seed: seed,
        packedWordsPerVector: packedWordsPerVector,
        packedIndices: packed,
        norms: norms
    )
    guard manifest.polarWHTKeyBytes == Int64(code.residentPayloadByteCount) else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT key payload bytes do not match manifest"
        )
    }
    _ = try turboQuantPolarWHTReferenceCode(attentionValueCode: code)
    return code
}

private func turboQuantSnapshotPolarWHTValueCode(
    manifest: TurboQuantKVSnapshotManifest,
    arrays: [String: MLXArray]
) throws -> TurboQuantPolarWHTAttentionValueCode? {
    let packedName = TurboQuantKVSnapshotArrayName.polarWHTValuePackedIndices
    let normsName = TurboQuantKVSnapshotArrayName.polarWHTValueNorms
    let packed = arrays[packedName]
    let norms = arrays[normsName]
    guard packed != nil || norms != nil || manifest.polarWHTValuePayloadAllocated else {
        return nil
    }
    guard manifest.kvCodec == .polarWHT else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot includes PolarWHT value payload for non-PolarWHT codec"
        )
    }
    guard let packed, let norms else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT value payload arrays must be all-or-none"
        )
    }
    guard let bits = manifest.polarWHTValueBits,
        let seed = manifest.polarWHTValueSeed,
        let packedWordsPerVector = manifest.polarWHTValuePackedWordsPerVector
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT value payload metadata is incomplete"
        )
    }
    let layout = TurboQuantAttentionLayout(
        batchSize: manifest.batchSize,
        kvHeadCount: manifest.kvHeadCount,
        capacity: manifest.capacity,
        logicalLength: manifest.logicalLength,
        ringOffset: manifest.ringOffset,
        pinnedPrefixLength: manifest.pinnedPrefixLength,
        headDimension: manifest.valueHeadDimension,
        groupsPerVector: 1,
        magnitudeWordsPerGroup: packedWordsPerVector,
        bitsetWordsPerGroup: 0
    )
    let code = TurboQuantPolarWHTAttentionValueCode(
        layout: layout,
        bits: bits,
        seed: seed,
        packedWordsPerVector: packedWordsPerVector,
        packedIndices: packed,
        norms: norms
    )
    guard manifest.polarWHTValueBytes == Int64(code.residentPayloadByteCount) else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT value payload bytes do not match manifest"
        )
    }
    _ = try turboQuantPolarWHTReferenceCode(attentionValueCode: code)
    return code
}

private func turboQuantValidateSnapshotManifest(
    _ manifest: TurboQuantKVSnapshotManifest,
    expectedIdentity: TurboQuantKVSnapshotIdentity,
    expectedCacheKind: String,
    expectedPreset: TurboQuantPreset,
    expectedRequestedBackend: TurboQuantBackend,
    expectedActiveBackend: TurboQuantBackend,
    expectedKVCodec: TurboQuantKVCodec,
    expectedGroupSize: Int,
    expectedValueBits: Int,
    expectedSeed: UInt64,
    expectedMode: QuantizationMode,
    arrays: [String: MLXArray]
) throws -> [MLXArray] {
    guard manifest.schemaVersion >= 1,
        manifest.schemaVersion <= TurboQuantKVSnapshotManifest.currentSchemaVersion
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "Unsupported TurboQuant snapshot schema \(manifest.schemaVersion)"
        )
    }
    guard manifest.cacheKind == expectedCacheKind else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "Snapshot cache kind \(manifest.cacheKind) cannot restore into \(expectedCacheKind)"
        )
    }
    guard manifest.preset == expectedPreset.rawValue,
        manifest.requestedBackend == expectedRequestedBackend.rawValue,
        manifest.activeBackend == expectedActiveBackend.rawValue,
        manifest.kvCodec == expectedKVCodec,
        manifest.groupSize == expectedGroupSize,
        manifest.valueBits == expectedValueBits,
        manifest.seed == expectedSeed,
        manifest.mode == expectedMode.rawValue
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot cache parameters do not match restore target"
        )
    }
    guard manifest.modelID == expectedIdentity.modelID,
        manifest.modelRevision == expectedIdentity.modelRevision,
        manifest.tokenizerHash == expectedIdentity.tokenizerHash,
        manifest.profileHash == expectedIdentity.profileHash,
        manifest.ropeConfigHash == expectedIdentity.ropeConfigHash,
        manifest.tokenPrefixHash == expectedIdentity.tokenPrefixHash,
        manifest.fallbackContractHash == expectedIdentity.fallbackContractHash
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot identity mismatch"
        )
    }
    guard manifest.turboQuantLayoutVersion == TurboQuantAttentionLayout.currentVersion else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "Unsupported TurboQuant snapshot layout \(manifest.turboQuantLayoutVersion)"
        )
    }
    guard manifest.capacity > 0,
        manifest.logicalLength >= 0,
        manifest.logicalLength <= manifest.capacity,
        manifest.pinnedPrefixLength >= 0,
        manifest.pinnedPrefixLength <= manifest.logicalLength,
        manifest.ringOffset >= 0,
        manifest.batchSize > 0,
        manifest.kvHeadCount > 0,
        manifest.keyHeadDimension > 0,
        manifest.valueHeadDimension > 0
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot manifest has invalid length, capacity, or shape metadata"
        )
    }
    let ringCapacity = manifest.capacity - manifest.pinnedPrefixLength
    if ringCapacity == 0 {
        guard manifest.ringOffset == 0 else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot ring offset must be zero when pinned prefix consumes capacity"
            )
        }
    } else {
        guard manifest.ringOffset < ringCapacity else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot ring offset \(manifest.ringOffset) exceeds rotating region \(ringCapacity)"
            )
        }
    }

    let ordered = try turboQuantSnapshotOrderedArrays(arrays)
    guard Int64(turboQuantArrayBytes(Array(ordered.prefix(5)))) == manifest.compressedKeyBytes,
        Int64(turboQuantArrayBytes(Array(ordered.suffix(5)))) == manifest.compressedValueBytes,
        manifest.blobByteCount
            >= manifest.compressedKeyBytes + manifest.compressedValueBytes
                + manifest.polarWHTKeyBytes
                + manifest.polarWHTValueBytes
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot byte counts do not match compressed arrays"
        )
    }
    var descriptors: [String: TurboQuantKVSnapshotArrayDescriptor] = [:]
    for descriptor in manifest.arrays {
        guard descriptors[descriptor.name] == nil else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot manifest has duplicate descriptor \(descriptor.name)"
            )
        }
        descriptors[descriptor.name] = descriptor
    }
    let requiredDescriptors = Set(TurboQuantKVSnapshotArrayName.ordered)
    let optionalKeyDescriptors = Set(TurboQuantKVSnapshotArrayName.polarWHTKeyOrdered)
    let optionalValueDescriptors = Set(TurboQuantKVSnapshotArrayName.polarWHTValueOrdered)
    let optionalDescriptors = optionalKeyDescriptors.union(optionalValueDescriptors)
    let knownDescriptors = requiredDescriptors.union(optionalDescriptors)
    let descriptorNames = Set(descriptors.keys)
    guard requiredDescriptors.isSubset(of: descriptorNames),
        descriptorNames.isSubset(of: knownDescriptors),
        descriptorNames == Set(arrays.keys)
    else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot manifest must describe compressed key/value arrays and optional PolarWHT payload arrays"
        )
    }
    let keySidecarDescriptors = descriptorNames.intersection(optionalKeyDescriptors)
    guard keySidecarDescriptors.isEmpty || keySidecarDescriptors == optionalKeyDescriptors else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT key payload descriptors must be all-or-none"
        )
    }
    let valueSidecarDescriptors = descriptorNames.intersection(optionalValueDescriptors)
    guard valueSidecarDescriptors.isEmpty || valueSidecarDescriptors == optionalValueDescriptors else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot PolarWHT value payload descriptors must be all-or-none"
        )
    }
    for name in TurboQuantKVSnapshotArrayName.allKnown where descriptorNames.contains(name) {
        guard let array = arrays[name] else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot payload is missing array for descriptor \(name)"
            )
        }
        guard let descriptor = descriptors[name] else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot manifest is missing descriptor for \(name)"
            )
        }
        guard descriptor.shape == array.shape,
            descriptor.dtype == String(describing: array.dtype),
            descriptor.byteCount == Int64(array.nbytes)
        else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot descriptor mismatch for \(name)"
            )
        }
    }
    return ordered
}

private struct TurboQuantSnapshotImportedCodes {
    var keys: TurboQuantAttentionCode
    var values: TurboQuantAttentionCode
}

private func turboQuantRequireSnapshotRank(
    _ array: MLXArray,
    name: String,
    rank: Int
) throws {
    guard array.ndim == rank else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot \(name) must be rank \(rank), found \(array.ndim)"
        )
    }
}

private func turboQuantRequireSnapshotRankOrCompact(
    _ array: MLXArray,
    name: String,
    rank: Int
) throws {
    guard array.shape == [1] || array.ndim == rank else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot \(name) must be compact or rank \(rank), found \(array.ndim)"
        )
    }
}

/// T1.4 stage 1: the K scale plane was dieted from 3 scales/group to 2 (norm, residual_norm);
/// the third slot was write-only (always 0.0) and never read. Snapshots taken before the diet
/// still carry a last-dim-3 key scale plane and must be rejected here rather than misread, since
/// schema-range validation alone does not catch this (schemaVersion 4 remains <= currentSchemaVersion).
private func turboQuantRequireKeyScalesPerGroup(_ keyScales: MLXArray) throws {
    guard keyScales.ndim == 5, keyScales.dim(4) == 2 else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot key scale plane has an unsupported layout"
                + " (expected 2 scales per group after the K scale-plane diet;"
                + " got shape \(keyScales.shape)). Recreate the snapshot with the current build."
        )
    }
}

private func turboQuantSnapshotImportedCodes(
    manifest: TurboQuantKVSnapshotManifest,
    ordered: [MLXArray],
    preset: TurboQuantPreset,
    groupSize: Int,
    seed: UInt64,
    valueBits: Int
) throws -> TurboQuantSnapshotImportedCodes {
    guard ordered.count == TurboQuantKVSnapshotArrayName.ordered.count else {
        throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
            "TurboQuant snapshot import requires exactly \(TurboQuantKVSnapshotArrayName.ordered.count) arrays"
        )
    }
    let keyPacked = ordered[0]
    let keySigns = ordered[1]
    let keyHighMask = ordered[2]
    let keyResidualSigns = ordered[3]
    let keyScales = ordered[4]
    let valuePacked = ordered[5]
    let valueSigns = ordered[6]
    let valueHighMask = ordered[7]
    let valueResidualSigns = ordered[8]
    let valueScales = ordered[9]
    let valueIsPlaceholder = valuePacked.shape == [1] && valueScales.shape == [1]

    try turboQuantRequireSnapshotRank(keyPacked, name: "key.packedMagnitudes", rank: 5)
    try turboQuantRequireSnapshotRank(keySigns, name: "key.signs", rank: 5)
    try turboQuantRequireSnapshotRankOrCompact(
        keyHighMask,
        name: "key.highPrecisionMask",
        rank: 5
    )
    try turboQuantRequireSnapshotRankOrCompact(
        keyResidualSigns,
        name: "key.residualSigns",
        rank: 5
    )
    try turboQuantRequireSnapshotRank(keyScales, name: "key.scales", rank: 5)
    try turboQuantRequireKeyScalesPerGroup(keyScales)
    if valueIsPlaceholder {
        try turboQuantRequireSnapshotRankOrCompact(
            valuePacked,
            name: "value.packedMagnitudes",
            rank: 5
        )
    } else {
        try turboQuantRequireSnapshotRank(valuePacked, name: "value.packedMagnitudes", rank: 5)
    }
    try turboQuantRequireSnapshotRankOrCompact(valueSigns, name: "value.signs", rank: 5)
    try turboQuantRequireSnapshotRankOrCompact(
        valueHighMask,
        name: "value.highPrecisionMask",
        rank: 5
    )
    try turboQuantRequireSnapshotRankOrCompact(
        valueResidualSigns,
        name: "value.residualSigns",
        rank: 5
    )
    if valueIsPlaceholder {
        try turboQuantRequireSnapshotRankOrCompact(valueScales, name: "value.scales", rank: 5)
    } else {
        try turboQuantRequireSnapshotRank(valueScales, name: "value.scales", rank: 5)
    }

    let keyLayout = MLX.TurboQuantAttentionLayout(
        layoutVersion: manifest.turboQuantLayoutVersion,
        batchSize: manifest.batchSize,
        kvHeadCount: manifest.kvHeadCount,
        capacity: manifest.capacity,
        logicalLength: manifest.logicalLength,
        ringOffset: manifest.ringOffset,
        pinnedPrefixLength: manifest.pinnedPrefixLength,
        headDimension: manifest.keyHeadDimension,
        groupsPerVector: keyPacked.dim(3),
        magnitudeWordsPerGroup: keyPacked.dim(4),
        bitsetWordsPerGroup: keySigns.dim(4)
    )
    let valueLayout = turboQuantRestoredValueLayout(
        valuePacked: valuePacked,
        keyLayout: keyLayout,
        valueHeadDimension: manifest.valueHeadDimension,
        groupSize: groupSize
    )
    let keys = TurboQuantAttentionCode(
        layout: keyLayout,
        preset: preset,
        role: .key,
        groupSize: groupSize,
        seed: seed,
        scalesPerGroup: keyScales.dim(4),
        packedMagnitudes: keyPacked,
        signs: keySigns,
        highPrecisionMask: keyHighMask,
        residualSigns: keyResidualSigns,
        scales: keyScales
    )
    let values = TurboQuantAttentionCode(
        layout: valueLayout,
        preset: preset,
        role: .value,
        groupSize: groupSize,
        seed: seed ^ turboQuantValueSeedSalt,
        valueBits: valueBits,
        scalesPerGroup: turboQuantRestoredScalesPerGroup(valueScales),
        packedMagnitudes: valuePacked,
        signs: valueSigns,
        highPrecisionMask: valueHighMask,
        residualSigns: valueResidualSigns,
        scales: valueScales
    )
    do {
        try validateTurboQuantPair(keys: keys, values: values, context: "snapshot import")
    } catch {
        throw TurboQuantRuntimeFailure(error)
    }
    return TurboQuantSnapshotImportedCodes(keys: keys, values: values)
}

public final class TurboQuantKVCache: QuantizedKVCache, TurboQuantCompressedKVCacheProtocol {
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?
    private var polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?
    private var polarWHTValueTailCode: TurboQuantPolarWHTAttentionValueCode?
    private var polarWHTDecodedValueBuffer: MLXArray?
    fileprivate var hybridAffineKeySidecar: TurboQuantAffineKeySidecar?
    fileprivate var hybridAffineKeyTailSidecar: TurboQuantAffineKeySidecar?
    public private(set) var keyPageSummary: MLXArray?
    public private(set) var keyCandidateSketch: MLXArray?
    private var compressedStep: Int = 256
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?
    private var restoredLayoutMetadata: RestoredAttentionLayoutMetadata?
    private var lastDecodedTransientBytes: Int = 0
    private var lastNativeAttentionDiagnostics: TurboQuantNativeAttentionDiagnostics?
    private var directRawFallbackCache: KVCacheSimple?
    private var keyPageSummaryUnavailableReason: String?
    private var keyCandidateSketchUnavailableReason: String?
    private let residentBudgetBytes: Int?
    public private(set) var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public private(set) var fallbackResults: [TurboQuantFallbackResult] = []

    public let preset: TurboQuantPreset
    public let kvCodec: TurboQuantKVCodec
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let fallbackPolicy: TurboQuantFallbackPolicy
    public let seed: UInt64
    public let valueBits: Int
    public let precisionPolicy: TurboQuantKVPrecisionPolicy
    public let requestedRuntimeMode: TurboQuantRuntimeMode
    public let resolvedRuntimeMode: TurboQuantRuntimeMode
    public let sparseValuePolicy: TurboQuantSparseValuePolicy
    public let sparseValueSelection: TurboQuantSparseValueSelection
    public let layerIndex: Int?
    public let boundaryProtectedLayerCount: Int
    public let boundaryProtectionReason: String?

    private var shouldMaintainPolarWHTKeySidecar: Bool {
        kvCodec == .polarWHT && !precisionPolicy.key.isHighPrecision
    }

    private var shouldMaintainPolarWHTValueSidecar: Bool {
        kvCodec == .polarWHT
    }

    private var shouldMaintainPolarWHTDecodedValueBuffer: Bool {
        guard usesPolarWHTValueOnlyStorage else { return false }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_DISABLE_POLARWHT_DECODED_VALUE_BUFFER") {
            return false
        }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_ENABLE_POLARWHT_DECODED_VALUE_BUFFER") {
            return true
        }
        return resolvedRuntimeMode == .throughputTurboQuant
    }

    private var shouldMaintainHybridAffineKeySidecar: Bool {
        usesPolarWHTValueOnlyStorage
    }

    fileprivate var usesPolarWHTValueOnlyStorage: Bool {
        turboQuantUsesPolarWHTValueOnlyStorage(kvCodec: kvCodec, precisionPolicy: precisionPolicy)
    }

    private var polarWHTDecodeAttentionPath: TurboQuantAttentionPath {
        precisionPolicy.key.isHighPrecision
            ? .metalHybridK8PolarWHTValue : .metalPolarWHTHybrid
    }

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        kvCodec: TurboQuantKVCodec? = nil,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        requestedRuntimeMode: TurboQuantRuntimeMode = .auto,
        resolvedRuntimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueSelection: TurboQuantSparseValueSelection = .off,
        layerIndex: Int? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        residentBudgetBytes: Int? = nil
    ) {
        let resolvedKVCodec = turboQuantCompressedKVCodec(
            requested: kvCodec,
            backend: backend
        )
        let resolvedValueBits = turboQuantDefaultValueBits(
            preset: preset,
            kvCodec: resolvedKVCodec,
            requestedValueBits: valueBits
        )
        let resolvedPrecisionPolicy =
            precisionPolicy ?? TurboQuantKVPrecisionPolicy.legacy(
                preset: preset,
                valueBits: resolvedValueBits
            )
        self.preset = resolvedPrecisionPolicy.compressedKeyPreset
        self.kvCodec = resolvedKVCodec
        self.requestedBackend = backend
        self.optimizationPolicy = optimizationPolicy
        self.fallbackPolicy = fallbackPolicy
        self.seed = seed
        self.valueBits =
            resolvedPrecisionPolicy.resolvedValueBits
            ?? resolvedValueBits
        self.precisionPolicy = resolvedPrecisionPolicy
        self.requestedRuntimeMode = requestedRuntimeMode
        self.resolvedRuntimeMode = resolvedRuntimeMode
        self.sparseValuePolicy = sparseValuePolicy
        self.sparseValueSelection = sparseValueSelection
        self.layerIndex = layerIndex
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.residentBudgetBytes = residentBudgetBytes
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        super.init(groupSize: groupSize, bits: resolvedPrecisionPolicy.compressedKeyPreset.effectiveBits, mode: mode)
    }

    public override var metaState: [String] {
        get {
            var meta =
                super.metaState + [
                    preset.rawValue,
                    requestedBackend.rawValue,
                    String(seed),
                    "valueBits=\(valueBits)",
                    "kvCodec=\(kvCodec.rawValue)",
                ]
            if let compressedKeys {
                let layout = compressedKeys.layout
                let valueLayout = compressedValues?.layout
                meta += [
                    "turboq-attn-v\(layout.layoutVersion)",
                    String(layout.capacity),
                    String(layout.logicalLength),
                    String(layout.ringOffset),
                    String(layout.headDimension),
                    String(layout.kvHeadCount),
                    lastAttentionPath.rawValue,
                    "keyHeadDimension=\(layout.headDimension)",
                    "valueHeadDimension=\(valueLayout?.headDimension ?? layout.headDimension)",
                    "kvHeadCount=\(layout.kvHeadCount)",
                ]
            }
            return meta
        }
        set {
            super.metaState = Array(newValue.prefix(4))
            keyPageSummary = nil
            keyCandidateSketch = nil
            keyCandidateSketchUnavailableReason = "key candidate sketch was cleared by metadata restore"
            let compressedBase =
                newValue.firstIndex {
                    $0.hasPrefix("turboq-attn-v")
                } ?? (UInt64(newValue.dropFirst(6).first ?? "") == nil ? 6 : 7)
            if newValue.count >= compressedBase + 7,
                let capacity = Int(newValue[compressedBase + 1]),
                let logicalLength = Int(newValue[compressedBase + 2]),
                let ringOffset = Int(newValue[compressedBase + 3]),
                let headDimension = Int(newValue[compressedBase + 4]),
                let kvHeadCount = Int(newValue[compressedBase + 5])
            {
                let keyHeadDimension =
                    turboQuantMetaInt(newValue, key: "keyHeadDimension") ?? headDimension
                let valueHeadDimension =
                    turboQuantMetaInt(newValue, key: "valueHeadDimension") ?? headDimension
                let restoredKVHeadCount =
                    turboQuantMetaInt(newValue, key: "kvHeadCount") ?? kvHeadCount
                offset = logicalLength
                restoredLayoutMetadata = RestoredAttentionLayoutMetadata(
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: ringOffset,
                    pinnedPrefixLength: 0,
                    headDimension: keyHeadDimension,
                    valueHeadDimension: valueHeadDimension,
                    kvHeadCount: restoredKVHeadCount
                )
                if var compressedKeys {
                    compressedKeys.layout.capacity = capacity
                    compressedKeys.layout.logicalLength = logicalLength
                    compressedKeys.layout.ringOffset = ringOffset
                    compressedKeys.layout.headDimension = keyHeadDimension
                    compressedKeys.layout.kvHeadCount = restoredKVHeadCount
                    self.compressedKeys = compressedKeys
                }
                if var compressedValues {
                    compressedValues.layout.capacity = capacity
                    compressedValues.layout.logicalLength = logicalLength
                    compressedValues.layout.ringOffset = ringOffset
                    compressedValues.layout.headDimension = valueHeadDimension
                    compressedValues.layout.kvHeadCount = restoredKVHeadCount
                    self.compressedValues = compressedValues
                }
                if let path = TurboQuantAttentionPath(rawValue: newValue[compressedBase + 6]) {
                    lastAttentionPath = path
                }
            }
        }
    }

    public override var state: [MLXArray] {
        get {
            if turboQuantIsMetalCompressedBackend(activeBackend),
                let compressedKeys,
                let compressedValues
            {
                return [
                    compressedKeys.packedMagnitudes,
                    compressedKeys.signs,
                    compressedKeys.highPrecisionMask,
                    compressedKeys.residualSigns,
                    compressedKeys.scales,
                    compressedValues.packedMagnitudes,
                    compressedValues.signs,
                    compressedValues.highPrecisionMask,
                    compressedValues.residualSigns,
                    compressedValues.scales,
                ]
            }
            if turboQuantIsMetalCompressedBackend(activeBackend), offset == 0 {
                return []
            }
            return super.state
        }
        set {
            if turboQuantIsMetalCompressedBackend(activeBackend), newValue.isEmpty {
                compressedKeys = nil
                compressedValues = nil
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                hybridAffineKeySidecar = nil
                hybridAffineKeyTailSidecar = nil
                keyPageSummary = nil
                keyCandidateSketch = nil
                keyCandidateSketchUnavailableReason = "compressed key state is empty"
                lastDecodedTransientBytes = 0
                cacheLifecycle = .empty
                return
            }
            if turboQuantIsMetalCompressedBackend(activeBackend), newValue.count == 10 {
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                hybridAffineKeySidecar = nil
                hybridAffineKeyTailSidecar = nil
                let capacity = restoredLayoutMetadata?.capacity ?? newValue[0].dim(2)
                let keyHeadDimension =
                    restoredLayoutMetadata?.keyHeadDimension
                    ?? max(groupSize, (newValue[0].dim(3) * groupSize))
                let valueHeadDimension =
                    restoredLayoutMetadata?.valueHeadDimension
                    ?? (
                        newValue[5].shape == [1]
                            ? keyHeadDimension : max(groupSize, (newValue[5].dim(3) * groupSize))
                    )
                let logicalLength =
                    restoredLayoutMetadata?.logicalLength
                    ?? (offset > 0 ? min(offset, capacity) : capacity)
                let keyLayout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[0].dim(0),
                    kvHeadCount: restoredLayoutMetadata?.kvHeadCount ?? newValue[0].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: restoredLayoutMetadata?.ringOffset ?? 0,
                    pinnedPrefixLength: restoredLayoutMetadata?.pinnedPrefixLength ?? 0,
                    headDimension: keyHeadDimension,
                    groupsPerVector: newValue[0].dim(3),
                    magnitudeWordsPerGroup: newValue[0].dim(4),
                    bitsetWordsPerGroup: newValue[1].dim(4)
                )
                let valueLayout = turboQuantRestoredValueLayout(
                    valuePacked: newValue[5],
                    keyLayout: keyLayout,
                    valueHeadDimension: valueHeadDimension,
                    groupSize: groupSize
                )
                // T1.4 stage 1: scalesPerGroup omitted here defaults to 2 (the dieted K scale
                // plane) regardless of newValue[4]'s actual last dim. This setter is non-throwing
                // (a `state` property setter), so a stale 3-scale snapshot restored through this
                // path is not rejected here; it fails closed at first compressed-attention use
                // instead, where mlx-swift's `validateTurboQuantAttentionCode` throws "compressed
                // attention scales per group ... does not match expected" (see
                // TurboQuantValidation.swift). The primary, throwing restore path
                // (`turboQuantSnapshotImportedCodes`) guards this explicitly via
                // `turboQuantRequireKeyScalesPerGroup`.
                compressedKeys = TurboQuantAttentionCode(
                    layout: keyLayout,
                    preset: preset,
                    role: .key,
                    groupSize: groupSize,
                    seed: seed,
                    packedMagnitudes: newValue[0],
                    signs: newValue[1],
                    highPrecisionMask: newValue[2],
                    residualSigns: newValue[3],
                    scales: newValue[4]
                )
                compressedValues = TurboQuantAttentionCode(
                    layout: valueLayout,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: valueBits,
                    scalesPerGroup: turboQuantRestoredScalesPerGroup(newValue[9]),
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
                if let compressedKeys {
                    refreshKeyPageSummary()
                    cacheLifecycle = .compressedCommitted(
                        logicalLength: compressedKeys.layout.logicalLength,
                        capacity: compressedKeys.layout.capacity
                    )
                }
            } else if newValue.count == 10 {
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                hybridAffineKeySidecar = nil
                hybridAffineKeyTailSidecar = nil
                keyPageSummary = nil
                keyCandidateSketch = nil
                keyCandidateSketchUnavailableReason =
                    "compressed TurboQuant state restored without Metal attention support"
                lastUnsupportedShape =
                    "compressed TurboQuant state restored without Metal attention support"
                cacheLifecycle = .failed(
                    reason: lastUnsupportedShape ?? "unsupported compressed state")
            } else {
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                hybridAffineKeySidecar = nil
                hybridAffineKeyTailSidecar = nil
                keyPageSummary = nil
                keyCandidateSketch = nil
                keyCandidateSketchUnavailableReason =
                    "raw or packed fallback state does not expose compressed key sketches"
                super.state = newValue
            }
        }
    }

    public override func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let previousOffset = offset
        guard supportsMLXAffineKVQuantization(
            keyDimension: keys.dim(3),
            valueDimension: values.dim(3),
            keyGroupSize: groupSize,
            valueGroupSize: groupSize
        ) else {
            let rawCache = directRawFallbackCache ?? KVCacheSimple()
            let (rawKeys, rawValues) = rawCache.update(keys: keys, values: values)
            directRawFallbackCache = rawCache
            offset = rawCache.offset
            updatePolarWHTKeySidecar(
                keys: keys,
                previousOffset: previousOffset,
                capacity: roundedLinearPolarWHTValueCapacity(requiredLength: offset)
            )
            updatePolarWHTValueSidecar(
                values: values,
                previousOffset: previousOffset,
                capacity: roundedLinearPolarWHTValueCapacity(requiredLength: offset)
            )
            super.state = [
                placeholderQuantizedTuple(for: rawKeys, bits: bits).0,
                placeholderQuantizedTuple(for: rawKeys, bits: bits).1,
                placeholderQuantizedTuple(for: rawKeys, bits: bits).2,
                placeholderQuantizedTuple(for: rawValues, bits: bits).0,
                placeholderQuantizedTuple(for: rawValues, bits: bits).1,
                placeholderQuantizedTuple(for: rawValues, bits: bits).2,
            ].compactMap { $0 }
            return (
                placeholderQuantizedTuple(for: rawKeys, bits: bits),
                placeholderQuantizedTuple(for: rawValues, bits: bits)
            )
        }
        directRawFallbackCache = nil
        let updated = super.updateQuantized(keys: keys, values: values)
        updatePolarWHTKeySidecar(
            keys: keys,
            previousOffset: previousOffset,
            capacity: roundedLinearPolarWHTValueCapacity(requiredLength: offset)
        )
        updatePolarWHTValueSidecar(
            values: values,
            previousOffset: previousOffset,
            capacity: roundedLinearPolarWHTValueCapacity(requiredLength: offset)
        )
        return updated
    }

    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        guard let compressedKeys, let compressedValues else { return nil }
        return (compressedKeys, compressedValues)
    }

    public var hybridAffineKeyState: QuantizedKVStorage? {
        turboQuantPackedStorage(hybridAffineKeySidecar) ?? super.getQuantizedState()?.0
    }

    public var hybridAffineKeyStateForAttention: QuantizedKVStorage? {
        turboQuantPackedStorage(hybridAffineKeySidecar?.packed) ?? super.getQuantizedState()?.0
    }

    public var hybridAffineKeyTailStateForAttention: QuantizedKVStorage? {
        turboQuantPackedStorage(hybridAffineKeyTailSidecar?.packed)
    }

    public var polarWHTKeyState: TurboQuantPolarWHTAttentionValueCode? {
        copiedTurboQuantPolarWHTCode(polarWHTKeyCode)
    }

    public var polarWHTValueState: TurboQuantPolarWHTAttentionValueCode? {
        copiedTurboQuantPolarWHTValueCode(polarWHTValueCode)
    }

    public var polarWHTKeyStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTKeyCode
    }

    public var polarWHTValueStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTValueCode
    }

    public var polarWHTValueTailStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTValueTailCode
    }

    public var polarWHTDecodedValueState: MLXArray? {
        guard let polarWHTDecodedValueBuffer else { return nil }
        let activeLength = min(offset, polarWHTDecodedValueBuffer.dim(2))
        guard activeLength > 0 else { return nil }
        return polarWHTDecodedValueBuffer[.ellipsis, 0 ..< activeLength, 0...]
    }

    private var polarWHTDecodedValueResidentBytes: Int {
        polarWHTDecodedValueBuffer?.nbytes ?? 0
    }

    private var polarWHTDecodedValueActiveBytes: Int {
        polarWHTDecodedValueState?.nbytes ?? 0
    }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let compressedBytes: Int
        let logicalLength: Int
        let capacity: Int
        if let compressedKeys, let compressedValues {
            compressedBytes =
                turboQuantCodeBytes(compressedKeys) + turboQuantCodeBytes(compressedValues)
                + turboQuantAffineKeySidecarBytes(hybridAffineKeySidecar)
                + turboQuantAffineKeySidecarBytes(hybridAffineKeyTailSidecar)
                + turboQuantArrayBytes([keyPageSummary, keyCandidateSketch].compactMap { $0 })
                + polarWHTKeyBytes
                + polarWHTValueBytes
            logicalLength = compressedKeys.layout.logicalLength
            capacity = compressedKeys.layout.capacity
        } else {
            compressedBytes = polarWHTKeyBytes + polarWHTValueBytes
            logicalLength = offset
            capacity = max(
                polarWHTKeyCode?.layout.capacity ?? 0,
                polarWHTValueCode?.layout.capacity ?? 0
            )
        }
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: logicalLength,
            capacity: capacity,
            compressedBytes: compressedBytes,
            packedFallbackBytes: turboQuantArrayBytes(super.state),
            rawShadowBytes: polarWHTDecodedValueResidentBytes,
            decodedTransientBytes: lastDecodedTransientBytes,
            lifecycle: cacheLifecycle
        )
    }

    public func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot {
        let keyBytes: Int
        let valueBytes: Int
        let logicalLength: Int
        let capacity: Int
        let pinnedPrefixLength: Int
        let ringOffset: Int

        if let compressedKeys, let compressedValues {
            keyBytes =
                turboQuantCodeBytes(compressedKeys)
                + turboQuantAffineKeySidecarBytes(hybridAffineKeySidecar)
            valueBytes = turboQuantCodeBytes(compressedValues)
            logicalLength = compressedKeys.layout.logicalLength
            capacity = compressedKeys.layout.capacity
            pinnedPrefixLength = compressedKeys.layout.pinnedPrefixLength
            ringOffset = compressedKeys.layout.ringOffset
        } else {
            let packedBytes = turboQuantKeyValueBytes(super.state)
            keyBytes = packedBytes.keyBytes
            valueBytes = packedBytes.valueBytes
            logicalLength = offset
            capacity = max(
                turboQuantStorageTokenCapacity(super.state),
                polarWHTKeyCode?.layout.capacity ?? 0,
                polarWHTValueCode?.layout.capacity ?? 0
            )
            pinnedPrefixLength = 0
            ringOffset = 0
        }

        return TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription: cacheLifecycle.turboQuantRuntimeDescription,
            logicalLength: logicalLength,
            capacity: capacity,
            pinnedPrefixLength: pinnedPrefixLength,
            ringOffset: ringOffset,
            keyBytes: keyBytes,
            valueBytes: valueBytes,
            rawShadowAllocated: false,
            packedFallbackAllocated: turboQuantArrayBytes(super.state) > 0,
            lastAttentionPath: lastAttentionPath.rawValue,
            lastFailure: cacheLifecycle.turboQuantRuntimeFailureReason ?? lastUnsupportedShape,
            kvCodec: kvCodec,
            quantizationMode: mode.rawValue,
            keyBits: bits,
            groupSize: groupSize,
            valueBits: valueBits,
            selectedPath: lastAttentionPath.rawValue,
            fallbackReason: cacheLifecycle.turboQuantRuntimeFailureReason ?? lastUnsupportedShape,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            boundaryPolicy: precisionPolicy.boundary,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            runtimeFallbackReason: backendFallbackReason,
            decodedActiveValueBytes: polarWHTDecodedValueActiveBytes,
            activeCacheAllocated: polarWHTDecodedValueResidentBytes > 0,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated
        )
    }

    public func exportSnapshot(
        identity: TurboQuantKVSnapshotIdentity,
        conversationID: UUID,
        snapshotID: UUID = UUID(),
        encryptionKeyID: String = "lm-local-unencrypted",
        createdAt: Date = Date()
    ) throws -> TurboQuantKVSnapshotPayload {
        guard case .compressedCommitted = cacheLifecycle else {
            throw TurboQuantRuntimeFailure.cacheLifecycleInvalid(
                "TurboQuant snapshot export requires committed compressed state; current lifecycle is \(cacheLifecycle)"
            )
        }
        guard turboQuantIsMetalCompressedBackend(activeBackend) else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot export requires active Metal compressed backend"
            )
        }
        try validateCompressedState(context: "snapshot export")
        guard let compressedKeys, let compressedValues else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot export missing compressed key/value state"
            )
        }
        let arrays = try turboQuantSnapshotArrays(
            from: state,
            polarWHTKeyCode: polarWHTKeyCode,
            polarWHTValueCode: polarWHTValueCode
        )
        let descriptors = turboQuantSnapshotArrayDescriptors(arrays)
        let manifest = TurboQuantKVSnapshotManifest(
            snapshotID: snapshotID,
            conversationID: conversationID,
            identity: identity,
            turboQuantLayoutVersion: compressedKeys.layout.layoutVersion,
            logicalLength: compressedKeys.layout.logicalLength,
            pinnedPrefixLength: compressedKeys.layout.pinnedPrefixLength,
            compressedKeyBytes: Int64(turboQuantCodeBytes(compressedKeys)),
            compressedValueBytes: Int64(turboQuantCodeBytes(compressedValues)),
            blobByteCount: Int64(turboQuantArrayBytes(Array(arrays.values))),
            encryptionKeyID: encryptionKeyID,
            createdAt: createdAt,
            cacheKind: "TurboQuantKVCache",
            kvCodec: kvCodec,
            preset: preset.rawValue,
            requestedBackend: requestedBackend.rawValue,
            activeBackend: activeBackend.rawValue,
            quantizationMode: mode.rawValue,
            keyBits: bits,
            groupSize: groupSize,
            valueBits: valueBits,
            seed: seed,
            mode: mode.rawValue,
            capacity: compressedKeys.layout.capacity,
            ringOffset: compressedKeys.layout.ringOffset,
            batchSize: compressedKeys.layout.batchSize,
            kvHeadCount: compressedKeys.layout.kvHeadCount,
            keyHeadDimension: compressedKeys.layout.headDimension,
            valueHeadDimension: compressedValues.layout.headDimension,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            boundaryPolicy: precisionPolicy.boundary,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            runtimeFallbackReason: backendFallbackReason,
            selectedPath: lastAttentionPath.rawValue,
            fallbackReason: backendFallbackReason,
            polarWHTKeyBytes: Int64(polarWHTKeyBytes),
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTKeyBits: polarWHTKeyCode?.bits,
            polarWHTKeySeed: polarWHTKeyCode?.seed,
            polarWHTKeyPackedWordsPerVector: polarWHTKeyCode?.packedWordsPerVector,
            polarWHTValueBytes: Int64(polarWHTValueBytes),
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated,
            polarWHTValueBits: polarWHTValueCode?.bits,
            polarWHTValueSeed: polarWHTValueCode?.seed,
            polarWHTValuePackedWordsPerVector: polarWHTValueCode?.packedWordsPerVector,
            arrays: descriptors
        )
        return TurboQuantKVSnapshotPayload(manifest: manifest, compressedArrays: arrays)
    }

    public func importSnapshot(
        _ payload: TurboQuantKVSnapshotPayload,
        expectedIdentity: TurboQuantKVSnapshotIdentity
    ) throws {
        guard turboQuantIsMetalCompressedBackend(activeBackend) else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant snapshot import requires active Metal compressed backend"
            )
        }
        let manifest = payload.manifest
        let ordered = try turboQuantValidateSnapshotManifest(
            manifest,
            expectedIdentity: expectedIdentity,
            expectedCacheKind: "TurboQuantKVCache",
            expectedPreset: preset,
            expectedRequestedBackend: requestedBackend,
            expectedActiveBackend: activeBackend,
            expectedKVCodec: kvCodec,
            expectedGroupSize: groupSize,
            expectedValueBits: valueBits,
            expectedSeed: seed,
            expectedMode: mode,
            arrays: payload.compressedArrays
        )
        let imported = try turboQuantSnapshotImportedCodes(
            manifest: manifest,
            ordered: ordered,
            preset: preset,
            groupSize: groupSize,
            seed: seed,
            valueBits: valueBits
        )
        let importedPolarWHTKeyCode = try turboQuantSnapshotPolarWHTKeyCode(
            manifest: manifest,
            arrays: payload.compressedArrays
        )
        let importedPolarWHTValueCode = try turboQuantSnapshotPolarWHTValueCode(
            manifest: manifest,
            arrays: payload.compressedArrays
        )
        let residentBytes = turboQuantCodeBytes(imported.keys) + turboQuantCodeBytes(imported.values)
            + (importedPolarWHTKeyCode?.residentPayloadByteCount ?? 0)
            + (importedPolarWHTValueCode?.residentPayloadByteCount ?? 0)
        if let residentBudgetBytes, residentBytes > residentBudgetBytes {
            throw TurboQuantRuntimeFailure.fallbackBudgetExceeded(
                "TurboQuant snapshot resident bytes \(residentBytes) exceed admitted budget \(residentBudgetBytes)"
            )
        }
        super.state = []
        restoredLayoutMetadata = RestoredAttentionLayoutMetadata(
            capacity: manifest.capacity,
            logicalLength: manifest.logicalLength,
            ringOffset: manifest.ringOffset,
            pinnedPrefixLength: manifest.pinnedPrefixLength,
            headDimension: manifest.keyHeadDimension,
            valueHeadDimension: manifest.valueHeadDimension,
            kvHeadCount: manifest.kvHeadCount
        )
        offset = manifest.logicalLength
        compressedKeys = imported.keys
        compressedValues = imported.values
        polarWHTKeyCode = importedPolarWHTKeyCode
        polarWHTValueCode = importedPolarWHTValueCode
        polarWHTValueTailCode = nil
        hybridAffineKeyTailSidecar = nil
        polarWHTDecodedValueBuffer = nil
        refreshKeyPageSummary()
        lastDecodedTransientBytes = 0
        lastUnsupportedShape = nil
        cacheLifecycle = .compressedCommitted(
            logicalLength: manifest.logicalLength,
            capacity: manifest.capacity
        )
    }

    public func recordFallback(_ result: TurboQuantFallbackResult) {
        fallbackResults.append(result)
        lastUnsupportedShape = result.reason
        if result.policy != .exactRequired {
            lastNativeAttentionDiagnostics = nil
        }
        if let toPath = result.toPath {
            lastAttentionPath = toPath
        }
        switch result.policy {
        case .packedAllowed:
            cacheLifecycle = .degradedPackedFallback(reason: result.reason)
        case .compressedDecodeAllowed:
            cacheLifecycle = .degradedDecodedFallback(reason: result.reason)
        case .fatalOnFailure:
            cacheLifecycle = .failed(reason: result.reason)
        case .exactRequired:
            break
        }
    }

    public func recordNativeAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        selection: TurboQuantSparseValueSelection
    ) {
        lastAttentionPath = .nativeMLXCompressed
        lastUnsupportedShape = nil
        lastNativeAttentionDiagnostics = diagnostics
        cacheLifecycle = .compressedCommitted(
            logicalLength: compressedKeys?.layout.logicalLength ?? offset,
            capacity: compressedKeys?.layout.capacity ?? compressedKeys?.layout.logicalLength ?? offset
        )
    }

    public func recordPolarWHTAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        path: TurboQuantAttentionPath
    ) {
        lastAttentionPath = path
        lastUnsupportedShape = nil
        lastNativeAttentionDiagnostics = diagnostics
        cacheLifecycle = .compressedCommitted(
            logicalLength: compressedKeys?.layout.logicalLength ?? offset,
            capacity: compressedKeys?.layout.capacity ?? compressedKeys?.layout.logicalLength ?? offset
        )
    }

    private func refreshKeyPageSummary() {
        guard activeBackend == .metalPolarQJL else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason =
                "key page summaries require metalPolarQJL backend; active backend is \(activeBackend.rawValue)"
            return
        }
        guard let compressedKeys else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "no compressed key state is available"
            return
        }
        guard compressedKeys.layout.ringOffset == 0 else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason =
                "ring offset \(compressedKeys.layout.ringOffset) makes page summaries unsafe"
            return
        }
        guard compressedKeys.layout.pinnedPrefixLength == 0 else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason =
                "pinned prefix length \(compressedKeys.layout.pinnedPrefixLength) makes page summaries unsafe"
            return
        }
        guard compressedKeys.layout.logicalLength > 0 else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "compressed key state is empty"
            return
        }
        do {
            keyPageSummary = try MLX.turboQuantKeyPageSummaries(keyCode: compressedKeys)
            keyPageSummaryUnavailableReason = nil
        } catch {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "key page summary build failed: \(error)"
        }
    }

    public func ensureKeyPageSummary() {
        if keyPageSummary == nil {
            refreshKeyPageSummary()
        }
    }

    private func invalidateKeyCandidateSketch(reason: String? = nil) {
        keyCandidateSketch = nil
        keyCandidateSketchUnavailableReason =
            reason
            ?? turboQuantKeyCandidateSketchUnavailableReason(
                activeBackend: activeBackend,
                compressedKeys: compressedKeys,
                artifactName: "key candidate sketches"
            )
            ?? "key candidate sketch has not been built"
    }

    private func refreshKeyCandidateSketch() {
        if let reason = turboQuantKeyCandidateSketchUnavailableReason(
            activeBackend: activeBackend,
            compressedKeys: compressedKeys,
            artifactName: "key candidate sketches"
        ) {
            invalidateKeyCandidateSketch(reason: reason)
            return
        }
        guard let compressedKeys else {
            invalidateKeyCandidateSketch(reason: "no compressed key state is available")
            return
        }
        do {
            keyCandidateSketch = try turboQuantBuildKeyCandidateSketch(keyCode: compressedKeys)
            keyCandidateSketchUnavailableReason = nil
        } catch {
            invalidateKeyCandidateSketch(reason: "key candidate sketch build failed: \(error)")
        }
    }

    public func ensureKeyCandidateSketch() {
        if keyCandidateSketch == nil {
            refreshKeyCandidateSketch()
        }
    }

    private func updateKeyCandidateSketchAfterLinearAppend(
        encodedKeys: TurboQuantAttentionCode,
        previousOffset: Int,
        tokenCount: Int
    ) {
        guard keyCandidateSketch != nil else {
            if sparseValueSelection.mode == .candidateSparse {
                refreshKeyCandidateSketch()
            } else {
                keyCandidateSketchUnavailableReason =
                    turboQuantKeyCandidateSketchUnavailableReason(
                        activeBackend: activeBackend,
                        compressedKeys: compressedKeys,
                        artifactName: "key candidate sketches"
                    )
                    ?? "key candidate sketch has not been built"
            }
            return
        }
        guard activeBackend == .metalPolarQJL,
            let compressedKeys,
            compressedKeys.layout.ringOffset == 0,
            compressedKeys.layout.logicalLength > 0
        else {
            invalidateKeyCandidateSketch()
            return
        }
        guard tokenCount == 1, encodedKeys.layout.logicalLength == 1 else {
            refreshKeyCandidateSketch()
            return
        }
        let pageIndex = previousOffset / MLX.turboQuantKeyPageSummaryPageSize
        guard let existingSketch = keyCandidateSketch else { return }
        do {
            keyCandidateSketch = try turboQuantUpdateKeyCandidateSketchPage(
                existingSketch: existingSketch,
                encodedKeys: encodedKeys,
                pageIndex: pageIndex
            )
            keyCandidateSketchUnavailableReason = nil
        } catch {
            invalidateKeyCandidateSketch(reason: "key candidate sketch update failed: \(error)")
        }
    }

    private func updateKeyPageSummaryAfterLinearAppend(
        encodedKeys: TurboQuantAttentionCode,
        previousOffset: Int,
        tokenCount: Int
    ) {
        updateKeyCandidateSketchAfterLinearAppend(
            encodedKeys: encodedKeys,
            previousOffset: previousOffset,
            tokenCount: tokenCount
        )
        guard activeBackend == .metalPolarQJL,
            let compressedKeys,
            compressedKeys.layout.ringOffset == 0,
            compressedKeys.layout.pinnedPrefixLength == 0,
            compressedKeys.layout.logicalLength > 0
        else {
            keyPageSummary = nil
            refreshKeyPageSummary()
            return
        }
        guard tokenCount == 1,
            let existingSummary = keyPageSummary,
            encodedKeys.layout.logicalLength == 1
        else {
            refreshKeyPageSummary()
            return
        }

        let pageIndex = previousOffset / MLX.turboQuantKeyPageSummaryPageSize
        guard pageIndex >= 0, pageIndex < existingSummary.dim(2) else {
            refreshKeyPageSummary()
            return
        }

        do {
            let updatedSummary = existingSummary
            let pageRange = pageIndex ..< (pageIndex + 1)
            let currentPage = updatedSummary[.ellipsis, pageRange, 0...]
            let tokenSummary = try MLX.turboQuantKeyPageSummaries(keyCode: encodedKeys)
            updatedSummary[.ellipsis, pageRange, 0...] = MLX.maximum(
                currentPage,
                tokenSummary.asType(.float32)
            )
            keyPageSummary = updatedSummary
        } catch {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "key page summary single-token update failed: \(error)"
        }
    }

    public func validateCompressedState(context: String) throws {
        guard let compressedKeys, let compressedValues else {
            if offset == 0 { return }
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): no compressed state exists for offset \(offset)"
            )
        }
        try validateTurboQuantPair(keys: compressedKeys, values: compressedValues, context: context)
        try enforceResidentBudget(context: context)
    }

    public func makeCompressedUpdateCheckpoint(appendingTokenCount tokenCount: Int)
        -> TurboQuantCompressedUpdateCheckpoint
    {
        guard let compressedKeys, let compressedValues else {
            return TurboQuantCompressedUpdateCheckpoint(
                payload: .fullState(
                    offset: offset,
                    metaState: metaState,
                    state: state.map { $0[.ellipsis] },
                    hybridAffineKeyState: copiedTurboQuantAffineKeySidecar(hybridAffineKeySidecar),
                    polarWHTKeyCode: copiedTurboQuantPolarWHTCode(polarWHTKeyCode),
                    polarWHTValueCode: copiedTurboQuantPolarWHTValueCode(polarWHTValueCode)
                ))
        }

        return TurboQuantCompressedUpdateCheckpoint(
            payload: .linearCompressed(
                offset: offset,
                compressedKeys: compressedKeys,
                compressedValues: compressedValues,
                hybridAffineKeyState: copiedTurboQuantAffineKeySidecar(hybridAffineKeySidecar),
                polarWHTKeyCode: copiedTurboQuantPolarWHTCode(polarWHTKeyCode),
                polarWHTValueCode: copiedTurboQuantPolarWHTValueCode(polarWHTValueCode),
                packedFallbackState: super.state.map { $0[.ellipsis] },
                fallbackResultCount: fallbackResults.count,
                lifecycle: cacheLifecycle,
                lastAttentionPath: lastAttentionPath,
                lastUnsupportedShape: lastUnsupportedShape,
                lastDecodedTransientBytes: lastDecodedTransientBytes
            ))
    }

    public func restoreCompressedUpdateCheckpoint(
        _ checkpoint: TurboQuantCompressedUpdateCheckpoint
    ) {
        switch checkpoint.payload {
        case .fullState(
            let previousOffset,
            let previousMetaState,
            let previousState,
            let previousHybridAffineKeyState,
            let previousPolarWHTKeyCode,
            let previousPolarWHTValueCode
        ):
            metaState = previousMetaState
            state = previousState
            offset = previousOffset
            hybridAffineKeySidecar = copiedTurboQuantAffineKeySidecar(previousHybridAffineKeyState)
            polarWHTKeyCode = copiedTurboQuantPolarWHTCode(previousPolarWHTKeyCode)
            polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(previousPolarWHTValueCode)
            polarWHTValueTailCode = nil
            hybridAffineKeyTailSidecar = nil
            polarWHTDecodedValueBuffer = nil

        case .linearCompressed(
            let previousOffset,
            let previousKeys,
            let previousValues,
            let previousHybridAffineKeyState,
            let previousPolarWHTKeyCode,
            let previousPolarWHTValueCode,
            let previousPackedFallbackState,
            let previousFallbackResultCount,
            let previousLifecycle,
            let previousAttentionPath,
            let previousUnsupportedShape,
            let previousDecodedTransientBytes
        ):
            offset = previousOffset
            compressedKeys = previousKeys
            compressedValues = previousValues
            hybridAffineKeySidecar = copiedTurboQuantAffineKeySidecar(previousHybridAffineKeyState)
            polarWHTKeyCode = copiedTurboQuantPolarWHTCode(previousPolarWHTKeyCode)
            polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(previousPolarWHTValueCode)
            polarWHTValueTailCode = nil
            hybridAffineKeyTailSidecar = nil
            polarWHTDecodedValueBuffer = nil
            refreshKeyPageSummary()
            super.state = previousPackedFallbackState
            cacheLifecycle = previousLifecycle
            lastAttentionPath = previousAttentionPath
            lastUnsupportedShape = previousUnsupportedShape
            lastDecodedTransientBytes = previousDecodedTransientBytes
            if fallbackResults.count > previousFallbackResultCount {
                fallbackResults.removeLast(fallbackResults.count - previousFallbackResultCount)
            }

        case .rotatingCompressed:
            break

        case .rotatingFullState:
            break
        }
    }

    private func enforceResidentBudget(context: String) throws {
        guard let residentBudgetBytes else { return }
        let residentBytes = cacheFootprint.residentBytes
        guard residentBytes <= residentBudgetBytes else {
            throw TurboQuantCacheError.residentBudgetExceeded(
                residentBytes: residentBytes,
                budgetBytes: residentBudgetBytes
            )
        }
    }

    public func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray) {
        try validateCompressedState(context: "decode compressed state")
        guard let compressedKeys, let compressedValues else {
            throw TurboQuantCacheError.compressedStorageInvalid("decode compressed state missing")
        }
        if kvCodec == .polarWHT,
            let polarWHTKeyCode,
            let polarWHTValueCode
        {
            cacheLifecycle = .decodeCompressed
            let decodedKeys =
                TurboQuantKernelAvailability.current.supportsMetalPolarWHTCodec
                ? try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    polarWHTKeyCode,
                    outputDType: outputDType
                )
                : try MLX.turboQuantPolarWHTReferenceDecodeAttentionValues(
                    polarWHTKeyCode
                ).asType(outputDType)
            let decodedValues =
                TurboQuantKernelAvailability.current.supportsMetalPolarWHTCodec
                ? try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    polarWHTValueCode,
                    outputDType: outputDType
                )
                : try MLX.turboQuantPolarWHTReferenceDecodeAttentionValues(
                    polarWHTValueCode
                ).asType(outputDType)
            lastDecodedTransientBytes = decodedKeys.nbytes + decodedValues.nbytes
            return (decodedKeys, decodedValues)
        }
        if turboQuantIsCompactValuePlaceholder(compressedValues) {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "hybrid K8+PolarWHT-V cache has no affine compressed value payload to decode"
            )
        }
        // 4.1: gate the full-context materialization so a decode under memory pressure degrades
        // (recoverable error) instead of aborting the process with a hard allocation failure.
        try turboQuantGuardFallbackMaterialization(
            keys: compressedKeys, values: compressedValues, dtype: outputDType)
        cacheLifecycle = .decodeCompressed
        let decodedKeys = try MLX.turboQuantMetalDecodeAttention(
            compressedKeys,
            outputDType: outputDType
        )
        let decodedValues = try MLX.turboQuantMetalDecodeAttention(
            compressedValues,
            outputDType: outputDType
        )
        lastDecodedTransientBytes = decodedKeys.nbytes + decodedValues.nbytes
        return (decodedKeys, decodedValues)
    }

    public func releaseRawShadow() {}

    public override func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        if let state = super.getQuantizedState() {
            return state
        }
        guard let compressedKeys,
            let compressedValues,
            turboQuantSupportsPackedFallback(
                keyCode: compressedKeys,
                valueCode: compressedValues,
                groupSize: groupSize
            ),
            let decodedKeys = try? MLX.turboQuantMetalDecodeAttention(
                compressedKeys,
                outputDType: .float16  // 4.1: fp16 scratch halves the fallback materialization spike
            ),
            let decodedValues = try? MLX.turboQuantMetalDecodeAttention(
                compressedValues,
                outputDType: .float16  // 4.1: fp16 scratch halves the fallback materialization spike
            )
        else {
            return nil
        }

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed ^ turboQuantValueSeedSalt,
            valueBits: valueBits
        )
        let packedKeys = turboQuantized(decodedKeys, configuration: keyConfiguration)
        let packedValues = turboQuantized(decodedValues, configuration: valueConfiguration)
        super.state = [
            packedKeys.weight, packedKeys.scales, packedKeys.biases,
            packedValues.weight, packedValues.scales, packedValues.biases,
        ].compactMap { $0 }
        return super.getQuantizedState()
    }

    public var attentionDiagnostics: TurboQuantAttentionDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        let sparseSelection = sparseValueSelection.resolved(
            runtimeMode: resolvedRuntimeMode,
            contextLength: compressedKeys?.layout.logicalLength ?? offset,
            policy: sparseValuePolicy
        )
        let sparseVInactiveReason = turboQuantSparseVInactiveReason(
            enabled: sparseSelection.isEnabled,
            kvCodec: kvCodec,
            activeBackend: activeBackend,
            nativeDiagnostics: lastNativeAttentionDiagnostics,
            fallbackReason: backendFallbackReason
        )
        return TurboQuantAttentionDiagnostics(
            layerIndex: layerIndex,
            metalAttentionAvailable: turboQuantMetalAttentionAvailable(
                kvCodec: kvCodec,
                availability: availability
            ),
            activeAttentionPath: lastAttentionPath,
            nativeBackend: availability.attentionCapabilities.nativeCompressedAttention == true
                ? "nativeMLX" : nil,
            nativeBackendVersion: availability.attentionCapabilities.nativeBackendVersion,
            nativeFallbackReason: availability.attentionCapabilities.nativeFallbackReason,
            nativeKernelKind: lastNativeAttentionDiagnostics?.kernelKind,
            nativeSparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: optimizationPolicy,
            fallbackReason: sparseVInactiveReason,
            lastUnsupportedShape: lastUnsupportedShape,
            rawFallbackAllocated: false,
            cacheLifecycle: cacheLifecycle,
            lastFallback: fallbackResults.last,
            sparseVEnabled: sparseSelection.isEnabled,
            sparseVThreshold: sparseSelection.resolvedThreshold,
            sparseVSelectionMode: sparseSelection.mode,
            sparseVTopK: sparseSelection.topK,
            sparseVCumulativeMass: sparseSelection.cumulativeMass,
            sparseVMaxTopK: sparseSelection.maxTopK,
            sparseVRecentTokenCount: sparseSelection.recentTokens,
            sparseVOlderTokenCount: sparseSelection.mode == .candidateSparse
                ? sparseSelection.topK : nil,
            sparseVPageCandidateCount: sparseSelection.candidatePages,
            sparseVSkippedTokens: lastNativeAttentionDiagnostics?.sparseSkippedTokens,
            sparseVTotalTokens: lastNativeAttentionDiagnostics?.sparseTotalTokens,
            sparseVActive: turboQuantSparseVActive(lastNativeAttentionDiagnostics),
            sparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            keyBits: bits,
            valueBits: valueBits,
            keyGroupSize: groupSize,
            valueGroupSize: groupSize,
            keyPageSummaryAvailable: keyPageSummary != nil,
            keyPageSummaryShape: keyPageSummary?.shape,
            keyPageSummaryUnavailableReason: keyPageSummaryUnavailableReason,
            keyCandidateSketchAvailable: keyCandidateSketch != nil,
            keyCandidateSketchShape: keyCandidateSketch?.shape,
            keyCandidateSketchUnavailableReason: keyCandidateSketchUnavailableReason,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated
        )
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        let sparseSelection = sparseValueSelection.resolved(
            runtimeMode: resolvedRuntimeMode,
            contextLength: compressedKeys?.layout.logicalLength ?? offset,
            policy: sparseValuePolicy
        )
        let sparseVInactiveReason = turboQuantSparseVInactiveReason(
            enabled: sparseSelection.isEnabled,
            kvCodec: kvCodec,
            activeBackend: activeBackend,
            nativeDiagnostics: lastNativeAttentionDiagnostics,
            fallbackReason: backendFallbackReason
        )
        return TurboQuantKVCacheDiagnostics(
            layerIndex: layerIndex,
            kvCodec: kvCodec,
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: sparseVInactiveReason,
            metalCodecAvailable: turboQuantMetalCodecAvailable(
                kvCodec: kvCodec,
                availability: availability
            ),
            metalAttentionAvailable: turboQuantMetalAttentionAvailable(
                kvCodec: kvCodec,
                availability: availability
            ),
            activeAttentionPath: lastAttentionPath,
            nativeBackend: availability.attentionCapabilities.nativeCompressedAttention == true
                ? "nativeMLX" : nil,
            nativeBackendVersion: availability.attentionCapabilities.nativeBackendVersion,
            nativeFallbackReason: availability.attentionCapabilities.nativeFallbackReason,
            nativeKernelKind: lastNativeAttentionDiagnostics?.kernelKind,
            nativeSparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: optimizationPolicy,
            lastUnsupportedShape: lastUnsupportedShape,
            groupSize: groupSize,
            bits: bits,
            valueBits: valueBits,
            maxSize: nil,
            rawFallbackAllocated: false,
            cacheLifecycle: cacheLifecycle,
            lastFallback: fallbackResults.last,
            footprint: cacheFootprint,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated,
            sparseVEnabled: sparseSelection.isEnabled,
            sparseVThreshold: sparseSelection.resolvedThreshold,
            sparseVSelectionMode: sparseSelection.mode,
            sparseVTopK: sparseSelection.topK,
            sparseVCumulativeMass: sparseSelection.cumulativeMass,
            sparseVMaxTopK: sparseSelection.maxTopK,
            sparseVRecentTokenCount: sparseSelection.recentTokens,
            sparseVOlderTokenCount: sparseSelection.mode == .candidateSparse
                ? sparseSelection.topK : nil,
            sparseVPageCandidateCount: sparseSelection.candidatePages,
            sparseVSkippedTokens: lastNativeAttentionDiagnostics?.sparseSkippedTokens,
            sparseVTotalTokens: lastNativeAttentionDiagnostics?.sparseTotalTokens,
            sparseVActive: turboQuantSparseVActive(lastNativeAttentionDiagnostics),
            sparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            keyCandidateSketchAvailable: keyCandidateSketch != nil,
            keyCandidateSketchShape: keyCandidateSketch?.shape,
            keyCandidateSketchUnavailableReason: keyCandidateSketchUnavailableReason
        )
    }

    public func supportsCompressedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        let availability = TurboQuantKernelAvailability.current
        let nativeAttentionAvailable =
            availability.attentionCapabilities.nativeCompressedAttention == true
        if kvCodec == .polarWHT {
            let requiresHybridValueKernel = precisionPolicy.key.isHighPrecision
            let polarWHTAttentionAvailable =
                requiresHybridValueKernel
                    ? availability.attentionCapabilities.hybridK8PolarWHTValueAttention
                    : availability.supportsMetalPolarWHTAttention
            guard activeBackend == .metalPolarWHT,
                polarWHTAttentionAvailable
            else {
                lastUnsupportedShape = turboQuantPolarWHTAttentionUnavailableReason(
                    backendFallbackReason: backendFallbackReason,
                    valueBytes: polarWHTValueBytes,
                    payloadAllocated: polarWHTValuePayloadAllocated
                )
                return false
            }
            guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else {
                lastUnsupportedShape = "queries/keys/values must be rank 4"
                return false
            }
            guard queries.dim(2) == 1 else {
                lastUnsupportedShape =
                    "PolarWHT native attention is decode-only; qLen=\(queries.dim(2))"
                return false
            }
            guard queries.dim(0) == keys.dim(0), queries.dim(0) == values.dim(0),
                keys.dim(2) == values.dim(2), keys.dim(1) == values.dim(1)
            else {
                lastUnsupportedShape = "query/key/value batch, token, or KV head counts differ"
                return false
            }
            guard queries.dim(3) == keys.dim(3),
                keys.dim(3) == values.dim(3),
                turboQuantSupportsPolarWHTAttentionDimension(queries.dim(3))
            else {
                lastUnsupportedShape =
                    "PolarWHT requires matching power-of-two head dimensions <= 256; q=\(queries.dim(3)) k=\(keys.dim(3)) v=\(values.dim(3))"
                return false
            }
            guard queries.dim(1) % keys.dim(1) == 0 else {
                lastUnsupportedShape = "query heads must be a multiple of KV heads"
                return false
            }
            lastAttentionPath = polarWHTDecodeAttentionPath
            lastUnsupportedShape = nil
            return true
        }
        guard kvCodec == .polarQJL else {
            lastUnsupportedShape = "unsupported TurboQuant KV codec \(kvCodec.rawValue)"
            return false
        }
        guard activeBackend == .metalPolarQJL,
            availability.supportsMetalPolarQJLAttention || nativeAttentionAvailable
        else {
            lastUnsupportedShape = "metal backend unavailable"
            return false
        }
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else {
            lastUnsupportedShape = "queries/keys/values must be rank 4"
            return false
        }
        guard queries.dim(0) == keys.dim(0), queries.dim(0) == values.dim(0) else {
            lastUnsupportedShape = "query/key/value batch sizes must match"
            return false
        }
        guard keys.dim(2) == values.dim(2), keys.dim(1) == values.dim(1) else {
            lastUnsupportedShape = "key/value token and head counts must match"
            return false
        }
        guard turboQuantSupportsAttentionDimension(queries.dim(3)),
            turboQuantSupportsAttentionDimension(keys.dim(3)),
            turboQuantSupportsAttentionDimension(values.dim(3)),
            queries.dim(3) == keys.dim(3)
        else {
            lastUnsupportedShape =
                "unsupported head dimension q=\(queries.dim(3)) k=\(keys.dim(3)) v=\(values.dim(3))"
            return false
        }
        guard queries.dim(1) % keys.dim(1) == 0 else {
            lastUnsupportedShape = "query heads must be a multiple of KV heads"
            return false
        }
        let keyCode: TurboQuantAttentionCode
        if let compressedKeys {
            keyCode = compressedKeys
        } else if let placeholder = try? placeholderCode(for: keys, role: .key) {
            keyCode = placeholder
        } else {
            lastUnsupportedShape = "failed to create compressed attention placeholder"
            return false
        }
        let supportsNative =
            optimizationPolicy != .conservative && nativeAttentionAvailable
            && queries.dim(2) <= 8 && turboQuantNativeSupportsMask(mask)
        let supportsTiled =
            queries.dim(3) == values.dim(3) && prefersOnlineFusedAttention
            && MLX.turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: mask
            )
        lastAttentionPath =
            supportsNative ? .nativeMLXCompressed : (supportsTiled ? .tiledOnlineFused : .twoStageCompressed)
        lastUnsupportedShape =
            supportsNative || supportsTiled
            ? nil
            : "online fused attention is not throughput-admitted for head dimension \(queries.dim(3)); using two-stage compressed attention"
        return true
    }

    public func updateCompressed(keys: MLXArray, values: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    ) {
        let previousOffset = offset
        let tokenCount = keys.dim(2)
        if tokenCount > 1 {
            cacheLifecycle = previousOffset == 0 ? .rawPrefillChunkOpen : cacheLifecycle
        }
        cacheLifecycle = .compressingChunk(start: previousOffset, count: tokenCount)
        try ensureCompressedCapacity(
            keys: keys, values: values, requiredLength: previousOffset + tokenCount)

        if usesPolarWHTValueOnlyStorage,
            previousOffset > 0,
            tokenCount == 1,
            TurboQuantRuntimeControl.enabled("TURBOQUANT_ENABLE_HYBRID_POLARWHT_TAIL"),
            !TurboQuantRuntimeControl.enabled("TURBOQUANT_DISABLE_HYBRID_POLARWHT_TAIL")
        {
            guard var currentKeys = compressedKeys, var currentValues = compressedValues else {
                throw TurboQuantCacheError.compressedStorageInvalid(
                    "hybrid PolarWHT tail append requires existing compressed placeholder state"
                )
            }
            guard updateHybridAffineKeyAndPolarWHTValueTailSidecars(keys: keys, values: values)
            else {
                throw TurboQuantCacheError.compressedStorageInvalid(
                    "hybrid PolarWHT tail append failed for token at offset \(previousOffset)"
                )
            }
            offset = previousOffset + tokenCount
            currentKeys.layout.logicalLength = offset
            currentValues.layout.logicalLength = offset
            compressedKeys = currentKeys
            compressedValues = currentValues
            lastDecodedTransientBytes = 0
            cacheLifecycle = .compressedCommitted(
                logicalLength: offset,
                capacity: currentKeys.layout.capacity
            )
            return (currentKeys, currentValues)
        }

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed ^ turboQuantValueSeedSalt,
            valueBits: valueBits
        )
        if previousOffset == 0, tokenCount > 0, compressedKeys == nil, compressedValues == nil {
            let capacity =
                ((compressedStep + tokenCount - 1) / compressedStep) * compressedStep
            let encodedKeys: TurboQuantAttentionCode
            if usesPolarWHTValueOnlyStorage {
                let keyLayout = try MLX.turboQuantAttentionLayout(
                    for: keys,
                    preset: preset,
                    role: .key,
                    groupSize: groupSize,
                    capacity: capacity,
                    logicalLength: tokenCount
                )
                encodedKeys = try MLX.turboQuantEmptyAttentionCode(
                    layout: keyLayout,
                    preset: preset,
                    role: .key,
                    groupSize: groupSize,
                    seed: seed
                )
            } else {
                encodedKeys = try MLX.turboQuantMetalEncodeAttention(
                    keys,
                    configuration: keyConfiguration,
                    capacity: capacity,
                    logicalLength: tokenCount,
                    stream: .gpu
                )
            }
            let encodedValues: TurboQuantAttentionCode
            if usesPolarWHTValueOnlyStorage {
                let valueLayout = try MLX.turboQuantAttentionLayout(
                    for: values,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    valueBits: valueBits,
                    capacity: capacity,
                    logicalLength: tokenCount
                )
                encodedValues = turboQuantCompactValuePlaceholderCode(
                    layout: valueLayout,
                    preset: preset,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: valueBits
                )
            } else {
                encodedValues = try MLX.turboQuantMetalEncodeAttention(
                    values,
                    configuration: valueConfiguration,
                    capacity: capacity,
                    logicalLength: tokenCount,
                    stream: .gpu
                )
            }
            compressedKeys = encodedKeys
            compressedValues = encodedValues
            refreshKeyPageSummary()
            if sparseValueSelection.mode == .candidateSparse {
                refreshKeyCandidateSketch()
            }
            offset = tokenCount
            if usesPolarWHTValueOnlyStorage,
                updateHybridAffineKeyAndPolarWHTValueSidecars(
                    keys: keys,
                    values: values,
                    previousOffset: previousOffset,
                    capacity: capacity
                )
            {
                updatePolarWHTKeySidecar(
                    keys: keys,
                    previousOffset: previousOffset,
                    capacity: capacity
                )
            } else {
                updateHybridAffineKeySidecar(keys: keys, previousOffset: previousOffset)
                updatePolarWHTKeySidecar(
                    keys: keys,
                    previousOffset: previousOffset,
                    capacity: capacity
                )
                updatePolarWHTValueSidecar(
                    values: values,
                    previousOffset: previousOffset,
                    capacity: capacity
                )
            }
            lastDecodedTransientBytes = 0
            try validateCompressedState(context: "compressed initial append")
            cacheLifecycle = .compressedCommitted(
                logicalLength: tokenCount,
                capacity: capacity
            )
            return (encodedKeys, encodedValues)
        }
        let encodedKeys =
            usesPolarWHTValueOnlyStorage
            ? nil
            : try MLX.turboQuantMetalEncodeAttention(
                keys,
                configuration: keyConfiguration,
                stream: .gpu
            )
        let encodedValues =
            usesPolarWHTValueOnlyStorage
            ? nil
            : try MLX.turboQuantMetalEncodeAttention(
                values,
                configuration: valueConfiguration,
                stream: .gpu
            )

        var currentKeys = compressedKeys!
        var currentValues = compressedValues!
        // 3A (audit 1.3): drop the stored references (struct copy → refcount 2) so the slice-update
        // writes below donate in place instead of reallocating each plane. Restored at the reassign.
        compressedKeys = nil
        compressedValues = nil
        let range = previousOffset ..< (previousOffset + tokenCount)
        if let encodedKeys {
            currentKeys.packedMagnitudes[.ellipsis, range, 0..., 0...] = encodedKeys.packedMagnitudes
            currentKeys.signs[.ellipsis, range, 0..., 0...] = encodedKeys.signs
            if currentKeys.highPrecisionMask.ndim == 5 {
                currentKeys.highPrecisionMask[.ellipsis, range, 0..., 0...] =
                    encodedKeys.highPrecisionMask
            }
            if currentKeys.residualSigns.ndim == 5 {
                currentKeys.residualSigns[.ellipsis, range, 0..., 0...] = encodedKeys.residualSigns
            }
            currentKeys.scales[.ellipsis, range, 0..., 0...] = encodedKeys.scales
        }

        if let encodedValues {
            currentValues.packedMagnitudes[.ellipsis, range, 0..., 0...] =
                encodedValues.packedMagnitudes
            if currentValues.signs.ndim == 5 {
                currentValues.signs[.ellipsis, range, 0..., 0...] = encodedValues.signs
            }
            if currentValues.highPrecisionMask.ndim == 5 {
                currentValues.highPrecisionMask[.ellipsis, range, 0..., 0...] =
                    encodedValues.highPrecisionMask
            }
            if currentValues.residualSigns.ndim == 5 {
                currentValues.residualSigns[.ellipsis, range, 0..., 0...] =
                    encodedValues.residualSigns
            }
            currentValues.scales[.ellipsis, range, 0..., 0...] = encodedValues.scales
        }

        if fallbackPolicy == .packedAllowed,
            turboQuantSupportsPackedFallback(keys: keys, values: values, groupSize: groupSize),
            super.getQuantizedState() != nil
        {
            _ = super.updateQuantized(keys: keys, values: values)
        } else {
            if turboQuantIsMetalCompressedBackend(activeBackend), !super.state.isEmpty {
                super.state = []
            }
            offset += tokenCount
        }
        let fusedHybridSidecarsUpdated =
            usesPolarWHTValueOnlyStorage
            && updateHybridAffineKeyAndPolarWHTValueSidecars(
                keys: keys,
                values: values,
                previousOffset: previousOffset,
                capacity: currentKeys.layout.capacity
            )
        if !fusedHybridSidecarsUpdated {
            updateHybridAffineKeySidecar(keys: keys, previousOffset: previousOffset)
        }
        currentKeys.layout.logicalLength = offset
        currentValues.layout.logicalLength = offset
        compressedKeys = currentKeys
        compressedValues = currentValues
        updatePolarWHTKeySidecar(
            keys: keys,
            previousOffset: previousOffset,
            capacity: currentKeys.layout.capacity
        )
        if !fusedHybridSidecarsUpdated {
            updatePolarWHTValueSidecar(
                values: values,
                previousOffset: previousOffset,
                capacity: currentKeys.layout.capacity
            )
        }
        if let encodedKeys {
            updateKeyPageSummaryAfterLinearAppend(
                encodedKeys: encodedKeys,
                previousOffset: previousOffset,
                tokenCount: tokenCount
            )
        }
        lastDecodedTransientBytes = 0
        try validateCompressedState(context: "compressed append")
        cacheLifecycle = .compressedCommitted(
            logicalLength: offset,
            capacity: currentKeys.layout.capacity
        )
        return (currentKeys, currentValues)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = super.trim(n)
        guard trimmed > 0 else { return 0 }

        if var compressedKeys, var compressedValues {
            compressedKeys.layout.logicalLength = offset
            compressedKeys.layout.ringOffset = 0
            compressedKeys.layout.pinnedPrefixLength = 0
            compressedValues.layout.logicalLength = offset
            compressedValues.layout.ringOffset = 0
            compressedValues.layout.pinnedPrefixLength = 0
            self.compressedKeys = compressedKeys
            self.compressedValues = compressedValues
            refreshKeyPageSummary()
            cacheLifecycle = .compressedCommitted(
                logicalLength: offset,
                capacity: compressedKeys.layout.capacity
            )
            lastDecodedTransientBytes = 0
        }
        trimHybridAffineKeySidecar(to: offset)
        if var polarWHTValueCode {
            polarWHTValueCode.layout.logicalLength = min(offset, polarWHTValueCode.layout.capacity)
            polarWHTValueCode.layout.ringOffset = 0
            polarWHTValueCode.layout.pinnedPrefixLength = 0
            self.polarWHTValueCode = polarWHTValueCode
        }
        if let polarWHTDecodedValueBuffer {
            if offset > 0 {
                let activeLength = min(offset, polarWHTDecodedValueBuffer.dim(2))
                self.polarWHTDecodedValueBuffer =
                    polarWHTDecodedValueBuffer[.ellipsis, 0 ..< activeLength, 0...]
            } else {
                self.polarWHTDecodedValueBuffer = nil
            }
        }
        if var polarWHTKeyCode {
            polarWHTKeyCode.layout.logicalLength = min(offset, polarWHTKeyCode.layout.capacity)
            polarWHTKeyCode.layout.ringOffset = 0
            polarWHTKeyCode.layout.pinnedPrefixLength = 0
            self.polarWHTKeyCode = polarWHTKeyCode
        }

        return trimmed
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        lastAttentionPath = .mlxPackedFallback
        lastUnsupportedShape = "compressed attention failed: \(message)"
        lastNativeAttentionDiagnostics = nil
        keyPageSummary = nil
        invalidateKeyCandidateSketch(reason: "compressed attention failed: \(message)")
        cacheLifecycle = .failed(reason: message)
    }

    public override func copy() -> any KVCache {
        let new = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend,
            kvCodec: kvCodec,
            optimizationPolicy: optimizationPolicy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: valueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            residentBudgetBytes: residentBudgetBytes
        )
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        new.hybridAffineKeySidecar = copiedTurboQuantAffineKeySidecar(hybridAffineKeySidecar)
        new.hybridAffineKeyTailSidecar = copiedTurboQuantAffineKeySidecar(
            hybridAffineKeyTailSidecar)
        new.polarWHTKeyCode = copiedTurboQuantPolarWHTCode(polarWHTKeyCode)
        new.polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(polarWHTValueCode)
        new.polarWHTValueTailCode = copiedTurboQuantPolarWHTValueCode(polarWHTValueTailCode)
        new.polarWHTDecodedValueBuffer = polarWHTDecodedValueBuffer?[.ellipsis]
        return new
    }

    private var polarWHTValueBytes: Int {
        guard kvCodec == .polarWHT else { return 0 }
        return (polarWHTValueCode?.residentPayloadByteCount ?? 0)
            + (polarWHTValueTailCode?.residentPayloadByteCount ?? 0)
    }

    private var polarWHTValuePayloadAllocated: Bool {
        polarWHTValueBytes > 0
    }

    private var polarWHTKeyBytes: Int {
        guard kvCodec == .polarWHT else { return 0 }
        return polarWHTKeyCode?.residentPayloadByteCount ?? 0
    }

    private var polarWHTKeyPayloadAllocated: Bool {
        polarWHTKeyBytes > 0
    }

    private func roundedLinearPolarWHTValueCapacity(requiredLength: Int) -> Int {
        guard requiredLength > 0 else { return 0 }
        return ((compressedStep + requiredLength - 1) / compressedStep) * compressedStep
    }

    private func updateHybridAffineKeySidecar(keys: MLXArray, previousOffset: Int) {
        guard shouldMaintainHybridAffineKeySidecar else {
            hybridAffineKeySidecar = nil
            return
        }
        guard supportsMLXAffineQuantization(dimension: keys.dim(3), groupSize: groupSize) else {
            hybridAffineKeySidecar = nil
            lastUnsupportedShape =
                "hybrid affine K sidecar requires key dimension \(keys.dim(3)) to be divisible by group size \(groupSize)"
            return
        }
        let configuration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: .metalPolarQJL,
            seed: seed
        )
        let encoded = turboQuantized(keys, configuration: configuration)
        let logicalLength = previousOffset + encoded.weight.dim(2)
        let requestedCapacity = max(
            compressedKeys?.layout.capacity ?? 0,
            encoded.weight.dim(2),
            logicalLength
        )
        guard var current = hybridAffineKeySidecar, previousOffset > 0 else {
            let sidecar = TurboQuantAffineKeySidecar(
                packed: expandedHybridAffineKeyStorage(
                    from: encoded,
                    logicalLength: logicalLength,
                    capacity: requestedCapacity
                ),
                logicalLength: logicalLength
            )
            hybridAffineKeySidecar = sidecar
            return
        }
        if current.capacity < requestedCapacity {
            current.packed = expandedHybridAffineKeyStorage(
                from: current.packed,
                logicalLength: current.logicalLength,
                capacity: requestedCapacity
            )
        }
        guard previousOffset <= current.capacity, logicalLength <= current.capacity else {
            hybridAffineKeySidecar = TurboQuantAffineKeySidecar(
                packed: expandedHybridAffineKeyStorage(
                    from: encoded,
                    logicalLength: encoded.weight.dim(2),
                    capacity: requestedCapacity
                ),
                logicalLength: encoded.weight.dim(2)
            )
            lastUnsupportedShape =
                "hybrid affine K sidecar append exceeded capacity \(current.capacity) for logical length \(logicalLength)"
            return
        }
        hybridAffineKeySidecar = nil
        let range = previousOffset ..< logicalLength
        current.packed.weight[.ellipsis, range, 0...] = encoded.weight
        current.packed.scales[.ellipsis, range, 0...] = encoded.scales
        if let encodedBiases = encoded.biases {
            current.packed.biases?[.ellipsis, range, 0...] = encodedBiases
        }
        current.logicalLength = logicalLength
        hybridAffineKeySidecar = current
    }

    private func trimHybridAffineKeySidecar(to logicalLength: Int) {
        guard var current = hybridAffineKeySidecar else { return }
        guard logicalLength > 0 else {
            hybridAffineKeySidecar = nil
            return
        }
        current.logicalLength = min(logicalLength, current.capacity)
        hybridAffineKeySidecar = current
    }

    private func polarWHTValueStorageMatches(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        values: MLXArray
    ) -> Bool {
        code.bits == valueBits
            && code.seed == seed ^ turboQuantValueSeedSalt
            && code.layout.batchSize == values.dim(0)
            && code.layout.kvHeadCount == values.dim(1)
            && code.layout.headDimension == values.dim(3)
    }

    private func polarWHTKeyStorageMatches(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        keys: MLXArray
    ) -> Bool {
        code.bits == bits
            && code.seed == seed
            && code.layout.batchSize == keys.dim(0)
            && code.layout.kvHeadCount == keys.dim(1)
            && code.layout.headDimension == keys.dim(3)
    }

    private func expandedPolarWHTValueCode(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        capacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode {
        var layout = code.layout
        layout.capacity = capacity
        let empty = try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: code.bits,
            seed: code.seed,
            normStorage: code.norms.dtype
        )
        var expanded = empty
        let copyRange = 0 ..< code.layout.capacity
        expanded.packedIndices[.ellipsis, copyRange, 0...] = code.packedIndices
        expanded.norms[.ellipsis, copyRange] = code.norms
        expanded.layout.logicalLength = code.layout.logicalLength
        return expanded
    }

    private func expandedPolarWHTKeyCode(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        capacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode {
        var layout = code.layout
        layout.capacity = capacity
        let empty = try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: code.bits,
            seed: code.seed,
            normStorage: code.norms.dtype
        )
        var expanded = empty
        let copyRange = 0 ..< code.layout.capacity
        expanded.packedIndices[.ellipsis, copyRange, 0...] = code.packedIndices
        expanded.norms[.ellipsis, copyRange] = code.norms
        expanded.layout.logicalLength = code.layout.logicalLength
        return expanded
    }

    private func ensurePolarWHTValueStorage(
        values: MLXArray,
        previousOffset: Int,
        capacity requestedCapacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode? {
        let capacity = max(requestedCapacity, previousOffset)
        guard capacity >= previousOffset else { return nil }
        if let current = polarWHTValueCode {
            guard polarWHTValueStorageMatches(current, values: values) else {
                polarWHTValueCode = nil
                return nil
            }
            if current.layout.capacity >= capacity {
                return current
            }
            return try expandedPolarWHTValueCode(current, capacity: capacity)
        }
        guard previousOffset == 0 else { return nil }
        let layout = MLX.TurboQuantAttentionLayout(
            batchSize: values.dim(0),
            kvHeadCount: values.dim(1),
            capacity: capacity,
            logicalLength: 0,
            headDimension: values.dim(3),
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 0,
            bitsetWordsPerGroup: 0
        )
        return try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: valueBits,
            seed: seed ^ turboQuantValueSeedSalt,
            normStorage: .float32
        )
    }

    private func ensurePolarWHTValueTailStorage(
        values: MLXArray,
        capacity requestedCapacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode? {
        let capacity = max(0, requestedCapacity)
        if let current = polarWHTValueTailCode {
            guard polarWHTValueStorageMatches(current, values: values) else {
                polarWHTValueTailCode = nil
                return nil
            }
            if current.layout.capacity >= capacity {
                return current
            }
            return try expandedPolarWHTValueCode(current, capacity: capacity)
        }
        let layout = MLX.TurboQuantAttentionLayout(
            batchSize: values.dim(0),
            kvHeadCount: values.dim(1),
            capacity: capacity,
            logicalLength: 0,
            headDimension: values.dim(3),
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 0,
            bitsetWordsPerGroup: 0
        )
        return try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: valueBits,
            seed: seed ^ turboQuantValueSeedSalt,
            normStorage: .float32
        )
    }

    private func ensurePolarWHTDecodedValueBuffer(
        values: MLXArray,
        capacity requestedCapacity: Int
    ) -> MLXArray {
        let capacity = max(0, requestedCapacity)
        if let current = polarWHTDecodedValueBuffer,
            current.dim(0) == values.dim(0),
            current.dim(1) == values.dim(1),
            current.dim(3) == values.dim(3),
            current.dtype == values.dtype
        {
            if current.dim(2) >= capacity {
                return current
            }
            var shape = current.shape
            shape[2] = capacity
            let expanded = MLXArray.zeros(shape, dtype: current.dtype)
            if current.dim(2) > 0 {
                expanded[.ellipsis, 0 ..< current.dim(2), 0...] = current
            }
            return expanded
        }
        return MLXArray.zeros(
            [values.dim(0), values.dim(1), capacity, values.dim(3)],
            dtype: values.dtype
        )
    }

    private func updatePolarWHTDecodedValueBuffer(
        decodedChunk: MLXArray,
        values: MLXArray,
        previousOffset: Int,
        capacity: Int
    ) {
        let tokenCount = decodedChunk.dim(2)
        guard tokenCount > 0 else { return }
        let buffer = ensurePolarWHTDecodedValueBuffer(values: values, capacity: capacity)
        let logicalLength = previousOffset + tokenCount
        guard logicalLength <= buffer.dim(2) else {
            polarWHTDecodedValueBuffer = nil
            return
        }
        let destination = previousOffset ..< logicalLength
        polarWHTDecodedValueBuffer = nil
        buffer[.ellipsis, destination, 0...] = decodedChunk
        polarWHTDecodedValueBuffer = buffer
    }

    private func ensurePolarWHTKeyStorage(
        keys: MLXArray,
        previousOffset: Int,
        capacity requestedCapacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode? {
        let capacity = max(requestedCapacity, previousOffset)
        guard capacity >= previousOffset else { return nil }
        if let current = polarWHTKeyCode {
            guard polarWHTKeyStorageMatches(current, keys: keys) else {
                polarWHTKeyCode = nil
                return nil
            }
            if current.layout.capacity >= capacity {
                return current
            }
            return try expandedPolarWHTKeyCode(current, capacity: capacity)
        }
        guard previousOffset == 0 else { return nil }
        let layout = MLX.TurboQuantAttentionLayout(
            batchSize: keys.dim(0),
            kvHeadCount: keys.dim(1),
            capacity: capacity,
            logicalLength: 0,
            headDimension: keys.dim(3),
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 0,
            bitsetWordsPerGroup: 0
        )
        return try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: bits,
            seed: seed,
            normStorage: .float32
        )
    }

    fileprivate func updatePolarWHTKeySidecar(
        keys: MLXArray,
        previousOffset: Int,
        capacity requestedCapacity: Int? = nil
    ) {
        guard shouldMaintainPolarWHTKeySidecar else {
            polarWHTKeyCode = nil
            return
        }
        let keys = keys.contiguous(stream: .gpu)
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return }
        let logicalLength = previousOffset + tokenCount
        let capacity =
            requestedCapacity
            ?? max(
                polarWHTKeyCode?.layout.capacity ?? 0,
                roundedLinearPolarWHTValueCapacity(requiredLength: logicalLength)
            )

        do {
            let encodedChunk = try turboQuantEncodePolarWHTAttentionValues(
                keys,
                bits: bits,
                seed: seed,
                capacity: tokenCount,
                logicalLength: tokenCount
            )
            guard var current = try ensurePolarWHTKeyStorage(
                keys: keys,
                previousOffset: previousOffset,
                capacity: max(capacity, logicalLength)
            ) else {
                return
            }
            polarWHTKeyCode = nil
            let destination = previousOffset ..< logicalLength
            let source = 0 ..< tokenCount
            current.packedIndices[.ellipsis, destination, 0...] =
                encodedChunk.packedIndices[.ellipsis, source, 0...]
            current.norms[.ellipsis, destination] = encodedChunk.norms[.ellipsis, source]
            current.layout.logicalLength = logicalLength
            current.layout.ringOffset = 0
            current.layout.pinnedPrefixLength = 0
            polarWHTKeyCode = current
        } catch {
            polarWHTKeyCode = nil
            lastUnsupportedShape = "PolarWHT key sidecar update failed: \(error)"
        }
    }

    fileprivate func updatePolarWHTValueSidecar(
        values: MLXArray,
        previousOffset: Int,
        capacity requestedCapacity: Int? = nil
    ) {
        guard shouldMaintainPolarWHTValueSidecar else {
            polarWHTValueCode = nil
            polarWHTDecodedValueBuffer = nil
            return
        }
        let values = values.contiguous(stream: .gpu)
        let tokenCount = values.dim(2)
        guard tokenCount > 0 else { return }
        let logicalLength = previousOffset + tokenCount
        let capacity =
            requestedCapacity
            ?? max(
                polarWHTValueCode?.layout.capacity ?? 0,
                roundedLinearPolarWHTValueCapacity(requiredLength: logicalLength)
            )

        do {
            let encodedChunk = try turboQuantEncodePolarWHTAttentionValues(
                values,
                bits: valueBits,
                seed: seed ^ turboQuantValueSeedSalt,
                capacity: tokenCount,
                logicalLength: tokenCount
            )
            guard var current = try ensurePolarWHTValueStorage(
                values: values,
                previousOffset: previousOffset,
                capacity: max(capacity, logicalLength)
            ) else {
                return
            }
            polarWHTValueCode = nil
            let destination = previousOffset ..< logicalLength
            let source = 0 ..< tokenCount
            current.packedIndices[.ellipsis, destination, 0...] =
                encodedChunk.packedIndices[.ellipsis, source, 0...]
            current.norms[.ellipsis, destination] = encodedChunk.norms[.ellipsis, source]
            current.layout.logicalLength = logicalLength
            current.layout.ringOffset = 0
            current.layout.pinnedPrefixLength = 0
            polarWHTValueCode = current
            if shouldMaintainPolarWHTDecodedValueBuffer {
                let decodedChunk = try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    encodedChunk,
                    outputDType: values.dtype
                )
                updatePolarWHTDecodedValueBuffer(
                    decodedChunk: decodedChunk,
                    values: values,
                    previousOffset: previousOffset,
                    capacity: max(capacity, logicalLength)
                )
            } else {
                polarWHTDecodedValueBuffer = nil
            }
        } catch {
            polarWHTValueCode = nil
            polarWHTDecodedValueBuffer = nil
            lastUnsupportedShape = "PolarWHT value sidecar update failed: \(error)"
        }
    }

    @discardableResult
    private func updateHybridAffineKeyAndPolarWHTValueSidecars(
        keys: MLXArray,
        values: MLXArray,
        previousOffset: Int,
        capacity requestedCapacity: Int? = nil
    ) -> Bool {
        guard shouldMaintainHybridAffineKeySidecar, shouldMaintainPolarWHTValueSidecar else {
            return false
        }
        guard supportsMLXAffineQuantization(dimension: keys.dim(3), groupSize: groupSize) else {
            return false
        }
        let keys = keys.contiguous(stream: .gpu)
        let values = values.contiguous(stream: .gpu)
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return true }
        guard values.dim(0) == keys.dim(0),
            values.dim(1) == keys.dim(1),
            values.dim(2) == tokenCount,
            values.dim(3) == keys.dim(3)
        else {
            return false
        }

        let logicalLength = previousOffset + tokenCount
        let capacity =
            requestedCapacity
            ?? max(
                hybridAffineKeySidecar?.capacity ?? 0,
                polarWHTValueCode?.layout.capacity ?? 0,
                roundedLinearPolarWHTValueCapacity(requiredLength: logicalLength)
            )
        let valueSeed = seed ^ turboQuantValueSeedSalt

        do {
            if previousOffset == 0 {
                hybridAffineKeyTailSidecar = nil
                polarWHTValueTailCode = nil
                let fused = try MLX.turboQuantMetalHybridAffineK8PolarWHTValueEncode(
                    keys: keys,
                    values: values,
                    keyGroupSize: groupSize,
                    valueBits: valueBits,
                    valueSeed: valueSeed,
                    capacity: max(capacity, logicalLength),
                    logicalLength: tokenCount
                )
                hybridAffineKeySidecar = TurboQuantAffineKeySidecar(
                    packed: fused.key,
                    logicalLength: logicalLength
                )
                polarWHTValueCode = fused.value
                if shouldMaintainPolarWHTDecodedValueBuffer {
                    polarWHTDecodedValueBuffer =
                        try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                            fused.value,
                            outputDType: values.dtype
                        )
                } else {
                    polarWHTDecodedValueBuffer = nil
                }
                return true
            }

            guard var currentKey = hybridAffineKeySidecar else {
                return false
            }
            let requiredCapacity = max(capacity, logicalLength)
            if currentKey.capacity < requiredCapacity {
                currentKey.packed = expandedHybridAffineKeyStorage(
                    from: currentKey.packed,
                    logicalLength: currentKey.logicalLength,
                    capacity: requiredCapacity
                )
            }
            guard logicalLength <= currentKey.capacity else {
                return false
            }

            let fused = try MLX.turboQuantMetalHybridAffineK8PolarWHTValueEncode(
                keys: keys,
                values: values,
                keyGroupSize: groupSize,
                valueBits: valueBits,
                valueSeed: valueSeed,
                capacity: tokenCount,
                logicalLength: tokenCount
            )
            let destination = previousOffset ..< logicalLength
            let source = 0 ..< tokenCount
            guard var currentValue = try ensurePolarWHTValueStorage(
                values: values,
                previousOffset: previousOffset,
                capacity: requiredCapacity
            ) else {
                return false
            }
            hybridAffineKeySidecar = nil
            polarWHTValueCode = nil
            currentKey.packed.weight[.ellipsis, destination, 0...] =
                fused.key.weight[.ellipsis, source, 0...]
            currentKey.packed.scales[.ellipsis, destination, 0...] =
                fused.key.scales[.ellipsis, source, 0...]
            if let fusedBiases = fused.key.biases {
                currentKey.packed.biases?[.ellipsis, destination, 0...] =
                    fusedBiases[.ellipsis, source, 0...]
            }
            currentKey.logicalLength = logicalLength

            currentValue.packedIndices[.ellipsis, destination, 0...] =
                fused.value.packedIndices[.ellipsis, source, 0...]
            currentValue.norms[.ellipsis, destination] = fused.value.norms[.ellipsis, source]
            currentValue.layout.logicalLength = logicalLength
            currentValue.layout.ringOffset = 0
            currentValue.layout.pinnedPrefixLength = 0

            hybridAffineKeySidecar = currentKey
            polarWHTValueCode = currentValue
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            if shouldMaintainPolarWHTDecodedValueBuffer {
                let decodedChunk = try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    fused.value,
                    outputDType: values.dtype
                )
                updatePolarWHTDecodedValueBuffer(
                    decodedChunk: decodedChunk,
                    values: values,
                    previousOffset: previousOffset,
                    capacity: requiredCapacity
                )
            } else {
                polarWHTDecodedValueBuffer = nil
            }
            return true
        } catch {
            polarWHTDecodedValueBuffer = nil
            return false
        }
    }

    @discardableResult
    private func updateHybridAffineKeyAndPolarWHTValueTailSidecars(
        keys: MLXArray,
        values: MLXArray
    ) -> Bool {
        guard shouldMaintainHybridAffineKeySidecar, shouldMaintainPolarWHTValueSidecar else {
            return false
        }
        guard supportsMLXAffineQuantization(dimension: keys.dim(3), groupSize: groupSize) else {
            return false
        }
        let keys = keys.contiguous(stream: .gpu)
        let values = values.contiguous(stream: .gpu)
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return true }
        guard values.dim(0) == keys.dim(0),
            values.dim(1) == keys.dim(1),
            values.dim(2) == tokenCount,
            values.dim(3) == keys.dim(3)
        else {
            return false
        }

        let previousTailOffset = hybridAffineKeyTailSidecar?.logicalLength ?? 0
        let logicalTailLength = previousTailOffset + tokenCount
        let capacity = max(
            hybridAffineKeyTailSidecar?.capacity ?? 0,
            polarWHTValueTailCode?.layout.capacity ?? 0,
            roundedLinearPolarWHTValueCapacity(requiredLength: logicalTailLength)
        )
        let valueSeed = seed ^ turboQuantValueSeedSalt

        do {
            let fused = try MLX.turboQuantMetalHybridAffineK8PolarWHTValueEncode(
                keys: keys,
                values: values,
                keyGroupSize: groupSize,
                valueBits: valueBits,
                valueSeed: valueSeed,
                capacity: tokenCount,
                logicalLength: tokenCount
            )

            if previousTailOffset == 0 {
                let tailCapacity = max(capacity, logicalTailLength)
                hybridAffineKeyTailSidecar = TurboQuantAffineKeySidecar(
                    packed: expandedHybridAffineKeyStorage(
                        from: fused.key,
                        logicalLength: tokenCount,
                        capacity: tailCapacity
                    ),
                    logicalLength: logicalTailLength
                )
                var tailValue = try expandedPolarWHTValueCode(fused.value, capacity: tailCapacity)
                tailValue.layout.logicalLength = logicalTailLength
                tailValue.layout.ringOffset = 0
                tailValue.layout.pinnedPrefixLength = 0
                polarWHTValueTailCode = tailValue
                polarWHTDecodedValueBuffer = nil
                return true
            }

            guard var currentKey = hybridAffineKeyTailSidecar else {
                return false
            }
            let requiredCapacity = max(capacity, logicalTailLength)
            if currentKey.capacity < requiredCapacity {
                currentKey.packed = expandedHybridAffineKeyStorage(
                    from: currentKey.packed,
                    logicalLength: currentKey.logicalLength,
                    capacity: requiredCapacity
                )
            }
            guard logicalTailLength <= currentKey.capacity,
                var currentValue = try ensurePolarWHTValueTailStorage(
                    values: values,
                    capacity: requiredCapacity
                )
            else {
                return false
            }

            let destination = previousTailOffset ..< logicalTailLength
            let source = 0 ..< tokenCount
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            currentKey.packed.weight[.ellipsis, destination, 0...] =
                fused.key.weight[.ellipsis, source, 0...]
            currentKey.packed.scales[.ellipsis, destination, 0...] =
                fused.key.scales[.ellipsis, source, 0...]
            if let fusedBiases = fused.key.biases {
                currentKey.packed.biases?[.ellipsis, destination, 0...] =
                    fusedBiases[.ellipsis, source, 0...]
            }
            currentKey.logicalLength = logicalTailLength

            currentValue.packedIndices[.ellipsis, destination, 0...] =
                fused.value.packedIndices[.ellipsis, source, 0...]
            currentValue.norms[.ellipsis, destination] = fused.value.norms[.ellipsis, source]
            currentValue.layout.logicalLength = logicalTailLength
            currentValue.layout.ringOffset = 0
            currentValue.layout.pinnedPrefixLength = 0

            hybridAffineKeyTailSidecar = currentKey
            polarWHTValueTailCode = currentValue
            polarWHTDecodedValueBuffer = nil
            return true
        } catch {
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            polarWHTDecodedValueBuffer = nil
            return false
        }
    }

    private func placeholderCode(for array: MLXArray, role: TurboQuantTensorRole) throws
        -> TurboQuantAttentionCode
    {
        let layout = try MLX.turboQuantAttentionLayout(
            for: array,
            preset: preset,
            role: role,
            groupSize: groupSize,
            valueBits: valueBits
        )
        return try MLX.turboQuantEmptyAttentionCode(
            layout: layout,
            preset: preset,
            role: role,
            groupSize: groupSize,
            seed: role == .value ? seed ^ turboQuantValueSeedSalt : seed,
            valueBits: valueBits
        )
    }

    private func ensureCompressedCapacity(
        keys: MLXArray,
        values: MLXArray,
        requiredLength: Int
    ) throws {
        if compressedKeys == nil || compressedValues == nil {
            let capacity = ((compressedStep + requiredLength - 1) / compressedStep) * compressedStep
            if offset > 0 {
                try backfillCompressedStorage(capacity: capacity)
                return
            }
            let keyLayout = try MLX.turboQuantAttentionLayout(
                for: keys,
                preset: preset,
                groupSize: groupSize,
                capacity: capacity,
                logicalLength: offset
            )
            let valueLayout = try MLX.turboQuantAttentionLayout(
                for: values,
                preset: preset,
                role: .value,
                groupSize: groupSize,
                valueBits: valueBits,
                capacity: capacity,
                logicalLength: offset
            )
            compressedKeys = try MLX.turboQuantEmptyAttentionCode(
                layout: keyLayout,
                preset: preset,
                role: .key,
                groupSize: groupSize,
                seed: seed
            )
            compressedValues =
                usesPolarWHTValueOnlyStorage
                ? turboQuantCompactValuePlaceholderCode(
                    layout: valueLayout,
                    preset: preset,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: valueBits
                )
                : try MLX.turboQuantEmptyAttentionCode(
                    layout: valueLayout,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: valueBits
                )
            return
        }

        guard let currentKeys = compressedKeys, let currentValues = compressedValues,
            requiredLength > currentKeys.layout.capacity
        else {
            return
        }

        let newCapacity = ((compressedStep + requiredLength - 1) / compressedStep) * compressedStep
        compressedKeys = try expandCompressedCode(currentKeys, newCapacity: newCapacity)
        compressedValues = try expandCompressedCode(currentValues, newCapacity: newCapacity)
        refreshKeyPageSummary()
    }

    private func backfillCompressedStorage(capacity: Int) throws {
        guard let (packedKeys, packedValues) = super.getQuantizedState() else {
            throw TurboQuantCacheError.compressedBackfillUnavailable(
                "no packed cache state exists for \(offset) previous tokens"
            )
        }

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed ^ turboQuantValueSeedSalt,
            valueBits: valueBits
        )
        let decodedKeys = turboDequantized(
            packedKeys,
            configuration: keyConfiguration,
            dtype: .float32
        )
        let decodedValues = turboDequantized(
            packedValues,
            configuration: valueConfiguration,
            dtype: .float32
        )
        compressedKeys = try MLX.turboQuantMetalEncodeAttention(
            decodedKeys,
            configuration: keyConfiguration,
            capacity: capacity,
            logicalLength: offset
        )
        if usesPolarWHTValueOnlyStorage {
            let valueLayout = try MLX.turboQuantAttentionLayout(
                for: decodedValues,
                preset: preset,
                role: .value,
                groupSize: groupSize,
                valueBits: valueBits,
                capacity: capacity,
                logicalLength: offset
            )
            compressedValues = turboQuantCompactValuePlaceholderCode(
                layout: valueLayout,
                preset: preset,
                groupSize: groupSize,
                seed: seed ^ turboQuantValueSeedSalt,
                valueBits: valueBits
            )
        } else {
            compressedValues = try MLX.turboQuantMetalEncodeAttention(
                decodedValues,
                configuration: valueConfiguration,
                capacity: capacity,
                logicalLength: offset
            )
        }
    }

    private func expandCompressedCode(
        _ code: TurboQuantAttentionCode,
        newCapacity: Int
    ) throws -> TurboQuantAttentionCode {
        var newLayout = code.layout
        let extra = newCapacity - code.layout.capacity
        newLayout.capacity = newCapacity
        if turboQuantIsCompactValuePlaceholder(code) {
            return turboQuantCompactValuePlaceholderCode(
                layout: newLayout,
                preset: code.preset,
                groupSize: code.groupSize,
                seed: code.seed,
                valueBits: code.valueBits
            )
        }
        let extraLayout = TurboQuantAttentionLayout(
            batchSize: code.layout.batchSize,
            kvHeadCount: code.layout.kvHeadCount,
            capacity: extra,
            logicalLength: 0,
            headDimension: code.layout.headDimension,
            groupsPerVector: code.layout.groupsPerVector,
            magnitudeWordsPerGroup: code.layout.magnitudeWordsPerGroup,
            bitsetWordsPerGroup: code.layout.bitsetWordsPerGroup
        )
        let zeros = try MLX.turboQuantEmptyAttentionCode(
            layout: extraLayout,
            preset: code.preset,
            role: code.role,
            groupSize: code.groupSize,
            seed: code.seed,
            valueBits: code.valueBits
        )

        func expandedBitsetPlane(_ current: MLXArray, _ padding: MLXArray) -> MLXArray {
            current.shape == [1] ? current : concatenated([current, padding], axis: 2)
        }

        let expandedSigns =
            code.role == .value
            ? code.signs
            : expandedBitsetPlane(code.signs, zeros.signs)
        let expanded = TurboQuantAttentionCode(
            layout: newLayout,
            preset: code.preset,
            role: code.role,
            groupSize: code.groupSize,
            seed: code.seed,
            valueBits: code.valueBits,
            scalesPerGroup: code.scalesPerGroup,
            packedMagnitudes: concatenated(
                [code.packedMagnitudes, zeros.packedMagnitudes], axis: 2),
            signs: expandedSigns,
            highPrecisionMask: expandedBitsetPlane(code.highPrecisionMask, zeros.highPrecisionMask),
            residualSigns: expandedBitsetPlane(code.residualSigns, zeros.residualSigns),
            scales: concatenated([code.scales, zeros.scales], axis: 2)
        )
        try validateExpandedCompressedCode(expanded)
        return expanded
    }

    private func validateExpandedCompressedCode(_ code: TurboQuantAttentionCode) throws {
        let capacity = code.layout.capacity
        func matchesBitsetCapacity(_ array: MLXArray) -> Bool {
            array.shape == [1] || (array.ndim == 5 && array.dim(2) == capacity)
        }
        let signsValid =
            code.role == .key
            ? (code.signs.ndim == 5 && code.signs.dim(2) == capacity)
            : matchesBitsetCapacity(code.signs)
        let highMaskValid =
            code.role == .key
            ? matchesBitsetCapacity(code.highPrecisionMask)
            : matchesBitsetCapacity(code.highPrecisionMask)
        guard code.packedMagnitudes.ndim == 5, code.packedMagnitudes.dim(2) == capacity,
            signsValid,
            highMaskValid,
            matchesBitsetCapacity(code.residualSigns),
            code.scales.ndim == 5, code.scales.dim(2) == capacity
        else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "expanded \(code.role.rawValue) compressed storage does not match capacity \(capacity)"
            )
        }
    }
}

public final class RotatingTurboQuantKVCache: BaseKVCache, QuantizedKVCacheProtocol,
    TurboQuantCompressedKVCacheProtocol,
    CustomDebugStringConvertible
{
    private var rawFallbackCache: RotatingKVCache?
    private var packedFallbackCache: RotatingQuantizedKVCache?
    fileprivate var packedKeys: TurboQuantPackedTensor?
    fileprivate var packedValues: TurboQuantPackedTensor?
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var polarWHTKeyCode: TurboQuantPolarWHTAttentionValueCode?
    private var polarWHTValueCode: TurboQuantPolarWHTAttentionValueCode?
    private var polarWHTValueTailCode: TurboQuantPolarWHTAttentionValueCode?
    private var polarWHTDecodedValueBuffer: MLXArray?
    fileprivate var hybridAffineKeyTailSidecar: TurboQuantAffineKeySidecar?
    public private(set) var keyPageSummary: MLXArray?
    public private(set) var keyCandidateSketch: MLXArray?
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?
    private var restoredLayoutMetadata: RestoredAttentionLayoutMetadata?
    private var lastDecodedTransientBytes: Int = 0
    private var lastNativeAttentionDiagnostics: TurboQuantNativeAttentionDiagnostics?
    private var keyPageSummaryUnavailableReason: String?
    private var keyCandidateSketchUnavailableReason: String?
    private let residentBudgetBytes: Int?
    private let keep: Int
    private let step: Int
    private let maxCacheSize: Int
    private var writeIndex: Int
    public private(set) var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public private(set) var fallbackResults: [TurboQuantFallbackResult] = []

    public let preset: TurboQuantPreset
    public let kvCodec: TurboQuantKVCodec
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let fallbackPolicy: TurboQuantFallbackPolicy
    public var groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let seed: UInt64
    public let valueBits: Int
    public let precisionPolicy: TurboQuantKVPrecisionPolicy
    public let requestedRuntimeMode: TurboQuantRuntimeMode
    public let resolvedRuntimeMode: TurboQuantRuntimeMode
    public let sparseValuePolicy: TurboQuantSparseValuePolicy
    public let sparseValueSelection: TurboQuantSparseValueSelection
    public let layerIndex: Int?
    public let boundaryProtectedLayerCount: Int
    public let boundaryProtectionReason: String?

    private var shouldMaintainPolarWHTKeySidecar: Bool {
        kvCodec == .polarWHT && !precisionPolicy.key.isHighPrecision
    }

    private var shouldMaintainPolarWHTValueSidecar: Bool {
        kvCodec == .polarWHT
    }

    private var shouldMaintainPolarWHTDecodedValueBuffer: Bool {
        guard usesPolarWHTValueOnlyStorage else { return false }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_DISABLE_POLARWHT_DECODED_VALUE_BUFFER") {
            return false
        }
        if TurboQuantRuntimeControl.enabled("TURBOQUANT_ENABLE_POLARWHT_DECODED_VALUE_BUFFER") {
            return true
        }
        return resolvedRuntimeMode == .throughputTurboQuant
    }

    private var shouldMaintainHybridAffineKeySidecar: Bool {
        usesPolarWHTValueOnlyStorage
    }

    fileprivate var usesPolarWHTValueOnlyStorage: Bool {
        turboQuantUsesPolarWHTValueOnlyStorage(kvCodec: kvCodec, precisionPolicy: precisionPolicy)
    }

    private var polarWHTDecodeAttentionPath: TurboQuantAttentionPath {
        precisionPolicy.key.isHighPrecision
            ? .metalHybridK8PolarWHTValue : .metalPolarWHTHybrid
    }

    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { offset < maxCacheSize }

    private func pinnedPrefixLength(forLogicalLength logicalLength: Int) -> Int {
        min(keep, maxCacheSize, max(0, logicalLength))
    }

    private var shouldMaintainExactRawShadow: Bool {
        activeBackend == .metalPolarQJL && optimizationPolicy == .conservative
    }

    private var exactRawShadowMaxSize: Int {
        min(maxCacheSize, max(keep + 1, 512))
    }

    private func mergedRotatingHybridAffineKeySidecar(
        encoded: TurboQuantPackedTensor,
        previousOffset: Int
    ) -> TurboQuantPackedTensor? {
        guard let current = packedKeys, previousOffset > 0 else {
            if encoded.weight.dim(2) > maxCacheSize {
                let start = encoded.weight.dim(2) - maxCacheSize
                return turboQuantSlicePackedTensor(
                    encoded,
                    tokens: start ..< encoded.weight.dim(2)
                )
            }
            return encoded
        }

        let currentLength = current.weight.dim(2)
        let appendedLength = encoded.weight.dim(2)
        let combined: TurboQuantPackedTensor?
        if previousOffset < maxCacheSize {
            combined = turboQuantConcatPackedTensors([current, encoded])
        } else {
            let retainedPrefix = min(keep, currentLength)
            let droppedTailStart = min(currentLength, retainedPrefix + appendedLength)
            var parts: [TurboQuantPackedTensor] = []
            if retainedPrefix > 0 {
                parts.append(turboQuantSlicePackedTensor(current, tokens: 0 ..< retainedPrefix))
            }
            if droppedTailStart < currentLength {
                parts.append(
                    turboQuantSlicePackedTensor(current, tokens: droppedTailStart ..< currentLength)
                )
            }
            parts.append(encoded)
            combined = turboQuantConcatPackedTensors(parts)
        }

        if let combined, combined.weight.dim(2) > maxCacheSize {
            let start = combined.weight.dim(2) - maxCacheSize
            return turboQuantSlicePackedTensor(
                combined,
                tokens: start ..< combined.weight.dim(2)
            )
        }
        return combined
    }

    private func updateHybridAffineKeySidecar(keys: MLXArray, previousOffset: Int) {
        guard shouldMaintainHybridAffineKeySidecar else {
            if packedValues == nil {
                packedKeys = nil
            }
            return
        }
        guard supportsMLXAffineQuantization(dimension: keys.dim(3), groupSize: groupSize) else {
            packedKeys = nil
            lastUnsupportedShape =
                "hybrid affine K sidecar requires key dimension \(keys.dim(3)) to be divisible by group size \(groupSize)"
            return
        }
        let configuration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: .metalPolarQJL,
            seed: seed
        )
        let encoded = turboQuantized(keys, configuration: configuration)
        packedKeys = mergedRotatingHybridAffineKeySidecar(
            encoded: encoded,
            previousOffset: previousOffset
        )
        packedValues = nil
    }

    public init(
        maxSize: Int,
        keep: Int = 4,
        step: Int = 256,
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        kvCodec: TurboQuantKVCodec? = nil,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        requestedRuntimeMode: TurboQuantRuntimeMode = .auto,
        resolvedRuntimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueSelection: TurboQuantSparseValueSelection = .off,
        layerIndex: Int? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        residentBudgetBytes: Int? = nil
    ) {
        let resolvedKVCodec = turboQuantCompressedKVCodec(
            requested: kvCodec,
            backend: backend
        )
        let resolvedValueBits = turboQuantDefaultValueBits(
            preset: preset,
            kvCodec: resolvedKVCodec,
            requestedValueBits: valueBits
        )
        let resolvedPrecisionPolicy =
            precisionPolicy ?? TurboQuantKVPrecisionPolicy.legacy(
                preset: preset,
                valueBits: resolvedValueBits
            )
        self.keep = max(0, min(keep, maxSize))
        self.step = step
        self.maxCacheSize = maxSize
        self.writeIndex = self.keep
        self.preset = resolvedPrecisionPolicy.compressedKeyPreset
        self.kvCodec = resolvedKVCodec
        self.requestedBackend = backend
        self.optimizationPolicy = optimizationPolicy
        self.fallbackPolicy = fallbackPolicy
        self.seed = seed
        self.valueBits =
            resolvedPrecisionPolicy.resolvedValueBits
            ?? resolvedValueBits
        self.precisionPolicy = resolvedPrecisionPolicy
        self.requestedRuntimeMode = requestedRuntimeMode
        self.resolvedRuntimeMode = resolvedRuntimeMode
        self.sparseValuePolicy = sparseValuePolicy
        self.sparseValueSelection = sparseValueSelection
        self.layerIndex = layerIndex
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.residentBudgetBytes = residentBudgetBytes
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        self.groupSize = groupSize
        self.bits = resolvedPrecisionPolicy.compressedKeyPreset.effectiveBits
        self.mode = mode
        super.init()
        if !turboQuantIsMetalCompressedBackend(self.activeBackend) {
            self.rawFallbackCache = RotatingKVCache(maxSize: maxSize, keep: self.keep, step: step)
        }
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let previousOffset = offset
        let rawCache = materializedRawFallbackCache()
        let (cachedKeys, cachedValues) = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset
        writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
        updateRotatingPolarWHTKeySidecar(keys: keys, previousOffset: previousOffset)
        updateRotatingPolarWHTValueSidecar(values: values, previousOffset: previousOffset)
        packedFallbackCache = nil
        if mode == .affine, packedKeys == nil, packedValues == nil,
            compressedKeys == nil, compressedValues == nil
        {
            groupSize = compatibleAffineGroupSize(
                configuredGroupSize: groupSize,
                keyDimension: cachedKeys.dim(3),
                valueDimension: cachedValues.dim(3)
            )
        }
        guard supportsMLXAffineKVQuantization(
            keyDimension: cachedKeys.dim(3),
            valueDimension: cachedValues.dim(3),
            keyGroupSize: groupSize,
            valueGroupSize: groupSize
        ) else {
            packedKeys = nil
            packedValues = nil
            return (
                placeholderQuantizedTuple(for: cachedKeys, bits: bits),
                placeholderQuantizedTuple(for: cachedValues, bits: bits)
            )
        }

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed ^ turboQuantValueSeedSalt,
            valueBits: valueBits
        )
        packedKeys = turboQuantized(cachedKeys, configuration: keyConfiguration)
        packedValues = turboQuantized(cachedValues, configuration: valueConfiguration)

        return (
            (packedKeys!.weight, packedKeys!.scales, packedKeys!.biases),
            (packedValues!.weight, packedValues!.scales, packedValues!.biases)
        )
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        if let packedKeys, let packedValues {
            return (
                (packedKeys.weight, packedKeys.scales, packedKeys.biases),
                (packedValues.weight, packedValues.scales, packedValues.biases)
            )
        }
        if let packedState = packedFallbackCache?.getQuantizedState() {
            return packedState
        }
        guard compressedKeys != nil, compressedValues != nil else {
            return nil
        }
        guard let compressedKeys, let compressedValues,
            turboQuantSupportsPackedFallback(
                keyCode: compressedKeys,
                valueCode: compressedValues,
                groupSize: groupSize
            )
        else {
            return nil
        }
        return materializedPackedFallbackCache().getQuantizedState()
    }

    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        guard let compressedKeys, let compressedValues else { return nil }
        return (compressedKeys, compressedValues)
    }

    public var hybridAffineKeyState: QuantizedKVStorage? {
        turboQuantPackedStorage(packedKeys)
    }

    public var hybridAffineKeyTailStateForAttention: QuantizedKVStorage? {
        turboQuantPackedStorage(hybridAffineKeyTailSidecar)
    }

    public var polarWHTKeyState: TurboQuantPolarWHTAttentionValueCode? {
        copiedTurboQuantPolarWHTCode(polarWHTKeyCode)
    }

    public var polarWHTValueState: TurboQuantPolarWHTAttentionValueCode? {
        copiedTurboQuantPolarWHTValueCode(polarWHTValueCode)
    }

    public var polarWHTKeyStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTKeyCode
    }

    public var polarWHTValueStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTValueCode
    }

    public var polarWHTValueTailStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        polarWHTValueTailCode
    }

    public var polarWHTDecodedValueState: MLXArray? {
        guard let polarWHTDecodedValueBuffer else { return nil }
        guard (polarWHTValueCode?.layout.logicalLength ?? min(offset, maxCacheSize)) > 0 else {
            return nil
        }
        return polarWHTDecodedValueBuffer
    }

    public var polarWHTDecodedValueLayout: MLX.TurboQuantAttentionLayout? {
        polarWHTValueCode?.layout
    }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let compressedBytes: Int
        let logicalLength: Int
        let capacity: Int
        if let compressedKeys, let compressedValues {
            compressedBytes =
                turboQuantCodeBytes(compressedKeys) + turboQuantCodeBytes(compressedValues)
                + turboQuantArrayBytes([keyPageSummary, keyCandidateSketch].compactMap { $0 })
                + polarWHTKeyBytes
                + polarWHTValueBytes
            logicalLength = compressedKeys.layout.logicalLength
            capacity = compressedKeys.layout.capacity
        } else {
            compressedBytes = polarWHTKeyBytes + polarWHTValueBytes
            logicalLength = min(offset, maxCacheSize)
            capacity = maxCacheSize
        }
        let packedBytes =
            turboQuantArrayBytes(packedFallbackCache?.state ?? [])
            + turboQuantArrayBytes(
                [packedKeys?.weight, packedKeys?.scales, packedKeys?.biases].compactMap { $0 })
            + turboQuantArrayBytes(
                [packedValues?.weight, packedValues?.scales, packedValues?.biases].compactMap { $0 }
            )
            + turboQuantAffineKeySidecarBytes(hybridAffineKeyTailSidecar)
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: logicalLength,
            capacity: capacity,
            compressedBytes: compressedBytes,
            packedFallbackBytes: packedBytes,
            rawShadowBytes: turboQuantArrayBytes(rawFallbackCache?.state ?? [])
                + polarWHTDecodedValueResidentBytes,
            decodedTransientBytes: lastDecodedTransientBytes,
            lifecycle: cacheLifecycle
        )
    }

    public func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot {
        let keyBytes: Int
        let valueBytes: Int
        let logicalLength: Int
        let capacity: Int
        let pinnedPrefixLength: Int
        let ringOffset: Int

        if let compressedKeys, let compressedValues {
            keyBytes = turboQuantCodeBytes(compressedKeys)
            valueBytes = turboQuantCodeBytes(compressedValues)
            logicalLength = compressedKeys.layout.logicalLength
            capacity = compressedKeys.layout.capacity
            pinnedPrefixLength = compressedKeys.layout.pinnedPrefixLength
            ringOffset = compressedKeys.layout.ringOffset
        } else {
            let rawBytes = turboQuantKeyValueBytes(rawFallbackCache?.state ?? [])
            let packedBytes = turboQuantKeyValueBytes(packedFallbackCache?.state ?? [])
            keyBytes = rawBytes.keyBytes + packedBytes.keyBytes
            valueBytes = rawBytes.valueBytes + packedBytes.valueBytes
            logicalLength = min(offset, maxCacheSize)
            capacity = maxCacheSize
            pinnedPrefixLength = self.pinnedPrefixLength(forLogicalLength: logicalLength)
            ringOffset = self.ringOffset(forOffset: offset)
        }

        return TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription: cacheLifecycle.turboQuantRuntimeDescription,
            logicalLength: logicalLength,
            capacity: capacity,
            pinnedPrefixLength: pinnedPrefixLength,
            ringOffset: ringOffset,
            keyBytes: keyBytes,
            valueBytes: valueBytes,
            rawShadowAllocated: turboQuantArrayBytes(rawFallbackCache?.state ?? []) > 0,
            packedFallbackAllocated: turboQuantArrayBytes(packedFallbackCache?.state ?? []) > 0
                || packedKeys != nil
                || packedValues != nil,
            lastAttentionPath: lastAttentionPath.rawValue,
            lastFailure: cacheLifecycle.turboQuantRuntimeFailureReason ?? lastUnsupportedShape,
            kvCodec: kvCodec,
            quantizationMode: mode.rawValue,
            keyBits: bits,
            groupSize: groupSize,
            valueBits: valueBits,
            selectedPath: lastAttentionPath.rawValue,
            fallbackReason: cacheLifecycle.turboQuantRuntimeFailureReason ?? lastUnsupportedShape,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            boundaryPolicy: precisionPolicy.boundary,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            runtimeFallbackReason: backendFallbackReason,
            decodedActiveKeyBytes: turboQuantKeyValueBytes(rawFallbackCache?.state ?? []).keyBytes,
            decodedActiveValueBytes: turboQuantKeyValueBytes(rawFallbackCache?.state ?? []).valueBytes
                + polarWHTDecodedValueActiveBytes,
            activeCacheAllocated: turboQuantArrayBytes(rawFallbackCache?.state ?? []) > 0
                || polarWHTDecodedValueResidentBytes > 0,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated
        )
    }

    public func exportSnapshot(
        identity: TurboQuantKVSnapshotIdentity,
        conversationID: UUID,
        snapshotID: UUID = UUID(),
        encryptionKeyID: String = "lm-local-unencrypted",
        createdAt: Date = Date()
    ) throws -> TurboQuantKVSnapshotPayload {
        guard case .compressedCommitted = cacheLifecycle else {
            throw TurboQuantRuntimeFailure.cacheLifecycleInvalid(
                "TurboQuant rotating snapshot export requires committed compressed state; current lifecycle is \(cacheLifecycle)"
            )
        }
        guard turboQuantIsMetalCompressedBackend(activeBackend) else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant rotating snapshot export requires active Metal compressed backend"
            )
        }
        try validateCompressedState(context: "rotating snapshot export")
        guard let compressedKeys, let compressedValues else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant rotating snapshot export missing compressed key/value state"
            )
        }
        let arrays = try turboQuantSnapshotArrays(
            from: state,
            polarWHTKeyCode: polarWHTKeyCode,
            polarWHTValueCode: polarWHTValueCode
        )
        let descriptors = turboQuantSnapshotArrayDescriptors(arrays)
        let manifest = TurboQuantKVSnapshotManifest(
            snapshotID: snapshotID,
            conversationID: conversationID,
            identity: identity,
            turboQuantLayoutVersion: compressedKeys.layout.layoutVersion,
            logicalLength: compressedKeys.layout.logicalLength,
            pinnedPrefixLength: compressedKeys.layout.pinnedPrefixLength,
            compressedKeyBytes: Int64(turboQuantCodeBytes(compressedKeys)),
            compressedValueBytes: Int64(turboQuantCodeBytes(compressedValues)),
            blobByteCount: Int64(turboQuantArrayBytes(Array(arrays.values))),
            encryptionKeyID: encryptionKeyID,
            createdAt: createdAt,
            cacheKind: "RotatingTurboQuantKVCache",
            kvCodec: kvCodec,
            preset: preset.rawValue,
            requestedBackend: requestedBackend.rawValue,
            activeBackend: activeBackend.rawValue,
            quantizationMode: mode.rawValue,
            keyBits: bits,
            groupSize: groupSize,
            valueBits: valueBits,
            seed: seed,
            mode: mode.rawValue,
            capacity: compressedKeys.layout.capacity,
            ringOffset: compressedKeys.layout.ringOffset,
            batchSize: compressedKeys.layout.batchSize,
            kvHeadCount: compressedKeys.layout.kvHeadCount,
            keyHeadDimension: compressedKeys.layout.headDimension,
            valueHeadDimension: compressedValues.layout.headDimension,
            rotatingKeep: keep,
            rotatingStep: step,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            boundaryPolicy: precisionPolicy.boundary,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            runtimeFallbackReason: backendFallbackReason,
            selectedPath: lastAttentionPath.rawValue,
            fallbackReason: backendFallbackReason,
            polarWHTKeyBytes: Int64(polarWHTKeyBytes),
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTKeyBits: polarWHTKeyCode?.bits,
            polarWHTKeySeed: polarWHTKeyCode?.seed,
            polarWHTKeyPackedWordsPerVector: polarWHTKeyCode?.packedWordsPerVector,
            polarWHTValueBytes: Int64(polarWHTValueBytes),
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated,
            polarWHTValueBits: polarWHTValueCode?.bits,
            polarWHTValueSeed: polarWHTValueCode?.seed,
            polarWHTValuePackedWordsPerVector: polarWHTValueCode?.packedWordsPerVector,
            arrays: descriptors
        )
        return TurboQuantKVSnapshotPayload(manifest: manifest, compressedArrays: arrays)
    }

    public func importSnapshot(
        _ payload: TurboQuantKVSnapshotPayload,
        expectedIdentity: TurboQuantKVSnapshotIdentity
    ) throws {
        guard turboQuantIsMetalCompressedBackend(activeBackend) else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant rotating snapshot import requires active Metal compressed backend"
            )
        }
        let manifest = payload.manifest
        let ordered = try turboQuantValidateSnapshotManifest(
            manifest,
            expectedIdentity: expectedIdentity,
            expectedCacheKind: "RotatingTurboQuantKVCache",
            expectedPreset: preset,
            expectedRequestedBackend: requestedBackend,
            expectedActiveBackend: activeBackend,
            expectedKVCodec: kvCodec,
            expectedGroupSize: groupSize,
            expectedValueBits: valueBits,
            expectedSeed: seed,
            expectedMode: mode,
            arrays: payload.compressedArrays
        )
        guard manifest.capacity == maxCacheSize else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant rotating snapshot capacity \(manifest.capacity) does not match cache maxSize \(maxCacheSize)"
            )
        }
        guard pinnedPrefixLength(forLogicalLength: manifest.logicalLength)
            == manifest.pinnedPrefixLength
        else {
            throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                "TurboQuant rotating snapshot pinned prefix \(manifest.pinnedPrefixLength) does not match cache keep \(keep)"
            )
        }
        let imported = try turboQuantSnapshotImportedCodes(
            manifest: manifest,
            ordered: ordered,
            preset: preset,
            groupSize: groupSize,
            seed: seed,
            valueBits: valueBits
        )
        let importedPolarWHTKeyCode = try turboQuantSnapshotPolarWHTKeyCode(
            manifest: manifest,
            arrays: payload.compressedArrays
        )
        let importedPolarWHTValueCode = try turboQuantSnapshotPolarWHTValueCode(
            manifest: manifest,
            arrays: payload.compressedArrays
        )
        let residentBytes = turboQuantCodeBytes(imported.keys) + turboQuantCodeBytes(imported.values)
            + (importedPolarWHTKeyCode?.residentPayloadByteCount ?? 0)
            + (importedPolarWHTValueCode?.residentPayloadByteCount ?? 0)
        if let residentBudgetBytes, residentBytes > residentBudgetBytes {
            throw TurboQuantRuntimeFailure.fallbackBudgetExceeded(
                "TurboQuant rotating snapshot resident bytes \(residentBytes) exceed admitted budget \(residentBudgetBytes)"
            )
        }
        restoredLayoutMetadata = RestoredAttentionLayoutMetadata(
            capacity: manifest.capacity,
            logicalLength: manifest.logicalLength,
            ringOffset: manifest.ringOffset,
            pinnedPrefixLength: manifest.pinnedPrefixLength,
            headDimension: manifest.keyHeadDimension,
            valueHeadDimension: manifest.valueHeadDimension,
            kvHeadCount: manifest.kvHeadCount
        )
        let ringCapacity = manifest.capacity - manifest.pinnedPrefixLength
        offset =
            manifest.logicalLength == manifest.capacity && ringCapacity > 0
            ? manifest.capacity + manifest.ringOffset
            : manifest.logicalLength
        writeIndex = nextWriteIndex(afterOffset: offset)
        rawFallbackCache = nil
        packedFallbackCache = nil
        packedKeys = nil
        packedValues = nil
        compressedKeys = imported.keys
        compressedValues = imported.values
        polarWHTKeyCode = importedPolarWHTKeyCode
        polarWHTValueCode = importedPolarWHTValueCode
        hybridAffineKeyTailSidecar = nil
        polarWHTValueTailCode = nil
        polarWHTDecodedValueBuffer = nil
        refreshKeyPageSummary()
        lastDecodedTransientBytes = 0
        lastUnsupportedShape = nil
        cacheLifecycle = .compressedCommitted(
            logicalLength: manifest.logicalLength,
            capacity: manifest.capacity
        )
    }

    public func recordFallback(_ result: TurboQuantFallbackResult) {
        fallbackResults.append(result)
        lastUnsupportedShape = result.reason
        if result.policy != .exactRequired {
            lastNativeAttentionDiagnostics = nil
        }
        if let toPath = result.toPath {
            lastAttentionPath = toPath
        }
        switch result.policy {
        case .packedAllowed:
            cacheLifecycle = .degradedPackedFallback(reason: result.reason)
        case .compressedDecodeAllowed:
            cacheLifecycle = .degradedDecodedFallback(reason: result.reason)
        case .fatalOnFailure:
            cacheLifecycle = .failed(reason: result.reason)
        case .exactRequired:
            break
        }
    }

    public func recordNativeAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        selection: TurboQuantSparseValueSelection
    ) {
        lastAttentionPath = .nativeMLXCompressed
        lastUnsupportedShape = nil
        lastNativeAttentionDiagnostics = diagnostics
        cacheLifecycle = .compressedCommitted(
            logicalLength: compressedKeys?.layout.logicalLength ?? min(offset, maxCacheSize),
            capacity: compressedKeys?.layout.capacity ?? maxCacheSize
        )
    }

    public func recordPolarWHTAttentionDiagnostics(
        _ diagnostics: TurboQuantNativeAttentionDiagnostics?,
        path: TurboQuantAttentionPath
    ) {
        lastAttentionPath = path
        lastUnsupportedShape = nil
        lastNativeAttentionDiagnostics = diagnostics
        cacheLifecycle = .compressedCommitted(
            logicalLength: compressedKeys?.layout.logicalLength ?? min(offset, maxCacheSize),
            capacity: compressedKeys?.layout.capacity ?? maxCacheSize
        )
    }

    private func refreshKeyPageSummary() {
        guard activeBackend == .metalPolarQJL else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason =
                "key page summaries require metalPolarQJL backend; active backend is \(activeBackend.rawValue)"
            return
        }
        guard let compressedKeys else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "no compressed key state is available"
            return
        }
        guard compressedKeys.layout.ringOffset == 0 else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason =
                "ring offset \(compressedKeys.layout.ringOffset) makes page summaries unsafe"
            return
        }
        guard compressedKeys.layout.pinnedPrefixLength == 0 else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason =
                "pinned prefix length \(compressedKeys.layout.pinnedPrefixLength) makes page summaries unsafe"
            return
        }
        guard compressedKeys.layout.logicalLength > 0 else {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "compressed key state is empty"
            return
        }
        do {
            keyPageSummary = try MLX.turboQuantKeyPageSummaries(keyCode: compressedKeys)
            keyPageSummaryUnavailableReason = nil
        } catch {
            keyPageSummary = nil
            keyPageSummaryUnavailableReason = "key page summary build failed: \(error)"
        }
    }

    public func ensureKeyPageSummary() {
        if keyPageSummary == nil {
            refreshKeyPageSummary()
        }
    }

    private func invalidateKeyCandidateSketch(reason: String? = nil) {
        keyCandidateSketch = nil
        keyCandidateSketchUnavailableReason =
            reason
            ?? turboQuantKeyCandidateSketchUnavailableReason(
                activeBackend: activeBackend,
                compressedKeys: compressedKeys,
                artifactName: "key candidate sketches"
            )
            ?? "key candidate sketch has not been built"
    }

    private func refreshKeyCandidateSketch() {
        if let reason = turboQuantKeyCandidateSketchUnavailableReason(
            activeBackend: activeBackend,
            compressedKeys: compressedKeys,
            artifactName: "key candidate sketches"
        ) {
            invalidateKeyCandidateSketch(reason: reason)
            return
        }
        guard let compressedKeys else {
            invalidateKeyCandidateSketch(reason: "no compressed key state is available")
            return
        }
        do {
            keyCandidateSketch = try turboQuantBuildKeyCandidateSketch(keyCode: compressedKeys)
            keyCandidateSketchUnavailableReason = nil
        } catch {
            invalidateKeyCandidateSketch(reason: "key candidate sketch build failed: \(error)")
        }
    }

    public func ensureKeyCandidateSketch() {
        if keyCandidateSketch == nil {
            refreshKeyCandidateSketch()
        }
    }

    private func updateKeyCandidateSketchAfterLinearAppend(
        encodedKeys: TurboQuantAttentionCode,
        previousOffset: Int,
        tokenCount: Int
    ) {
        guard keyCandidateSketch != nil else {
            if sparseValueSelection.mode == .candidateSparse {
                refreshKeyCandidateSketch()
            } else {
                keyCandidateSketchUnavailableReason =
                    turboQuantKeyCandidateSketchUnavailableReason(
                        activeBackend: activeBackend,
                        compressedKeys: compressedKeys,
                        artifactName: "key candidate sketches"
                    )
                    ?? "key candidate sketch has not been built"
            }
            return
        }
        guard activeBackend == .metalPolarQJL,
            let compressedKeys,
            compressedKeys.layout.ringOffset == 0,
            compressedKeys.layout.logicalLength > 0
        else {
            invalidateKeyCandidateSketch()
            return
        }
        guard tokenCount == 1, encodedKeys.layout.logicalLength == 1 else {
            refreshKeyCandidateSketch()
            return
        }
        let pageIndex = previousOffset / MLX.turboQuantKeyPageSummaryPageSize
        guard let existingSketch = keyCandidateSketch else { return }
        do {
            keyCandidateSketch = try turboQuantUpdateKeyCandidateSketchPage(
                existingSketch: existingSketch,
                encodedKeys: encodedKeys,
                pageIndex: pageIndex
            )
            keyCandidateSketchUnavailableReason = nil
        } catch {
            invalidateKeyCandidateSketch(reason: "key candidate sketch update failed: \(error)")
        }
    }

    public func validateCompressedState(context: String) throws {
        guard let compressedKeys, let compressedValues else {
            if offset == 0 { return }
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): no compressed rotating state exists for offset \(offset)"
            )
        }
        try validateTurboQuantPair(keys: compressedKeys, values: compressedValues, context: context)
        try enforceResidentBudget(context: context)
    }

    public func makeCompressedUpdateCheckpoint(appendingTokenCount tokenCount: Int)
        -> TurboQuantCompressedUpdateCheckpoint
    {
        guard let compressedKeys, let compressedValues else {
            return TurboQuantCompressedUpdateCheckpoint(
                payload: .fullState(
                    offset: offset,
                    metaState: metaState,
                    state: state.map { $0[.ellipsis] },
                    hybridAffineKeyState: copiedTurboQuantPackedTensor(packedKeys).map {
                        TurboQuantAffineKeySidecar(
                            packed: $0,
                            logicalLength: $0.weight.dim(2)
                        )
                    },
                    polarWHTKeyCode: copiedTurboQuantPolarWHTCode(polarWHTKeyCode),
                    polarWHTValueCode: copiedTurboQuantPolarWHTValueCode(polarWHTValueCode)
                ))
        }

        let canHideAppendedSlots = offset + tokenCount <= maxCacheSize
        if !canHideAppendedSlots {
            return TurboQuantCompressedUpdateCheckpoint(
                payload: .rotatingFullState(
                    offset: offset,
                    writeIndex: writeIndex,
                    metaState: metaState,
                    state: state.map { $0[.ellipsis] },
                    rawFallbackState: rawFallbackCache?.state.map { $0[.ellipsis] },
                    rawFallbackMetaState: rawFallbackCache?.metaState,
                    packedFallbackState: packedFallbackCache?.state.map { $0[.ellipsis] },
                    packedFallbackMetaState: packedFallbackCache?.metaState,
                    packedKeys: packedKeys,
                    packedValues: packedValues,
                    polarWHTKeyCode: copiedTurboQuantPolarWHTCode(polarWHTKeyCode),
                    polarWHTValueCode: copiedTurboQuantPolarWHTValueCode(polarWHTValueCode),
                    fallbackResultCount: fallbackResults.count,
                    lifecycle: cacheLifecycle,
                    lastAttentionPath: lastAttentionPath,
                    lastUnsupportedShape: lastUnsupportedShape,
                    lastDecodedTransientBytes: lastDecodedTransientBytes
                ))
        }

        return TurboQuantCompressedUpdateCheckpoint(
            payload: .rotatingCompressed(
                offset: offset,
                writeIndex: writeIndex,
                compressedKeys: compressedKeys,
                compressedValues: compressedValues,
                rawFallbackState: rawFallbackCache?.state.map { $0[.ellipsis] },
                rawFallbackMetaState: rawFallbackCache?.metaState,
                packedFallbackState: packedFallbackCache?.state.map { $0[.ellipsis] },
                packedFallbackMetaState: packedFallbackCache?.metaState,
                packedKeys: packedKeys,
                packedValues: packedValues,
                polarWHTKeyCode: copiedTurboQuantPolarWHTCode(polarWHTKeyCode),
                polarWHTValueCode: copiedTurboQuantPolarWHTValueCode(polarWHTValueCode),
                fallbackResultCount: fallbackResults.count,
                lifecycle: cacheLifecycle,
                lastAttentionPath: lastAttentionPath,
                lastUnsupportedShape: lastUnsupportedShape,
                lastDecodedTransientBytes: lastDecodedTransientBytes
            ))
    }

    public func restoreCompressedUpdateCheckpoint(
        _ checkpoint: TurboQuantCompressedUpdateCheckpoint
    ) {
        switch checkpoint.payload {
        case .fullState(
            let previousOffset,
            let previousMetaState,
            let previousState,
            let previousHybridAffineKeyState,
            let previousPolarWHTKeyCode,
            let previousPolarWHTValueCode
        ):
            metaState = previousMetaState
            state = previousState
            offset = previousOffset
            writeIndex = nextWriteIndex(afterOffset: previousOffset)
            packedKeys = copiedTurboQuantPackedTensor(previousHybridAffineKeyState?.packed)
            packedValues = nil
            polarWHTKeyCode = copiedTurboQuantPolarWHTCode(previousPolarWHTKeyCode)
            polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(previousPolarWHTValueCode)
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            polarWHTDecodedValueBuffer = nil

        case .rotatingFullState(
            let previousOffset,
            let previousWriteIndex,
            let previousMetaState,
            let previousState,
            let previousRawFallbackState,
            let previousRawFallbackMetaState,
            let previousPackedFallbackState,
            let previousPackedFallbackMetaState,
            let previousPackedKeys,
            let previousPackedValues,
            let previousPolarWHTKeyCode,
            let previousPolarWHTValueCode,
            let previousFallbackResultCount,
            let previousLifecycle,
            let previousAttentionPath,
            let previousUnsupportedShape,
            let previousDecodedTransientBytes
        ):
            metaState = previousMetaState
            state = previousState
            restoreRawFallbackCache(
                state: previousRawFallbackState,
                metaState: previousRawFallbackMetaState
            )
            restorePackedFallbackCache(
                state: previousPackedFallbackState,
                metaState: previousPackedFallbackMetaState
            )
            packedKeys = copiedTurboQuantPackedTensor(previousPackedKeys)
            packedValues = copiedTurboQuantPackedTensor(previousPackedValues)
            polarWHTKeyCode = copiedTurboQuantPolarWHTCode(previousPolarWHTKeyCode)
            polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(previousPolarWHTValueCode)
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            polarWHTDecodedValueBuffer = nil
            offset = previousOffset
            writeIndex = previousWriteIndex
            refreshKeyPageSummary()
            cacheLifecycle = previousLifecycle
            lastAttentionPath = previousAttentionPath
            lastUnsupportedShape = previousUnsupportedShape
            lastDecodedTransientBytes = previousDecodedTransientBytes
            if fallbackResults.count > previousFallbackResultCount {
                fallbackResults.removeLast(fallbackResults.count - previousFallbackResultCount)
            }

        case .rotatingCompressed(
            let previousOffset,
            let previousWriteIndex,
            let previousKeys,
            let previousValues,
            let previousRawFallbackState,
            let previousRawFallbackMetaState,
            let previousPackedFallbackState,
            let previousPackedFallbackMetaState,
            let previousPackedKeys,
            let previousPackedValues,
            let previousPolarWHTKeyCode,
            let previousPolarWHTValueCode,
            let previousFallbackResultCount,
            let previousLifecycle,
            let previousAttentionPath,
            let previousUnsupportedShape,
            let previousDecodedTransientBytes
        ):
            offset = previousOffset
            writeIndex = previousWriteIndex
            compressedKeys = previousKeys
            compressedValues = previousValues
            refreshKeyPageSummary()
            restoreRawFallbackCache(
                state: previousRawFallbackState,
                metaState: previousRawFallbackMetaState
            )
            restorePackedFallbackCache(
                state: previousPackedFallbackState,
                metaState: previousPackedFallbackMetaState
            )
            packedKeys = copiedTurboQuantPackedTensor(previousPackedKeys)
            packedValues = copiedTurboQuantPackedTensor(previousPackedValues)
            polarWHTKeyCode = copiedTurboQuantPolarWHTCode(previousPolarWHTKeyCode)
            polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(previousPolarWHTValueCode)
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            polarWHTDecodedValueBuffer = nil
            cacheLifecycle = previousLifecycle
            lastAttentionPath = previousAttentionPath
            lastUnsupportedShape = previousUnsupportedShape
            lastDecodedTransientBytes = previousDecodedTransientBytes
            if fallbackResults.count > previousFallbackResultCount {
                fallbackResults.removeLast(fallbackResults.count - previousFallbackResultCount)
            }

        case .linearCompressed:
            break
        }
    }

    private func enforceResidentBudget(context: String) throws {
        guard let residentBudgetBytes else { return }
        let residentBytes = cacheFootprint.residentBytes
        guard residentBytes <= residentBudgetBytes else {
            throw TurboQuantCacheError.residentBudgetExceeded(
                residentBytes: residentBytes,
                budgetBytes: residentBudgetBytes
            )
        }
    }

    public func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray) {
        try validateCompressedState(context: "decode rotating compressed state")
        guard let compressedKeys, let compressedValues else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "decode rotating compressed state missing"
            )
        }
        if kvCodec == .polarWHT,
            let polarWHTKeyCode,
            let polarWHTValueCode
        {
            cacheLifecycle = .decodeCompressed
            let decodedKeys =
                TurboQuantKernelAvailability.current.supportsMetalPolarWHTCodec
                ? try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    polarWHTKeyCode,
                    outputDType: outputDType
                )
                : try MLX.turboQuantPolarWHTReferenceDecodeAttentionValues(
                    polarWHTKeyCode
                ).asType(outputDType)
            let decodedValues =
                TurboQuantKernelAvailability.current.supportsMetalPolarWHTCodec
                ? try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    polarWHTValueCode,
                    outputDType: outputDType
                )
                : try MLX.turboQuantPolarWHTReferenceDecodeAttentionValues(
                    polarWHTValueCode
                ).asType(outputDType)
            lastDecodedTransientBytes = decodedKeys.nbytes + decodedValues.nbytes
            return (decodedKeys, decodedValues)
        }
        if turboQuantIsCompactValuePlaceholder(compressedValues) {
            throw TurboQuantRuntimeFailure.compressedAttentionUnavailable(
                "hybrid K8+PolarWHT-V rotating cache has no affine compressed value payload to decode"
            )
        }
        // 4.1: gate the full-context materialization (recoverable instead of a crash under pressure).
        try turboQuantGuardFallbackMaterialization(
            keys: compressedKeys, values: compressedValues, dtype: outputDType)
        cacheLifecycle = .decodeCompressed
        let decodedKeys = try MLX.turboQuantMetalDecodeAttention(
            compressedKeys,
            outputDType: outputDType
        )
        let decodedValues = try MLX.turboQuantMetalDecodeAttention(
            compressedValues,
            outputDType: outputDType
        )
        lastDecodedTransientBytes = decodedKeys.nbytes + decodedValues.nbytes
        return (decodedKeys, decodedValues)
    }

    public func releaseRawShadow() {
        if compressedKeys != nil, compressedValues != nil {
            rawFallbackCache = nil
        }
    }

    public var attentionDiagnostics: TurboQuantAttentionDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        let sparseSelection = sparseValueSelection.resolved(
            runtimeMode: resolvedRuntimeMode,
            contextLength: compressedKeys?.layout.logicalLength ?? min(offset, maxCacheSize),
            policy: sparseValuePolicy
        )
        let sparseVInactiveReason = turboQuantSparseVInactiveReason(
            enabled: sparseSelection.isEnabled,
            kvCodec: kvCodec,
            activeBackend: activeBackend,
            nativeDiagnostics: lastNativeAttentionDiagnostics,
            fallbackReason: backendFallbackReason
        )
        return TurboQuantAttentionDiagnostics(
            layerIndex: layerIndex,
            metalAttentionAvailable: turboQuantMetalAttentionAvailable(
                kvCodec: kvCodec,
                availability: availability
            ),
            activeAttentionPath: lastAttentionPath,
            nativeBackend: availability.attentionCapabilities.nativeCompressedAttention == true
                ? "nativeMLX" : nil,
            nativeBackendVersion: availability.attentionCapabilities.nativeBackendVersion,
            nativeFallbackReason: availability.attentionCapabilities.nativeFallbackReason,
            nativeKernelKind: lastNativeAttentionDiagnostics?.kernelKind,
            nativeSparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: optimizationPolicy,
            fallbackReason: sparseVInactiveReason,
            lastUnsupportedShape: lastUnsupportedShape,
            rawFallbackAllocated: rawFallbackCache != nil,
            cacheLifecycle: cacheLifecycle,
            lastFallback: fallbackResults.last,
            sparseVEnabled: sparseSelection.isEnabled,
            sparseVThreshold: sparseSelection.resolvedThreshold,
            sparseVSelectionMode: sparseSelection.mode,
            sparseVTopK: sparseSelection.topK,
            sparseVCumulativeMass: sparseSelection.cumulativeMass,
            sparseVMaxTopK: sparseSelection.maxTopK,
            sparseVRecentTokenCount: sparseSelection.recentTokens,
            sparseVOlderTokenCount: sparseSelection.mode == .candidateSparse
                ? sparseSelection.topK : nil,
            sparseVPageCandidateCount: sparseSelection.candidatePages,
            sparseVSkippedTokens: lastNativeAttentionDiagnostics?.sparseSkippedTokens,
            sparseVTotalTokens: lastNativeAttentionDiagnostics?.sparseTotalTokens,
            sparseVActive: turboQuantSparseVActive(lastNativeAttentionDiagnostics),
            sparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            keyBits: bits,
            valueBits: valueBits,
            keyGroupSize: groupSize,
            valueGroupSize: groupSize,
            keyPageSummaryAvailable: keyPageSummary != nil,
            keyPageSummaryShape: keyPageSummary?.shape,
            keyPageSummaryUnavailableReason: keyPageSummaryUnavailableReason,
            keyCandidateSketchAvailable: keyCandidateSketch != nil,
            keyCandidateSketchShape: keyCandidateSketch?.shape,
            keyCandidateSketchUnavailableReason: keyCandidateSketchUnavailableReason,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated
        )
    }

    public func supportsCompressedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        let availability = TurboQuantKernelAvailability.current
        let nativeAttentionAvailable =
            availability.attentionCapabilities.nativeCompressedAttention == true
        if kvCodec == .polarWHT {
            let requiresHybridValueKernel = precisionPolicy.key.isHighPrecision
            let polarWHTAttentionAvailable =
                requiresHybridValueKernel
                    ? availability.attentionCapabilities.hybridK8PolarWHTValueAttention
                    : availability.supportsMetalPolarWHTAttention
            guard activeBackend == .metalPolarWHT,
                polarWHTAttentionAvailable
            else {
                lastUnsupportedShape = turboQuantPolarWHTAttentionUnavailableReason(
                    backendFallbackReason: backendFallbackReason,
                    valueBytes: polarWHTValueBytes,
                    payloadAllocated: polarWHTValuePayloadAllocated
                )
                return false
            }
            guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else {
                lastUnsupportedShape = "queries/keys/values must be rank 4"
                return false
            }
            guard queries.dim(2) == 1 else {
                lastUnsupportedShape =
                    "PolarWHT native attention is decode-only; qLen=\(queries.dim(2))"
                return false
            }
            guard queries.dim(0) == keys.dim(0), queries.dim(0) == values.dim(0),
                keys.dim(2) == values.dim(2), keys.dim(1) == values.dim(1)
            else {
                lastUnsupportedShape = "query/key/value batch, token, or KV head counts differ"
                return false
            }
            guard queries.dim(3) == keys.dim(3),
                keys.dim(3) == values.dim(3),
                turboQuantSupportsPolarWHTAttentionDimension(queries.dim(3))
            else {
                lastUnsupportedShape =
                    "PolarWHT requires matching power-of-two head dimensions <= 256; q=\(queries.dim(3)) k=\(keys.dim(3)) v=\(values.dim(3))"
                return false
            }
            guard queries.dim(1) % keys.dim(1) == 0 else {
                lastUnsupportedShape = "query heads must be a multiple of KV heads"
                return false
            }
            lastAttentionPath = polarWHTDecodeAttentionPath
            lastUnsupportedShape = nil
            return true
        }
        guard kvCodec == .polarQJL else {
            lastUnsupportedShape = "unsupported TurboQuant KV codec \(kvCodec.rawValue)"
            return false
        }
        guard activeBackend == .metalPolarQJL,
            availability.supportsMetalPolarQJLAttention || nativeAttentionAvailable
        else {
            lastUnsupportedShape = "metal backend unavailable"
            return false
        }
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else {
            lastUnsupportedShape = "queries/keys/values must be rank 4"
            return false
        }
        guard queries.dim(0) == keys.dim(0), queries.dim(0) == values.dim(0) else {
            lastUnsupportedShape = "query/key/value batch sizes must match"
            return false
        }
        guard keys.dim(2) == values.dim(2), keys.dim(1) == values.dim(1) else {
            lastUnsupportedShape = "key/value token and head counts must match"
            return false
        }
        guard turboQuantSupportsAttentionDimension(queries.dim(3)),
            turboQuantSupportsAttentionDimension(keys.dim(3)),
            turboQuantSupportsAttentionDimension(values.dim(3)),
            queries.dim(3) == keys.dim(3)
        else {
            lastUnsupportedShape =
                "unsupported head dimension q=\(queries.dim(3)) k=\(keys.dim(3)) v=\(values.dim(3))"
            return false
        }
        guard queries.dim(1) % keys.dim(1) == 0 else {
            lastUnsupportedShape = "query heads must be a multiple of KV heads"
            return false
        }
        if compressedKeys == nil, offset > 0, rawFallbackCache != nil {
            lastUnsupportedShape = "raw fallback cache already owns previous rotating context"
            return false
        }
        let keyCode: TurboQuantAttentionCode
        if let compressedKeys {
            keyCode = compressedKeys
        } else if let placeholder = try? placeholderCode(for: keys, role: .key) {
            keyCode = placeholder
        } else {
            lastUnsupportedShape = "failed to create compressed attention placeholder"
            return false
        }
        let supportsNative =
            optimizationPolicy != .conservative && nativeAttentionAvailable
            && queries.dim(2) <= 8 && turboQuantNativeSupportsMask(mask)
        let supportsTiled =
            queries.dim(3) == values.dim(3) && prefersOnlineFusedAttention
            && MLX.turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: mask
            )
        lastAttentionPath =
            supportsNative ? .nativeMLXCompressed : (supportsTiled ? .tiledOnlineFused : .twoStageCompressed)
        lastUnsupportedShape =
            supportsNative || supportsTiled
            ? nil
            : "online fused attention is not throughput-admitted for head dimension \(queries.dim(3)); using two-stage compressed attention"
        return true
    }

    public func updateCompressed(keys: MLXArray, values: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    ) {
        let previousOffset = offset
        let tokenCount = keys.dim(2)
        if tokenCount > 1 {
            cacheLifecycle = previousOffset == 0 ? .rawPrefillChunkOpen : cacheLifecycle
        }
        cacheLifecycle = .compressingChunk(start: previousOffset, count: tokenCount)
        let shouldUpdatePackedFallback =
            packedFallbackCache != nil
            && turboQuantSupportsPackedFallback(keys: keys, values: values, groupSize: groupSize)
        let keyConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .key,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend,
            seed: seed ^ turboQuantValueSeedSalt,
            valueBits: valueBits
        )
        if previousOffset == 0, tokenCount > 0, tokenCount <= maxCacheSize,
            compressedKeys == nil, compressedValues == nil
        {
            let logicalLength = min(tokenCount, maxCacheSize)
            let pinnedPrefixLength = pinnedPrefixLength(forLogicalLength: logicalLength)
            let encodedKeys = try MLX.turboQuantMetalEncodeAttention(
                keys,
                configuration: keyConfiguration,
                capacity: maxCacheSize,
                logicalLength: logicalLength,
                ringOffset: 0,
                pinnedPrefixLength: pinnedPrefixLength
            )
            let encodedValues: TurboQuantAttentionCode
            if usesPolarWHTValueOnlyStorage {
                let valueLayout = try MLX.turboQuantAttentionLayout(
                    for: values,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    valueBits: valueBits,
                    capacity: maxCacheSize,
                    logicalLength: logicalLength,
                    ringOffset: 0,
                    pinnedPrefixLength: pinnedPrefixLength
                )
                encodedValues = turboQuantCompactValuePlaceholderCode(
                    layout: valueLayout,
                    preset: preset,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: valueBits
                )
            } else {
                encodedValues = try MLX.turboQuantMetalEncodeAttention(
                    values,
                    configuration: valueConfiguration,
                    capacity: maxCacheSize,
                    logicalLength: logicalLength,
                    ringOffset: 0,
                    pinnedPrefixLength: pinnedPrefixLength
                )
            }
            if shouldMaintainExactRawShadow {
                _ = materializedExactRawShadowCache().update(keys: keys, values: values)
            }
            offset = tokenCount
            writeIndex = nextWriteIndex(afterOffset: offset)
            compressedKeys = encodedKeys
            compressedValues = encodedValues
            let fusedHybridSidecarsUpdated =
                usesPolarWHTValueOnlyStorage
                && updateRotatingHybridAffineKeyAndPolarWHTValueSidecars(
                    keys: keys,
                    values: values,
                    previousOffset: previousOffset
                )
            updateRotatingPolarWHTKeySidecar(
                keys: keys,
                previousOffset: previousOffset
            )
            if !fusedHybridSidecarsUpdated {
                updateRotatingPolarWHTValueSidecar(
                    values: values,
                    previousOffset: previousOffset
                )
                updateHybridAffineKeySidecar(keys: keys, previousOffset: previousOffset)
            }
            refreshKeyPageSummary()
            if sparseValueSelection.mode == .candidateSparse {
                refreshKeyCandidateSketch()
            }
            packedValues = nil
            packedFallbackCache = nil
            lastDecodedTransientBytes = 0
            if !shouldMaintainExactRawShadow {
                releaseRawShadow()
            }
            try validateCompressedState(context: "rotating compressed initial append")
            cacheLifecycle = .compressedCommitted(
                logicalLength: logicalLength,
                capacity: maxCacheSize
            )
            return (encodedKeys, encodedValues)
        }
        if usesPolarWHTValueOnlyStorage, previousOffset > 0, tokenCount == 1,
            TurboQuantRuntimeControl.enabled("TURBOQUANT_ENABLE_HYBRID_POLARWHT_TAIL"),
            !TurboQuantRuntimeControl.enabled("TURBOQUANT_DISABLE_HYBRID_POLARWHT_TAIL")
        {
            guard var currentKeys = compressedKeys, var currentValues = compressedValues else {
                throw TurboQuantCacheError.compressedStorageInvalid(
                    "rotating hybrid PolarWHT value tail append requires committed compressed storage"
                )
            }
            guard updateRotatingHybridAffineKeyAndPolarWHTValueTailSidecars(
                keys: keys,
                values: values
            ) else {
                throw TurboQuantCacheError.compressedStorageInvalid(
                    lastUnsupportedShape
                        ?? "rotating hybrid PolarWHT value tail append failed"
                )
            }
            if shouldMaintainExactRawShadow {
                _ = materializedExactRawShadowCache().update(keys: keys, values: values)
            }
            offset += tokenCount
            writeIndex = nextWriteIndex(afterOffset: offset)
            if shouldUpdatePackedFallback {
                _ = packedFallbackCache?.updateQuantized(keys: keys, values: values)
            } else {
                packedFallbackCache = nil
            }
            updateCompressedLayouts(keys: &currentKeys, values: &currentValues)
            compressedKeys = currentKeys
            compressedValues = currentValues
            packedValues = nil
            lastDecodedTransientBytes = 0
            if !shouldMaintainExactRawShadow {
                releaseRawShadow()
            }
            try validateCompressedState(context: "rotating hybrid PolarWHT value tail append")
            cacheLifecycle = .compressedCommitted(
                logicalLength: currentKeys.layout.logicalLength,
                capacity: currentKeys.layout.capacity
            )
            return (currentKeys, currentValues)
        }
        let encodedKeys = try MLX.turboQuantMetalEncodeAttention(
            keys,
            configuration: keyConfiguration
        )
        let encodedValues =
            usesPolarWHTValueOnlyStorage
            ? nil
            : try MLX.turboQuantMetalEncodeAttention(
                values,
                configuration: valueConfiguration
            )
        try ensureCompressedStorage(keys: keys, values: values)

        var currentKeys = compressedKeys!
        var currentValues = compressedValues!
        // 3A (audit 1.3): TurboQuantAttentionCode is a struct, so the copy above gives every
        // packed plane refcount 2 (currentKeys + compressedKeys), which forces MLX to reallocate
        // a fresh full-capacity buffer for each plane on the slice-update writes below — O(context)
        // bytes/token, a prime OOM suspect. Releasing the stored references here drops them to
        // refcount 1 so the writes donate in place. Restored at the reassign below; verified that
        // nothing between reads compressedKeys/compressedValues.
        compressedKeys = nil
        compressedValues = nil
        let physicalStart = physicalSlot(forAbsoluteToken: offset)
        if tokenCount > 0, physicalStart + tokenCount <= maxCacheSize {
            let target = physicalStart ..< (physicalStart + tokenCount)
            currentKeys.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                encodedKeys.packedMagnitudes
            currentKeys.signs[.ellipsis, target, 0..., 0...] = encodedKeys.signs
            if currentKeys.highPrecisionMask.ndim == 5 {
                currentKeys.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                    encodedKeys.highPrecisionMask
            }
            if currentKeys.residualSigns.ndim == 5 {
                currentKeys.residualSigns[.ellipsis, target, 0..., 0...] =
                    encodedKeys.residualSigns
            }
            currentKeys.scales[.ellipsis, target, 0..., 0...] = encodedKeys.scales

            if let encodedValues {
                currentValues.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                    encodedValues.packedMagnitudes
                if currentValues.signs.ndim == 5 {
                    currentValues.signs[.ellipsis, target, 0..., 0...] = encodedValues.signs
                }
                if currentValues.highPrecisionMask.ndim == 5 {
                    currentValues.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                        encodedValues.highPrecisionMask
                }
                if currentValues.residualSigns.ndim == 5 {
                    currentValues.residualSigns[.ellipsis, target, 0..., 0...] =
                        encodedValues.residualSigns
                }
                currentValues.scales[.ellipsis, target, 0..., 0...] = encodedValues.scales
            }
        } else {
            for token in 0 ..< tokenCount {
                let physical = physicalSlot(forAbsoluteToken: offset + token)
                let source = token ..< (token + 1)
                let target = physical ..< (physical + 1)
                currentKeys.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                    encodedKeys.packedMagnitudes[.ellipsis, source, 0..., 0...]
                currentKeys.signs[.ellipsis, target, 0..., 0...] =
                    encodedKeys.signs[.ellipsis, source, 0..., 0...]
                if currentKeys.highPrecisionMask.ndim == 5 {
                    currentKeys.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                        encodedKeys.highPrecisionMask[.ellipsis, source, 0..., 0...]
                }
                if currentKeys.residualSigns.ndim == 5 {
                    currentKeys.residualSigns[.ellipsis, target, 0..., 0...] =
                        encodedKeys.residualSigns[.ellipsis, source, 0..., 0...]
                }
                currentKeys.scales[.ellipsis, target, 0..., 0...] =
                    encodedKeys.scales[.ellipsis, source, 0..., 0...]

                if let encodedValues {
                    currentValues.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                        encodedValues.packedMagnitudes[.ellipsis, source, 0..., 0...]
                    if currentValues.signs.ndim == 5 {
                        currentValues.signs[.ellipsis, target, 0..., 0...] =
                            encodedValues.signs[.ellipsis, source, 0..., 0...]
                    }
                    if currentValues.highPrecisionMask.ndim == 5 {
                        currentValues.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                            encodedValues.highPrecisionMask[.ellipsis, source, 0..., 0...]
                    }
                    if currentValues.residualSigns.ndim == 5 {
                        currentValues.residualSigns[.ellipsis, target, 0..., 0...] =
                            encodedValues.residualSigns[.ellipsis, source, 0..., 0...]
                    }
                    currentValues.scales[.ellipsis, target, 0..., 0...] =
                        encodedValues.scales[.ellipsis, source, 0..., 0...]
                }
            }
        }

        if shouldMaintainExactRawShadow {
            _ = materializedExactRawShadowCache().update(keys: keys, values: values)
        }

        offset += tokenCount
        writeIndex = nextWriteIndex(afterOffset: offset)
        if shouldUpdatePackedFallback {
            _ = packedFallbackCache?.updateQuantized(keys: keys, values: values)
        } else {
            packedFallbackCache = nil
        }
        let fusedHybridSidecarsUpdated =
            usesPolarWHTValueOnlyStorage
            && updateRotatingHybridAffineKeyAndPolarWHTValueSidecars(
                keys: keys,
                values: values,
                previousOffset: previousOffset
            )
        if !fusedHybridSidecarsUpdated {
            updateHybridAffineKeySidecar(keys: keys, previousOffset: previousOffset)
        }
        updateCompressedLayouts(keys: &currentKeys, values: &currentValues)
        compressedKeys = currentKeys
        compressedValues = currentValues
        updateRotatingPolarWHTKeySidecar(
            keys: keys,
            previousOffset: previousOffset
        )
        if !fusedHybridSidecarsUpdated {
            updateRotatingPolarWHTValueSidecar(
                values: values,
                previousOffset: previousOffset
            )
        }
        updateKeyCandidateSketchAfterLinearAppend(
            encodedKeys: encodedKeys,
            previousOffset: min(previousOffset, maxCacheSize),
            tokenCount: tokenCount
        )
        refreshKeyPageSummary()
        if !shouldMaintainHybridAffineKeySidecar {
            packedKeys = nil
        }
        packedValues = nil
        lastDecodedTransientBytes = 0
        if !shouldMaintainExactRawShadow {
            releaseRawShadow()
        }
        try validateCompressedState(context: "rotating compressed append")
        cacheLifecycle = .compressedCommitted(
            logicalLength: currentKeys.layout.logicalLength,
            capacity: currentKeys.layout.capacity
        )
        return (currentKeys, currentValues)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previousOffset = offset
        let rawCache = materializedRawFallbackCache()
        let result = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset
        writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
        updateRotatingPolarWHTKeySidecar(keys: keys, previousOffset: previousOffset)
        updateRotatingPolarWHTValueSidecar(values: values, previousOffset: previousOffset)
        return result
    }

    public func exactRawStateIfComplete() -> (keys: MLXArray, values: MLXArray)? {
        guard shouldMaintainExactRawShadow,
              let rawFallbackCache,
              let compressedKeys,
              rawFallbackCache.offset == offset
        else {
            return nil
        }
        let state = rawFallbackCache.state
        guard state.count == 2,
              state[0].dim(2) == compressedKeys.layout.logicalLength,
              state[1].dim(2) == compressedKeys.layout.logicalLength
        else {
            return nil
        }
        return (state[0], state[1])
    }

    public override var state: [MLXArray] {
        get {
            if turboQuantIsMetalCompressedBackend(activeBackend),
                let compressedKeys,
                let compressedValues
            {
                return [
                    compressedKeys.packedMagnitudes,
                    compressedKeys.signs,
                    compressedKeys.highPrecisionMask,
                    compressedKeys.residualSigns,
                    compressedKeys.scales,
                    compressedValues.packedMagnitudes,
                    compressedValues.signs,
                    compressedValues.highPrecisionMask,
                    compressedValues.residualSigns,
                    compressedValues.scales,
                ]
            }
            return rawFallbackCache?.state ?? []
        }
        set {
            if newValue.isEmpty {
                packedKeys = nil
                packedValues = nil
                packedFallbackCache = nil
                compressedKeys = nil
                compressedValues = nil
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                hybridAffineKeyTailSidecar = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                keyPageSummary = nil
                keyCandidateSketch = nil
                keyCandidateSketchUnavailableReason = "compressed key state is empty"
                rawFallbackCache = nil
                lastDecodedTransientBytes = 0
                cacheLifecycle = .empty
                return
            }
            if turboQuantIsMetalCompressedBackend(activeBackend), newValue.count == 10 {
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                hybridAffineKeyTailSidecar = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                packedFallbackCache = nil
                let capacity = restoredLayoutMetadata?.capacity ?? newValue[0].dim(2)
                let keyHeadDimension =
                    restoredLayoutMetadata?.keyHeadDimension
                    ?? max(groupSize, (newValue[0].dim(3) * groupSize))
                let valueHeadDimension =
                    restoredLayoutMetadata?.valueHeadDimension
                    ?? (
                        newValue[5].shape == [1]
                            ? keyHeadDimension : max(groupSize, (newValue[5].dim(3) * groupSize))
                    )
                let logicalLength = restoredLayoutMetadata?.logicalLength ?? min(offset, capacity)
                let pinnedPrefixLength =
                    restoredLayoutMetadata?.pinnedPrefixLength
                    ?? min(keep, capacity, max(0, logicalLength))
                let keyLayout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[0].dim(0),
                    kvHeadCount: restoredLayoutMetadata?.kvHeadCount ?? newValue[0].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: restoredLayoutMetadata?.ringOffset ?? ringOffset(forOffset: offset),
                    pinnedPrefixLength: pinnedPrefixLength,
                    headDimension: keyHeadDimension,
                    groupsPerVector: newValue[0].dim(3),
                    magnitudeWordsPerGroup: newValue[0].dim(4),
                    bitsetWordsPerGroup: newValue[1].dim(4)
                )
                let valueLayout = turboQuantRestoredValueLayout(
                    valuePacked: newValue[5],
                    keyLayout: keyLayout,
                    valueHeadDimension: valueHeadDimension,
                    groupSize: groupSize
                )
                // T1.4 stage 1: scalesPerGroup omitted here defaults to 2 (the dieted K scale
                // plane) regardless of newValue[4]'s actual last dim. This setter is non-throwing
                // (a `state` property setter), so a stale 3-scale snapshot restored through this
                // path is not rejected here; it fails closed at first compressed-attention use
                // instead, where mlx-swift's `validateTurboQuantAttentionCode` throws "compressed
                // attention scales per group ... does not match expected" (see
                // TurboQuantValidation.swift). The primary, throwing restore path
                // (`turboQuantSnapshotImportedCodes`) guards this explicitly via
                // `turboQuantRequireKeyScalesPerGroup`.
                compressedKeys = TurboQuantAttentionCode(
                    layout: keyLayout,
                    preset: preset,
                    role: .key,
                    groupSize: groupSize,
                    seed: seed,
                    packedMagnitudes: newValue[0],
                    signs: newValue[1],
                    highPrecisionMask: newValue[2],
                    residualSigns: newValue[3],
                    scales: newValue[4]
                )
                compressedValues = TurboQuantAttentionCode(
                    layout: valueLayout,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: valueBits,
                    scalesPerGroup: turboQuantRestoredScalesPerGroup(newValue[9]),
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
                if let compressedKeys {
                    refreshKeyPageSummary()
                    cacheLifecycle = .compressedCommitted(
                        logicalLength: compressedKeys.layout.logicalLength,
                        capacity: compressedKeys.layout.capacity
                    )
                }
            } else if newValue.count == 10 {
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                hybridAffineKeyTailSidecar = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                keyPageSummary = nil
                keyCandidateSketch = nil
                keyCandidateSketchUnavailableReason =
                    "compressed rotating TurboQuant state restored without Metal attention support"
                lastUnsupportedShape =
                    "compressed rotating TurboQuant state restored without Metal attention support"
                cacheLifecycle = .failed(
                    reason: lastUnsupportedShape ?? "unsupported compressed state")
            } else {
                let rawCache = materializedRawFallbackCache()
                rawCache.state = newValue
                offset = rawCache.offset
                writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
                packedFallbackCache = nil
                compressedKeys = nil
                compressedValues = nil
                polarWHTKeyCode = nil
                polarWHTValueCode = nil
                hybridAffineKeyTailSidecar = nil
                polarWHTValueTailCode = nil
                polarWHTDecodedValueBuffer = nil
                keyPageSummary = nil
                keyCandidateSketch = nil
                keyCandidateSketchUnavailableReason =
                    "raw fallback state does not expose compressed key sketches"
                cacheLifecycle = .degradedDecodedFallback(reason: "restored raw fallback state")
            }
            packedKeys = nil
            packedValues = nil
        }
    }

    public override var metaState: [String] {
        get {
            var meta = [
                String(keep),
                String(maxCacheSize),
                String(step),
                String(offset),
                String(writeIndex),
                preset.rawValue,
                String(groupSize),
                requestedBackend.rawValue,
                String(seed),
                "valueBits=\(valueBits)",
                "kvCodec=\(kvCodec.rawValue)",
            ]
            if let compressedKeys {
                let layout = compressedKeys.layout
                let valueLayout = compressedValues?.layout
                meta += [
                    "turboq-rot-v\(layout.layoutVersion)",
                    String(layout.logicalLength),
                    String(layout.ringOffset),
                    String(layout.pinnedPrefixLength),
                    String(layout.headDimension),
                    lastAttentionPath.rawValue,
                    rawFallbackCache == nil ? "raw-free" : "raw-fallback",
                    "keyHeadDimension=\(layout.headDimension)",
                    "valueHeadDimension=\(valueLayout?.headDimension ?? layout.headDimension)",
                    "kvHeadCount=\(layout.kvHeadCount)",
                ]
            }
            return meta
        }
        set {
            guard newValue.count >= 5 else { return }
            keyPageSummary = nil
            keyCandidateSketch = nil
            keyCandidateSketchUnavailableReason = "key candidate sketch was cleared by metadata restore"
            offset = Int(newValue[3]) ?? 0
            writeIndex = Int(newValue[4]) ?? nextWriteIndex(afterOffset: offset)
            if let rawFallbackCache {
                rawFallbackCache.metaState = Array(newValue.prefix(5))
            }
            let compressedBase =
                newValue.firstIndex {
                    $0.hasPrefix("turboq-rot-v")
                } ?? 9
            if newValue.count >= compressedBase + 4,
                let logicalLength = Int(newValue[compressedBase + 1]),
                let ringOffset = Int(newValue[compressedBase + 2]),
                let pinnedPrefixLength = Int(newValue[compressedBase + 3])
            {
                var headDimension: Int?
                var pathIndex = compressedBase + 4
                if newValue.count > compressedBase + 5,
                    let parsedHeadDimension = Int(newValue[compressedBase + 4])
                {
                    headDimension = parsedHeadDimension
                    pathIndex = compressedBase + 5
                }
                restoredLayoutMetadata = RestoredAttentionLayoutMetadata(
                    capacity: maxCacheSize,
                    logicalLength: logicalLength,
                    ringOffset: ringOffset,
                    pinnedPrefixLength: pinnedPrefixLength,
                    headDimension: headDimension,
                    valueHeadDimension: turboQuantMetaInt(
                        newValue, key: "valueHeadDimension") ?? headDimension,
                    kvHeadCount: nil
                )
                if var compressedKeys {
                    compressedKeys.layout.logicalLength = logicalLength
                    compressedKeys.layout.ringOffset = ringOffset
                    compressedKeys.layout.pinnedPrefixLength = pinnedPrefixLength
                    if let keyHeadDimension =
                        turboQuantMetaInt(newValue, key: "keyHeadDimension") ?? headDimension
                    {
                        compressedKeys.layout.headDimension = keyHeadDimension
                    }
                    self.compressedKeys = compressedKeys
                }
                if var compressedValues {
                    compressedValues.layout.logicalLength = logicalLength
                    compressedValues.layout.ringOffset = ringOffset
                    compressedValues.layout.pinnedPrefixLength = pinnedPrefixLength
                    if let valueHeadDimension =
                        turboQuantMetaInt(newValue, key: "valueHeadDimension") ?? headDimension
                    {
                        compressedValues.layout.headDimension = valueHeadDimension
                    }
                    self.compressedValues = compressedValues
                }
                if pathIndex < newValue.count,
                    let path = TurboQuantAttentionPath(rawValue: newValue[pathIndex])
                {
                    lastAttentionPath = path
                }
                refreshKeyPageSummary()
            }
        }
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if activeBackend != .metalPolarQJL, let rawFallbackCache {
            return rawFallbackCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }
        if n > 1 {
            let actualWindowSize = windowSize ?? maxCacheSize
            let cappedOffset = min(maxCacheSize - 1, offset)
            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        }
        guard let windowSize else { return .none }
        if offset >= windowSize, maxCacheSize > windowSize {
            let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
            let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
            var currentWriteIndex = writeIndex
            if currentWriteIndex >= maxCacheSize {
                currentWriteIndex = 0
            }
            return .array(roll(mask, shift: currentWriteIndex + 1))
        }
        return .none
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        writeIndex = nextWriteIndex(afterOffset: offset)
        if let rawFallbackCache {
            _ = rawFallbackCache.trim(n)
        }
        if let packedFallbackCache {
            _ = packedFallbackCache.trim(n)
        }
        if var compressedKeys, var compressedValues {
            updateCompressedLayouts(keys: &compressedKeys, values: &compressedValues)
            self.compressedKeys = compressedKeys
            self.compressedValues = compressedValues
            refreshKeyPageSummary()
        }
        if var polarWHTValueCode {
            updateRotatingPolarWHTValueLayout(&polarWHTValueCode, offset: offset)
            self.polarWHTValueCode = polarWHTValueCode
        }
        if offset == 0 {
            polarWHTDecodedValueBuffer = nil
        }
        if var polarWHTKeyCode {
            updateRotatingPolarWHTValueLayout(&polarWHTKeyCode, offset: offset)
            self.polarWHTKeyCode = polarWHTKeyCode
        }
        if shouldMaintainHybridAffineKeySidecar, let current = packedKeys {
            if offset > 0, current.weight.dim(2) > offset {
                packedKeys = turboQuantSlicePackedTensor(current, tokens: 0 ..< offset)
            } else if offset == 0 {
                packedKeys = nil
            }
        } else {
            packedKeys = nil
        }
        packedValues = nil
        return trimmed
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        lastAttentionPath = .mlxPackedFallback
        lastUnsupportedShape = "compressed attention failed: \(message)"
        lastNativeAttentionDiagnostics = nil
        keyPageSummary = nil
        invalidateKeyCandidateSketch(reason: "compressed attention failed: \(message)")
        cacheLifecycle = .failed(reason: message)
    }

    public override func copy() -> any KVCache {
        let new = RotatingTurboQuantKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend,
            kvCodec: kvCodec,
            optimizationPolicy: optimizationPolicy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: valueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            residentBudgetBytes: residentBudgetBytes
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        new.packedKeys = copiedTurboQuantPackedTensor(packedKeys)
        new.packedValues = copiedTurboQuantPackedTensor(packedValues)
        new.polarWHTKeyCode = copiedTurboQuantPolarWHTCode(polarWHTKeyCode)
        new.polarWHTValueCode = copiedTurboQuantPolarWHTValueCode(polarWHTValueCode)
        new.hybridAffineKeyTailSidecar = copiedTurboQuantAffineKeySidecar(
            hybridAffineKeyTailSidecar
        )
        new.polarWHTValueTailCode = copiedTurboQuantPolarWHTValueCode(polarWHTValueTailCode)
        new.polarWHTDecodedValueBuffer = polarWHTDecodedValueBuffer?[.ellipsis]
        return new
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        let sparseSelection = sparseValueSelection.resolved(
            runtimeMode: resolvedRuntimeMode,
            contextLength: compressedKeys?.layout.logicalLength ?? min(offset, maxCacheSize),
            policy: sparseValuePolicy
        )
        let sparseVInactiveReason = turboQuantSparseVInactiveReason(
            enabled: sparseSelection.isEnabled,
            kvCodec: kvCodec,
            activeBackend: activeBackend,
            nativeDiagnostics: lastNativeAttentionDiagnostics,
            fallbackReason: backendFallbackReason
        )
        return TurboQuantKVCacheDiagnostics(
            layerIndex: layerIndex,
            kvCodec: kvCodec,
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: sparseVInactiveReason,
            metalCodecAvailable: turboQuantMetalCodecAvailable(
                kvCodec: kvCodec,
                availability: availability
            ),
            metalAttentionAvailable: turboQuantMetalAttentionAvailable(
                kvCodec: kvCodec,
                availability: availability
            ),
            activeAttentionPath: lastAttentionPath,
            nativeBackend: availability.attentionCapabilities.nativeCompressedAttention == true
                ? "nativeMLX" : nil,
            nativeBackendVersion: availability.attentionCapabilities.nativeBackendVersion,
            nativeFallbackReason: availability.attentionCapabilities.nativeFallbackReason,
            nativeKernelKind: lastNativeAttentionDiagnostics?.kernelKind,
            nativeSparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: optimizationPolicy,
            lastUnsupportedShape: lastUnsupportedShape,
            groupSize: groupSize,
            bits: bits,
            valueBits: valueBits,
            maxSize: maxSize,
            rawFallbackAllocated: rawFallbackCache != nil,
            cacheLifecycle: cacheLifecycle,
            lastFallback: fallbackResults.last,
            footprint: cacheFootprint,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: polarWHTKeyPayloadAllocated,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: polarWHTValuePayloadAllocated,
            sparseVEnabled: sparseSelection.isEnabled,
            sparseVThreshold: sparseSelection.resolvedThreshold,
            sparseVSelectionMode: sparseSelection.mode,
            sparseVTopK: sparseSelection.topK,
            sparseVCumulativeMass: sparseSelection.cumulativeMass,
            sparseVMaxTopK: sparseSelection.maxTopK,
            sparseVRecentTokenCount: sparseSelection.recentTokens,
            sparseVOlderTokenCount: sparseSelection.mode == .candidateSparse
                ? sparseSelection.topK : nil,
            sparseVPageCandidateCount: sparseSelection.candidatePages,
            sparseVSkippedTokens: lastNativeAttentionDiagnostics?.sparseSkippedTokens,
            sparseVTotalTokens: lastNativeAttentionDiagnostics?.sparseTotalTokens,
            sparseVActive: turboQuantSparseVActive(lastNativeAttentionDiagnostics),
            sparseVSkipRatio: lastNativeAttentionDiagnostics?.sparseSkipRatio,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            keyCandidateSketchAvailable: keyCandidateSketch != nil,
            keyCandidateSketchShape: keyCandidateSketch?.shape,
            keyCandidateSketchUnavailableReason: keyCandidateSketchUnavailableReason
        )
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxSize?.description ?? "-"), preset: \(preset.rawValue), backend: \(activeBackend.rawValue), rawFallback: \(rawFallbackCache != nil)"
    }

    private var polarWHTValueBytes: Int {
        guard kvCodec == .polarWHT else { return 0 }
        return (polarWHTValueCode?.residentPayloadByteCount ?? 0)
            + (polarWHTValueTailCode?.residentPayloadByteCount ?? 0)
    }

    private var polarWHTDecodedValueResidentBytes: Int {
        polarWHTDecodedValueBuffer?.nbytes ?? 0
    }

    private var polarWHTDecodedValueActiveBytes: Int {
        guard let polarWHTDecodedValueBuffer else { return 0 }
        let activeLength = min(
            polarWHTValueCode?.layout.logicalLength ?? min(offset, maxCacheSize),
            polarWHTDecodedValueBuffer.dim(2)
        )
        guard activeLength > 0, polarWHTDecodedValueBuffer.dim(2) > 0 else { return 0 }
        return polarWHTDecodedValueBuffer.nbytes * activeLength / polarWHTDecodedValueBuffer.dim(2)
    }

    private var polarWHTValuePayloadAllocated: Bool {
        polarWHTValueBytes > 0
    }

    private var polarWHTKeyBytes: Int {
        guard kvCodec == .polarWHT else { return 0 }
        return polarWHTKeyCode?.residentPayloadByteCount ?? 0
    }

    private var polarWHTKeyPayloadAllocated: Bool {
        polarWHTKeyBytes > 0
    }

    private func polarWHTKeyStorageMatches(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        keys: MLXArray
    ) -> Bool {
        code.bits == bits
            && code.seed == seed
            && code.layout.batchSize == keys.dim(0)
            && code.layout.kvHeadCount == keys.dim(1)
            && code.layout.capacity == maxCacheSize
            && code.layout.headDimension == keys.dim(3)
    }

    private func polarWHTValueStorageMatches(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        values: MLXArray
    ) -> Bool {
        code.bits == valueBits
            && code.seed == seed ^ turboQuantValueSeedSalt
            && code.layout.batchSize == values.dim(0)
            && code.layout.kvHeadCount == values.dim(1)
            && code.layout.capacity == maxCacheSize
            && code.layout.headDimension == values.dim(3)
    }

    private func polarWHTValueTailStorageMatches(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        values: MLXArray
    ) -> Bool {
        code.bits == valueBits
            && code.seed == seed ^ turboQuantValueSeedSalt
            && code.layout.batchSize == values.dim(0)
            && code.layout.kvHeadCount == values.dim(1)
            && code.layout.headDimension == values.dim(3)
    }

    private func roundedRotatingPolarWHTTailCapacity(requiredLength: Int) -> Int {
        guard requiredLength > 0 else { return 0 }
        let quantum = max(1, step)
        return ((quantum + requiredLength - 1) / quantum) * quantum
    }

    private func expandedRotatingPolarWHTValueCode(
        _ code: TurboQuantPolarWHTAttentionValueCode,
        capacity requestedCapacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode {
        let capacity = max(0, requestedCapacity)
        if capacity == code.layout.capacity {
            return code
        }
        var layout = code.layout
        layout.capacity = capacity
        let empty = try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: code.bits,
            seed: code.seed,
            normStorage: code.norms.dtype
        )
        var expanded = empty
        let copyLength = min(code.layout.capacity, capacity)
        if copyLength > 0 {
            let copyRange = 0 ..< copyLength
            expanded.packedIndices[.ellipsis, copyRange, 0...] =
                code.packedIndices[.ellipsis, copyRange, 0...]
            expanded.norms[.ellipsis, copyRange] = code.norms[.ellipsis, copyRange]
        }
        expanded.layout.logicalLength = min(code.layout.logicalLength, capacity)
        expanded.layout.ringOffset = 0
        expanded.layout.pinnedPrefixLength = 0
        return expanded
    }

    private func updateRotatingPolarWHTValueLayout(
        _ code: inout TurboQuantPolarWHTAttentionValueCode,
        offset targetOffset: Int
    ) {
        let logicalLength = min(targetOffset, maxCacheSize)
        code.layout.logicalLength = logicalLength
        code.layout.ringOffset = ringOffset(forOffset: targetOffset)
        code.layout.pinnedPrefixLength = pinnedPrefixLength(forLogicalLength: logicalLength)
    }

    private func ensureRotatingPolarWHTKeyStorage(
        keys: MLXArray,
        previousOffset: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode? {
        if let current = polarWHTKeyCode {
            guard polarWHTKeyStorageMatches(current, keys: keys) else {
                polarWHTKeyCode = nil
                return nil
            }
            return current
        }
        guard previousOffset == 0 else { return nil }
        let layout = MLX.TurboQuantAttentionLayout(
            batchSize: keys.dim(0),
            kvHeadCount: keys.dim(1),
            capacity: maxCacheSize,
            logicalLength: 0,
            ringOffset: 0,
            pinnedPrefixLength: 0,
            headDimension: keys.dim(3),
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 0,
            bitsetWordsPerGroup: 0
        )
        return try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: bits,
            seed: seed,
            normStorage: .float32
        )
    }

    fileprivate func updateRotatingPolarWHTKeySidecar(
        keys: MLXArray,
        previousOffset: Int
    ) {
        guard shouldMaintainPolarWHTKeySidecar else {
            polarWHTKeyCode = nil
            return
        }
        let keys = keys.contiguous(stream: .gpu)
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return }

        do {
            let encodedChunk = try turboQuantEncodePolarWHTAttentionValues(
                keys,
                bits: bits,
                seed: seed,
                capacity: tokenCount,
                logicalLength: tokenCount
            )
            guard var current = try ensureRotatingPolarWHTKeyStorage(
                keys: keys,
                previousOffset: previousOffset
            ) else {
                return
            }
            polarWHTKeyCode = nil
            for token in 0 ..< tokenCount {
                let physical = physicalSlot(forAbsoluteToken: previousOffset + token)
                let source = token ..< (token + 1)
                let target = physical ..< (physical + 1)
                current.packedIndices[.ellipsis, target, 0...] =
                    encodedChunk.packedIndices[.ellipsis, source, 0...]
                current.norms[.ellipsis, target] = encodedChunk.norms[.ellipsis, source]
            }
            updateRotatingPolarWHTValueLayout(
                &current,
                offset: previousOffset + tokenCount
            )
            polarWHTKeyCode = current
        } catch {
            polarWHTKeyCode = nil
            lastUnsupportedShape = "rotating PolarWHT key sidecar update failed: \(error)"
        }
    }

    private func ensureRotatingPolarWHTValueStorage(
        values: MLXArray,
        previousOffset: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode? {
        if let current = polarWHTValueCode {
            guard polarWHTValueStorageMatches(current, values: values) else {
                polarWHTValueCode = nil
                polarWHTDecodedValueBuffer = nil
                return nil
            }
            return current
        }
        guard previousOffset == 0 else { return nil }
        let layout = MLX.TurboQuantAttentionLayout(
            batchSize: values.dim(0),
            kvHeadCount: values.dim(1),
            capacity: maxCacheSize,
            logicalLength: 0,
            ringOffset: 0,
            pinnedPrefixLength: 0,
            headDimension: values.dim(3),
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 0,
            bitsetWordsPerGroup: 0
        )
        return try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: valueBits,
            seed: seed ^ turboQuantValueSeedSalt,
            normStorage: .float32
        )
    }

    private func ensureRotatingPolarWHTValueTailStorage(
        values: MLXArray,
        capacity requestedCapacity: Int
    ) throws -> TurboQuantPolarWHTAttentionValueCode? {
        let capacity = max(0, requestedCapacity)
        if let current = polarWHTValueTailCode {
            guard polarWHTValueTailStorageMatches(current, values: values) else {
                polarWHTValueTailCode = nil
                return nil
            }
            if current.layout.capacity >= capacity {
                return current
            }
            return try expandedRotatingPolarWHTValueCode(current, capacity: capacity)
        }
        let layout = MLX.TurboQuantAttentionLayout(
            batchSize: values.dim(0),
            kvHeadCount: values.dim(1),
            capacity: capacity,
            logicalLength: 0,
            ringOffset: 0,
            pinnedPrefixLength: 0,
            headDimension: values.dim(3),
            groupsPerVector: 1,
            magnitudeWordsPerGroup: 0,
            bitsetWordsPerGroup: 0
        )
        return try MLX.turboQuantEmptyPolarWHTAttentionValueCode(
            layout: layout,
            bits: valueBits,
            seed: seed ^ turboQuantValueSeedSalt,
            normStorage: .float32
        )
    }

    private func mergedRotatingPolarWHTDecodedValueBuffer(
        decodedChunk: MLXArray,
        values: MLXArray,
        previousOffset: Int
    ) -> MLXArray? {
        let tokenCount = decodedChunk.dim(2)
        guard tokenCount > 0 else { return polarWHTDecodedValueBuffer }
        guard decodedChunk.ndim == 4,
            decodedChunk.dim(0) == values.dim(0),
            decodedChunk.dim(1) == values.dim(1),
            decodedChunk.dim(3) == values.dim(3),
            decodedChunk.dtype == values.dtype
        else {
            return nil
        }
        let buffer: MLXArray
        if let current = polarWHTDecodedValueBuffer,
            current.ndim == 4,
            current.dim(0) == values.dim(0),
            current.dim(1) == values.dim(1),
            current.dim(2) == maxCacheSize,
            current.dim(3) == values.dim(3),
            current.dtype == values.dtype
        {
            buffer = current
        } else {
            buffer = MLXArray.zeros(
                [values.dim(0), values.dim(1), maxCacheSize, values.dim(3)],
                dtype: values.dtype
            )
        }

        let writeCount = min(tokenCount, maxCacheSize)
        let sourceBase = tokenCount - writeCount
        polarWHTDecodedValueBuffer = nil
        if writeCount > 0 {
            let physicalStart = physicalSlot(forAbsoluteToken: previousOffset + sourceBase)
            if physicalStart + writeCount <= maxCacheSize {
                buffer[.ellipsis, physicalStart ..< (physicalStart + writeCount), 0...] =
                    decodedChunk[.ellipsis, sourceBase ..< (sourceBase + writeCount), 0...]
            } else {
                for token in 0 ..< writeCount {
                    let physical = physicalSlot(
                        forAbsoluteToken: previousOffset + sourceBase + token
                    )
                    let source = (sourceBase + token) ..< (sourceBase + token + 1)
                    let target = physical ..< (physical + 1)
                    buffer[.ellipsis, target, 0...] = decodedChunk[.ellipsis, source, 0...]
                }
            }
        }
        return buffer
    }

    @discardableResult
    private func updateRotatingHybridAffineKeyAndPolarWHTValueSidecars(
        keys: MLXArray,
        values: MLXArray,
        previousOffset: Int
    ) -> Bool {
        guard shouldMaintainHybridAffineKeySidecar, shouldMaintainPolarWHTValueSidecar else {
            return false
        }
        guard supportsMLXAffineQuantization(dimension: keys.dim(3), groupSize: groupSize) else {
            return false
        }
        let keys = keys.contiguous(stream: .gpu)
        let values = values.contiguous(stream: .gpu)
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return true }
        guard values.dim(0) == keys.dim(0),
            values.dim(1) == keys.dim(1),
            values.dim(2) == tokenCount,
            values.dim(3) == keys.dim(3)
        else {
            return false
        }

        let logicalLength = min(previousOffset + tokenCount, maxCacheSize)
        let isInitialStorage = previousOffset == 0
        let fusedCapacity = isInitialStorage ? maxCacheSize : tokenCount
        let fusedLogicalLength = isInitialStorage ? logicalLength : tokenCount
        let valueSeed = seed ^ turboQuantValueSeedSalt

        do {
            let fused = try MLX.turboQuantMetalHybridAffineK8PolarWHTValueEncode(
                keys: keys,
                values: values,
                keyGroupSize: groupSize,
                valueBits: valueBits,
                valueSeed: valueSeed,
                capacity: fusedCapacity,
                logicalLength: fusedLogicalLength,
                ringOffset: isInitialStorage ? 0 : 0,
                pinnedPrefixLength: isInitialStorage
                    ? pinnedPrefixLength(forLogicalLength: logicalLength) : 0
            )

            let encodedKey =
                isInitialStorage
                ? turboQuantSlicePackedTensor(fused.key, tokens: 0 ..< logicalLength)
                : fused.key
            guard let updatedKey = mergedRotatingHybridAffineKeySidecar(
                encoded: encodedKey,
                previousOffset: previousOffset
            ) else {
                return false
            }

            if isInitialStorage {
                packedKeys = updatedKey
                packedValues = nil
                polarWHTValueCode = fused.value
                hybridAffineKeyTailSidecar = nil
                polarWHTValueTailCode = nil
                if shouldMaintainPolarWHTDecodedValueBuffer {
                    let decodedChunk = try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                        fused.value,
                        outputDType: values.dtype
                    )
                    polarWHTDecodedValueBuffer = mergedRotatingPolarWHTDecodedValueBuffer(
                        decodedChunk: decodedChunk,
                        values: values,
                        previousOffset: previousOffset
                    )
                } else {
                    polarWHTDecodedValueBuffer = nil
                }
                return true
            }

            guard var currentValue = try ensureRotatingPolarWHTValueStorage(
                values: values,
                previousOffset: previousOffset
            ) else {
                return false
            }
            polarWHTValueCode = nil
            for token in 0 ..< tokenCount {
                let physical = physicalSlot(forAbsoluteToken: previousOffset + token)
                let source = token ..< (token + 1)
                let target = physical ..< (physical + 1)
                currentValue.packedIndices[.ellipsis, target, 0...] =
                    fused.value.packedIndices[.ellipsis, source, 0...]
                currentValue.norms[.ellipsis, target] = fused.value.norms[.ellipsis, source]
            }
            updateRotatingPolarWHTValueLayout(
                &currentValue,
                offset: previousOffset + tokenCount
            )
            packedKeys = updatedKey
            packedValues = nil
            polarWHTValueCode = currentValue
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            if shouldMaintainPolarWHTDecodedValueBuffer {
                let decodedChunk = try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    fused.value,
                    outputDType: values.dtype
                )
                polarWHTDecodedValueBuffer = mergedRotatingPolarWHTDecodedValueBuffer(
                    decodedChunk: decodedChunk,
                    values: values,
                    previousOffset: previousOffset
                )
            } else {
                polarWHTDecodedValueBuffer = nil
            }
            return true
        } catch {
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            return false
        }
    }

    @discardableResult
    private func updateRotatingHybridAffineKeyAndPolarWHTValueTailSidecars(
        keys: MLXArray,
        values: MLXArray
    ) -> Bool {
        guard shouldMaintainHybridAffineKeySidecar, shouldMaintainPolarWHTValueSidecar else {
            return false
        }
        guard supportsMLXAffineQuantization(dimension: keys.dim(3), groupSize: groupSize) else {
            lastUnsupportedShape =
                "hybrid affine K tail sidecar requires key dimension \(keys.dim(3)) to be divisible by group size \(groupSize)"
            return false
        }
        let keys = keys.contiguous(stream: .gpu)
        let values = values.contiguous(stream: .gpu)
        let tokenCount = keys.dim(2)
        guard tokenCount > 0 else { return true }
        guard values.dim(0) == keys.dim(0),
            values.dim(1) == keys.dim(1),
            values.dim(2) == tokenCount,
            values.dim(3) == keys.dim(3)
        else {
            lastUnsupportedShape =
                "hybrid affine K tail sidecar shape mismatch: keys=\(keys.shape) values=\(values.shape)"
            return false
        }

        let previousTailOffset = hybridAffineKeyTailSidecar?.logicalLength ?? 0
        let logicalTailLength = previousTailOffset + tokenCount
        let capacity = max(
            hybridAffineKeyTailSidecar?.capacity ?? 0,
            polarWHTValueTailCode?.layout.capacity ?? 0,
            roundedRotatingPolarWHTTailCapacity(requiredLength: logicalTailLength)
        )
        let valueSeed = seed ^ turboQuantValueSeedSalt

        do {
            let fused = try MLX.turboQuantMetalHybridAffineK8PolarWHTValueEncode(
                keys: keys,
                values: values,
                keyGroupSize: groupSize,
                valueBits: valueBits,
                valueSeed: valueSeed,
                capacity: tokenCount,
                logicalLength: tokenCount
            )

            if previousTailOffset == 0 {
                let tailCapacity = max(capacity, logicalTailLength)
                hybridAffineKeyTailSidecar = TurboQuantAffineKeySidecar(
                    packed: expandedHybridAffineKeyStorage(
                        from: fused.key,
                        logicalLength: tokenCount,
                        capacity: tailCapacity
                    ),
                    logicalLength: logicalTailLength
                )
                var tailValue = try expandedRotatingPolarWHTValueCode(
                    fused.value,
                    capacity: tailCapacity
                )
                tailValue.layout.logicalLength = logicalTailLength
                tailValue.layout.ringOffset = 0
                tailValue.layout.pinnedPrefixLength = 0
                polarWHTValueTailCode = tailValue
                polarWHTDecodedValueBuffer = nil
                lastUnsupportedShape = nil
                return true
            }

            guard var currentKey = hybridAffineKeyTailSidecar else {
                lastUnsupportedShape = "hybrid affine K tail sidecar missing for append"
                return false
            }
            let requiredCapacity = max(capacity, logicalTailLength)
            if currentKey.capacity < requiredCapacity {
                currentKey.packed = expandedHybridAffineKeyStorage(
                    from: currentKey.packed,
                    logicalLength: currentKey.logicalLength,
                    capacity: requiredCapacity
                )
            }
            guard logicalTailLength <= currentKey.capacity,
                var currentValue = try ensureRotatingPolarWHTValueTailStorage(
                    values: values,
                    capacity: requiredCapacity
                )
            else {
                lastUnsupportedShape =
                    "hybrid affine K tail sidecar append exceeded capacity for logical length \(logicalTailLength)"
                return false
            }

            let destination = previousTailOffset ..< logicalTailLength
            let source = 0 ..< tokenCount
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            currentKey.packed.weight[.ellipsis, destination, 0...] =
                fused.key.weight[.ellipsis, source, 0...]
            currentKey.packed.scales[.ellipsis, destination, 0...] =
                fused.key.scales[.ellipsis, source, 0...]
            if let fusedBiases = fused.key.biases {
                currentKey.packed.biases?[.ellipsis, destination, 0...] =
                    fusedBiases[.ellipsis, source, 0...]
            }
            currentKey.logicalLength = logicalTailLength

            currentValue.packedIndices[.ellipsis, destination, 0...] =
                fused.value.packedIndices[.ellipsis, source, 0...]
            currentValue.norms[.ellipsis, destination] = fused.value.norms[.ellipsis, source]
            currentValue.layout.logicalLength = logicalTailLength
            currentValue.layout.ringOffset = 0
            currentValue.layout.pinnedPrefixLength = 0

            hybridAffineKeyTailSidecar = currentKey
            polarWHTValueTailCode = currentValue
            polarWHTDecodedValueBuffer = nil
            lastUnsupportedShape = nil
            return true
        } catch {
            hybridAffineKeyTailSidecar = nil
            polarWHTValueTailCode = nil
            polarWHTDecodedValueBuffer = nil
            lastUnsupportedShape = "rotating hybrid affine K/PolarWHT V tail update failed: \(error)"
            return false
        }
    }

    fileprivate func updateRotatingPolarWHTValueSidecar(
        values: MLXArray,
        previousOffset: Int
    ) {
        guard shouldMaintainPolarWHTValueSidecar else {
            polarWHTValueCode = nil
            polarWHTValueTailCode = nil
            polarWHTDecodedValueBuffer = nil
            return
        }
        let values = values.contiguous(stream: .gpu)
        let tokenCount = values.dim(2)
        guard tokenCount > 0 else { return }

        do {
            let encodedChunk = try turboQuantEncodePolarWHTAttentionValues(
                values,
                bits: valueBits,
                seed: seed ^ turboQuantValueSeedSalt,
                capacity: tokenCount,
                logicalLength: tokenCount
            )
            guard var current = try ensureRotatingPolarWHTValueStorage(
                values: values,
                previousOffset: previousOffset
            ) else {
                return
            }
            polarWHTValueCode = nil
            for token in 0 ..< tokenCount {
                let physical = physicalSlot(forAbsoluteToken: previousOffset + token)
                let source = token ..< (token + 1)
                let target = physical ..< (physical + 1)
                current.packedIndices[.ellipsis, target, 0...] =
                    encodedChunk.packedIndices[.ellipsis, source, 0...]
                current.norms[.ellipsis, target] = encodedChunk.norms[.ellipsis, source]
            }
            updateRotatingPolarWHTValueLayout(
                &current,
                offset: previousOffset + tokenCount
            )
            polarWHTValueCode = current
            polarWHTValueTailCode = nil
            if shouldMaintainPolarWHTDecodedValueBuffer {
                let decodedChunk = try MLX.turboQuantMetalPolarWHTDecodeAttentionValues(
                    encodedChunk,
                    outputDType: values.dtype
                )
                polarWHTDecodedValueBuffer = mergedRotatingPolarWHTDecodedValueBuffer(
                    decodedChunk: decodedChunk,
                    values: values,
                    previousOffset: previousOffset
                )
            } else {
                polarWHTDecodedValueBuffer = nil
            }
        } catch {
            polarWHTValueCode = nil
            polarWHTDecodedValueBuffer = nil
            lastUnsupportedShape = "rotating PolarWHT value sidecar update failed: \(error)"
        }
    }

    private func placeholderCode(for array: MLXArray, role: TurboQuantTensorRole) throws
        -> TurboQuantAttentionCode
    {
        let logicalLength = min(offset + array.dim(2), maxCacheSize)
        let layout = try MLX.turboQuantAttentionLayout(
            for: array,
            preset: preset,
            role: role,
            groupSize: groupSize,
            valueBits: valueBits,
            capacity: maxCacheSize,
            logicalLength: logicalLength,
            ringOffset: ringOffset(forOffset: offset + array.dim(2)),
            pinnedPrefixLength: pinnedPrefixLength(forLogicalLength: logicalLength)
        )
        return try MLX.turboQuantEmptyAttentionCode(
            layout: layout,
            preset: preset,
            role: role,
            groupSize: groupSize,
            seed: role == .value ? seed ^ turboQuantValueSeedSalt : seed,
            valueBits: valueBits
        )
    }

    private func ensureCompressedStorage(keys: MLXArray, values: MLXArray) throws {
        if compressedKeys != nil, compressedValues != nil { return }
        if offset > 0, rawFallbackCache != nil {
            throw TurboQuantCacheError.compressedBackfillUnavailable(
                "rotating raw fallback cache already owns \(offset) previous tokens"
            )
        }
        let logicalLength = min(offset, maxCacheSize)
        let pinnedPrefixLength = pinnedPrefixLength(forLogicalLength: logicalLength)
        let keyLayout = try MLX.turboQuantAttentionLayout(
            for: keys,
            preset: preset,
            groupSize: groupSize,
            capacity: maxCacheSize,
            logicalLength: logicalLength,
            ringOffset: ringOffset(forOffset: offset),
            pinnedPrefixLength: pinnedPrefixLength
        )
        let valueLayout = try MLX.turboQuantAttentionLayout(
            for: values,
            preset: preset,
            role: .value,
            groupSize: groupSize,
            valueBits: valueBits,
            capacity: maxCacheSize,
            logicalLength: logicalLength,
            ringOffset: ringOffset(forOffset: offset),
            pinnedPrefixLength: pinnedPrefixLength
        )
        compressedKeys = try MLX.turboQuantEmptyAttentionCode(
            layout: keyLayout,
            preset: preset,
            role: .key,
            groupSize: groupSize,
            seed: seed
        )
        compressedValues =
            usesPolarWHTValueOnlyStorage
            ? turboQuantCompactValuePlaceholderCode(
                layout: valueLayout,
                preset: preset,
                groupSize: groupSize,
                seed: seed ^ turboQuantValueSeedSalt,
                valueBits: valueBits
            )
            : try MLX.turboQuantEmptyAttentionCode(
                layout: valueLayout,
                preset: preset,
                role: .value,
                groupSize: groupSize,
                seed: seed ^ turboQuantValueSeedSalt,
                valueBits: valueBits
            )
    }

    private func updateCompressedLayouts(
        keys: inout TurboQuantAttentionCode,
        values: inout TurboQuantAttentionCode
    ) {
        let logicalLength = min(offset, maxCacheSize)
        let ringOffset = ringOffset(forOffset: offset)
        let pinnedPrefixLength = pinnedPrefixLength(forLogicalLength: logicalLength)
        keys.layout.logicalLength = logicalLength
        keys.layout.ringOffset = ringOffset
        keys.layout.pinnedPrefixLength = pinnedPrefixLength
        values.layout.logicalLength = logicalLength
        values.layout.ringOffset = ringOffset
        values.layout.pinnedPrefixLength = pinnedPrefixLength
    }

    private func physicalSlot(forAbsoluteToken absolute: Int) -> Int {
        if absolute < keep { return absolute }
        let ringCapacity = max(1, maxCacheSize - keep)
        return keep + ((absolute - keep) % ringCapacity)
    }

    private func ringOffset(forOffset offset: Int) -> Int {
        let pinned = min(keep, maxCacheSize)
        let ringCapacity = maxCacheSize - pinned
        guard ringCapacity > 0, offset > pinned else { return 0 }
        let activeRing = min(offset - pinned, ringCapacity)
        return (offset - pinned - activeRing) % ringCapacity
    }

    private func nextWriteIndex(afterOffset offset: Int) -> Int {
        if offset < keep { return offset }
        let ringCapacity = max(1, maxCacheSize - keep)
        return keep + ((offset - keep) % ringCapacity)
    }

    private func rawFallbackWriteIndex(forOffset offset: Int) -> Int {
        if offset <= maxCacheSize { return min(offset, maxCacheSize) }
        return nextWriteIndex(afterOffset: offset)
    }

    private func rotatingCacheParameters(from metaState: [String]?) -> (
        keep: Int, maxSize: Int, step: Int
    ) {
        guard let metaState, metaState.count >= 3 else {
            return (keep, maxCacheSize, step)
        }
        return (
            Int(metaState[0]) ?? keep,
            Int(metaState[1]) ?? maxCacheSize,
            Int(metaState[2]) ?? step
        )
    }

    private func restoreRawFallbackCache(state: [MLXArray]?, metaState: [String]?) {
        guard let state else {
            rawFallbackCache = nil
            return
        }
        let parameters = rotatingCacheParameters(from: metaState)
        let cache = RotatingKVCache(
            maxSize: parameters.maxSize,
            keep: parameters.keep,
            step: parameters.step
        )
        cache.state = state
        if let metaState {
            cache.metaState = metaState
        }
        rawFallbackCache = cache
    }

    private func restorePackedFallbackCache(state: [MLXArray]?, metaState: [String]?) {
        guard let state else {
            packedFallbackCache = nil
            return
        }
        let parameters = rotatingCacheParameters(from: metaState)
        let cache = RotatingQuantizedKVCache(
            maxSize: parameters.maxSize,
            keep: parameters.keep,
            step: parameters.step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        cache.state = state
        if let metaState {
            cache.metaState = metaState
        }
        packedFallbackCache = cache
    }

    private func materializedRawFallbackCache() -> RotatingKVCache {
        if let rawFallbackCache, rawFallbackCache.maxSize == maxCacheSize {
            return rawFallbackCache
        }
        let rawCache = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        if let compressedKeys, let compressedValues {
            do {
                let decodedKeys = try MLX.turboQuantMetalDecodeAttention(
                    compressedKeys,
                    outputDType: .float16  // 4.1: fp16 scratch halves the fallback materialization spike
                )
                let decodedValues = try MLX.turboQuantMetalDecodeAttention(
                    compressedValues,
                    outputDType: .float16  // 4.1: fp16 scratch halves the fallback materialization spike
                )
                rawCache.state = [decodedKeys, decodedValues]
            } catch {
                lastUnsupportedShape = "failed to materialize raw fallback: \(error)"
            }
        }
        rawCache.metaState = [
            String(keep),
            String(maxCacheSize),
            String(step),
            String(offset),
            String(rawFallbackWriteIndex(forOffset: offset)),
        ]
        rawFallbackCache = rawCache
        return rawCache
    }

    private func materializedExactRawShadowCache() -> RotatingKVCache {
        if let rawFallbackCache,
           rawFallbackCache.maxSize == exactRawShadowMaxSize,
           rawFallbackCache.offset == offset {
            return rawFallbackCache
        }
        let rawCache = RotatingKVCache(
            maxSize: exactRawShadowMaxSize,
            keep: min(keep, exactRawShadowMaxSize),
            step: min(step, max(1, exactRawShadowMaxSize))
        )
        rawCache.metaState = [
            String(min(keep, exactRawShadowMaxSize)),
            String(exactRawShadowMaxSize),
            String(min(step, max(1, exactRawShadowMaxSize))),
            String(offset),
            String(min(offset, exactRawShadowMaxSize)),
        ]
        rawFallbackCache = rawCache
        return rawCache
    }

    private func materializedPackedFallbackCache() -> RotatingQuantizedKVCache {
        if let packedFallbackCache { return packedFallbackCache }
        let cache = RotatingQuantizedKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        if let rawFallbackCache {
            let state = rawFallbackCache.state
            if state.count == 2 {
                cache.setUnquantizedState(
                    keys: state[0],
                    values: state[1],
                    offset: rawFallbackCache.offset,
                    writeIndex: currentWriteIndexFromRawMeta(rawFallbackCache.metaState)
                )
            }
        } else if let compressedKeys, let compressedValues,
            turboQuantSupportsPackedFallback(
                keyCode: compressedKeys,
                valueCode: compressedValues,
                groupSize: groupSize
            ),
            let decodedKeys = try? MLX.turboQuantMetalDecodeAttention(
                compressedKeys,
                outputDType: .float16  // 4.1: fp16 scratch halves the fallback materialization spike
            ),
            let decodedValues = try? MLX.turboQuantMetalDecodeAttention(
                compressedValues,
                outputDType: .float16  // 4.1: fp16 scratch halves the fallback materialization spike
            )
        {
            cache.setUnquantizedState(
                keys: decodedKeys,
                values: decodedValues,
                offset: offset,
                writeIndex: rawFallbackWriteIndex(forOffset: offset)
            )
        }
        packedFallbackCache = cache
        return cache
    }

    fileprivate func setPackedFallbackState(
        keys: MLXArray,
        values: MLXArray,
        offset: Int,
        writeIndex: Int
    ) {
        let cache = RotatingQuantizedKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        cache.setUnquantizedState(
            keys: keys,
            values: values,
            offset: offset,
            writeIndex: writeIndex
        )
        packedFallbackCache = cache
        packedKeys = nil
        packedValues = nil
    }

    private func currentWriteIndexFromRawMeta(_ meta: [String]) -> Int {
        guard meta.count >= 5 else { return nextWriteIndex(afterOffset: offset) }
        return Int(meta[4]) ?? nextWriteIndex(afterOffset: offset)
    }
}

extension RotatingKVCache {
    public func toTurboQuant(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        kvCodec: TurboQuantKVCodec? = nil,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueSelection: TurboQuantSparseValueSelection = .off,
        layerIndex: Int? = nil,
        residentBudgetBytes: Int? = nil
    ) -> RotatingTurboQuantKVCache {
        let resolvedKVCodec = turboQuantCompressedKVCodec(
            requested: kvCodec,
            backend: backend
        )
        let resolvedValueBits = turboQuantDefaultValueBits(
            preset: preset,
            kvCodec: resolvedKVCodec,
            requestedValueBits: valueBits
        )
        let capacity = maxSize ?? rotatingStep
        let cache = RotatingTurboQuantKVCache(
            maxSize: capacity,
            keep: rotatingKeep,
            step: rotatingStep,
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend,
            kvCodec: resolvedKVCodec,
            optimizationPolicy: optimizationPolicy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: resolvedValueBits,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            layerIndex: layerIndex,
            residentBudgetBytes: residentBudgetBytes
        )
        cache.offset = offset
        cache.metaState = metaState

        let currentState = state
        guard currentState.count == 2 else { return cache }
        cache.setPackedFallbackState(
            keys: currentState[0],
            values: currentState[1],
            offset: offset,
            writeIndex: rotatingWriteIndex
        )

        let availability = TurboQuantKernelAvailability.current
        let compressedBackendAvailable =
            cache.activeBackend == .metalPolarQJL
            ? availability.supportsMetalPolarQJLAttention
            : availability.supportsMetalPolarWHTCodec
        if turboQuantIsMetalCompressedBackend(cache.activeBackend),
            compressedBackendAvailable
        {
            do {
                let keyConfiguration = TurboQuantConfiguration(
                    preset: preset,
                    role: .key,
                    groupSize: groupSize,
                    mode: mode,
                    backend: cache.activeBackend,
                    seed: seed
                )
                let valueConfiguration = TurboQuantConfiguration(
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    mode: mode,
                    backend: cache.activeBackend,
                    seed: seed ^ turboQuantValueSeedSalt,
                    valueBits: resolvedValueBits
                )
                let keys = turboQuantPhysicalizedState(currentState[0], maxSize: capacity)
                let values = turboQuantPhysicalizedState(currentState[1], maxSize: capacity)
                let logicalLength = min(offset, capacity)
                let pinnedPrefixLength = min(rotatingKeep, capacity, max(0, logicalLength))
                let keyCode = try MLX.turboQuantMetalEncodeAttention(
                    keys,
                    configuration: keyConfiguration,
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: rotatingRingOffset,
                    pinnedPrefixLength: pinnedPrefixLength
                )
                let valueCode: TurboQuantAttentionCode
                if cache.usesPolarWHTValueOnlyStorage {
                    let valueLayout = try MLX.turboQuantAttentionLayout(
                        for: values,
                        preset: cache.preset,
                        role: .value,
                        groupSize: groupSize,
                        valueBits: resolvedValueBits,
                        capacity: capacity,
                        logicalLength: logicalLength,
                        ringOffset: rotatingRingOffset,
                        pinnedPrefixLength: pinnedPrefixLength
                    )
                    valueCode = turboQuantCompactValuePlaceholderCode(
                        layout: valueLayout,
                        preset: cache.preset,
                        groupSize: groupSize,
                        seed: seed ^ turboQuantValueSeedSalt,
                        valueBits: resolvedValueBits
                    )
                } else {
                    valueCode = try MLX.turboQuantMetalEncodeAttention(
                        values,
                        configuration: valueConfiguration,
                        capacity: capacity,
                        logicalLength: logicalLength,
                        ringOffset: rotatingRingOffset,
                        pinnedPrefixLength: pinnedPrefixLength
                    )
                }
                cache.state = [
                    keyCode.packedMagnitudes,
                    keyCode.signs,
                    keyCode.highPrecisionMask,
                    keyCode.residualSigns,
                    keyCode.scales,
                    valueCode.packedMagnitudes,
                    valueCode.signs,
                    valueCode.highPrecisionMask,
                    valueCode.residualSigns,
                    valueCode.scales,
                ]
                if cache.usesPolarWHTValueOnlyStorage {
                    cache.packedKeys = turboQuantized(currentState[0], configuration: keyConfiguration)
                    cache.packedValues = nil
                }
                cache.updateRotatingPolarWHTKeySidecar(
                    keys: keys,
                    previousOffset: 0
                )
                cache.updateRotatingPolarWHTValueSidecar(
                    values: values,
                    previousOffset: 0
                )
                cache.metaState = metaState
                return cache
            } catch {
                cache.recordCompressedAttentionFailure(String(describing: error))
            }
        }

        cache.metaState = metaState
        cache.state = currentState
        cache.metaState = metaState
        return cache
    }

    private func turboQuantPhysicalizedState(_ array: MLXArray, maxSize: Int) -> MLXArray {
        let length = array.dim(2)
        guard length > maxSize else { return array }

        let pinned = min(rotatingKeep, maxSize)
        let ringCapacity = maxSize - pinned
        let physical = MLXArray.zeros(
            [array.dim(0), array.dim(1), maxSize, array.dim(3)],
            dtype: array.dtype
        )

        if pinned > 0 {
            let prefixLength = min(pinned, length)
            physical[.ellipsis, 0 ..< prefixLength, 0...] =
                array[.ellipsis, 0 ..< prefixLength, 0...]
        }

        guard ringCapacity > 0 else { return physical }

        let ringCount = min(ringCapacity, max(0, length - pinned))
        let sourceStart = length - ringCount
        for logical in 0 ..< ringCount {
            let source = sourceStart + logical
            let target = pinned + ((rotatingRingOffset + logical) % ringCapacity)
            physical[.ellipsis, target ..< (target + 1), 0...] =
                array[.ellipsis, source ..< (source + 1), 0...]
        }
        return physical
    }
}

extension KVCacheSimple {
    public func toTurboQuant(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        kvCodec: TurboQuantKVCodec? = nil,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueSelection: TurboQuantSparseValueSelection = .off,
        layerIndex: Int? = nil,
        residentBudgetBytes: Int? = nil
    ) -> TurboQuantKVCache {
        let resolvedKVCodec = turboQuantCompressedKVCodec(
            requested: kvCodec,
            backend: backend
        )
        let resolvedValueBits = turboQuantDefaultValueBits(
            preset: preset,
            kvCodec: resolvedKVCodec,
            requestedValueBits: valueBits
        )
        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend,
            kvCodec: resolvedKVCodec,
            optimizationPolicy: optimizationPolicy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: resolvedValueBits,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            layerIndex: layerIndex,
            residentBudgetBytes: residentBudgetBytes
        )
        cache.offset = self.offset

        let currentState = self.state
        if currentState.count == 2 {
            let keyConfiguration = TurboQuantConfiguration(
                preset: preset,
                role: .key,
                groupSize: groupSize,
                mode: mode,
                backend: cache.activeBackend,
                seed: seed
            )
            let valueConfiguration = TurboQuantConfiguration(
                preset: preset,
                role: .value,
                groupSize: groupSize,
                mode: mode,
                backend: cache.activeBackend,
                seed: seed ^ turboQuantValueSeedSalt,
                valueBits: resolvedValueBits
            )
            let keys = turboQuantized(currentState[0], configuration: keyConfiguration)
            let values = turboQuantized(currentState[1], configuration: valueConfiguration)
            cache.state = [
                keys.weight, keys.scales, keys.biases,
                values.weight, values.scales, values.biases,
            ].compactMap { $0 }
            cache.updatePolarWHTKeySidecar(
                keys: currentState[0],
                previousOffset: 0
            )
            cache.updatePolarWHTValueSidecar(
                values: currentState[1],
                previousOffset: 0
            )
            let availability = TurboQuantKernelAvailability.current
            let compressedBackendAvailable =
                cache.activeBackend == .metalPolarQJL
                ? availability.supportsMetalPolarQJLAttention
                : availability.supportsMetalPolarWHTCodec
            if turboQuantIsMetalCompressedBackend(cache.activeBackend),
                compressedBackendAvailable
            {
                do {
                    let keyCode = try MLX.turboQuantMetalEncodeAttention(
                        currentState[0],
                        configuration: keyConfiguration,
                        logicalLength: self.offset
                    )
                    let valueCode: TurboQuantAttentionCode
                    if cache.usesPolarWHTValueOnlyStorage {
                        let valueLayout = try MLX.turboQuantAttentionLayout(
                            for: currentState[1],
                            preset: cache.preset,
                            role: .value,
                            groupSize: groupSize,
                            valueBits: resolvedValueBits,
                            logicalLength: self.offset
                        )
                        valueCode = turboQuantCompactValuePlaceholderCode(
                            layout: valueLayout,
                            preset: cache.preset,
                            groupSize: groupSize,
                            seed: seed ^ turboQuantValueSeedSalt,
                            valueBits: resolvedValueBits
                        )
                    } else {
                        valueCode = try MLX.turboQuantMetalEncodeAttention(
                            currentState[1],
                            configuration: valueConfiguration,
                            logicalLength: self.offset
                        )
                    }
                    cache.state = [
                        keyCode.packedMagnitudes,
                        keyCode.signs,
                        keyCode.highPrecisionMask,
                        keyCode.residualSigns,
                        keyCode.scales,
                        valueCode.packedMagnitudes,
                        valueCode.signs,
                        valueCode.highPrecisionMask,
                        valueCode.residualSigns,
                        valueCode.scales,
                    ]
                    if cache.usesPolarWHTValueOnlyStorage {
                        cache.hybridAffineKeySidecar = TurboQuantAffineKeySidecar(
                            packed: expandedHybridAffineKeyStorage(
                                from: keys,
                                logicalLength: self.offset,
                                capacity: keyCode.layout.capacity
                            ),
                            logicalLength: self.offset
                        )
                    }
                    cache.updatePolarWHTKeySidecar(
                        keys: currentState[0],
                        previousOffset: 0,
                        capacity: keyCode.layout.capacity
                    )
                    cache.updatePolarWHTValueSidecar(
                        values: currentState[1],
                        previousOffset: 0,
                        capacity: keyCode.layout.capacity
                    )
                } catch {
                    cache.recordCompressedAttentionFailure(String(describing: error))
                }
            }
        }

        return cache
    }
}
