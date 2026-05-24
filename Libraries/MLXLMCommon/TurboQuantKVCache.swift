// Copyright © 2026 RNT56.

import Foundation
import MLX

public typealias TurboQuantPreset = MLX.TurboQuantPreset
public typealias TurboQuantBackend = MLX.TurboQuantBackend
public typealias TurboQuantKernelAvailability = MLX.TurboQuantKernelAvailability
public typealias TurboQuantAttentionCode = MLX.TurboQuantAttentionCode
public typealias TurboQuantAttentionPath = MLX.TurboQuantAttentionPath
public typealias TurboQuantKernelProfile = MLX.TurboQuantKernelProfile
public typealias TurboQuantDeviceCapabilities = MLX.TurboQuantDeviceCapabilities
public typealias TurboQuantRuntimeProbeResult = MLX.TurboQuantRuntimeProbeResult
public typealias TurboQuantRuntimeSelfTestStatus = MLX.TurboQuantRuntimeSelfTestStatus

let defaultTurboQuantSeed: UInt64 = 0x9E37_79B9_7F4A_7C15
private let turboQuantValueSeedSalt: UInt64 = 0xD1B5_4A32_D192_ED03

public enum TurboQuantFallbackPath: String, Equatable, Codable, Sendable, CaseIterable {
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
    case turboQuant
}

public enum TurboQuantOptimizationPolicy: String, Codable, Sendable, CaseIterable {
    case auto
    case conservative
    case preferMemory
    case preferThroughput
}

public struct TurboQuantAttentionDiagnostics: Equatable, Codable, Sendable {
    public var metalAttentionAvailable: Bool
    public var activeAttentionPath: TurboQuantAttentionPath
    public var selectedKernelProfile: TurboQuantKernelProfile
    public var selfTestStatus: TurboQuantRuntimeSelfTestStatus
    public var selfTestFailureReason: String?
    public var optimizationPolicy: TurboQuantOptimizationPolicy
    public var fallbackReason: String?
    public var lastUnsupportedShape: String?
    public var rawFallbackAllocated: Bool
    public var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public var lastFallback: TurboQuantFallbackResult?
}

public struct TurboQuantKVCacheDiagnostics: Equatable, Codable, Sendable {
    public var preset: TurboQuantPreset
    public var requestedBackend: TurboQuantBackend
    public var activeBackend: TurboQuantBackend
    public var fallbackReason: String?
    public var metalCodecAvailable: Bool
    public var metalAttentionAvailable: Bool
    public var activeAttentionPath: TurboQuantAttentionPath
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
}

public protocol TurboQuantCompressedKVCacheProtocol: KVCache, AnyObject {
    var preset: TurboQuantPreset { get }
    var requestedBackend: TurboQuantBackend { get }
    var activeBackend: TurboQuantBackend { get }
    var optimizationPolicy: TurboQuantOptimizationPolicy { get }
    var attentionDiagnostics: TurboQuantAttentionDiagnostics { get }
    var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? { get }
    var cacheLifecycle: TurboQuantCacheLifecycle { get }
    var fallbackResults: [TurboQuantFallbackResult] { get }
    var cacheFootprint: TurboQuantRuntimeCacheFootprint { get }

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

    func recordCompressedAttentionFailure(_ message: String)
    func recordFallback(_ result: TurboQuantFallbackResult)
    func validateCompressedState(context: String) throws
    func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray)
    func releaseRawShadow()
}

extension TurboQuantCompressedKVCacheProtocol {
    public var prefersOnlineFusedAttention: Bool {
        optimizationPolicy != .conservative
    }

    public var prefersExactInitialPrefill: Bool {
        switch optimizationPolicy {
        case .auto, .conservative:
            true
        case .preferMemory, .preferThroughput:
            false
        }
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
    return keyCode.layout.headDimension.isMultiple(of: groupSize)
        && valueCode.layout.headDimension.isMultiple(of: groupSize)
}

private enum TurboQuantCacheError: Error, CustomStringConvertible {
    case compressedBackfillUnavailable(String)
    case compressedStorageInvalid(String)

    var description: String {
        switch self {
        case .compressedBackfillUnavailable(let message):
            "TurboQuant compressed cache backfill unavailable: \(message)"
        case .compressedStorageInvalid(let message):
            "TurboQuant compressed cache storage invalid: \(message)"
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
        || array.shape == turboQuantStorageShape(
            code,
            wordsPerGroup: code.layout.bitsetWordsPerGroup
        )
}

private func validateTurboQuantCode(_ code: TurboQuantAttentionCode, context: String) throws {
    let layout = code.layout
    guard layout.layoutVersion == TurboQuantAttentionLayout.currentVersion else {
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
            code.highPrecisionMask.dtype == .uint32, code.highPrecisionMask.shape == bitsetShape,
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

public final class TurboQuantKVCache: QuantizedKVCache, TurboQuantCompressedKVCacheProtocol {
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var compressedStep: Int = 256
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?
    private var restoredLayoutMetadata: RestoredAttentionLayoutMetadata?
    private var lastDecodedTransientBytes: Int = 0
    public private(set) var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public private(set) var fallbackResults: [TurboQuantFallbackResult] = []

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let seed: UInt64
    public let valueBits: Int

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil
    ) {
        self.preset = preset
        self.requestedBackend = backend
        self.optimizationPolicy = optimizationPolicy
        self.seed = seed
        self.valueBits = valueBits ?? preset.defaultValueBits
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        super.init(groupSize: groupSize, bits: preset.effectiveBits, mode: mode)
    }

    public override var metaState: [String] {
        get {
            var meta =
                super.metaState + [
                    preset.rawValue,
                    requestedBackend.rawValue,
                    String(seed),
                    "valueBits=\(valueBits)",
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
            if activeBackend == .metalPolarQJL,
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
            if activeBackend == .metalPolarQJL, offset == 0 {
                return []
            }
            return super.state
        }
        set {
            if activeBackend == .metalPolarQJL, newValue.isEmpty {
                compressedKeys = nil
                compressedValues = nil
                lastDecodedTransientBytes = 0
                cacheLifecycle = .empty
                return
            }
            if activeBackend == .metalPolarQJL, newValue.count == 10 {
                let capacity = restoredLayoutMetadata?.capacity ?? newValue[0].dim(2)
                let keyHeadDimension =
                    restoredLayoutMetadata?.keyHeadDimension
                    ?? max(groupSize, (newValue[0].dim(3) * groupSize))
                let valueHeadDimension =
                    restoredLayoutMetadata?.valueHeadDimension
                    ?? max(groupSize, (newValue[5].dim(3) * groupSize))
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
                let valueLayout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[5].dim(0),
                    kvHeadCount: restoredLayoutMetadata?.kvHeadCount ?? newValue[5].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: restoredLayoutMetadata?.ringOffset ?? 0,
                    pinnedPrefixLength: restoredLayoutMetadata?.pinnedPrefixLength ?? 0,
                    headDimension: valueHeadDimension,
                    groupsPerVector: newValue[5].dim(3),
                    magnitudeWordsPerGroup: newValue[5].dim(4),
                    bitsetWordsPerGroup: newValue[1].dim(4)
                )
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
                    scalesPerGroup: newValue[9].dim(4),
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
                if let compressedKeys {
                    cacheLifecycle = .compressedCommitted(
                        logicalLength: compressedKeys.layout.logicalLength,
                        capacity: compressedKeys.layout.capacity
                    )
                }
            } else if newValue.count == 10 {
                lastUnsupportedShape =
                    "compressed TurboQuant state restored without Metal attention support"
                cacheLifecycle = .failed(reason: lastUnsupportedShape ?? "unsupported compressed state")
            } else {
                super.state = newValue
            }
        }
    }

    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        guard let compressedKeys, let compressedValues else { return nil }
        return (compressedKeys, compressedValues)
    }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let compressedBytes: Int
        let logicalLength: Int
        let capacity: Int
        if let compressedKeys, let compressedValues {
            compressedBytes = turboQuantCodeBytes(compressedKeys) + turboQuantCodeBytes(compressedValues)
            logicalLength = compressedKeys.layout.logicalLength
            capacity = compressedKeys.layout.capacity
        } else {
            compressedBytes = 0
            logicalLength = offset
            capacity = 0
        }
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: logicalLength,
            capacity: capacity,
            compressedBytes: compressedBytes,
            packedFallbackBytes: turboQuantArrayBytes(super.state),
            decodedTransientBytes: lastDecodedTransientBytes,
            lifecycle: cacheLifecycle
        )
    }

    public func recordFallback(_ result: TurboQuantFallbackResult) {
        fallbackResults.append(result)
        lastUnsupportedShape = result.reason
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

    public func validateCompressedState(context: String) throws {
        guard let compressedKeys, let compressedValues else {
            if offset == 0 { return }
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): no compressed state exists for offset \(offset)"
            )
        }
        try validateTurboQuantPair(keys: compressedKeys, values: compressedValues, context: context)
    }

    public func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray) {
        try validateCompressedState(context: "decode compressed state")
        guard let compressedKeys, let compressedValues else {
            throw TurboQuantCacheError.compressedStorageInvalid("decode compressed state missing")
        }
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
                outputDType: .float32
            ),
            let decodedValues = try? MLX.turboQuantMetalDecodeAttention(
                compressedValues,
                outputDType: .float32
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
        return TurboQuantAttentionDiagnostics(
            metalAttentionAvailable: availability.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: optimizationPolicy,
            fallbackReason: backendFallbackReason,
            lastUnsupportedShape: lastUnsupportedShape,
            rawFallbackAllocated: false,
            cacheLifecycle: cacheLifecycle,
            lastFallback: fallbackResults.last
        )
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        return TurboQuantKVCacheDiagnostics(
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: backendFallbackReason,
            metalCodecAvailable: availability.supportsMetalPolarQJLCodec,
            metalAttentionAvailable: availability.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
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
            footprint: cacheFootprint
        )
    }

    public func supportsCompressedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        guard activeBackend == .metalPolarQJL,
            TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention
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
        let supportsTiled =
            queries.dim(3) == values.dim(3) && prefersOnlineFusedAttention
            && MLX.turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: mask
            )
        lastAttentionPath = supportsTiled ? .tiledOnlineFused : .twoStageCompressed
        lastUnsupportedShape = supportsTiled
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
        let encodedKeys = try MLX.turboQuantMetalEncodeAttention(
            keys,
            configuration: keyConfiguration,
            stream: .gpu
        )
        let encodedValues = try MLX.turboQuantMetalEncodeAttention(
            values,
            configuration: valueConfiguration,
            stream: .gpu
        )

        var currentKeys = compressedKeys!
        var currentValues = compressedValues!
        let range = previousOffset ..< (previousOffset + tokenCount)
        currentKeys.packedMagnitudes[.ellipsis, range, 0..., 0...] = encodedKeys.packedMagnitudes
        currentKeys.signs[.ellipsis, range, 0..., 0...] = encodedKeys.signs
        currentKeys.highPrecisionMask[.ellipsis, range, 0..., 0...] = encodedKeys.highPrecisionMask
        if currentKeys.residualSigns.ndim == 5 {
            currentKeys.residualSigns[.ellipsis, range, 0..., 0...] = encodedKeys.residualSigns
        }
        currentKeys.scales[.ellipsis, range, 0..., 0...] = encodedKeys.scales

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

        if turboQuantSupportsPackedFallback(keys: keys, values: values, groupSize: groupSize),
            super.getQuantizedState() != nil
        {
            _ = super.updateQuantized(keys: keys, values: values)
        } else {
            offset += tokenCount
        }
        currentKeys.layout.logicalLength = offset
        currentValues.layout.logicalLength = offset
        compressedKeys = currentKeys
        compressedValues = currentValues
        lastDecodedTransientBytes = 0
        try validateCompressedState(context: "compressed append")
        cacheLifecycle = .compressedCommitted(
            logicalLength: offset,
            capacity: currentKeys.layout.capacity
        )
        return (currentKeys, currentValues)
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        lastAttentionPath = .mlxPackedFallback
        lastUnsupportedShape = "compressed attention failed: \(message)"
        cacheLifecycle = .failed(reason: message)
    }

    public override func copy() -> any KVCache {
        let new = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend,
            optimizationPolicy: optimizationPolicy,
            seed: seed,
            valueBits: valueBits
        )
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
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
            compressedValues = try MLX.turboQuantEmptyAttentionCode(
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
        compressedValues = try MLX.turboQuantMetalEncodeAttention(
            decodedValues,
            configuration: valueConfiguration,
            capacity: capacity,
            logicalLength: offset
        )
    }

    private func expandCompressedCode(
        _ code: TurboQuantAttentionCode,
        newCapacity: Int
    ) throws -> TurboQuantAttentionCode {
        var newLayout = code.layout
        let extra = newCapacity - code.layout.capacity
        newLayout.capacity = newCapacity
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
        let expandedSigns =
            code.role == .value
            ? zeros.signs
            : concatenated([code.signs, zeros.signs], axis: 2)
        let expandedHighPrecisionMask =
            code.role == .value
            ? zeros.highPrecisionMask
            : concatenated([code.highPrecisionMask, zeros.highPrecisionMask], axis: 2)
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
            highPrecisionMask: expandedHighPrecisionMask,
            residualSigns: zeros.residualSigns,
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
            ? (code.highPrecisionMask.ndim == 5 && code.highPrecisionMask.dim(2) == capacity)
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
    private var packedKeys: TurboQuantPackedTensor?
    private var packedValues: TurboQuantPackedTensor?
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?
    private var restoredLayoutMetadata: RestoredAttentionLayoutMetadata?
    private var lastDecodedTransientBytes: Int = 0
    private let keep: Int
    private let step: Int
    private let maxCacheSize: Int
    private var writeIndex: Int
    public private(set) var cacheLifecycle: TurboQuantCacheLifecycle = .empty
    public private(set) var fallbackResults: [TurboQuantFallbackResult] = []

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let seed: UInt64
    public let valueBits: Int

    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { offset < maxCacheSize }

    private func pinnedPrefixLength(forLogicalLength logicalLength: Int) -> Int {
        min(keep, maxCacheSize, max(0, logicalLength))
    }

    public init(
        maxSize: Int,
        keep: Int = 4,
        step: Int = 256,
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil
    ) {
        self.keep = max(0, min(keep, maxSize))
        self.step = step
        self.maxCacheSize = maxSize
        self.writeIndex = self.keep
        self.preset = preset
        self.requestedBackend = backend
        self.optimizationPolicy = optimizationPolicy
        self.seed = seed
        self.valueBits = valueBits ?? preset.defaultValueBits
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        self.groupSize = groupSize
        self.bits = preset.effectiveBits
        self.mode = mode
        super.init()
        if self.activeBackend != .metalPolarQJL {
            self.rawFallbackCache = RotatingKVCache(maxSize: maxSize, keep: self.keep, step: step)
        }
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let rawCache = materializedRawFallbackCache()
        let (cachedKeys, cachedValues) = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset
        writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
        packedFallbackCache = nil

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
        return materializedPackedFallbackCache().getQuantizedState()
    }

    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        guard let compressedKeys, let compressedValues else { return nil }
        return (compressedKeys, compressedValues)
    }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let compressedBytes: Int
        let logicalLength: Int
        let capacity: Int
        if let compressedKeys, let compressedValues {
            compressedBytes = turboQuantCodeBytes(compressedKeys) + turboQuantCodeBytes(compressedValues)
            logicalLength = compressedKeys.layout.logicalLength
            capacity = compressedKeys.layout.capacity
        } else {
            compressedBytes = 0
            logicalLength = min(offset, maxCacheSize)
            capacity = maxCacheSize
        }
        let packedBytes =
            turboQuantArrayBytes(packedFallbackCache?.state ?? [])
            + turboQuantArrayBytes([packedKeys?.weight, packedKeys?.scales, packedKeys?.biases].compactMap { $0 })
            + turboQuantArrayBytes([packedValues?.weight, packedValues?.scales, packedValues?.biases].compactMap { $0 })
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: logicalLength,
            capacity: capacity,
            compressedBytes: compressedBytes,
            packedFallbackBytes: packedBytes,
            rawShadowBytes: turboQuantArrayBytes(rawFallbackCache?.state ?? []),
            decodedTransientBytes: lastDecodedTransientBytes,
            lifecycle: cacheLifecycle
        )
    }

    public func recordFallback(_ result: TurboQuantFallbackResult) {
        fallbackResults.append(result)
        lastUnsupportedShape = result.reason
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

    public func validateCompressedState(context: String) throws {
        guard let compressedKeys, let compressedValues else {
            if offset == 0 { return }
            throw TurboQuantCacheError.compressedStorageInvalid(
                "\(context): no compressed rotating state exists for offset \(offset)"
            )
        }
        try validateTurboQuantPair(keys: compressedKeys, values: compressedValues, context: context)
    }

    public func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray) {
        try validateCompressedState(context: "decode rotating compressed state")
        guard let compressedKeys, let compressedValues else {
            throw TurboQuantCacheError.compressedStorageInvalid(
                "decode rotating compressed state missing"
            )
        }
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
        return TurboQuantAttentionDiagnostics(
            metalAttentionAvailable: availability.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: optimizationPolicy,
            fallbackReason: backendFallbackReason,
            lastUnsupportedShape: lastUnsupportedShape,
            rawFallbackAllocated: rawFallbackCache != nil,
            cacheLifecycle: cacheLifecycle,
            lastFallback: fallbackResults.last
        )
    }

    public func supportsCompressedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        guard activeBackend == .metalPolarQJL,
            TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention
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
        let supportsTiled =
            queries.dim(3) == values.dim(3) && prefersOnlineFusedAttention
            && MLX.turboQuantMetalSupportsOnlineFusedAttention(
                queries: queries,
                keyCode: keyCode,
                mask: mask
            )
        lastAttentionPath = supportsTiled ? .tiledOnlineFused : .twoStageCompressed
        lastUnsupportedShape = supportsTiled
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
        let encodedKeys = try MLX.turboQuantMetalEncodeAttention(
            keys,
            configuration: keyConfiguration
        )
        let encodedValues = try MLX.turboQuantMetalEncodeAttention(
            values,
            configuration: valueConfiguration
        )
        try ensureCompressedStorage(keys: keys, values: values)

        var currentKeys = compressedKeys!
        var currentValues = compressedValues!
        let physicalStart = physicalSlot(forAbsoluteToken: offset)
        if tokenCount > 0, physicalStart + tokenCount <= maxCacheSize {
            let target = physicalStart ..< (physicalStart + tokenCount)
            currentKeys.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                encodedKeys.packedMagnitudes
            currentKeys.signs[.ellipsis, target, 0..., 0...] = encodedKeys.signs
            currentKeys.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                encodedKeys.highPrecisionMask
            if currentKeys.residualSigns.ndim == 5 {
                currentKeys.residualSigns[.ellipsis, target, 0..., 0...] =
                    encodedKeys.residualSigns
            }
            currentKeys.scales[.ellipsis, target, 0..., 0...] = encodedKeys.scales

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
        } else {
            for token in 0 ..< tokenCount {
                let physical = physicalSlot(forAbsoluteToken: offset + token)
                let source = token ..< (token + 1)
                let target = physical ..< (physical + 1)
                currentKeys.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                    encodedKeys.packedMagnitudes[.ellipsis, source, 0..., 0...]
                currentKeys.signs[.ellipsis, target, 0..., 0...] =
                    encodedKeys.signs[.ellipsis, source, 0..., 0...]
                currentKeys.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                    encodedKeys.highPrecisionMask[.ellipsis, source, 0..., 0...]
                if currentKeys.residualSigns.ndim == 5 {
                    currentKeys.residualSigns[.ellipsis, target, 0..., 0...] =
                        encodedKeys.residualSigns[.ellipsis, source, 0..., 0...]
                }
                currentKeys.scales[.ellipsis, target, 0..., 0...] =
                    encodedKeys.scales[.ellipsis, source, 0..., 0...]

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
        packedKeys = nil
        packedValues = nil
        lastDecodedTransientBytes = 0
        releaseRawShadow()
        try validateCompressedState(context: "rotating compressed append")
        cacheLifecycle = .compressedCommitted(
            logicalLength: currentKeys.layout.logicalLength,
            capacity: currentKeys.layout.capacity
        )
        return (currentKeys, currentValues)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let rawCache = materializedRawFallbackCache()
        let result = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset
        writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
        return result
    }

    public override var state: [MLXArray] {
        get {
            if activeBackend == .metalPolarQJL,
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
                rawFallbackCache = nil
                lastDecodedTransientBytes = 0
                cacheLifecycle = .empty
                return
            }
            if activeBackend == .metalPolarQJL, newValue.count == 10 {
                packedFallbackCache = nil
                let capacity = restoredLayoutMetadata?.capacity ?? newValue[0].dim(2)
                let keyHeadDimension =
                    restoredLayoutMetadata?.keyHeadDimension
                    ?? max(groupSize, (newValue[0].dim(3) * groupSize))
                let valueHeadDimension =
                    restoredLayoutMetadata?.valueHeadDimension
                    ?? max(groupSize, (newValue[5].dim(3) * groupSize))
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
                let valueLayout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[5].dim(0),
                    kvHeadCount: restoredLayoutMetadata?.kvHeadCount ?? newValue[5].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: restoredLayoutMetadata?.ringOffset ?? ringOffset(forOffset: offset),
                    pinnedPrefixLength: pinnedPrefixLength,
                    headDimension: valueHeadDimension,
                    groupsPerVector: newValue[5].dim(3),
                    magnitudeWordsPerGroup: newValue[5].dim(4),
                    bitsetWordsPerGroup: newValue[1].dim(4)
                )
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
                    scalesPerGroup: newValue[9].dim(4),
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
                if let compressedKeys {
                    cacheLifecycle = .compressedCommitted(
                        logicalLength: compressedKeys.layout.logicalLength,
                        capacity: compressedKeys.layout.capacity
                    )
                }
            } else if newValue.count == 10 {
                lastUnsupportedShape =
                    "compressed rotating TurboQuant state restored without Metal attention support"
                cacheLifecycle = .failed(reason: lastUnsupportedShape ?? "unsupported compressed state")
            } else {
                let rawCache = materializedRawFallbackCache()
                rawCache.state = newValue
                offset = rawCache.offset
                writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
                packedFallbackCache = nil
                compressedKeys = nil
                compressedValues = nil
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
            return .array(MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize))
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
        }
        packedKeys = nil
        packedValues = nil
        return trimmed
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        lastAttentionPath = .mlxPackedFallback
        lastUnsupportedShape = "compressed attention failed: \(message)"
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
            optimizationPolicy: optimizationPolicy,
            seed: seed,
            valueBits: valueBits
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        return TurboQuantKVCacheDiagnostics(
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: backendFallbackReason,
            metalCodecAvailable: availability.supportsMetalPolarQJLCodec,
            metalAttentionAvailable: availability.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
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
            footprint: cacheFootprint
        )
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxSize?.description ?? "-"), preset: \(preset.rawValue), backend: \(activeBackend.rawValue), rawFallback: \(rawFallbackCache != nil)"
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
        compressedValues = try MLX.turboQuantEmptyAttentionCode(
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

    private func materializedRawFallbackCache() -> RotatingKVCache {
        if let rawFallbackCache { return rawFallbackCache }
        let rawCache = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        if let compressedKeys, let compressedValues {
            do {
                let decodedKeys = try MLX.turboQuantMetalDecodeAttention(
                    compressedKeys,
                    outputDType: .float32
                )
                let decodedValues = try MLX.turboQuantMetalDecodeAttention(
                    compressedValues,
                    outputDType: .float32
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
                outputDType: .float32
            ),
            let decodedValues = try? MLX.turboQuantMetalDecodeAttention(
                compressedValues,
                outputDType: .float32
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
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil
    ) -> RotatingTurboQuantKVCache {
        let resolvedValueBits = valueBits ?? preset.defaultValueBits
        let capacity = maxSize ?? rotatingStep
        let cache = RotatingTurboQuantKVCache(
            maxSize: capacity,
            keep: rotatingKeep,
            step: rotatingStep,
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend,
            optimizationPolicy: optimizationPolicy,
            seed: seed,
            valueBits: resolvedValueBits
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

        if cache.activeBackend == .metalPolarQJL,
            TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention
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
                let valueCode = try MLX.turboQuantMetalEncodeAttention(
                    values,
                    configuration: valueConfiguration,
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: rotatingRingOffset,
                    pinnedPrefixLength: pinnedPrefixLength
                )
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
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil
    ) -> TurboQuantKVCache {
        let resolvedValueBits = valueBits ?? preset.defaultValueBits
        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend,
            optimizationPolicy: optimizationPolicy,
            seed: seed,
            valueBits: resolvedValueBits
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
            if cache.activeBackend == .metalPolarQJL,
                TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention
            {
                do {
                    let keyCode = try MLX.turboQuantMetalEncodeAttention(
                        currentState[0],
                        configuration: keyConfiguration,
                        logicalLength: self.offset
                    )
                    let valueCode = try MLX.turboQuantMetalEncodeAttention(
                        currentState[1],
                        configuration: valueConfiguration,
                        logicalLength: self.offset
                    )
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
                } catch {
                    cache.recordCompressedAttentionFailure(String(describing: error))
                }
            }
        }

        return cache
    }
}
