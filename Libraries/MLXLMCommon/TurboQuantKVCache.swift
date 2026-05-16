// Copyright © 2026 Schtack.

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
    public var maxSize: Int?
    public var rawFallbackAllocated: Bool
}

public protocol TurboQuantCompressedKVCacheProtocol: KVCache {
    var preset: TurboQuantPreset { get }
    var requestedBackend: TurboQuantBackend { get }
    var activeBackend: TurboQuantBackend { get }
    var optimizationPolicy: TurboQuantOptimizationPolicy { get }
    var attentionDiagnostics: TurboQuantAttentionDiagnostics { get }
    var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? { get }

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
}

public extension TurboQuantCompressedKVCacheProtocol {
    var prefersOnlineFusedAttention: Bool {
        optimizationPolicy != .conservative
    }
}

private struct RestoredAttentionLayoutMetadata {
    var capacity: Int?
    var logicalLength: Int
    var ringOffset: Int
    var pinnedPrefixLength: Int
    var headDimension: Int?
    var kvHeadCount: Int?
}

private enum TurboQuantCacheError: Error, CustomStringConvertible {
    case compressedBackfillUnavailable(String)

    var description: String {
        switch self {
        case .compressedBackfillUnavailable(let message):
            "TurboQuant compressed cache backfill unavailable: \(message)"
        }
    }
}

public final class TurboQuantKVCache: QuantizedKVCache, TurboQuantCompressedKVCacheProtocol {
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var compressedStep: Int = 256
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?
    private var restoredLayoutMetadata: RestoredAttentionLayoutMetadata?

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let seed: UInt64

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    ) {
        self.preset = preset
        self.requestedBackend = backend
        self.optimizationPolicy = optimizationPolicy
        self.seed = seed
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        super.init(groupSize: groupSize, bits: preset.effectiveBits, mode: mode)
    }

    public override var metaState: [String] {
        get {
            var meta = super.metaState + [
                preset.rawValue,
                requestedBackend.rawValue,
                String(seed),
            ]
            if let compressedKeys {
                let layout = compressedKeys.layout
                meta += [
                    "turboq-attn-v\(layout.layoutVersion)",
                    String(layout.capacity),
                    String(layout.logicalLength),
                    String(layout.ringOffset),
                    String(layout.headDimension),
                    String(layout.kvHeadCount),
                    lastAttentionPath.rawValue,
                ]
            }
            return meta
        }
        set {
            super.metaState = Array(newValue.prefix(4))
            let compressedBase = newValue.firstIndex {
                $0.hasPrefix("turboq-attn-v")
            } ?? (UInt64(newValue.dropFirst(6).first ?? "") == nil ? 6 : 7)
            if newValue.count >= compressedBase + 7,
                let capacity = Int(newValue[compressedBase + 1]),
                let logicalLength = Int(newValue[compressedBase + 2]),
                let ringOffset = Int(newValue[compressedBase + 3]),
                let headDimension = Int(newValue[compressedBase + 4]),
                let kvHeadCount = Int(newValue[compressedBase + 5])
            {
                offset = logicalLength
                restoredLayoutMetadata = RestoredAttentionLayoutMetadata(
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: ringOffset,
                    pinnedPrefixLength: 0,
                    headDimension: headDimension,
                    kvHeadCount: kvHeadCount
                )
                if var compressedKeys {
                    compressedKeys.layout.capacity = capacity
                    compressedKeys.layout.logicalLength = logicalLength
                    compressedKeys.layout.ringOffset = ringOffset
                    compressedKeys.layout.headDimension = headDimension
                    compressedKeys.layout.kvHeadCount = kvHeadCount
                    self.compressedKeys = compressedKeys
                }
                if var compressedValues {
                    compressedValues.layout.capacity = capacity
                    compressedValues.layout.logicalLength = logicalLength
                    compressedValues.layout.ringOffset = ringOffset
                    compressedValues.layout.headDimension = headDimension
                    compressedValues.layout.kvHeadCount = kvHeadCount
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
            return super.state
        }
        set {
            if activeBackend == .metalPolarQJL, newValue.isEmpty {
                compressedKeys = nil
                compressedValues = nil
                return
            }
            if activeBackend == .metalPolarQJL, newValue.count == 10 {
                let capacity = restoredLayoutMetadata?.capacity ?? newValue[0].dim(2)
                let headDimension = restoredLayoutMetadata?.headDimension
                    ?? max(groupSize, (newValue[0].dim(3) * groupSize))
                let logicalLength = restoredLayoutMetadata?.logicalLength
                    ?? (offset > 0 ? min(offset, capacity) : capacity)
                let layout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[0].dim(0),
                    kvHeadCount: restoredLayoutMetadata?.kvHeadCount ?? newValue[0].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: restoredLayoutMetadata?.ringOffset ?? 0,
                    pinnedPrefixLength: restoredLayoutMetadata?.pinnedPrefixLength ?? 0,
                    headDimension: headDimension,
                    groupsPerVector: newValue[0].dim(3),
                    magnitudeWordsPerGroup: newValue[0].dim(4),
                    bitsetWordsPerGroup: newValue[1].dim(4)
                )
                compressedKeys = TurboQuantAttentionCode(
                    layout: layout,
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
                    layout: layout,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
            } else if newValue.count == 10 {
                lastUnsupportedShape =
                    "compressed TurboQuant state restored without Metal attention support"
            } else {
                super.state = newValue
            }
        }
    }

    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        guard let compressedKeys, let compressedValues else { return nil }
        return (compressedKeys, compressedValues)
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
            rawFallbackAllocated: false
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
            maxSize: nil,
            rawFallbackAllocated: false
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
        guard queries.dim(0) == 1, keys.dim(0) == 1, values.dim(0) == 1 else {
            lastUnsupportedShape = "batch sizes greater than 1 use packed fallback"
            return false
        }
        guard [64, 80, 96, 128, 256].contains(queries.dim(3)),
            queries.dim(3) == keys.dim(3),
            queries.dim(3) == values.dim(3)
        else {
            lastUnsupportedShape = "unsupported head dimension q=\(queries.dim(3)) k=\(keys.dim(3)) v=\(values.dim(3))"
            return false
        }
        guard queries.dim(1) % keys.dim(1) == 0 else {
            lastUnsupportedShape = "query heads must be a multiple of KV heads"
            return false
        }
        let supportsTiled = prefersOnlineFusedAttention && MLX.turboQuantMetalSupportsOnlineFusedAttention(
            queries: queries,
            keyCode: compressedKeys ?? placeholderCode(for: keys, role: .key),
            mask: mask
        )
        lastAttentionPath = supportsTiled ? .tiledOnlineFused : .twoStageCompressed
        lastUnsupportedShape = nil
        return true
    }

    public func updateCompressed(keys: MLXArray, values: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    ) {
        let previousOffset = offset
        let tokenCount = keys.dim(2)
        try ensureCompressedCapacity(keys: keys, values: values, requiredLength: previousOffset + tokenCount)

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
            seed: seed ^ turboQuantValueSeedSalt
        )
        let encodedKeys = try MLX.turboQuantMetalEncodeAttention(
            keys,
            configuration: keyConfiguration,
            stream: .default
        )
        let encodedValues = try MLX.turboQuantMetalEncodeAttention(
            values,
            configuration: valueConfiguration,
            stream: .default
        )

        var currentKeys = compressedKeys!
        var currentValues = compressedValues!
        let range = previousOffset ..< (previousOffset + tokenCount)
        currentKeys.packedMagnitudes[.ellipsis, range, 0..., 0...] = encodedKeys.packedMagnitudes
        currentKeys.signs[.ellipsis, range, 0..., 0...] = encodedKeys.signs
        currentKeys.highPrecisionMask[.ellipsis, range, 0..., 0...] = encodedKeys.highPrecisionMask
        currentKeys.residualSigns[.ellipsis, range, 0..., 0...] = encodedKeys.residualSigns
        currentKeys.scales[.ellipsis, range, 0..., 0...] = encodedKeys.scales

        currentValues.packedMagnitudes[.ellipsis, range, 0..., 0...] = encodedValues.packedMagnitudes
        currentValues.signs[.ellipsis, range, 0..., 0...] = encodedValues.signs
        currentValues.highPrecisionMask[.ellipsis, range, 0..., 0...] = encodedValues.highPrecisionMask
        currentValues.residualSigns[.ellipsis, range, 0..., 0...] = encodedValues.residualSigns
        currentValues.scales[.ellipsis, range, 0..., 0...] = encodedValues.scales

        offset += tokenCount
        currentKeys.layout.logicalLength = offset
        currentValues.layout.logicalLength = offset
        compressedKeys = currentKeys
        compressedValues = currentValues
        return (currentKeys, currentValues)
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        lastAttentionPath = .mlxPackedFallback
        lastUnsupportedShape = "compressed attention failed: \(message)"
    }

    public override func copy() -> any KVCache {
        let new = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend,
            optimizationPolicy: optimizationPolicy,
            seed: seed
        )
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    private func placeholderCode(for array: MLXArray, role: TurboQuantTensorRole) -> TurboQuantAttentionCode {
        let layout = try! MLX.turboQuantAttentionLayout(
            for: array,
            preset: preset,
            groupSize: groupSize
        )
        return try! MLX.turboQuantEmptyAttentionCode(
            layout: layout,
            preset: preset,
            role: role,
            groupSize: groupSize,
            seed: role == .value ? seed ^ turboQuantValueSeedSalt : seed
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
                groupSize: groupSize,
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
                seed: seed ^ turboQuantValueSeedSalt
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
            seed: seed ^ turboQuantValueSeedSalt
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
            seed: code.seed
        )
        return TurboQuantAttentionCode(
            layout: newLayout,
            preset: code.preset,
            role: code.role,
            groupSize: code.groupSize,
            seed: code.seed,
            packedMagnitudes: concatenated([code.packedMagnitudes, zeros.packedMagnitudes], axis: 2),
            signs: concatenated([code.signs, zeros.signs], axis: 2),
            highPrecisionMask: concatenated([code.highPrecisionMask, zeros.highPrecisionMask], axis: 2),
            residualSigns: concatenated([code.residualSigns, zeros.residualSigns], axis: 2),
            scales: concatenated([code.scales, zeros.scales], axis: 2)
        )
    }
}

public final class RotatingTurboQuantKVCache: BaseKVCache, QuantizedKVCacheProtocol,
    TurboQuantCompressedKVCacheProtocol,
    CustomDebugStringConvertible
{
    private var rawFallbackCache: RotatingKVCache?
    private var packedKeys: TurboQuantPackedTensor?
    private var packedValues: TurboQuantPackedTensor?
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?
    private var restoredLayoutMetadata: RestoredAttentionLayoutMetadata?
    private let keep: Int
    private let step: Int
    private let maxCacheSize: Int
    private var writeIndex: Int

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let optimizationPolicy: TurboQuantOptimizationPolicy
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let seed: UInt64

    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { offset < maxCacheSize }

    public init(
        maxSize: Int,
        keep: Int = 4,
        step: Int = 256,
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    ) {
        self.keep = max(0, min(keep, maxSize))
        self.step = step
        self.maxCacheSize = maxSize
        self.writeIndex = self.keep
        self.preset = preset
        self.requestedBackend = backend
        self.optimizationPolicy = optimizationPolicy
        self.seed = seed
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
            seed: seed ^ turboQuantValueSeedSalt
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
        guard let packedKeys, let packedValues else { return nil }
        return (
            (packedKeys.weight, packedKeys.scales, packedKeys.biases),
            (packedValues.weight, packedValues.scales, packedValues.biases)
        )
    }

    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        guard let compressedKeys, let compressedValues else { return nil }
        return (compressedKeys, compressedValues)
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
            rawFallbackAllocated: rawFallbackCache != nil
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
        guard queries.dim(0) == 1, keys.dim(0) == 1, values.dim(0) == 1 else {
            lastUnsupportedShape = "batch sizes greater than 1 use packed fallback"
            return false
        }
        guard [64, 80, 96, 128, 256].contains(queries.dim(3)),
            queries.dim(3) == keys.dim(3),
            queries.dim(3) == values.dim(3)
        else {
            lastUnsupportedShape = "unsupported head dimension q=\(queries.dim(3)) k=\(keys.dim(3)) v=\(values.dim(3))"
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
        let keyCode = compressedKeys ?? placeholderCode(for: keys, role: .key)
        let supportsTiled = prefersOnlineFusedAttention && MLX.turboQuantMetalSupportsOnlineFusedAttention(
            queries: queries,
            keyCode: keyCode,
            mask: mask
        )
        lastAttentionPath = supportsTiled ? .tiledOnlineFused : .twoStageCompressed
        lastUnsupportedShape = nil
        return true
    }

    public func updateCompressed(keys: MLXArray, values: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    ) {
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
            seed: seed ^ turboQuantValueSeedSalt
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
        for token in 0 ..< keys.dim(2) {
            let physical = physicalSlot(forAbsoluteToken: offset + token)
            let source = token ..< (token + 1)
            let target = physical ..< (physical + 1)
            currentKeys.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                encodedKeys.packedMagnitudes[.ellipsis, source, 0..., 0...]
            currentKeys.signs[.ellipsis, target, 0..., 0...] =
                encodedKeys.signs[.ellipsis, source, 0..., 0...]
            currentKeys.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                encodedKeys.highPrecisionMask[.ellipsis, source, 0..., 0...]
            currentKeys.residualSigns[.ellipsis, target, 0..., 0...] =
                encodedKeys.residualSigns[.ellipsis, source, 0..., 0...]
            currentKeys.scales[.ellipsis, target, 0..., 0...] =
                encodedKeys.scales[.ellipsis, source, 0..., 0...]

            currentValues.packedMagnitudes[.ellipsis, target, 0..., 0...] =
                encodedValues.packedMagnitudes[.ellipsis, source, 0..., 0...]
            currentValues.signs[.ellipsis, target, 0..., 0...] =
                encodedValues.signs[.ellipsis, source, 0..., 0...]
            currentValues.highPrecisionMask[.ellipsis, target, 0..., 0...] =
                encodedValues.highPrecisionMask[.ellipsis, source, 0..., 0...]
            currentValues.residualSigns[.ellipsis, target, 0..., 0...] =
                encodedValues.residualSigns[.ellipsis, source, 0..., 0...]
            currentValues.scales[.ellipsis, target, 0..., 0...] =
                encodedValues.scales[.ellipsis, source, 0..., 0...]
        }

        offset += keys.dim(2)
        writeIndex = nextWriteIndex(afterOffset: offset)
        updateCompressedLayouts(keys: &currentKeys, values: &currentValues)
        compressedKeys = currentKeys
        compressedValues = currentValues
        packedKeys = nil
        packedValues = nil
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
                compressedKeys = nil
                compressedValues = nil
                rawFallbackCache = nil
                return
            }
            if activeBackend == .metalPolarQJL, newValue.count == 10 {
                let capacity = restoredLayoutMetadata?.capacity ?? newValue[0].dim(2)
                let headDimension = restoredLayoutMetadata?.headDimension
                    ?? max(groupSize, (newValue[0].dim(3) * groupSize))
                let logicalLength = restoredLayoutMetadata?.logicalLength ?? min(offset, capacity)
                let layout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[0].dim(0),
                    kvHeadCount: restoredLayoutMetadata?.kvHeadCount ?? newValue[0].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: restoredLayoutMetadata?.ringOffset ?? ringOffset(forOffset: offset),
                    pinnedPrefixLength: restoredLayoutMetadata?.pinnedPrefixLength ?? min(keep, capacity),
                    headDimension: headDimension,
                    groupsPerVector: newValue[0].dim(3),
                    magnitudeWordsPerGroup: newValue[0].dim(4),
                    bitsetWordsPerGroup: newValue[1].dim(4)
                )
                compressedKeys = TurboQuantAttentionCode(
                    layout: layout,
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
                    layout: layout,
                    preset: preset,
                    role: .value,
                    groupSize: groupSize,
                    seed: seed ^ turboQuantValueSeedSalt,
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
            } else if newValue.count == 10 {
                lastUnsupportedShape =
                    "compressed rotating TurboQuant state restored without Metal attention support"
            } else {
                let rawCache = materializedRawFallbackCache()
                rawCache.state = newValue
                offset = rawCache.offset
                writeIndex = currentWriteIndexFromRawMeta(rawCache.metaState)
                compressedKeys = nil
                compressedValues = nil
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
            ]
            if let compressedKeys {
                let layout = compressedKeys.layout
                meta += [
                    "turboq-rot-v\(layout.layoutVersion)",
                    String(layout.logicalLength),
                    String(layout.ringOffset),
                    String(layout.pinnedPrefixLength),
                    String(layout.headDimension),
                    lastAttentionPath.rawValue,
                    rawFallbackCache == nil ? "raw-free" : "raw-fallback",
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
            if newValue.count >= 13,
                let logicalLength = Int(newValue[10]),
                let ringOffset = Int(newValue[11]),
                let pinnedPrefixLength = Int(newValue[12])
            {
                var headDimension: Int?
                var pathIndex = 13
                if newValue.count > 14, let parsedHeadDimension = Int(newValue[13]) {
                    headDimension = parsedHeadDimension
                    pathIndex = 14
                }
                restoredLayoutMetadata = RestoredAttentionLayoutMetadata(
                    capacity: maxCacheSize,
                    logicalLength: logicalLength,
                    ringOffset: ringOffset,
                    pinnedPrefixLength: pinnedPrefixLength,
                    headDimension: headDimension,
                    kvHeadCount: nil
                )
                if var compressedKeys {
                    compressedKeys.layout.logicalLength = logicalLength
                    compressedKeys.layout.ringOffset = ringOffset
                    compressedKeys.layout.pinnedPrefixLength = pinnedPrefixLength
                    if let headDimension {
                        compressedKeys.layout.headDimension = headDimension
                    }
                    self.compressedKeys = compressedKeys
                }
                if var compressedValues {
                    compressedValues.layout.logicalLength = logicalLength
                    compressedValues.layout.ringOffset = ringOffset
                    compressedValues.layout.pinnedPrefixLength = pinnedPrefixLength
                    if let headDimension {
                        compressedValues.layout.headDimension = headDimension
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
                return .array(createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
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
    }

    public override func copy() -> any KVCache {
        guard let maxSize else {
            fatalError("RotatingTurboQuantKVCache requires maxSize")
        }
        let new = RotatingTurboQuantKVCache(
            maxSize: maxSize,
            keep: keep,
            step: step,
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend,
            optimizationPolicy: optimizationPolicy,
            seed: seed
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
            maxSize: maxSize,
            rawFallbackAllocated: rawFallbackCache != nil
        )
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxSize?.description ?? "-"), preset: \(preset.rawValue), backend: \(activeBackend.rawValue), rawFallback: \(rawFallbackCache != nil)"
    }

    private func placeholderCode(for array: MLXArray, role: TurboQuantTensorRole) -> TurboQuantAttentionCode {
        let layout = try! MLX.turboQuantAttentionLayout(
            for: array,
            preset: preset,
            groupSize: groupSize,
            capacity: maxCacheSize,
            logicalLength: min(offset + array.dim(2), maxCacheSize),
            ringOffset: ringOffset(forOffset: offset + array.dim(2)),
            pinnedPrefixLength: min(keep, maxCacheSize)
        )
        return try! MLX.turboQuantEmptyAttentionCode(
            layout: layout,
            preset: preset,
            role: role,
            groupSize: groupSize,
            seed: role == .value ? seed ^ turboQuantValueSeedSalt : seed
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
        let keyLayout = try MLX.turboQuantAttentionLayout(
            for: keys,
            preset: preset,
            groupSize: groupSize,
            capacity: maxCacheSize,
            logicalLength: logicalLength,
            ringOffset: ringOffset(forOffset: offset),
            pinnedPrefixLength: min(keep, maxCacheSize)
        )
        let valueLayout = try MLX.turboQuantAttentionLayout(
            for: values,
            preset: preset,
            groupSize: groupSize,
            capacity: maxCacheSize,
            logicalLength: logicalLength,
            ringOffset: ringOffset(forOffset: offset),
            pinnedPrefixLength: min(keep, maxCacheSize)
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
            seed: seed ^ turboQuantValueSeedSalt
        )
    }

    private func updateCompressedLayouts(
        keys: inout TurboQuantAttentionCode,
        values: inout TurboQuantAttentionCode
    ) {
        let logicalLength = min(offset, maxCacheSize)
        let ringOffset = ringOffset(forOffset: offset)
        keys.layout.logicalLength = logicalLength
        keys.layout.ringOffset = ringOffset
        keys.layout.pinnedPrefixLength = min(keep, maxCacheSize)
        values.layout.logicalLength = logicalLength
        values.layout.ringOffset = ringOffset
        values.layout.pinnedPrefixLength = min(keep, maxCacheSize)
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

    private func currentWriteIndexFromRawMeta(_ meta: [String]) -> Int {
        guard meta.count >= 5 else { return nextWriteIndex(afterOffset: offset) }
        return Int(meta[4]) ?? nextWriteIndex(afterOffset: offset)
    }
}

public extension RotatingKVCache {
    func toTurboQuant(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    ) -> RotatingTurboQuantKVCache {
        guard let maxSize else {
            fatalError("RotatingKVCache requires maxSize for TurboQuant conversion")
        }

        let cache = RotatingTurboQuantKVCache(
            maxSize: maxSize,
            keep: rotatingKeep,
            step: rotatingStep,
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend,
            optimizationPolicy: optimizationPolicy,
            seed: seed
        )
        cache.offset = offset
        cache.metaState = metaState

        let currentState = state
        guard currentState.count == 2 else { return cache }

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
                    seed: seed ^ turboQuantValueSeedSalt
                )
                let keys = turboQuantPhysicalizedState(currentState[0], maxSize: maxSize)
                let values = turboQuantPhysicalizedState(currentState[1], maxSize: maxSize)
                let logicalLength = min(offset, maxSize)
                let pinnedPrefixLength = min(rotatingKeep, maxSize)
                let keyCode = try MLX.turboQuantMetalEncodeAttention(
                    keys,
                    configuration: keyConfiguration,
                    capacity: maxSize,
                    logicalLength: logicalLength,
                    ringOffset: rotatingRingOffset,
                    pinnedPrefixLength: pinnedPrefixLength
                )
                let valueCode = try MLX.turboQuantMetalEncodeAttention(
                    values,
                    configuration: valueConfiguration,
                    capacity: maxSize,
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

public extension KVCacheSimple {
    func toTurboQuant(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    ) -> TurboQuantKVCache {
        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend,
            optimizationPolicy: optimizationPolicy,
            seed: seed
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
                seed: seed ^ turboQuantValueSeedSalt
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
