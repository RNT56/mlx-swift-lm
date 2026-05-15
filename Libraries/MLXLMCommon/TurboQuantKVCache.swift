import Foundation
import MLX

public typealias TurboQuantPreset = MLX.TurboQuantPreset
public typealias TurboQuantBackend = MLX.TurboQuantBackend
public typealias TurboQuantKernelAvailability = MLX.TurboQuantKernelAvailability
public typealias TurboQuantAttentionCode = MLX.TurboQuantAttentionCode
public typealias TurboQuantAttentionPath = MLX.TurboQuantAttentionPath

public enum KVCacheStrategy: String, Codable, Sendable, CaseIterable {
    case none
    case mlxAffine
    case turboQuant
}

public struct TurboQuantAttentionDiagnostics: Equatable, Codable, Sendable {
    public var metalAttentionAvailable: Bool
    public var activeAttentionPath: TurboQuantAttentionPath
    public var fallbackReason: String?
    public var lastUnsupportedShape: String?
}

public struct TurboQuantKVCacheDiagnostics: Equatable, Codable, Sendable {
    public var preset: TurboQuantPreset
    public var requestedBackend: TurboQuantBackend
    public var activeBackend: TurboQuantBackend
    public var fallbackReason: String?
    public var metalCodecAvailable: Bool
    public var metalAttentionAvailable: Bool
    public var activeAttentionPath: TurboQuantAttentionPath
    public var lastUnsupportedShape: String?
    public var groupSize: Int
    public var bits: Int
    public var maxSize: Int?
}

public protocol TurboQuantCompressedKVCacheProtocol: KVCache {
    var preset: TurboQuantPreset { get }
    var requestedBackend: TurboQuantBackend { get }
    var activeBackend: TurboQuantBackend { get }
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

public final class TurboQuantKVCache: QuantizedKVCache, TurboQuantCompressedKVCacheProtocol {
    private var compressedKeys: TurboQuantAttentionCode?
    private var compressedValues: TurboQuantAttentionCode?
    private var compressedStep: Int = 256
    private var lastAttentionPath: TurboQuantAttentionPath = .mlxPackedFallback
    private var lastUnsupportedShape: String?

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?

    public init(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked
    ) {
        self.preset = preset
        self.requestedBackend = backend
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        super.init(groupSize: groupSize, bits: preset.effectiveBits, mode: mode)
    }

    public override var metaState: [String] {
        get {
            var meta = super.metaState + [preset.rawValue, requestedBackend.rawValue]
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
            if newValue.count >= 13,
                let capacity = Int(newValue[7]),
                let logicalLength = Int(newValue[8]),
                let ringOffset = Int(newValue[9]),
                let headDimension = Int(newValue[10]),
                let kvHeadCount = Int(newValue[11])
            {
                offset = logicalLength
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
                if let path = TurboQuantAttentionPath(rawValue: newValue[12]) {
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
                let capacity = newValue[0].dim(2)
                let headDimension = max(groupSize, (newValue[0].dim(3) * groupSize))
                let logicalLength = offset > 0 ? min(offset, capacity) : capacity
                let layout = MLX.TurboQuantAttentionLayout(
                    batchSize: newValue[0].dim(0),
                    kvHeadCount: newValue[0].dim(1),
                    capacity: capacity,
                    logicalLength: logicalLength,
                    ringOffset: 0,
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
                    seed: 0x9E37_79B9_7F4A_7C15,
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
                    seed: 0x9E37_79B9_7F4A_7C15,
                    packedMagnitudes: newValue[5],
                    signs: newValue[6],
                    highPrecisionMask: newValue[7],
                    residualSigns: newValue[8],
                    scales: newValue[9]
                )
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
        TurboQuantAttentionDiagnostics(
            metalAttentionAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
            fallbackReason: backendFallbackReason,
            lastUnsupportedShape: lastUnsupportedShape
        )
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        TurboQuantKVCacheDiagnostics(
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: backendFallbackReason,
            metalCodecAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec,
            metalAttentionAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
            lastUnsupportedShape: lastUnsupportedShape,
            groupSize: groupSize,
            bits: bits,
            maxSize: nil
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
        lastAttentionPath = MLX.turboQuantMetalSupportsOnlineFusedAttention(
            queries: queries,
            keyCode: compressedKeys ?? placeholderCode(for: keys, role: .key),
            mask: mask
        ) ? .onlineFused : .twoStageCompressed
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
            backend: activeBackend
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset,
            role: .value,
            groupSize: groupSize,
            mode: mode,
            backend: activeBackend
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
            backend: requestedBackend
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
            groupSize: groupSize
        )
    }

    private func ensureCompressedCapacity(
        keys: MLXArray,
        values: MLXArray,
        requiredLength: Int
    ) throws {
        if compressedKeys == nil || compressedValues == nil {
            let capacity = ((compressedStep + requiredLength - 1) / compressedStep) * compressedStep
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
                groupSize: groupSize
            )
            compressedValues = try MLX.turboQuantEmptyAttentionCode(
                layout: valueLayout,
                preset: preset,
                role: .value,
                groupSize: groupSize
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
    CustomDebugStringConvertible
{
    private var rawCache: RotatingKVCache
    private var packedKeys: TurboQuantPackedTensor?
    private var packedValues: TurboQuantPackedTensor?

    public let preset: TurboQuantPreset
    public let requestedBackend: TurboQuantBackend
    public let activeBackend: TurboQuantBackend
    public let backendFallbackReason: String?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public override var maxSize: Int? { rawCache.maxSize }
    public override var isTrimmable: Bool { rawCache.isTrimmable }
    public var ropeOffset: RoPEOffset { rawCache.ropeOffset }

    public init(
        maxSize: Int,
        keep: Int = 4,
        step: Int = 256,
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked
    ) {
        self.rawCache = RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
        self.preset = preset
        self.requestedBackend = backend
        let availability = TurboQuantKernelAvailability.current
        self.activeBackend = availability.runtimeBackend(for: backend)
        self.backendFallbackReason = availability.fallbackReason(for: backend)
        self.groupSize = groupSize
        self.bits = preset.effectiveBits
        self.mode = mode
        super.init()
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let (cachedKeys, cachedValues) = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset, role: .key, groupSize: groupSize, mode: mode, backend: activeBackend)
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset, role: .value, groupSize: groupSize, mode: mode, backend: activeBackend)
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

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = rawCache.update(keys: keys, values: values)
        offset = rawCache.offset
        return result
    }

    public override var state: [MLXArray] {
        get { rawCache.state }
        set {
            if newValue.isEmpty {
                let meta = rawCache.metaState
                let keep = Int(meta[0]) ?? 4
                let maxSize = Int(meta[1]) ?? self.maxSize ?? 0
                let step = Int(meta[2]) ?? 256
                rawCache = RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
                rawCache.metaState = meta
                offset = rawCache.offset
                packedKeys = nil
                packedValues = nil
                compressedKeys = nil
                compressedValues = nil
                return
            }
            rawCache.state = newValue
            offset = rawCache.offset
            packedKeys = nil
            packedValues = nil
        }
    }

    public override var metaState: [String] {
        get { rawCache.metaState + [preset.rawValue, String(groupSize), requestedBackend.rawValue] }
        set {
            rawCache.metaState = Array(newValue.prefix(5))
            offset = rawCache.offset
        }
    }

    public override func innerState() -> [MLXArray] {
        rawCache.innerState()
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        rawCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = rawCache.trim(n)
        offset = rawCache.offset
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
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: requestedBackend
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    public var diagnostics: TurboQuantKVCacheDiagnostics {
        TurboQuantKVCacheDiagnostics(
            preset: preset,
            requestedBackend: requestedBackend,
            activeBackend: activeBackend,
            fallbackReason: backendFallbackReason,
            metalCodecAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec,
            metalAttentionAvailable: TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention,
            activeAttentionPath: activeBackend == .metalPolarQJL ? .twoStageCompressed : .mlxPackedFallback,
            lastUnsupportedShape: nil,
            groupSize: groupSize,
            bits: bits,
            maxSize: maxSize
        )
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxSize?.description ?? "-"), preset: \(preset.rawValue), backend: \(activeBackend.rawValue)"
    }
}

public extension KVCacheSimple {
    func toTurboQuant(
        preset: TurboQuantPreset = .turbo3_5,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .mlxPacked
    ) -> TurboQuantKVCache {
        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            mode: mode,
            backend: backend
        )
        cache.offset = self.offset

        let currentState = self.state
        if currentState.count == 2 {
            let keyConfiguration = TurboQuantConfiguration(
                preset: preset,
                role: .key,
                groupSize: groupSize,
                mode: mode,
                backend: cache.activeBackend
            )
            let valueConfiguration = TurboQuantConfiguration(
                preset: preset,
                role: .value,
                groupSize: groupSize,
                mode: mode,
                backend: cache.activeBackend
            )
            let keys = turboQuantized(currentState[0], configuration: keyConfiguration)
            let values = turboQuantized(currentState[1], configuration: valueConfiguration)
            cache.state = [
                keys.weight, keys.scales, keys.biases,
                values.weight, values.scales, values.biases,
            ].compactMap { $0 }
        }

        return cache
    }
}
