// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
///
/// See ``KVCache/ropeOffset``.
public enum RoPEOffset {
    case scalar(Int)
    case batch(MLXArray)
}

/// Implementation of KV cache functionality for MLX Swift
///
///
/// ## Quantized Cache Usage
///
/// **Standard caches:**
/// ```swift
/// let cache = KVCacheSimple()
/// let (keys, values) = cache.update(keys: keys, values: values)
/// let output = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, ...)
/// ```
///
/// **Quantized cache:**
/// ```swift
/// let quantizedCache = QuantizedKVCache(groupSize: 64, bits: 4)
/// let (qKeys, qValues) = quantizedCache.updateQuantized(keys: keys, values: values)
///
/// let output = quantizedScaledDotProductAttention(
///     queries: queries,
///     quantizedKeys: qKeys,
///     quantizedValues: qValues,
///     scale: scale,
///     mask: mask,
///     groupSize: quantizedCache.groupSize,
///     bits: quantizedCache.bits
/// )
/// ```
///
/// Interface for Key/Value cache for LLMs.
///
/// See ``LanguageModel/newCache(parameters:)``
public protocol KVCache: Evaluatable {
    /// get the current offset
    var offset: Int { get }

    /// Offset to use with ``applyRotaryPosition(_:to:offset:)``.
    var ropeOffset: RoPEOffset { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int

    /// Create an attention mask for this cache
    ///
    /// This method encapsulates cache-specific mask creation logic. Implementations should handle offset capping, window size logic,
    /// and optimization decisions (symbolic vs array masks).
    ///
    /// - Parameters:
    ///   - n: The sequence length for the new tokens
    ///   - windowSize: Optional sliding window size
    ///   - returnArray: Force return of array mask instead of symbolic
    /// - Returns: Attention mask mode for scaled dot product attention
    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode

    /// Create an independent deep copy of this cache.
    func copy() -> any KVCache

    /// Prepare cache metadata for a batched sequence.
    func prepare(lengths: [Int]?)

    /// Prepare cache metadata for a batched sequence.
    func prepare(lengths: MLXArray?)

    /// Clear transient cache metadata after generation.
    func finalize()
}

/// Marker for architecture-specific caches that must remain raw because model attention
/// reads shared or latent KV state directly.
public protocol RawOnlyKVCache: KVCache {}

extension KVCache {
    public var ropeOffset: RoPEOffset {
        .scalar(offset)
    }

    public func prepare(lengths: [Int]?) {}

    public func prepare(lengths: MLXArray?) {}

    public func finalize() {}
}

public func withPreparedCache<Result>(
    _ cache: [any KVCache],
    lengths: [Int]?,
    _ body: () throws -> Result
) rethrows -> Result {
    guard let lengths else {
        return try body()
    }
    for cache in cache {
        cache.prepare(lengths: lengths)
    }
    defer {
        for cache in cache {
            cache.finalize()
        }
    }
    return try body()
}

/// Protocol for caches that support efficient quantized operations
///
/// **Usage Example:**
/// ```swift
/// // Efficient quantized path
/// if let quantizedCache = cache as? QuantizedKVCacheProtocol {
///     let (qKeys, qValues) = quantizedCache.updateQuantized(keys: k, values: v)
///     // Use native quantized operations
///     let scores = quantizedMM(queries, w: qKeys.0, scales: qKeys.1, biases: qKeys.2, ...)
/// } else {
///     // Regular path
///     let (k, v) = cache.update(keys: k, values: v)
///     let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, ...)
/// }
/// ```
public protocol QuantizedKVCacheProtocol: KVCache {
    /// The quantization group size used
    var groupSize: Int { get }

    /// The number of quantization bits used
    var bits: Int { get }

    /// Quantization mode
    var mode: QuantizationMode { get }

    /// Update cache and return quantized tuples for maximum efficiency
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )

    /// Get current quantized state without updating
    ///
    /// Useful for accessing cached data without adding new tokens.
    /// - Returns: Current quantized state, or nil if cache is empty
    func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))?
}

/// Marker for first-class affine int4 caches that must use native quantized SDPA only.
public protocol NativeAffineInt4KVCacheProtocol: QuantizedKVCacheProtocol {}

public struct AffineK8VxResidualState: @unchecked Sendable {
    public var laneIndices: MLXArray
    public var values: MLXArray
    public var residualsPerGroup: Int

    public init(laneIndices: MLXArray, values: MLXArray, residualsPerGroup: Int) {
        self.laneIndices = laneIndices
        self.values = values
        self.residualsPerGroup = max(0, residualsPerGroup)
    }
}

/// Marker for mixed affine caches that keep keys at K8 and values at Vx.
public protocol NativeAffineK8V4KVCacheProtocol: QuantizedKVCacheProtocol {
    var keyGroupSize: Int { get }
    var keyBits: Int { get }
    var valueGroupSize: Int { get }
    var valueBits: Int { get }
    var residualsPerGroup: Int { get }
    var sparseValuePolicy: TurboQuantSparseValuePolicy { get }
    var sparseValueRuntimeMode: TurboQuantRuntimeMode { get }
    var attentionDiagnostics: TurboQuantAttentionDiagnostics { get }
    func recordNativeAffineK8V4AttentionPath(
        _ path: TurboQuantAttentionPath,
        failureReason: String?
    )
    func getValueResidualState() -> AffineK8VxResidualState?
}

extension NativeAffineK8V4KVCacheProtocol {
    public func recordNativeAffineK8V4AttentionPath(
        _ path: TurboQuantAttentionPath,
        failureReason: String?
    ) {}

    public var sparseValuePolicy: TurboQuantSparseValuePolicy { .off }
    public var sparseValueRuntimeMode: TurboQuantRuntimeMode { .capacityTurboQuant }
}

private struct ResolvedQuantizationParameters {
    var groupSize: Int
    var bits: Int
}

private func resolvedQuantizationParameters(
    groupSize: Int,
    bits: Int,
    mode: QuantizationMode
) -> ResolvedQuantizationParameters {
    switch mode {
    case .affine:
        return .init(groupSize: groupSize, bits: bits)
    case .mxfp4:
        return .init(groupSize: 32, bits: 4)
    case .mxfp8:
        return .init(groupSize: 32, bits: 8)
    case .nvfp4:
        return .init(groupSize: 16, bits: 4)
    }
}

func compatibleAffineGroupSize(
    configuredGroupSize: Int,
    keyDimension: Int,
    valueDimension: Int
) -> Int {
    let configuredGroupSize = max(1, configuredGroupSize)
    guard keyDimension > 0, valueDimension > 0 else { return configuredGroupSize }
    guard !keyDimension.isMultiple(of: configuredGroupSize)
        || !valueDimension.isMultiple(of: configuredGroupSize)
    else {
        return configuredGroupSize
    }

    for candidate in [128, 64, 32] where candidate <= configuredGroupSize {
        if keyDimension.isMultiple(of: candidate), valueDimension.isMultiple(of: candidate) {
            return candidate
        }
    }
    return configuredGroupSize
}

func supportsMLXAffineQuantization(dimension: Int, groupSize: Int) -> Bool {
    [32, 64, 128].contains(groupSize) && dimension.isMultiple(of: groupSize)
}

func supportsMLXAffineKVQuantization(
    keyDimension: Int,
    valueDimension: Int,
    keyGroupSize: Int,
    valueGroupSize: Int
) -> Bool {
    supportsMLXAffineQuantization(dimension: keyDimension, groupSize: keyGroupSize)
        && supportsMLXAffineQuantization(dimension: valueDimension, groupSize: valueGroupSize)
}

func placeholderQuantizedTuple(
    for array: MLXArray,
    bits: Int,
    includeBiases: Bool = true
) -> (MLXArray, MLXArray, MLXArray?) {
    let packedWidth = max(1, (array.dim(3) * max(1, bits) + 31) / 32)
    let shape = [array.dim(0), array.dim(1), array.dim(2)]
    let weights = MLXArray.zeros(shape + [packedWidth], dtype: .uint32)
    let scales = MLXArray.ones(shape + [1], dtype: .float32)
    let biases = includeBiases ? MLXArray.zeros(shape + [1], dtype: .float32) : nil
    return (weights, scales, biases)
}

private func inferredQuantizationMode(
    stateCount: Int,
    groupSize: Int,
    bits: Int
) -> QuantizationMode {
    guard stateCount == 4 else { return .affine }
    if groupSize == 16 && bits == 4 {
        return .nvfp4
    }
    if groupSize == 32 && bits == 8 {
        return .mxfp8
    }
    return .mxfp4
}

/// Base cache implementation providing default behaviors
open class BaseKVCache: KVCache {
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get { [""] }
        set {
            guard newValue.count == 1 && newValue[0].isEmpty else {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    open func copy() -> any KVCache {
        fatalError("copy() must be implemented by subclass")
    }

    open func prepare(lengths: [Int]?) {}

    open func prepare(lengths: MLXArray?) {}

    open func finalize() {}

    /// Default implementation for caches without special mask requirements
    open func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // For single token, no mask needed
        if n == 1 {
            return .none
        }

        // For multi-token sequences
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }
}

public func createCausalMask(
    n: Int,
    offset: Int,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    if var lengths {
        lengths = lengths[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rinds .< lengths)
    }

    return mask
}

/// Create an attention mask matching mlx-lm's create_attention_mask helper.
///
/// This returns `.causal` when a symbolic mask is sufficient, avoiding
/// materializing a full mask array.
public func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }

    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }

    return .causal
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also `MultiHeadAttention.createAdditiveCausalMask(_:dtype:)` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

@available(
    *, deprecated,
    message: "Use createAttentionMask(h:cache:windowSize:returnArray:) with a single cache instead"
)
public func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if t > 1 {
        var returnArray = returnArray
        var offset = 0
        var windowSize: Int? = nil
        if let c = cache?.first {
            offset = c.offset
            if let maxSize = c.maxSize {
                windowSize = maxSize
                offset = min(maxSize - 1, offset)
                if !returnArray {
                    returnArray = offset + t > maxSize
                }
            }
        }

        if returnArray {
            return .array(createCausalMask(n: t, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    }
    return .none
}

/// Create an attention mask with explicit window size parameter.
///
/// - Parameters:
///   - h: The input array (used to determine sequence length)
///   - cache: Optional single KV cache
///   - windowSize: Optional sliding window size (if provided, creates windowed attention)
///   - returnArray: Force return of array mask instead of symbolic "causal"
/// - Returns: Attention mask mode for scaled dot product attention
public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    // Delegate to cache's makeMask if available
    if let cache = cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    // Fallback for no cache
    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

public func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

/// Standard KV cache implementation based on Python's KVCache
/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
public class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(2)
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    /// Convert to quantized cache for maximum efficiency
    ///
    /// Use `updateQuantized()` and `quantizedScaledDotProductAttention()` for zero-overhead operation.
    public func toQuantized(
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) -> QuantizedKVCache {
        let requestedParameters = resolvedQuantizationParameters(
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        if let keys = self.keys, let values = self.values {
            // Quantize the current keys and values
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            guard
                let effectiveGroupSize = resolvedKVQuantizationGroupSize(
                    requested: requestedParameters.groupSize,
                    keyHeadDim: currentKeys.dim(3),
                    valueHeadDim: currentValues.dim(3)
                )
            else {
                fatalError(
                    "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(requestedParameters.groupSize). Key head dim: \(currentKeys.dim(3)). Value head dim: \(currentValues.dim(3))."
                )
            }
            let quantizedCache = QuantizedKVCache(
                groupSize: effectiveGroupSize,
                bits: requestedParameters.bits,
                mode: mode
            )
            quantizedCache.offset = self.offset

            let quantizedKeys = quantized(
                currentKeys,
                groupSize: quantizedCache.groupSize,
                bits: quantizedCache.bits,
                mode: quantizedCache.mode
            )
            let quantizedValues = quantized(
                currentValues,
                groupSize: quantizedCache.groupSize,
                bits: quantizedCache.bits,
                mode: quantizedCache.mode
            )

            // Set the quantized state
            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }

            return quantizedCache
        }

        let quantizedCache = QuantizedKVCache(
            groupSize: requestedParameters.groupSize,
            bits: requestedParameters.bits,
            mode: mode
        )
        quantizedCache.offset = self.offset
        return quantizedCache
    }

    public func toAffineInt4(groupSize: Int = TurboQuantKVCodec.affineInt4DefaultGroupSize)
        -> AffineInt4KVCache
    {
        let cache = AffineInt4KVCache(groupSize: groupSize)
        cache.offset = self.offset
        let parameters = resolvedQuantizationParameters(
            groupSize: cache.groupSize,
            bits: cache.bits,
            mode: cache.mode
        )

        if let keys = self.keys, let values = self.values {
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            let quantizedKeys = quantized(
                currentKeys,
                groupSize: parameters.groupSize,
                bits: parameters.bits,
                mode: cache.mode
            )
            let quantizedValues = quantized(
                currentValues,
                groupSize: parameters.groupSize,
                bits: parameters.bits,
                mode: cache.mode
            )
            cache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }
        }

        return cache
    }

    public func toAffineK8V4(
        valueBits: Int = TurboQuantKVCodec.affineK8V4ValueBits,
        valueGroupSize: Int = TurboQuantKVCodec.affineK8V4ValueGroupSize,
        residualsPerGroup: Int = 0,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueRuntimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        layerIndex: Int? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil
    ) -> AffineK8V4KVCache {
        let cache = AffineK8V4KVCache(
            valueGroupSize: valueGroupSize,
            valueBits: valueBits,
            residualsPerGroup: residualsPerGroup,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueRuntimeMode: sparseValueRuntimeMode,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason
        )
        let s = state
        if !s.isEmpty {
            cache.setUnquantizedState(keys: s[0], values: s[1], offset: offset)
        }
        return cache
    }

    public override func copy() -> any KVCache {
        let new = KVCacheSimple()
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

public final class RawOnlyKVCacheSimple: KVCacheSimple, RawOnlyKVCache {
    public override func copy() -> any KVCache {
        let new = RawOnlyKVCacheSimple()
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }
}

/// Rotating KV cache for sliding window attention
public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var maxCacheSize: Int
    private var step: Int
    private var idx: Int = 0

    public override var maxSize: Int? { maxCacheSize }
    internal var rotatingKeep: Int { keep }
    internal var rotatingStep: Int { step }
    internal var rotatingWriteIndex: Int { idx }
    internal var rotatingRingOffset: Int {
        let pinned = min(keep, maxCacheSize)
        let ringCapacity = maxCacheSize - pinned
        guard ringCapacity > 0, offset > pinned else { return 0 }
        let activeRing = min(offset - pinned, ringCapacity)
        return (offset - pinned - activeRing) % ringCapacity
    }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            // Allow temporary cache growth during multi-token processing (e.g., prompt prefill).
            // The largest size is maxCacheSize + S - 1 to ensure
            // every token gets at least maxCacheSize context
            let trimSize = idx - maxCacheSize + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // May not have hit the max size yet, so potentially keep growing the cache
        if self.keys == nil
            || (prev >= self.keys!.dim(2) && self.keys!.dim(2) < maxCacheSize)
        {
            let newSize = min(step, maxCacheSize - prev)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        // Rotate if we've hit the end
        if idx == maxCacheSize {
            idx = keep
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // Return the appropriate cache slice
        if offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<offset, 0...],
                self.values![.ellipsis, ..<offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result =
            if keys.dim(2) == 1 {
                updateInPlace(keys: keys, values: values)
            } else {
                updateConcat(keys: keys, values: values)
            }
        return result
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Note: RotatingKVCache doesn't set offset from keys like KVCache does
            // The offset is managed through meta_state
        }
    }

    public override var metaState: [String] {
        get {
            return [String(keep), String(maxCacheSize), String(step), String(offset), String(idx)]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("RotatingKVCache metaState must have exactly 5 values")
            }
            guard let keepVal = Int(newValue[0]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            if newValue[1] == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(newValue[1]) else {
                fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
        }
    }

    public override var isTrimmable: Bool {
        return offset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    /// Optimized mask creation for rotating cache with offset capping
    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            // Multi-token case
            let actualWindowSize = windowSize ?? maxCacheSize
            let cappedOffset = min(maxCacheSize - 1, offset)

            // Decide if we need an array mask
            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            // Single token case (n == 1)
            guard let windowSize = windowSize else {
                return .none
            }

            // May need a mask when window_size < max_size and cache has wrapped
            if offset >= windowSize, maxCacheSize > windowSize {
                var currentIdx = idx
                if currentIdx >= maxCacheSize {
                    currentIdx = 0
                }

                let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
                let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)

                // Roll the mask to account for rotation
                let rolledMask = roll(mask, shift: currentIdx + 1)

                return .array(rolledMask)
            }
            return .none
        }
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize.description), keep: \(keep), idx: \(idx)"
    }

    public override func copy() -> any KVCache {
        let new = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Convert to quantized cache
    /// Note: This is complex due to the rotating nature and temporal ordering
    public func toQuantized(
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) -> QuantizedKVCache {
        let cache = RotatingQuantizedKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        let s = state
        if !s.isEmpty {
            cache.setUnquantizedState(keys: s[0], values: s[1], offset: offset, writeIndex: idx)
        }
        cache.metaState = metaState
        return cache
    }

    public func toAffineInt4(groupSize: Int = TurboQuantKVCodec.affineInt4DefaultGroupSize)
        -> RotatingAffineInt4KVCache
    {
        let cache = RotatingAffineInt4KVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize
        )
        let s = state
        if !s.isEmpty {
            cache.setUnquantizedState(keys: s[0], values: s[1], offset: offset, writeIndex: idx)
        }
        cache.metaState = metaState
        return cache
    }

    public func toAffineK8V4(
        valueBits: Int = TurboQuantKVCodec.affineK8V4ValueBits,
        valueGroupSize: Int = TurboQuantKVCodec.affineK8V4ValueGroupSize,
        residualsPerGroup: Int = 0,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueRuntimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        layerIndex: Int? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil
    ) -> AffineK8V4KVCache {
        let cache = AffineK8V4KVCache(
            maxSize: maxCacheSize,
            keep: keep,
            valueGroupSize: valueGroupSize,
            valueBits: valueBits,
            residualsPerGroup: residualsPerGroup,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueRuntimeMode: sparseValueRuntimeMode,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason
        )
        let s = state
        if !s.isEmpty {
            cache.setUnquantizedState(keys: s[0], values: s[1], offset: offset)
        }
        return cache
    }
}

public final class RawOnlyRotatingKVCache: RotatingKVCache, RawOnlyKVCache {
    public override func copy() -> any KVCache {
        let new = RawOnlyRotatingKVCache(
            maxSize: self.maxSize ?? 0,
            keep: rotatingKeep,
            step: rotatingStep
        )
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }
}

private func resolvedKVQuantizationGroupSize(
    requested: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Int? {
    let requested = max(1, requested)
    let compatible = [32, 64, 128].filter {
        keyHeadDim.isMultiple(of: $0) && valueHeadDim.isMultiple(of: $0)
    }
    guard !compatible.isEmpty else { return nil }
    return compatible.min { lhs, rhs in
        let lhsDistance = abs(lhs - requested)
        let rhsDistance = abs(rhs - requested)
        if lhsDistance == rhsDistance {
            return lhs < rhs
        }
        return lhsDistance < rhsDistance
    }
}

/// Quantized KV cache for memory efficiency using MLX quantization
public class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    public var groupSize: Int
    public var bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        let parameters = resolvedQuantizationParameters(
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        self.groupSize = parameters.groupSize
        self.bits = parameters.bits
        self.step = 256
        self.mode = mode
        super.init()
    }

    fileprivate func adoptCompatibleAffineGroupSize(
        keyDimension: Int,
        valueDimension: Int,
        storageIsEmpty: Bool
    ) {
        guard storageIsEmpty, mode == .affine else { return }
        groupSize = compatibleAffineGroupSize(
            configuredGroupSize: groupSize,
            keyDimension: keyDimension,
            valueDimension: valueDimension
        )
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    /// Tree map equivalent for applying function to tuple elements
    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
        -> (T, T, T?)
    {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))

        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    /// Tree map for two tuples (like Python's tree_map over (keys, values))
    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
        _ tuple2: (MLXArray, MLXArray, MLXArray?)
    ) -> ((T, T, T?), (T, T, T?)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    /// Create initial quantized tuples (like Python's init_quant)
    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        // Create temporary zero arrays and quantize them using native MLX Swift
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits, mode: mode)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    /// Expand quantized tuple
    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray?
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    /// Update cache and return quantized tuples (Python's update_and_fetch)
    /// This is needed because `update` in Swift must return `(MLXArray, MLXArray)`
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset
        adoptCompatibleAffineGroupSize(
            keyDimension: kHeadDim,
            valueDimension: vHeadDim,
            storageIsEmpty: self.keys == nil && self.values == nil
        )
        guard
            resolvedKVQuantizationGroupSize(
            requested: groupSize,
            keyHeadDim: kHeadDim,
            valueHeadDim: vHeadDim
            ) != nil
        else {
            fatalError(
                "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(kHeadDim). Value head dim: \(vHeadDim)."
            )
        }

        // Check if we need to expand the cache
        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                // Trim if needed
                if prev % step != 0 {
                    // Use tree_map equivalent to trim both keys and values
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                // Expand using tree_map equivalent (Python's tree_map(expand_quant, ...))
                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
            } else {
                // Initialize new quantized cache
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits, mode: mode)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits, mode: mode)

        // Convert named tuples to positional tuples
        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        // Assign to storage
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        // Update each component of the quantized tuples
        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        if let qKeysBiases = qKeys.2 {
            currentKeys.2![.ellipsis, prev ..< offset, 0...] = qKeysBiases
        }

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        if let qValuesBiases = qValues.2 {
            currentValues.2![.ellipsis, prev ..< offset, 0...] = qValuesBiases
        }

        self.keys = currentKeys
        self.values = currentValues

        // Return quantized tuples
        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with QuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    /// Array of keys and values -- this will have either 6 elements or 4 elements (if biases are nil).
    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                // Trim to current offset using tree_map
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                // Flatten tuples to array for serialization
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ].compactMap { $0 }
            } else {
                // Flatten tuples to array for serialization
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
            }
        }
        set {
            switch newValue.count {
            case 0:
                keys = nil
                values = nil
                offset = 0
            case 4:
                // nil biases case
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }
            guard
                let offset = Int(newValue[1]),
                let groupSize = Int(newValue[2]),
                let bits = Int(newValue[3])
            else {
                fatalError("Failed to convert QuantizedKVCache metaState values to integers")
            }

            self.offset = offset
            self.groupSize = groupSize
            self.bits = bits
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    /// Convert to unquantized cache
    public func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            // Dequantize the current state using tree_map approach
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            // Set the unquantized state
            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

public final class AffineInt4KVCache: QuantizedKVCache, NativeAffineInt4KVCacheProtocol {
    public init(groupSize: Int = TurboQuantKVCodec.affineInt4DefaultGroupSize) {
        super.init(groupSize: groupSize, bits: TurboQuantKVCodec.affineInt4Bits, mode: .affine)
    }

    public override func copy() -> any KVCache {
        let new = AffineInt4KVCache(groupSize: groupSize)
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }
}

public final class AffineK8V4KVCache: BaseKVCache, NativeAffineK8V4KVCacheProtocol,
    TurboQuantCompressedKVCacheProtocol
{
    private typealias QuantizedTuple = (MLXArray, MLXArray, MLXArray?)
    private typealias ResidualTuple = (laneIndices: MLXArray, values: MLXArray)

    private var keys: QuantizedTuple?
    private var values: QuantizedTuple?
    private var valueResiduals: ResidualTuple?
    private var rawFallbackCache: KVCacheSimple?
    private let capacity: Int?
    private let keep: Int
    private let step = 256
    private var activeLength: Int = 0
    private var lastAttentionPath: TurboQuantAttentionPath
    private var lastAttentionFailureReason: String?

    public private(set) var keyGroupSize: Int
    public private(set) var keyBits: Int
    public private(set) var valueGroupSize: Int
    public private(set) var valueBits: Int
    public private(set) var residualsPerGroup: Int
    public let sparseValuePolicy: TurboQuantSparseValuePolicy
    public let sparseValueRuntimeMode: TurboQuantRuntimeMode
    public let layerIndex: Int?
    public let boundaryProtectedLayerCount: Int
    public let boundaryProtectionReason: String?
    public var groupSize: Int { keyGroupSize }
    public var bits: Int { keyBits }
    public let mode = QuantizationMode.affine
    public var precisionPolicy: TurboQuantKVPrecisionPolicy {
        TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: valueBits),
            boundary: .disabled
        )
    }

    public override var maxSize: Int? { capacity }
    public override var isTrimmable: Bool { true }

    public init(
        maxSize: Int? = nil,
        keep: Int = 0,
        keyGroupSize: Int = TurboQuantKVCodec.affineK8V4KeyGroupSize,
        keyBits: Int = TurboQuantKVCodec.affineK8V4KeyBits,
        valueGroupSize: Int = TurboQuantKVCodec.affineK8V4ValueGroupSize,
        valueBits: Int = TurboQuantKVCodec.affineK8V4ValueBits,
        residualsPerGroup: Int = 0,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueRuntimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        layerIndex: Int? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil
    ) {
        precondition(keyBits == TurboQuantKVCodec.affineK8V4KeyBits, "Affine K8/Vx requires 8-bit keys")
        precondition(
            TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits),
            "Affine K8/Vx value bits must be 2, 3, or 4"
        )
        precondition(
            residualsPerGroup == 0 || (valueBits == 2 && residualsPerGroup == 1),
            "Residual affine K8/Vx currently supports only V2 with residualsPerGroup == 1"
        )
        self.capacity = maxSize
        self.keep = max(0, keep)
        self.keyGroupSize = keyGroupSize
        self.keyBits = keyBits
        self.valueGroupSize = valueGroupSize
        self.valueBits = valueBits
        self.residualsPerGroup = max(0, residualsPerGroup)
        self.sparseValuePolicy = sparseValuePolicy
        self.sparseValueRuntimeMode = sparseValueRuntimeMode
        self.layerIndex = layerIndex
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.lastAttentionPath = (
            valueBits == TurboQuantKVCodec.affineK8V4ValueBits ? .affineK8V4Native
                : residualsPerGroup > 0 ? .affineK8VxResidual : .affineK8VxNative
        )
        super.init()
    }

    private func adoptCompatibleGroupSizes(keyDimension: Int, valueDimension: Int) {
        guard keys == nil, values == nil else { return }
        keyGroupSize = compatibleAffineGroupSize(
            configuredGroupSize: keyGroupSize,
            keyDimension: keyDimension,
            valueDimension: keyDimension
        )
        valueGroupSize = compatibleAffineGroupSize(
            configuredGroupSize: valueGroupSize,
            keyDimension: valueDimension,
            valueDimension: valueDimension
        )
    }

    public func setUnquantizedState(keys: MLXArray, values: MLXArray, offset: Int) {
        adoptCompatibleGroupSizes(keyDimension: keys.dim(3), valueDimension: values.dim(3))
        guard supportsMLXAffineKVQuantization(
            keyDimension: keys.dim(3),
            valueDimension: values.dim(3),
            keyGroupSize: keyGroupSize,
            valueGroupSize: valueGroupSize
        ) else {
            let rawCache = KVCacheSimple()
            _ = rawCache.update(keys: keys, values: values)
            rawCache.offset = offset
            rawFallbackCache = rawCache
            self.keys = nil
            self.values = nil
            self.valueResiduals = nil
            self.activeLength = keys.dim(2)
            self.offset = offset
            return
        }
        rawFallbackCache = nil
        self.keys = quantizedTuple(keys, groupSize: keyGroupSize, bits: keyBits)
        self.values = quantizedTuple(values, groupSize: valueGroupSize, bits: valueBits)
        if residualsPerGroup > 0, let quantizedValues = self.values {
            self.valueResiduals = residualTuple(values, quantizedValues: quantizedValues)
        }
        self.activeLength = keys.dim(2)
        self.offset = offset
        enforceCapacity()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    private func quantizedTuple(
        _ array: MLXArray,
        groupSize: Int,
        bits: Int
    ) -> QuantizedTuple {
        let q = quantized(array, groupSize: groupSize, bits: bits, mode: .affine)
        return (q.wq, q.scales, q.biases)
    }

    private func mapTuple(_ tuple: QuantizedTuple, _ transform: (MLXArray) -> MLXArray)
        -> QuantizedTuple
    {
        (transform(tuple.0), transform(tuple.1), tuple.2.map(transform))
    }

    private func tupleLength(_ tuple: QuantizedTuple) -> Int {
        tuple.0.dim(-2)
    }

    private func residualLength(_ tuple: ResidualTuple) -> Int {
        tuple.values.dim(-2)
    }

    private func sliceTuple(_ tuple: QuantizedTuple, _ range: Range<Int>) -> QuantizedTuple {
        mapTuple(tuple) { $0[.ellipsis, range, 0...] }
    }

    private func sliceResidualTuple(_ tuple: ResidualTuple, _ range: Range<Int>) -> ResidualTuple {
        (
            tuple.laneIndices[.ellipsis, range, 0...],
            tuple.values[.ellipsis, range, 0...]
        )
    }

    private func concatTuple(_ lhs: QuantizedTuple, _ rhs: QuantizedTuple) -> QuantizedTuple {
        let biases: MLXArray?
        if let leftBiases = lhs.2, let rightBiases = rhs.2 {
            biases = concatenated([leftBiases, rightBiases], axis: -2)
        } else {
            biases = nil
        }
        return (
            concatenated([lhs.0, rhs.0], axis: -2),
            concatenated([lhs.1, rhs.1], axis: -2),
            biases
        )
    }

    private func concatResidualTuple(_ lhs: ResidualTuple, _ rhs: ResidualTuple) -> ResidualTuple {
        (
            concatenated([lhs.laneIndices, rhs.laneIndices], axis: -2),
            concatenated([lhs.values, rhs.values], axis: -2)
        )
    }

    private func initQuant(
        dim: Int,
        shape: [Int],
        dtype: DType,
        groupSize: Int,
        bits: Int
    ) -> QuantizedTuple {
        quantizedTuple(
            MLXArray.zeros(shape + [dim], dtype: dtype),
            groupSize: groupSize,
            bits: bits
        )
    }

    private func initResiduals(shape: [Int], groupCount: Int, dtype: DType) -> ResidualTuple? {
        guard residualsPerGroup > 0 else { return nil }
        return (
            MLXArray.zeros(shape + [groupCount], dtype: .uint8),
            MLXArray.zeros(shape + [groupCount], dtype: dtype)
        )
    }

    private func expandTuple(_ tuple: QuantizedTuple, newShape: [Int]) -> QuantizedTuple {
        mapTuple(tuple) { array in
            concatenated(
                [array, MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)],
                axis: -2
            )
        }
    }

    private func expandResidualTuple(_ tuple: ResidualTuple, newShape: [Int]) -> ResidualTuple {
        (
            concatenated(
                [
                    tuple.laneIndices,
                    MLXArray.zeros(
                        newShape + [tuple.laneIndices.dim(-1)],
                        dtype: tuple.laneIndices.dtype
                    ),
                ],
                axis: -2
            ),
            concatenated(
                [
                    tuple.values,
                    MLXArray.zeros(newShape + [tuple.values.dim(-1)], dtype: tuple.values.dtype),
                ],
                axis: -2
            )
        )
    }

    private func residualTuple(
        _ originalValues: MLXArray,
        quantizedValues: QuantizedTuple
    ) -> ResidualTuple? {
        guard residualsPerGroup > 0 else { return nil }
        let valueDim = originalValues.dim(-1)
        guard valueDim > 0, valueDim % valueGroupSize == 0 else { return nil }
        let groupCount = valueDim / valueGroupSize
        let decoded = dequantized(
            quantizedValues.0,
            scales: quantizedValues.1,
            biases: quantizedValues.2,
            groupSize: valueGroupSize,
            bits: valueBits,
            mode: .affine,
            dtype: originalValues.dtype
        )
        let groupedResiduals = (originalValues - decoded).reshaped(
            [
                originalValues.dim(0),
                originalValues.dim(1),
                originalValues.dim(2),
                groupCount,
                valueGroupSize,
            ])
        let laneIndices = argMax(groupedResiduals.abs(), axis: -1)
        let residualValues = takeAlong(
            groupedResiduals,
            expandedDimensions(laneIndices, axis: -1),
            axis: -1
        ).squeezed(axis: -1)
        return (laneIndices.asType(.uint8), residualValues)
    }

    private func roundedStorageLength(for requiredLength: Int) -> Int {
        guard capacity == nil else { return max(0, capacity ?? requiredLength) }
        return ((requiredLength + step - 1) / step) * step
    }

    private func ensureStorage(
        batch: Int,
        kvHeads: Int,
        requiredLength: Int,
        keyDim: Int,
        valueDim: Int,
        dtype: DType
    ) {
        let targetLength = roundedStorageLength(for: requiredLength)
        guard targetLength > 0 else { return }

        if let currentKeys = keys, let currentValues = values {
            let storageLength = tupleLength(currentKeys)
            guard storageLength < targetLength else { return }
            let growBy = targetLength - storageLength
            let shape = [batch, kvHeads, growBy]
            keys = expandTuple(currentKeys, newShape: shape)
            values = expandTuple(currentValues, newShape: shape)
            if let currentResiduals = valueResiduals {
                valueResiduals = expandResidualTuple(currentResiduals, newShape: shape)
            } else if residualsPerGroup > 0 {
                valueResiduals = initResiduals(
                    shape: [batch, kvHeads, targetLength],
                    groupCount: valueDim / valueGroupSize,
                    dtype: dtype
                )
            }
        } else {
            let shape = [batch, kvHeads, targetLength]
            keys = initQuant(
                dim: keyDim,
                shape: shape,
                dtype: dtype,
                groupSize: keyGroupSize,
                bits: keyBits
            )
            values = initQuant(
                dim: valueDim,
                shape: shape,
                dtype: dtype,
                groupSize: valueGroupSize,
                bits: valueBits
            )
            valueResiduals = initResiduals(
                shape: shape,
                groupCount: valueDim / valueGroupSize,
                dtype: dtype
            )
        }
    }

    private func enforceCapacity() {
        guard let capacity, capacity > 0, let keys, let values else { return }
        let currentActiveLength = min(activeLength, tupleLength(keys))
        guard currentActiveLength > capacity else {
            activeLength = currentActiveLength
            return
        }
        let preservedPrefix = min(keep, capacity)
        let trimCount = currentActiveLength - capacity
        let prefix: QuantizedTuple? =
            preservedPrefix > 0 ? sliceTuple(keys, 0 ..< preservedPrefix) : nil
        let valuePrefix: QuantizedTuple? =
            preservedPrefix > 0 ? sliceTuple(values, 0 ..< preservedPrefix) : nil
        let tailStart = max(preservedPrefix, trimCount + preservedPrefix)
        let keyTail = sliceTuple(keys, tailStart ..< currentActiveLength)
        let valueTail = sliceTuple(values, tailStart ..< currentActiveLength)
        let residualPrefix: ResidualTuple? =
            preservedPrefix > 0 ? valueResiduals.map { sliceResidualTuple($0, 0 ..< preservedPrefix) } : nil
        let residualTail = valueResiduals.map {
            sliceResidualTuple($0, tailStart ..< currentActiveLength)
        }
        self.keys = prefix.map { concatTuple($0, keyTail) } ?? keyTail
        self.values = valuePrefix.map { concatTuple($0, valueTail) } ?? valueTail
        if let residualTail {
            self.valueResiduals = residualPrefix.map { concatResidualTuple($0, residualTail) }
                ?? residualTail
        }
        activeLength = min(capacity, currentActiveLength)
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let batch = keys.dim(0)
        let kvHeads = keys.dim(1)
        let steps = keys.dim(2)
        let keyDim = keys.dim(3)
        let valueDim = values.dim(3)
        adoptCompatibleGroupSizes(keyDimension: keyDim, valueDimension: valueDim)
        guard supportsMLXAffineKVQuantization(
            keyDimension: keyDim,
            valueDimension: valueDim,
            keyGroupSize: keyGroupSize,
            valueGroupSize: valueGroupSize
        ) else {
            let rawCache = rawFallbackCache ?? KVCacheSimple()
            let (rawKeys, rawValues) = rawCache.update(keys: keys, values: values)
            rawFallbackCache = rawCache
            self.keys = nil
            self.values = nil
            self.valueResiduals = nil
            offset = rawCache.offset
            activeLength = rawKeys.dim(2)
            return (
                placeholderQuantizedTuple(for: rawKeys, bits: keyBits),
                placeholderQuantizedTuple(for: rawValues, bits: valueBits)
            )
        }
        rawFallbackCache = nil
        let keyUpdate = quantizedTuple(keys, groupSize: keyGroupSize, bits: keyBits)
        let valueUpdate = quantizedTuple(values, groupSize: valueGroupSize, bits: valueBits)
        let residualUpdate = residualTuple(values, quantizedValues: valueUpdate)

        if capacity.map({ activeLength + steps <= $0 }) ?? true {
            ensureStorage(
                batch: batch,
                kvHeads: kvHeads,
                requiredLength: activeLength + steps,
                keyDim: keyDim,
                valueDim: valueDim,
                dtype: keys.dtype
            )
            guard self.keys != nil, self.values != nil else {
                fatalError("AffineK8VxKVCache storage was not initialized")
            }
            let range = activeLength ..< (activeLength + steps)
            self.keys!.0[.ellipsis, range, 0...] = keyUpdate.0
            self.keys!.1[.ellipsis, range, 0...] = keyUpdate.1
            if let keyBiases = keyUpdate.2 {
                self.keys!.2![.ellipsis, range, 0...] = keyBiases
            }
            self.values!.0[.ellipsis, range, 0...] = valueUpdate.0
            self.values!.1[.ellipsis, range, 0...] = valueUpdate.1
            if let valueBiases = valueUpdate.2 {
                self.values!.2![.ellipsis, range, 0...] = valueBiases
            }
            if let residualUpdate {
                if valueResiduals == nil {
                    valueResiduals = initResiduals(
                        shape: [batch, kvHeads, tupleLength(self.values!)],
                        groupCount: valueDim / valueGroupSize,
                        dtype: values.dtype
                    )
                }
                valueResiduals!.laneIndices[.ellipsis, range, 0...] = residualUpdate.laneIndices
                valueResiduals!.values[.ellipsis, range, 0...] = residualUpdate.values
            }
            activeLength += steps
        } else {
            // Capacity overflow is not the normal long-context benchmark path: it only
            // happens after the admitted window is exceeded. Preserve the configured
            // prefix and latest tail while keeping the hot update semantics correct.
            let currentKeys = getQuantizedState()?.0
            let currentValues = getQuantizedState()?.1
            let currentResiduals = getValueResidualTuple()
            let combinedKeys = currentKeys.map { concatTuple($0, keyUpdate) } ?? keyUpdate
            let combinedValues = currentValues.map { concatTuple($0, valueUpdate) } ?? valueUpdate
            self.keys = combinedKeys
            self.values = combinedValues
            self.valueResiduals =
                if let residualUpdate {
                    currentResiduals.map { concatResidualTuple($0, residualUpdate) }
                        ?? residualUpdate
                } else {
                    nil
                }
            activeLength = tupleLength(combinedKeys)
            enforceCapacity()
        }

        offset += keys.dim(2)
        return getQuantizedState()!
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys, let values else { return nil }
        guard activeLength > 0 else { return nil }
        return (
            sliceTuple(keys, 0 ..< min(activeLength, tupleLength(keys))),
            sliceTuple(values, 0 ..< min(activeLength, tupleLength(values)))
        )
    }

    public func exactRawStateIfComplete() -> (keys: MLXArray, values: MLXArray)? {
        guard let rawState = rawFallbackCache?.state, rawState.count == 2 else { return nil }
        return (rawState[0], rawState[1])
    }

    private func getValueResidualTuple() -> ResidualTuple? {
        guard let valueResiduals, activeLength > 0 else { return nil }
        return sliceResidualTuple(
            valueResiduals,
            0 ..< min(activeLength, residualLength(valueResiduals))
        )
    }

    public func getValueResidualState() -> AffineK8VxResidualState? {
        guard let residuals = getValueResidualTuple(), residualsPerGroup > 0 else { return nil }
        return AffineK8VxResidualState(
            laneIndices: residuals.laneIndices,
            values: residuals.values,
            residualsPerGroup: residualsPerGroup
        )
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `AffineK8V4KVCache`. Use `updateQuantized` instead."
        )
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let activeKeys = sliceTuple(keys, 0 ..< min(activeLength, tupleLength(keys)))
            let activeValues = sliceTuple(values, 0 ..< min(activeLength, tupleLength(values)))
            var arrays = [
                activeKeys.0, activeKeys.1, activeKeys.2, activeValues.0, activeValues.1,
                activeValues.2,
            ].compactMap { $0 }
            if let residuals = getValueResidualTuple() {
                arrays.append(residuals.laneIndices)
                arrays.append(residuals.values)
            }
            return arrays
        }
        set {
            switch newValue.count {
            case 4:
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
                valueResiduals = nil
                activeLength = tupleLength(keys!)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
                valueResiduals = nil
                activeLength = tupleLength(keys!)
            case 8:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
                valueResiduals = (
                    newValue[6].dtype == .uint8 ? newValue[6] : newValue[6].asType(.uint8),
                    newValue[7]
                )
                activeLength = tupleLength(keys!)
            default:
                keys = nil
                values = nil
                valueResiduals = nil
                offset = 0
                activeLength = 0
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(offset),
                capacity.map(String.init) ?? "None",
                String(keep),
                String(keyGroupSize),
                String(keyBits),
                String(valueGroupSize),
                String(valueBits),
                mode.rawValue,
                String(activeLength),
                lastAttentionPath.rawValue,
                lastAttentionFailureReason ?? "None",
                String(residualsPerGroup),
            ]
        }
        set {
            guard let offsetValue = newValue.first.flatMap(Int.init) else { return }
            offset = offsetValue
            if newValue.count >= 7 {
                keyGroupSize = Int(newValue[3]) ?? keyGroupSize
                keyBits = Int(newValue[4]) ?? keyBits
                valueGroupSize = Int(newValue[5]) ?? valueGroupSize
                valueBits = Int(newValue[6]) ?? valueBits
            }
            if newValue.count >= 12 {
                residualsPerGroup = max(0, Int(newValue[11]) ?? residualsPerGroup)
            }
            if newValue.count >= 9, let restoredActiveLength = Int(newValue[8]) {
                activeLength = restoredActiveLength
            } else if let keys {
                activeLength = tupleLength(keys)
            }
            if newValue.count >= 10, let path = TurboQuantAttentionPath(rawValue: newValue[9]) {
                lastAttentionPath = path
            }
            if newValue.count >= 11, newValue[10] != "None" {
                lastAttentionFailureReason = newValue[10]
            }
        }
    }

    public func recordNativeAffineK8V4AttentionPath(
        _ path: TurboQuantAttentionPath,
        failureReason: String? = nil
    ) {
        lastAttentionPath = path
        lastAttentionFailureReason = failureReason
    }

    public var activeAttentionPath: TurboQuantAttentionPath {
        lastAttentionPath
    }

    public var attentionFailureReason: String? {
        lastAttentionFailureReason
    }

    public var attentionDiagnostics: TurboQuantAttentionDiagnostics {
        let availability = TurboQuantKernelAvailability.current
        let sparseSelection = TurboQuantSparseValueSelection.off.resolved(
            runtimeMode: sparseValueRuntimeMode,
            contextLength: activeLength,
            policy: sparseValuePolicy
        )
        let lifecycle: TurboQuantCacheLifecycle =
            activeLength > 0
            ? .compressedCommitted(
                logicalLength: activeLength,
                capacity: keys.map(tupleLength) ?? activeLength
            )
            : .empty
        return TurboQuantAttentionDiagnostics(
            layerIndex: layerIndex,
            metalAttentionAvailable: availability.supportsMetalPolarQJLAttention,
            activeAttentionPath: lastAttentionPath,
            nativeBackend: availability.attentionCapabilities.nativeCompressedAttention == true
                ? "nativeMLX" : nil,
            nativeBackendVersion: availability.attentionCapabilities.nativeBackendVersion,
            nativeFallbackReason: lastAttentionFailureReason
                ?? availability.attentionCapabilities.nativeFallbackReason,
            selectedKernelProfile: availability.selectedKernelProfile,
            selfTestStatus: availability.selfTestStatus,
            selfTestFailureReason: availability.selfTestFailureReason,
            optimizationPolicy: .preferThroughput,
            fallbackReason: lastAttentionFailureReason,
            lastUnsupportedShape: nil,
            rawFallbackAllocated: false,
            cacheLifecycle: lifecycle,
            sparseVEnabled: sparseSelection.isEnabled,
            sparseVThreshold: sparseSelection.resolvedThreshold,
            sparseVSelectionMode: sparseSelection.mode,
            sparseVTopK: sparseSelection.topK,
            sparseVCumulativeMass: sparseSelection.cumulativeMass,
            sparseVMaxTopK: sparseSelection.maxTopK,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            keyBits: keyBits,
            valueBits: valueBits,
            keyGroupSize: keyGroupSize,
            valueGroupSize: valueGroupSize
        )
    }

    public func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot {
        let activeKeys = keys.map { sliceTuple($0, 0 ..< min(activeLength, tupleLength($0))) }
        let activeValues = values.map { sliceTuple($0, 0 ..< min(activeLength, tupleLength($0))) }
        let activeResiduals = getValueResidualTuple()
        let keyBytes = activeKeys.map(tupleBytes) ?? 0
        let valueBytes = (activeValues.map(tupleBytes) ?? 0)
            + (activeResiduals.map(residualBytes) ?? 0)
        let storageCapacity = keys.map(tupleLength) ?? capacity ?? 0
        let residualSuffix = residualsPerGroup > 0 ? "ResidualR\(residualsPerGroup)" : ""

        return TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription:
                "affineK8V\(valueBits)\(residualSuffix)Native(logicalLength:\(activeLength),capacity:\(storageCapacity))",
            logicalLength: activeLength,
            capacity: storageCapacity,
            pinnedPrefixLength: min(keep, activeLength),
            ringOffset: 0,
            keyBytes: keyBytes,
            valueBytes: valueBytes,
            rawShadowAllocated: false,
            packedFallbackAllocated: false,
            lastAttentionPath: lastAttentionPath.rawValue,
            lastFailure: lastAttentionFailureReason,
            kvCodec: valueBits == TurboQuantKVCodec.affineK8V4ValueBits ? .affineK8V4 : .affineK8Vx,
            quantizationMode: mode.rawValue,
            keyBits: keyBits,
            groupSize: keyGroupSize,
            valueBits: valueBits,
            valueGroupSize: valueGroupSize,
            selectedPath: lastAttentionPath.rawValue,
            fallbackReason: lastAttentionFailureReason,
            activeCacheAllocated: false
        )
    }

    private func tupleBytes(_ tuple: QuantizedTuple) -> Int {
        tuple.0.nbytes + tuple.1.nbytes + (tuple.2?.nbytes ?? 0)
    }

    private func residualBytes(_ tuple: ResidualTuple) -> Int {
        tuple.laneIndices.nbytes + tuple.values.nbytes
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        guard trimmed > 0, let keys, let values else {
            offset -= trimmed
            return trimmed
        }
        let retainedLength = max(0, activeLength - min(trimmed, activeLength))
        activeLength = retainedLength
        if retainedLength == 0 {
            self.keys = nil
            self.values = nil
            self.valueResiduals = nil
        } else if retainedLength < tupleLength(keys) && retainedLength < step {
            // Trim small speculative prefixes to avoid carrying oversized scratch
            // storage after a rollback to the beginning of a prompt.
            let keepRange = 0 ..< retainedLength
            self.keys = sliceTuple(keys, keepRange)
            self.values = sliceTuple(values, keepRange)
            self.valueResiduals = valueResiduals.map { sliceResidualTuple($0, keepRange) }
        } else {
            self.keys = keys
            self.values = values
        }
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = AffineK8V4KVCache(
            maxSize: capacity,
            keep: keep,
            keyGroupSize: keyGroupSize,
            keyBits: keyBits,
            valueGroupSize: valueGroupSize,
            valueBits: valueBits,
            residualsPerGroup: residualsPerGroup,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueRuntimeMode: sparseValueRuntimeMode,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }
}

/// Rotating quantized KV cache for sliding-window attention.
///
/// Storage keeps the same rotating metadata as ``RotatingKVCache`` while
/// `updateQuantized` returns tuples in temporal order for quantized attention.
public class RotatingQuantizedKVCache: QuantizedKVCache {
    private typealias QuantizedTuple = (MLXArray, MLXArray, MLXArray?)

    private var keys: QuantizedTuple?
    private var values: QuantizedTuple?
    private var keep: Int
    private var step: Int
    private var maxCacheSize: Int
    private var writeIndex: Int = 0

    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { offset < maxCacheSize }
    public var ropeOffset: RoPEOffset { .scalar(offset) }

    public init(
        maxSize: Int,
        keep: Int = 0,
        step: Int = 256,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init(groupSize: groupSize, bits: bits, mode: mode)
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    internal func setUnquantizedState(
        keys: MLXArray,
        values: MLXArray,
        offset: Int,
        writeIndex: Int
    ) {
        self.keys = quantizedTuple(keys)
        self.values = quantizedTuple(values)
        self.offset = offset
        self.writeIndex = writeIndex
    }

    private func quantizedTuple(_ array: MLXArray) -> QuantizedTuple {
        let q = quantized(array, groupSize: groupSize, bits: bits, mode: mode)
        return (q.wq, q.scales, q.biases)
    }

    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> QuantizedTuple {
        quantizedTuple(MLXArray.zeros(shape + [dim], dtype: dtype))
    }

    private func tupleLength(_ tuple: QuantizedTuple) -> Int {
        tuple.0.dim(-2)
    }

    private func mapTuple(_ tuple: QuantizedTuple, _ transform: (MLXArray) -> MLXArray)
        -> QuantizedTuple
    {
        (transform(tuple.0), transform(tuple.1), tuple.2.map(transform))
    }

    private func concatTuple(_ lhs: QuantizedTuple, _ rhs: QuantizedTuple) -> QuantizedTuple {
        let biases: MLXArray?
        if let leftBiases = lhs.2, let rightBiases = rhs.2 {
            biases = concatenated([leftBiases, rightBiases], axis: -2)
        } else {
            biases = nil
        }
        return (
            concatenated([lhs.0, rhs.0], axis: -2),
            concatenated([lhs.1, rhs.1], axis: -2),
            biases
        )
    }

    private func sliceTuple(_ tuple: QuantizedTuple, _ range: Range<Int>) -> QuantizedTuple {
        mapTuple(tuple) { $0[.ellipsis, range, 0...] }
    }

    private func temporalOrder(_ tuple: QuantizedTuple) -> QuantizedTuple {
        let length = tupleLength(tuple)
        if writeIndex == length {
            return tuple
        } else if writeIndex < offset {
            let prefix = sliceTuple(tuple, 0 ..< min(keep, length))
            let tail = sliceTuple(tuple, writeIndex ..< length)
            let middle = writeIndex > keep ? sliceTuple(tuple, keep ..< writeIndex) : nil
            return middle.map { concatTuple(concatTuple(prefix, tail), $0) }
                ?? concatTuple(prefix, tail)
        } else {
            return sliceTuple(tuple, 0 ..< writeIndex)
        }
    }

    private func trimTuple(
        trimSize: Int,
        _ tuple: QuantizedTuple,
        append: QuantizedTuple? = nil
    ) -> QuantizedTuple {
        let trimmed: QuantizedTuple
        if trimSize > 0 {
            let prefix = sliceTuple(tuple, 0 ..< keep)
            let suffix = sliceTuple(tuple, (trimSize + keep) ..< tupleLength(tuple))
            trimmed = concatTuple(prefix, suffix)
        } else {
            trimmed = tuple
        }
        if let append {
            return concatTuple(trimmed, append)
        }
        return trimmed
    }

    private func expandTuple(_ tuple: QuantizedTuple, newShape: [Int]) -> QuantizedTuple {
        mapTuple(tuple) { array in
            concatenated(
                [array, MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)],
                axis: -2
            )
        }
    }

    private func assign(
        _ update: QuantizedTuple, to storage: inout QuantizedTuple, range: Range<Int>
    ) {
        storage.0[.ellipsis, range, 0...] = update.0
        storage.1[.ellipsis, range, 0...] = update.1
        if let updateBiases = update.2 {
            storage.2![.ellipsis, range, 0...] = updateBiases
        }
    }

    private func temporalQuantizedState() -> (QuantizedTuple, QuantizedTuple)? {
        guard let keys, let values else { return nil }
        let orderedKeys = temporalOrder(keys)
        let orderedValues = temporalOrder(values)
        let activeLength = min(tupleLength(orderedKeys), max(0, min(offset, maxCacheSize)))
        guard activeLength > 0 else { return nil }
        return (
            sliceTuple(orderedKeys, 0 ..< activeLength),
            sliceTuple(orderedValues, 0 ..< activeLength)
        )
    }

    public override func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        temporalQuantizedState()
    }

    public override func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let steps = keys.dim(2)
        let keyDim = keys.dim(3)
        let valueDim = values.dim(3)
        adoptCompatibleAffineGroupSize(
            keyDimension: keyDim,
            valueDimension: valueDim,
            storageIsEmpty: self.keys == nil && self.values == nil
        )
        let keyUpdate = quantizedTuple(keys)
        let valueUpdate = quantizedTuple(values)

        if steps == 1 {
            if self.keys == nil
                || (offset >= tupleLength(self.keys!) && tupleLength(self.keys!) < maxCacheSize)
            {
                let growBy = min(step, maxCacheSize - min(offset, maxCacheSize))
                let newShape = [B, nKVHeads, growBy]
                if let currentKeys = self.keys, let currentValues = self.values {
                    self.keys = expandTuple(currentKeys, newShape: newShape)
                    self.values = expandTuple(currentValues, newShape: newShape)
                } else {
                    self.keys = initQuant(dim: keyDim, shape: newShape, dtype: keys.dtype)
                    self.values = initQuant(dim: valueDim, shape: newShape, dtype: values.dtype)
                }
                writeIndex = min(offset, maxCacheSize)
            }

            if tupleLength(self.keys!) > maxCacheSize {
                let trimSize = tupleLength(self.keys!) - maxCacheSize
                self.keys = trimTuple(trimSize: trimSize, self.keys!)
                self.values = trimTuple(trimSize: trimSize, self.values!)
                writeIndex = maxCacheSize
            }

            if writeIndex == maxCacheSize {
                writeIndex = min(keep, maxCacheSize)
            }

            var currentKeys = self.keys!
            var currentValues = self.values!
            let range = writeIndex ..< (writeIndex + steps)
            assign(keyUpdate, to: &currentKeys, range: range)
            assign(valueUpdate, to: &currentValues, range: range)
            self.keys = currentKeys
            self.values = currentValues
            offset += steps
            writeIndex += steps
        } else {
            if let currentKeys = self.keys, let currentValues = self.values {
                var orderedKeys = temporalOrder(currentKeys)
                var orderedValues = temporalOrder(currentValues)
                writeIndex = tupleLength(orderedKeys)
                let trimSize = writeIndex - maxCacheSize + 1
                orderedKeys = trimTuple(trimSize: trimSize, orderedKeys, append: keyUpdate)
                orderedValues = trimTuple(trimSize: trimSize, orderedValues, append: valueUpdate)
                self.keys = orderedKeys
                self.values = orderedValues
            } else {
                self.keys = keyUpdate
                self.values = valueUpdate
            }
            offset += steps
            writeIndex = tupleLength(self.keys!)
        }

        return temporalQuantizedState()!
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (qKeys, qValues) = updateQuantized(keys: keys, values: values)
        return (
            dequantized(
                qKeys.0, scales: qKeys.1, biases: qKeys.2, groupSize: groupSize, bits: bits,
                mode: mode),
            dequantized(
                qValues.0, scales: qValues.1, biases: qValues.2, groupSize: groupSize, bits: bits,
                mode: mode)
        )
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
        }
        set {
            switch newValue.count {
            case 4:
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                return
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(keep), String(maxCacheSize), String(step), String(offset),
                String(writeIndex), String(groupSize), String(bits), mode.rawValue,
            ]
        }
        set {
            guard newValue.count >= 5,
                let keepValue = Int(newValue[0]),
                let maxSizeValue = Int(newValue[1]),
                let stepValue = Int(newValue[2]),
                let offsetValue = Int(newValue[3]),
                let writeIndexValue = Int(newValue[4])
            else {
                return
            }
            keep = keepValue
            maxCacheSize = maxSizeValue
            step = stepValue
            offset = offsetValue
            writeIndex = writeIndexValue
        }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        writeIndex = min(writeIndex, max(0, writeIndex - trimmed))
        return trimmed
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }
        let contextLength = min(offset, maxCacheSize)
        if returnArray || (windowSize != nil && contextLength + n > windowSize!) {
            return .array(createCausalMask(n: n, offset: contextLength, windowSize: windowSize))
        }
        return .causal
    }

    public override func copy() -> any KVCache {
        let new = RotatingQuantizedKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    public override func toUnquantized() -> KVCacheSimple {
        let cache = KVCacheSimple()
        guard let (keys, values) = temporalQuantizedState() else { return cache }
        let decodedKeys = dequantized(
            keys.0, scales: keys.1, biases: keys.2, groupSize: groupSize, bits: bits, mode: mode)
        let decodedValues = dequantized(
            values.0, scales: values.1, biases: values.2, groupSize: groupSize, bits: bits,
            mode: mode)
        cache.state = [decodedKeys, decodedValues]
        return cache
    }
}

public final class RotatingAffineInt4KVCache: RotatingQuantizedKVCache,
    NativeAffineInt4KVCacheProtocol
{
    public init(
        maxSize: Int,
        keep: Int = 0,
        step: Int = 256,
        groupSize: Int = TurboQuantKVCodec.affineInt4DefaultGroupSize
    ) {
        super.init(
            maxSize: maxSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: TurboQuantKVCodec.affineInt4Bits,
            mode: .affine
        )
    }

    public override func copy() -> any KVCache {
        let meta = metaState
        let new = RotatingAffineInt4KVCache(
            maxSize: Int(meta.dropFirst().first ?? "") ?? maxSize ?? 0,
            keep: Int(meta.first ?? "") ?? 0,
            step: meta.count > 2 ? Int(meta[2]) ?? 256 : 256,
            groupSize: groupSize
        )
        let s = state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = meta
        return new
    }
}

/// Chunked KV cache for processing large contexts in chunks
public class ChunkedKVCache: KVCacheSimple {
    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    public func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            keys.dim(2) >= chunkSize
        else { return }

        startPosition += keys.dim(2) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)
        let end = offset - startPosition
        self.keys![.ellipsis, prev ..< end, 0...] = keys
        self.values![.ellipsis, prev ..< end, 0...] = values

        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = ChunkedKVCache(chunkSize: chunkSize)
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public override var metaState: [String] {
        get {
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            } else {
                self.chunkSize = Int(newValue[0])
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }
}

/// Base cache for array-based state storage
public class ArraysCache: BaseKVCache {
    fileprivate var cache: [MLXArray?]
    internal var leftPadding: MLXArray?
    internal var lengths: MLXArray?

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get {
            return cache.compactMap { $0 }
        }
        set {
            cache = newValue.map { $0 as MLXArray? }
        }
    }

    public override func copy() -> any KVCache {
        let new = ArraysCache(size: cache.count)
        copyContents(to: new)
        return new
    }

    internal func copyContents(to new: ArraysCache) {
        new.cache = cache.map { $0?[.ellipsis] }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        new.lengths = self.lengths
    }

    internal var batchSize: Int {
        cache.lazy.compactMap { $0?.dim(0) }.first ?? leftPadding?.size ?? lengths?.size ?? 1
    }

    /// In-place filter to keep just the given indices in the cache
    public func filter(batchIndices: MLXArray) {
        cache = cache.map { c in
            c?[batchIndices]
        }
        leftPadding = leftPadding?[batchIndices]
        lengths = lengths?[batchIndices]
    }

    /// In-place extend this cache with the other cache
    public func extend(other: ArraysCache) {
        let aBatch = batchSize
        let bBatch = other.batchSize

        func concatenate(_ a: MLXArray?, _ b: MLXArray?) -> MLXArray? {
            guard let example = a ?? b else {
                return nil
            }

            let suffixShape = Array(example.shape.dropFirst())
            let dtype = example.dtype
            let lhs = a ?? MLXArray.zeros([aBatch] + suffixShape, dtype: dtype)
            let rhs = b ?? MLXArray.zeros([bBatch] + suffixShape, dtype: dtype)
            return MLX.concatenated([lhs, rhs])
        }

        cache = zip(cache, other.cache).map { c, o in
            concatenate(c, o)
        }
        leftPadding = concatenate(leftPadding, other.leftPadding)
        lengths = concatenate(lengths, other.lengths)
    }

    public override func prepare(lengths: [Int]?) {
        self.lengths = lengths.map { MLXArray($0) }
    }

    public override func prepare(lengths: MLXArray?) {
        self.lengths = lengths
    }

    public override func finalize() {
        lengths = nil
        leftPadding = nil
    }

    public func advance(_ N: Int) {
        if let currentLengths = lengths {
            lengths = currentLengths - N
        }
        if let currentLeftPadding = leftPadding {
            leftPadding = currentLeftPadding - N
        }
    }

    public var currentLengths: MLXArray? {
        lengths
    }

    internal var leftPaddingValues: [Int]? {
        guard let leftPadding else { return nil }
        return leftPadding.asArray(Int.self)
    }

    internal var lengthsValues: [Int]? {
        guard let lengths else { return nil }
        return lengths.asArray(Int.self)
    }

    internal var presentSlotIndices: [Int] {
        cache.enumerated().compactMap { (i, v) in v != nil ? i : nil }
    }

    internal var slotCount: Int { cache.count }

    /// Create attention mask based on left padding or prepared sequence lengths
    public func makeMask(N: Int) -> MLXArray? {
        let positions = MLXArray(0 ..< N)
        if let leftPadding {
            return positions .>= leftPadding[0..., .newAxis]
        } else if let lengths {
            return positions .< lengths[0..., .newAxis]
        } else {
            return nil
        }
    }

    // MARK: - Serialization

    /// metaState format: [slotCount, presentSlots, leftPadding?, lengths?]
    /// Legacy format (BaseKVCache default): [""]
    public override var metaState: [String] {
        get {
            let leftPaddingState = Self.serializeMetadata(leftPadding)
            let lengthsState = Self.serializeMetadata(lengths)
            var result = [
                "\(cache.count)",
                presentSlotIndices.map(String.init).joined(separator: ","),
            ]
            if let leftPaddingState {
                result.append(leftPaddingState)
            } else if lengthsState != nil {
                result.append("")
            }
            if let lengthsState {
                result.append(lengthsState)
            }
            return result
        }
        set {
            assertionFailure(
                "ArraysCache.metaState should not be set directly. Use restoreFromMetaState() instead"
            )
        }
    }

    /// Restore from saved metaState + state arrays. Handles both new (slot-aware) and legacy formats.
    internal func restoreFromMetaState(state: [MLXArray], savedMetaState: [String]) {
        // Detect new format: first element parses as int (slotCount), second element is present slots
        if savedMetaState.count >= 2, let slotCount = Int(savedMetaState[0]) {
            let presentSlots =
                savedMetaState[1].isEmpty
                ? [] : savedMetaState[1].split(separator: ",").compactMap { Int($0) }

            self.cache = Array(repeating: nil, count: slotCount)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < state.count {
                self.cache[slotIdx] = state[arrayIdx]
            }
            self.leftPadding = Self.metadataArray(savedMetaState, at: 2)
            self.lengths = Self.metadataArray(savedMetaState, at: 3)
        } else {
            // Legacy: best-effort, state is compacted
            self.cache = state.map { $0 as MLXArray? }
        }
    }

    private static func serializeMetadata(_ array: MLXArray?) -> String? {
        array?.asArray(Int.self).map(String.init).joined(separator: ",")
    }

    private static func metadataArray(_ state: [String], at index: Int) -> MLXArray? {
        guard state.indices.contains(index), !state[index].isEmpty else { return nil }
        return MLXArray(state[index].split(separator: ",").compactMap { Int($0) })
    }
}

/// Simple cache for Mamba-style state space models
public class MambaCache: ArraysCache {
    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }

    // MARK: - §4 speculative-rollback support (lever ① on the hybrid)
    //
    // A recurrent cache cannot be `trim`ed (no per-token KV to drop), so the hybrid is
    // excluded from speculation via `canTrimPromptCache`. Instead it supports
    // INDEX rollback: during a q_seq=k verify forward the GatedDeltaNet scan runs
    // step-by-step and calls `recordVerifyStep()` after each consumed verify-input
    // token, so `verifyStepStates[i]` is the (conv, ssm) state after consuming verify
    // token i. Accepting `consumed` committed tokens then commits `verifyStepStates
    // [consumed-1]` — an index, not a recompute or snapshot-restore (correct by
    // construction). `discardVerify()` restores the pre-verify state. This keeps the
    // projections batched (weight-amortized) and needs no Metal kernel change.
    private var verifyPreState: [MLXArray]?
    private var verifyPreOffset: Int = 0
    private var verifyStepStates: [[MLXArray]] = []
    private(set) public var inVerify = false

    /// True: this cache supports speculative rollback via index selection (vs `trim`).
    public var supportsSpeculativeRollback: Bool { true }

    /// Snapshot the pre-verify recurrent state and arm per-step recording.
    public func beginVerify() {
        verifyPreState = state.map { $0[.ellipsis] }
        verifyPreOffset = offset
        verifyStepStates.removeAll(keepingCapacity: true)
        inVerify = true
    }

    /// Record the recurrent state after one verify-input token was consumed.
    public func recordVerifyStep() {
        guard inVerify else { return }
        verifyStepStates.append(state.map { $0[.ellipsis] })
    }

    /// Commit the recurrent state to the point after `consumed` verify-input tokens
    /// (1-based; the greedy round always commits ≥1 token). Ends verify mode.
    public func selectAcceptedStep(consumed: Int) {
        defer { endVerify() }
        guard inVerify, consumed >= 1, consumed <= verifyStepStates.count else { return }
        state = verifyStepStates[consumed - 1].map { $0[.ellipsis] }
        offset = verifyPreOffset + consumed
    }

    /// Restore the pre-verify recurrent state (full discard / misprediction rollback).
    public func discardVerify() {
        defer { endVerify() }
        guard inVerify, let pre = verifyPreState else { return }
        state = pre.map { $0[.ellipsis] }
        offset = verifyPreOffset
    }

    private func endVerify() {
        inVerify = false
        verifyStepStates.removeAll(keepingCapacity: true)
        verifyPreState = nil
    }

    fileprivate func detachedGenerationState() -> [MLXArray] {
        var detached = [MLXArray]()
        for index in 0 ..< slotCount {
            guard let array = self[index] else { continue }
            let state = stopGradient(array)
            self[index] = state
            detached.append(state)
        }
        return detached
    }

    public override func copy() -> any KVCache {
        let new = MambaCache()
        copyContents(to: new)
        return new
    }
}

/// Composite cache that manages multiple sub-caches
public class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    /// Internal initializer for reconstruction from deserialized children
    internal init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        get { caches[index] }
        set { caches[index] = newValue }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override func copy() -> any KVCache {
        let copiedCaches = caches.map { $0.copy() }
        let new = CacheList(caches: copiedCaches)
        return new
    }

    /// Recursively apply a transformation to every non-composite child cache.
    ///
    /// `CacheList` children are descended into; any other cache is passed to
    /// `transform` and replaced by the returned value. This is the primitive
    /// used by dynamic cache quantization and other cache-wide rewrites for
    /// models with hybrid attention/recurrent caches (e.g. Falcon-H1).
    public func mapChildren(_ transform: (KVCache) -> KVCache) {
        caches = caches.map { child in
            if let list = child as? CacheList {
                list.mapChildren(transform)
                return list
            }
            return transform(child)
        }
    }

    public override func prepare(lengths: [Int]?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func prepare(lengths: MLXArray?) {
        caches.forEach { $0.prepare(lengths: lengths) }
    }

    public override func finalize() {
        caches.forEach { $0.finalize() }
    }

    public override var isTrimmable: Bool {
        caches.allSatisfy { $0.isTrimmable }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }

    /// Internal accessor for child caches (used by serialization)
    internal var children: [KVCache] { caches }

    internal func replaceChildren(_ transform: (KVCache) -> KVCache) {
        for index in caches.indices {
            caches[index] = transform(caches[index])
        }
    }

    // MARK: - Serialization

    /// metaState format: [childCount, (className, stateCount, metaStateCount, ...metaState)*]
    ///
    /// Like Python's CacheList.meta_state which returns [child_class_names, child_meta_states],
    /// but flattened for Swift's [String] format.
    public override var metaState: [String] {
        get {
            var result = ["\(caches.count)"]
            for cache in caches {
                let className = cacheClassName(cache)
                let meta = cache.metaState
                result.append(className)
                result.append("\(cache.state.count)")
                result.append("\(meta.count)")
                result.append(contentsOf: meta)
            }
            return result
        }
        set {
            assertionFailure(
                "CacheList.metaState should not be set directly. Use CacheList.fromState() instead")
        }
    }

    /// Reconstruct a CacheList from flattened state + metaState, like Python's from_state()
    internal static func fromState(state: [MLXArray], metaState: [String]) throws -> CacheList {
        guard let childCount = metaState.first.flatMap({ Int($0) }) else {
            throw KVCacheError(message: "CacheList metaState missing child count")
        }

        var children: [KVCache] = []
        var metaIdx = 1  // skip childCount
        var stateIdx = 0

        for _ in 0 ..< childCount {
            guard metaIdx + 2 < metaState.count else {
                throw KVCacheError(message: "CacheList metaState truncated")
            }
            let className = metaState[metaIdx]
            guard let stateCount = Int(metaState[metaIdx + 1]) else {
                throw KVCacheError(message: "CacheList: invalid stateCount for child")
            }
            guard let metaCount = Int(metaState[metaIdx + 2]) else {
                throw KVCacheError(message: "CacheList: invalid metaStateCount for child")
            }
            metaIdx += 3

            let childMeta = Array(metaState[metaIdx ..< min(metaIdx + metaCount, metaState.count)])
            metaIdx += metaCount

            let childState = Array(state[stateIdx ..< min(stateIdx + stateCount, state.count)])
            stateIdx += stateCount

            let child = try restoreCacheFromMetaState(
                className: className, state: childState, metaState: childMeta)
            children.append(child)
        }

        return CacheList(caches: children)
    }
}

/// Materialize recurrent/native cache state after generation steps.
///
/// Hybrid models such as Qwen3.5 and LFM2 use TurboQuant attention caches for
/// attention layers but retain Mamba-style native state for recurrent/linear
/// layers. Token generation evaluates the sampled token asynchronously, which
/// does not guarantee those native cache arrays are materialized before the
/// next step stores them. Detaching and evaluating only recurrent cache state
/// prevents lazy graph chains from accumulating while leaving standard
/// attention KV caches untouched.
@discardableResult
public func materializeRecurrentKVCacheState(_ caches: [KVCache], synchronize: Bool = true) -> Bool {
    let state = recurrentGenerationStateArrays(in: caches)
    guard !state.isEmpty else { return false }
    // `eval` is synchronous, so the recurrent state is materialized at this call regardless
    // of `synchronize`. Pass `synchronize: false` when the caller will immediately do its own
    // `synchronize()`+`clearCache()` over the generated token (the TokenIterator path), folding
    // the two per-token barriers on the hybrid recurrent path into one — see N6.
    eval(state)
    if synchronize {
        Stream.gpu.synchronize()
        Memory.clearCache()
    }
    return true
}

private func recurrentGenerationStateArrays(in caches: [KVCache]) -> [MLXArray] {
    caches.flatMap { recurrentGenerationStateArrays(in: $0) }
}

private func recurrentGenerationStateArrays(in cache: KVCache) -> [MLXArray] {
    if let cache = cache as? MambaCache {
        return cache.detachedGenerationState()
    }
    if let cache = cache as? CacheList {
        return recurrentGenerationStateArrays(in: cache.children)
    }
    return []
}

// MARK: - Error Types

struct KVCacheError: Error {
    let message: String
}

// MARK: - Utility Functions

/// Map a cache instance to its Python-compatible class name for serialization.
private func cacheClassName(_ cache: KVCache) -> String {
    switch cache {
    case is ChunkedKVCache: return "ChunkedKVCache"
    case is MambaCache: return "MambaCache"
    case is ArraysCache: return "ArraysCache"
    case is HybridTurboQuantKVCache: return "HybridTurboQuantKVCache"
    case is RotatingTurboQuantKVCache: return "RotatingTurboQuantKVCache"
    case is AffineK8V4KVCache: return "AffineK8V4KVCache"
    case is RotatingKVCache: return "RotatingKVCache"
    case is TurboQuantKVCache: return "TurboQuantKVCache"
    case is QuantizedKVCache: return "QuantizedKVCache"
    case is KVCacheSimple: return "KVCache"
    case is CacheList: return "CacheList"
    default: return "KVCache"
    }
}

/// Save a pre-computed prompt cache to a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws {
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    let cacheClasses = cache.map { cacheClassName($0) }

    // Flatten cache data using tree_flatten compatible structure: "i.j" format
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    // Create cache_metadata structure compatible with Python: [cache_info, metadata, cache_classes]
    var flattenedMetadata: [String: String] = [:]

    // Flatten cache_info as "0.i.j" (first element of cache_metadata)
    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    // Flatten user metadata as "1.key" (second element of cache_metadata)
    for (key, value) in metadata {
        flattenedMetadata["1.\(key)"] = value
    }

    // Flatten cache_classes as "2.i" (third element of cache_metadata)
    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    try save(arrays: flattenedData, metadata: flattenedMetadata, url: url)
}

/// Load a prompt cache from a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
/// - Returns: The prompt cache and the metadata
public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)

    // Unflatten arrays using tree_unflatten compatible logic
    let cacheData = unflattenArrays(arrays)

    // Unflatten metadata using tree_unflatten compatible logic
    let unflattenedMetadata = unflattenMetadata(metadata)

    // Extract cache_info, user_metadata, and cache_classes from unflattened structure
    // Structure: [cache_info, user_metadata, cache_classes]
    guard unflattenedMetadata.count >= 3 else {
        throw KVCacheError(message: "Invalid cache metadata format")
    }

    let cacheInfo = unflattenedMetadata[0] as? [[String]] ?? []
    let userMetadata = unflattenedMetadata[1] as? [String: String] ?? [:]
    let cacheClasses = unflattenedMetadata[2] as? [String] ?? []

    guard cacheData.count == cacheInfo.count && cacheData.count == cacheClasses.count else {
        throw KVCacheError(message: "Mismatch in cache counts")
    }

    // Reconstruct cache instances
    var caches: [KVCache] = []
    for i in 0 ..< cacheData.count {
        let className = cacheClasses[i]
        let info = i < cacheInfo.count ? cacheInfo[i] : []

        let cache = try restoreCacheFromMetaState(
            className: className, state: cacheData[i], metaState: info)
        caches.append(cache)
    }

    return (caches, userMetadata)
}

/// Reconstruct a single cache from its class name, state arrays, and metaState.
///
/// Like Python's `globals()[className].from_state(state, meta_state)`, each cache type
/// encodes enough info in `metaState` to reconstruct itself.
private func restoreCacheFromMetaState(
    className: String,
    state: [MLXArray],
    metaState: [String]
) throws -> KVCache {
    switch className {
    case "KVCache", "KVCacheSimple":
        let cache = KVCacheSimple()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "RotatingKVCache":
        guard metaState.count >= 5 else {
            throw KVCacheError(
                message: "Invalid RotatingKVCache metaState - expected 5 values")
        }
        if metaState[1] == "None" {
            throw KVCacheError(
                message:
                    "RotatingKVCache with maxSize=None is not supported.")
        }
        guard let maxSize = Int(metaState[1]) else {
            throw KVCacheError(
                message: "Failed to parse RotatingKVCache maxSize from: \(metaState[1])")
        }
        let cache = RotatingKVCache(maxSize: maxSize)
        cache.state = state
        cache.metaState = metaState
        return cache

    case "QuantizedKVCache":
        let groupSize = metaState.count > 2 ? Int(metaState[2]) ?? 64 : 64
        let bits = metaState.count > 3 ? Int(metaState[3]) ?? 8 : 8
        let mode = inferredQuantizationMode(
            stateCount: state.count,
            groupSize: groupSize,
            bits: bits
        )
        let cache = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        cache.state = state
        cache.metaState = metaState
        return cache

    case "AffineK8V4KVCache":
        let maxSize =
            metaState.count > 1 && metaState[1] != "None" ? Int(metaState[1]) : nil
        let keep = metaState.count > 2 ? Int(metaState[2]) ?? 0 : 0
        let cache = AffineK8V4KVCache(maxSize: maxSize, keep: keep)
        cache.state = state
        cache.metaState = metaState
        return cache

    case "TurboQuantKVCache":
        let preset =
            metaState.count > 4 ? TurboQuantPreset(rawValue: metaState[4]) ?? .turbo3_5 : .turbo3_5
        let groupSize = metaState.count > 2 ? Int(metaState[2]) ?? 64 : 64
        let backend =
            metaState.count > 5
            ? TurboQuantBackend(rawValue: metaState[5]) ?? .mlxPacked : .mlxPacked
        let seed =
            metaState.count > 6
            ? UInt64(metaState[6]) ?? defaultTurboQuantSeed : defaultTurboQuantSeed
        let valueBits = turboQuantValueBits(from: metaState)
        let kvCodec = turboQuantKVCodec(from: metaState, backend: backend)
        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            backend: backend,
            kvCodec: kvCodec,
            optimizationPolicy: .auto,
            seed: seed,
            valueBits: valueBits
        )
        if state.count == 10, cache.activeBackend != .metalPolarQJL {
            throw KVCacheError(
                message:
                    "Cannot restore compressed TurboQuantKVCache without a verified Metal TurboQuant backend"
            )
        }
        cache.state = state
        cache.metaState = metaState
        return cache

    case "RotatingTurboQuantKVCache":
        guard metaState.count >= 7 else {
            throw KVCacheError(
                message: "Invalid RotatingTurboQuantKVCache metaState - expected 7 values")
        }
        guard let maxSize = Int(metaState[1]) else {
            throw KVCacheError(
                message: "Failed to parse RotatingTurboQuantKVCache maxSize from: \(metaState[1])")
        }
        let keep = Int(metaState[0]) ?? 4
        let step = metaState.count > 2 ? Int(metaState[2]) ?? 256 : 256
        let preset = TurboQuantPreset(rawValue: metaState[5]) ?? .turbo3_5
        let groupSize = Int(metaState[6]) ?? 64
        let backend =
            metaState.count > 7
            ? TurboQuantBackend(rawValue: metaState[7]) ?? .mlxPacked : .mlxPacked
        let seed =
            metaState.count > 8
            ? UInt64(metaState[8]) ?? defaultTurboQuantSeed : defaultTurboQuantSeed
        let valueBits = turboQuantValueBits(from: metaState)
        let kvCodec = turboQuantKVCodec(from: metaState, backend: backend)
        let cache = RotatingTurboQuantKVCache(
            maxSize: maxSize,
            keep: keep,
            step: step,
            preset: preset,
            groupSize: groupSize,
            backend: backend,
            kvCodec: kvCodec,
            optimizationPolicy: .auto,
            seed: seed,
            valueBits: valueBits
        )
        if state.count == 10, cache.activeBackend != .metalPolarQJL {
            throw KVCacheError(
                message:
                    "Cannot restore compressed RotatingTurboQuantKVCache without a verified Metal TurboQuant backend"
            )
        }
        cache.state = state
        cache.metaState = metaState
        return cache

    case "HybridTurboQuantKVCache":
        return try HybridTurboQuantKVCache.restoreFromMetaState(
            state: state,
            metaState: metaState
        )

    case "ChunkedKVCache":
        let cache = ChunkedKVCache()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "MambaCache":
        let cache = MambaCache()
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache

    case "ArraysCache":
        let cache = ArraysCache(size: 0)
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache

    case "CacheList":
        return try CacheList.fromState(state: state, metaState: metaState)

    default:
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
}

private func turboQuantValueBits(from metaState: [String]) -> Int? {
    for item in metaState {
        if item.hasPrefix("valueBits=") {
            return Int(item.dropFirst("valueBits=".count))
        }
    }
    return nil
}

private func turboQuantKVCodec(
    from metaState: [String],
    backend: TurboQuantBackend
) -> TurboQuantKVCodec {
    for item in metaState {
        if item.hasPrefix("kvCodec="),
            let codec = TurboQuantKVCodec(rawValue: String(item.dropFirst("kvCodec=".count)))
        {
            return codec
        }
    }
    return turboQuantCompressedKVCodec(backend: backend)
}

/// Unflatten arrays from tree_flatten format (e.g., "0.1", "1.0") to nested structure
private func unflattenArrays(_ flatArrays: [String: MLXArray]) -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    // Parse all keys and organize by indices
    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        if components.count >= 2,
            let i = Int(components[0]),
            let j = Int(components[1])
        {
            if arrayMap[i] == nil {
                arrayMap[i] = [:]
            }
            arrayMap[i]![j] = array
        }
    }

    // Convert to ordered array structure
    var result: [[MLXArray]] = []
    let maxI = arrayMap.keys.max() ?? -1

    for i in 0 ... maxI {
        if let innerMap = arrayMap[i] {
            let maxJ = innerMap.keys.max() ?? -1
            var innerArray: [MLXArray] = []
            for j in 0 ... maxJ {
                if let array = innerMap[j] {
                    innerArray.append(array)
                }
            }
            result.append(innerArray)
        } else {
            result.append([])
        }
    }

    return result
}

/// Unflatten metadata from tree_flatten format to nested structure
private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            // Cache info: "0.i.j" format
            if let i = Int(components[1]), let j = Int(components[2]) {
                // Ensure cacheInfo is large enough
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                // Ensure inner array is large enough
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            // User metadata: "1.key" format
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            // Cache classes: "2.i" format
            if let i = Int(components[1]) {
                // Ensure cacheClasses is large enough
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}

/// Construct the model's cache for use when generating.
///
/// This function will defer the cache construction to the model if it has a
/// `newCache` method, otherwise it will make a default KV cache.
public func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    // The model already conforms to LanguageModel which has newCache
    // If it also conforms to KVCacheDimensionProvider, the extension will provide the implementation
    return model.newCache(parameters: parameters)
}

/// Legacy function for backwards compatibility
public func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return makePromptCache(model: model, parameters: parameters)
}

/// Fallback function to create cache when layer count is known
///
/// This function creates a default cache structure when the number of layers is known.
/// Use this when `makePromptCache` cannot determine the layer count automatically.
public func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    (0 ..< numLayers).map { layerIndex in
        makeAttentionKVCache(
            parameters: parameters,
            maxKVSize: maxKVSize,
            layerIndex: layerIndex,
            layerCount: numLayers
        )
    }
}

private func turboQuantProtectsBoundaryLayer(
    parameters: GenerateParameters?,
    layerIndex: Int?,
    layerCount: Int?,
    boundaryLayerIndex: Int? = nil,
    boundaryLayerCount: Int? = nil
) -> Bool {
    let effectiveLayerIndex = boundaryLayerIndex ?? layerIndex
    let effectiveLayerCount = boundaryLayerCount ?? layerCount
    guard parameters?.kvCacheStrategy.createsTurboQuantCacheImmediately == true,
        let effectiveLayerIndex,
        let effectiveLayerCount,
        effectiveLayerIndex >= 0,
        effectiveLayerCount > 0
    else {
        return false
    }
    return parameters?.effectiveTurboQuantPrecisionPolicy.protectsBoundaryLayer(
        layerIndex: effectiveLayerIndex,
        layerCount: effectiveLayerCount
    ) ?? false
}

private func turboQuantBoundaryDiagnosticsMetadata(
    parameters: GenerateParameters?,
    layerCount: Int?
) -> (protectedLayerCount: Int, reason: String?) {
    let precisionPolicy = parameters?.effectiveTurboQuantPrecisionPolicy
    let protectedLayerCount =
        layerCount.map {
            precisionPolicy?.protectedBoundaryLayerIndexes(layerCount: $0).count ?? 0
        } ?? 0
    let reason =
        protectedLayerCount > 0
        ? {
            switch precisionPolicy?.boundaryCachePrecision ?? .affineK8V4 {
            case .affineK8V4:
                return "K8/V4 boundary protection for low-bit K/V policy"
            case .raw:
                return "rawKV boundary protection for explicit compatibility policy"
            }
        }()
        : nil
    return (protectedLayerCount, reason)
}

private func resolvedAffineInt4GroupSize(
    kvGroupSize: Int,
    kvBits: Int?,
    kvCodec: TurboQuantKVCodec
) -> Int {
    if kvGroupSize == 64, kvBits != TurboQuantKVCodec.affineInt4Bits, kvCodec != .affineInt4 {
        return TurboQuantKVCodec.affineInt4DefaultGroupSize
    }
    return kvGroupSize
}

private func resolvedAffineK8VxValueBits(
    kvCodec: TurboQuantKVCodec,
    requestedValueBits: Int?
) -> Int {
    let valueBits = requestedValueBits ?? TurboQuantKVCodec.affineK8V4ValueBits
    guard kvCodec == .affineK8Vx || requestedValueBits != nil else {
        return TurboQuantKVCodec.affineK8V4ValueBits
    }
    return TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits)
        ? valueBits : TurboQuantKVCodec.affineK8V4ValueBits
}

private func resolvedAffineK8VxValueGroupSize(requestedValueGroupSize: Int?) -> Int {
    guard let requestedValueGroupSize else {
        return TurboQuantKVCodec.affineK8V4ValueGroupSize
    }
    return (requestedValueGroupSize == 32 || requestedValueGroupSize == 64)
        ? requestedValueGroupSize
        : TurboQuantKVCodec.affineK8V4ValueGroupSize
}

private func resolvedTurboQuantValueBits(
    preset: TurboQuantPreset,
    kvCodec: TurboQuantKVCodec,
    requestedValueBits: Int?
) -> Int? {
    requestedValueBits
        ?? (kvCodec == .polarWHT ? TurboQuantKVCodec.polarWHTDefaultValueBits : nil)
}

private func resolvedTurboQuantPrecisionPolicy(
    parameters: GenerateParameters?,
    preset: TurboQuantPreset,
    kvCodec: TurboQuantKVCodec,
    valueBits: Int?
) -> TurboQuantKVPrecisionPolicy? {
    if let resolved = parameters?.turboQuantResolvedPrecisionPolicy {
        return resolved
    }
    if let explicit = parameters?.turboQuantPrecisionPolicy {
        return explicit
    }
    guard parameters != nil else { return nil }
    guard kvCodec == .polarWHT else {
        return parameters?.effectiveTurboQuantPrecisionPolicy
    }
    return TurboQuantKVPrecisionPolicy.legacy(preset: preset, valueBits: valueBits)
}

private func rawAttentionKVCache(maxKVSize: Int?, keep: Int) -> KVCache {
    if let maxKVSize {
        return RotatingKVCache(maxSize: maxKVSize, keep: keep)
    }
    return KVCacheSimple()
}

private func makeTurboQuantBoundaryKVCache(
    parameters: GenerateParameters?,
    maxKVSize: Int?,
    keep: Int,
    layerIndex: Int?,
    boundaryProtectedLayerCount: Int,
    boundaryProtectionReason: String?
) -> KVCache {
    switch parameters?.effectiveTurboQuantPrecisionPolicy.boundaryCachePrecision ?? .affineK8V4 {
    case .affineK8V4:
        return AffineK8V4KVCache(
            maxSize: maxKVSize,
            keep: keep,
            sparseValuePolicy: parameters?.effectiveTurboQuantSparseValuePolicy ?? .off,
            sparseValueRuntimeMode: parameters?.turboQuantResolvedRuntimeMode
                ?? parameters?.turboQuantRuntimeMode
                ?? .capacityTurboQuant,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason
        )
    case .raw:
        return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
    }
}

private func resolvedKVLayerCodec(
    parameters: GenerateParameters?,
    layerIndex: Int?
) -> KVLayerCodec? {
    guard let policy = parameters?.kvLayerPolicy else { return nil }
    let codec =
        if let layerIndex {
            policy.codec(forLayerIndex: layerIndex)
        } else {
            policy.defaultCodec ?? .inherit
        }
    if case .inherit = codec { return nil }
    return codec
}

private func makeAttentionKVCache(
    codec: KVLayerCodec,
    parameters: GenerateParameters?,
    maxKVSize: Int?,
    keep: Int,
    layerIndex: Int?,
    layerCount: Int?
) -> KVCache {
    let quantizedKVStart = parameters?.quantizedKVStart ?? 0
    let boundaryMetadata = turboQuantBoundaryDiagnosticsMetadata(
        parameters: parameters,
        layerCount: layerCount
    )
    switch codec {
    case .inherit:
        return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
    case .rawFP16:
        return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
    case .mlxAffine(let bits, let groupSize):
        guard quantizedKVStart <= 0 else {
            return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
        }
        if let maxKVSize {
            return RotatingQuantizedKVCache(
                maxSize: maxKVSize,
                keep: keep,
                groupSize: groupSize,
                bits: bits,
                mode: .affine
            )
        }
        return QuantizedKVCache(groupSize: groupSize, bits: bits, mode: .affine)
    case .affineK8V4:
        guard quantizedKVStart <= 0 else {
            return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
        }
        return AffineK8V4KVCache(
            maxSize: maxKVSize,
            keep: keep,
            sparseValuePolicy: parameters?.effectiveTurboQuantSparseValuePolicy ?? .off,
            sparseValueRuntimeMode: parameters?.turboQuantResolvedRuntimeMode
                ?? parameters?.turboQuantRuntimeMode
                ?? .capacityTurboQuant,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
            boundaryProtectionReason: boundaryMetadata.reason
        )
    case .affineK8Vx(let valueBits):
        guard quantizedKVStart <= 0 else {
            return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
        }
        let resolvedValueBits = (
            TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits)
                ? valueBits : TurboQuantKVCodec.affineK8V4ValueBits
        )
        let resolvedValueGroupSize = resolvedAffineK8VxValueGroupSize(
            requestedValueGroupSize: parameters?.turboQuantValueGroupSize
        )
        return AffineK8V4KVCache(
            maxSize: maxKVSize,
            keep: keep,
            valueGroupSize: resolvedValueGroupSize,
            valueBits: resolvedValueBits,
            sparseValuePolicy: parameters?.effectiveTurboQuantSparseValuePolicy ?? .off,
            sparseValueRuntimeMode: parameters?.turboQuantResolvedRuntimeMode
                ?? parameters?.turboQuantRuntimeMode
                ?? .capacityTurboQuant,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
            boundaryProtectionReason: boundaryMetadata.reason
        )
    case .affineK8VxResidual(let valueBits, let residualsPerGroup):
        guard quantizedKVStart <= 0 else {
            return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
        }
        let resolvedValueBits = valueBits == 2 ? valueBits : 2
        let resolvedResidualsPerGroup = residualsPerGroup == 1 ? 1 : 0
        return AffineK8V4KVCache(
            maxSize: maxKVSize,
            keep: keep,
            valueBits: resolvedValueBits,
            residualsPerGroup: resolvedResidualsPerGroup,
            sparseValuePolicy: parameters?.effectiveTurboQuantSparseValuePolicy ?? .off,
            sparseValueRuntimeMode: parameters?.turboQuantResolvedRuntimeMode
                ?? parameters?.turboQuantRuntimeMode
                ?? .capacityTurboQuant,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
            boundaryProtectionReason: boundaryMetadata.reason
        )
    case .affineInt4:
        guard quantizedKVStart <= 0 else {
            return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
        }
        let groupSize = resolvedAffineInt4GroupSize(
            kvGroupSize: parameters?.kvGroupSize ?? TurboQuantKVCodec.affineInt4DefaultGroupSize,
            kvBits: parameters?.kvBits,
            kvCodec: parameters?.kvCodec ?? .polarQJL
        )
        if let maxKVSize {
            return RotatingAffineInt4KVCache(
                maxSize: maxKVSize,
                keep: keep,
                groupSize: groupSize
            )
        }
        return AffineInt4KVCache(groupSize: groupSize)
    case .turboQuant(let preset, let valueBits, let groupSize, let backend):
        guard quantizedKVStart <= 0 else {
            return rawAttentionKVCache(maxKVSize: maxKVSize, keep: keep)
        }
        let policy = parameters?.turboQuantOptimizationPolicy ?? .auto
        let fallbackPolicy = parameters?.turboQuantFallbackPolicy ?? .compressedDecodeAllowed
        let seed = parameters?.turboQuantSeed ?? defaultTurboQuantSeed
        let runtimeMode = parameters?.turboQuantResolvedRuntimeMode
            ?? parameters?.turboQuantRuntimeMode
            ?? .capacityTurboQuant
        let kvCodec = turboQuantCompressedKVCodec(backend: backend)
        let resolvedValueBits = resolvedTurboQuantValueBits(
            preset: preset,
            kvCodec: kvCodec,
            requestedValueBits: valueBits
        )
        let precisionPolicy = TurboQuantKVPrecisionPolicy.legacy(
            preset: preset,
            valueBits: resolvedValueBits
        )
        let sparseValuePolicy = parameters?.effectiveTurboQuantSparseValuePolicy ?? .off
        let sparseValueSelection = parameters?.effectiveTurboQuantSparseValueSelection ?? .off
        if runtimeMode == .throughputTurboQuant {
            return ThroughputTurboQuantKVCache(
                maxSize: maxKVSize,
                keep: keep,
                preset: preset,
                groupSize: groupSize,
                backend: backend,
                kvCodec: kvCodec,
                optimizationPolicy: policy,
                fallbackPolicy: fallbackPolicy,
                seed: seed,
                valueBits: resolvedValueBits,
                precisionPolicy: precisionPolicy,
                requestedRuntimeMode: parameters?.turboQuantRuntimeMode ?? .auto,
                sparseValuePolicy: sparseValuePolicy,
                sparseValueSelection: sparseValueSelection,
                residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes
            )
        }
        if let maxKVSize {
            return RotatingTurboQuantKVCache(
                maxSize: maxKVSize,
                keep: keep,
                preset: preset,
                groupSize: groupSize,
                backend: backend,
                kvCodec: kvCodec,
                optimizationPolicy: policy,
                fallbackPolicy: fallbackPolicy,
                seed: seed,
                valueBits: resolvedValueBits,
                precisionPolicy: precisionPolicy,
                requestedRuntimeMode: parameters?.turboQuantRuntimeMode ?? .auto,
                resolvedRuntimeMode: runtimeMode,
                sparseValuePolicy: sparseValuePolicy,
                sparseValueSelection: sparseValueSelection,
                layerIndex: layerIndex,
                residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes
            )
        }
        return TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            backend: backend,
            kvCodec: kvCodec,
            optimizationPolicy: policy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: resolvedValueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: parameters?.turboQuantRuntimeMode ?? .auto,
            resolvedRuntimeMode: runtimeMode,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            layerIndex: layerIndex,
            residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes
        )
    }
}

/// Construct a single standard attention KV cache, honoring TurboQuant generation parameters.
public func makeAttentionKVCache(
    parameters: GenerateParameters? = nil,
    maxKVSize: Int? = nil,
    keep: Int = 4,
    layerIndex: Int? = nil,
    layerCount: Int? = nil,
    boundaryLayerIndex: Int? = nil,
    boundaryLayerCount: Int? = nil
) -> KVCache {
    let resolvedMaxKVSize: Int? =
        if let requested = parameters?.maxKVSize, let local = maxKVSize {
            min(requested, local)
        } else {
            parameters?.maxKVSize ?? maxKVSize
        }
    let effectiveBoundaryLayerCount = boundaryLayerCount ?? layerCount
    let boundaryMetadata = turboQuantBoundaryDiagnosticsMetadata(
        parameters: parameters,
        layerCount: effectiveBoundaryLayerCount
    )
    let effectiveKeep =
        parameters?.effectiveTurboQuantSparseValueSelection.mode == .pageTopK ? 0 : keep

    if let codec = resolvedKVLayerCodec(parameters: parameters, layerIndex: layerIndex) {
        return makeAttentionKVCache(
            codec: codec,
            parameters: parameters,
            maxKVSize: resolvedMaxKVSize,
            keep: effectiveKeep,
            layerIndex: layerIndex,
            layerCount: layerCount
        )
    }

    if turboQuantProtectsBoundaryLayer(
        parameters: parameters,
        layerIndex: layerIndex,
        layerCount: layerCount,
        boundaryLayerIndex: boundaryLayerIndex,
        boundaryLayerCount: boundaryLayerCount
    ) {
        return makeTurboQuantBoundaryKVCache(
            parameters: parameters,
            maxKVSize: resolvedMaxKVSize,
            keep: effectiveKeep,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
            boundaryProtectionReason: boundaryMetadata.reason
        )
    }

    if parameters?.kvCacheStrategy == .hybridTurboQuant {
        return makeHybridTurboQuantKVCache(
            parameters: parameters,
            maxKVSize: resolvedMaxKVSize,
            layerIndex: layerIndex,
            layerCount: layerCount
        )
    }

    if parameters?.kvCacheStrategy.createsAffineK8VxCacheImmediately == true,
        (parameters?.quantizedKVStart ?? 0) <= 0
    {
        let valueBits = resolvedAffineK8VxValueBits(
            kvCodec: parameters?.kvCodec ?? .affineK8V4,
            requestedValueBits: parameters?.turboQuantValueBits
        )
        let valueGroupSize = resolvedAffineK8VxValueGroupSize(
            requestedValueGroupSize: parameters?.turboQuantValueGroupSize
        )
        let sparseValueRuntimeMode =
            parameters?.turboQuantResolvedRuntimeMode
            ?? parameters?.turboQuantRuntimeMode
            ?? .capacityTurboQuant
        return AffineK8V4KVCache(
            maxSize: resolvedMaxKVSize,
            keep: effectiveKeep,
            valueGroupSize: valueGroupSize,
            valueBits: valueBits,
            sparseValuePolicy: parameters?.effectiveTurboQuantSparseValuePolicy ?? .off,
            sparseValueRuntimeMode: sparseValueRuntimeMode,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
            boundaryProtectionReason: boundaryMetadata.reason
        )
    }

    if parameters?.kvCacheStrategy.createsAffineInt4CacheImmediately == true,
        (parameters?.quantizedKVStart ?? 0) <= 0
    {
        let groupSize = resolvedAffineInt4GroupSize(
            kvGroupSize: parameters?.kvGroupSize ?? TurboQuantKVCodec.affineInt4DefaultGroupSize,
            kvBits: parameters?.kvBits,
            kvCodec: parameters?.kvCodec ?? .polarQJL
        )
        if let maxKVSize = resolvedMaxKVSize {
            return RotatingAffineInt4KVCache(
                maxSize: maxKVSize,
                keep: effectiveKeep,
                groupSize: groupSize
            )
        }
        return AffineInt4KVCache(groupSize: groupSize)
    }

    if parameters?.kvCacheStrategy.createsTurboQuantCacheImmediately == true {
        let preset = parameters?.turboQuantPreset ?? .turbo3_5
        let backend = parameters?.turboQuantBackend ?? .metalPolarQJL
        let kvCodec = parameters?.kvCodec ?? turboQuantCompressedKVCodec(backend: backend)
        let groupSize = parameters?.kvGroupSize ?? 64
        let policy = parameters?.turboQuantOptimizationPolicy ?? .auto
        let fallbackPolicy = parameters?.turboQuantFallbackPolicy ?? .compressedDecodeAllowed
        let seed = parameters?.turboQuantSeed ?? defaultTurboQuantSeed
        let valueBits = resolvedTurboQuantValueBits(
            preset: preset,
            kvCodec: kvCodec,
            requestedValueBits: parameters?.turboQuantValueBits
        )
        let runtimeMode = parameters?.turboQuantResolvedRuntimeMode
            ?? parameters?.turboQuantRuntimeMode
            ?? .capacityTurboQuant
        let precisionPolicy = resolvedTurboQuantPrecisionPolicy(
            parameters: parameters,
            preset: preset,
            kvCodec: kvCodec,
            valueBits: valueBits
        )
        let sparseValuePolicy = parameters?.effectiveTurboQuantSparseValuePolicy ?? .off
        let sparseValueSelection = parameters?.effectiveTurboQuantSparseValueSelection ?? .off
        if runtimeMode == .throughputTurboQuant {
            return ThroughputTurboQuantKVCache(
                maxSize: resolvedMaxKVSize,
                keep: effectiveKeep,
                preset: preset,
                groupSize: groupSize,
                backend: backend,
                kvCodec: kvCodec,
                optimizationPolicy: policy,
                fallbackPolicy: fallbackPolicy,
                seed: seed,
                valueBits: valueBits,
                precisionPolicy: precisionPolicy,
                requestedRuntimeMode: parameters?.turboQuantRuntimeMode ?? .auto,
                sparseValuePolicy: sparseValuePolicy,
                sparseValueSelection: sparseValueSelection,
                boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
                boundaryProtectionReason: boundaryMetadata.reason,
                residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes
            )
        }
        if let maxKVSize = resolvedMaxKVSize {
            return RotatingTurboQuantKVCache(
                maxSize: maxKVSize,
                keep: effectiveKeep,
                preset: preset,
                groupSize: groupSize,
                backend: backend,
                kvCodec: kvCodec,
                optimizationPolicy: policy,
                fallbackPolicy: fallbackPolicy,
                seed: seed,
                valueBits: valueBits,
                precisionPolicy: precisionPolicy,
                requestedRuntimeMode: parameters?.turboQuantRuntimeMode ?? .auto,
                resolvedRuntimeMode: runtimeMode,
                sparseValuePolicy: sparseValuePolicy,
                sparseValueSelection: sparseValueSelection,
                layerIndex: layerIndex,
                boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
                boundaryProtectionReason: boundaryMetadata.reason,
                residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes
            )
        }
        return TurboQuantKVCache(
            preset: preset,
            groupSize: groupSize,
            backend: backend,
            kvCodec: kvCodec,
            optimizationPolicy: policy,
            fallbackPolicy: fallbackPolicy,
            seed: seed,
            valueBits: valueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: parameters?.turboQuantRuntimeMode ?? .auto,
            resolvedRuntimeMode: runtimeMode,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            layerIndex: layerIndex,
            boundaryProtectedLayerCount: boundaryMetadata.protectedLayerCount,
            boundaryProtectionReason: boundaryMetadata.reason,
            residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes
        )
    }

    if let maxKVSize = resolvedMaxKVSize {
        return RotatingKVCache(maxSize: maxKVSize, keep: effectiveKeep)
    }
    return KVCacheSimple()
}

private func makeHybridTurboQuantKVCache(
    parameters: GenerateParameters?,
    maxKVSize: Int?,
    layerIndex: Int? = nil,
    layerCount: Int? = nil
) -> HybridTurboQuantKVCache {
    let resolvedMaxKVSize =
        parameters?.maxKVSize
        ?? maxKVSize
        ?? parameters?.turboQuantRequestedContextLength
    return HybridTurboQuantKVCache(
        maxSize: resolvedMaxKVSize,
        hotWindowTokens: parameters?.turboQuantHotWindowTokens,
        coldBlockTokens: parameters?.turboQuantColdBlockTokens ?? 1024,
        coldBudgetTokens: parameters?.turboQuantColdBudgetTokens ?? 4096,
        maxColdBudgetTokens: parameters?.turboQuantMaxColdBudgetTokens ?? 8192,
        coldAttentionMode: parameters?.turboQuantColdAttentionMode ?? .selected,
        layerPolicy: parameters?.turboQuantLayerPolicy ?? .auto,
        preset: parameters?.turboQuantPreset ?? .turbo4v2,
        groupSize: parameters?.kvGroupSize ?? 64,
        backend: parameters?.turboQuantBackend ?? .metalPolarQJL,
        optimizationPolicy: parameters?.turboQuantOptimizationPolicy ?? .auto,
        fallbackPolicy: parameters?.turboQuantFallbackPolicy ?? .compressedDecodeAllowed,
        seed: parameters?.turboQuantSeed ?? defaultTurboQuantSeed,
        valueBits: parameters?.turboQuantValueBits,
        residentBudgetBytes: parameters?.turboQuantPerCacheResidentBudgetBytes,
        layerIndex: layerIndex,
        layerCount: layerCount,
        sparseValuePolicy: parameters?.effectiveTurboQuantSparseValuePolicy ?? .off,
        selectorPolicy: parameters?.turboQuantColdSelectorPolicy ?? .automatic,
        selectorHints: parameters?.turboQuantColdSelectorHints ?? []
    )
}

/// Construct a raw-only KV cache for model paths that read cache state directly.
public func makeRawAttentionKVCache(
    parameters: GenerateParameters? = nil,
    maxKVSize: Int? = nil,
    keep: Int = 4
) -> KVCache {
    if let maxKVSize = parameters?.maxKVSize ?? maxKVSize {
        return RawOnlyRotatingKVCache(maxSize: maxKVSize, keep: keep)
    }
    return RawOnlyKVCacheSimple()
}

/// Check if model's cache can be trimmed.
public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    cache.dropFirst().forEach { $0.trim(numTokens) }
    return cache.first?.trim(numTokens) ?? 0
}

// MARK: - Type Aliases

/// Standard KV cache - alias to KVCacheSimple for compatibility
public typealias StandardKVCache = KVCacheSimple

// MARK: - Quantized Attention Operations

private func quantizedHeadDimension(_ packed: MLXArray, bits: Int) -> Int? {
    guard bits > 0 else { return nil }
    let unpackedElements = packed.dim(-1) * 32
    guard unpackedElements % bits == 0 else { return nil }
    return unpackedElements / bits
}

private func normalizedQuantizedAttentionMask(
    _ mask: MLXFast.ScaledDotProductAttentionMaskMode
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch mask {
    case .arrays(let masks):
        return masks.first.map { .array($0) } ?? .none
    default:
        return mask
    }
}

private func supportsNativeQuantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?,
    parameters: ResolvedQuantizationParameters,
    mode: QuantizationMode
) -> Bool {
    guard parameters.groupSize > 0, parameters.bits > 0 else { return false }
    guard queries.ndim == 4,
        quantizedKeys.0.ndim == 4,
        quantizedKeys.1.ndim == 4,
        quantizedValues.0.ndim == 4,
        quantizedValues.1.ndim == 4
    else { return false }

    guard queries.dtype == .float16 || queries.dtype == .bfloat16 || queries.dtype == .float32,
        quantizedKeys.0.dtype == .uint32,
        quantizedValues.0.dtype == .uint32
    else { return false }

    let keyHeadDimension = quantizedHeadDimension(quantizedKeys.0, bits: parameters.bits)
    let valueHeadDimension = quantizedHeadDimension(quantizedValues.0, bits: parameters.bits)
    guard keyHeadDimension == queries.dim(-1),
        valueHeadDimension == queries.dim(-1),
        queries.dim(-1) % parameters.groupSize == 0
    else { return false }

    guard queries.dim(0) == quantizedKeys.0.dim(0),
        queries.dim(0) == quantizedValues.0.dim(0),
        quantizedKeys.0.dim(-3) > 0,
        quantizedKeys.0.dim(-3) == quantizedValues.0.dim(-3),
        quantizedKeys.0.dim(-2) == quantizedValues.0.dim(-2),
        queries.dim(-3) % quantizedKeys.0.dim(-3) == 0
    else { return false }

    let expectedScaleDimension = queries.dim(-1) / parameters.groupSize
    guard quantizedKeys.1.dim(-1) == expectedScaleDimension,
        quantizedValues.1.dim(-1) == expectedScaleDimension,
        quantizedKeys.1.dim(-3) == quantizedKeys.0.dim(-3),
        quantizedValues.1.dim(-3) == quantizedValues.0.dim(-3),
        quantizedKeys.1.dim(-2) == quantizedKeys.0.dim(-2),
        quantizedValues.1.dim(-2) == quantizedValues.0.dim(-2)
    else { return false }

    switch mode {
    case .affine:
        guard parameters.groupSize == 32 || parameters.groupSize == 64,
            parameters.bits == 4 || parameters.bits == 6 || parameters.bits == 8,
            let keyBiases = quantizedKeys.2,
            let valueBiases = quantizedValues.2,
            keyBiases.ndim == 4,
            valueBiases.ndim == 4,
            keyBiases.shape == quantizedKeys.1.shape,
            valueBiases.shape == quantizedValues.1.shape
        else { return false }
    case .mxfp4, .mxfp8, .nvfp4:
        guard quantizedKeys.2 == nil,
            quantizedValues.2 == nil,
            quantizedKeys.1.dtype == .uint8,
            quantizedValues.1.dtype == .uint8
        else { return false }
    }

    if case .array(let maskArray) = mask, maskArray.ndim > 4 {
        return false
    }

    if let sinks {
        guard sinks.ndim == 1,
            sinks.dim(0) == queries.dim(1),
            sinks.dtype.isFloatingPoint
        else { return false }
    }

    return true
}

public func supportsNativeAffineInt4ScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    groupSize: Int = TurboQuantKVCodec.affineInt4DefaultGroupSize
) -> Bool {
    let parameters = resolvedQuantizationParameters(
        groupSize: groupSize,
        bits: TurboQuantKVCodec.affineInt4Bits,
        mode: .affine
    )
    return supportsNativeQuantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        mask: normalizedQuantizedAttentionMask(mask),
        sinks: sinks,
        parameters: parameters,
        mode: .affine
    )
}

public func affineInt4NativeScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    groupSize: Int = TurboQuantKVCodec.affineInt4DefaultGroupSize
) throws -> MLXArray {
    let parameters = resolvedQuantizationParameters(
        groupSize: groupSize,
        bits: TurboQuantKVCodec.affineInt4Bits,
        mode: .affine
    )
    let nativeMask = normalizedQuantizedAttentionMask(mask)
    guard supportsNativeQuantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        mask: nativeMask,
        sinks: sinks,
        parameters: parameters,
        mode: .affine
    ) else {
        throw TurboQuantRuntimeFailure.unsupportedAttentionShape(
            "affine int4 native SDPA unsupported for q=\(queries.shape), k=\(quantizedKeys.0.shape), v=\(quantizedValues.0.shape), group_size=\(parameters.groupSize), mask=\(nativeMask), sinks=\(sinks?.shape.description ?? "nil")"
        )
    }

    return MLXFast.quantizedScaledDotProductAttention(
        queries: queries,
        keys: quantizedKeys.0,
        keyScales: quantizedKeys.1,
        values: quantizedValues.0,
        valueScales: quantizedValues.1,
        scale: scale,
        keyBiases: quantizedKeys.2,
        valueBiases: quantizedValues.2,
        mask: nativeMask,
        sinks: sinks,
        groupSize: parameters.groupSize,
        bits: parameters.bits,
        mode: .affine
    )
}

public func supportsNativeAffineK8V4ScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    keyGroupSize: Int = TurboQuantKVCodec.affineK8V4KeyGroupSize,
    keyBits: Int = TurboQuantKVCodec.affineK8V4KeyBits,
    valueGroupSize: Int = TurboQuantKVCodec.affineK8V4ValueGroupSize,
    valueBits: Int = TurboQuantKVCodec.affineK8V4ValueBits
) -> Bool {
    let mask = normalizedQuantizedAttentionMask(mask)
    if ProcessInfo.processInfo.environment["MLX_TURBOQUANT_DISABLE_K8V4_NATIVE"] == "1" {
        return false
    }
    guard Device.defaultDevice().deviceType == .gpu else {
        return false
    }

    guard keyGroupSize > 0, valueGroupSize > 0, keyBits > 0, valueBits > 0 else {
        return false
    }
    guard queries.ndim == 4,
        quantizedKeys.0.ndim == 4,
        quantizedKeys.1.ndim == 4,
        quantizedValues.0.ndim == 4,
        quantizedValues.1.ndim == 4,
        let keyBiases = quantizedKeys.2,
        let valueBiases = quantizedValues.2,
        keyBiases.ndim == 4,
        valueBiases.ndim == 4
    else { return false }

    guard queries.dim(-2) <= 32,
        queries.dim(-2) <= quantizedKeys.0.dim(-2)
    else { return false }

    guard queries.dtype == .float16 || queries.dtype == .bfloat16 || queries.dtype == .float32,
        quantizedKeys.0.dtype == .uint32,
        quantizedValues.0.dtype == .uint32
    else { return false }

    let keyHeadDimension = quantizedHeadDimension(quantizedKeys.0, bits: keyBits)
    let valueHeadDimension = quantizedHeadDimension(quantizedValues.0, bits: valueBits)
    guard keyHeadDimension == queries.dim(-1),
        valueHeadDimension == queries.dim(-1),
        queries.dim(-1) % keyGroupSize == 0,
        queries.dim(-1) % valueGroupSize == 0
    else { return false }
    guard [64, 128, 256, 512].contains(queries.dim(-1)) else {
        return false
    }

    guard queries.dim(0) == quantizedKeys.0.dim(0),
        queries.dim(0) == quantizedValues.0.dim(0),
        quantizedKeys.0.dim(-3) > 0,
        quantizedKeys.0.dim(-3) == quantizedValues.0.dim(-3),
        quantizedKeys.0.dim(-2) == quantizedValues.0.dim(-2),
        queries.dim(-3) % quantizedKeys.0.dim(-3) == 0
    else { return false }
    let gqaFactor = queries.dim(-3) / quantizedKeys.0.dim(-3)
    guard gqaFactor <= 32 else {
        return false
    }

    let expectedKeyScaleDimension = queries.dim(-1) / keyGroupSize
    let expectedValueScaleDimension = queries.dim(-1) / valueGroupSize
    guard quantizedKeys.1.dim(-1) == expectedKeyScaleDimension,
        quantizedValues.1.dim(-1) == expectedValueScaleDimension,
        quantizedKeys.1.dim(-3) == quantizedKeys.0.dim(-3),
        quantizedValues.1.dim(-3) == quantizedValues.0.dim(-3),
        quantizedKeys.1.dim(-2) == quantizedKeys.0.dim(-2),
        quantizedValues.1.dim(-2) == quantizedValues.0.dim(-2),
        keyBiases.shape == quantizedKeys.1.shape,
        valueBiases.shape == quantizedValues.1.shape
    else { return false }

    guard (keyGroupSize == 32 || keyGroupSize == 64),
        (valueGroupSize == 32 || valueGroupSize == 64),
        keyBits == TurboQuantKVCodec.affineK8V4KeyBits,
        TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits)
    else { return false }

    if case .array(let maskArray) = mask, maskArray.ndim > 4 {
        return false
    }

    if let sinks {
        guard sinks.ndim == 1,
            sinks.dim(0) == queries.dim(1),
            sinks.dtype.isFloatingPoint
        else { return false }
    }

    return true
}

public func mixedAffineK8V4ScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    keyGroupSize: Int = TurboQuantKVCodec.affineK8V4KeyGroupSize,
    keyBits: Int = TurboQuantKVCodec.affineK8V4KeyBits,
    valueGroupSize: Int = TurboQuantKVCodec.affineK8V4ValueGroupSize,
    valueBits: Int = TurboQuantKVCodec.affineK8V4ValueBits,
    sparseVThreshold: Float? = nil
) throws -> MLXArray {
    let nativeMask = normalizedQuantizedAttentionMask(mask)
    let resolvedSparseVThreshold = sparseVThreshold.flatMap { $0 > 0 ? $0 : nil }
    let sparseNativeSupported: Bool = {
        guard resolvedSparseVThreshold != nil else { return true }
        if sinks != nil || queries.dim(2) != 1 { return false }
        switch nativeMask {
        case .none, .causal:
            return true
        case .array, .arrays:
            return false
        }
    }()
    if sparseNativeSupported,
        supportsNativeAffineK8V4ScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        mask: nativeMask,
        sinks: sinks,
        keyGroupSize: keyGroupSize,
        keyBits: keyBits,
        valueGroupSize: valueGroupSize,
        valueBits: valueBits
    ), let keyBiases = quantizedKeys.2, let valueBiases = quantizedValues.2 {
        if let resolvedSparseVThreshold {
            return try MLXFast.mixedQuantizedScaledDotProductAttention(
                queries: queries,
                keys: quantizedKeys.0,
                keyScales: quantizedKeys.1,
                values: quantizedValues.0,
                valueScales: quantizedValues.1,
                keyBiases: keyBiases,
                valueBiases: valueBiases,
                mask: nativeMask,
                sinks: sinks,
                options: MixedQuantizedScaledDotProductAttentionOptions(
                    scale: scale,
                    keyGroupSize: keyGroupSize,
                    keyBits: keyBits,
                    valueGroupSize: valueGroupSize,
                    valueBits: valueBits,
                    sparseVThreshold: resolvedSparseVThreshold
                )
            )
        }
        return MLXFast.mixedQuantizedScaledDotProductAttention(
            queries: queries,
            keys: quantizedKeys.0,
            keyScales: quantizedKeys.1,
            values: quantizedValues.0,
            valueScales: quantizedValues.1,
            scale: scale,
            keyBiases: keyBiases,
            valueBiases: valueBiases,
            mask: nativeMask,
            sinks: sinks,
            keyGroupSize: keyGroupSize,
            keyBits: keyBits,
            valueGroupSize: valueGroupSize,
            valueBits: valueBits
        )
    }

    let queryHeadCount = queries.dim(1)
    let kvHeadCount = quantizedKeys.0.dim(1)
    guard kvHeadCount > 0, queryHeadCount % kvHeadCount == 0 else {
        throw TurboQuantRuntimeFailure.unsupportedAttentionShape(
            "affine K8/Vx native attention requires query heads to be a multiple of KV heads"
        )
    }

    func expandTuple(
        _ tuple: (MLXArray, MLXArray, MLXArray?),
        axis: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        (
            expandedDimensions(tuple.0, axis: axis),
            expandedDimensions(tuple.1, axis: axis),
            tuple.2.map { expandedDimensions($0, axis: axis) }
        )
    }

    let repeats = queryHeadCount / kvHeadCount
    var q = queries * scale
    var k = quantizedKeys
    var v = quantizedValues
    if repeats > 1 {
        q = unflatten(q, axis: 1, shape: [kvHeadCount, repeats])
        k = expandTuple(k, axis: 2)
        v = expandTuple(v, axis: 2)
    }

    var scores = quantizedMM(
        q,
        k.0,
        scales: k.1,
        biases: k.2,
        transpose: true,
        groupSize: keyGroupSize,
        bits: keyBits,
        mode: .affine
    )

    let maskArray: MLXArray? =
        switch mask {
        case .none:
            nil
        case .causal:
            createCausalMask(
                n: queries.dim(2),
                offset: max(0, quantizedKeys.0.dim(-2) - queries.dim(2))
            )
        case .array(let array):
            array
        case .arrays(let arrays):
            arrays.first
        }
    if let maskArray {
        if maskArray.dtype == .bool {
            scores = MLX.where(
                maskArray,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: scores.dtype)
            )
        } else {
            scores = scores + maskArray
        }
    }

    var weights = softmax(scores, axes: [-1], precise: true)
    if let resolvedSparseVThreshold {
        weights = MLX.where(
            weights .>= resolvedSparseVThreshold,
            weights,
            MLXArray.zeros(like: weights)
        )
    }
    var output = quantizedMM(
        weights,
        v.0,
        scales: v.1,
        biases: v.2,
        transpose: false,
        groupSize: valueGroupSize,
        bits: valueBits,
        mode: .affine
    )
    if repeats > 1 {
        output = flatten(output, startAxis: 1, endAxis: 2)
    }
    return output
}

public func mixedAffineK8VxResidualScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    residualState: AffineK8VxResidualState,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    keyGroupSize: Int = TurboQuantKVCodec.affineK8V4KeyGroupSize,
    keyBits: Int = TurboQuantKVCodec.affineK8V4KeyBits,
    valueGroupSize: Int = TurboQuantKVCodec.affineK8V4ValueGroupSize,
    valueBits: Int = 2,
    sparseVThreshold: Float? = nil
) throws -> MLXArray {
    guard residualState.residualsPerGroup == 1, valueBits == 2 else {
        throw TurboQuantRuntimeFailure.unsupportedAttentionShape(
            "affine K8/Vx residual attention currently supports only V2 residualsPerGroup == 1"
        )
    }

    let base = try mixedAffineK8V4ScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        scale: scale,
        mask: mask,
        sinks: sinks,
        keyGroupSize: keyGroupSize,
        keyBits: keyBits,
        valueGroupSize: valueGroupSize,
        valueBits: valueBits,
        sparseVThreshold: sparseVThreshold
    )

    let headDim = queries.dim(-1)
    let groupCount = headDim / valueGroupSize
    guard groupCount > 0,
        residualState.laneIndices.dim(-1) == groupCount,
        residualState.values.dim(-1) == groupCount
    else {
        throw TurboQuantRuntimeFailure.unsupportedAttentionShape(
            "affine K8/Vx residual state shape does not match query head dimension"
        )
    }

    let lanes = MLXArray(0 ..< Int32(valueGroupSize))
        .asType(residualState.laneIndices.dtype)
        .reshaped([1, 1, 1, 1, valueGroupSize])
    let residualMask = expandedDimensions(residualState.laneIndices, axis: -1) .== lanes
    var residualValues = MLX.where(
        residualMask,
        expandedDimensions(residualState.values, axis: -1),
        MLXArray.zeros(
            residualState.values.shape + [valueGroupSize],
            dtype: residualState.values.dtype
        )
    ).reshaped([
        residualState.values.dim(0),
        residualState.values.dim(1),
        residualState.values.dim(2),
        headDim,
    ])

    let queryHeadCount = queries.dim(1)
    let kvHeadCount = quantizedKeys.0.dim(1)
    guard kvHeadCount > 0, queryHeadCount % kvHeadCount == 0 else {
        throw TurboQuantRuntimeFailure.unsupportedAttentionShape(
            "affine K8/Vx residual attention requires query heads to be a multiple of KV heads"
        )
    }

    func expandTuple(
        _ tuple: (MLXArray, MLXArray, MLXArray?),
        axis: Int
    ) -> (MLXArray, MLXArray, MLXArray?) {
        (
            expandedDimensions(tuple.0, axis: axis),
            expandedDimensions(tuple.1, axis: axis),
            tuple.2.map { expandedDimensions($0, axis: axis) }
        )
    }

    let repeats = queryHeadCount / kvHeadCount
    var q = queries * scale
    var k = quantizedKeys
    if repeats > 1 {
        q = unflatten(q, axis: 1, shape: [kvHeadCount, repeats])
        k = expandTuple(k, axis: 2)
        residualValues = expandedDimensions(residualValues, axis: 2)
    }

    var scores = quantizedMM(
        q,
        k.0,
        scales: k.1,
        biases: k.2,
        transpose: true,
        groupSize: keyGroupSize,
        bits: keyBits,
        mode: .affine
    )

    let maskArray: MLXArray? =
        switch mask {
        case .none:
            nil
        case .causal:
            createCausalMask(
                n: queries.dim(2),
                offset: max(0, quantizedKeys.0.dim(-2) - queries.dim(2))
            )
        case .array(let array):
            array
        case .arrays(let arrays):
            arrays.first
        }
    if let maskArray {
        if maskArray.dtype == .bool {
            scores = MLX.where(
                maskArray,
                scores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: scores.dtype)
            )
        } else {
            scores = scores + maskArray
        }
    }

    var weights = softmax(scores, axes: [-1], precise: true)
    if let sparseVThreshold, sparseVThreshold > 0 {
        weights = MLX.where(
            weights .>= sparseVThreshold,
            weights,
            MLXArray.zeros(like: weights)
        )
    }

    var correction = matmul(weights.asType(.float32), residualValues.asType(.float32))
    if repeats > 1 {
        correction = flatten(correction, startAxis: 1, endAxis: 2)
    }
    return base + correction.asType(base.dtype)
}

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine
) -> MLXArray {
    let parameters = resolvedQuantizationParameters(groupSize: groupSize, bits: bits, mode: mode)
    let nativeMask = normalizedQuantizedAttentionMask(mask)

    if supportsNativeQuantizedScaledDotProductAttention(
        queries: queries,
        quantizedKeys: quantizedKeys,
        quantizedValues: quantizedValues,
        mask: nativeMask,
        sinks: sinks,
        parameters: parameters,
        mode: mode
    ) {
        return MLXFast.quantizedScaledDotProductAttention(
            queries: queries,
            keys: quantizedKeys.0,
            keyScales: quantizedKeys.1,
            values: quantizedValues.0,
            valueScales: quantizedValues.1,
            scale: scale,
            keyBiases: quantizedKeys.2,
            valueBiases: quantizedValues.2,
            mask: nativeMask,
            sinks: sinks,
            groupSize: parameters.groupSize,
            bits: parameters.bits,
            mode: mode
        )
    }

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: parameters.groupSize, bits: parameters.bits,
        mode: mode
    )

    // Apply mask
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, MLXArray(-Float.greatestFiniteMagnitude))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }

    if let sinks {
        precondition(
            sinks.ndim == 1 && sinks.dim(0) == nQHeads && sinks.dtype.isFloatingPoint,
            "attention sinks must be a floating point array with shape [query heads]"
        )
        let sinkScores: MLXArray
        if nRepeats > 1 {
            sinkScores = broadcast(
                expandedDimensions(
                    sinks.asType(scores.dtype).reshaped([nKVHeads, nRepeats]),
                    axes: [0, 3, 4]
                ),
                to: [B, nKVHeads, nRepeats, L, 1]
            )
        } else {
            sinkScores = broadcast(
                expandedDimensions(sinks.asType(scores.dtype), axes: [0, 2, 3]),
                to: [B, nQHeads, L, 1]
            )
        }
        scores = concatenated([sinkScores, scores], axis: -1)
    }

    let attentionWeights = softmax(scores, axis: -1)
    let attentionValues =
        sinks == nil
        ? attentionWeights
        : attentionWeights[.ellipsis, 1...]

    // Compute output using quantized matmul
    var output = quantizedMM(
        attentionValues, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: parameters.groupSize, bits: parameters.bits,
        mode: mode
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, output.dim(-1)])
    }

    return output
}

// MARK: - Dynamic Cache Quantization

/// Dynamically quantize KV caches during generation if conditions are met
///
/// Resolve a kvScheme string to (bits, groupSize) for affine quantization.
/// Returns nil for unrecognized schemes (custom schemes handle their own caches).
public func resolveAffineScheme(_ scheme: String?) -> (bits: Int, groupSize: Int)? {
    switch scheme {
    case "affine4": return (4, 64)
    case "affine8": return (8, 64)
    default: return nil
    }
}

/// Converts regular caches to quantized caches when:
/// - kvBits is specified (or kvScheme resolves to a built-in affine scheme)
/// - The cache is not already quantized
/// - The cache offset is at or beyond quantizedKVStart
///
/// - Parameters:
///   - cache: Array of KV caches to potentially quantize
///   - kvBits: Number of bits for quantization (nil = no quantization)
///   - kvGroupSize: Group size for quantization
///   - quantizedKVStart: Token count threshold to begin quantizing
///   - kvCacheStrategy: Cache strategy used for dynamic quantization
///   - kvCodec: KV codec used for affine and TurboQuant cache conversion
///   - turboQuantPreset: TurboQuant preset used when `kvCacheStrategy` is `.turboQuant`
///   - turboQuantBackend: Requested TurboQuant backend
///   - turboQuantOptimizationPolicy: TurboQuant optimization policy
///   - turboQuantFallbackPolicy: Fallback policy for compressed cache conversion and decode
///   - turboQuantSeed: Optional deterministic seed for TurboQuant encoding
///   - turboQuantValueBits: Optional value-bit override for TurboQuant caches
///   - turboQuantPrecisionPolicy: Optional key/value precision policy override
///   - turboQuantValueGroupSize: Optional value group-size override for TurboQuant caches
///   - turboQuantSparseValuePolicy: Sparse-value policy used during TurboQuant conversion
///   - turboQuantSparseValueSelection: Sparse-value selection mode used during TurboQuant conversion
///   - turboQuantResidentBudgetBytes: Optional resident-byte budget for converted TurboQuant caches
///   - spillMemoryWatermarkBytes: Optional live-memory watermark that triggers cache spill
///   - kvLayerPolicy: Optional per-layer precision policy for mixed KV conversion
///   - kvScheme: Scheme selector; overrides kvBits when it names a built-in
///     affine scheme ("affine4", "affine8"). Unrecognized schemes are left to
///     custom cache implementations and do not quantize here.
public func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Int?,
    kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0,
    kvCacheStrategy: KVCacheStrategy = .mlxAffine,
    kvCodec: TurboQuantKVCodec = .polarQJL,
    turboQuantPreset: TurboQuantPreset = .turbo3_5,
    turboQuantBackend: TurboQuantBackend = .metalPolarQJL,
    turboQuantOptimizationPolicy: TurboQuantOptimizationPolicy = .auto,
    turboQuantFallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
    turboQuantSeed: UInt64? = nil,
    turboQuantValueBits: Int? = nil,
    turboQuantPrecisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
    turboQuantValueGroupSize: Int? = nil,
    turboQuantSparseValuePolicy: TurboQuantSparseValuePolicy = .off,
    turboQuantSparseValueSelection: TurboQuantSparseValueSelection = .off,
    turboQuantResidentBudgetBytes: Int? = nil,
    spillMemoryWatermarkBytes: Int? = nil,
    kvLayerPolicy: KVLayerPolicy? = nil,
    kvScheme: String? = nil
) {
    guard !cache.isEmpty else { return }
    let resolvedAffineScheme = kvScheme.flatMap(resolveAffineScheme)
    let hasLayerPolicy = kvLayerPolicy != nil
    if kvCacheStrategy == .hybridTurboQuant && !hasLayerPolicy { return }
    // 2A spill: when live available memory falls below the watermark, re-encode the FP16 cache to
    // compressed mid-generation — even a plain-FP16 (.none) cache that fit at admission but outgrew
    // its projection. This converts an impending OOM into a graceful precision/throughput tradeoff.
    // The watermark is an on-device tuning constant; nil disables memory-triggered spilling.
    let underMemoryPressure: Bool = {
        guard let watermark = spillMemoryWatermarkBytes, watermark > 0 else { return false }
        return turboQuantAvailableProcessMemoryBytes() < watermark
    }()
    if kvCacheStrategy == .none && !underMemoryPressure && !hasLayerPolicy
        && resolvedAffineScheme == nil
    {
        return
    }
    let useAffineK8Vx = kvCacheStrategy.createsAffineK8VxCacheImmediately
    let useAffineInt4 = kvCacheStrategy == .affineInt4
    let spillToTurboQuant =
        kvCacheStrategy.canUseTurboQuant || (kvCacheStrategy == .none && underMemoryPressure)
    let effectiveAffineGroupSize = resolvedAffineScheme?.groupSize ?? kvGroupSize
    let resolvedBits =
        useAffineK8Vx ? TurboQuantKVCodec.affineK8V4KeyBits
        : useAffineInt4 ? TurboQuantKVCodec.affineInt4Bits
        : (spillToTurboQuant ? turboQuantPreset.effectiveBits : (resolvedAffineScheme?.bits ?? kvBits))
    // Under memory pressure, spill immediately — ignore the static token threshold.
    let effectiveQuantizedKVStart = underMemoryPressure ? 0 : quantizedKVStart
    let affineK8VxValueBits = resolvedAffineK8VxValueBits(
        kvCodec: kvCodec,
        requestedValueBits: turboQuantValueBits
    )
    let affineK8VxValueGroupSize = resolvedAffineK8VxValueGroupSize(
        requestedValueGroupSize: turboQuantValueGroupSize
    )
    let affineBoundaryMetadata: (protectedLayerCount: Int, reason: String?) = {
        guard let kvLayerPolicy else { return (0, nil) }
        let protectedLayerCount = cache.indices.filter { layerIndex in
            switch kvLayerPolicy.codec(forLayerIndex: layerIndex) {
            case .affineK8V4:
                true
            default:
                false
            }
        }.count
        guard protectedLayerCount > 0 else { return (0, nil) }
        return (
            protectedLayerCount,
            "K8/V4 boundary protection for low-bit K/V policy"
        )
    }()

    func policyCodec(for layerIndex: Int) -> KVLayerCodec? {
        guard let kvLayerPolicy else { return nil }
        let codec = kvLayerPolicy.codec(forLayerIndex: layerIndex)
        if case .inherit = codec { return nil }
        return codec
    }

    func codecRequiresConversion(_ codec: KVLayerCodec) -> Bool {
        switch codec {
        case .inherit, .rawFP16:
            false
        case .mlxAffine, .affineK8V4, .affineK8Vx, .affineK8VxResidual, .affineInt4,
            .turboQuant:
            true
        }
    }

    func isReadyForQuantization(_ item: KVCache, layerIndex: Int) -> Bool {
        if let list = item as? CacheList {
            return list.children.contains { isReadyForQuantization($0, layerIndex: layerIndex) }
        }
        if item is RawOnlyKVCache {
            return false
        }
        if item is QuantizedKVCacheProtocol {
            return false
        }
        if let codec = policyCodec(for: layerIndex) {
            guard codecRequiresConversion(codec) else { return false }
        } else if resolvedBits == nil {
            return false
        }
        if item is KVCacheSimple {
            return item.offset > 0 && item.offset >= effectiveQuantizedKVStart
        }
        if item is RotatingKVCache {
            return item.offset > 0 && item.offset >= effectiveQuantizedKVStart
        }
        return false
    }

    func convertedCache(_ item: KVCache, layerIndex: Int) -> KVCache {
        if let list = item as? CacheList {
            list.replaceChildren { convertedCache($0, layerIndex: layerIndex) }
            return list
        }
        if item is RawOnlyKVCache {
            return item
        }
        if item is QuantizedKVCacheProtocol {
            return item
        }
        if let codec = policyCodec(for: layerIndex) {
            switch codec {
            case .inherit, .rawFP16:
                return item
            case .mlxAffine(let bits, let groupSize):
                if let simpleCache = item as? KVCacheSimple {
                    return simpleCache.toQuantized(groupSize: groupSize, bits: bits)
                }
                if let rotatingCache = item as? RotatingKVCache {
                    return rotatingCache.toQuantized(groupSize: groupSize, bits: bits)
                }
            case .affineK8V4:
                if let simpleCache = item as? KVCacheSimple {
                    return simpleCache.toAffineK8V4(
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        layerIndex: layerIndex,
                        boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                        boundaryProtectionReason: affineBoundaryMetadata.reason
                    )
                }
                if let rotatingCache = item as? RotatingKVCache {
                    return rotatingCache.toAffineK8V4(
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        layerIndex: layerIndex,
                        boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                        boundaryProtectionReason: affineBoundaryMetadata.reason
                    )
                }
            case .affineK8Vx(let valueBits):
                let resolvedValueBits = (
                    TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits)
                        ? valueBits : TurboQuantKVCodec.affineK8V4ValueBits
                )
                if let simpleCache = item as? KVCacheSimple {
                    return simpleCache.toAffineK8V4(
                        valueBits: resolvedValueBits,
                        valueGroupSize: affineK8VxValueGroupSize,
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        layerIndex: layerIndex,
                        boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                        boundaryProtectionReason: affineBoundaryMetadata.reason
                    )
                }
                if let rotatingCache = item as? RotatingKVCache {
                    return rotatingCache.toAffineK8V4(
                        valueBits: resolvedValueBits,
                        valueGroupSize: affineK8VxValueGroupSize,
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        layerIndex: layerIndex,
                        boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                        boundaryProtectionReason: affineBoundaryMetadata.reason
                    )
                }
            case .affineK8VxResidual(let valueBits, let residualsPerGroup):
                let resolvedValueBits = valueBits == 2 ? valueBits : 2
                let resolvedResidualsPerGroup = residualsPerGroup == 1 ? 1 : 0
                if let simpleCache = item as? KVCacheSimple {
                    return simpleCache.toAffineK8V4(
                        valueBits: resolvedValueBits,
                        residualsPerGroup: resolvedResidualsPerGroup,
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        layerIndex: layerIndex,
                        boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                        boundaryProtectionReason: affineBoundaryMetadata.reason
                    )
                }
                if let rotatingCache = item as? RotatingKVCache {
                    return rotatingCache.toAffineK8V4(
                        valueBits: resolvedValueBits,
                        residualsPerGroup: resolvedResidualsPerGroup,
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        layerIndex: layerIndex,
                        boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                        boundaryProtectionReason: affineBoundaryMetadata.reason
                    )
                }
            case .affineInt4:
                let groupSize = resolvedAffineInt4GroupSize(
                    kvGroupSize: kvGroupSize,
                    kvBits: TurboQuantKVCodec.affineInt4Bits,
                    kvCodec: .affineInt4
                )
                if let simpleCache = item as? KVCacheSimple {
                    return simpleCache.toAffineInt4(groupSize: groupSize)
                }
                if let rotatingCache = item as? RotatingKVCache {
                    return rotatingCache.toAffineInt4(groupSize: groupSize)
                }
            case .turboQuant(let preset, let valueBits, let groupSize, let backend):
                if let simpleCache = item as? KVCacheSimple {
                    return simpleCache.toTurboQuant(
                        preset: preset,
                        groupSize: groupSize,
                        backend: backend,
                        kvCodec: turboQuantCompressedKVCodec(backend: backend),
                        optimizationPolicy: turboQuantOptimizationPolicy,
                        fallbackPolicy: turboQuantFallbackPolicy,
                        seed: turboQuantSeed ?? defaultTurboQuantSeed,
                        valueBits: valueBits,
                        precisionPolicy: turboQuantPrecisionPolicy,
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        sparseValueSelection: turboQuantSparseValueSelection,
                        layerIndex: layerIndex,
                        residentBudgetBytes: turboQuantResidentBudgetBytes
                    )
                }
                if let rotatingCache = item as? RotatingKVCache {
                    return rotatingCache.toTurboQuant(
                        preset: preset,
                        groupSize: groupSize,
                        backend: backend,
                        kvCodec: turboQuantCompressedKVCodec(backend: backend),
                        optimizationPolicy: turboQuantOptimizationPolicy,
                        fallbackPolicy: turboQuantFallbackPolicy,
                        seed: turboQuantSeed ?? defaultTurboQuantSeed,
                        valueBits: valueBits,
                        precisionPolicy: turboQuantPrecisionPolicy,
                        sparseValuePolicy: turboQuantSparseValuePolicy,
                        sparseValueSelection: turboQuantSparseValueSelection,
                        layerIndex: layerIndex,
                        residentBudgetBytes: turboQuantResidentBudgetBytes
                    )
                }
            }
            return item
        }
        guard let kvBits = resolvedBits else { return item }
        if let simpleCache = item as? KVCacheSimple {
            if useAffineK8Vx {
                return simpleCache.toAffineK8V4(
                    valueBits: affineK8VxValueBits,
                    valueGroupSize: affineK8VxValueGroupSize,
                    sparseValuePolicy: turboQuantSparseValuePolicy,
                    layerIndex: layerIndex,
                    boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                    boundaryProtectionReason: affineBoundaryMetadata.reason
                )
            }
            if useAffineInt4 {
                return simpleCache.toAffineInt4(
                    groupSize: resolvedAffineInt4GroupSize(
                        kvGroupSize: kvGroupSize,
                        kvBits: kvBits,
                        kvCodec: kvCodec
                    )
                )
            }
            if spillToTurboQuant {
                return simpleCache.toTurboQuant(
                    preset: turboQuantPreset,
                    groupSize: kvGroupSize,
                    backend: turboQuantBackend,
                    kvCodec: kvCodec,
                    optimizationPolicy: turboQuantOptimizationPolicy,
                    fallbackPolicy: turboQuantFallbackPolicy,
                    seed: turboQuantSeed ?? defaultTurboQuantSeed,
                    valueBits: turboQuantValueBits,
                    precisionPolicy: turboQuantPrecisionPolicy,
                    sparseValuePolicy: turboQuantSparseValuePolicy,
                    sparseValueSelection: turboQuantSparseValueSelection,
                    layerIndex: layerIndex,
                    residentBudgetBytes: turboQuantResidentBudgetBytes
                )
            }
            return simpleCache.toQuantized(groupSize: effectiveAffineGroupSize, bits: kvBits)
        }
        if spillToTurboQuant, let rotatingCache = item as? RotatingKVCache {
            return rotatingCache.toTurboQuant(
                preset: turboQuantPreset,
                groupSize: kvGroupSize,
                backend: turboQuantBackend,
                kvCodec: kvCodec,
                optimizationPolicy: turboQuantOptimizationPolicy,
                fallbackPolicy: turboQuantFallbackPolicy,
                seed: turboQuantSeed ?? defaultTurboQuantSeed,
                valueBits: turboQuantValueBits,
                precisionPolicy: turboQuantPrecisionPolicy,
                sparseValuePolicy: turboQuantSparseValuePolicy,
                sparseValueSelection: turboQuantSparseValueSelection,
                layerIndex: layerIndex,
                residentBudgetBytes: turboQuantResidentBudgetBytes
            )
        }
        if let rotatingCache = item as? RotatingKVCache {
            if useAffineK8Vx {
                return rotatingCache.toAffineK8V4(
                    valueBits: affineK8VxValueBits,
                    valueGroupSize: affineK8VxValueGroupSize,
                    sparseValuePolicy: turboQuantSparseValuePolicy,
                    layerIndex: layerIndex,
                    boundaryProtectedLayerCount: affineBoundaryMetadata.protectedLayerCount,
                    boundaryProtectionReason: affineBoundaryMetadata.reason
                )
            }
            if useAffineInt4 {
                return rotatingCache.toAffineInt4(
                    groupSize: resolvedAffineInt4GroupSize(
                        kvGroupSize: kvGroupSize,
                        kvBits: kvBits,
                        kvCodec: kvCodec
                    )
                )
            }
            return rotatingCache.toQuantized(groupSize: effectiveAffineGroupSize, bits: kvBits)
        }
        return item
    }

    guard cache.indices.contains(where: { isReadyForQuantization(cache[$0], layerIndex: $0) }) else {
        return
    }

    let timingStart = TurboQuantTiming.start()
    for i in 0 ..< cache.count {
        cache[i] = convertedCache(cache[i], layerIndex: i)
    }
    let convertedState = cache.flatMap { $0.state }
    if !convertedState.isEmpty {
        eval(convertedState)
        Stream.gpu.synchronize()
        Memory.clearCache()
    }
    TurboQuantTiming.record(.dynamicCacheQuantization, startedAt: timingStart)
}
