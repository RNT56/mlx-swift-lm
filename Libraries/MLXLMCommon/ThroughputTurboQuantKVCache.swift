// Copyright © 2026 RNT56.

import Foundation
import MLX

public final class ThroughputTurboQuantKVCache: BaseKVCache, TurboQuantCompressedKVCacheProtocol,
    CustomDebugStringConvertible
{
    private var activeCache: any KVCache
    private var backingCache: any TurboQuantCompressedKVCacheProtocol
    private var lastAttentionPath: TurboQuantAttentionPath = .baseline
    private var lastFailure: String?

    public let requestedRuntimeMode: TurboQuantRuntimeMode
    public let resolvedRuntimeMode: TurboQuantRuntimeMode
    public let precisionPolicy: TurboQuantKVPrecisionPolicy
    public let sparseValuePolicy: TurboQuantSparseValuePolicy
    public let sparseValueSelection: TurboQuantSparseValueSelection
    public let boundaryProtectedLayerCount: Int
    public let boundaryProtectionReason: String?

    public init(
        maxSize: Int? = nil,
        keep: Int = 4,
        step: Int = 256,
        preset: TurboQuantPreset = .turbo8,
        groupSize: Int = 64,
        mode: QuantizationMode = .affine,
        backend: TurboQuantBackend = .metalPolarQJL,
        kvCodec: TurboQuantKVCodec? = nil,
        optimizationPolicy: TurboQuantOptimizationPolicy = .preferThroughput,
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed,
        seed: UInt64 = 0x9E37_79B9_7F4A_7C15,
        valueBits: Int? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        requestedRuntimeMode: TurboQuantRuntimeMode = .auto,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        sparseValueSelection: TurboQuantSparseValueSelection = .off,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        residentBudgetBytes: Int? = nil
    ) {
        let backingKVCodec = turboQuantCompressedKVCodec(
            requested: kvCodec,
            backend: backend
        )
        let requestedValueBits = turboQuantDefaultValueBits(
            preset: preset,
            kvCodec: backingKVCodec,
            requestedValueBits: valueBits
        )
        let policy =
            precisionPolicy
            ?? TurboQuantKVPrecisionPolicy.legacy(preset: preset, valueBits: requestedValueBits)
        self.requestedRuntimeMode = requestedRuntimeMode
        self.resolvedRuntimeMode = .throughputTurboQuant
        self.precisionPolicy = policy
        self.sparseValuePolicy = sparseValuePolicy
        self.sparseValueSelection = sparseValueSelection
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        let backingPreset = policy.compressedKeyPreset
        let backingValueBits = policy.resolvedValueBits ?? requestedValueBits

        if let maxSize {
            self.activeCache = RotatingKVCache(maxSize: maxSize, keep: keep, step: step)
            self.backingCache = RotatingTurboQuantKVCache(
                maxSize: maxSize,
                keep: keep,
                step: step,
                preset: backingPreset,
                groupSize: groupSize,
                mode: mode,
                backend: backend,
                kvCodec: backingKVCodec,
                optimizationPolicy: optimizationPolicy,
                fallbackPolicy: fallbackPolicy,
                seed: seed,
                valueBits: backingValueBits,
                sparseValuePolicy: sparseValuePolicy,
                sparseValueSelection: sparseValueSelection,
                residentBudgetBytes: residentBudgetBytes
            )
        } else {
            let active = KVCacheSimple()
            active.step = step
            self.activeCache = active
            self.backingCache = TurboQuantKVCache(
                preset: backingPreset,
                groupSize: groupSize,
                mode: mode,
                backend: backend,
                kvCodec: backingKVCodec,
                optimizationPolicy: optimizationPolicy,
                fallbackPolicy: fallbackPolicy,
                seed: seed,
                valueBits: backingValueBits,
                sparseValuePolicy: sparseValuePolicy,
                sparseValueSelection: sparseValueSelection,
                residentBudgetBytes: residentBudgetBytes
            )
        }
        super.init()
    }

    public override var maxSize: Int? { activeCache.maxSize }
    public var ropeOffset: RoPEOffset { activeCache.ropeOffset }
    public override var isTrimmable: Bool { activeCache.isTrimmable }

    public var preset: TurboQuantPreset { backingCache.preset }
    public var kvCodec: TurboQuantKVCodec { backingCache.kvCodec }
    public var requestedBackend: TurboQuantBackend { backingCache.requestedBackend }
    public var activeBackend: TurboQuantBackend { backingCache.activeBackend }
    public var optimizationPolicy: TurboQuantOptimizationPolicy { backingCache.optimizationPolicy }
    public var fallbackPolicy: TurboQuantFallbackPolicy { backingCache.fallbackPolicy }
    public var compressedState: (TurboQuantAttentionCode, TurboQuantAttentionCode)? {
        backingCache.compressedState
    }
    public var hybridAffineKeyState: QuantizedKVStorage? {
        backingCache.hybridAffineKeyState
    }
    public var hybridAffineKeyStateForAttention: QuantizedKVStorage? {
        backingCache.hybridAffineKeyStateForAttention
    }
    public var polarWHTKeyState: TurboQuantPolarWHTAttentionValueCode? {
        backingCache.polarWHTKeyState
    }
    public var polarWHTValueState: TurboQuantPolarWHTAttentionValueCode? {
        backingCache.polarWHTValueState
    }
    public var polarWHTKeyStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        backingCache.polarWHTKeyStateForAttention
    }
    public var polarWHTValueStateForAttention: TurboQuantPolarWHTAttentionValueCode? {
        backingCache.polarWHTValueStateForAttention
    }
    public var keyPageSummary: MLXArray? { backingCache.keyPageSummary }
    public func ensureKeyPageSummary() { backingCache.ensureKeyPageSummary() }
    public var keyCandidateSketch: MLXArray? { backingCache.keyCandidateSketch }
    public func ensureKeyCandidateSketch() { backingCache.ensureKeyCandidateSketch() }
    public var cacheLifecycle: TurboQuantCacheLifecycle { backingCache.cacheLifecycle }
    public var fallbackResults: [TurboQuantFallbackResult] { backingCache.fallbackResults }

    public var cacheFootprint: TurboQuantRuntimeCacheFootprint {
        let backing = backingCache.cacheFootprint
        return TurboQuantRuntimeCacheFootprint(
            logicalLength: backing.logicalLength,
            capacity: backing.capacity,
            compressedBytes: backing.compressedBytes,
            packedFallbackBytes: backing.packedFallbackBytes,
            rawShadowBytes: activeKVBytes + backing.rawShadowBytes,
            decodedTransientBytes: backing.decodedTransientBytes,
            lifecycle: backing.lifecycle
        )
    }

    public var attentionDiagnostics: TurboQuantAttentionDiagnostics {
        let diagnostics = backingCache.attentionDiagnostics
        return TurboQuantAttentionDiagnostics(
            metalAttentionAvailable: diagnostics.metalAttentionAvailable,
            activeAttentionPath: lastAttentionPath,
            nativeBackend: diagnostics.nativeBackend,
            nativeBackendVersion: diagnostics.nativeBackendVersion,
            nativeFallbackReason: diagnostics.nativeFallbackReason,
            nativeSparseVSkipRatio: diagnostics.nativeSparseVSkipRatio,
            selectedKernelProfile: diagnostics.selectedKernelProfile,
            selfTestStatus: diagnostics.selfTestStatus,
            selfTestFailureReason: diagnostics.selfTestFailureReason,
            optimizationPolicy: diagnostics.optimizationPolicy,
            fallbackReason: diagnostics.fallbackReason ?? lastFailure,
            lastUnsupportedShape: diagnostics.lastUnsupportedShape ?? lastFailure,
            rawFallbackAllocated: true,
            cacheLifecycle: diagnostics.cacheLifecycle,
            lastFallback: diagnostics.lastFallback,
            sparseVEnabled: diagnostics.sparseVEnabled,
            sparseVThreshold: diagnostics.sparseVThreshold,
            sparseVSelectionMode: diagnostics.sparseVSelectionMode,
            sparseVTopK: diagnostics.sparseVTopK,
            sparseVCumulativeMass: diagnostics.sparseVCumulativeMass,
            sparseVMaxTopK: diagnostics.sparseVMaxTopK,
            sparseVRecentTokenCount: diagnostics.sparseVRecentTokenCount,
            sparseVOlderTokenCount: diagnostics.sparseVOlderTokenCount,
            sparseVPageCandidateCount: diagnostics.sparseVPageCandidateCount,
            sparseVSkippedTokens: diagnostics.sparseVSkippedTokens,
            sparseVTotalTokens: diagnostics.sparseVTotalTokens,
            sparseVActive: diagnostics.sparseVActive,
            sparseVSkipRatio: diagnostics.sparseVSkipRatio,
            sparseVRetainedMass: diagnostics.sparseVRetainedMass,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            keyBits: diagnostics.keyBits,
            valueBits: diagnostics.valueBits,
            keyGroupSize: diagnostics.keyGroupSize,
            valueGroupSize: diagnostics.valueGroupSize,
            keyPageSummaryAvailable: diagnostics.keyPageSummaryAvailable,
            keyPageSummaryShape: diagnostics.keyPageSummaryShape,
            keyPageSummaryUnavailableReason: diagnostics.keyPageSummaryUnavailableReason,
            keyCandidateSketchAvailable: diagnostics.keyCandidateSketchAvailable,
            keyCandidateSketchShape: diagnostics.keyCandidateSketchShape,
            keyCandidateSketchUnavailableReason: diagnostics.keyCandidateSketchUnavailableReason,
            polarWHTValueBytes: diagnostics.polarWHTValueBytes,
            polarWHTValuePayloadAllocated: diagnostics.polarWHTValuePayloadAllocated
        )
    }

    public func updateThroughput(keys: MLXArray, values: MLXArray) throws -> (MLXArray, MLXArray) {
        _ = try backingCache.updateCompressed(keys: keys, values: values)
        let (activeKeys, activeValues) = activeCache.update(keys: keys, values: values)
        offset = activeCache.offset
        lastAttentionPath = .baseline
        lastFailure = nil
        return (activeKeys, activeValues)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        do {
            return try updateThroughput(keys: keys, values: values)
        } catch {
            lastFailure = String(describing: error)
            backingCache.recordCompressedAttentionFailure(lastFailure ?? "throughput update failed")
            let result = activeCache.update(keys: keys, values: values)
            offset = activeCache.offset
            return result
        }
    }

    public override var state: [MLXArray] {
        get { backingCache.state }
        set {
            backingCache.state = newValue
            offset = backingCache.offset
            rehydrateActiveCacheFromBacking()
        }
    }

    public override var metaState: [String] {
        get {
            [
                "throughput-turboquant-v1",
                requestedRuntimeMode.rawValue,
                resolvedRuntimeMode.rawValue,
            ] + backingCache.metaState
        }
        set {
            let backingMeta =
                newValue.first == "throughput-turboquant-v1"
                ? Array(newValue.dropFirst(3))
                : newValue
            backingCache.metaState = backingMeta
            offset = backingCache.offset
            rehydrateActiveCacheFromBacking()
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
        activeCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let activeTrimmed = activeCache.trim(n)
        let backingTrimmed = backingCache.trim(n)
        offset = activeCache.offset
        return max(activeTrimmed, backingTrimmed)
    }

    public override func copy() -> any KVCache {
        let copy = ThroughputTurboQuantKVCache(
            maxSize: maxSize,
            preset: preset,
            groupSize: (backingCache as? any QuantizedKVCacheProtocol)?.groupSize ?? 64,
            backend: requestedBackend,
            kvCodec: kvCodec,
            optimizationPolicy: optimizationPolicy,
            fallbackPolicy: fallbackPolicy,
            seed: (backingCache as? TurboQuantKVCache)?.seed
                ?? (backingCache as? RotatingTurboQuantKVCache)?.seed
                ?? defaultTurboQuantSeed,
            valueBits: precisionPolicy.resolvedValueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: requestedRuntimeMode,
            sparseValuePolicy: sparseValuePolicy,
            sparseValueSelection: sparseValueSelection,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason
        )
        copy.state = state.map { $0[.ellipsis] }
        copy.metaState = metaState
        return copy
    }

    public func runtimeSnapshot() -> TurboQuantCacheRuntimeSnapshot {
        var snapshot = backingCache.runtimeSnapshot()
        snapshot.requestedRuntimeMode = requestedRuntimeMode
        snapshot.resolvedRuntimeMode = resolvedRuntimeMode
        snapshot.precisionPolicy = precisionPolicy
        snapshot.sparseValuePolicy = sparseValuePolicy
        snapshot.boundaryPolicy = precisionPolicy.boundary
        snapshot.boundaryProtectedLayerCount = boundaryProtectedLayerCount
        snapshot.boundaryProtectionReason = boundaryProtectionReason
        snapshot.runtimeFallbackReason = lastFailure
        snapshot.decodedActiveKeyBytes = activeKeyBytes
        snapshot.decodedActiveValueBytes = activeValueBytes
        snapshot.activeCacheAllocated = activeKVBytes > 0
        return snapshot
    }

    public func exportSnapshot(
        identity: TurboQuantKVSnapshotIdentity,
        conversationID: UUID,
        snapshotID: UUID,
        encryptionKeyID: String,
        createdAt: Date
    ) throws -> TurboQuantKVSnapshotPayload {
        var payload = try backingCache.exportSnapshot(
            identity: identity,
            conversationID: conversationID,
            snapshotID: snapshotID,
            encryptionKeyID: encryptionKeyID,
            createdAt: createdAt
        )
        payload.manifest.requestedRuntimeMode = requestedRuntimeMode
        payload.manifest.resolvedRuntimeMode = resolvedRuntimeMode
        payload.manifest.precisionPolicy = precisionPolicy
        payload.manifest.sparseValuePolicy = sparseValuePolicy
        payload.manifest.boundaryPolicy = precisionPolicy.boundary
        payload.manifest.boundaryProtectedLayerCount = boundaryProtectedLayerCount
        payload.manifest.boundaryProtectionReason = boundaryProtectionReason
        payload.manifest.runtimeFallbackReason = lastFailure
        return payload
    }

    public func importSnapshot(
        _ payload: TurboQuantKVSnapshotPayload,
        expectedIdentity: TurboQuantKVSnapshotIdentity
    ) throws {
        try backingCache.importSnapshot(payload, expectedIdentity: expectedIdentity)
        offset = backingCache.offset
        rehydrateActiveCacheFromBacking()
    }

    public func supportsCompressedAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> Bool {
        false
    }

    public func updateCompressed(keys: MLXArray, values: MLXArray) throws -> (
        TurboQuantAttentionCode,
        TurboQuantAttentionCode
    ) {
        try backingCache.updateCompressed(keys: keys, values: values)
    }

    public func makeCompressedUpdateCheckpoint(appendingTokenCount tokenCount: Int)
        -> TurboQuantCompressedUpdateCheckpoint
    {
        backingCache.makeCompressedUpdateCheckpoint(appendingTokenCount: tokenCount)
    }

    public func restoreCompressedUpdateCheckpoint(_ checkpoint: TurboQuantCompressedUpdateCheckpoint) {
        backingCache.restoreCompressedUpdateCheckpoint(checkpoint)
        offset = backingCache.offset
        rehydrateActiveCacheFromBacking()
    }

    public func recordCompressedAttentionFailure(_ message: String) {
        lastFailure = message
        backingCache.recordCompressedAttentionFailure(message)
    }

    public func recordFallback(_ result: TurboQuantFallbackResult) {
        lastFailure = result.reason
        backingCache.recordFallback(result)
    }

    public func validateCompressedState(context: String) throws {
        try backingCache.validateCompressedState(context: context)
    }

    public func decodedCompressedState(outputDType: DType) throws -> (MLXArray, MLXArray) {
        try backingCache.decodedCompressedState(outputDType: outputDType)
    }

    public func releaseRawShadow() {
        backingCache.releaseRawShadow()
    }

    public func exactRawStateIfComplete() -> (keys: MLXArray, values: MLXArray)? {
        let state = activeCache.state
        guard state.count == 2, activeCache.offset == offset else { return nil }
        return (state[0], state[1])
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), mode: \(resolvedRuntimeMode.rawValue), backing: \(String(describing: type(of: backingCache)))"
    }

    private var activeKeyBytes: Int {
        let state = activeCache.state
        return state.count > 0 ? state[0].nbytes : 0
    }

    private var activeValueBytes: Int {
        let state = activeCache.state
        return state.count > 1 ? state[1].nbytes : 0
    }

    private var activeKVBytes: Int {
        activeKeyBytes + activeValueBytes
    }

    private func rehydrateActiveCacheFromBacking() {
        guard backingCache.offset > 0 else {
            _ = activeCache.trim(activeCache.offset)
            offset = 0
            return
        }
        do {
            let decoded = try backingCache.decodedCompressedState(outputDType: .float16)
            activeCache.state = [decoded.0, decoded.1]
            offset = activeCache.offset
            lastFailure = nil
        } catch {
            offset = backingCache.offset
            lastFailure = "failed to rehydrate throughput active cache: \(error)"
        }
    }
}
