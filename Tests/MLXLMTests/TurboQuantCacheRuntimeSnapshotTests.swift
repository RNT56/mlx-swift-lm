import Foundation
import MLX
import MLXLMCommon
import Testing

@Suite("TurboQuant cache runtime snapshot")
struct TurboQuantCacheRuntimeSnapshotTests {
    @Test func emptyCacheSnapshotIsStableAndCodable() throws {
        let cache = TurboQuantKVCache()

        let snapshot = cache.runtimeSnapshot()
        let data = try JSONEncoder().encode(snapshot)
        let decoded = try JSONDecoder().decode(TurboQuantCacheRuntimeSnapshot.self, from: data)

        #expect(decoded == snapshot)
        #expect(snapshot.schemaVersion == TurboQuantCacheRuntimeSnapshot.currentSchemaVersion)
        #expect(snapshot.lifecycleDescription == "empty")
        #expect(snapshot.logicalLength == 0)
        #expect(snapshot.capacity == 0)
        #expect(snapshot.keyBytes == 0)
        #expect(snapshot.valueBytes == 0)
        #expect(!snapshot.rawShadowAllocated)
        #expect(!snapshot.packedFallbackAllocated)
        #expect(snapshot.polarWHTValueBytes == 0)
        #expect(!snapshot.polarWHTValuePayloadAllocated)
    }

    @Test func legacyV1SnapshotDecodesWithWave1Defaults() throws {
        let json = """
            {
              "schemaVersion": 1,
              "lifecycleDescription": "empty",
              "logicalLength": 0,
              "capacity": 0,
              "pinnedPrefixLength": 0,
              "ringOffset": 0,
              "keyBytes": 0,
              "valueBytes": 0,
              "rawShadowAllocated": false,
              "packedFallbackAllocated": false
            }
            """

        let snapshot = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: Data(json.utf8)
        )

        #expect(snapshot.schemaVersion == 1)
        #expect(snapshot.kvCodec == .polarQJL)
        #expect(snapshot.quantizationMode == nil)
        #expect(snapshot.keyBits == nil)
        #expect(snapshot.groupSize == nil)
        #expect(snapshot.requestedRuntimeMode == nil)
        #expect(snapshot.resolvedRuntimeMode == nil)
        #expect(snapshot.precisionPolicy == nil)
        #expect(snapshot.sparseValuePolicy == nil)
        #expect(snapshot.boundaryPolicy == nil)
        #expect(snapshot.boundaryProtectedLayerCount == 0)
        #expect(snapshot.decodedActiveKeyBytes == 0)
        #expect(snapshot.decodedActiveValueBytes == 0)
        #expect(!snapshot.activeCacheAllocated)
        #expect(snapshot.polarWHTValueBytes == 0)
        #expect(!snapshot.polarWHTValuePayloadAllocated)
    }

    @Test func runtimeSnapshotRoundTripsWave2Metadata() throws {
        let policy = TurboQuantKVPrecisionPolicy(
            key: .turbo4v2,
            value: .turbo4v2,
            boundary: .protectedEdges(first: 2, last: 2)
        )
        let cache = TurboQuantKVCache(
            preset: .turbo4v2,
            precisionPolicy: policy,
            resolvedRuntimeMode: .capacityTurboQuant,
            sparseValuePolicy: .force(threshold: 1e-5),
            boundaryProtectedLayerCount: 4,
            boundaryProtectionReason: "test"
        )

        let snapshot = cache.runtimeSnapshot()
        let decoded = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: try JSONEncoder().encode(snapshot)
        )

        #expect(decoded.sparseValuePolicy == .force(threshold: 1e-5))
        #expect(decoded.boundaryPolicy == .protectedEdges(first: 2, last: 2))
        #expect(decoded.boundaryProtectedLayerCount == 4)
        #expect(decoded.boundaryProtectionReason == "test")
    }

    @Test func runtimeSnapshotRoundTripsPolarWHTMetadata() throws {
        let snapshot = TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription: "polarWHT(logicalLength:8,capacity:16)",
            logicalLength: 8,
            capacity: 16,
            pinnedPrefixLength: 0,
            ringOffset: 0,
            keyBytes: 1024,
            valueBytes: 384,
            rawShadowAllocated: false,
            packedFallbackAllocated: true,
            lastAttentionPath: TurboQuantAttentionPath.mlxPackedFallback.rawValue,
            lastFailure: nil,
            kvCodec: .polarWHT,
            valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
            selectedPath: TurboQuantAttentionPath.mlxPackedFallback.rawValue,
            fallbackReason: "PolarWHT Metal kernels unavailable; using MLX packed TurboQuant lanes.",
            requestedRuntimeMode: .auto,
            resolvedRuntimeMode: .capacityTurboQuant,
            runtimeFallbackReason: "PolarWHT Metal kernels unavailable; using MLX packed TurboQuant lanes.",
            polarWHTValueBytes: 0,
            polarWHTValuePayloadAllocated: false
        )

        let decoded = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: try JSONEncoder().encode(snapshot)
        )

        #expect(decoded.kvCodec == .polarWHT)
        #expect(decoded.valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)
        #expect(decoded.selectedPath == TurboQuantAttentionPath.mlxPackedFallback.rawValue)
        #expect(decoded.fallbackReason?.contains("PolarWHT Metal kernels unavailable") == true)
        #expect(decoded.runtimeFallbackReason?.contains("PolarWHT Metal kernels unavailable") == true)
        #expect(decoded.polarWHTValueBytes == 0)
        #expect(!decoded.polarWHTValuePayloadAllocated)
    }

    @Test func runtimeSnapshotRoundTripsAffineInt4Metadata() throws {
        let snapshot = TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription: "affineInt4Native(logicalLength:8)",
            logicalLength: 8,
            capacity: 8,
            pinnedPrefixLength: 0,
            ringOffset: 0,
            keyBytes: 1024,
            valueBytes: 1024,
            rawShadowAllocated: false,
            packedFallbackAllocated: false,
            lastAttentionPath: TurboQuantAttentionPath.affineInt4Native.rawValue,
            lastFailure: nil,
            kvCodec: .affineInt4,
            quantizationMode: QuantizationMode.affine.rawValue,
            keyBits: TurboQuantKVCodec.affineInt4Bits,
            groupSize: TurboQuantKVCodec.affineInt4DefaultGroupSize,
            selectedPath: TurboQuantAttentionPath.affineInt4Native.rawValue,
            fallbackReason: nil
        )

        let decoded = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: try JSONEncoder().encode(snapshot)
        )

        #expect(decoded.kvCodec == .affineInt4)
        #expect(decoded.quantizationMode == QuantizationMode.affine.rawValue)
        #expect(decoded.keyBits == TurboQuantKVCodec.affineInt4Bits)
        #expect(decoded.groupSize == TurboQuantKVCodec.affineInt4DefaultGroupSize)
        #expect(decoded.selectedPath == TurboQuantAttentionPath.affineInt4Native.rawValue)
    }

    @Test func runtimeSnapshotRoundTripsAffineK8V4Metadata() throws {
        let snapshot = TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription: "affineK8V4Native(logicalLength:8,capacity:16)",
            logicalLength: 8,
            capacity: 16,
            pinnedPrefixLength: 0,
            ringOffset: 0,
            keyBytes: 1024,
            valueBytes: 512,
            rawShadowAllocated: false,
            packedFallbackAllocated: false,
            lastAttentionPath: TurboQuantAttentionPath.affineK8V4Native.rawValue,
            lastFailure: nil,
            kvCodec: .affineK8V4,
            quantizationMode: QuantizationMode.affine.rawValue,
            keyBits: TurboQuantKVCodec.affineK8V4KeyBits,
            groupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            valueBits: TurboQuantKVCodec.affineK8V4ValueBits,
            valueGroupSize: TurboQuantKVCodec.affineK8V4ValueGroupSize,
            selectedPath: TurboQuantAttentionPath.affineK8V4Native.rawValue,
            fallbackReason: nil
        )

        let decoded = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: try JSONEncoder().encode(snapshot)
        )

        #expect(decoded.kvCodec == .affineK8V4)
        #expect(decoded.quantizationMode == QuantizationMode.affine.rawValue)
        #expect(decoded.keyBits == TurboQuantKVCodec.affineK8V4KeyBits)
        #expect(decoded.groupSize == TurboQuantKVCodec.affineK8V4KeyGroupSize)
        #expect(decoded.valueBits == TurboQuantKVCodec.affineK8V4ValueBits)
        #expect(decoded.valueGroupSize == TurboQuantKVCodec.affineK8V4ValueGroupSize)
        #expect(decoded.selectedPath == TurboQuantAttentionPath.affineK8V4Native.rawValue)
    }

    @Test func runtimeSnapshotRoundTripsAffineK8VxMetadata() throws {
        let snapshot = TurboQuantCacheRuntimeSnapshot(
            lifecycleDescription: "affineK8V3Native(logicalLength:8,capacity:16)",
            logicalLength: 8,
            capacity: 16,
            pinnedPrefixLength: 0,
            ringOffset: 0,
            keyBytes: 1024,
            valueBytes: 384,
            rawShadowAllocated: false,
            packedFallbackAllocated: false,
            lastAttentionPath: TurboQuantAttentionPath.affineK8VxNative.rawValue,
            lastFailure: nil,
            kvCodec: .affineK8Vx,
            quantizationMode: QuantizationMode.affine.rawValue,
            keyBits: TurboQuantKVCodec.affineK8V4KeyBits,
            groupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            valueBits: 3,
            valueGroupSize: TurboQuantKVCodec.affineK8V4ValueGroupSize,
            selectedPath: TurboQuantAttentionPath.affineK8VxNative.rawValue,
            fallbackReason: nil
        )

        let decoded = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: try JSONEncoder().encode(snapshot)
        )

        #expect(decoded.kvCodec == .affineK8Vx)
        #expect(decoded.keyBits == TurboQuantKVCodec.affineK8V4KeyBits)
        #expect(decoded.groupSize == TurboQuantKVCodec.affineK8V4KeyGroupSize)
        #expect(decoded.valueBits == 3)
        #expect(decoded.valueGroupSize == TurboQuantKVCodec.affineK8V4ValueGroupSize)
        #expect(decoded.selectedPath == TurboQuantAttentionPath.affineK8VxNative.rawValue)
    }

    @Test func throughputSnapshotRecordsActiveDecodedResidency() {
        let cache = ThroughputTurboQuantKVCache(
            maxSize: 16,
            preset: .turbo8,
            valueBits: 4,
            precisionPolicy: .qwenQ4Default,
            requestedRuntimeMode: .auto
        )
        let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
        let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)

        _ = cache.update(keys: keys, values: values)
        let snapshot = cache.runtimeSnapshot()

        #expect(snapshot.requestedRuntimeMode == .auto)
        #expect(snapshot.resolvedRuntimeMode == .throughputTurboQuant)
        #expect(snapshot.precisionPolicy == .qwenQ4Default)
        #expect(snapshot.activeCacheAllocated)
        #expect(snapshot.decodedActiveKeyBytes == keys.nbytes)
        #expect(snapshot.decodedActiveValueBytes == values.nbytes)
    }

    @Test func failureSnapshotRecordsReasonAndPath() {
        let cache = TurboQuantKVCache()

        cache.recordCompressedAttentionFailure("forced failure")

        let snapshot = cache.runtimeSnapshot()
        #expect(snapshot.lifecycleDescription == "failed(reason:forced failure)")
        #expect(snapshot.lastAttentionPath == TurboQuantAttentionPath.mlxPackedFallback.rawValue)
        #expect(snapshot.lastFailure == "forced failure")
    }

    @Test func fallbackSnapshotRecordsLifecycleReason() {
        let cache = TurboQuantKVCache()

        cache.recordFallback(
            TurboQuantFallbackResult(
                fromPath: .twoStageCompressed,
                toPath: .mlxPackedFallback,
                policy: .packedAllowed,
                reason: "qk unavailable",
                isSemanticallyExact: true
            )
        )

        let snapshot = cache.runtimeSnapshot()
        #expect(snapshot.lifecycleDescription == "degradedPackedFallback(reason:qk unavailable)")
        #expect(snapshot.lastAttentionPath == TurboQuantAttentionPath.mlxPackedFallback.rawValue)
        #expect(snapshot.lastFailure == "qk unavailable")
    }

    @Test func hybridSnapshotCarriesSelectorDiagnostics() throws {
        let cache = HybridTurboQuantKVCache(
            maxSize: 32,
            hotWindowTokens: 8,
            coldBlockTokens: 4,
            coldBudgetTokens: 4,
            maxColdBudgetTokens: 8
        )
        let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
        let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
        _ = cache.update(keys: keys, values: values)

        let snapshot = cache.runtimeSnapshot()

        #expect(snapshot.lifecycleDescription == "hybridRawHot(logicalLength:3,hotTokens:3)")
        #expect(snapshot.logicalLength == 3)
        #expect(snapshot.rawShadowAllocated)
        #expect(snapshot.lastAttentionPath == "hybrid_raw_hot")
        #expect(snapshot.hybridDiagnostics?.hotTokens == 3)
        #expect(snapshot.hybridDiagnostics?.coldBudgetTokens == 0)
        #expect(snapshot.hybridDiagnostics?.maxColdBudgetTokens == 8)

        let decoded = try JSONDecoder().decode(
            TurboQuantCacheRuntimeSnapshot.self,
            from: try JSONEncoder().encode(snapshot)
        )
        #expect(decoded.hybridDiagnostics?.selectedBudgetedColdTokens == 0)
        #expect(decoded.hybridDiagnostics?.anchorColdTokens == 0)
        #expect(decoded.hybridDiagnostics?.anchorOverflowTokens == 0)
        #expect(decoded.hybridDiagnostics?.selectorInitialConfidence == 1)
        #expect(decoded.hybridDiagnostics?.selectorFinalConfidence == 1)
        #expect(
            decoded.hybridDiagnostics?.selectorEscalation
                == TurboQuantColdSelectorEscalation.none
        )
        #expect(decoded.hybridDiagnostics?.selectorReasonFlags == ["empty"])
    }

    @Test func rotatingSnapshotReportsCapacityAndPinnedPrefix() {
        let cache = RotatingTurboQuantKVCache(maxSize: 32, keep: 4)

        let snapshot = cache.runtimeSnapshot()

        #expect(snapshot.capacity == 32)
        #expect(snapshot.logicalLength == 0)
        #expect(snapshot.pinnedPrefixLength == 0)
        #expect(snapshot.ringOffset == 0)
        #expect(!snapshot.rawShadowAllocated)
    }
}
