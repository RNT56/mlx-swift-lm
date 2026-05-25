import Foundation
import MLX
import MLXLMCommon
import Testing

extension MLXRuntimeSwiftTests {
    @Suite("TurboQuant KV snapshot export/import")
    struct TurboQuantKVSnapshotTests {
    @Test func manifestIsCodableAndCarriesIdentity() throws {
        let manifest = TurboQuantKVSnapshotManifest(
            snapshotID: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000002")!,
            identity: Self.identity(),
            turboQuantLayoutVersion: 4,
            logicalLength: 8,
            pinnedPrefixLength: 4,
            compressedKeyBytes: 10,
            compressedValueBytes: 11,
            blobByteCount: 21,
            encryptionKeyID: "key-1",
            createdAt: Date(timeIntervalSinceReferenceDate: 1),
            cacheKind: "TurboQuantKVCache",
            preset: TurboQuantPreset.turbo3_5.rawValue,
            requestedBackend: TurboQuantBackend.metalPolarQJL.rawValue,
            activeBackend: TurboQuantBackend.metalPolarQJL.rawValue,
            groupSize: 64,
            valueBits: 4,
            capacity: 8,
            ringOffset: 0,
            batchSize: 1,
            kvHeadCount: 2,
            keyHeadDimension: 64,
            valueHeadDimension: 64,
            arrays: []
        )

        let data = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(TurboQuantKVSnapshotManifest.self, from: data)

        #expect(decoded == manifest)
        #expect(decoded.schemaVersion == 1)
        #expect(decoded.modelID == "model-a")
        #expect(decoded.tokenPrefixHash == "prefix-a")
    }

    @Test func nonRotatingSnapshotRoundTripsCompressedStateWithoutRawKV() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            return
        }
        let cache = TurboQuantKVCache(backend: .metalPolarQJL)
        let keys = MLXArray.ones([1, 2, 4, 64], dtype: .float32)
        let values = keys + 0.25
        _ = try cache.updateCompressed(keys: keys, values: values)

        let payload = try cache.exportSnapshot(
            identity: Self.identity(),
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000003")!,
            encryptionKeyID: "local-key"
        )
        let restored = TurboQuantKVCache(backend: .metalPolarQJL)
        try restored.importSnapshot(payload, expectedIdentity: Self.identity())

        let originalState = try #require(cache.compressedState)
        let restoredState = try #require(restored.compressedState)
        let originalDecoded = try turboQuantMetalDecodeAttention(originalState.0, outputDType: .float32)
        let restoredDecoded = try turboQuantMetalDecodeAttention(restoredState.0, outputDType: .float32)

        #expect(payload.manifest.cacheKind == "TurboQuantKVCache")
        #expect(payload.manifest.encryptionKeyID == "local-key")
        #expect(payload.manifest.logicalLength == 4)
        #expect(restored.runtimeSnapshot().logicalLength == cache.runtimeSnapshot().logicalLength)
        #expect(restored.runtimeSnapshot().keyBytes == cache.runtimeSnapshot().keyBytes)
        #expect(restored.runtimeSnapshot().rawShadowAllocated == false)
        #expect(allClose(originalDecoded, restoredDecoded, rtol: 1e-5, atol: 1e-5).item(Bool.self))
    }

    @Test func rotatingSnapshotPreservesRingOffsetAndPinnedPrefix() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            return
        }
        let cache = RotatingTurboQuantKVCache(maxSize: 8, keep: 2, backend: .metalPolarQJL)
        let keys = MLXArray.ones([1, 2, 12, 64], dtype: .float32)
        let values = keys + 0.5
        _ = try cache.updateCompressed(keys: keys, values: values)

        let payload = try cache.exportSnapshot(
            identity: Self.identity(),
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000004")!
        )
        let restored = RotatingTurboQuantKVCache(maxSize: 8, keep: 2, backend: .metalPolarQJL)
        try restored.importSnapshot(payload, expectedIdentity: Self.identity())

        let original = cache.runtimeSnapshot()
        let restoredSnapshot = restored.runtimeSnapshot()
        #expect(payload.manifest.cacheKind == "RotatingTurboQuantKVCache")
        #expect(original.logicalLength == 8)
        #expect(restoredSnapshot.logicalLength == original.logicalLength)
        #expect(restoredSnapshot.capacity == original.capacity)
        #expect(restoredSnapshot.ringOffset == original.ringOffset)
        #expect(restoredSnapshot.pinnedPrefixLength == original.pinnedPrefixLength)

        _ = try restored.updateCompressed(
            keys: MLXArray.ones([1, 2, 1, 64], dtype: .float32) * 2,
            values: MLXArray.ones([1, 2, 1, 64], dtype: .float32) * 3
        )
        let advanced = restored.runtimeSnapshot()
        let ringCapacity = original.capacity - original.pinnedPrefixLength
        #expect(advanced.ringOffset == (original.ringOffset + 1) % ringCapacity)
    }

    @Test func importRejectsIdentityAndSchemaMismatchBeforeUse() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            return
        }
        let cache = TurboQuantKVCache(backend: .metalPolarQJL)
        _ = try cache.updateCompressed(
            keys: MLXArray.ones([1, 2, 2, 64], dtype: .float32),
            values: MLXArray.ones([1, 2, 2, 64], dtype: .float32)
        )
        var payload = try cache.exportSnapshot(
            identity: Self.identity(),
            conversationID: UUID()
        )

        let wrongIdentity = TurboQuantKVSnapshotIdentity(
            modelID: "model-a",
            tokenizerHash: "tokenizer-a",
            profileHash: "profile-a",
            ropeConfigHash: "rope-a",
            tokenPrefixHash: "different-prefix"
        )
        let target = TurboQuantKVCache(backend: .metalPolarQJL)
        #expect(throws: TurboQuantRuntimeFailure.self) {
            try target.importSnapshot(payload, expectedIdentity: wrongIdentity)
        }

        payload.manifest.schemaVersion = 999
        #expect(throws: TurboQuantRuntimeFailure.self) {
            try target.importSnapshot(payload, expectedIdentity: Self.identity())
        }
        #expect(target.runtimeSnapshot().logicalLength == 0)
    }

    @Test func importRejectsInvalidLayoutShapeAndDTypeBeforeUse() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            return
        }
        let cache = TurboQuantKVCache(backend: .metalPolarQJL)
        _ = try cache.updateCompressed(
            keys: MLXArray.ones([1, 2, 2, 64], dtype: .float32),
            values: MLXArray.ones([1, 2, 2, 64], dtype: .float32)
        )
        let payload = try cache.exportSnapshot(
            identity: Self.identity(),
            conversationID: UUID()
        )

        var invalidLayout = payload
        invalidLayout.manifest.turboQuantLayoutVersion = 999
        let layoutTarget = TurboQuantKVCache(backend: .metalPolarQJL)
        #expect(throws: TurboQuantRuntimeFailure.self) {
            try layoutTarget.importSnapshot(invalidLayout, expectedIdentity: Self.identity())
        }
        #expect(layoutTarget.runtimeSnapshot().logicalLength == 0)

        var invalidShape = payload
        Self.replaceArray(
            in: &invalidShape,
            name: "key.packedMagnitudes",
            with: MLXArray.zeros([1], dtype: .uint32)
        )
        let shapeTarget = TurboQuantKVCache(backend: .metalPolarQJL)
        #expect(throws: TurboQuantRuntimeFailure.self) {
            try shapeTarget.importSnapshot(invalidShape, expectedIdentity: Self.identity())
        }
        #expect(shapeTarget.runtimeSnapshot().logicalLength == 0)

        var invalidDType = payload
        let scales = try #require(payload.compressedArrays["value.scales"])
        Self.replaceArray(
            in: &invalidDType,
            name: "value.scales",
            with: MLXArray.zeros(scales.shape, dtype: .uint32)
        )
        let dtypeTarget = TurboQuantKVCache(backend: .metalPolarQJL)
        #expect(throws: TurboQuantRuntimeFailure.self) {
            try dtypeTarget.importSnapshot(invalidDType, expectedIdentity: Self.identity())
        }
        #expect(dtypeTarget.runtimeSnapshot().logicalLength == 0)
    }

    @Test func exportRejectsUncommittedOrEmptyState() {
        let cache = TurboQuantKVCache(backend: .metalPolarQJL)

        #expect(throws: TurboQuantRuntimeFailure.self) {
            _ = try cache.exportSnapshot(
                identity: Self.identity(),
                conversationID: UUID()
            )
        }
    }

    private static func identity() -> TurboQuantKVSnapshotIdentity {
        TurboQuantKVSnapshotIdentity(
            modelID: "model-a",
            modelRevision: "rev-a",
            tokenizerHash: "tokenizer-a",
            profileHash: "profile-a",
            ropeConfigHash: "rope-a",
            tokenPrefixHash: "prefix-a",
            fallbackContractHash: "fallback-a"
        )
    }

    private static func replaceArray(
        in payload: inout TurboQuantKVSnapshotPayload,
        name: String,
        with array: MLXArray
    ) {
        payload.compressedArrays[name] = array
        payload.manifest.arrays = payload.manifest.arrays.map { descriptor in
            descriptor.name == name
                ? TurboQuantKVSnapshotArrayDescriptor(name: name, array: array)
                : descriptor
        }
        let keyBytes = TurboQuantKVSnapshotArrayName.ordered.prefix(5).reduce(Int64(0)) {
            partial, name in
            partial + Int64(payload.compressedArrays[name]?.nbytes ?? 0)
        }
        let valueBytes = TurboQuantKVSnapshotArrayName.ordered.suffix(5).reduce(Int64(0)) {
            partial, name in
            partial + Int64(payload.compressedArrays[name]?.nbytes ?? 0)
        }
        payload.manifest.compressedKeyBytes = keyBytes
        payload.manifest.compressedValueBytes = valueBytes
        payload.manifest.blobByteCount = keyBytes + valueBytes
    }
    }
}
