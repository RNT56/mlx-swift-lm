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
        #expect(decoded.schemaVersion == TurboQuantKVSnapshotManifest.currentSchemaVersion)
        #expect(decoded.kvCodec == .polarQJL)
        #expect(decoded.quantizationMode == QuantizationMode.affine.rawValue)
        #expect(decoded.keyBits == TurboQuantPreset.turbo3_5.effectiveBits)
        #expect(decoded.modelID == "model-a")
        #expect(decoded.tokenPrefixHash == "prefix-a")
    }

    @Test func manifestRoundTripsAffineInt4Metadata() throws {
        let manifest = TurboQuantKVSnapshotManifest(
            snapshotID: UUID(uuidString: "00000000-0000-0000-0000-000000000011")!,
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000012")!,
            identity: Self.identity(),
            turboQuantLayoutVersion: 4,
            logicalLength: 8,
            pinnedPrefixLength: 0,
            compressedKeyBytes: 10,
            compressedValueBytes: 11,
            blobByteCount: 21,
            encryptionKeyID: "key-1",
            createdAt: Date(timeIntervalSinceReferenceDate: 1),
            cacheKind: "AffineInt4KVCache",
            kvCodec: .affineInt4,
            preset: "affine_int4",
            requestedBackend: TurboQuantBackend.mlxPacked.rawValue,
            activeBackend: TurboQuantBackend.mlxPacked.rawValue,
            quantizationMode: QuantizationMode.affine.rawValue,
            keyBits: TurboQuantKVCodec.affineInt4Bits,
            groupSize: TurboQuantKVCodec.affineInt4DefaultGroupSize,
            valueBits: TurboQuantKVCodec.affineInt4Bits,
            capacity: 8,
            ringOffset: 0,
            batchSize: 1,
            kvHeadCount: 2,
            keyHeadDimension: 256,
            valueHeadDimension: 256,
            selectedPath: TurboQuantAttentionPath.affineInt4Native.rawValue,
            arrays: [
                TurboQuantKVSnapshotArrayDescriptor(
                    name: "key_scales",
                    dtype: "float32",
                    shape: [1, 2, 8, 8],
                    byteCount: 512
                ),
                TurboQuantKVSnapshotArrayDescriptor(
                    name: "key_biases",
                    dtype: "float32",
                    shape: [1, 2, 8, 8],
                    byteCount: 512
                ),
            ]
        )

        let decoded = try JSONDecoder().decode(
            TurboQuantKVSnapshotManifest.self,
            from: try JSONEncoder().encode(manifest)
        )

        #expect(decoded.kvCodec == .affineInt4)
        #expect(decoded.quantizationMode == QuantizationMode.affine.rawValue)
        #expect(decoded.keyBits == TurboQuantKVCodec.affineInt4Bits)
        #expect(decoded.groupSize == TurboQuantKVCodec.affineInt4DefaultGroupSize)
        #expect(decoded.selectedPath == TurboQuantAttentionPath.affineInt4Native.rawValue)
        #expect(decoded.arrays.map(\.name).contains("key_biases"))
    }

    @Test func manifestAcceptsOptionalPolarWHTValuePayloadArrays() throws {
        let compressedArrays = Dictionary(
            uniqueKeysWithValues: TurboQuantKVSnapshotArrayName.ordered.enumerated().map {
                index, name in
                (
                    name,
                    MLXArray.zeros(
                        index == 4 || index == 9 ? [1, 2, 8, 1, 1] : [1, 2, 8, 1, 1],
                        dtype: index == 4 || index == 9 ? .float32 : .uint32
                    )
                )
            }
        )
        let sidecarArrays = [
            TurboQuantKVSnapshotArrayName.polarWHTKeyPackedIndices:
                MLXArray.zeros([1, 2, 8, 8], dtype: .uint32),
            TurboQuantKVSnapshotArrayName.polarWHTKeyNorms:
                MLXArray.zeros([1, 2, 8], dtype: .float32),
            TurboQuantKVSnapshotArrayName.polarWHTValuePackedIndices:
                MLXArray.zeros([1, 2, 8, 6], dtype: .uint32),
            TurboQuantKVSnapshotArrayName.polarWHTValueNorms:
                MLXArray.zeros([1, 2, 8], dtype: .float32),
        ]
        let arrays = compressedArrays.merging(sidecarArrays) { current, _ in current }
        let keyBytes = TurboQuantKVSnapshotArrayName.ordered.prefix(5).reduce(Int64(0)) {
            $0 + Int64(arrays[$1]?.nbytes ?? 0)
        }
        let valueBytes = TurboQuantKVSnapshotArrayName.ordered.suffix(5).reduce(Int64(0)) {
            $0 + Int64(arrays[$1]?.nbytes ?? 0)
        }
        let polarWHTKeyBytes = TurboQuantKVSnapshotArrayName.polarWHTKeyOrdered.reduce(
            Int64(0)
        ) {
            $0 + Int64(arrays[$1]?.nbytes ?? 0)
        }
        let polarWHTValueBytes = TurboQuantKVSnapshotArrayName.polarWHTValueOrdered.reduce(
            Int64(0)
        ) {
            $0 + Int64(arrays[$1]?.nbytes ?? 0)
        }
        var manifest = TurboQuantKVSnapshotManifest(
            conversationID: UUID(),
            identity: Self.identity(),
            turboQuantLayoutVersion: TurboQuantAttentionLayout.currentVersion,
            logicalLength: 8,
            pinnedPrefixLength: 0,
            compressedKeyBytes: keyBytes,
            compressedValueBytes: valueBytes,
            blobByteCount: keyBytes + valueBytes + polarWHTKeyBytes + polarWHTValueBytes,
            encryptionKeyID: "key-1",
            cacheKind: "TurboQuantKVCache",
            kvCodec: .polarWHT,
            preset: TurboQuantPreset.turbo4v2.rawValue,
            requestedBackend: TurboQuantBackend.metalPolarWHT.rawValue,
            activeBackend: TurboQuantBackend.mlxPacked.rawValue,
            groupSize: 64,
            valueBits: 3,
            capacity: 8,
            ringOffset: 0,
            batchSize: 1,
            kvHeadCount: 2,
            keyHeadDimension: 64,
            valueHeadDimension: 64,
            polarWHTKeyBytes: polarWHTKeyBytes,
            polarWHTKeyPayloadAllocated: true,
            polarWHTKeyBits: 4,
            polarWHTKeySeed: 0xBEEF,
            polarWHTKeyPackedWordsPerVector: 8,
            polarWHTValueBytes: polarWHTValueBytes,
            polarWHTValuePayloadAllocated: true,
            polarWHTValueBits: 3,
            polarWHTValueSeed: 0xA5A5,
            polarWHTValuePackedWordsPerVector: 6,
            arrays: TurboQuantKVSnapshotArrayName.allKnown.compactMap { name in
                arrays[name].map { TurboQuantKVSnapshotArrayDescriptor(name: name, array: $0) }
            }
        )

        try manifest.validateArrayDescriptors(against: arrays)

        var keyManifest = manifest
        keyManifest.arrays.removeAll {
            $0.name == TurboQuantKVSnapshotArrayName.polarWHTKeyNorms
        }
        #expect(throws: TurboQuantKVSnapshotError.self) {
            try keyManifest.validateArrayDescriptors(against: arrays)
        }

        manifest.arrays.removeAll {
            $0.name == TurboQuantKVSnapshotArrayName.polarWHTValueNorms
        }
        #expect(throws: TurboQuantKVSnapshotError.self) {
            try manifest.validateArrayDescriptors(against: arrays)
        }
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

    @Test func polarWHTSnapshotRoundTripsValueSidecars() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            return
        }
        let cache = TurboQuantKVCache(
            preset: .turbo4v2,
            backend: .metalPolarQJL,
            kvCodec: .polarWHT
        )
        let keys = MLXArray.ones([1, 2, 4, 64], dtype: .float32)
        let values = MLXArray((0 ..< 512).map { Float(($0 % 29) - 14) / 29 }, [1, 2, 4, 64])
        _ = try cache.updateCompressed(keys: keys, values: values)

        let originalKeySidecar = try #require(cache.polarWHTKeyState)
        let originalValueSidecar = try #require(cache.polarWHTValueState)
        let payload = try cache.exportSnapshot(
            identity: Self.identity(),
            conversationID: UUID(uuidString: "00000000-0000-0000-0000-000000000023")!
        )
        let restored = TurboQuantKVCache(
            preset: .turbo4v2,
            backend: .metalPolarQJL,
            kvCodec: .polarWHT
        )
        try restored.importSnapshot(payload, expectedIdentity: Self.identity())
        let restoredKeySidecar = try #require(restored.polarWHTKeyState)
        let restoredValueSidecar = try #require(restored.polarWHTValueState)

        #expect(payload.manifest.kvCodec == .polarWHT)
        #expect(payload.manifest.polarWHTKeyPayloadAllocated)
        #expect(payload.manifest.polarWHTKeyBytes > 0)
        #expect(payload.manifest.polarWHTValuePayloadAllocated)
        #expect(payload.manifest.polarWHTValueBytes > 0)
        #expect(
            payload.compressedArrays[
                TurboQuantKVSnapshotArrayName.polarWHTKeyPackedIndices
            ] != nil
        )
        #expect(
            payload.compressedArrays[
                TurboQuantKVSnapshotArrayName.polarWHTValuePackedIndices
            ] != nil
        )
        #expect(restored.runtimeSnapshot().polarWHTKeyPayloadAllocated)
        #expect(restored.runtimeSnapshot().polarWHTKeyBytes == cache.runtimeSnapshot().polarWHTKeyBytes)
        #expect(restored.runtimeSnapshot().polarWHTValuePayloadAllocated)
        #expect(restored.runtimeSnapshot().polarWHTValueBytes == cache.runtimeSnapshot().polarWHTValueBytes)
        #expect(
            allClose(
                try turboQuantPolarWHTReferenceDecodeAttentionValues(originalKeySidecar),
                try turboQuantPolarWHTReferenceDecodeAttentionValues(restoredKeySidecar),
                rtol: 1e-5,
                atol: 1e-5
            )
            .item(Bool.self)
        )
        #expect(
            allClose(
                try turboQuantPolarWHTReferenceDecodeAttentionValues(originalValueSidecar),
                try turboQuantPolarWHTReferenceDecodeAttentionValues(restoredValueSidecar),
                rtol: 1e-5,
                atol: 1e-5
            )
            .item(Bool.self)
        )
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

    /// T1.4 stage 1: the K scale plane was dieted from 3 scales/group to 2 (norm, residual_norm);
    /// the third slot was write-only (always 0.0) and never read. Snapshots written before the
    /// diet still carry a last-dim-3 key scale plane. Schema-range validation alone would accept
    /// this (schemaVersion 4 <= currentSchemaVersion 5), so the decisive gate is the explicit
    /// `turboQuantRequireKeyScalesPerGroup` shape check in the restore path — confirm it rejects
    /// a stale 3-scale key plane instead of silently misreading it.
    @Test func importRejectsStaleThreeScaleKeyPlaneBeforeUse() throws {
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

        var staleKeyScales = payload
        let keyScales = try #require(payload.compressedArrays["key.scales"])
        #expect(keyScales.dim(4) == 2, "current encoder must already produce the dieted 2-scale K plane")
        let staleScales = concatenated(
            [keyScales, MLXArray.zeros([keyScales.dim(0), keyScales.dim(1), keyScales.dim(2), keyScales.dim(3), 1], dtype: keyScales.dtype)],
            axis: 4
        )
        Self.replaceArray(in: &staleKeyScales, name: "key.scales", with: staleScales)

        let target = TurboQuantKVCache(backend: .metalPolarQJL)
        #expect(throws: TurboQuantRuntimeFailure.self) {
            try target.importSnapshot(staleKeyScales, expectedIdentity: Self.identity())
        }
        #expect(target.runtimeSnapshot().logicalLength == 0)
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
