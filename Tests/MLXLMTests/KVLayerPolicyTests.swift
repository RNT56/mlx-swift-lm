import Darwin
import Foundation
import IntegrationTestHelpers
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {
    @Suite(.serialized)
    struct KVLayerPolicyTests {
        private func patternedArray(
            shape: [Int],
            modulus: Int,
            centeredAt center: Int,
            denominator: Float
        ) -> MLXArray {
            let count = shape.reduce(1, *)
            var values = [Float]()
            values.reserveCapacity(count)
            for index in 0 ..< count {
                values.append(Float((index % modulus) - center) / denominator)
            }
            return MLXArray(values, shape)
        }

        private func expectPolarWHTValueState(
            _ code: TurboQuantPolarWHTAttentionValueCode,
            matches values: MLXArray
        ) throws {
            let decoded = try turboQuantPolarWHTReferenceDecodeAttentionValues(code)
            let reference = try turboQuantPolarWHTReferenceDecode(
                turboQuantPolarWHTReferenceEncode(
                    values,
                    bits: code.bits,
                    seed: code.seed
                )
            )

            #expect(decoded.shape == reference.shape)
            #expect(allClose(decoded, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }

        @Test func testOptIQConfigDecodingAndStableHash() throws {
            let json = """
                [
                  { "layer_idx": 3, "bits": 8, "group_size": 64 },
                  { "layer_idx": 7, "bits": 4, "group_size": 64 }
                ]
                """.data(using: .utf8)!

            let policy = try KVLayerPolicy.optiQKVConfig(data: json)

            #expect(policy.codec(forLayerIndex: 3) == .affineK8V4)
            #expect(
                policy.codec(forLayerIndex: 7)
                    == .turboQuant(
                        preset: .turbo4v2,
                        valueBits: 4,
                        groupSize: 64,
                        backend: .metalPolarQJL
                    )
            )
            #expect(policy.codec(forLayerIndex: 0) == .inherit)
            let decodedAgain = try KVLayerPolicy.optiQKVConfig(data: json)
            #expect(policy.stableHash == decodedAgain.stableHash)
            #expect(policy.summary().contains("3:affineK8V4"))
        }

        @Test func testPolarWHTCodecAndBackendRoundTrip() throws {
            let codec = try JSONDecoder().decode(
                TurboQuantKVCodec.self,
                from: Data(#""polar_wht""#.utf8)
            )
            let policy = KVLayerPolicy(
                defaultCodec: .turboQuant(
                    preset: .turbo4v2,
                    valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    groupSize: 64,
                    backend: .metalPolarWHT
                )
            )
            let decoded = try JSONDecoder().decode(
                KVLayerPolicy.self,
                from: try JSONEncoder().encode(policy)
            )
            let cache = TurboQuantKVCache(
                preset: .turbo4v2,
                backend: .metalPolarWHT,
                valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits
            )
            let polarWHTAvailable = TurboQuantKernelAvailability.current.supports(.metalPolarWHT)

            #expect(codec == .polarWHT)
            #expect(TurboQuantKVCodec.polarWHTDefaultValueBits == 3)
            #expect(decoded == policy)
            #expect(policy.summary().contains("metalPolarWHT"))
            #expect(cache.requestedBackend == .metalPolarWHT)
            #expect(cache.activeBackend == (polarWHTAvailable ? .metalPolarWHT : .mlxPacked))
            #expect(cache.kvCodec == .polarWHT)
            #expect(cache.valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)
            #expect(cache.diagnostics.kvCodec == .polarWHT)
            #expect(cache.diagnostics.metalCodecAvailable == polarWHTAvailable)
            #expect(cache.diagnostics.metalAttentionAvailable == polarWHTAvailable)
            #expect(cache.diagnostics.polarWHTValueBytes == 0)
            #expect(!cache.diagnostics.polarWHTValuePayloadAllocated)
            #expect(cache.runtimeSnapshot().kvCodec == .polarWHT)
            #expect(cache.runtimeSnapshot().valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)
            #expect(cache.runtimeSnapshot().polarWHTValueBytes == 0)
            #expect(!cache.runtimeSnapshot().polarWHTValuePayloadAllocated)
            if polarWHTAvailable {
                #expect(cache.backendFallbackReason == nil)
                #expect(cache.diagnostics.fallbackReason == nil)
                #expect(cache.attentionDiagnostics.fallbackReason == nil)
            } else {
                #expect(cache.backendFallbackReason?.contains("PolarWHT") == true)
                #expect(cache.diagnostics.fallbackReason?.contains("PolarWHT") == true)
                #expect(cache.attentionDiagnostics.fallbackReason?.contains("PolarWHT") == true)
            }
            let queries = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
            let keys = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
            let values = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
            #expect(
                cache.supportsCompressedAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    mask: .none
                ) == polarWHTAvailable
            )
            #expect(cache.attentionDiagnostics.polarWHTValueBytes == 0)
            #expect(!cache.attentionDiagnostics.polarWHTValuePayloadAllocated)
            if !polarWHTAvailable {
                #expect(
                    cache.attentionDiagnostics.lastUnsupportedShape?
                        .contains("value payload unavailable") == true
                )
            }
        }

        @Test func testPolarWHTFactoryAndConversionPreserveCodecMetadata() throws {
            try Device.withDefaultDevice(.cpu) {
                let parameters = GenerateParameters(
                    quantizedKVStart: 0,
                    kvCacheStrategy: .turboQuant,
                    kvCodec: .polarWHT,
                    turboQuantPreset: .turbo4v2,
                    turboQuantBackend: .metalPolarWHT
                )
                let direct = try #require(makeAttentionKVCache(parameters: parameters) as? TurboQuantKVCache)
                let polarWHTAvailable = TurboQuantKernelAvailability.current.supports(.metalPolarWHT)

                #expect(direct.kvCodec == .polarWHT)
                #expect(direct.requestedBackend == .metalPolarWHT)
                #expect(direct.activeBackend == (polarWHTAvailable ? .metalPolarWHT : .mlxPacked))
                #expect(direct.valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)
                #expect(direct.runtimeSnapshot().kvCodec == .polarWHT)
                if polarWHTAvailable {
                    #expect(direct.runtimeSnapshot().runtimeFallbackReason == nil)
                } else {
                    #expect(direct.runtimeSnapshot().runtimeFallbackReason?.contains("PolarWHT") == true)
                }
                #expect(direct.diagnostics.polarWHTKeyBytes == 0)
                #expect(!direct.diagnostics.polarWHTKeyPayloadAllocated)
                #expect(direct.diagnostics.polarWHTValueBytes == 0)
                #expect(!direct.diagnostics.polarWHTValuePayloadAllocated)

                var cache: [KVCache] = [KVCacheSimple()]
                let keys = MLXArray.ones([1, 1, 2, 64], dtype: .float32)
                let values = keys + 0.5
                _ = cache[0].update(keys: keys, values: values)

                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: nil,
                    kvGroupSize: 64,
                    quantizedKVStart: 0,
                    kvCacheStrategy: .turboQuant,
                    kvCodec: .polarWHT,
                    turboQuantPreset: .turbo4v2,
                    turboQuantBackend: .metalPolarWHT
                )

                let converted = try #require(cache[0] as? TurboQuantKVCache)
                #expect(converted.kvCodec == .polarWHT)
                #expect(converted.requestedBackend == .metalPolarWHT)
                #expect(converted.activeBackend == (polarWHTAvailable ? .metalPolarWHT : .mlxPacked))
                #expect(converted.valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)
                #expect(converted.runtimeSnapshot().kvCodec == .polarWHT)
                if polarWHTAvailable {
                    #expect(converted.runtimeSnapshot().runtimeFallbackReason == nil)
                } else {
                    #expect(converted.runtimeSnapshot().runtimeFallbackReason?.contains("PolarWHT") == true)
                }
                #expect(converted.diagnostics.polarWHTKeyBytes > 0)
                #expect(converted.diagnostics.polarWHTKeyPayloadAllocated)
                #expect(converted.diagnostics.polarWHTValueBytes > 0)
                #expect(converted.diagnostics.polarWHTValuePayloadAllocated)
                #expect(converted.runtimeSnapshot().polarWHTKeyBytes > 0)
                #expect(converted.runtimeSnapshot().polarWHTKeyPayloadAllocated)
                #expect(converted.runtimeSnapshot().polarWHTValueBytes > 0)
                #expect(converted.runtimeSnapshot().polarWHTValuePayloadAllocated)
            }
        }

        @Test func testHybridPolarWHTConversionKeepsAffineKeySidecarOnly() throws {
            try Device.withDefaultDevice(.cpu) {
                let polarWHTAvailable = TurboQuantKernelAvailability.current.supports(.metalPolarWHT)
                guard polarWHTAvailable else { return }

                var cache: [KVCache] = [KVCacheSimple()]
                let keys = MLXArray.ones([1, 8, 4, 128], dtype: .float32)
                let values = keys + 0.25
                let precisionPolicy = TurboQuantKVPrecisionPolicy(
                    key: .affineQ8,
                    value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                    boundary: .disabled
                )
                _ = cache[0].update(keys: keys, values: values)

                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: nil,
                    kvGroupSize: 64,
                    quantizedKVStart: 0,
                    kvCacheStrategy: .turboQuant,
                    kvCodec: .polarWHT,
                    turboQuantPreset: .turbo8,
                    turboQuantBackend: .metalPolarWHT,
                    turboQuantValueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    turboQuantPrecisionPolicy: precisionPolicy
                )

                let converted = try #require(cache[0] as? TurboQuantKVCache)
                let affineKeyState = try #require(converted.hybridAffineKeyState)
                #expect(converted.precisionPolicy.key == .affineQ8)
                #expect(converted.kvCodec == .polarWHT)
                #expect(converted.activeBackend == .metalPolarWHT)
                #expect(converted.diagnostics.polarWHTKeyBytes == 0)
                #expect(!converted.diagnostics.polarWHTKeyPayloadAllocated)
                #expect(converted.diagnostics.polarWHTValueBytes > 0)
                #expect(converted.diagnostics.polarWHTValuePayloadAllocated)
                #expect(affineKeyState.0.dim(-3) == 8)
                #expect(affineKeyState.0.dim(-2) == 4)
                #expect(converted.runtimeSnapshot().polarWHTKeyBytes == 0)
                #expect(!converted.runtimeSnapshot().polarWHTKeyPayloadAllocated)
                #expect(converted.runtimeSnapshot().polarWHTValuePayloadAllocated)
            }
        }

        @Test func testPolarWHTPackedFallbackAllocatesValueSidecar() throws {
            try Device.withDefaultDevice(.cpu) {
                let cache = TurboQuantKVCache(
                    preset: .turbo4v2,
                    backend: .metalPolarWHT
                )
                let keys = MLXArray.ones([1, 1, 2, 64], dtype: .float32)
                let values = MLXArray((0 ..< 128).map { Float($0) / 128 }, [1, 1, 2, 64])

                _ = cache.updateQuantized(keys: keys, values: values)

                let snapshot = cache.runtimeSnapshot()
                let diagnostics = cache.diagnostics
                let polarWHTAvailable = TurboQuantKernelAvailability.current.supports(.metalPolarWHT)
                #expect(cache.kvCodec == .polarWHT)
                #expect(cache.activeBackend == (polarWHTAvailable ? .metalPolarWHT : .mlxPacked))
                #expect(snapshot.logicalLength == 2)
                #expect(snapshot.capacity >= snapshot.logicalLength)
                #expect(snapshot.polarWHTKeyPayloadAllocated)
                #expect(snapshot.polarWHTKeyBytes > 0)
                #expect(snapshot.polarWHTValuePayloadAllocated)
                #expect(snapshot.polarWHTValueBytes > 0)
                #expect(diagnostics.polarWHTKeyPayloadAllocated)
                #expect(diagnostics.polarWHTKeyBytes == snapshot.polarWHTKeyBytes)
                #expect(diagnostics.polarWHTValuePayloadAllocated)
                #expect(diagnostics.polarWHTValueBytes == snapshot.polarWHTValueBytes)
                #expect(
                    cache.cacheFootprint.compressedBytes
                        >= snapshot.polarWHTKeyBytes + snapshot.polarWHTValueBytes
                )
                #expect(cache.attentionDiagnostics.polarWHTKeyBytes == snapshot.polarWHTKeyBytes)
                #expect(cache.attentionDiagnostics.polarWHTKeyPayloadAllocated)
                #expect(cache.attentionDiagnostics.polarWHTValueBytes == snapshot.polarWHTValueBytes)
                #expect(cache.attentionDiagnostics.polarWHTValuePayloadAllocated)
                let queries = MLXArray.ones([1, 1, 1, 64], dtype: .float32)
                #expect(
                    cache.supportsCompressedAttention(
                        queries: queries,
                        keys: keys,
                        values: values,
                        mask: .none
                    ) == polarWHTAvailable
                )
                if !polarWHTAvailable {
                    #expect(
                        cache.attentionDiagnostics.lastUnsupportedShape?
                            .contains("value payload present") == true
                    )
                }
                let prefillQueries = MLXArray.ones([1, 1, 2, 64], dtype: .float32)
                #expect(
                    !cache.supportsCompressedAttention(
                        queries: prefillQueries,
                        keys: keys,
                        values: values,
                        mask: .none
                    )
                )
                if polarWHTAvailable {
                    #expect(
                        cache.attentionDiagnostics.lastUnsupportedShape?
                            .contains("decode-only") == true
                    )
                }
                try expectPolarWHTValueState(
                    try #require(cache.polarWHTKeyState),
                    matches: keys
                )
                try expectPolarWHTValueState(
                    try #require(cache.polarWHTValueState),
                    matches: values
                )

                let copied = try #require(cache.copy() as? TurboQuantKVCache)
                #expect(copied.runtimeSnapshot().polarWHTKeyBytes == snapshot.polarWHTKeyBytes)
                #expect(copied.diagnostics.polarWHTKeyPayloadAllocated)
                #expect(copied.runtimeSnapshot().polarWHTValueBytes == snapshot.polarWHTValueBytes)
                #expect(copied.diagnostics.polarWHTValuePayloadAllocated)
                try expectPolarWHTValueState(
                    try #require(copied.polarWHTKeyState),
                    matches: keys
                )
                try expectPolarWHTValueState(
                    try #require(copied.polarWHTValueState),
                    matches: values
                )

                let nextKeys = MLXArray.ones([1, 1, 1, 64], dtype: .float32) * 2
                let nextValues = MLXArray.ones([1, 1, 1, 64], dtype: .float32) * 0.25
                _ = cache.updateQuantized(keys: nextKeys, values: nextValues)

                let appended = cache.runtimeSnapshot()
                #expect(appended.logicalLength == 3)
                #expect(appended.polarWHTKeyPayloadAllocated)
                #expect(appended.polarWHTKeyBytes == snapshot.polarWHTKeyBytes)
                #expect(appended.polarWHTValuePayloadAllocated)
                #expect(appended.polarWHTValueBytes == snapshot.polarWHTValueBytes)
                let appendedKeys = MLXArray(
                    [Float](repeating: 1, count: 128)
                        + [Float](repeating: 2, count: 64),
                    [1, 1, 3, 64]
                )
                let appendedValues = MLXArray(
                    (0 ..< 128).map { Float($0) / 128 }
                        + [Float](repeating: 0.25, count: 64),
                    [1, 1, 3, 64]
                )
                try expectPolarWHTValueState(
                    try #require(cache.polarWHTKeyState),
                    matches: appendedKeys
                )
                try expectPolarWHTValueState(
                    try #require(cache.polarWHTValueState),
                    matches: appendedValues
                )
            }
        }

        @Test func testMetalPolarWHTPrefillCommitsCompressedStorageForDecode() throws {
            guard TurboQuantKernelAvailability.current.supports(.metalPolarWHT) else {
                return
            }

            let cache = RotatingTurboQuantKVCache(
                maxSize: 8,
                keep: 1,
                preset: .turbo4v2,
                backend: .metalPolarWHT
            )
            let prefillQueries = MLXArray(
                (0 ..< 128).map { Float(($0 % 19) - 9) / 19 },
                [1, 1, 2, 64]
            )
            let prefillKeys = MLXArray(
                (0 ..< 128).map { Float(($0 % 23) - 11) / 23 },
                [1, 1, 2, 64]
            )
            let prefillValues = MLXArray(
                (0 ..< 128).map { Float(($0 % 29) - 14) / 29 },
                [1, 1, 2, 64]
            )
            let scale = Float(0.125)

            let prefill = try attentionWithCacheUpdateReturningStateThrowing(
                queries: prefillQueries,
                keys: prefillKeys,
                values: prefillValues,
                cache: cache,
                scale: scale,
                mask: .causal
            )
            guard case .turboQuant = prefill.state else {
                Issue.record("Expected PolarWHT prefill to commit compressed state")
                return
            }
            let expectedPrefill = MLXFast.scaledDotProductAttention(
                queries: prefillQueries,
                keys: prefillKeys,
                values: prefillValues,
                scale: scale,
                mask: .causal
            )
            let prefillSnapshot = cache.runtimeSnapshot()
            #expect(allClose(prefill.output, expectedPrefill, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(prefillSnapshot.logicalLength == 2)
            #expect(prefillSnapshot.rawShadowAllocated == false)
            #expect(prefillSnapshot.polarWHTKeyPayloadAllocated)
            #expect(prefillSnapshot.polarWHTValuePayloadAllocated)
            #expect(cache.compressedState != nil)

            let decodeQueries = MLXArray(
                (0 ..< 64).map { Float(($0 % 17) - 8) / 17 },
                [1, 1, 1, 64]
            )
            let decodeKeys = MLXArray(
                (0 ..< 64).map { Float(($0 % 31) - 15) / 31 },
                [1, 1, 1, 64]
            )
            let decodeValues = MLXArray(
                (0 ..< 64).map { Float(($0 % 37) - 18) / 37 },
                [1, 1, 1, 64]
            )

            let decode = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: cache,
                scale: scale,
                mask: .causal
            )
            guard case .turboQuant = decode.state else {
                Issue.record("Expected PolarWHT decode to remain compressed")
                return
            }

            #expect(decode.output.shape == [1, 1, 1, 64])
            #expect(cache.runtimeSnapshot().logicalLength == 3)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
            #expect(cache.attentionDiagnostics.activeAttentionPath == .metalPolarWHTHybrid)
            #expect(cache.fallbackResults.last?.toPath == .metalPolarWHTHybrid)
        }

        @Test func testMetalHybridK8PolarWHTPrefillCommitsValueOnlySidecarForDecode() throws {
            guard TurboQuantKernelAvailability.current.attentionCapabilities
                .hybridK8PolarWHTValueAttention
            else {
                return
            }
            let precisionPolicy = TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                boundary: .disabled
            )
            let cache = RotatingTurboQuantKVCache(
                maxSize: 8,
                keep: 1,
                preset: .turbo8,
                backend: .metalPolarWHT,
                kvCodec: .polarWHT,
                valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                precisionPolicy: precisionPolicy
            )
            let prefillQueries = MLXArray(
                (0 ..< 128).map { Float(($0 % 19) - 9) / 19 },
                [1, 1, 2, 64]
            )
            let prefillKeys = MLXArray(
                (0 ..< 128).map { Float(($0 % 23) - 11) / 23 },
                [1, 1, 2, 64]
            )
            let prefillValues = MLXArray(
                (0 ..< 128).map { Float(($0 % 29) - 14) / 29 },
                [1, 1, 2, 64]
            )
            let scale = Float(0.125)

            let prefill = try attentionWithCacheUpdateReturningStateThrowing(
                queries: prefillQueries,
                keys: prefillKeys,
                values: prefillValues,
                cache: cache,
                scale: scale,
                mask: .causal
            )
            guard case .turboQuant = prefill.state else {
                Issue.record("Expected hybrid PolarWHT prefill to commit compressed state")
                return
            }
            let prefillSnapshot = cache.runtimeSnapshot()
            #expect(prefillSnapshot.rawShadowAllocated == false)
            #expect(!prefillSnapshot.polarWHTKeyPayloadAllocated)
            #expect(prefillSnapshot.polarWHTValuePayloadAllocated)

            let decodeQueries = MLXArray(
                (0 ..< 64).map { Float(($0 % 17) - 8) / 17 },
                [1, 1, 1, 64]
            )
            let decodeKeys = MLXArray(
                (0 ..< 64).map { Float(($0 % 31) - 15) / 31 },
                [1, 1, 1, 64]
            )
            let decodeValues = MLXArray(
                (0 ..< 64).map { Float(($0 % 37) - 18) / 37 },
                [1, 1, 1, 64]
            )

            let decode = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: cache,
                scale: scale,
                mask: .causal
            )
            guard case .turboQuant = decode.state else {
                Issue.record("Expected hybrid PolarWHT decode to remain compressed")
                return
            }

            #expect(decode.output.shape == [1, 1, 1, 64])
            #expect(cache.runtimeSnapshot().logicalLength == 3)
            #expect(!cache.attentionDiagnostics.polarWHTKeyPayloadAllocated)
            #expect(cache.attentionDiagnostics.polarWHTValuePayloadAllocated)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
            #expect(cache.attentionDiagnostics.activeAttentionPath == .metalHybridK8PolarWHTValue)
            #expect(cache.fallbackResults.last?.toPath == .metalHybridK8PolarWHTValue)
        }

        @Test func testMetalHybridK8PolarWHTRotatingFullWindowMatchesReferenceHybrid()
            throws
        {
            Stream.gpu.synchronize()
            Memory.clearCache()
            defer {
                Stream.gpu.synchronize()
                Memory.clearCache()
            }
            guard TurboQuantKernelAvailability.current.attentionCapabilities
                .hybridK8PolarWHTValueAttention
            else {
                return
            }
            let precisionPolicy = TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                boundary: .disabled
            )
            func makeCache(backend: TurboQuantBackend) -> RotatingTurboQuantKVCache {
                RotatingTurboQuantKVCache(
                    maxSize: 8,
                    keep: 1,
                    preset: .turbo8,
                    backend: backend,
                    kvCodec: .polarWHT,
                    valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    precisionPolicy: precisionPolicy,
                    requestedRuntimeMode: .throughputTurboQuant,
                    resolvedRuntimeMode: .throughputTurboQuant
                )
            }
            let metalCache = makeCache(backend: .metalPolarWHT)
            let referenceCache = makeCache(backend: .polarWHTReference)
            let prefillQueryCount = 4 * 8 * 64
            let prefillTokenCount = 8 * 64
            let decodeQueryCount = 4 * 64
            let prefillQueryValues = (0 ..< prefillQueryCount).map { index -> Float in
                Float((index % 19) - 9) / 19
            }
            let prefillKeyValues = (0 ..< prefillTokenCount).map { index -> Float in
                Float((index % 23) - 11) / 23
            }
            let prefillValueValues = (0 ..< prefillTokenCount).map { index -> Float in
                Float((index % 29) - 14) / 29
            }
            let decodeQueryValues = (0 ..< decodeQueryCount).map { index -> Float in
                Float((index % 17) - 8) / 17
            }
            let decodeKeyValues = (0 ..< 64).map { index -> Float in
                Float((index % 31) - 15) / 31
            }
            let decodeValueValues = (0 ..< 64).map { index -> Float in
                Float((index % 37) - 18) / 37
            }
            let prefillQueries = MLXArray(prefillQueryValues, [1, 4, 8, 64])
            let prefillKeys = MLXArray(prefillKeyValues, [1, 1, 8, 64])
            let prefillValues = MLXArray(prefillValueValues, [1, 1, 8, 64])
            let decodeQueries = MLXArray(decodeQueryValues, [1, 4, 1, 64])
            let decodeKeys = MLXArray(decodeKeyValues, [1, 1, 1, 64])
            let decodeValues = MLXArray(decodeValueValues, [1, 1, 1, 64])
            let scale = Float(0.125)

            _ = try attentionWithCacheUpdateReturningStateThrowing(
                queries: prefillQueries,
                keys: prefillKeys,
                values: prefillValues,
                cache: metalCache,
                scale: scale,
                mask: .causal
            )
            _ = try attentionWithCacheUpdateReturningStateThrowing(
                queries: prefillQueries,
                keys: prefillKeys,
                values: prefillValues,
                cache: referenceCache,
                scale: scale,
                mask: .causal
            )
            let metal = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: metalCache,
                scale: scale,
                mask: .causal
            )
            let reference = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: referenceCache,
                scale: scale,
                mask: .causal
            )

            eval(metal.output, reference.output)
            #expect(metalCache.attentionDiagnostics.activeAttentionPath == .metalHybridK8PolarWHTValue)
            #expect(referenceCache.attentionDiagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
            let snapshot = metalCache.runtimeSnapshot()
            let decodedValueState = try #require(metalCache.polarWHTDecodedValueState)
            #expect(snapshot.ringOffset == 1)
            #expect(snapshot.decodedActiveValueBytes > 0)
            #expect(decodedValueState.shape == [1, 1, 8, 64])
            #expect(metalCache.polarWHTDecodedValueLayout?.ringOffset == 1)
            #expect(
                metalCache.fallbackResults.last?.reason
                    .contains("buffered PolarWHT-decoded V") == true
            )
            let maxDelta = abs(metal.output.asType(.float32) - reference.output.asType(.float32))
                .max().item(Float.self)
            #expect(maxDelta < 0.025)
        }

        @Test func testMetalHybridK8PolarWHTLongLinearChunkedMatchesReferenceHybrid() throws {
            guard TurboQuantKernelAvailability.current.attentionCapabilities
                .hybridK8PolarWHTValueAttention
            else {
                return
            }
            let precisionPolicy = TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                boundary: .disabled
            )
            func makeCache(backend: TurboQuantBackend) -> TurboQuantKVCache {
                TurboQuantKVCache(
                    preset: .turbo8,
                    backend: backend,
                    kvCodec: .polarWHT,
                    valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    precisionPolicy: precisionPolicy,
                    requestedRuntimeMode: .throughputTurboQuant,
                    resolvedRuntimeMode: .throughputTurboQuant
                )
            }
            let metalCache = makeCache(backend: .metalPolarWHT)
            let referenceCache = makeCache(backend: .polarWHTReference)
            let contextLength = 4096
            let chunkSize = 512
            let scale = Float(1.0 / sqrt(128.0))

            for chunkStart in stride(from: 0, to: contextLength, by: chunkSize) {
                let queryOffset = chunkStart * 16 * 128
                let keyOffset = chunkStart * 8 * 128
                let prefillQueries = patternedArray(
                    shape: [1, chunkSize, 16, 128],
                    modulus: 31,
                    centeredAt: 15,
                    denominator: 31
                ) + MLXArray(Float(queryOffset % 17) / 257)
                let prefillKeys = patternedArray(
                    shape: [1, chunkSize, 8, 128],
                    modulus: 37,
                    centeredAt: 18,
                    denominator: 37
                ) + MLXArray(Float(keyOffset % 19) / 263)
                let prefillValues = patternedArray(
                    shape: [1, chunkSize, 8, 128],
                    modulus: 41,
                    centeredAt: 20,
                    denominator: 41
                ) + MLXArray(Float(keyOffset % 23) / 269)

                let metalPrefill = try attentionWithCacheUpdateReturningStateThrowing(
                    queries: prefillQueries.transposed(0, 2, 1, 3).asType(.bfloat16),
                    keys: prefillKeys.transposed(0, 2, 1, 3).asType(.bfloat16),
                    values: prefillValues.transposed(0, 2, 1, 3).asType(.bfloat16),
                    cache: metalCache,
                    scale: scale,
                    mask: .causal
                )
                let referencePrefill = try attentionWithCacheUpdateReturningStateThrowing(
                    queries: prefillQueries.transposed(0, 2, 1, 3).asType(.bfloat16),
                    keys: prefillKeys.transposed(0, 2, 1, 3).asType(.bfloat16),
                    values: prefillValues.transposed(0, 2, 1, 3).asType(.bfloat16),
                    cache: referenceCache,
                    scale: scale,
                    mask: .causal
                )
                eval(metalPrefill.output, referencePrefill.output)
            }

            let decodedValueState = try #require(metalCache.polarWHTDecodedValueState)
            let preDecodeSnapshot = metalCache.runtimeSnapshot()
            #expect(decodedValueState.shape == [1, 8, contextLength, 128])
            #expect(preDecodeSnapshot.activeCacheAllocated)
            #expect(preDecodeSnapshot.decodedActiveValueBytes == decodedValueState.nbytes)
            #expect(metalCache.cacheFootprint.rawShadowBytes >= decodedValueState.nbytes)

            let decodeQueries = patternedArray(
                shape: [1, 16, 1, 128],
                modulus: 29,
                centeredAt: 14,
                denominator: 29
            )
            let decodeKeys = patternedArray(
                shape: [1, 8, 1, 128],
                modulus: 43,
                centeredAt: 21,
                denominator: 43
            )
            let decodeValues = patternedArray(
                shape: [1, 8, 1, 128],
                modulus: 47,
                centeredAt: 23,
                denominator: 47
            )

            let metal = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries.asType(.bfloat16),
                keys: decodeKeys.asType(.bfloat16),
                values: decodeValues.asType(.bfloat16),
                cache: metalCache,
                scale: scale,
                mask: .causal
            )
            let reference = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries.asType(.bfloat16),
                keys: decodeKeys.asType(.bfloat16),
                values: decodeValues.asType(.bfloat16),
                cache: referenceCache,
                scale: scale,
                mask: .causal
            )

            eval(metal.output, reference.output)
            #expect(metalCache.attentionDiagnostics.activeAttentionPath == .metalHybridK8PolarWHTValue)
            #expect(referenceCache.attentionDiagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
            #expect(
                metalCache.fallbackResults.last?.reason
                    .contains("buffered PolarWHT-decoded V") == true
            )
            #expect(allClose(metal.output, reference.output, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        }

        @Test func testMetalHybridK8PolarWHTExactPrefillConversionMatchesReferenceHybrid() throws {
            guard TurboQuantKernelAvailability.current.attentionCapabilities
                .hybridK8PolarWHTValueAttention
            else {
                return
            }
            let precisionPolicy = TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                boundary: .disabled
            )
            func convertedRawCache(backend: TurboQuantBackend) throws -> TurboQuantKVCache {
                var cache: [KVCache] = [KVCacheSimple()]
                let contextLength = 4096
                let chunkSize = 512
                for chunkStart in stride(from: 0, to: contextLength, by: chunkSize) {
                    let keyOffset = chunkStart * 8 * 128
                    let keys = patternedArray(
                        shape: [1, 8, chunkSize, 128],
                        modulus: 37,
                        centeredAt: 18,
                        denominator: 37
                    ) + MLXArray(Float(keyOffset % 19) / 263)
                    let values = patternedArray(
                        shape: [1, 8, chunkSize, 128],
                        modulus: 41,
                        centeredAt: 20,
                        denominator: 41
                    ) + MLXArray(Float(keyOffset % 23) / 269)
                    _ = cache[0].update(keys: keys, values: values)
                }
                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: nil,
                    kvGroupSize: 64,
                    quantizedKVStart: 0,
                    kvCacheStrategy: .turboQuant,
                    kvCodec: .polarWHT,
                    turboQuantPreset: .turbo8,
                    turboQuantBackend: backend,
                    turboQuantValueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    turboQuantPrecisionPolicy: precisionPolicy
                )
                return try #require(cache[0] as? TurboQuantKVCache)
            }

            let metalCache = try convertedRawCache(backend: .metalPolarWHT)
            let referenceCache = try convertedRawCache(backend: .polarWHTReference)
            let decodeQueries = patternedArray(
                shape: [1, 16, 1, 128],
                modulus: 29,
                centeredAt: 14,
                denominator: 29
            )
            let decodeKeys = patternedArray(
                shape: [1, 8, 1, 128],
                modulus: 43,
                centeredAt: 21,
                denominator: 43
            )
            let decodeValues = patternedArray(
                shape: [1, 8, 1, 128],
                modulus: 47,
                centeredAt: 23,
                denominator: 47
            )
            let scale = Float(1.0 / sqrt(128.0))

            let metal = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: metalCache,
                scale: scale,
                mask: .causal
            )
            let reference = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: referenceCache,
                scale: scale,
                mask: .causal
            )

            eval(metal.output, reference.output)
            #expect(metalCache.attentionDiagnostics.activeAttentionPath == .metalHybridK8PolarWHTValue)
            #expect(referenceCache.attentionDiagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
            #expect(!metalCache.attentionDiagnostics.polarWHTKeyPayloadAllocated)
            #expect(referenceCache.attentionDiagnostics.polarWHTValuePayloadAllocated)
            #expect(allClose(metal.output, reference.output, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        }

        @Test func testMetalHybridK8PolarWHTLongRotatingNoWrapMatchesReferenceHybrid() throws {
            guard TurboQuantKernelAvailability.current.attentionCapabilities
                .hybridK8PolarWHTValueAttention
            else {
                return
            }
            let precisionPolicy = TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                boundary: .disabled
            )
            func makeCache(backend: TurboQuantBackend) -> RotatingTurboQuantKVCache {
                RotatingTurboQuantKVCache(
                    maxSize: 4113,
                    keep: 4,
                    preset: .turbo8,
                    backend: backend,
                    kvCodec: .polarWHT,
                    valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    precisionPolicy: precisionPolicy
                )
            }
            let metalCache = makeCache(backend: .metalPolarWHT)
            let referenceCache = makeCache(backend: .polarWHTReference)
            let contextLength = 4096
            let chunkSize = 512
            let scale = Float(1.0 / sqrt(128.0))

            for chunkStart in stride(from: 0, to: contextLength, by: chunkSize) {
                let queryOffset = chunkStart * 16 * 128
                let keyOffset = chunkStart * 8 * 128
                let prefillQueries = patternedArray(
                    shape: [1, chunkSize, 16, 128],
                    modulus: 31,
                    centeredAt: 15,
                    denominator: 31
                ) + MLXArray(Float(queryOffset % 17) / 257)
                let prefillKeys = patternedArray(
                    shape: [1, chunkSize, 8, 128],
                    modulus: 37,
                    centeredAt: 18,
                    denominator: 37
                ) + MLXArray(Float(keyOffset % 19) / 263)
                let prefillValues = patternedArray(
                    shape: [1, chunkSize, 8, 128],
                    modulus: 41,
                    centeredAt: 20,
                    denominator: 41
                ) + MLXArray(Float(keyOffset % 23) / 269)

                let metalPrefill = try attentionWithCacheUpdateReturningStateThrowing(
                    queries: prefillQueries.transposed(0, 2, 1, 3).asType(.bfloat16),
                    keys: prefillKeys.transposed(0, 2, 1, 3).asType(.bfloat16),
                    values: prefillValues.transposed(0, 2, 1, 3).asType(.bfloat16),
                    cache: metalCache,
                    scale: scale,
                    mask: .causal
                )
                let referencePrefill = try attentionWithCacheUpdateReturningStateThrowing(
                    queries: prefillQueries.transposed(0, 2, 1, 3).asType(.bfloat16),
                    keys: prefillKeys.transposed(0, 2, 1, 3).asType(.bfloat16),
                    values: prefillValues.transposed(0, 2, 1, 3).asType(.bfloat16),
                    cache: referenceCache,
                    scale: scale,
                    mask: .causal
                )
                eval(metalPrefill.output, referencePrefill.output)
            }

            let decodeQueries = patternedArray(
                shape: [1, 16, 1, 128],
                modulus: 29,
                centeredAt: 14,
                denominator: 29
            )
            let decodeKeys = patternedArray(
                shape: [1, 8, 1, 128],
                modulus: 43,
                centeredAt: 21,
                denominator: 43
            )
            let decodeValues = patternedArray(
                shape: [1, 8, 1, 128],
                modulus: 47,
                centeredAt: 23,
                denominator: 47
            )

            let metal = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries.asType(.bfloat16),
                keys: decodeKeys.asType(.bfloat16),
                values: decodeValues.asType(.bfloat16),
                cache: metalCache,
                scale: scale,
                mask: .causal
            )
            let reference = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries.asType(.bfloat16),
                keys: decodeKeys.asType(.bfloat16),
                values: decodeValues.asType(.bfloat16),
                cache: referenceCache,
                scale: scale,
                mask: .causal
            )

            eval(metal.output, reference.output)
            #expect(metalCache.runtimeSnapshot().logicalLength == 4097)
            #expect(metalCache.runtimeSnapshot().ringOffset == 0)
            #expect(metalCache.attentionDiagnostics.activeAttentionPath == .metalHybridK8PolarWHTValue)
            #expect(referenceCache.attentionDiagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
            #expect(allClose(metal.output, reference.output, rtol: 1e-3, atol: 1e-3).item(Bool.self))
        }

        @Test func testPolarWHTReferenceBackendUsesHybridAttention() throws {
            try Device.withDefaultDevice(.cpu) {
                let cache = TurboQuantKVCache(
                    preset: .turbo4v2,
                    backend: .polarWHTReference
                )
                let keys = MLXArray((0 ..< 128).map { Float(($0 % 17) - 8) / 17 }, [1, 1, 2, 64])
                let values = MLXArray((0 ..< 128).map { Float(($0 % 23) - 11) / 23 }, [1, 1, 2, 64])
                let queries = MLXArray((0 ..< 64).map { Float(($0 % 13) - 6) / 13 }, [1, 1, 1, 64])
                let scale = Float(0.125)

                let quantized = cache.updateQuantized(keys: keys, values: values)
                let output = try attentionWithKVStateThrowing(
                    queries: queries,
                    state: .quantized(keys: quantized.0, values: quantized.1, cache: cache),
                    scale: scale
                )
                let decodedKeys = dequantized(
                    quantized.0.0,
                    scales: quantized.0.1,
                    biases: quantized.0.2,
                    groupSize: cache.groupSize,
                    bits: cache.bits,
                    mode: cache.mode,
                    dtype: .float32
                )
                let decodedPolarWHTKeys = try turboQuantPolarWHTReferenceDecodeAttentionValues(
                    try #require(cache.polarWHTKeyState)
                )
                .asType(.float32)
                let decodedValues = try turboQuantPolarWHTReferenceDecodeAttentionValues(
                    try #require(cache.polarWHTValueState)
                )
                .asType(.float32)
                let reason = try #require(cache.fallbackResults.last?.reason)
                let expectedKeys =
                    reason.contains("Metal PolarWHT QK scoring")
                    ? decodedPolarWHTKeys
                    : decodedKeys
                let expected = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: expectedKeys,
                    values: decodedValues,
                    scale: scale,
                    mask: .none
                )

                #expect(cache.activeBackend == .polarWHTReference)
                #expect(allClose(output, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
                #expect(cache.attentionDiagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
                #expect(cache.fallbackResults.last?.toPath == .polarWHTReferenceHybrid)
                #expect(reason.contains("WHT-pulled"))
                #expect(
                    reason.contains("Metal PolarWHT QK scoring")
                        || reason.contains("affine K scoring")
                )
                #expect(cache.attentionDiagnostics.polarWHTKeyPayloadAllocated)
                #expect(cache.attentionDiagnostics.polarWHTValuePayloadAllocated)
            }
        }

        @Test func testPolarWHTReferenceHybridCanForceAffineKScoring() throws {
            try Device.withDefaultDevice(.cpu) {
                let cache = TurboQuantKVCache(
                    preset: .turbo8,
                    backend: .polarWHTReference,
                    valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                    precisionPolicy: TurboQuantKVPrecisionPolicy(
                        key: .affineQ8,
                        value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
                        boundary: .disabled
                    )
                )
                let keys = MLXArray((0 ..< 128).map { Float(($0 % 17) - 8) / 17 }, [1, 1, 2, 64])
                let values = MLXArray((0 ..< 128).map { Float(($0 % 23) - 11) / 23 }, [1, 1, 2, 64])
                let queries = MLXArray((0 ..< 64).map { Float(($0 % 13) - 6) / 13 }, [1, 1, 1, 64])
                let scale = Float(0.125)

                let quantized = cache.updateQuantized(keys: keys, values: values)
                let output = try attentionWithKVStateThrowing(
                    queries: queries,
                    state: .quantized(keys: quantized.0, values: quantized.1, cache: cache),
                    scale: scale
                )
                let decodedKeys = dequantized(
                    quantized.0.0,
                    scales: quantized.0.1,
                    biases: quantized.0.2,
                    groupSize: cache.groupSize,
                    bits: cache.bits,
                    mode: cache.mode,
                    dtype: .float32
                )
                let decodedValues = try turboQuantPolarWHTReferenceDecodeAttentionValues(
                    try #require(cache.polarWHTValueState)
                )
                .asType(.float32)
                let expected = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: decodedKeys,
                    values: decodedValues,
                    scale: scale,
                    mask: .none
                )
                let reason = try #require(cache.fallbackResults.last?.reason)

                #expect(cache.bits == TurboQuantPreset.turbo8.effectiveBits)
                #expect(allClose(output, expected, rtol: 1e-4, atol: 1e-4).item(Bool.self))
                #expect(cache.attentionDiagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
                #expect(reason.contains("affine K scoring"))
                #expect(!reason.contains("Metal PolarWHT QK scoring"))
                #expect(!cache.attentionDiagnostics.polarWHTKeyPayloadAllocated)
                #expect(cache.attentionDiagnostics.polarWHTValuePayloadAllocated)
            }
        }

        @Test func testPolarWHTReferenceReportsSparseVInactiveReason() throws {
            try Device.withDefaultDevice(.cpu) {
                let cache = TurboQuantKVCache(
                    preset: .turbo4v2,
                    backend: .polarWHTReference,
                    resolvedRuntimeMode: .capacityTurboQuant,
                    sparseValueSelection: .threshold(1e-4)
                )
                let keys = MLXArray((0 ..< 128).map { Float(($0 % 17) - 8) / 17 }, [1, 1, 2, 64])
                let values = MLXArray((0 ..< 128).map { Float(($0 % 23) - 11) / 23 }, [1, 1, 2, 64])
                let queries = MLXArray((0 ..< 64).map { Float(($0 % 13) - 6) / 13 }, [1, 1, 1, 64])

                let quantized = cache.updateQuantized(keys: keys, values: values)
                _ = try attentionWithKVStateThrowing(
                    queries: queries,
                    state: .quantized(keys: quantized.0, values: quantized.1, cache: cache),
                    scale: 0.125
                )
                let diagnostics = cache.attentionDiagnostics

                #expect(diagnostics.activeAttentionPath == .polarWHTReferenceHybrid)
                #expect(diagnostics.sparseVEnabled)
                #expect(diagnostics.sparseVActive == false)
                #expect(
                    diagnostics.fallbackReason?
                        .contains("Sparse-V is not implemented for PolarWHT reference hybrid")
                        == true
                )
            }
        }

        @Test func testPolarWHTRotatingPackedFallbackMaintainsValueSidecar() throws {
            try Device.withDefaultDevice(.cpu) {
                let cache = RotatingTurboQuantKVCache(
                    maxSize: 4,
                    keep: 1,
                    preset: .turbo4v2,
                    backend: .metalPolarWHT
                )
                let keys = MLXArray.ones([1, 1, 4, 64], dtype: .float32)
                let values = MLXArray((0 ..< 256).map { Float($0 + 1) / 257 }, [1, 1, 4, 64])

                _ = cache.updateQuantized(keys: keys, values: values)

                let initial = cache.runtimeSnapshot()
                let polarWHTAvailable = TurboQuantKernelAvailability.current.supports(.metalPolarWHT)
                #expect(cache.kvCodec == .polarWHT)
                #expect(cache.activeBackend == (polarWHTAvailable ? .metalPolarWHT : .mlxPacked))
                #expect(initial.logicalLength == 4)
                #expect(initial.capacity == 4)
                #expect(initial.pinnedPrefixLength == 1)
                #expect(initial.ringOffset == 0)
                #expect(initial.polarWHTKeyPayloadAllocated)
                #expect(initial.polarWHTKeyBytes > 0)
                #expect(initial.polarWHTValuePayloadAllocated)
                #expect(initial.polarWHTValueBytes > 0)
                #expect(cache.diagnostics.polarWHTKeyBytes == initial.polarWHTKeyBytes)
                #expect(cache.attentionDiagnostics.polarWHTKeyBytes == initial.polarWHTKeyBytes)
                #expect(cache.attentionDiagnostics.polarWHTKeyPayloadAllocated)
                #expect(cache.diagnostics.polarWHTValueBytes == initial.polarWHTValueBytes)
                #expect(cache.attentionDiagnostics.polarWHTValueBytes == initial.polarWHTValueBytes)
                #expect(cache.attentionDiagnostics.polarWHTValuePayloadAllocated)
                #expect(
                    cache.cacheFootprint.compressedBytes
                        >= initial.polarWHTKeyBytes + initial.polarWHTValueBytes
                )

                let nextKeys = MLXArray.ones([1, 1, 1, 64], dtype: .float32) * 2
                let nextValues = MLXArray.ones([1, 1, 1, 64], dtype: .float32) * 0.5
                _ = cache.updateQuantized(keys: nextKeys, values: nextValues)

                let wrapped = cache.runtimeSnapshot()
                #expect(wrapped.logicalLength == 4)
                #expect(wrapped.ringOffset == 1)
                #expect(wrapped.polarWHTKeyPayloadAllocated)
                #expect(wrapped.polarWHTKeyBytes == initial.polarWHTKeyBytes)
                #expect(wrapped.polarWHTValuePayloadAllocated)
                #expect(wrapped.polarWHTValueBytes == initial.polarWHTValueBytes)
                let expectedWrappedKeys = MLXArray(
                    [Float](repeating: 1, count: 64)
                        + [Float](repeating: 1, count: 128)
                        + [Float](repeating: 2, count: 64),
                    [1, 1, 4, 64]
                )
                let valueData = (0 ..< 256).map { Float($0 + 1) / 257 }
                let expectedWrappedValues = MLXArray(
                    Array(valueData[0 ..< 64])
                        + Array(valueData[128 ..< 256])
                        + [Float](repeating: 0.5, count: 64),
                    [1, 1, 4, 64]
                )
                try expectPolarWHTValueState(
                    try #require(cache.polarWHTKeyState),
                    matches: expectedWrappedKeys
                )
                try expectPolarWHTValueState(
                    try #require(cache.polarWHTValueState),
                    matches: expectedWrappedValues
                )

                let copied = try #require(cache.copy() as? RotatingTurboQuantKVCache)
                #expect(copied.runtimeSnapshot().polarWHTKeyBytes == wrapped.polarWHTKeyBytes)
                #expect(copied.diagnostics.polarWHTKeyPayloadAllocated)
                #expect(copied.runtimeSnapshot().polarWHTValueBytes == wrapped.polarWHTValueBytes)
                #expect(copied.diagnostics.polarWHTValuePayloadAllocated)
                try expectPolarWHTValueState(
                    try #require(copied.polarWHTKeyState),
                    matches: expectedWrappedKeys
                )
                try expectPolarWHTValueState(
                    try #require(copied.polarWHTValueState),
                    matches: expectedWrappedValues
                )
            }
        }

        @Test func testValidationReportsDuplicateAndOutOfRangeLayers() {
            let policy = KVLayerPolicy(
                defaultCodec: .rawFP16,
                rules: [
                    KVLayerRule(layerIndex: 2, codec: .affineK8V4),
                    KVLayerRule(layerIndex: 2, codec: .rawFP16),
                    KVLayerRule(layerIndex: 9, codec: .affineInt4),
                ]
            )

            let errors = policy.validationErrors(layerCount: 8)

            #expect(errors.contains { $0.contains("duplicate") })
            #expect(errors.contains { $0.contains("out of range") })
        }

        @Test func testValidationReportsUnsupportedAffineK8VxBits() {
            let policy = KVLayerPolicy(
                defaultCodec: .affineK8Vx(valueBits: 5),
                rules: [KVLayerRule(layerIndex: 0, codec: .affineK8Vx(valueBits: 1))]
            )

            let errors = policy.validationErrors(layerCount: 4)

            #expect(errors.count == 2)
            #expect(errors.allSatisfy { $0.contains("value bits must be 2, 3, or 4") })
        }

        @Test func testResidualAffineK8VxPolicyRoundTripAndValidation() throws {
            let policy = KVLayerPolicy(
                defaultCodec: .affineK8VxResidual(valueBits: 2, residualsPerGroup: 1),
                rules: [KVLayerRule(layerIndex: 0, codec: .affineK8V4)]
            )
            let decoded = try JSONDecoder().decode(
                KVLayerPolicy.self,
                from: try JSONEncoder().encode(policy)
            )
            let invalid = KVLayerPolicy(
                defaultCodec: .affineK8VxResidual(valueBits: 3, residualsPerGroup: 2)
            )

            #expect(decoded == policy)
            #expect(decoded.stableHash == policy.stableHash)
            #expect(policy.summary().contains("affineK8V2ResidualR1"))
            #expect(invalid.validationErrors(layerCount: 4).count == 2)
        }

        @Test func testAffineK8VxProtectedEdgesCreatesLowerVMiddleAndBoundaryRules() {
            let k8v4Boundary = KVLayerPolicy.affineK8VxProtectedEdges(
                layerCount: 8,
                valueBits: 3
            )
            let rawBoundary = KVLayerPolicy.affineK8VxProtectedEdges(
                layerCount: 8,
                valueBits: 2,
                boundaryCachePrecision: .raw,
                first: 1,
                last: 1
            )

            #expect(k8v4Boundary.defaultCodec == .affineK8Vx(valueBits: 3))
            #expect(k8v4Boundary.codec(forLayerIndex: 0) == .affineK8V4)
            #expect(k8v4Boundary.codec(forLayerIndex: 1) == .affineK8V4)
            #expect(k8v4Boundary.codec(forLayerIndex: 3) == .affineK8Vx(valueBits: 3))
            #expect(k8v4Boundary.codec(forLayerIndex: 6) == .affineK8V4)
            #expect(k8v4Boundary.codec(forLayerIndex: 7) == .affineK8V4)
            #expect(rawBoundary.defaultCodec == .affineK8Vx(valueBits: 2))
            #expect(rawBoundary.codec(forLayerIndex: 0) == .rawFP16)
            #expect(rawBoundary.codec(forLayerIndex: 3) == .affineK8Vx(valueBits: 2))
            #expect(rawBoundary.codec(forLayerIndex: 7) == .rawFP16)
        }

        @Test func testCacheFactoryRoutesPerLayerCodecs() {
            let policy = KVLayerPolicy(
                defaultCodec: .mlxAffine(bits: 4, groupSize: 64),
                rules: [
                    KVLayerRule(layerIndex: 0, codec: .rawFP16),
                    KVLayerRule(layerIndex: 3, codec: .affineK8V4),
                    KVLayerRule(layerIndex: 4, codec: .affineK8Vx(valueBits: 3)),
                    KVLayerRule(
                        layerIndex: 6,
                        codec: .affineK8VxResidual(valueBits: 2, residualsPerGroup: 1)
                    ),
                    KVLayerRule(
                        layerIndex: 2,
                        codec: .turboQuant(
                            preset: .turbo4v2,
                            valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
                            groupSize: 64,
                            backend: .polarWHTReference
                        )
                    ),
                    KVLayerRule(
                        layerIndex: 7,
                        codec: .turboQuant(
                            preset: .turbo4v2,
                            valueBits: 4,
                            groupSize: 64,
                            backend: .metalPolarQJL
                        )
                    ),
                ]
            )
            let parameters = GenerateParameters(
                maxKVSize: 128,
                quantizedKVStart: 0,
                kvCacheStrategy: .none,
                kvLayerPolicy: policy
            )

            let raw = makeAttentionKVCache(parameters: parameters, layerIndex: 0, layerCount: 8)
            let k8v4 = makeAttentionKVCache(parameters: parameters, layerIndex: 3, layerCount: 8)
            let k8v3 = makeAttentionKVCache(parameters: parameters, layerIndex: 4, layerCount: 8)
            let k8v2Residual = makeAttentionKVCache(
                parameters: parameters,
                layerIndex: 6,
                layerCount: 8
            )
            let polarWHTReference = makeAttentionKVCache(
                parameters: parameters,
                layerIndex: 2,
                layerCount: 8
            )
            let turbo = makeAttentionKVCache(parameters: parameters, layerIndex: 7, layerCount: 8)
            let inheritedDefault = makeAttentionKVCache(
                parameters: parameters,
                layerIndex: 5,
                layerCount: 8
            )

            #expect(raw is RotatingKVCache)
            #expect(k8v4 is AffineK8V4KVCache)
            #expect((k8v3 as? AffineK8V4KVCache)?.valueBits == 3)
            #expect((k8v2Residual as? AffineK8V4KVCache)?.valueBits == 2)
            #expect((k8v2Residual as? AffineK8V4KVCache)?.residualsPerGroup == 1)
            #expect((polarWHTReference as? RotatingTurboQuantKVCache)?.kvCodec == .polarWHT)
            #expect(
                (polarWHTReference as? RotatingTurboQuantKVCache)?.requestedBackend
                    == .polarWHTReference
            )
            #expect(
                (polarWHTReference as? RotatingTurboQuantKVCache)?.activeBackend
                    == .polarWHTReference
            )
            #expect(turbo is RotatingTurboQuantKVCache)
            #expect(inheritedDefault is RotatingQuantizedKVCache)
        }

        @Test func testResidualAffineK8VxCacheStateCopyAndTrim() throws {
            Device.withDefaultDevice(.cpu) {
                let cache = AffineK8V4KVCache(valueBits: 2, residualsPerGroup: 1)
                let keys = MLXArray.ones([1, 1, 4, 64], dtype: .float32)
                let values = MLXArray((0 ..< 256).map { Float($0) / 256 }, [1, 1, 4, 64])
                _ = cache.updateQuantized(keys: keys, values: values)

                let residualState = cache.getValueResidualState()
                #expect(residualState?.laneIndices.shape == [1, 1, 4, 2])
                #expect(residualState?.laneIndices.dtype == .uint8)
                #expect(residualState?.values.shape == [1, 1, 4, 2])
                #expect(cache.state.count == 8)
                #expect(cache.metaState.last == "1")

                let copied = cache.copy() as? AffineK8V4KVCache
                #expect(copied?.residualsPerGroup == 1)
                #expect(copied?.getValueResidualState()?.values.shape == [1, 1, 4, 2])

                #expect(cache.trim(2) == 2)
                #expect(cache.getValueResidualState()?.values.shape == [1, 1, 2, 2])
            }
        }

        @Test func testResidualAffineK8VxAttentionRouteRecordsResidualPath() throws {
            try Device.withDefaultDevice(.cpu) {
                setenv("MLX_TURBOQUANT_DISABLE_K8V4_NATIVE", "1", 1)
                turboQuantResetAffineK8V4NativeKillSwitchForTesting()
                defer {
                    unsetenv("MLX_TURBOQUANT_DISABLE_K8V4_NATIVE")
                    turboQuantResetAffineK8V4NativeKillSwitchForTesting()
                }
                let cache = AffineK8V4KVCache(valueBits: 2, residualsPerGroup: 1)
                let keys = MLXArray.ones([1, 1, 4, 64], dtype: .float32) * 0.1
                let values = MLXArray((0 ..< 256).map { Float($0) / 128 }, [1, 1, 4, 64])
                let queries = MLXArray.ones([1, 1, 1, 64], dtype: .float32) * 0.05
                let state = cache.updateQuantized(keys: keys, values: values)

                let output = try attentionWithKVStateThrowing(
                    queries: queries,
                    state: .quantized(keys: state.0, values: state.1, cache: cache),
                    scale: 0.125,
                    mask: .causal
                )

                eval(output)
                #expect(output.shape == [1, 1, 1, 64])
                #expect(cache.activeAttentionPath == .affineK8VxResidual)
                #expect(cache.attentionFailureReason?.contains("residual correction") == true)
            }
        }

        @Test func testInferenceParityRecognizesLowerV2ConfigLabels() throws {
            let configs = try #require(InferenceParityBenchmark.configs(
                fromCSV:
                    "affineK8V2-calibrated,affineK8V2-residual-r1,affineK8V2-calibrated-residual-r1,affineK8V2-protectedK8V4-edge8,affineK8V2-protectedRaw-edge4"
            ))

            #expect(configs.map(\.label) == [
                "affineK8V2-calibrated",
                "affineK8V2-residual-r1",
                "affineK8V2-calibrated-residual-r1",
                "affineK8V2-protectedK8V4-edge8",
                "affineK8V2-protectedRaw-edge4",
            ])
            #expect(configs[1].kvLayerPolicy?.defaultCodec == .affineK8VxResidual(
                valueBits: 2,
                residualsPerGroup: 1
            ))
        }

        @Test func testInferenceParityRecognizesPolarWHTConfigLabel() throws {
            let configs = try #require(
                InferenceParityBenchmark.configs(
                    fromCSV:
                        "polarWHTV3,polarWHTReferenceV3,wht_reference_v3,hybrid_k8_polar_wht_v3,hybrid_k8_polar_wht_v4,hybrid_k8_polar_wht_v3_reference"
                )
            )

            #expect(configs.count == 6)
            #expect(configs[0].label == "polarWHTV3")
            #expect(configs[0].kvCodec == .polarWHT)
            #expect(configs[0].turboQuantBackend == .metalPolarWHT)
            #expect(configs[0].valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)

            #expect(configs[1].label == "polarWHTReferenceV3")
            #expect(configs[1].kvCodec == .polarWHT)
            #expect(configs[1].turboQuantBackend == .polarWHTReference)
            #expect(configs[1].valueBits == TurboQuantKVCodec.polarWHTDefaultValueBits)

            #expect(configs[2].label == "polarWHTReferenceV3")
            #expect(configs[2].turboQuantBackend == .polarWHTReference)

            #expect(configs[3].label == "hybridK8PolarWHTV3")
            #expect(configs[3].kvCodec == .polarWHT)
            #expect(configs[3].turboQuantBackend == .metalPolarWHT)
            #expect(configs[3].strategy == .adaptiveTurboQuant)
            #expect(configs[3].exactPrefill)
            #expect(configs[3].precisionPolicy?.key == .affineQ8)
            #expect(configs[3].precisionPolicy?.value == .turbo3_5)

            #expect(configs[4].label == "hybridK8PolarWHTV4")
            #expect(configs[4].kvCodec == .polarWHT)
            #expect(configs[4].turboQuantBackend == .metalPolarWHT)
            #expect(configs[4].strategy == .adaptiveTurboQuant)
            #expect(configs[4].exactPrefill)
            #expect(configs[4].precisionPolicy?.key == .affineQ8)
            #expect(configs[4].precisionPolicy?.value == .turbo4v2)

            #expect(configs[5].label == "hybridK8PolarWHTV3Reference")
            #expect(configs[5].kvCodec == .polarWHT)
            #expect(configs[5].turboQuantBackend == .polarWHTReference)
            #expect(configs[5].strategy == .adaptiveTurboQuant)
            #expect(configs[5].exactPrefill)
            #expect(configs[5].precisionPolicy?.key == .affineQ8)
            #expect(configs[5].precisionPolicy?.value == .turbo3_5)
        }

        @Test func testDynamicQuantizationUsesSelectedLayerCodecAndSkipsRaw() throws {
            Device.withDefaultDevice(.cpu) {
                var cache: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
                let keys = MLXArray.ones([1, 1, 4, 64], dtype: .float32)
                let values = MLXArray.ones([1, 1, 4, 64], dtype: .float32)
                _ = cache[0].update(keys: keys, values: values)
                _ = cache[1].update(keys: keys, values: values)
                let policy = KVLayerPolicy(
                    rules: [
                        KVLayerRule(layerIndex: 0, codec: .rawFP16),
                        KVLayerRule(layerIndex: 1, codec: .affineK8V4),
                    ]
                )

                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: nil,
                    quantizedKVStart: 0,
                    kvCacheStrategy: .none,
                    kvLayerPolicy: policy
                )

                #expect(cache[0] is KVCacheSimple)
                #expect(cache[1] is AffineK8V4KVCache)
            }
        }

        @Test func testQwen35UsesModelLayerIndexesAndLeavesLinearLayersRecurrent() throws {
            let config = try Self.qwen35TextConfiguration(hiddenLayers: 4, fullAttentionInterval: 2)
            let model = Qwen35TextModel(config)
            let policy = KVLayerPolicy(
                defaultCodec: .rawFP16,
                rules: [
                    KVLayerRule(layerIndex: 1, codec: .affineK8V4),
                    KVLayerRule(
                        layerIndex: 3,
                        codec: .turboQuant(
                            preset: .turbo4v2,
                            valueBits: 4,
                            groupSize: 64,
                            backend: .metalPolarQJL
                        )
                    ),
                ]
            )
            let caches = model.newCache(
                parameters: GenerateParameters(
                    maxKVSize: nil,
                    quantizedKVStart: 0,
                    kvCacheStrategy: .none,
                    kvLayerPolicy: policy
                )
            )

            #expect(caches.count == 4)
            #expect(caches[0] is MambaCache)
            #expect(caches[1] is AffineK8V4KVCache)
            #expect(caches[2] is MambaCache)
            #expect(caches[3] is TurboQuantKVCache)
        }

        @Test func testProfileCarriesKVLayerPolicyIntoGenerateParametersAndValidation() {
            let policy = KVLayerPolicy(
                defaultCodec: .turboQuant(
                    preset: .turbo4v2,
                    valueBits: 4,
                    groupSize: 64,
                    backend: .metalPolarQJL
                ),
                rules: [KVLayerRule(layerIndex: 3, codec: .affineK8V4)]
            )
            let profile = TurboQuantProfile(
                id: "policy-test",
                modelPatterns: ["policy-test"],
                supportedKeyHeadDimensions: [128],
                modelFingerprint: TurboQuantModelFingerprint(layerCount: 8),
                turboQuant: TurboQuantProfileTurboQuantManifest(
                    keyPreset: .turbo4v2,
                    valueBits: 4,
                    groupSize: 64,
                    kvLayerPolicy: policy
                )
            )

            let parameters = GenerateParameters(turboQuantProfile: profile)

            #expect(parameters.kvLayerPolicy == policy)
            #expect(
                !profile.productManifestValidation(requireMeasuredOutcomes: false).issues
                    .contains { $0.field == "turbo_quant.kv_layer_policy" }
            )
        }

        private static func qwen35TextConfiguration(
            hiddenLayers: Int,
            fullAttentionInterval: Int
        ) throws -> Qwen35TextConfiguration {
            let json = """
                {
                  "model_type": "qwen3_5",
                  "hidden_size": 64,
                  "num_hidden_layers": \(hiddenLayers),
                  "intermediate_size": 128,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "linear_num_value_heads": 64,
                  "linear_num_key_heads": 16,
                  "linear_key_head_dim": 192,
                  "linear_value_head_dim": 128,
                  "linear_conv_kernel_dim": 4,
                  "rms_norm_eps": 1e-6,
                  "vocab_size": 128,
                  "rope_theta": 10000.0,
                  "max_position_embeddings": 128,
                  "full_attention_interval": \(fullAttentionInterval),
                  "num_nextn_predict_layers": 0,
                  "tie_word_embeddings": true
                }
                """
            return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
        }
    }
}
