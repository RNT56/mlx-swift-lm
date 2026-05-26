import Foundation
import MLX
import Testing

@testable import MLXLMCommon

private let cacheCreators: [@Sendable () -> any KVCache] = [
    { KVCacheSimple() },
    { RotatingKVCache(maxSize: 32) },
    { QuantizedKVCache() },
    { TurboQuantKVCache() },
    { RotatingTurboQuantKVCache(maxSize: 32) },
    { ChunkedKVCache(chunkSize: 16) },
    { ArraysCache(size: 2) },
    { MambaCache() },
]

// MARK: - Helper

private func tempURL() -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")
}

/// Assert two arrays of MLXArray are element-wise close
private func assertArraysClose(_ lhs: [MLXArray], _ rhs: [MLXArray], label: String = "") {
    #expect(lhs.count == rhs.count, "state count mismatch \(label)")
    for (i, (a, b)) in zip(lhs, rhs).enumerated() {
        #expect(a.shape == b.shape, "shape mismatch at index \(i) \(label)")
        let close = allClose(a, b).item(Bool.self)
        #expect(close, "values not close at index \(i) \(label)")
    }
}

extension MLXRuntimeSwiftTests {

    @Suite
    struct KVCacheTests {

        // MARK: - Original parameterized test (updated with value assertions)

        @Test(
            .serialized,
            arguments: cacheCreators)
        func testCacheSerialization(creator: (() -> any KVCache)) async throws {
            let cache = (0 ..< 10).map { _ in creator() }
            let keys = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
            let values = MLXArray.ones([1, 8, 32, 64], dtype: .bfloat16)
            for item in cache {
                switch item {
                case let arrays as ArraysCache:
                    arrays[0] = keys
                    arrays[1] = values
                case let turbo as TurboQuantKVCache:
                    _ = turbo.updateQuantized(keys: keys, values: values)
                case let turbo as RotatingTurboQuantKVCache:
                    _ = turbo.updateQuantized(keys: keys, values: values)
                case let quantized as QuantizedKVCache:
                    _ = quantized.updateQuantized(keys: keys, values: values)
                default:
                    _ = item.update(keys: keys, values: values)
                }
            }

            let url = tempURL()

            try savePromptCache(url: url, cache: cache, metadata: [:])
            let (loadedCache, _) = try loadPromptCache(url: url)

            #expect(cache.count == loadedCache.count)
            for (lhs, rhs) in zip(cache, loadedCache) {
                #expect(type(of: lhs) == type(of: rhs))
                #expect(lhs.metaState == rhs.metaState)
                assertArraysClose(lhs.state, rhs.state)
            }
        }

        // MARK: - ArraysCache sparse slot round-trip

        @Test func testArraysCacheSparseSlots() throws {
            let cache = ArraysCache(size: 3)
            let a = MLXArray.ones([2, 4], dtype: .float32) * 3.0
            let b = MLXArray.ones([2, 4], dtype: .float32) * 7.0
            cache[0] = a
            // slot 1 stays nil
            cache[2] = b

            let url = tempURL()
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)

            #expect(loaded.count == 1)
            let restored = try #require(loaded[0] as? ArraysCache)
            #expect(restored.slotCount == 3)
            #expect(restored[0] != nil)
            #expect(restored[1] == nil)
            #expect(restored[2] != nil)
            #expect(allClose(restored[0]!, a).item(Bool.self))
            #expect(allClose(restored[2]!, b).item(Bool.self))
        }

        // MARK: - ArraysCache leftPadding round-trip

        @Test func testArraysCacheLeftPadding() throws {
            let cache = ArraysCache(size: 2, leftPadding: [0, 5])
            let a = MLXArray.ones([2, 4], dtype: .float32)
            let b = MLXArray.ones([2, 4], dtype: .float32) * 2.0
            cache[0] = a
            cache[1] = b

            let url = tempURL()
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)

            let restored = try #require(loaded[0] as? ArraysCache)
            #expect(restored.leftPaddingValues == [0, 5])
            assertArraysClose(restored.state, cache.state)
        }

        @Test func testMaterializeRecurrentKVCacheStateWalksCacheLists() throws {
            let recurrent = MambaCache()
            let source = MLXArray(0 ..< 12, [1, 6, 2])
            let convState = source[0..., 3..., 0...] + 1
            let nativeState = source[0..., ..<3, 0...] * 2
            recurrent[0] = convState
            recurrent[1] = nativeState

            let attention = KVCacheSimple()
            let keys = MLXArray.ones([1, 1, 2, 4], dtype: .float32)
            let values = MLXArray.ones([1, 1, 2, 4], dtype: .float32)
            _ = attention.update(keys: keys, values: values)

            #expect(materializeRecurrentKVCacheState([CacheList(recurrent, attention)]))

            let state = recurrent.state
            #expect(state.count == 2)
            #expect(allClose(state[0], convState).item(Bool.self))
            #expect(allClose(state[1], nativeState).item(Bool.self))
            #expect(attention.state.count == 2)
            #expect(!materializeRecurrentKVCacheState([attention]))
        }

        @Test func testTurboQuantCacheStrategyConvertsSimpleCache() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            try Device.withDefaultDevice(.cpu) {
                var cache: [KVCache] = [KVCacheSimple()]
                let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
                let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
                _ = cache[0].update(keys: keys, values: values)

                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: nil,
                    kvGroupSize: 64,
                    quantizedKVStart: 0,
                    kvCacheStrategy: .turboQuant,
                    turboQuantPreset: .turbo3_5,
                    turboQuantBackend: .metalPolarQJL,
                    turboQuantOptimizationPolicy: .preferThroughput,
                    turboQuantSeed: 0xDEAD_BEEF_0000_0017
                )

                #expect(cache[0] is TurboQuantKVCache)
                let turbo = try #require(cache[0] as? TurboQuantKVCache)
                #expect(turbo.preset == .turbo3_5)
                #expect(turbo.requestedBackend == .metalPolarQJL)
                #expect(turbo.optimizationPolicy == .preferThroughput)
                #expect(turbo.seed == 0xDEAD_BEEF_0000_0017)
                #expect(turbo.diagnostics.optimizationPolicy == .preferThroughput)
                #expect(
                    turbo.diagnostics.selectedKernelProfile
                        == TurboQuantKernelAvailability.current.selectedKernelProfile)
                let expectedBackend: TurboQuantBackend =
                    TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention
                    ? .metalPolarQJL
                    : .mlxPacked
                #expect(turbo.activeBackend == expectedBackend)
                if expectedBackend == .mlxPacked {
                    #expect(turbo.backendFallbackReason != nil)
                } else {
                    #expect(turbo.state.count == 10)
                    let compressed = try #require(turbo.compressedState)
                    #expect(compressed.0.layout.logicalLength == 16)
                    #expect(compressed.1.layout.logicalLength == 16)
                }
            }
        }

        @Test func testTurboQuantDefaultsRequestVerifiedMetalBackend() {
            let availability = TurboQuantKernelAvailability.current
            let expectedBackend: TurboQuantBackend =
                availability.supportsMetalPolarQJLAttention ? .metalPolarQJL : .mlxPacked

            let simple = TurboQuantKVCache()
            #expect(simple.requestedBackend == .metalPolarQJL)
            #expect(simple.activeBackend == expectedBackend)

            let rotating = RotatingTurboQuantKVCache(maxSize: 32)
            #expect(rotating.requestedBackend == .metalPolarQJL)
            #expect(rotating.activeBackend == expectedBackend)
        }

        @Test func testDefaultTurboQuantCacheUsesCompressedAttentionWhenAvailable() {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache()
            let queries = MLXArray.ones([2, 4, 1, 64], dtype: .float32)
            let keys = MLXArray.ones([2, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([2, 2, 3, 64], dtype: .float32)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 0.125
            )

            #expect(output.shape == [2, 4, 1, 64])
            #expect(cache.activeBackend == .metalPolarQJL)
            #expect(cache.compressedState != nil)
            #expect(cache.attentionDiagnostics.activeAttentionPath == .tiledOnlineFused)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
        }

        @Test func testTurboQuantPrefillUsesRawAttentionOutputWhenAvailable() {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let qValues: [Float] = (0 ..< (1 * 4 * 3 * 256)).map { index in
                let position = Double(index)
                return Float(0.27 * sin(position * 0.013) + 0.09 * cos(position * 0.071))
            }
            let kValues: [Float] = (0 ..< (1 * 2 * 3 * 256)).map { index in
                let position = Double(index)
                return Float(0.22 * cos(position * 0.017) - 0.08 * sin(position * 0.057))
            }
            let vValues: [Float] = (0 ..< (1 * 2 * 3 * 256)).map { index in
                let position = Double(index)
                return Float(0.18 * sin(position * 0.023) + 0.12 * cos(position * 0.043))
            }
            let queries = MLXArray(qValues, [1, 4, 3, 256])
            let keys = MLXArray(kValues, [1, 2, 3, 256])
            let values = MLXArray(vValues, [1, 2, 3, 256])
            let cache = TurboQuantKVCache(
                preset: .turbo4v2,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xA17_0000_0000_0001,
                valueBits: 4
            )

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 1 / sqrt(Float(256)),
                mask: .causal
            )
            let reference = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: 1 / sqrt(Float(256)),
                mask: .causal
            )

            #expect(output.shape == [1, 4, 3, 256])
            #expect(allClose(output, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(cache.offset == 3)
            #expect(cache.compressedState != nil)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
        }

        @Test func testThrowingAttentionWithCacheUpdateMatchesNonThrowingRawPath() throws {
            let queries = MLXArray.ones([1, 1, 2, 8], dtype: .float32)
            let keys = MLXArray.ones([1, 1, 2, 8], dtype: .float32)
            let values = MLXArray.ones([1, 1, 2, 8], dtype: .float32) * 3.0

            let throwingCache = KVCacheSimple()
            let throwingResult = try attentionWithCacheUpdateReturningStateThrowing(
                queries: queries,
                keys: keys,
                values: values,
                cache: throwingCache,
                scale: 1 / sqrt(Float(8)),
                mask: .causal
            )

            let nonThrowingCache = KVCacheSimple()
            let nonThrowingResult = attentionWithCacheUpdateReturningState(
                queries: queries,
                keys: keys,
                values: values,
                cache: nonThrowingCache,
                scale: 1 / sqrt(Float(8)),
                mask: .causal
            )

            #expect(throwingResult.output.shape == nonThrowingResult.output.shape)
            #expect(
                allClose(
                    throwingResult.output,
                    nonThrowingResult.output,
                    rtol: 1e-5,
                    atol: 1e-5
                ).item(Bool.self)
            )
            guard case .raw = throwingResult.state else {
                Issue.record("Expected raw attention state")
                return
            }
        }

        @Test func testTurboQuantPrefillRecordsLifecycleFallbackAndFootprint() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(
                preset: .turbo4v2,
                groupSize: 64,
                backend: .metalPolarQJL,
                seed: 0xA17_0000_0000_0002,
                valueBits: 4
            )
            let queries = MLXArray.ones([1, 4, 3, 64], dtype: .float32)
            let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)

            _ = attentionWithCacheUpdateReturningState(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 0.125,
                mask: .causal
            )

            let compressed = try #require(cache.compressedState)
            #expect(compressed.0.layout.logicalLength == 3)
            #expect(cache.fallbackResults.last?.toPath == .baseline)
            #expect(cache.fallbackResults.last?.policy == .exactRequired)
            #expect(cache.fallbackResults.last?.isSemanticallyExact == true)
            if case .compressedCommitted(let logicalLength, let capacity) = cache.cacheLifecycle {
                #expect(logicalLength == 3)
                #expect(capacity >= 3)
            } else {
                Issue.record("Expected compressed committed lifecycle")
            }
            #expect(cache.cacheFootprint.compressedBytes > 0)
            #expect(cache.cacheFootprint.rawShadowBytes == 0)
            #expect(
                cache.diagnostics.footprint?.compressedBytes == cache.cacheFootprint.compressedBytes
            )
            #expect(turboQuantAggregateCacheFootprint([cache]).compressedBytes > 0)
        }

        @Test func testTurboQuantDecodedFallbackReportsTransientFootprint() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            _ = try cache.updateCompressed(keys: keys, values: values)

            let decoded = try cache.decodedCompressedState(outputDType: .float32)
            #expect(
                cache.cacheFootprint.decodedTransientBytes == decoded.0.nbytes + decoded.1.nbytes)
            if case .decodeCompressed = cache.cacheLifecycle {
            } else {
                Issue.record("Expected decode-compressed lifecycle")
            }

            _ = try cache.updateCompressed(
                keys: MLXArray.ones([1, 2, 1, 64], dtype: .float32),
                values: MLXArray.ones([1, 2, 1, 64], dtype: .float32)
            )
            #expect(cache.cacheFootprint.decodedTransientBytes == 0)
        }

        @Test func testTurboQuantMemoryPolicyUsesCompressedInitialPrefillWhenAvailable() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let qValues: [Float] = (0 ..< (1 * 4 * 3 * 64)).map { index in
                let position = Double(index)
                return Float(0.24 * sin(position * 0.019) + 0.11 * cos(position * 0.061))
            }
            let kValues: [Float] = (0 ..< (1 * 2 * 3 * 64)).map { index in
                let position = Double(index)
                return Float(0.20 * cos(position * 0.029) - 0.07 * sin(position * 0.073))
            }
            let vValues: [Float] = (0 ..< (1 * 2 * 3 * 64)).map { index in
                let position = Double(index)
                return Float(0.17 * sin(position * 0.031) + 0.09 * cos(position * 0.047))
            }
            let queries = MLXArray(qValues, [1, 4, 3, 64])
            let keys = MLXArray(kValues, [1, 2, 3, 64])
            let values = MLXArray(vValues, [1, 2, 3, 64])
            let cache = TurboQuantKVCache(
                backend: .metalPolarQJL,
                optimizationPolicy: .preferMemory
            )

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 0.125,
                mask: .causal
            )
            let compressed = try #require(cache.compressedState)
            let directCompressed = try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: compressed.0,
                valueCode: compressed.1,
                scale: 0.125,
                mask: .causal,
                preferOnlineFused: cache.prefersOnlineFusedAttention,
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
            )

            #expect(output.shape == [1, 4, 3, 64])
            #expect(cache.offset == 3)
            #expect(cache.prefersExactInitialPrefill == false)
            #expect(allClose(output, directCompressed, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }

        @Test func testMakeAttentionKVCacheHonorsTurboQuantParameters() throws {
            let parameters = GenerateParameters(
                maxKVSize: 48,
                kvCacheStrategy: .turboQuant,
                turboQuantPreset: .turbo2_5,
                turboQuantBackend: .metalPolarQJL,
                turboQuantOptimizationPolicy: .conservative,
                turboQuantSeed: 0x1234_5678_9ABC_DEF0,
                turboQuantValueBits: 4
            )

            let cache = try #require(
                makeAttentionKVCache(parameters: parameters, maxKVSize: 16, keep: 2)
                    as? RotatingTurboQuantKVCache)

            #expect(cache.maxSize == 16)
            #expect(cache.metaState.first == "2")
            #expect(cache.preset == .turbo2_5)
            #expect(cache.requestedBackend == .metalPolarQJL)
            #expect(cache.optimizationPolicy == .conservative)
            #expect(cache.seed == 0x1234_5678_9ABC_DEF0)
            #expect(cache.valueBits == 4)
        }

        @Test func testRawOnlyAttentionCacheIgnoresTurboQuantConversion() {
            let parameters = GenerateParameters(
                maxKVSize: 16,
                kvCacheStrategy: .turboQuant,
                turboQuantPreset: .turbo3_5
            )
            var cache: [KVCache] = [
                makeRawAttentionKVCache(
                    parameters: GenerateParameters(kvCacheStrategy: .turboQuant)),
                makeRawAttentionKVCache(parameters: parameters, maxKVSize: 8, keep: 0),
            ]
            let keys = MLXArray.ones([1, 2, 4, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 4, 64], dtype: .float32)
            _ = cache[0].update(keys: keys, values: values)
            _ = cache[1].update(keys: keys, values: values)

            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: nil,
                kvGroupSize: 64,
                quantizedKVStart: 0,
                kvCacheStrategy: .turboQuant,
                turboQuantPreset: .turbo3_5,
                turboQuantBackend: .metalPolarQJL
            )

            #expect(cache[0] is RawOnlyKVCacheSimple)
            #expect(cache[1] is RawOnlyRotatingKVCache)
            #expect(!(cache[0] is QuantizedKVCacheProtocol))
            #expect(!(cache[1] is QuantizedKVCacheProtocol))
        }

        @Test func testTurboQuantCompressedCacheUpdateRollsBackOnFailure() {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            enum TestFailure: Error {
                case forced
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let queries = MLXArray.ones([1, 2, 1, 64], dtype: .float32)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 64], dtype: .float32)

            let output: MLXArray? = withTurboQuantCompressedCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                mask: .none
            ) { _, _, _ in
                throw TestFailure.forced
            }

            #expect(output == nil)
            #expect(cache.offset == 0)
            #expect(cache.state.isEmpty)
            #expect(cache.compressedState == nil)
            #expect(
                cache.attentionDiagnostics.lastUnsupportedShape?
                    .contains("compressed attention failed") == true)
        }

        @Test func testThrowingTurboQuantCompressedCacheUpdateRollsBackOnFailure() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            enum TestFailure: Error {
                case forced
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let queries = MLXArray.ones([1, 2, 1, 64], dtype: .float32)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 64], dtype: .float32)

            do {
                _ = try withTurboQuantCompressedCacheUpdateThrowing(
                    queries: queries,
                    keys: keys,
                    values: values,
                    cache: cache,
                    mask: .none
                ) { _, _, _ in
                    throw TestFailure.forced
                }
                Issue.record("Expected forced compressed update failure")
            } catch {
                #expect(String(describing: error).contains("forced"))
            }

            #expect(cache.offset == 0)
            #expect(cache.state.isEmpty)
            #expect(cache.compressedState == nil)
            if case .failed = cache.cacheLifecycle {
            } else {
                Issue.record("Expected failed lifecycle after forced compressed update failure")
            }
        }

        @Test func testTurboQuantOptimizationPolicyPropagatesToPromptCaches() throws {
            let parameters = GenerateParameters(
                maxKVSize: 32,
                kvCacheStrategy: .turboQuant,
                turboQuantBackend: .metalPolarQJL,
                turboQuantOptimizationPolicy: .conservative
            )
            let cache = makePromptCacheWithLayerCount(numLayers: 2, parameters: parameters)

            #expect(cache.count == 2)
            let rotating = try #require(cache[0] as? RotatingTurboQuantKVCache)
            #expect(rotating.optimizationPolicy == .conservative)
            #expect(rotating.attentionDiagnostics.optimizationPolicy == .conservative)
            #expect(
                rotating.diagnostics.selfTestStatus
                    == TurboQuantKernelAvailability.current.selfTestStatus)
        }

        @Test func testAttentionWithCacheUpdateSupportsSinksWithTurboQuantCache() {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }
            let expectsCompressedPath = TurboQuantKernelAvailability.current
                .supportsMetalPolarQJLAttention

            Device.withDefaultDevice(.cpu) {
                let cache = TurboQuantKVCache()
                let queries = MLXArray.ones([1, 2, 1, 64], dtype: .float32)
                let keys = MLXArray.ones([1, 2, 1, 64], dtype: .float32)
                let values = MLXArray.ones([1, 2, 1, 64], dtype: .float32)
                let sinks = MLXArray.zeros([2], dtype: .float32)

                let output = attentionWithCacheUpdate(
                    queries: queries,
                    keys: keys,
                    values: values,
                    cache: cache,
                    scale: 0.125,
                    sinks: sinks
                )

                #expect(output.shape == [1, 2, 1, 64])
                #expect(cache.offset == 1)
                if expectsCompressedPath {
                    #expect(
                        cache.attentionDiagnostics.activeAttentionPath == .twoStageCompressed
                            || cache.attentionDiagnostics.activeAttentionPath == .tiledOnlineFused)
                    #expect(cache.compressedState != nil)
                } else {
                    #expect(cache.attentionDiagnostics.activeAttentionPath == .mlxPackedFallback)
                }
            }
        }

        @Test func testTurboQuantPackedFallbackHonorsBooleanMasks() {
            let queries = MLXArray.zeros([1, 1, 2, 64], dtype: .float32)
            let keys = MLXArray.zeros([1, 1, 2, 64], dtype: .float32)
            let values = MLXArray(
                Array(repeating: Float(1), count: 64)
                    + Array(repeating: Float(9), count: 64),
                [1, 1, 2, 64]
            )
            let expected = MLXArray(
                Array(repeating: Float(1), count: 64)
                    + Array(repeating: Float(5), count: 64),
                [1, 1, 2, 64]
            )

            let directCache = QuantizedKVCache(groupSize: 64, bits: 4)
            let (quantizedKeys, quantizedValues) = directCache.updateQuantized(
                keys: keys, values: values)
            let causal = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: quantizedKeys,
                quantizedValues: quantizedValues,
                scale: 1,
                mask: .causal,
                groupSize: directCache.groupSize,
                bits: directCache.bits,
                mode: directCache.mode
            )
            let materialized = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: quantizedKeys,
                quantizedValues: quantizedValues,
                scale: 1,
                mask: .array(createCausalMask(n: 2, offset: 0)),
                groupSize: directCache.groupSize,
                bits: directCache.bits,
                mode: directCache.mode
            )

            let turboCache = TurboQuantKVCache(backend: .mlxPacked)
            let turboFallback = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: turboCache,
                scale: 1,
                mask: .causal
            )

            #expect(allClose(causal, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(allClose(materialized, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(allClose(turboFallback, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(turboCache.attentionDiagnostics.activeAttentionPath == .mlxPackedFallback)
        }

        @Test func testQuantizedPackedAttentionSupportsSinks() {
            let queries = MLXArray.zeros([1, 2, 1, 64], dtype: .float32)
            let keys = MLXArray.zeros([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let sinks = MLXArray([Float(0), Float(1)], [2])
            let cache = QuantizedKVCache(groupSize: 32, bits: 4)

            let packed = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 1,
                sinks: sinks
            )
            let reference = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: 1,
                mask: .none,
                sinks: sinks
            )

            #expect(packed.shape == reference.shape)
            #expect(allClose(packed, reference, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }

        @Test func testQuantizedCachesPreserveFloatingPointModeParameters() {
            guard Device.defaultDevice().deviceType == .gpu else { return }

            let keys = MLXArray.ones([1, 1, 2, 64], dtype: .float16)
            let values = MLXArray.ones([1, 1, 2, 64], dtype: .float16)
            let queries = MLXArray.ones([1, 1, 1, 64], dtype: .float16)
            let simple = KVCacheSimple()
            _ = simple.update(keys: keys, values: values)

            let quantized = simple.toQuantized(groupSize: 64, bits: 4, mode: .mxfp4)
            #expect(quantized.groupSize == 32)
            #expect(quantized.bits == 4)
            #expect(quantized.mode == .mxfp4)
            #expect(quantized.state.count == 4)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: QuantizedKVCache(groupSize: 64, bits: 4, mode: .mxfp4),
                scale: 0.125
            )
            #expect(output.shape == [1, 1, 1, 64])

            let rotating = RotatingQuantizedKVCache(
                maxSize: 4,
                groupSize: 64,
                bits: 4,
                mode: .mxfp4
            )
            let state = rotating.updateQuantized(keys: keys, values: values)
            #expect(rotating.groupSize == 32)
            #expect(rotating.mode == .mxfp4)
            #expect(state.0.2 == nil)
            #expect(state.1.2 == nil)
        }

        @Test func testTurboQuantCompressedCacheMaterializesPackedFallbackState() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)
            _ = try cache.updateCompressed(keys: keys, values: values)

            let packedState = try #require(cache.getQuantizedState())
            #expect(packedState.0.0.dim(-2) == 3)
            #expect(packedState.1.0.dim(-2) == 3)

            let fallback = try #require(
                packedQuantizedAttentionFallback(
                    queries: queries,
                    cache: cache,
                    scale: 0.125
                ))
            let directPacked = attentionWithKVState(
                queries: queries,
                state: .quantized(keys: packedState.0, values: packedState.1, cache: cache),
                scale: 0.125
            )

            #expect(fallback.shape == [1, 4, 1, 64])
            #expect(allClose(fallback, directPacked, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }

        @Test func testRotatingTurboQuantCompressedCacheMaterializesPackedFallbackState() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = RotatingTurboQuantKVCache(
                maxSize: 4,
                keep: 1,
                backend: .metalPolarQJL
            )
            let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)
            _ = try cache.updateCompressed(keys: keys, values: values)

            let packedState = try #require(cache.getQuantizedState())
            #expect(packedState.0.0.dim(-2) == 3)
            #expect(packedState.1.0.dim(-2) == 3)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)

            let fallback = try #require(
                packedQuantizedAttentionFallback(
                    queries: queries,
                    cache: cache,
                    scale: 0.125
                ))

            #expect(fallback.shape == [1, 4, 1, 64])
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
        }

        @Test func testRotatingTurboQuantPackedFallbackTracksUpdatesAfterMaterialization() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = RotatingTurboQuantKVCache(
                maxSize: 4,
                keep: 1,
                backend: .metalPolarQJL
            )
            let firstKeys = (MLXArray(0 ..< Int32(1 * 2 * 3 * 64)).asType(.float32) / 128)
                .reshaped([1, 2, 3, 64])
            let firstValues = firstKeys + 0.5
            let secondKeys = firstKeys + 1.0
            let secondValues = firstValues + 1.0
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)

            _ = try cache.updateCompressed(keys: firstKeys, values: firstValues)
            let firstPackedState = try #require(cache.getQuantizedState())
            #expect(firstPackedState.0.0.dim(-2) == 3)
            #expect(firstPackedState.1.0.dim(-2) == 3)
            _ = try cache.updateCompressed(keys: secondKeys, values: secondValues)

            let packedState = try #require(cache.getQuantizedState())
            #expect(packedState.0.0.dim(-2) == 4)
            #expect(packedState.1.0.dim(-2) == 4)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)

            let fallback = try #require(
                packedQuantizedAttentionFallback(
                    queries: queries,
                    cache: cache,
                    scale: 0.125
                ))
            let directPacked = attentionWithKVState(
                queries: queries,
                state: .quantized(
                    keys: packedState.0,
                    values: packedState.1,
                    cache: cache
                ),
                scale: 0.125
            )

            #expect(fallback.shape == directPacked.shape)
            #expect(allClose(fallback, directPacked, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
        }

        @Test func testTurboQuantCompressedAttentionStateWhenAvailable() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)

            #expect(
                cache.supportsCompressedAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    mask: .none
                )
            )
            let (compressedKeys, compressedValues) = try cache.updateCompressed(
                keys: keys, values: values)

            #expect(cache.compressedState != nil)
            #expect(compressedKeys.layout.logicalLength == 2)
            #expect(compressedValues.layout.logicalLength == 2)
            #expect(cache.attentionDiagnostics.activeAttentionPath == .tiledOnlineFused)
            #expect(cache.attentionDiagnostics.rawFallbackAllocated == false)
        }

        @Test func testConservativeTurboQuantUsesExactRawShadowForQwenLikeDecode() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let headDimension = 256
            let queryHeadCount = 16
            let kvHeadCount = 4
            let prefillLength = 8
            let scale = 1 / sqrt(Float(headDimension))

            func values(count: Int, sinFactor: Double, cosFactor: Double) -> [Float] {
                (0 ..< count).map { index in
                    let position = Double(index)
                    return Float(0.31 * sin(position * sinFactor) + 0.23 * cos(position * cosFactor))
                }
            }

            let prefillQueries = MLXArray(
                values(
                    count: queryHeadCount * prefillLength * headDimension,
                    sinFactor: 0.013,
                    cosFactor: 0.041
                ),
                [1, queryHeadCount, prefillLength, headDimension]
            )
            let prefillKeys = MLXArray(
                values(
                    count: kvHeadCount * prefillLength * headDimension,
                    sinFactor: 0.019,
                    cosFactor: 0.071
                ),
                [1, kvHeadCount, prefillLength, headDimension]
            )
            let prefillValues = MLXArray(
                values(
                    count: kvHeadCount * prefillLength * headDimension,
                    sinFactor: 0.029,
                    cosFactor: 0.053
                ),
                [1, kvHeadCount, prefillLength, headDimension]
            )
            let decodeQueries = MLXArray(
                values(count: queryHeadCount * headDimension, sinFactor: 0.017, cosFactor: 0.037),
                [1, queryHeadCount, 1, headDimension]
            )
            let decodeKeys = MLXArray(
                values(count: kvHeadCount * headDimension, sinFactor: 0.023, cosFactor: 0.067),
                [1, kvHeadCount, 1, headDimension]
            )
            let decodeValues = MLXArray(
                values(count: kvHeadCount * headDimension, sinFactor: 0.031, cosFactor: 0.047),
                [1, kvHeadCount, 1, headDimension]
            )

            let rawCache = RotatingKVCache(maxSize: 512)
            _ = rawCache.update(keys: prefillKeys, values: prefillValues)

            let turboCache = RotatingTurboQuantKVCache(
                maxSize: 512,
                preset: .turbo8,
                backend: .metalPolarQJL,
                optimizationPolicy: .conservative,
                valueBits: 8
            )
            _ = try attentionWithCacheUpdateReturningStateThrowing(
                queries: prefillQueries,
                keys: prefillKeys,
                values: prefillValues,
                cache: turboCache,
                scale: scale,
                mask: .causal
            )

            let rawDecodeState = rawCache.update(keys: decodeKeys, values: decodeValues)
            let expected = MLXFast.scaledDotProductAttention(
                queries: decodeQueries,
                keys: rawDecodeState.0,
                values: rawDecodeState.1,
                scale: scale,
                mask: .causal
            )
            let actual = try attentionWithCacheUpdateReturningStateThrowing(
                queries: decodeQueries,
                keys: decodeKeys,
                values: decodeValues,
                cache: turboCache,
                scale: scale,
                mask: .causal
            ).output

            #expect(turboCache.attentionDiagnostics.rawFallbackAllocated)
            #expect(turboCache.attentionDiagnostics.lastFallback?.toPath == .baseline)
            #expect(turboCache.attentionDiagnostics.lastFallback?.isSemanticallyExact == true)
            #expect(allClose(actual, expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }

        @Test func testTurboQuantCompressedAttentionStateSupportsSplitValueDimension() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 80], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)

            let result = attentionWithCacheUpdateReturningState(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 0.125
            )

            #expect(result.output.shape == [1, 4, 1, 80])
            guard case .turboQuant(let compressedKeys, let compressedValues, _) = result.state
            else {
                Issue.record("Expected TurboQuant compressed attention state")
                return
            }
            #expect(compressedKeys.layout.headDimension == 64)
            #expect(compressedValues.layout.headDimension == 80)
            #expect(cache.attentionDiagnostics.activeAttentionPath == .twoStageCompressed)
        }

        @Test func testTurboQuantCompressedStateSerializationKeepsSplitValueDimension() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 80], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)
            _ = attentionWithCacheUpdateReturningState(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 0.125
            )

            let url = tempURL()
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)
            let restored = try #require(loaded.first as? TurboQuantKVCache)
            let compressed = try #require(restored.compressedState)

            #expect(compressed.0.layout.headDimension == 64)
            #expect(compressed.1.layout.headDimension == 80)
        }

        @Test func testQuantizedAttentionStateSupportsSplitValueDimension() {
            let cache = QuantizedKVCache(groupSize: 64, bits: 4)
            let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 128], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)

            let result = attentionWithCacheUpdateReturningState(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: 0.125
            )

            #expect(result.output.shape == [1, 4, 1, 128])
            guard case .quantized = result.state else {
                Issue.record("Expected quantized attention state")
                return
            }
        }

        @Test func testTurboQuantConservativePolicyUsesTwoStageCompressedAttentionWhenAvailable()
            throws
        {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(
                backend: .metalPolarQJL, optimizationPolicy: .conservative)
            let keys = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 2, 64], dtype: .float32)
            let queries = MLXArray.ones([1, 4, 1, 64], dtype: .float32)

            #expect(
                cache.supportsCompressedAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    mask: .none
                )
            )
            #expect(cache.attentionDiagnostics.activeAttentionPath == .twoStageCompressed)
        }

        @Test func testRotatingTurboQuantCompressedStateIsRawFreeWhenAvailable() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = RotatingTurboQuantKVCache(maxSize: 8, backend: .metalPolarQJL)
            let keys = MLXArray.ones([1, 2, 10, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 10, 64], dtype: .float32)
            _ = try cache.updateCompressed(keys: keys, values: values)

            let compressed = try #require(cache.compressedState)
            #expect(cache.state.count == 10)
            #expect(compressed.0.layout.layoutVersion == TurboQuantAttentionLayout.currentVersion)
            #expect(compressed.0.layout.capacity == 8)
            #expect(compressed.0.layout.logicalLength == 8)
            #expect(compressed.0.layout.pinnedPrefixLength == 4)
            #expect(cache.diagnostics.rawFallbackAllocated == false)
        }

        @Test func testRotatingTurboQuantCompressedBulkWritePreservesLogicalOrder() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let tokenCount = 4
            let headDimension = 64
            let keysValues: [Float] = (0 ..< (1 * 2 * tokenCount * headDimension)).map {
                index in
                let token = (index / headDimension) % tokenCount
                return Float(token * 16 + 1)
            }
            let keys = MLXArray(keysValues, [1, 2, tokenCount, headDimension])
            let values = keys + 0.5
            let cache = RotatingTurboQuantKVCache(
                maxSize: 8,
                keep: 2,
                backend: .metalPolarQJL
            )
            _ = try cache.updateCompressed(keys: keys, values: values)

            let compressed = try #require(cache.compressedState)
            let decodedKeys = try turboQuantMetalDecodeAttention(
                compressed.0,
                outputDType: .float32
            )
            let decoded = decodedKeys.asArray(Float.self)
            let means = (0 ..< tokenCount).map { token in
                let start = token * headDimension
                let end = start + headDimension
                return decoded[start ..< end].reduce(Float(0), +) / Float(headDimension)
            }

            #expect(compressed.0.layout.logicalLength == tokenCount)
            #expect(compressed.0.layout.capacity == 8)
            #expect(zip(means, means.dropFirst()).allSatisfy { pair in pair.0 < pair.1 })
            #expect(means[1] - means[0] > 4)
            #expect(means[2] - means[1] > 4)
            #expect(means[3] - means[2] > 4)
        }

        @Test func testTurboQuantCompressedCacheExpansionKeepsCompactUnusedBitsets() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
                return
            }

            let cache = TurboQuantKVCache(backend: .metalPolarQJL)
            let firstTokenCount = 255
            let secondTokenCount = 3
            let firstKeys = MLXArray.ones([1, 2, firstTokenCount, 64], dtype: .float32)
            let firstValues = firstKeys + 0.25
            let secondKeys = MLXArray.ones([1, 2, secondTokenCount, 64], dtype: .float32) * 2.0
            let secondValues = secondKeys + 0.25

            _ = try cache.updateCompressed(keys: firstKeys, values: firstValues)
            let beforeExpansion = try #require(cache.compressedState)
            #expect(beforeExpansion.0.layout.capacity == 256)
            #expect(beforeExpansion.0.signs.dim(2) == 256)
            #expect(beforeExpansion.0.residualSigns.shape == [1])
            #expect(beforeExpansion.1.signs.shape == [1])
            #expect(beforeExpansion.1.highPrecisionMask.shape == [1])
            #expect(beforeExpansion.1.residualSigns.shape == [1])

            _ = try cache.updateCompressed(keys: secondKeys, values: secondValues)
            let expanded = try #require(cache.compressedState)
            #expect(expanded.0.layout.logicalLength == firstTokenCount + secondTokenCount)
            #expect(expanded.0.layout.capacity == 512)
            #expect(expanded.1.layout.capacity == 512)
            #expect(expanded.1.packedMagnitudes.dim(2) == 512)
            #expect(expanded.0.signs.dim(2) == 512)
            #expect(expanded.0.highPrecisionMask.dim(2) == 512)
            #expect(expanded.0.residualSigns.shape == [1])
            #expect(expanded.1.signs.shape == [1])
            #expect(expanded.1.highPrecisionMask.shape == [1])
            #expect(expanded.1.residualSigns.shape == [1])
            #expect(expanded.1.scales.dim(2) == 512)

            let decodedValues = try turboQuantMetalDecodeAttention(
                expanded.1,
                outputDType: .float32
            )
            #expect(decodedValues.shape == [1, 2, firstTokenCount + secondTokenCount, 64])

            let restored = TurboQuantKVCache(backend: .metalPolarQJL)
            restored.metaState = cache.metaState
            restored.state = cache.state
            let restoredCompressed = try #require(restored.compressedState)
            #expect(restoredCompressed.1.signs.shape == [1])
            let restoredValues = try turboQuantMetalDecodeAttention(
                restoredCompressed.1,
                outputDType: .float32
            )
            #expect(restoredValues.shape == decodedValues.shape)
        }

        // MARK: - MambaCache type preservation

        @Test func testMambaCacheRoundTrip() throws {
            let cache = MambaCache()
            let a = MLXArray.ones([2, 4], dtype: .float32) * 5.0
            let b = MLXArray.ones([2, 4], dtype: .float32) * 9.0
            cache[0] = a
            cache[1] = b

            let url = tempURL()
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)

            #expect(loaded.count == 1)
            let restored = try #require(loaded[0] as? MambaCache)
            #expect(restored.slotCount == 2)
            assertArraysClose(restored.state, cache.state)
        }

        // MARK: - CacheList with KV caches

        @Test func testCacheListKVCaches() throws {
            let simple = KVCacheSimple()
            let rotating = RotatingKVCache(maxSize: 32)

            let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
            let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
            _ = simple.update(keys: keys, values: values)
            _ = rotating.update(keys: keys * 2.0, values: values * 2.0)

            let cacheList = CacheList(simple, rotating)

            let url = tempURL()
            try savePromptCache(url: url, cache: [cacheList], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)

            #expect(loaded.count == 1)
            let restored = try #require(loaded[0] as? CacheList)
            let child0 = try #require(restored[0] as? KVCacheSimple)
            let child1 = try #require(restored[1] as? RotatingKVCache)

            assertArraysClose(child0.state, simple.state, label: "child0")
            assertArraysClose(child1.state, rotating.state, label: "child1")
            #expect(child1.metaState == rotating.metaState)
        }

        // MARK: - CacheList with hybrid (MambaCache + KVCacheSimple)

        @Test func testCacheListHybrid() throws {
            let mamba = MambaCache()
            mamba[0] = MLXArray.ones([2, 4], dtype: .float32) * 3.0
            mamba[1] = MLXArray.ones([2, 4], dtype: .float32) * 4.0

            let simple = KVCacheSimple()
            let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
            let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
            _ = simple.update(keys: keys, values: values)

            let cacheList = CacheList(mamba, simple)

            let url = tempURL()
            try savePromptCache(url: url, cache: [cacheList], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)

            #expect(loaded.count == 1)
            let restored = try #require(loaded[0] as? CacheList)
            let restoredMamba = try #require(restored[0] as? MambaCache)
            let restoredSimple = try #require(restored[1] as? KVCacheSimple)

            assertArraysClose(restoredMamba.state, mamba.state, label: "mamba")
            assertArraysClose(restoredSimple.state, simple.state, label: "simple")
        }

        // MARK: - Simple cache round-trip with value assertions

        @Test func testSimpleCacheRoundTrip() throws {
            let cache = KVCacheSimple()
            let keys = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
            let values = MLXArray.ones([1, 8, 16, 64], dtype: .bfloat16)
            _ = cache.update(keys: keys, values: values)

            let url = tempURL()
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)
            #expect(loaded.count == 1)
            assertArraysClose(loaded[0].state, cache.state)
        }

        // MARK: - ArraysCache fully populated round-trip

        @Test func testArraysCacheFullyPopulated() throws {
            let cache = ArraysCache(size: 2)
            cache[0] = MLXArray.ones([2, 4], dtype: .float32)
            cache[1] = MLXArray.ones([2, 4], dtype: .float32) * 2.0

            let url = tempURL()
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)

            #expect(loaded.count == 1)
            let restored = try #require(loaded[0] as? ArraysCache)
            #expect(restored.slotCount == 2)
            assertArraysClose(restored.state, cache.state)
        }

        /// Verify that copy() produces an independent cache: same type, same state,
        /// but mutating the copy does not affect the original.
        @Test(
            .serialized,
            arguments: cacheCreators)
        func testCacheCopyIsIndependent(creator: (() -> any KVCache)) async throws {
            let original = creator()

            let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
            let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)

            // populate the original
            switch original {
            case let arrays as ArraysCache:
                arrays[0] = keys
                arrays[1] = values
            case let turbo as TurboQuantKVCache:
                _ = turbo.updateQuantized(keys: keys, values: values)
            case let turbo as RotatingTurboQuantKVCache:
                _ = turbo.updateQuantized(keys: keys, values: values)
            case let quantized as QuantizedKVCache:
                _ = quantized.updateQuantized(keys: keys, values: values)
            default:
                _ = original.update(keys: keys, values: values)
            }

            let originalOffset = original.offset
            let originalState = original.state
            eval(originalState)
            let originalMeta = original.metaState

            // copy
            let copied = original.copy()

            // same type
            #expect(type(of: original) == type(of: copied))

            // same offset and metadata
            #expect(copied.offset == originalOffset)
            #expect(copied.metaState == originalMeta)

            // same state values
            let copiedState = copied.state
            eval(copiedState)
            #expect(copiedState.count == originalState.count)
            for (origArr, copyArr) in zip(originalState, copiedState) {
                #expect(origArr.shape == copyArr.shape)
                #expect(allClose(origArr, copyArr).item(Bool.self))
            }

            // mutate the copy — push more tokens through it
            let moreKeys = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
            let moreValues = MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)

            switch copied {
            case let arrays as ArraysCache:
                // overwrite slot 0 with a different array
                arrays[0] = moreKeys
            case let quantized as QuantizedKVCache:
                _ = quantized.updateQuantized(keys: moreKeys, values: moreValues)
            default:
                _ = copied.update(keys: moreKeys, values: moreValues)
            }

            // original must be unchanged
            #expect(original.offset == originalOffset)
            #expect(original.metaState == originalMeta)
            let currentState = original.state
            eval(currentState)
            #expect(currentState.count == originalState.count)
            for (origArr, savedArr) in zip(currentState, originalState) {
                #expect(origArr.shape == savedArr.shape)
                #expect(allClose(origArr, savedArr).item(Bool.self))
            }
        }

        /// copy() on an empty (unpopulated) cache must not crash.
        @Test(
            .serialized,
            arguments: cacheCreators)
        func testCacheCopyOnEmptyCache(creator: (() -> any KVCache)) async throws {
            let empty = creator()
            let copied = empty.copy()

            #expect(type(of: empty) == type(of: copied))
            #expect(copied.offset == 0)
            #expect(copied.state.count == empty.state.count)
        }

        /// CacheList.copy() produces independent sub-caches.
        @Test
        func testCacheListCopyIsIndependent() async throws {
            let sub1 = KVCacheSimple()
            let sub2 = RotatingKVCache(maxSize: 32)
            let composite = CacheList(sub1, sub2)

            let keys = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
            let values = MLXArray.ones([1, 8, 4, 64], dtype: .bfloat16)
            _ = sub1.update(keys: keys, values: values)
            _ = sub2.update(keys: keys, values: values)

            // snapshot original state — eval to materialize before copy
            let originalState = composite.state
            eval(originalState)
            let originalOffset0 = sub1.offset
            let originalOffset1 = sub2.offset

            let copied = composite.copy()

            #expect(copied is CacheList)
            let copiedState = copied.state
            eval(copiedState)
            #expect(copiedState.count == originalState.count)
            for (orig, copy) in zip(originalState, copiedState) {
                #expect(orig.shape == copy.shape)
                #expect(allClose(orig, copy).item(Bool.self))
            }

            // mutate inside the copy
            let copiedList = copied as! CacheList
            _ = copiedList[0].update(
                keys: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16),
                values: MLXArray.zeros([1, 8, 2, 64], dtype: .bfloat16)
            )

            // originals unchanged
            #expect(sub1.offset == originalOffset0)
            #expect(sub2.offset == originalOffset1)
            let currentState = composite.state
            eval(currentState)
            #expect(currentState.count == originalState.count)
            for (orig, saved) in zip(currentState, originalState) {
                #expect(orig.shape == saved.shape)
                #expect(allClose(orig, saved).item(Bool.self))
            }
        }

    }

}
