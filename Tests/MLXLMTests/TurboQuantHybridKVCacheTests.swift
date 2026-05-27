import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {
    @Suite
    struct TurboQuantHybridKVCacheTests {
        @Test func testHybridStrategyCreatesHybridCacheImmediately() {
            let parameters = GenerateParameters(
                maxKVSize: 65_536,
                kvCacheStrategy: .hybridTurboQuant,
                turboQuantHotWindowTokens: 8192,
                turboQuantColdBlockTokens: 1024,
                turboQuantColdBudgetTokens: 4096,
                turboQuantMaxColdBudgetTokens: 8192,
                turboQuantColdAttentionMode: .selected
            )

            let cache = makeAttentionKVCache(parameters: parameters)

            let hybrid = cache as? HybridTurboQuantKVCache
            #expect(hybrid != nil)
            #expect(hybrid?.hotWindowTokens == 8192)
            #expect(hybrid?.coldBlockTokens == 1024)
            #expect(hybrid?.coldBudgetTokens == 4096)
            #expect(hybrid?.maxColdBudgetTokens == 8192)
            #expect(hybrid?.coldAttentionMode == .selected)
        }

        @Test func testHybridPromptCacheAssignsLayerPolicyIdentity() {
            let parameters = GenerateParameters(
                maxKVSize: 65_536,
                kvCacheStrategy: .hybridTurboQuant,
                turboQuantLayerPolicy: .auto
            )

            let caches = makePromptCacheWithLayerCount(
                numLayers: 8,
                parameters: parameters
            )

            let first = caches[0] as? HybridTurboQuantKVCache
            let final = caches[7] as? HybridTurboQuantKVCache
            #expect(first?.layerIndex == 0)
            #expect(first?.layerCount == 8)
            #expect(final?.layerIndex == 7)
            #expect(final?.layerCount == 8)
        }

        @Test func testHybridSelectorIncludesAnchorsAndNearestColdBlock() {
            let descriptors = [
                descriptor(id: 0, start: 0, end: 1024, anchorFlags: [.system]),
                descriptor(id: 1, start: 1024, end: 2048),
                descriptor(id: 2, start: 2048, end: 3072),
                descriptor(id: 3, start: 3072, end: 4096),
            ]

            let selection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: 2048,
                maxBudgetTokens: 2048
            )

            #expect(selection.selectedBlockIDs.contains(0))
            #expect(selection.selectedBlockIDs.contains(3))
            #expect(selection.selectedTokenCount <= 2048)
            #expect(selection.reasonFlags.contains("anchor"))
            #expect(selection.reasonFlags.contains("nearest"))
        }

        @Test func testHybridSelectorHonorsOffAndExhaustiveModes() {
            let descriptors = [
                descriptor(id: 0, start: 0, end: 512),
                descriptor(id: 1, start: 512, end: 1024),
            ]

            let offSelection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .off,
                budgetTokens: 4096,
                maxBudgetTokens: 4096
            )
            #expect(offSelection.selectedBlockIDs.isEmpty)

            let exhaustiveSelection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .exhaustive,
                budgetTokens: 0,
                maxBudgetTokens: 0
            )
            #expect(exhaustiveSelection.selectedBlockIDs == [0, 1])
            #expect(exhaustiveSelection.selectedTokenCount == 1024)
            #expect(exhaustiveSelection.reasonFlags == ["exhaustive"])
        }

        @Test func testHybridHotWindowSealsOldFullBlocks() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 16,
                    hotWindowTokens: 4,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 4,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL
                )
                let keys = MLXArray.ones([1, 2, 9, 64], dtype: .float32)
                let values = MLXArray.ones([1, 2, 9, 64], dtype: .float32)

                _ = cache.update(keys: keys, values: values)

                #expect(cache.offset == 9)
                #expect(cache.coldBlockCount == 2)
                #expect(cache.coldTokenCount == 4)
                #expect(cache.rawHotLength == 5)
                #expect(cache.coldBlockDescriptors.map(\.startToken) == [0, 2])
                #expect(cache.coldBlockDescriptors.map(\.endToken) == [2, 4])
            }
        }

        @Test func testHybridDecodeUsesSegmentedSelectedColdAttentionWhenAvailable() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else { return }

            let cache = HybridTurboQuantKVCache(
                maxSize: 32,
                hotWindowTokens: 4,
                coldBlockTokens: 2,
                coldBudgetTokens: 4,
                maxColdBudgetTokens: 4,
                preset: .turbo4v2,
                groupSize: 64,
                backend: .metalPolarQJL
            )
            let keys = MLXArray.ones([1, 2, 9, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 9, 64], dtype: .float32) * 0.75
            _ = cache.update(keys: keys, values: values)
            #expect(cache.coldBlockCount > 0)

            let queries = MLXArray.ones([1, 2, 1, 64], dtype: .float32)
            let nextKeys = MLXArray.ones([1, 2, 1, 64], dtype: .float32) * 0.25
            let nextValues = MLXArray.ones([1, 2, 1, 64], dtype: .float32) * 0.5

            let result = try turboQuantHybridAttentionThrowing(
                queries: queries,
                keys: nextKeys,
                values: nextValues,
                cache: cache,
                scale: 1 / sqrt(Float(64)),
                mask: .causal,
                sinks: nil
            )

            #expect(result.output.shape == [1, 2, 1, 64])
            #expect(cache.lastSelection.selectedTokenCount > 0)
            #expect(result.state.keyLength == cache.rawHotLength)
            #expect(cache.diagnostics.fallbackReason == nil)
        }

        @Test func testHybridPromptCacheRoundTripsHotState() throws {
            let cache = HybridTurboQuantKVCache(
                maxSize: 32,
                hotWindowTokens: 16,
                coldBlockTokens: 4,
                coldBudgetTokens: 4,
                maxColdBudgetTokens: 8
            )
            let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32) * 2
            _ = cache.update(keys: keys, values: values)

            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("safetensors")
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)
            let restored = try #require(loaded.first as? HybridTurboQuantKVCache)

            #expect(restored.offset == 3)
            #expect(restored.rawHotLength == 3)
            #expect(restored.coldBlockCount == 0)
            #expect(restored.state.count == 2)
            #expect(allClose(restored.state[0], keys).item(Bool.self))
            #expect(allClose(restored.state[1], values).item(Bool.self))
        }

        private func descriptor(
            id: Int,
            start: Int,
            end: Int,
            anchorFlags: TurboQuantColdBlockAnchorFlags = []
        ) -> TurboQuantColdBlockDescriptor {
            TurboQuantColdBlockDescriptor(
                blockID: id,
                startToken: start,
                endToken: end,
                compressedSlotStart: 0,
                compressedSlotEnd: end - start,
                logicalTokenCount: end - start,
                recencyRank: id,
                anchorFlags: anchorFlags,
                maxKeyNormEstimate: 0,
                relevanceScore: Float(end)
            )
        }
    }
}
