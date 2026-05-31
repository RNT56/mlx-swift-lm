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

            let middle = caches[4] as? HybridTurboQuantKVCache
            #expect(middle?.layerIndex == 4)
            #expect(middle?.layerCount == 8)
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
            #expect(selection.budgetedTokenCount <= 2048)
            #expect(selection.anchorTokenCount == 1024)
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
            #expect(exhaustiveSelection.selectorEscalation == .exhaustive)
            #expect(exhaustiveSelection.reasonFlags == ["exhaustive"])
        }

        @Test func testHybridSelectorKeepsAnchorsOutsideStrictBudget() {
            let descriptors = [
                descriptor(id: 0, start: 0, end: 1024, anchorFlags: [.system]),
                descriptor(id: 1, start: 1024, end: 2048, anchorFlags: [.tool]),
                descriptor(id: 2, start: 2048, end: 3072),
            ]

            let selection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: 1024,
                maxBudgetTokens: 1024,
                selectorPolicy: TurboQuantColdSelectorPolicy(
                    nearestBlockCount: 1,
                    minimumConfidence: 0,
                    allowMaxBudgetEscalation: false
                )
            )

            #expect(selection.selectedBlockIDs.contains(0))
            #expect(selection.selectedBlockIDs.contains(1))
            #expect(selection.selectedBlockIDs.contains(2))
            #expect(selection.budgetedTokenCount == 1024)
            #expect(selection.anchorTokenCount == 2048)
            #expect(selection.anchorOverflowTokens == 1024)
        }

        @Test func testHybridSelectorRanksSemanticLexicalKeyNormThenRecency() {
            let descriptors = [
                descriptor(id: 0, start: 0, end: 512, keyNorm: 10),
                descriptor(id: 1, start: 512, end: 1024, lexicalScore: 1),
                descriptor(id: 2, start: 1024, end: 1536, semanticScore: 1),
                descriptor(id: 3, start: 1536, end: 2048),
            ]

            let selection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: 512,
                maxBudgetTokens: 512,
                selectorPolicy: TurboQuantColdSelectorPolicy(
                    nearestBlockCount: 0,
                    minimumConfidence: 0,
                    allowMaxBudgetEscalation: false
                )
            )

            #expect(selection.selectedBlockIDs == [2])
            #expect(selection.reasonFlags.contains("semantic"))
        }

        @Test func testHybridSelectorEscalatesToMaxBudgetWhenConfidenceIsLow() {
            let descriptors = (0 ..< 8).map {
                descriptor(id: $0, start: $0 * 512, end: ($0 + 1) * 512)
            }

            let selection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: 512,
                maxBudgetTokens: 2048,
                selectorPolicy: TurboQuantColdSelectorPolicy(
                    nearestBlockCount: 1,
                    minimumConfidence: 0.75,
                    allowMaxBudgetEscalation: true,
                    allowExhaustiveEscalation: false
                )
            )

            #expect(selection.selectorEscalation == .maxBudget)
            #expect(selection.selectedTokenBudget == 2048)
            #expect(selection.budgetedTokenCount <= 2048)
            #expect(selection.reasonFlags.contains("max_budget_escalation"))
            #expect(!selection.requiresExhaustiveFallback)
        }

        @Test func testHybridSelectorExhaustiveRequiresExplicitPolicy() {
            let descriptors = (0 ..< 8).map {
                descriptor(id: $0, start: $0 * 512, end: ($0 + 1) * 512)
            }

            let defaultSelection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: 512,
                maxBudgetTokens: 512,
                selectorPolicy: TurboQuantColdSelectorPolicy(
                    nearestBlockCount: 1,
                    minimumConfidence: 0.95,
                    allowMaxBudgetEscalation: false,
                    allowExhaustiveEscalation: false
                )
            )
            let exhaustiveSelection = HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: 512,
                maxBudgetTokens: 512,
                selectorPolicy: TurboQuantColdSelectorPolicy(
                    nearestBlockCount: 1,
                    minimumConfidence: 0.95,
                    allowMaxBudgetEscalation: false,
                    allowExhaustiveEscalation: true
                )
            )

            #expect(defaultSelection.selectorEscalation == .none)
            #expect(defaultSelection.reasonFlags.contains("low_confidence_exhaustive_disabled"))
            #expect(exhaustiveSelection.selectorEscalation == .exhaustive)
            #expect(exhaustiveSelection.reasonFlags.contains("quality_guard_exhaustive"))
            #expect(exhaustiveSelection.requiresExhaustiveFallback)
        }

        @Test func testHybridLayerPolicyBudgetsEarlyMiddleFinalAndCustomLayers() {
            let auto = HybridTurboQuantKVCache(
                hotWindowTokens: 16,
                coldBlockTokens: 4,
                coldBudgetTokens: 4096,
                maxColdBudgetTokens: 8192,
                layerPolicy: .auto
            )
            #expect(auto.coldBudgetForLayer(layerIndex: 0, layerCount: 8, defaultBudget: 4096) == 0)
            #expect(auto.coldBudgetForLayer(layerIndex: 4, layerCount: 8, defaultBudget: 4096) == 2048)
            #expect(auto.coldBudgetForLayer(layerIndex: 5, layerCount: 8, defaultBudget: 4096) == 0)
            #expect(auto.coldBudgetForLayer(layerIndex: 6, layerCount: 8, defaultBudget: 4096) == 4096)

            let all = HybridTurboQuantKVCache(
                hotWindowTokens: 16,
                coldBlockTokens: 4,
                coldBudgetTokens: 4096,
                maxColdBudgetTokens: 8192,
                layerPolicy: .all
            )
            #expect(all.coldBudgetForLayer(layerIndex: 0, layerCount: 8, defaultBudget: 4096) == 4096)

            let finalQuarter = HybridTurboQuantKVCache(
                hotWindowTokens: 16,
                coldBlockTokens: 4,
                coldBudgetTokens: 4096,
                maxColdBudgetTokens: 8192,
                layerPolicy: .finalQuarter
            )
            #expect(finalQuarter.coldBudgetForLayer(layerIndex: 1, layerCount: 8, defaultBudget: 4096) == 0)
            #expect(finalQuarter.coldBudgetForLayer(layerIndex: 4, layerCount: 8, defaultBudget: 4096) == 0)
            #expect(finalQuarter.coldBudgetForLayer(layerIndex: 6, layerCount: 8, defaultBudget: 4096) == 4096)

            let custom = HybridTurboQuantKVCache(
                hotWindowTokens: 16,
                coldBlockTokens: 4,
                coldBudgetTokens: 4096,
                maxColdBudgetTokens: 8192,
                layerPolicy: .custom([1: 1024, 6: 3072])
            )
            #expect(custom.coldBudgetForLayer(layerIndex: nil, layerCount: 8, defaultBudget: 4096) == 4096)
            #expect(custom.coldBudgetForLayer(layerIndex: 1, layerCount: 8, defaultBudget: 4096) == 1024)
            #expect(custom.coldBudgetForLayer(layerIndex: 3, layerCount: 8, defaultBudget: 4096) == 0)
            #expect(custom.coldBudgetForLayer(layerIndex: 6, layerCount: 8, defaultBudget: 4096) == 3072)
        }

        @Test func testHybridSelectedModeExhaustiveFallbackDiagnosticsAreExplicit() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 32,
                    hotWindowTokens: 4,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 2,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    selectorPolicy: TurboQuantColdSelectorPolicy(
                        nearestBlockCount: 1,
                        minimumConfidence: 0.95,
                        allowMaxBudgetEscalation: false,
                        allowExhaustiveEscalation: true
                    )
                )
                _ = cache.update(
                    keys: tokenRamp(start: 0, count: 9),
                    values: tokenRamp(start: 0, count: 9, scale: 2)
                )

                let selection = cache.selectColdBlocks()

                #expect(selection.selectorEscalation == .exhaustive)
                #expect(selection.requiresExhaustiveFallback)
                #expect(cache.diagnostics.route == "hybrid_exhaustive_cold")
                #expect(cache.diagnostics.fullScanFallbackCount == 1)
                #expect(
                    cache.diagnostics.fallbackReason?.contains(
                        "selected_mode_exhaustive_fallback"
                    ) == true
                )
            }
        }

        @Test func testHybridSelectorUsesQuerySummaryWhenAvailable() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 16,
                    hotWindowTokens: 1,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 2,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL,
                    selectorPolicy: TurboQuantColdSelectorPolicy(
                        nearestBlockCount: 0,
                        semanticWeight: 100,
                        lexicalWeight: 0,
                        keyNormWeight: 0,
                        recencyWeight: 0,
                        minimumConfidence: 0,
                        allowMaxBudgetEscalation: false
                    )
                )
                var keyValues = [Float](repeating: 0, count: 1 * 2 * 7 * 64)
                for head in 0 ..< 2 {
                    for token in 0 ..< 2 {
                        keyValues[((head * 7 + token) * 64) + 0] = 1
                    }
                    for token in 2 ..< 4 {
                        keyValues[((head * 7 + token) * 64) + 4] = 1
                    }
                }
                let keys = MLXArray(keyValues, [1, 2, 7, 64])
                let values = MLXArray.ones([1, 2, 7, 64], dtype: .float32)
                _ = cache.update(keys: keys, values: values)

                var firstQueryValues = [Float](repeating: 0, count: 1 * 2 * 1 * 64)
                firstQueryValues[0] = 1
                firstQueryValues[64] = 1
                let firstSelection = cache.selectColdBlocks(
                    query: MLXArray(firstQueryValues, [1, 2, 1, 64])
                )

                var secondQueryValues = [Float](repeating: 0, count: 1 * 2 * 1 * 64)
                secondQueryValues[4] = 1
                secondQueryValues[64 + 4] = 1
                let secondSelection = cache.selectColdBlocks(
                    query: MLXArray(secondQueryValues, [1, 2, 1, 64])
                )

                #expect(firstSelection.selectedBlockIDs == [0])
                #expect(secondSelection.selectedBlockIDs == [1])
                #expect(firstSelection.reasonFlags.contains("semantic"))
                #expect(secondSelection.reasonFlags.contains("semantic"))
            }
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
                #expect(cache.coldBlockCount == 3)
                #expect(cache.coldTokenCount == 6)
                #expect(cache.rawHotLength == 3)
                #expect(cache.rawHotLength <= cache.hotWindowTokens)
                #expect(cache.coldBlockDescriptors.map(\.startToken) == [0, 2, 4])
                #expect(cache.coldBlockDescriptors.map(\.endToken) == [2, 4, 6])
                #expect(cache.coldBlockDescriptors.allSatisfy { $0.maxKeyNormEstimate > 0 })
            }
        }

        @Test func testHybridHotResidencyRetainsExactTailAcrossRepeatedSeals() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 32,
                    hotWindowTokens: 4,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 4,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL
                )

                for token in 0 ..< 12 {
                    _ = cache.update(
                        keys: tokenRamp(start: token, count: 1),
                        values: tokenRamp(start: token, count: 1, scale: 2)
                    )
                    #expect(cache.rawHotLength <= cache.hotWindowTokens)
                }

                let state = cache.state
                #expect(cache.offset == 12)
                #expect(cache.coldBlockDescriptors.map(\.startToken) == [0, 2, 4, 6])
                #expect(cache.coldBlockDescriptors.map(\.endToken) == [2, 4, 6, 8])
                #expect(cache.rawHotLength == 4)
                #expect(state.count >= 2)
                #expect(allClose(state[0], tokenRamp(start: 8, count: 4)).item(Bool.self))
                #expect(allClose(state[1], tokenRamp(start: 8, count: 4, scale: 2)).item(Bool.self))
            }
        }

        @Test func testHybridTrimReportsWholeCompressedBlockForPartialColdRollback() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 16,
                    hotWindowTokens: 1,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 4,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL
                )

                _ = cache.update(
                    keys: tokenRamp(start: 0, count: 6),
                    values: tokenRamp(start: 0, count: 6, scale: 2)
                )

                #expect(cache.offset == 6)
                #expect(cache.rawHotLength == 0)
                #expect(cache.coldBlockDescriptors.map(\.startToken) == [0, 2, 4])

                let trimmed = cache.trim(3)

                #expect(trimmed == 4)
                #expect(cache.offset == 2)
                #expect(cache.rawHotLength == 0)
                #expect(cache.coldBlockDescriptors.map(\.startToken) == [0])
                #expect(cache.coldBlockDescriptors.map(\.endToken) == [2])
            }
        }

        @Test func testHybridCheckpointRestoreRestoresSelectionAfterTrimRollback() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 16,
                    hotWindowTokens: 4,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 2,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL
                )
                _ = cache.update(
                    keys: tokenRamp(start: 0, count: 9),
                    values: tokenRamp(start: 0, count: 9, scale: 2)
                )
                let selection = cache.selectColdBlocks()
                let checkpoint = cache.makeCompressedUpdateCheckpoint(appendingTokenCount: 0)

                _ = cache.trim(5)

                #expect(cache.lastSelection.isEmpty)

                cache.restoreCompressedUpdateCheckpoint(checkpoint)

                #expect(cache.offset == 9)
                #expect(cache.rawHotLength == 3)
                #expect(cache.coldBlockDescriptors.map(\.startToken) == [0, 2, 4])
                #expect(cache.lastSelection == selection)
                #expect(cache.diagnostics.selectedColdBlocks == selection.selectedBlockIDs)
            }
        }

        @Test func testHybridPromptCacheRoundTripsColdDescriptorSummaries() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            let cache = HybridTurboQuantKVCache(
                maxSize: 16,
                hotWindowTokens: 2,
                coldBlockTokens: 2,
                coldBudgetTokens: 2,
                maxColdBudgetTokens: 4,
                preset: .turbo4v2,
                groupSize: 64,
                backend: .metalPolarQJL
            )
            let keys = MLXArray.ones([1, 2, 7, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 7, 64], dtype: .float32)
            _ = cache.update(keys: keys, values: values)
            let originalDescriptor = try #require(cache.coldBlockDescriptors.first)

            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("safetensors")
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)
            let restored = try #require(loaded.first as? HybridTurboQuantKVCache)
            let restoredDescriptor = try #require(restored.coldBlockDescriptors.first)

            #expect(restoredDescriptor.keyMeanSummary == originalDescriptor.keyMeanSummary)
            #expect(restoredDescriptor.scoreUpperBoundEstimate > 0)
            #expect(restoredDescriptor.startToken == originalDescriptor.startToken)
            #expect(restoredDescriptor.endToken == originalDescriptor.endToken)
        }

        @Test func testHybridSelectorHintsApplyBeforeAndAfterSealing() throws {
            guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLCodec else { return }

            Device.withDefaultDevice(.cpu) {
                let cache = HybridTurboQuantKVCache(
                    maxSize: 16,
                    hotWindowTokens: 2,
                    coldBlockTokens: 2,
                    coldBudgetTokens: 2,
                    maxColdBudgetTokens: 4,
                    preset: .turbo4v2,
                    groupSize: 64,
                    backend: .metalPolarQJL
                )
                cache.applySelectorHints([
                    TurboQuantColdSelectorHint(
                        startToken: 0,
                        endToken: 2,
                        semanticScore: 1,
                        anchorFlags: [.system],
                        sourceID: "before"
                    )
                ])

                let keys = MLXArray.ones([1, 2, 7, 64], dtype: .float32)
                let values = MLXArray.ones([1, 2, 7, 64], dtype: .float32)
                _ = cache.update(keys: keys, values: values)

                #expect(cache.coldBlockDescriptors.first?.semanticScore == 1)
                #expect(cache.coldBlockDescriptors.first?.anchorFlags.contains(.system) == true)

                cache.applySelectorHints([
                    TurboQuantColdSelectorHint(
                        startToken: 0,
                        endToken: 2,
                        semanticScore: 1,
                        anchorFlags: [.system],
                        sourceID: "before"
                    ),
                    TurboQuantColdSelectorHint(
                        startToken: 2,
                        endToken: 4,
                        lexicalScore: 1,
                        sourceID: "after"
                    ),
                ])
                #expect(cache.coldBlockDescriptors.dropFirst().first?.lexicalScore == 1)
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
            #expect(
                result.state.keyLength == cache.rawHotLength + cache.lastSelection.selectedTokenCount
            )
            #expect(cache.diagnostics.fallbackReason == nil)
            #expect(cache.diagnostics.route == "hybrid_selected_segmented_attention")
            #expect(cache.runtimeSnapshot().selectedPath == "hybrid_selected_segmented_attention")

            let stateOutput = try attentionWithKVStateThrowing(
                queries: queries,
                state: result.state,
                scale: 1 / sqrt(Float(64)),
                mask: .causal
            )
            #expect(stateOutput.shape == [1, 2, 1, 64])
            #expect(cache.diagnostics.route == "hybrid_selected_segmented_attention")
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

        @Test func testHybridPromptCacheRoundTripsSelectorMetadata() throws {
            let policy = TurboQuantColdSelectorPolicy(
                nearestBlockCount: 3,
                minimumConfidence: 0.5,
                allowMaxBudgetEscalation: true,
                allowExhaustiveEscalation: true
            )
            let hints = [
                TurboQuantColdSelectorHint(
                    startToken: 0,
                    endToken: 4,
                    lexicalScore: 0.25,
                    semanticScore: 0.75,
                    anchorFlags: [.userAnchor],
                    sourceID: "test"
                )
            ]
            let cache = HybridTurboQuantKVCache(
                maxSize: 32,
                hotWindowTokens: 16,
                coldBlockTokens: 4,
                coldBudgetTokens: 4,
                maxColdBudgetTokens: 8,
                selectorPolicy: policy,
                selectorHints: hints
            )
            let keys = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            let values = MLXArray.ones([1, 2, 3, 64], dtype: .float32)
            _ = cache.update(keys: keys, values: values)

            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("safetensors")
            try savePromptCache(url: url, cache: [cache], metadata: [:])
            let (loaded, _) = try loadPromptCache(url: url)
            let restored = try #require(loaded.first as? HybridTurboQuantKVCache)

            #expect(restored.selectorPolicy == policy)
            #expect(restored.selectorHints == hints)
        }

        @Test func testHybridLegacyV1MetaStateStillRestores() throws {
            let restored = try HybridTurboQuantKVCache.restoreFromMetaState(
                state: [],
                metaState: [
                    "HybridTurboQuantKVCache.v1",
                    "offset=0",
                    "maxSize=32",
                    "hotWindowTokens=8",
                    "coldBlockTokens=4",
                    "coldBudgetTokens=4",
                    "maxColdBudgetTokens=8",
                    "coldAttentionMode=selected",
                    "preset=turbo4v2",
                    "backend=metalPolarQJL",
                    "activeBackend=metalPolarQJL",
                    "groupSize=64",
                    "seed=11400714819323198485",
                    "valueBits=None",
                    "layerIndex=None",
                    "layerCount=None",
                    "hotStateCount=0",
                    "coldBlockCount=0",
                ]
            )

            #expect(restored.offset == 0)
            #expect(restored.selectorPolicy == .automatic)
            #expect(restored.selectorHints.isEmpty)
        }

        private func descriptor(
            id: Int,
            start: Int,
            end: Int,
            anchorFlags: TurboQuantColdBlockAnchorFlags = [],
            keyNorm: Float = 0,
            lexicalScore: Float = 0,
            semanticScore: Float = 0
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
                maxKeyNormEstimate: keyNorm,
                lexicalScore: lexicalScore,
                semanticScore: semanticScore,
                relevanceScore: Float(end)
            )
        }

        private func tokenRamp(
            start: Int,
            count: Int,
            heads: Int = 2,
            headDimension: Int = 64,
            scale: Float = 1
        ) -> MLXArray {
            var values: [Float] = []
            values.reserveCapacity(heads * count * headDimension)
            for head in 0 ..< heads {
                for token in 0 ..< count {
                    for _ in 0 ..< headDimension {
                        values.append(Float(start + token + head) * scale)
                    }
                }
            }
            return MLXArray(values, [1, heads, count, headDimension])
        }
    }
}
