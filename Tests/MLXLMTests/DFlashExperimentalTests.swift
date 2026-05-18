import MLX
import Testing

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {
    @Suite
    struct DFlashExperimentalTests {
        @Test func acceptanceLengthCountsConsecutiveMatches() {
            let drafted = MLXArray([1, 2, 3, 4])
            let posterior = MLXArray([1, 2, 9, 4])

            let accepted = DFlashRuntime.matchAcceptanceLength(
                draftedTokens: drafted,
                posteriorTokens: posterior
            )

            #expect(accepted.item(Int.self) == 2)
        }

        @Test func greedySelectionHonorsSuppressMask() {
            let logits = MLXArray([0.1 as Float, 9.0 as Float, 3.0 as Float]).reshaped(1, 3)
            let mask = DFlashRuntime.buildSuppressTokenMask(vocabSize: 3, suppressTokenIDs: [1])

            let token = DFlashRuntime.greedyTokensWithMask(
                logits: logits,
                suppressTokenMask: mask
            )

            #expect(token.item(Int.self) == 2)
        }

        @Test func registryResolvesExplicitExactAndPrefixRefs() {
            #expect(
                DFlashDraftRegistry.resolveDraftRef(
                    modelRef: "anything",
                    draftRef: "local/draft"
                ) == "local/draft"
            )
            #expect(
                DFlashDraftRegistry.resolveDraftRef(modelRef: "org/Qwen3-4B")
                    == "z-lab/Qwen3-4B-DFlash-b16"
            )
            #expect(
                DFlashDraftRegistry.resolveDraftRef(modelRef: "Qwen3.5-4B-4bit")
                    == "z-lab/Qwen3.5-4B-DFlash"
            )
        }

        @Test func draftConfigurationBuildsDeterministicLayerIDs() {
            #expect(
                buildDFlashTargetLayerIDs(numTargetLayers: 36, numDraftLayers: 4)
                    == [1, 12, 22, 33]
            )

            let config = DFlashDraftConfiguration(
                hiddenSize: 4,
                numHiddenLayers: 1,
                intermediateSize: 8,
                numAttentionHeads: 1,
                numKeyValueHeads: 1,
                headDim: 4,
                numTargetLayers: 8,
                blockSize: 3,
                dflashConfig: .init(targetLayerIds: [2], maskTokenId: 99)
            )
            let model = DFlashDraftModel(config)

            #expect(model.targetLayerIDs == [2])
            #expect(model.blockSize == 3)
            #expect(model.maskTokenID == 99)
        }

        @Test func contextOnlyDraftCacheKeepsSinkAndWindow() {
            let cache = ContextOnlyDraftKVCache(sinkSize: 2, windowSize: 3)
            let first = MLXArray.ones([1, 1, 3, 2], dtype: .float32)
            let second = MLXArray.ones([1, 1, 4, 2], dtype: .float32) * 2

            cache.appendContext(contextKeys: first, contextValues: first, numPositions: 3)
            cache.appendContext(contextKeys: second, contextValues: second, numPositions: 4)

            #expect(cache.offset == 7)
            #expect(cache.cacheLength == 5)
            #expect(cache.keys?.shape == [1, 1, 5, 2])
            #expect(cache.values?.shape == [1, 1, 5, 2])
        }

        @Test func snapshotCacheRollbackRestoresCapturedState() {
            let cache = MambaSnapshotCache()
            let conv = MLXArray.ones([1, 3, 4], dtype: .float32)
            let recurrent = MLXArray.ones([1, 1, 4, 4], dtype: .float32)
            cache[0] = conv
            cache[1] = recurrent
            cache.armRollback(prefixLen: 0)

            cache[0] = MLXArray.zeros([1, 3, 4], dtype: .float32)
            cache[1] = MLXArray.zeros([1, 1, 4, 4], dtype: .float32)
            cache.rollback(nAccepted: 0)

            #expect(cache.isArmed == false)
            #expect(allClose(cache[0]!, conv).item(Bool.self))
            #expect(allClose(cache[1]!, recurrent).item(Bool.self))
        }
    }
}
