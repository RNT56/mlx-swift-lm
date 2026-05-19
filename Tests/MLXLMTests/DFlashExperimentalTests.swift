import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
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

        @Test func concreteLLMModelsExposeDFlashTargetConformance() throws {
            let qwen3 = Qwen3Model(try Self.makeQwen3Config())
            let qwen35 = Qwen35TextModel(try Self.makeQwen35TextConfig())
            let qwen3MoE = Qwen3MoEModel(try Self.makeQwen3MoEConfig())
            let llama = LlamaModel(Self.makeLlamaConfig())
            let qwen3Next = Qwen3NextModel(try Self.makeQwen3NextConfig())

            #expect((qwen3 as any LanguageModel) is any DFlashTargetModel)
            #expect((qwen35 as any LanguageModel) is any DFlashTargetModel)
            #expect((qwen3MoE as any LanguageModel) is any DFlashTargetModel)
            #expect((llama as any LanguageModel) is any DFlashTargetModel)
            #expect((qwen3Next as any LanguageModel) is any DFlashTargetModel)
        }

        @Test func dflashRuntimeGeneratesSummaryWithToyTarget() {
            let target = ToyDFlashTargetModel(hiddenSize: 8, vocabSize: 16)
            let draft = DFlashDraftModel(
                DFlashDraftConfiguration(
                    hiddenSize: 8,
                    numHiddenLayers: 1,
                    intermediateSize: 16,
                    numAttentionHeads: 2,
                    numKeyValueHeads: 1,
                    headDim: 4,
                    numTargetLayers: 2,
                    blockSize: 2,
                    dflashConfig: .init(targetLayerIds: [0], maskTokenId: 0)
                )
            )

            let events = DFlashRuntime.generateSync(
                targetModel: target,
                draftModel: draft,
                promptTokens: [1, 2],
                maxNewTokens: 3,
                blockTokens: 2,
                prefillStepSize: 1
            )
            let summaries = events.compactMap { event -> DFlashSummary? in
                if case .summary(let summary) = event { return summary }
                return nil
            }

            #expect(summaries.count == 1)
            #expect(summaries[0].generationTokens == 3)
            #expect(summaries[0].cyclesCompleted > 0)
        }

        private static func makeQwen3Config() throws -> Qwen3Configuration {
            let json = """
                {
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 16,
                    "head_dim": 4
                }
                """
            return try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
        }

        private static func makeQwen35TextConfig() throws -> Qwen35TextConfiguration {
            let json = """
                {
                    "model_type": "qwen3_5",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "linear_num_value_heads": 1,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_conv_kernel_dim": 4,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 16,
                    "full_attention_interval": 2
                }
                """
            return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
        }

        private static func makeQwen3MoEConfig() throws -> Qwen3MoEConfiguration {
            let json = """
                {
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 16,
                    "head_dim": 4,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "moe_intermediate_size": 8
                }
                """
            return try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: Data(json.utf8))
        }

        private static func makeLlamaConfig() -> LlamaConfiguration {
            LlamaConfiguration(
                hiddenSize: 8,
                hiddenLayers: 2,
                intermediateSize: 16,
                attentionHeads: 2,
                headDimensions: 4,
                rmsNormEps: 1e-6,
                vocabularySize: 16,
                kvHeads: 1
            )
        }

        private static func makeQwen3NextConfig() throws -> Qwen3NextConfiguration {
            let json = """
                {
                    "model_type": "qwen3_next",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "linear_num_value_heads": 1,
                    "linear_num_key_heads": 1,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_conv_kernel_dim": 4,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "decoder_sparse_step": 1,
                    "shared_expert_intermediate_size": 8,
                    "mlp_only_layers": [],
                    "moe_intermediate_size": 8,
                    "rms_norm_eps": 1e-6,
                    "vocab_size": 16,
                    "num_key_value_heads": 1,
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "max_position_embeddings": 64,
                    "full_attention_interval": 2
                }
                """
            return try JSONDecoder().decode(Qwen3NextConfiguration.self, from: Data(json.utf8))
        }
    }
}

private final class ToyDFlashTargetModel: Module, DFlashTargetModel {
    let hiddenSize: Int
    let vocabularySize: Int
    let kvHeads: [Int] = [1]

    init(hiddenSize: Int, vocabSize: Int) {
        self.hiddenSize = hiddenSize
        self.vocabularySize = vocabSize
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        dflashForwardWithCapture(inputIDs: inputs, cache: cache ?? [], captureLayerIDs: [1]).0
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    func dflashEmbedTokens(_ tokens: MLXArray) -> MLXArray {
        MLXArray.zeros([tokens.dim(0), tokens.dim(1), hiddenSize])
    }

    func dflashLmHeadLogits(_ hiddenStates: MLXArray) -> MLXArray {
        logits(batch: hiddenStates.dim(0), length: hiddenStates.dim(1))
    }

    func dflashForwardWithCapture(
        inputIDs: MLXArray,
        cache: [KVCache],
        captureLayerIDs: Set<Int>
    ) -> (MLXArray, [Int: MLXArray]) {
        let batch = inputIDs.dim(0)
        let length = inputIDs.dim(1)
        let hidden = MLXArray.ones([batch, length, hiddenSize])
        var captured = [Int: MLXArray]()
        for layerID in captureLayerIDs {
            captured[layerID] = hidden
        }
        return (logits(batch: batch, length: length), captured)
    }

    var dflashIsHybridGDN: Bool { false }

    private func logits(batch: Int, length: Int) -> MLXArray {
        var values = Array(repeating: Float(-1_000), count: batch * length * vocabularySize)
        for row in 0 ..< (batch * length) {
            values[row * vocabularySize + 1] = 1_000
        }
        return MLXArray(values).reshaped(batch, length, vocabularySize)
    }
}
