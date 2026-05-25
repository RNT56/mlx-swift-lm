import Foundation
import MLX
import MLXLLM
import MLXNN
import Testing

@testable import MLXLMCommon

struct TurboQuantRuntimeFailureTests {
    @Test func mapsAttentionStateErrorsLosslessly() {
        let compressed = TurboQuantRuntimeFailure(
            attentionStateError: .compressedAttentionUnavailable("kernel self-test failed")
        )
        #expect(compressed == .compressedAttentionUnavailable("kernel self-test failed"))
        #expect(compressed.pinesFailureKind == .turboQuantPathUnavailable)

        let fallback = TurboQuantRuntimeFailure(
            attentionStateError: .noSemanticallyCorrectFallback("decode disabled by policy")
        )
        #expect(fallback == .noBudgetedFallback("decode disabled by policy"))
        #expect(fallback.pinesFailureKind == .turboQuantFallbackUnavailable)
    }

    @Test func classifiesCacheAndBudgetFailuresWithoutDroppingDetails() {
        let budget = TurboQuantRuntimeFailure(
            TestError.message(
                "TurboQuant compressed cache resident bytes 4096 exceed admitted budget 1024"
            )
        )
        #expect(
            budget
                == .fallbackBudgetExceeded(
                    "TurboQuant compressed cache resident bytes 4096 exceed admitted budget 1024"
                )
        )
        #expect(budget.pinesFailureKind == .fallbackBudgetExceeded)

        let layout = TurboQuantRuntimeFailure(
            TestError.message("TurboQuant compressed cache storage invalid: ring offset outside capacity")
        )
        #expect(
            layout
                == .cacheLayoutInvalid(
                    "TurboQuant compressed cache storage invalid: ring offset outside capacity"
                )
        )
        #expect(layout.pinesFailureKind == .cacheLayoutInvalid)
    }

    @Test func mapsUnsupportedRuntimeFailuresForPines() {
        #expect(
            TurboQuantRuntimeFailure(TestError.message("unsupported attention mask rank"))
                == .unsupportedAttentionMask("unsupported attention mask rank")
        )
        #expect(
            TurboQuantRuntimeFailure(TestError.message("unsupported tensor dtype int64"))
                == .unsupportedTensorDType("unsupported tensor dtype int64")
        )
        #expect(
            TurboQuantRuntimeFailure(TestError.message("unsupported head dimension 768"))
                == .unsupportedAttentionShape("unsupported head dimension 768")
        )
    }

    @Test func runtimeFailureCodableRoundTripKeepsCaseAndMessage() throws {
        let original = TurboQuantRuntimeFailure.decodedFallbackUnavailable(
            "decode compressed K/V fallback failed"
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(TurboQuantRuntimeFailure.self, from: data)
        #expect(decoded == original)
        #expect(decoded.pinesFailureKind == .turboQuantFallbackUnavailable)
    }

    @Test func turboQuantGenerationRejectsNonThrowingModelsBeforeRuntimeAttention() throws {
        let model = NonThrowingTinyLanguageModel()
        let input = LMInput(text: .init(tokens: MLXArray([1, 2] as [Int32]).reshaped([1, 2])))
        let parameters = GenerateParameters(
            maxTokens: 1,
            kvCacheStrategy: .turboQuant,
            turboQuantAdmissionPolicy: .automatic
        )

        do {
            _ = try TokenIterator(input: input, model: model, parameters: parameters)
            Issue.record("TurboQuant generation should reject non-throwing models before runtime attention")
        } catch let error as TurboQuantGenerationError {
            guard case .modelRequiresThrowingAttention(let modelName) = error else {
                Issue.record("Unexpected TurboQuant generation error: \(error)")
                return
            }
            #expect(modelName.contains("NonThrowingTinyLanguageModel"))
            #expect(!model.wasCalled)
        }
    }

    @Test func profileBackedFamiliesExposeThrowingTurboQuantRuntime() throws {
        try Self.assertTurboQuantRuntimeReady(
            LlamaModel(
                LlamaConfiguration(
                    hiddenSize: 64,
                    hiddenLayers: 2,
                    intermediateSize: 128,
                    attentionHeads: 4,
                    headDimensions: 16,
                    rmsNormEps: 1e-5,
                    vocabularySize: 128,
                    kvHeads: 2
                )
            )
        )
        try Self.assertTurboQuantRuntimeReady(
            Gemma3TextModel(
                Gemma3TextConfiguration(
                    modelType: "gemma3",
                    hiddenSize: 64,
                    hiddenLayers: 2,
                    intermediateSize: 128,
                    attentionHeads: 4,
                    headDim: 16,
                    rmsNormEps: 1e-5,
                    vocabularySize: 128,
                    kvHeads: 2,
                    ropeTheta: 1_000_000,
                    ropeLocalBaseFreq: 10_000,
                    ropeTraditional: false,
                    queryPreAttnScalar: 256,
                    slidingWindow: 8,
                    slidingWindowPattern: 2,
                    maxPositionEmbeddings: 128
                )
            )
        )
        try Self.assertTurboQuantRuntimeReady(Qwen3Model(Self.qwen3Configuration()))
        try Self.assertTurboQuantRuntimeReady(Qwen3MoEModel(Self.qwen3MoEConfiguration()))
        try Self.assertTurboQuantRuntimeReady(Qwen35TextModel(Self.qwen35TextConfiguration()))
    }

    private enum TestError: Error, CustomStringConvertible {
        case message(String)

        var description: String {
            switch self {
            case .message(let message):
                message
            }
        }
    }

    private final class NonThrowingTinyLanguageModel: Module, LanguageModel {
        private(set) var wasCalled = false

        func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
            wasCalled = true
            return .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            wasCalled = true
            return MLXArray.zeros([1, inputs.dim(-1), 8])
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            wasCalled = true
            return []
        }
    }

    private static func assertTurboQuantRuntimeReady(_ model: any LanguageModel) throws {
        #expect(model is any ThrowingLanguageModel)
        let parameters = GenerateParameters(
            maxTokens: 1,
            maxKVSize: 8,
            kvCacheStrategy: .turboQuant,
            turboQuantAdmissionPolicy: .automatic
        )
        let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 1)
        #expect(resolved.kvCacheStrategy == .turboQuant)
        #expect(!model.newCache(parameters: resolved).isEmpty)
    }

    private static func qwen3Configuration() throws -> Qwen3Configuration {
        let json = """
        {
          "model_type": "qwen3",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "rope_theta": 1000000,
          "max_position_embeddings": 128,
          "tie_word_embeddings": true
        }
        """
        return try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    private static func qwen3MoEConfiguration() throws -> Qwen3MoEConfiguration {
        let json = """
        {
          "model_type": "qwen3_moe",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "moe_intermediate_size": 32,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "decoder_sparse_step": 1,
          "mlp_only_layers": [],
          "norm_topk_prob": false,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "rope_theta": 1000000,
          "max_position_embeddings": 128,
          "tie_word_embeddings": true
        }
        """
        return try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: Data(json.utf8))
    }

    private static func qwen35TextConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
        {
          "model_type": "qwen3_5",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "linear_num_value_heads": 1,
          "linear_num_key_heads": 1,
          "linear_key_head_dim": 8,
          "linear_value_head_dim": 8,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "rope_theta": 10000.0,
          "max_position_embeddings": 128,
          "full_attention_interval": 2,
          "num_nextn_predict_layers": 0,
          "tie_word_embeddings": true
        }
        """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }
}
