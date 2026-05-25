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
        try Self.assertTurboQuantRuntimeReady(Qwen35Model(Self.qwen35Configuration()))
        try Self.assertTurboQuantRuntimeReady(Gemma4Model(Self.gemma4Configuration()))
        try Self.assertTurboQuantRuntimeReady(Mistral3TextModel(Self.mistral3TextConfiguration()))
        try Self.assertTurboQuantRuntimeReady(Mistral4Model(Self.mistral4Configuration()))
        try Self.assertTurboQuantRuntimeReady(Qwen2Model(Self.qwen2Configuration()))
        try Self.assertTurboQuantRuntimeReady(PhiModel(Self.phiConfiguration()))
        try Self.assertTurboQuantRuntimeReady(Phi3Model(Self.phi3Configuration()))
        try Self.assertTurboQuantRuntimeReady(SmolLM3Model(Self.smolLM3Configuration()))
        try Self.assertTurboQuantRuntimeReady(GraniteModel(Self.graniteConfiguration()))
        try Self.assertTurboQuantRuntimeReady(Exaone4Model(Self.exaone4Configuration()))
        try Self.assertTurboQuantRuntimeReady(LFM2Model(Self.lfm2Configuration()))
        try Self.assertTurboQuantRuntimeReady(GLM4MoELiteModel(Self.glm4MoELiteConfiguration()))
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
        let caches = model.newCache(parameters: resolved)
        #expect(!caches.isEmpty, "Expected \(type(of: model)) to create runtime caches")
    }

    private static func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
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
        return try decode(Qwen3Configuration.self, json)
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
        return try decode(Qwen3MoEConfiguration.self, json)
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
        return try decode(Qwen35TextConfiguration.self, json)
    }

    private static func qwen35Configuration() throws -> Qwen35Configuration {
        let json = """
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
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
          },
          "vocab_size": 128
        }
        """
        return try decode(Qwen35Configuration.self, json)
    }

    private static func gemma4Configuration() throws -> Gemma4Configuration {
        let json = """
        {
          "model_type": "gemma4",
          "vocab_size": 128,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "global_head_dim": 16,
            "global_partial_rotary_factor": 1.0,
            "rms_norm_eps": 1e-6,
            "vocab_size": 128,
            "vocab_size_per_layer_input": 128,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 2,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 64,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 128,
            "attention_k_eq_v": false,
            "use_double_wide_mlp": false,
            "layer_types": ["sliding_attention", "full_attention"],
            "tie_word_embeddings": true
          }
        }
        """
        return try decode(Gemma4Configuration.self, json)
    }

    private static func mistral3TextConfiguration() throws -> Mistral3TextConfiguration {
        let json = """
        {
          "model_type": "ministral3",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "max_position_embeddings": 128,
          "rope_theta": 10000.0,
          "layer_types": ["full_attention", "sliding_attention"],
          "sliding_window": 8,
          "tie_word_embeddings": true
        }
        """
        return try decode(Mistral3TextConfiguration.self, json)
    }

    private static func mistral4Configuration() throws -> Mistral4Configuration {
        let json = """
        {
          "hidden_size": 64,
          "moe_intermediate_size": 32,
          "intermediate_size": 64,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "n_routed_experts": 2,
          "n_shared_experts": 1,
          "num_experts_per_tok": 1,
          "first_k_dense_replace": 0,
          "routed_scaling_factor": 1.0,
          "norm_topk_prob": false,
          "kv_lora_rank": 8,
          "q_lora_rank": 16,
          "qk_rope_head_dim": 8,
          "qk_nope_head_dim": 8,
          "v_head_dim": 16,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "vocab_size": 128,
          "tie_word_embeddings": true,
          "max_position_embeddings": 128
        }
        """
        return try decode(Mistral4Configuration.self, json)
    }

    private static func qwen2Configuration() throws -> Qwen2Configuration {
        let json = """
        {
          "model_type": "qwen2",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-6,
          "vocab_size": 128,
          "rope_theta": 1000000,
          "tie_word_embeddings": true
        }
        """
        return try decode(Qwen2Configuration.self, json)
    }

    private static func phiConfiguration() throws -> PhiConfiguration {
        let json = """
        {
          "max_position_embeddings": 128,
          "vocab_size": 128,
          "hidden_size": 64,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "num_key_value_heads": 2,
          "partial_rotary_factor": 1.0,
          "intermediate_size": 128,
          "layer_norm_eps": 1e-5,
          "rope_theta": 10000.0
        }
        """
        return try decode(PhiConfiguration.self, json)
    }

    private static func phi3Configuration() throws -> Phi3Configuration {
        let json = """
        {
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "rope_theta": 10000.0,
          "partial_rotary_factor": 1.0,
          "max_position_embeddings": 128,
          "original_max_position_embeddings": 128,
          "tie_word_embeddings": true
        }
        """
        return try decode(Phi3Configuration.self, json)
    }

    private static func smolLM3Configuration() -> SmolLM3Configuration {
        SmolLM3Configuration(
            hiddenSize: 64,
            hiddenLayers: 2,
            intermediateSize: 128,
            attentionHeads: 4,
            headDimensions: 16,
            rmsNormEps: 1e-5,
            vocabularySize: 128,
            kvHeads: 2,
            maxPositionEmbeddings: 128
        )
    }

    private static func graniteConfiguration() throws -> GraniteConfiguration {
        let json = """
        {
          "model_type": "granite",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "logits_scaling": 1.0,
          "attention_multiplier": 1.0,
          "embedding_multiplier": 1.0,
          "residual_multiplier": 1.0,
          "max_position_embeddings": 128,
          "attention_bias": false,
          "mlp_bias": false,
          "rope_theta": 10000.0,
          "tie_word_embeddings": true
        }
        """
        return try decode(GraniteConfiguration.self, json)
    }

    private static func exaone4Configuration() throws -> Exaone4Configuration {
        let json = """
        {
          "model_type": "exaone4",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "max_position_embeddings": 128,
          "rope_theta": 10000.0,
          "head_dim": 16,
          "tie_word_embeddings": true,
          "sliding_window": 8,
          "sliding_window_pattern": "LG"
        }
        """
        return try decode(Exaone4Configuration.self, json)
    }

    private static func lfm2Configuration() throws -> LFM2Configuration {
        let json = """
        {
          "model_type": "lfm2",
          "vocab_size": 128,
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "max_position_embeddings": 128,
          "norm_eps": 1e-5,
          "conv_bias": false,
          "conv_L_cache": 3,
          "block_dim": 64,
          "block_ff_dim": 128,
          "block_multiple_of": 1,
          "block_ffn_dim_multiplier": 1.0,
          "block_auto_adjust_ff_dim": false,
          "layer_types": ["full_attention", "conv"],
          "rope_theta": 10000.0
        }
        """
        return try decode(LFM2Configuration.self, json)
    }

    private static func glm4MoELiteConfiguration() throws -> GLM4MoELiteConfiguration {
        let json = """
        {
          "model_type": "glm4_moe_lite",
          "vocab_size": 128,
          "hidden_size": 64,
          "intermediate_size": 128,
          "moe_intermediate_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 4,
          "n_shared_experts": 1,
          "n_routed_experts": 2,
          "routed_scaling_factor": 1.0,
          "kv_lora_rank": 8,
          "q_lora_rank": 16,
          "qk_rope_head_dim": 8,
          "qk_nope_head_dim": 8,
          "v_head_dim": 16,
          "topk_method": "noaux_tc",
          "scoring_func": "sigmoid",
          "norm_topk_prob": false,
          "n_group": 1,
          "topk_group": 1,
          "num_experts_per_tok": 1,
          "moe_layer_freq": 1,
          "first_k_dense_replace": 0,
          "max_position_embeddings": 128,
          "rms_norm_eps": 1e-6,
          "rope_theta": 10000.0,
          "rope_traditional": false,
          "attention_bias": false,
          "attention_dropout": 0.0,
          "partial_rotary_factor": 1.0,
          "tie_word_embeddings": false,
          "num_nextn_predict_layers": 0
        }
        """
        return try decode(GLM4MoELiteConfiguration.self, json)
    }
}
