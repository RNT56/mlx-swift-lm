import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    @Suite
    struct TurboQuantModelCacheTests {

        private func decodeConfig<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
            try JSONDecoder().decode(T.self, from: Data(json.utf8))
        }

        @Test func testGemma4SharedKVModelCreatesTurboQuantCaches() throws {
            let config = try decodeConfig(
                Gemma4TextConfiguration.self,
                """
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 128,
                  "num_hidden_layers": 4,
                  "intermediate_size": 256,
                  "num_attention_heads": 2,
                  "head_dim": 64,
                  "global_head_dim": 64,
                  "num_key_value_heads": 1,
                  "num_global_key_value_heads": 1,
                  "num_kv_shared_layers": 2,
                  "vocab_size": 128,
                  "vocab_size_per_layer_input": 128,
                  "hidden_size_per_layer_input": 0,
                  "sliding_window": 8,
                  "sliding_window_pattern": 2,
                  "layer_types": [
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention"
                  ]
                }
                """)
            let model = Gemma4TextModel(config)
            let caches = model.newCache(
                parameters: GenerateParameters(kvCacheStrategy: .turboQuant))

            #expect(caches.count == 2)
            #expect(caches[0] is RotatingTurboQuantKVCache)
            #expect(caches[1] is TurboQuantKVCache)
            #expect(!(caches[0] is RawOnlyKVCache))
            #expect(!(caches[1] is RawOnlyKVCache))
        }

        @Test func testGemma3nCreatesTurboQuantCaches() {
            let model = Gemma3nTextModel(config: Gemma3nTextConfiguration())
            let caches = model.newCache(
                parameters: GenerateParameters(kvCacheStrategy: .turboQuant))

            #expect(caches.count == 4)
            #expect(caches.contains { $0 is TurboQuantCompressedKVCacheProtocol })
            #expect(!caches.contains { $0 is RawOnlyKVCache })
        }

        @Test func testGLM4MoELiteLatentAttentionCreatesTurboQuantCaches() throws {
            let config = try decodeConfig(
                GLM4MoELiteConfiguration.self,
                """
                {
                  "model_type": "glm4_moe_lite",
                  "vocab_size": 128,
                  "hidden_size": 64,
                  "intermediate_size": 128,
                  "moe_intermediate_size": 64,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "n_shared_experts": 1,
                  "n_routed_experts": 2,
                  "routed_scaling_factor": 1.0,
                  "kv_lora_rank": 64,
                  "qk_rope_head_dim": 64,
                  "qk_nope_head_dim": 32,
                  "v_head_dim": 64,
                  "topk_method": "noaux_tc",
                  "scoring_func": "sigmoid",
                  "norm_topk_prob": true,
                  "n_group": 1,
                  "topk_group": 1,
                  "num_experts_per_tok": 1,
                  "moe_layer_freq": 1,
                  "first_k_dense_replace": 1,
                  "max_position_embeddings": 128,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 1000000.0,
                  "rope_traditional": true,
                  "attention_bias": false,
                  "attention_dropout": 0.0,
                  "partial_rotary_factor": 1.0,
                  "tie_word_embeddings": false,
                  "num_nextn_predict_layers": 1
                }
                """)
            let model = GLM4MoELiteModel(config)
            let caches = model.newCache(
                parameters: GenerateParameters(kvCacheStrategy: .turboQuant))

            #expect(caches.count == 2)
            #expect(caches.allSatisfy { $0 is TurboQuantKVCache })
            #expect(!caches.contains { $0 is RawOnlyKVCache })
        }
    }
}
