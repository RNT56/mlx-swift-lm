import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct Worker8ModelSafetyTests {
    @Test func gemma4AssistantReportsUnsupportedDirectUse() throws {
        let config = try makeGemma4Config(assistant: true)
        let assistant = Gemma4AssistantModel(config)
        let input = LMInput(text: .init(tokens: MLXArray([1, 2] as [Int32]).reshaped([1, 2])))

        let direct = assistant(input.text.tokens, cache: nil)
        #expect(direct.shape == [1, 2, 0])

        do {
            _ = try assistant.prepare(
                input, cache: assistant.newCache(parameters: nil), windowSize: nil)
            Issue.record("Gemma4 assistant prepare should require dual-model MTP orchestration")
        } catch let error as Gemma4AssistantError {
            #expect(error == .requiresDualModelMTP)
        } catch {
            Issue.record("Unexpected Gemma4 assistant prepare error: \(error)")
        }

        do {
            _ = try assistant.callMTPChecked(input.text.tokens, cache: nil, mtpCaches: nil)
            Issue.record("Gemma4 assistant MTP should require a verifier model")
        } catch let error as Gemma4AssistantError {
            #expect(error == .requiresDualModelMTP)
        } catch {
            Issue.record("Unexpected Gemma4 assistant MTP error: \(error)")
        }

        do {
            try assistant.validateMTPOrchestration(mainModel: DummyBaseLanguageModel())
            Issue.record("Gemma4 assistant should reject non-Gemma4 verifier models")
        } catch Gemma4AssistantError.incompatibleMainModel(let type) {
            #expect(type.contains("DummyBaseLanguageModel"))
        } catch {
            Issue.record("Unexpected Gemma4 assistant validation error: \(error)")
        }
    }

    @Test func gemma4AssistantRunsWithExplicitDualModelMTP() throws {
        let mainConfig = try makeGemma4Config(assistant: false)
        let assistantConfig = try makeGemma4Config(assistant: true)
        let mainModel = Gemma4TextModel(mainConfig.textConfig)
        let assistant = Gemma4AssistantModel(assistantConfig)
        assistant.mainModelRef = mainModel

        let tokens = MLXArray([1, 2] as [Int32]).reshaped([1, 2])
        let cache = mainModel.newCache(parameters: nil)
        let logits = try assistant.callMTPChecked(tokens, cache: cache, mtpCaches: nil)

        #expect(logits.count == 2)
        #expect(logits[0].shape == [1, 2, mainConfig.textConfig.vocabSize])
        #expect(logits[1].shape == [1, 1, assistantConfig.textConfig.vocabSize])
    }

    @Test func deepseekV4DecodesRoutingAndCompressionMetadata() throws {
        let config = try makeDeepseekV4Config()

        #expect(config.layerTypes == [.sliding, .compressedSparse, .heavilyCompressed])
        #expect(config.mlpLayerTypes == [.hashMoE, .moe, .hashMoE])
        #expect(config.compressRateCSA == 2)
        #expect(config.compressRateHCA == 4)
        #expect(config.indexHeads == 2)
        #expect(config.indexHeadDim == 2)
        #expect(config.indexTopK == 2)
        #expect(config.tieWordEmbeddings)
    }

    @Test func deepseekV4SanitizePreservesCanonicalRuntimeWeights() throws {
        let previous = MTPConfig.retainMTPWeights
        MTPConfig.retainMTPWeights = false
        defer { MTPConfig.retainMTPWeights = previous }

        let model = DeepseekV4Model(try makeDeepseekV4Config())
        let weights: [String: MLXArray] = [
            "model.layers.0.attn.compressor.wkv.weight": MLXArray.zeros([8, 8]),
            "model.layers.0.attn.compressor.indexer.wq_b.weight": MLXArray.zeros([8, 8]),
            "model.layers.0.ffn.gate.tid2eid": MLXArray.zeros([8, 1], dtype: .int32),
            "model.layers.0.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.layers.2.attn.compressor.wkv.weight": MLXArray.zeros([8, 8]),
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.layers.0.attn.compressor.wkv.weight"] != nil)
        #expect(sanitized["model.layers.0.attn.compressor.indexer.wq_b.weight"] != nil)
        #expect(sanitized["model.layers.0.ffn.gate.tid2eid"] != nil)
        #expect(sanitized["model.layers.0.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.layers.2.attn.compressor.wkv.weight"] == nil)
    }

    @Test func deepseekV4CacheAndMTPCacheUseLayerSemantics() throws {
        let previous = MTPConfig.retainMTPWeights
        MTPConfig.retainMTPWeights = true
        defer { MTPConfig.retainMTPWeights = previous }

        let model = DeepseekV4Model(try makeDeepseekV4Config())
        let caches = model.newCache(parameters: nil)
        let mtpCaches = model.makeMTPCaches(parameters: nil)

        #expect(caches.count == 2)
        #expect(caches[0] is DeepseekV4KVCache)
        #expect(caches[1] is DeepseekV4KVCache)
        #expect(String(reflecting: caches[0]).contains("sliding_attention"))
        #expect(String(reflecting: caches[1]).contains("compressed_sparse_attention"))
        #expect(mtpCaches.count == 1)
        #expect(mtpCaches[0].count == 1)
        #expect(String(reflecting: mtpCaches[0][0]).contains("heavily_compressed_attention"))
    }

    @Test func deepseekV4HashRoutingUsesTid2Eid() throws {
        let config = try makeDeepseekV4Config(numHiddenLayers: 1, numNextnPredictLayers: 0)
        let gate = DeepseekV4Gate(config: config, isHash: true)
        let tokenToExpert = MLXArray(
            [
                0, 1,
                2, 3,
                1, 2,
                3, 0,
                0, 2,
                1, 3,
                2, 0,
                3, 1,
            ] as [Int32]
        ).reshaped([8, 2])
        try gate.update(
            parameters: ModuleParameters.unflattened([
                "tid2eid": tokenToExpert
            ]),
            verify: [])

        let hidden = MLXArray.zeros([1, 2, config.hiddenSize])
        let inputIds = MLXArray([1, 3] as [Int32]).reshaped([1, 2])
        let (indices, weights) = gate(hidden, inputIds: inputIds)

        #expect(indices.asArray(Int32.self) == [2, 3, 3, 0])
        #expect(weights.shape == [1, 2, 2])
    }
}

private final class DummyBaseLanguageModel: Module, BaseLanguageModel {}

private func makeGemma4Config(assistant: Bool) throws -> Gemma4Configuration {
    let sharedLayers = assistant ? 2 : 0
    let json = """
        {
          "model_type": "gemma4",
          "vocab_size": 32,
          "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "head_dim": 4,
            "global_head_dim": 4,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": \(sharedLayers),
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 4,
            "sliding_window_pattern": 2,
            "final_logit_softcapping": null,
            "layer_types": [
              "sliding_attention",
              "full_attention"
            ]
          }
        }
        """
    return try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))
}

private func makeDeepseekV4Config(
    numHiddenLayers: Int = 3,
    numNextnPredictLayers: Int = 1
) throws -> DeepseekV4Configuration {
    let layerTypes = [
        "\"sliding_attention\"",
        "\"compressed_sparse_attention\"",
        "\"heavily_compressed_attention\"",
    ].prefix(numHiddenLayers).joined(separator: ", ")
    let mlpTypes = [
        "\"hash_moe\"",
        "\"moe\"",
        "\"hash_moe\"",
    ].prefix(numHiddenLayers).joined(separator: ", ")
    let json = """
        {
          "vocab_size": 8,
          "hidden_size": 4,
          "moe_intermediate_size": 4,
          "num_hidden_layers": \(numHiddenLayers),
          "num_attention_heads": 2,
          "head_dim": 4,
          "q_lora_rank": 4,
          "qk_rope_head_dim": 2,
          "rms_norm_eps": 1e-6,
          "o_groups": 1,
          "o_lora_rank": 4,
          "sliding_window": 4,
          "compress_rates": {
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4
          },
          "compress_rope_theta": 160000,
          "layer_types": [\(layerTypes)],
          "n_routed_experts": 4,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "scoring_func": "sqrtsoftplus",
          "routed_scaling_factor": 1.0,
          "swiglu_limit": 10,
          "num_hash_layers": 1,
          "num_nextn_predict_layers": \(numNextnPredictLayers),
          "norm_topk_prob": true,
          "mlp_layer_types": [\(mlpTypes)],
          "hc_mult": 1,
          "hc_sinkhorn_iters": 1,
          "hc_eps": 1e-6,
          "rope_theta": 10000,
          "max_position_embeddings": 64,
          "index_n_heads": 2,
          "index_head_dim": 2,
          "index_topk": 2,
          "tie_word_embeddings": true
        }
        """
    return try JSONDecoder().decode(DeepseekV4Configuration.self, from: Data(json.utf8))
}
