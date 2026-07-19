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

    @Test func mapsTypedMLXTurboQuantErrorsForPines() {
        let backend = TurboQuantRuntimeFailure(
            TurboQuantError.unsupportedBackend(
                .metalPolarQJL,
                "kernel self-test failed"
            )
        )
        #expect(backend == .unsupportedBackend("metalPolarQJL: kernel self-test failed"))
        #expect(backend.pinesFailureKind == .turboQuantPathUnavailable)

        let dtype = TurboQuantRuntimeFailure(
            TurboQuantError.invalidMetalConfiguration("unsupported tensor dtype int64")
        )
        #expect(dtype == .unsupportedTensorDType("Invalid TurboQuant Metal configuration: unsupported tensor dtype int64"))
        #expect(dtype.pinesFailureKind == .unsupportedTensorDType)
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

    @Test func qwen35HybridBudgetLayersExcludeLinearNativeStateCaches() throws {
        let model = Qwen35TextModel(try Self.qwen35TextConfiguration())
        #expect(model.kvHeads == [0, 2])

        let parameters = GenerateParameters(
            maxTokens: 1,
            maxKVSize: 8,
            kvCacheStrategy: .turboQuant,
            turboQuantAdmissionPolicy: .automatic
        )
        let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 1)
        let caches = model.newCache(parameters: resolved)

        #expect(caches.count == 2)
        #expect(caches[0] is MambaCache)
        #expect(caches[1] is TurboQuantCompressedKVCacheProtocol)
    }

    @Test func qwen35HybridTokenIteratorMaterializesNativeStateWithTurboQuantAttention() throws {
        let model = Qwen35TextModel(try Self.qwen35TextConfiguration())
        let parameters = GenerateParameters(
            maxTokens: 3,
            maxKVSize: 16,
            kvCacheStrategy: .turboQuant,
            turboQuantAdmissionPolicy: .automatic,
            temperature: 0
        )
        let resolved = try parameters.resolvedForTurboQuantRuntime(layerCount: 1)
        let cache = model.newCache(parameters: resolved)
        #expect(cache[0] is MambaCache)
        #expect(cache[1] is TurboQuantCompressedKVCacheProtocol)

        let prompt = MLXArray([1, 2] as [Int32])
        var iterator = try TokenIterator(
            input: LMInput(text: .init(tokens: prompt)),
            model: model,
            cache: cache,
            parameters: parameters
        )

        var emitted = [Int]()
        while let token = iterator.next() {
            emitted.append(token)
        }
        if let error = iterator.lastRuntimeError {
            Issue.record("Unexpected Qwen3.5 hybrid TurboQuant iterator error: \(error)")
        }

        #expect(emitted.count == 3)
        let recurrent = try #require(cache[0] as? MambaCache)
        let recurrentState = recurrent.state
        #expect(recurrentState.count == 2)
        eval(recurrentState)
        #expect(recurrentState[0].shape == [1, 3, 14336])
        #expect(recurrentState[1].shape == [1, 64, 128, 192])
    }

    @Test func exportedCapabilityRegistryMatchesThrowingRuntimeCoverage() async throws {
        let registeredModelTypes = Set(await LLMTypeRegistry.shared.registeredModelTypes)
        let textConfigOnlyModelTypes: Set<String> = [
            "gemma3n_text",
            "mistral4",
            "qwen3_5_moe_text",
        ]

        for capability in MLXTurboQuantRuntimeCapabilityRegistry.capabilities {
            if capability.supportsThrowingTurboQuantAttention {
                #expect(
                    registeredModelTypes.contains(capability.modelType)
                        || textConfigOnlyModelTypes.contains(capability.modelType),
                    "\(capability.modelType) is exported as TurboQuant-capable but is not registered or documented as text-config-only"
                )
            }
        }

        let supported = MLXTurboQuantRuntimeCapabilityRegistry.throwingTurboQuantAttentionModelTypes
        let expected: Set<String> = [
            "llama", "mistral", "ministral3", "mistral3", "mistral4",
            "gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma3n_text",
            "gemma4", "gemma4_text",
            "qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_text",
            "qwen3_5_moe", "qwen3_5_moe_text", "acereason",
            "phi", "phi3", "granite", "exaone4", "smollm3", "lfm2", "glm4_moe_lite",
        ]
        #expect(supported == expected)
        #expect(!MLXTurboQuantRuntimeCapabilityRegistry.supportsThrowingTurboQuantAttention(modelType: "gemma4_assistant"))
        #expect(!MLXTurboQuantRuntimeCapabilityRegistry.supportsThrowingTurboQuantAttention(modelType: "pixtral"))
    }

    @Test func profileBackedFactoryTypesInstantiateAsThrowingLanguageModels() async throws {
        for fixture in Self.factoryFixtures {
            let capability = MLXTurboQuantRuntimeCapabilityRegistry.capability(for: fixture.capabilityModelType)
            #expect(capability?.supportsThrowingTurboQuantAttention == true)
            let model = try await LLMTypeRegistry.shared.createModel(
                configuration: fixture.configurationData,
                modelType: fixture.registryModelType
            )
            try Self.assertTurboQuantRuntimeReady(model)
        }
    }

    @Test func profileBackedFamiliesHaveTinyThrowingPrefillDecodePath() throws {
        let models: [any LanguageModel] = [
            LlamaModel(try Self.llamaConfiguration()),
            GemmaModel(try Self.gemmaConfiguration()),
            Gemma2Model(try Self.gemma2Configuration()),
            Gemma3TextModel(try Self.gemma3TextConfiguration()),
            Gemma3nTextModel(config: try Self.gemma3nTextConfiguration()),
            Gemma4TextModel(try Self.gemma4TextConfiguration()),
            Qwen2Model(try Self.qwen2Configuration()),
            Qwen3Model(try Self.qwen3Configuration()),
            Qwen3MoEModel(try Self.qwen3MoEConfiguration()),
            Mistral3TextModel(try Self.mistral3TextConfiguration()),
            Mistral4Model(try Self.mistral4Configuration()),
            PhiModel(try Self.phiConfiguration()),
            Phi3Model(try Self.phi3Configuration()),
            SmolLM3Model(Self.smolLM3Configuration()),
            GraniteModel(try Self.graniteConfiguration()),
            Exaone4Model(try Self.exaone4Configuration()),
            LFM2Model(try Self.lfm2Configuration()),
            GLM4MoELiteModel(try Self.glm4MoELiteConfiguration()),
        ]

        for model in models {
            try Self.assertTinyPrefillDecodePath(model)
        }
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

    private static func assertTinyPrefillDecodePath(_ model: any LanguageModel) throws {
        let throwingModel = try #require(model as? any ThrowingLanguageModel)
        let parameters = GenerateParameters(maxTokens: 1, maxKVSize: 8, kvCacheStrategy: .mlxAffine)
        let caches = model.newCache(parameters: parameters)
        #expect(!caches.isEmpty, "Expected \(type(of: model)) to create runtime caches")

        let prefillTokens = MLXArray([1, 2] as [Int32]).reshaped([1, 2])
        let prefill = try throwingModel.callAsFunctionThrowing(prefillTokens, cache: caches)
        eval(prefill)
        #expect(prefill.dim(0) == 1)
        #expect(prefill.dim(1) == 2)

        let decodeTokens = MLXArray([3] as [Int32]).reshaped([1, 1])
        let decode = try throwingModel.callAsFunctionThrowing(decodeTokens, cache: caches)
        eval(decode)
        #expect(decode.dim(0) == 1)
        #expect(decode.dim(1) == 1)
    }

    private struct FactoryFixture {
        var registryModelType: String
        var capabilityModelType: String
        var configurationData: Data
    }

    private static var factoryFixtures: [FactoryFixture] {
        [
            .init(registryModelType: "llama", capabilityModelType: "llama", configurationData: llamaConfigurationJSON),
            .init(registryModelType: "mistral", capabilityModelType: "mistral", configurationData: llamaConfigurationJSON),
            .init(registryModelType: "phi", capabilityModelType: "phi", configurationData: phiConfigurationJSON),
            .init(registryModelType: "phi3", capabilityModelType: "phi3", configurationData: phi3ConfigurationJSON),
            .init(registryModelType: "gemma", capabilityModelType: "gemma", configurationData: gemmaConfigurationJSON),
            .init(registryModelType: "gemma2", capabilityModelType: "gemma2", configurationData: gemma2ConfigurationJSON),
            .init(registryModelType: "gemma3", capabilityModelType: "gemma3", configurationData: gemma3TextConfigurationJSON),
            .init(registryModelType: "gemma3_text", capabilityModelType: "gemma3_text", configurationData: gemma3TextConfigurationJSON),
            .init(registryModelType: "gemma3n", capabilityModelType: "gemma3n", configurationData: gemma3nTextConfigurationJSON),
            .init(registryModelType: "gemma4", capabilityModelType: "gemma4", configurationData: gemma4ConfigurationJSON),
            .init(registryModelType: "gemma4_text", capabilityModelType: "gemma4_text", configurationData: gemma4TextConfigurationJSON),
            .init(registryModelType: "qwen2", capabilityModelType: "qwen2", configurationData: qwen2ConfigurationJSON),
            .init(registryModelType: "qwen3", capabilityModelType: "qwen3", configurationData: qwen3ConfigurationJSON),
            .init(registryModelType: "qwen3_moe", capabilityModelType: "qwen3_moe", configurationData: qwen3MoEConfigurationJSON),
            .init(registryModelType: "qwen3_5", capabilityModelType: "qwen3_5", configurationData: qwen35ConfigurationJSON),
            .init(registryModelType: "qwen3_5_text", capabilityModelType: "qwen3_5_text", configurationData: qwen35TextConfigurationJSON),
            .init(registryModelType: "qwen3_5_moe", capabilityModelType: "qwen3_5_moe", configurationData: qwen35ConfigurationJSON),
            .init(registryModelType: "acereason", capabilityModelType: "acereason", configurationData: qwen2ConfigurationJSON),
            .init(registryModelType: "granite", capabilityModelType: "granite", configurationData: graniteConfigurationJSON),
            .init(registryModelType: "glm4_moe_lite", capabilityModelType: "glm4_moe_lite", configurationData: glm4MoELiteConfigurationJSON),
            .init(registryModelType: "smollm3", capabilityModelType: "smollm3", configurationData: smolLM3ConfigurationJSON),
            .init(registryModelType: "lfm2", capabilityModelType: "lfm2", configurationData: lfm2ConfigurationJSON),
            .init(registryModelType: "exaone4", capabilityModelType: "exaone4", configurationData: exaone4ConfigurationJSON),
            .init(registryModelType: "ministral3", capabilityModelType: "ministral3", configurationData: mistral3TextConfigurationJSON),
            .init(registryModelType: "mistral3", capabilityModelType: "mistral3", configurationData: mistral3TextConfigurationJSON),
            .init(registryModelType: "mistral3", capabilityModelType: "mistral4", configurationData: mistral4WrappedConfigurationJSON),
        ]
    }

    private static func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(type, from: Data(json.utf8))
    }

    private static func data(_ json: String) -> Data {
        Data(json.utf8)
    }

    private static var llamaConfigurationJSON: Data {
        data("""
        {
          "model_type": "llama",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "rope_theta": 10000.0,
          "max_position_embeddings": 128,
          "tie_word_embeddings": true
        }
        """)
    }

    private static var gemmaConfigurationJSON: Data {
        data("""
        {
          "model_type": "gemma",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "rope_theta": 10000.0
        }
        """)
    }

    private static var gemma2ConfigurationJSON: Data {
        data("""
        {
          "model_type": "gemma2",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "rope_theta": 10000.0,
          "attn_logit_softcapping": 50.0,
          "final_logit_softcapping": 30.0,
          "query_pre_attn_scalar": 256.0
        }
        """)
    }

    private static var gemma3TextConfigurationJSON: Data {
        data("""
        {
          "model_type": "gemma3",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "rope_theta": 1000000,
          "rope_local_base_freq": 10000,
          "rope_traditional": false,
          "query_pre_attn_scalar": 256,
          "sliding_window": 8,
          "sliding_window_pattern": 2,
          "max_position_embeddings": 128,
          "tie_word_embeddings": true
        }
        """)
    }

    private static var gemma3nTextConfigurationJSON: Data {
        data("""
        {
          "model_type": "gemma3n",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "num_key_value_heads": 2,
          "num_kv_shared_layers": 0,
          "query_pre_attn_scalar": 256,
          "vocab_size_per_layer_input": 128,
          "sliding_window": 8,
          "max_position_embeddings": 128,
          "rope_local_base_freq": 10000,
          "rope_theta": 1000000,
          "final_logit_softcapping": 30.0,
          "layer_types": ["sliding_attention", "full_attention"],
          "hidden_size_per_layer_input": 64,
          "altup_num_inputs": 1,
          "altup_correct_scale": false,
          "altup_active_idx": 0,
          "laurel_rank": 4,
          "sliding_window_pattern": 2
        }
        """)
    }

    private static var gemma4TextConfigurationJSON: Data {
        data("""
        {
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
        """)
    }

    private static var gemma4ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var qwen2ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var qwen3ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var qwen3MoEConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var qwen35TextConfigurationJSON: Data {
        data("""
        {
          "model_type": "qwen3_5",
          "hidden_size": 64,
          "num_hidden_layers": 2,
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
          "full_attention_interval": 2,
          "num_nextn_predict_layers": 0,
          "tie_word_embeddings": true
        }
        """)
    }

    private static var qwen35ConfigurationJSON: Data {
        data("""
        {
          "model_type": "qwen3_5",
          "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
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
            "full_attention_interval": 2,
            "num_nextn_predict_layers": 0,
            "tie_word_embeddings": true
          },
          "vocab_size": 128
        }
        """)
    }

    private static var mistral3TextConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var mistral4ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var mistral4WrappedConfigurationJSON: Data {
        data("""
        {
          "model_type": "mistral3",
          "text_config": {
            "model_type": "mistral4",
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
        }
        """)
    }

    private static var phiConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var phi3ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var smolLM3ConfigurationJSON: Data {
        data("""
        {
          "model_type": "smollm3",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-5,
          "vocab_size": 128,
          "max_position_embeddings": 128
        }
        """)
    }

    private static var graniteConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var exaone4ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var lfm2ConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static var glm4MoELiteConfigurationJSON: Data {
        data("""
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
        """)
    }

    private static func llamaConfiguration() throws -> LlamaConfiguration {
        try JSONDecoder().decode(LlamaConfiguration.self, from: llamaConfigurationJSON)
    }

    private static func gemmaConfiguration() throws -> GemmaConfiguration {
        try JSONDecoder().decode(GemmaConfiguration.self, from: gemmaConfigurationJSON)
    }

    private static func gemma2Configuration() throws -> Gemma2Configuration {
        try JSONDecoder().decode(Gemma2Configuration.self, from: gemma2ConfigurationJSON)
    }

    private static func gemma3TextConfiguration() throws -> Gemma3TextConfiguration {
        try JSONDecoder().decode(Gemma3TextConfiguration.self, from: gemma3TextConfigurationJSON)
    }

    private static func gemma3nTextConfiguration() throws -> Gemma3nTextConfiguration {
        try JSONDecoder().decode(Gemma3nTextConfiguration.self, from: gemma3nTextConfigurationJSON)
    }

    private static func gemma4TextConfiguration() throws -> Gemma4TextConfiguration {
        try JSONDecoder().decode(Gemma4TextConfiguration.self, from: gemma4TextConfigurationJSON)
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
          "linear_num_value_heads": 64,
          "linear_num_key_heads": 16,
          "linear_key_head_dim": 192,
          "linear_value_head_dim": 128,
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
            "linear_num_value_heads": 64,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 192,
            "linear_value_head_dim": 128,
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
