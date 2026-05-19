import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

@Suite(.serialized)
struct MTPTokenIteratorTests {
    @Test func deterministicMTPFullyAcceptsDraftAndEmitsVerifierToken() throws {
        let model = FakeMTPModel(
            scriptedCalls: [
                [[10], [20]],
                [[20, 21], [22]],
            ])
        var iterator = try makeIterator(model: model, numMTPTokens: 1, maxTokens: 3)

        #expect(iterator.next() == 10)
        #expect(iterator.next() == 20)
        #expect(iterator.next() == 21)
        #expect(iterator.acceptedDraftTokens == 1)
        #expect(iterator.totalDraftTokens == 1)
    }

    @Test func qwen35DecodesAndAllocatesMTPHeadsWhenRetained() throws {
        let previous = MTPConfig.retainMTPWeights
        MTPConfig.retainMTPWeights = true
        defer { MTPConfig.retainMTPWeights = previous }

        let config = try makeQwen35TextConfig(numMTPLayers: 2)
        let model = Qwen35TextModel(config)

        #expect(config.numNextnPredictLayers == 2)
        #expect(model.mtp.count == 2)
        #expect((model as any LanguageModel) is any MTPLanguageModel)
    }

    @Test func qwen35DropsMTPHeadsAndWeightsByDefault() throws {
        let previous = MTPConfig.retainMTPWeights
        MTPConfig.retainMTPWeights = false
        defer { MTPConfig.retainMTPWeights = previous }

        let config = try makeQwen35TextConfig(numMTPLayers: 1)
        let model = Qwen35TextModel(config)
        let weights = [
            "model.embed_tokens.weight": MLXArray.zeros([16, 8]),
            "mtp.fc.weight": MLXArray.zeros([8, 16]),
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(model.mtp.isEmpty)
        #expect(sanitized.keys.contains { $0.contains("mtp.") } == false)
    }

    @Test func qwen35RetainedMTPWeightsAreIndexedForModuleLoading() throws {
        let previous = MTPConfig.retainMTPWeights
        MTPConfig.retainMTPWeights = true
        defer { MTPConfig.retainMTPWeights = previous }

        let config = try makeQwen35TextConfig(numMTPLayers: 1)
        let model = Qwen35TextModel(config)
        let weights = [
            "mtp.fc.weight": MLXArray.zeros([8, 16])
        ]
        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["mtp.0.fc.weight"] != nil)
    }

    @Test func deterministicMTPRejectsDraftAndFallsBackToVerifierToken() throws {
        let model = FakeMTPModel(
            scriptedCalls: [
                [[10], [20]],
                [[22, 23]],
            ])
        var iterator = try makeIterator(model: model, numMTPTokens: 1, maxTokens: 2)

        #expect(iterator.next() == 10)
        #expect(iterator.next() == 22)
        #expect(iterator.acceptedDraftTokens == 0)
        #expect(iterator.totalDraftTokens == 1)
    }

    @Test func deterministicMTPPartiallyAcceptsMultiTokenDraft() throws {
        let model = FakeMTPModel(
            scriptedCalls: [
                [[10], [20], [30]],
                [[20, 31, 32]],
            ])
        var iterator = try makeIterator(model: model, numMTPTokens: 2, maxTokens: 3)

        #expect(iterator.next() == 10)
        #expect(iterator.next() == 20)
        #expect(iterator.next() == 31)
        #expect(iterator.acceptedDraftTokens == 1)
        #expect(iterator.totalDraftTokens == 2)
    }

    private func makeIterator(
        model: FakeMTPModel,
        numMTPTokens: Int,
        maxTokens: Int
    ) throws -> MTPTokenIterator {
        try MTPTokenIterator(
            input: LMInput(text: .init(tokens: MLXArray([1] as [Int32]))),
            model: model,
            parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
            numMTPTokens: numMTPTokens)
    }
}

private func makeQwen35TextConfig(
    numMTPLayers: Int = 0,
    numHiddenLayers: Int = 4,
    hiddenSize: Int = 8,
    vocabSize: Int = 16
) throws -> Qwen35TextConfiguration {
    let json = """
        {
            "model_type": "qwen3_5",
            "hidden_size": \(hiddenSize),
            "num_hidden_layers": \(numHiddenLayers),
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "linear_num_value_heads": 1,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": \(vocabSize),
            "rope_theta": 10000.0,
            "max_position_embeddings": 64,
            "full_attention_interval": 2,
            "num_nextn_predict_layers": \(numMTPLayers)
        }
        """
    return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
}

private final class FakeMTPModel: Module, MTPLanguageModel {
    private let vocabSize = 64
    private var scriptedCalls: [[[Int]]]
    private var callIndex = 0

    init(scriptedCalls: [[[Int]]]) {
        self.scriptedCalls = scriptedCalls
        super.init()
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(for: [0])
    }

    func callMTP(_ inputs: MLXArray, cache: [KVCache]?, mtpCaches: [[KVCache]]?) -> [MLXArray] {
        precondition(callIndex < scriptedCalls.count, "Unexpected MTP call")
        defer { callIndex += 1 }
        return scriptedCalls[callIndex].map(logits(for:))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    private func logits(for tokens: [Int]) -> MLXArray {
        var values = Array(repeating: Float(-1_000), count: tokens.count * vocabSize)
        for (row, token) in tokens.enumerated() {
            values[row * vocabSize + token] = 1_000
        }
        return MLXArray(values).reshaped([1, tokens.count, vocabSize])
    }
}
