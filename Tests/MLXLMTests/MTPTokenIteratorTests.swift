import MLX
import MLXLMCommon
import MLXNN
import Testing

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
