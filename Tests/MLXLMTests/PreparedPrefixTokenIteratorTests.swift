import MLX
import MLXLMCommon
import MLXNN
import Testing

struct PreparedPrefixTokenIteratorTests {
    @Test func preparedPrefixEvaluatesOnlyUncachedSuffix() throws {
        let model = PreparedPrefixModel()
        let cache: [KVCache] = [KVCacheSimple()]
        seed(cache: cache, tokenCount: 3)
        let input = LMInput(tokens: MLXArray(Int32(0) ..< Int32(5)))

        _ = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            cachedPrefixTokenCount: 3,
            parameters: GenerateParameters(maxTokens: 1)
        )

        #expect(model.preparedTokenCounts == [2])
        #expect(model.stepTokenCounts == [2])
    }

    @Test func completePreparedPrefixTrimsOneTokenAndEvaluatesTail() throws {
        let model = PreparedPrefixModel()
        let cache: [KVCache] = [KVCacheSimple()]
        seed(cache: cache, tokenCount: 5)
        let input = LMInput(tokens: MLXArray(Int32(0) ..< Int32(5)))

        _ = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            cachedPrefixTokenCount: 5,
            parameters: GenerateParameters(maxTokens: 1)
        )

        #expect(model.preparedTokenCounts == [1])
        #expect(model.stepTokenCounts == [1])
    }

    @Test func preparedPrefixClampsToCacheOffset() throws {
        let model = PreparedPrefixModel()
        let cache: [KVCache] = [KVCacheSimple()]
        seed(cache: cache, tokenCount: 1)
        let input = LMInput(tokens: MLXArray(Int32(0) ..< Int32(5)))

        _ = try TokenIterator(
            input: input,
            model: model,
            cache: cache,
            cachedPrefixTokenCount: 3,
            parameters: GenerateParameters(maxTokens: 1)
        )

        #expect(model.preparedTokenCounts == [4])
        #expect(model.stepTokenCounts == [4])
    }

    private func seed(cache: [KVCache], tokenCount: Int) {
        guard tokenCount > 0 else { return }
        let keys = MLXArray.ones([1, 1, tokenCount, 1], dtype: .float32)
        for entry in cache {
            _ = entry.update(keys: keys, values: keys)
        }
    }
}

private final class PreparedPrefixModel: Module, LanguageModel {
    var preparedTokenCounts: [Int] = []
    var stepTokenCounts: [Int] = []

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        preparedTokenCounts.append(input.text.tokens.size)
        return .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        stepTokenCounts.append(input.tokens.size)
        if input.tokens.size > 0 {
            let keys = MLXArray.ones([1, 1, input.tokens.size, 1], dtype: .float32)
            for entry in cache ?? [] {
                _ = entry.update(keys: keys, values: keys)
            }
        }
        return LMOutput(logits: logits(length: max(1, input.tokens.size)))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(length: max(1, inputs.size))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    private func logits(length: Int) -> MLXArray {
        var values = Array(repeating: Float(-1_000), count: length * 8)
        for row in 0 ..< length {
            values[row * 8] = 1_000
        }
        return MLXArray(values, [1, length, 8])
    }
}
