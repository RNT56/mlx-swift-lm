import Foundation
import MLX
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
}
