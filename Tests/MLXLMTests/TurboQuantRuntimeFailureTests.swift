import Foundation
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

    private enum TestError: Error, CustomStringConvertible {
        case message(String)

        var description: String {
            switch self {
            case .message(let message):
                message
            }
        }
    }
}
