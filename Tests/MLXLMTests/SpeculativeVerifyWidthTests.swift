import Foundation
import Testing

@testable import MLXLMCommon

/// Verify-width quantization (P1-3 spec-pipeline hygiene): the native
/// compressed SDPA route specializes on q_seq via a function constant, so the
/// draft width must be truncated to a fixed small set when the flag is on —
/// bounding the verify pipelines to q_seq in {2, 4} and staying below the
/// measured qL=8 spill width. Truncation is correctness-neutral (a prefix of
/// a draft is a valid draft; greedy argmax verify corrects any width).
struct SpeculativeVerifyWidthTests {

    @Test func quantizationDisabledIsIdentity() {
        for n in 0 ... 12 {
            #expect(NgramSpeculativeTokenIterator.quantizedDraftWidth(n, enabled: false) == n)
        }
    }

    @Test func quantizationMapsToAllowedWidthsOnly() {
        // n -> expected: 0->0, 1->1, 2->1, 3->3, 4->3, 7->3, 12->3
        let expected = [0: 0, 1: 1, 2: 1, 3: 3, 4: 3, 7: 3, 12: 3]
        for (n, want) in expected {
            #expect(NgramSpeculativeTokenIterator.quantizedDraftWidth(n, enabled: true) == want)
        }
    }

    @Test func quantizedWidthNeverExceedsInputOrSpillCap() {
        for n in 0 ... 32 {
            let q = NgramSpeculativeTokenIterator.quantizedDraftWidth(n, enabled: true)
            #expect(q <= n)
            // verify q_seq = draft + 1 must stay strictly below the measured
            // qL=8 register-spill width on the compressed route
            #expect(q + 1 < 8)
            #expect(q == 0 || NgramSpeculativeTokenIterator.allowedDraftWidths.contains(q))
        }
    }
}
