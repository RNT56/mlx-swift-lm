import Foundation
import Testing

@testable import MLXLMCommon

@Suite
struct PromptLookupSpeculatorTests {

    @Test func proposeReturnsContinuationAfterMostRecentMatch() {
        var speculator = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 8)
        // [5,6,7,8, 5,6,7] -> last window [5,6,7] matched earlier at index 0,
        // whose verbatim continuation in the sequence is [8,5,6,7].
        speculator.append(contentsOf: [5, 6, 7, 8, 5, 6, 7])
        #expect(speculator.propose() == [8, 5, 6, 7])
    }

    @Test func proposeRespectsMaxProposalTokens() {
        var speculator = PromptLookupSpeculator(ngram: 2, maxProposalTokens: 3)
        // window [1,2] first followed by [3,4,5,6,7]; only 3 proposed.
        speculator.append(contentsOf: [1, 2, 3, 4, 5, 6, 7, 1, 2])
        #expect(speculator.propose() == [3, 4, 5])
    }

    @Test func proposeEmptyWhenNoEarlierMatch() {
        var speculator = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 8)
        speculator.append(contentsOf: [1, 2, 3, 4, 5])  // no repeated 3-gram
        #expect(speculator.propose().isEmpty)
    }

    @Test func proposeEmptyWhenShorterThanNgram() {
        var speculator = PromptLookupSpeculator(ngram: 4, maxProposalTokens: 8)
        speculator.append(contentsOf: [1, 2, 3])
        #expect(speculator.propose().isEmpty)
    }

    @Test func proposePrefersMostRecentOccurrence() {
        var speculator = PromptLookupSpeculator(ngram: 2, maxProposalTokens: 4)
        // window [9,9] occurs followed by 1, then later followed by 2; the most
        // recent earlier occurrence (->2) wins.
        speculator.append(contentsOf: [9, 9, 1, 0, 9, 9, 2, 0, 9, 9])
        #expect(speculator.propose() == [2, 0, 9, 9])
    }

    // MARK: - Acceptance simulation (the lever-① gate logic)

    @Test func simulateNoRepetitionGivesNoSpeculativeGain() {
        // Strictly unique tokens -> no proposal ever matches -> exactly one
        // forward per token, zero accepted.
        let tokens = Array(0 ..< 64)
        let stats = PromptLookupSpeculator.simulate(
            tokens: tokens, promptLength: 8, ngram: 3, maxProposalTokens: 8)
        #expect(stats.generatedTokens == 56)
        #expect(stats.acceptedTokens == 0)
        #expect(stats.proposalRounds == 0)
        #expect(stats.forwards == 56)
        #expect(stats.tokensPerForward == 1.0)
    }

    @Test func simulatePeriodicSequenceAmortizesForwards() {
        // Period-4 repetition: prompt-lookup should accept long runs and emit
        // many tokens per forward.
        let tokens = (0 ..< 48).map { $0 % 4 }
        let stats = PromptLookupSpeculator.simulate(
            tokens: tokens, promptLength: 4, ngram: 3, maxProposalTokens: 8)
        #expect(stats.generatedTokens == 44)
        #expect(stats.acceptedTokens > 0)
        #expect(stats.proposalRounds > 0)
        #expect(stats.meanAcceptedPerProposal > 1.0)
        #expect(stats.tokensPerForward > 2.0)  // strong amortization on pure repetition
        #expect(stats.forwards < stats.generatedTokens)
    }

    @Test func simulateVerbatimCopyIsAcceptedAfterWarmup() {
        // Generated half is an exact copy of the prompt half (the canonical
        // retrieval/quote case).
        let pattern = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        let tokens = pattern + pattern
        let stats = PromptLookupSpeculator.simulate(
            tokens: tokens, promptLength: pattern.count, ngram: 3, maxProposalTokens: 8)
        #expect(stats.acceptedTokens > 0)
        #expect(stats.tokensPerForward > 1.5)
    }

    @Test func simulateEmptyGenerationIsZero() {
        let tokens = [1, 2, 3, 4]
        let stats = PromptLookupSpeculator.simulate(
            tokens: tokens, promptLength: 4, ngram: 3, maxProposalTokens: 8)
        #expect(stats.generatedTokens == 0)
        #expect(stats.forwards == 0)
        #expect(stats.tokensPerForward == 0)
    }

    @Test func simulateNeverEmitsMoreForwardsThanTokens() {
        // Invariant: forwards <= generated tokens, and emitted == generated.
        let tokens = (0 ..< 100).map { ($0 * 7) % 13 }  // mildly repetitive
        let stats = PromptLookupSpeculator.simulate(
            tokens: tokens, promptLength: 5, ngram: 3, maxProposalTokens: 8)
        #expect(stats.forwards <= stats.generatedTokens)
        #expect(stats.forwards >= 1)
        #expect(stats.tokensPerForward >= 1.0)
    }

    // ----- truncate (N7 optimistic-prefetch rollback building block) -----

    @Test func truncateThenProposeEqualsNeverAppended() {
        // The core invariant: appending then truncating leaves the proposer in the
        // SAME state as if the truncated tokens were never appended.
        let base = [5, 6, 7, 8, 9, 5, 6, 7]
        let speculative = [99, 42, 7]  // wrong drafts appended on a full-accept guess
        var rolled = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 8)
        rolled.append(contentsOf: base)
        rolled.append(contentsOf: speculative)
        rolled.truncate(to: base.count)
        var clean = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 8)
        clean.append(contentsOf: base)
        #expect(rolled.count == clean.count)
        #expect(rolled.propose() == clean.propose())
    }

    @Test func truncateRestoresIndexForFutureMatches() {
        // After rollback, a future append that recreates a window must still match the
        // original earlier occurrence (index not corrupted by the popped speculative window).
        var s = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 8)
        s.append(contentsOf: [1, 2, 3, 4, 5])  // window [1,2,3] recorded at 0
        s.append(contentsOf: [1, 2, 3, 77])  // optimistic, wrong
        s.truncate(to: 5)  // back to [1,2,3,4,5]
        s.append(contentsOf: [1, 2, 3])  // now [1,2,3,4,5,1,2,3]; last [1,2,3] matches at 0
        #expect(s.propose() == [4, 5, 1, 2, 3])
    }

    @Test func truncateNoOpWhenCountAtOrAboveLength() {
        var s = PromptLookupSpeculator(ngram: 2, maxProposalTokens: 4)
        s.append(contentsOf: [3, 1, 4, 1, 5])
        let before = s.propose()
        s.truncate(to: 5)  // == length: no-op
        s.truncate(to: 99)  // > length: no-op
        #expect(s.count == 5)
        #expect(s.propose() == before)
    }

    @Test func repeatedAppendTruncateCyclesAreStable() {
        // Mimics many prefetch mispredicts: append-then-truncate repeatedly must keep the
        // proposer identical to the committed-only baseline.
        var pipelined = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 6)
        var baseline = PromptLookupSpeculator(ngram: 3, maxProposalTokens: 6)
        let committed = [2, 3, 5, 7, 11, 2, 3, 5, 13, 2, 3, 5]
        for (i, tok) in committed.enumerated() {
            // Optimistically append two wrong tokens, then roll them back before committing.
            let mark = pipelined.count
            pipelined.append(contentsOf: [9990 + i, 9991 + i])
            pipelined.truncate(to: mark)
            pipelined.append(tok)
            baseline.append(tok)
        }
        #expect(pipelined.count == baseline.count)
        #expect(pipelined.propose() == baseline.propose())
    }
}
