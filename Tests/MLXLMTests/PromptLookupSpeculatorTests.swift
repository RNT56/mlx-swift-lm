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
}
