// Copyright © 2026 Apple Inc.

import Foundation

/// Draft-model-free speculative proposer ("prompt-lookup" / LLMA decoding).
///
/// Maintains the running token sequence (prompt + generated) and, at each step,
/// proposes the next `k` tokens by finding the most recent earlier occurrence of
/// the last `ngram` tokens and returning what followed it. This captures verbatim
/// repetition — code, JSON, quoted/retrieved spans, edits — which is exactly the
/// long-context regime where TurboQuant is used to reach the context.
///
/// It is the proposal source for ``NgramSpeculativeTokenIterator`` (lever ①). The
/// verify step (one multi-query forward over the affine cache) makes every emitted
/// token bit-exact regardless of proposal quality, so this struct never affects
/// correctness — only how many tokens one forward can emit.
///
/// This type is pure CPU logic with no model/MLX dependency, so the acceptance
/// behaviour is unit-testable via ``simulate(tokens:promptLength:ngram:maxProposalTokens:)``
/// without loading a model.
public struct PromptLookupSpeculator {

    /// Match-window size: the last `ngram` tokens are the lookup key.
    public let ngram: Int
    /// Maximum number of tokens proposed per round.
    public let maxProposalTokens: Int

    private var tokens: [Int] = []
    // hash(window starting at s) -> start positions s, in append order.
    private var index: [Int: [Int]] = [:]

    public init(ngram: Int = 3, maxProposalTokens: Int = 8) {
        precondition(ngram >= 1, "ngram must be >= 1")
        precondition(maxProposalTokens >= 1, "maxProposalTokens must be >= 1")
        self.ngram = ngram
        self.maxProposalTokens = maxProposalTokens
    }

    public var count: Int { tokens.count }

    public mutating func reset() {
        tokens.removeAll(keepingCapacity: true)
        index.removeAll(keepingCapacity: true)
    }

    public mutating func append(_ token: Int) {
        tokens.append(token)
        let n = tokens.count
        if n >= ngram {
            let start = n - ngram
            index[windowHash(start: start), default: []].append(start)
        }
    }

    public mutating func append(contentsOf newTokens: [Int]) {
        for token in newTokens { append(token) }
    }

    /// Truncate the running sequence back to `count` tokens, exactly reversing the
    /// `append`s that grew it past `count`. Used by optimistic-prefetch speculation
    /// (lever ①/N7) to roll back tokens that were appended on a full-acceptance
    /// assumption when the verify later rejected them. O(removed × ngram).
    public mutating func truncate(to count: Int) {
        guard count >= 0, count < tokens.count else { return }
        while tokens.count > count {
            let n = tokens.count
            if n >= ngram {
                // `append` added a window starting at (n - ngram) when the count reached n.
                let start = n - ngram
                let h = windowHash(start: start)
                if var arr = index[h] {
                    if arr.last == start { arr.removeLast() }
                    if arr.isEmpty { index[h] = nil } else { index[h] = arr }
                }
            }
            tokens.removeLast()
        }
    }

    /// Propose up to `maxProposalTokens` guesses for the tokens immediately
    /// following the current sequence. Empty if no earlier match exists.
    public func propose() -> [Int] {
        let n = tokens.count
        guard n >= ngram else { return [] }
        let curStart = n - ngram
        guard let starts = index[windowHash(start: curStart)] else { return [] }
        // Most-recent earlier occurrence (skip the current window; verify equality
        // to guard against hash collisions).
        for start in starts.reversed() where start < curStart {
            if windowEquals(start, curStart) {
                let from = start + ngram
                let to = Swift.min(n, from + maxProposalTokens)
                if from < to { return Array(tokens[from ..< to]) }
            }
        }
        return []
    }

    private func windowHash(start: Int) -> Int {
        // FNV-1a style fold over the window (overflow-wrapping).
        var h = -3_750_763_034_362_895_579  // 0xcbf29ce484222325 as Int
        for k in 0 ..< ngram {
            h = (h ^ tokens[start + k]) &* 1_099_511_628_211
        }
        return h
    }

    private func windowEquals(_ a: Int, _ b: Int) -> Bool {
        for k in 0 ..< ngram where tokens[a + k] != tokens[b + k] { return false }
        return true
    }
}

/// Acceptance statistics for prompt-lookup speculation over a *known* greedy
/// token sequence. Pure replay — no model, no cache, no rollback — so it measures
/// the upper-bound acceptance behaviour exactly and is deterministically testable.
public struct PromptLookupAcceptanceStats: Sendable, Codable, Equatable {
    public let generatedTokens: Int
    public let proposalRounds: Int
    public let acceptedTokens: Int
    public let fullAcceptRounds: Int
    public let forwards: Int
    /// Mean accepted draft tokens per round that actually proposed something.
    public let meanAcceptedPerProposal: Double
    /// Emitted tokens per model forward — the honest throughput proxy (a verify
    /// forward reads the weight stream once and can emit `accepted + 1` tokens).
    public let tokensPerForward: Double
}

extension PromptLookupSpeculator {

    /// Replay prompt-lookup speculation over an already-generated greedy sequence
    /// and report how many model forwards it would have taken. This is the lever-①
    /// acceptance gate: no speed claim, just `tokensPerForward` and mean accepted
    /// length, which determine whether the iterator is worth building for a workload.
    ///
    /// - Parameters:
    ///   - tokens: the full greedy token sequence (prompt + generated).
    ///   - promptLength: number of leading prompt tokens (generation starts here).
    public static func simulate(
        tokens: [Int],
        promptLength: Int,
        ngram: Int,
        maxProposalTokens: Int
    ) -> PromptLookupAcceptanceStats {
        let total = tokens.count
        let start = Swift.max(0, Swift.min(promptLength, total))
        let generated = total - start
        guard generated > 0 else {
            return PromptLookupAcceptanceStats(
                generatedTokens: 0, proposalRounds: 0, acceptedTokens: 0,
                fullAcceptRounds: 0, forwards: 0,
                meanAcceptedPerProposal: 0, tokensPerForward: 0)
        }

        var speculator = PromptLookupSpeculator(ngram: ngram, maxProposalTokens: maxProposalTokens)
        speculator.append(contentsOf: Array(tokens[0 ..< start]))

        var i = start
        var forwards = 0
        var proposalRounds = 0
        var acceptedTokens = 0
        var fullAcceptRounds = 0

        while i < total {
            let proposal = speculator.propose()
            forwards += 1  // one verify/decode forward this round

            var accepted = 0
            if !proposal.isEmpty {
                proposalRounds += 1
                while accepted < proposal.count
                    && i + accepted < total
                    && proposal[accepted] == tokens[i + accepted] {
                    accepted += 1
                }
                acceptedTokens += accepted
                if accepted == proposal.count { fullAcceptRounds += 1 }
            }

            // The forward always emits the true token at position i; matching draft
            // tokens i+1..i+accepted are accepted for free.
            let advance = accepted + 1
            speculator.append(contentsOf: Array(tokens[i ..< Swift.min(total, i + advance)]))
            i += advance
        }

        return PromptLookupAcceptanceStats(
            generatedTokens: generated,
            proposalRounds: proposalRounds,
            acceptedTokens: acceptedTokens,
            fullAcceptRounds: fullAcceptRounds,
            forwards: forwards,
            meanAcceptedPerProposal: proposalRounds > 0
                ? Double(acceptedTokens) / Double(proposalRounds) : 0,
            tokensPerForward: forwards > 0 ? Double(generated) / Double(forwards) : 0
        )
    }
}
