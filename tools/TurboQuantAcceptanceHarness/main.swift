// Copyright © 2026 Apple Inc.
//
// Lever ① gate: measure prompt-lookup (n-gram) speculation acceptance on a real
// model. NO speed claim — it generates greedily (the model is the only thing run)
// then replays the prompt-lookup speculator offline to report mean accepted length
// and tokens-per-forward. This is the cheap "first experiment" that decides whether
// the no-draft self-speculation iterator (lever ①) is worth building for a workload.
//
// In-package tools have no swift-transformers dependency, so prompts are supplied
// as token IDs (tokenize real text externally, e.g. with `transformers`, and pass
// `--prompt-ids-file`). The model only ever sees token IDs anyway.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// Token-ID identity tokenizer (the model operates on IDs directly; we never decode).
private struct IdentityTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? { token.utf8.first.map(Int.init) }
    func convertIdToToken(_ id: Int) -> String? {
        String(UnicodeScalar(UInt8(truncatingIfNeeded: id)))
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { "\($0["role"] ?? ""): \($0["content"] ?? "")\n".utf8.map { Int($0) } }
    }
}
private struct IdentityTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { IdentityTokenizer() }
}

private struct PromptIDs: Codable { var label: String; var ids: [Int] }

private struct AcceptanceRow: Codable {
    var prompt: String
    var ngram: Int
    var maxProposalTokens: Int
    var promptTokens: Int
    var generatedTokens: Int
    var forwards: Int
    var proposalRounds: Int
    var acceptedTokens: Int
    var fullAcceptRounds: Int
    var meanAcceptedPerProposal: Double
    var tokensPerForward: Double
}
private struct AcceptanceReport: Codable {
    var schemaVersion = 1
    var model: String
    var maxTokens: Int
    var rows: [AcceptanceRow]
}
private struct GeneratedRun: Sendable {
    var label: String
    var tokens: [Int]
    var promptLength: Int
}

private struct ValidationResult: Sendable {
    var label: String
    var generated: Int
    var identical: Bool
    var firstMismatch: Int
    var plainTokensPerSec: Double
    var specTokensPerSec: Double
    var speedup: Double
    var acceptanceRate: Double
    var forwards: Int
    var tokensPerForward: Double
}

private func arg(_ name: String, default def: String? = nil) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), a.indices.contains(i + 1) else { return def }
    return a[i + 1]
}
private func argInt(_ name: String, default def: Int) -> Int { arg(name).flatMap(Int.init) ?? def }
private func argInts(_ name: String, default def: [Int]) -> [Int] {
    guard let raw = arg(name) else { return def }
    let v = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return v.isEmpty ? def : v
}

@main
struct TurboQuantAcceptanceHarness {
    static func main() async throws {
        if CommandLine.arguments.contains("--help") {
            print(
                """
                TurboQuantAcceptanceHarness — lever ① acceptance gate (no speed claim)

                  --model-dir <path>       MLX model dir, or set TQ_MODEL_DIR
                  --prompt-ids-file <path> JSON [{"label","ids":[Int]}] (tokenize text externally)
                  --max-tokens <n>         generated tokens per prompt (default 128)
                  --ngrams <csv>           match-window sizes to sweep (default 2,3,4)
                  --max-proposal <n>       max proposed tokens per round (default 8)
                  --eos <id>               stop token id (optional)
                  --output <path>          write JSON report
                """)
            return
        }
        guard let modelPath = arg("--model-dir", default: ProcessInfo.processInfo.environment["TQ_MODEL_DIR"]) else {
            FileHandle.standardError.write(Data("missing --model-dir / TQ_MODEL_DIR\n".utf8)); exit(2)
        }
        guard let promptsFile = arg("--prompt-ids-file") else {
            FileHandle.standardError.write(Data("missing --prompt-ids-file (JSON of token-id prompts)\n".utf8)); exit(2)
        }
        guard Device.defaultDevice().deviceType == .gpu else {
            FileHandle.standardError.write(Data("GPU device required\n".utf8)); exit(3)
        }

        let maxTokens = argInt("--max-tokens", default: 128)
        let ngrams = argInts("--ngrams", default: [2, 3, 4])
        let maxProposal = argInt("--max-proposal", default: 8)
        let eos = arg("--eos").flatMap(Int.init)

        let prompts = try JSONDecoder().decode(
            [PromptIDs].self, from: Data(contentsOf: URL(fileURLWithPath: promptsFile)))

        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL, using: IdentityTokenizerLoader())

        // ----- End-to-end ① validation + speed: plain greedy vs n-gram speculative -----
        if CommandLine.arguments.contains("--validate-speculative") {
            let ngram = argInt("--ngram", default: 3)
            let results: [ValidationResult] = try await container.perform { (ctx: ModelContext) in
                var out: [ValidationResult] = []
                for prompt in prompts {
                    func greedyParams() -> GenerateParameters {
                        var p = GenerateParameters(maxTokens: maxTokens)
                        p.temperature = 0
                        return p
                    }
                    // Plain greedy reference.
                    var plain = try TokenIterator(
                        prompt: MLXArray(prompt.ids.map { Int32($0) }),
                        model: ctx.model, parameters: greedyParams())
                    var plainTokens: [Int] = []
                    let t0 = Date.timeIntervalSinceReferenceDate
                    for _ in 0 ..< maxTokens { guard let t = plain.next() else { break }; plainTokens.append(t) }
                    Stream().synchronize()
                    let plainElapsed = Date.timeIntervalSinceReferenceDate - t0

                    // N-gram speculative.
                    var spec = try NgramSpeculativeTokenIterator(
                        input: LMInput(text: LMInput.Text(tokens: MLXArray(prompt.ids.map { Int32($0) }))),
                        model: ctx.model, parameters: greedyParams(),
                        ngram: ngram, maxProposalTokens: maxProposal)
                    var specTokens: [Int] = []
                    let t1 = Date.timeIntervalSinceReferenceDate
                    for _ in 0 ..< maxTokens { guard let t = spec.next() else { break }; specTokens.append(t) }
                    Stream().synchronize()
                    let specElapsed = Date.timeIntervalSinceReferenceDate - t1

                    let n = Swift.min(plainTokens.count, specTokens.count)
                    var firstMismatch = -1
                    for i in 0 ..< n where plainTokens[i] != specTokens[i] { firstMismatch = i; break }
                    let identical = (plainTokens == specTokens)
                    let plainTPS = plainElapsed > 0 ? Double(plainTokens.count) / plainElapsed : 0
                    let specTPS = specElapsed > 0 ? Double(specTokens.count) / specElapsed : 0
                    out.append(
                        ValidationResult(
                            label: prompt.label, generated: specTokens.count, identical: identical,
                            firstMismatch: firstMismatch, plainTokensPerSec: plainTPS,
                            specTokensPerSec: specTPS,
                            speedup: plainTPS > 0 ? specTPS / plainTPS : 0,
                            acceptanceRate: spec.acceptanceRate, forwards: spec.modelForwards,
                            tokensPerForward: spec.modelForwards > 0
                                ? Double(specTokens.count) / Double(spec.modelForwards) : 0))
                }
                return out
            }
            print(
                "prompt            identical  genTok  plain tok/s  spec tok/s  speedup  acceptRate  tok/forward")
            print(
                "---------------   ---------  ------  -----------  ----------  -------  ----------  -----------")
            var allIdentical = true
            for r in results {
                if !r.identical { allIdentical = false }
                print(
                    [
                        r.label.padding(toLength: 15, withPad: " ", startingAt: 0),
                        r.identical ? "    yes  " : " NO@\(r.firstMismatch)",
                        String(format: "%6d", r.generated),
                        String(format: "%11.2f", r.plainTokensPerSec),
                        String(format: "%10.2f", r.specTokensPerSec),
                        String(format: "%7.3f", r.speedup),
                        String(format: "%10.3f", r.acceptanceRate),
                        String(format: "%11.3f", r.tokensPerForward),
                    ].joined(separator: "   "))
            }
            print(
                "\n=== determinism gate: \(allIdentical ? "PASS (all byte-identical)" : "FAIL (mismatch)") ===")
            return
        }

        let promptList = prompts
        let runs: [GeneratedRun] = try await container.perform { (ctx: ModelContext) in
            var out: [GeneratedRun] = []
            for prompt in promptList {
                var params = GenerateParameters(maxTokens: maxTokens)
                params.temperature = 0  // greedy / deterministic
                var iterator = try TokenIterator(
                    prompt: MLXArray(prompt.ids.map { Int32($0) }),
                    model: ctx.model, parameters: params)
                var generated: [Int] = []
                generated.reserveCapacity(maxTokens)
                for _ in 0 ..< maxTokens {
                    guard let token = iterator.next() else { break }
                    if let eos, token == eos { break }
                    generated.append(token)
                }
                out.append(
                    GeneratedRun(
                        label: prompt.label, tokens: prompt.ids + generated,
                        promptLength: prompt.ids.count))
            }
            return out
        }

        var rows: [AcceptanceRow] = []
        print(
            "prompt            ngram  promptTok  genTok  forwards  rounds  accepted  meanAcc  tok/forward")
        print(
            "---------------   -----  ---------  ------  --------  ------  --------  -------  -----------")
        for run in runs {
            for ngram in ngrams {
                let s = PromptLookupSpeculator.simulate(
                    tokens: run.tokens, promptLength: run.promptLength,
                    ngram: ngram, maxProposalTokens: maxProposal)
                rows.append(
                    AcceptanceRow(
                        prompt: run.label, ngram: ngram, maxProposalTokens: maxProposal,
                        promptTokens: run.promptLength, generatedTokens: s.generatedTokens,
                        forwards: s.forwards, proposalRounds: s.proposalRounds,
                        acceptedTokens: s.acceptedTokens, fullAcceptRounds: s.fullAcceptRounds,
                        meanAcceptedPerProposal: s.meanAcceptedPerProposal,
                        tokensPerForward: s.tokensPerForward))
                print(
                    [
                        run.label.padding(toLength: 15, withPad: " ", startingAt: 0),
                        String(format: "%5d", ngram), String(format: "%9d", run.promptLength),
                        String(format: "%6d", s.generatedTokens), String(format: "%8d", s.forwards),
                        String(format: "%6d", s.proposalRounds), String(format: "%8d", s.acceptedTokens),
                        String(format: "%7.2f", s.meanAcceptedPerProposal),
                        String(format: "%11.3f", s.tokensPerForward),
                    ].joined(separator: "   "))
            }
        }

        print("\n=== aggregate tokens/forward by ngram (all prompts) ===")
        for ngram in ngrams {
            let r = rows.filter { $0.ngram == ngram }
            let gen = r.reduce(0) { $0 + $1.generatedTokens }
            let fwd = r.reduce(0) { $0 + $1.forwards }
            let agg = fwd > 0 ? Double(gen) / Double(fwd) : 0
            print(String(format: "  ngram=%d  tokens/forward=%.3f  (%d tok / %d forwards)", ngram, agg, gen, fwd))
        }

        if let outputPath = arg("--output") {
            let report = AcceptanceReport(model: modelURL.lastPathComponent, maxTokens: maxTokens, rows: rows)
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(report).write(to: URL(fileURLWithPath: outputPath), options: .atomic)
            print("\nwrote report: \(outputPath)")
        }
    }
}
