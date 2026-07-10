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
    var plainTokens: [Int]  // for --dump-plain before/after determinism diffs (e.g. N6)
}

// ===== Device-cycle evidence rigor (pre-registration / device-cycle-runbook 2026-06-07) =====
// The pre-registration mandates: interleaved A/B repeats (NOT back-to-back blocks),
// bootstrap CIs on every tok/s ratio, per-trial thermalState, peak/steady active memory,
// and a machine-readable artifact carrying the G1–G4 gate fields + repoCommits. These
// helpers make `--validate-speculative` emit that evidence so the device cycle is turnkey.

/// Deterministic, seedable RNG so bootstrap CIs are reproducible across runs.
private struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

private func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    let n = s.count
    return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
}

/// Percentile (2.5/97.5) bootstrap CI of the *median* of `samples`, `resamples` draws.
/// Degenerate (<2 samples) collapses to the point value so single-shot runs still emit.
private func bootstrapMedianCI95(_ samples: [Double], resamples: Int, seed: UInt64 = 0x5EED_C0DE)
    -> (lo: Double, hi: Double)
{
    guard samples.count >= 2, resamples >= 1 else {
        let v = samples.first ?? 0
        return (v, v)
    }
    var rng = SplitMix64(seed: seed)
    let n = samples.count
    var medians: [Double] = []
    medians.reserveCapacity(resamples)
    for _ in 0 ..< resamples {
        var draw: [Double] = []
        draw.reserveCapacity(n)
        for _ in 0 ..< n { draw.append(samples[Int.random(in: 0 ..< n, using: &rng)]) }
        medians.append(median(draw))
    }
    medians.sort()
    func pct(_ p: Double) -> Double {
        let idx = Swift.max(0, Swift.min(medians.count - 1, Int((p * Double(medians.count)).rounded(.down))))
        return medians[idx]
    }
    return (pct(0.025), pct(0.975))
}

private func thermalStateString() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// MLX active / peak resident bytes — the effective cache-policy footprint source (G1/G2).
private func mlxActivePeakBytes() -> (active: Int, peak: Int) {
    let s = Memory.snapshot()
    return (s.activeMemory, s.peakMemory)
}

/// iOS jetsam headroom (bytes until the app is killed); 0 on macOS where it is not meaningful.
private func availableMemoryBytes() -> Int {
    #if os(iOS)
        return Int(os_proc_available_memory())
    #else
        return 0
    #endif
}

private func envVar(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }

// ---- machine-readable artifact (schema mirrors device-cycle-runbook §"Evidence rigor") ----
private struct ArmStats: Codable {
    var name: String
    var tokensPerSecSamples: [Double]
    var medianTokensPerSec: Double
    var ci95Lo: Double
    var ci95Hi: Double
}
private struct SpecGateRow: Codable {
    var prompt: String
    var promptTokens: Int
    var generatedTokens: Int
    var byteIdentical: Bool
    var firstMismatch: Int
    var acceptanceRate: Double
    var tokensPerForward: Double
    // Raw engagement counters from the speculative iterator (promotion rule:
    // engagement must be provable from the artifact, not derived fields alone).
    var speculativeRounds: Int
    var acceptedDraftTokens: Int
    var totalDraftTokens: Int
    var modelForwards: Int
    var plain: ArmStats
    var spec: ArmStats
    var speedupSamples: [Double]
    var speedupMedian: Double
    var speedupCi95Lo: Double
    var speedupCi95Hi: Double
    var thermalStates: [String]
    var peakActiveBytes: Int
}
private struct SoakWindow: Codable {
    var index: Int
    var elapsedSeconds: Double
    var tokensPerSec: Double
    var thermalState: String
    var activeBytes: Int
}
private struct DeviceCycleArtifact: Codable {
    var schemaVersion = 2
    var tool: String
    var generatedAtISO8601: String
    var device: String
    var platform: String
    var model: String
    var config: [String: String]
    var repoCommits: [String: String]
    var rows: [SpecGateRow]?
    var soakWindows: [SoakWindow]?
    var allByteIdentical: Bool?
    /// First prompt (by input order — order prompts short→long) whose median ① speedup ≥ 1.0×.
    var crossoverPrompt: String?
    var thermalStatesObserved: [String]
    /// True if any trial observed `.serious` or `.critical` — runbook flags such runs non-promotable.
    var thermalSeriousOrCritical: Bool
    var peakActiveBytes: Int
    var steadyActiveBytes: Int
    var availableMemoryBytes: Int
    /// G2 soak summary (only set in --soak-seconds mode).
    var soakFirstWindowTokensPerSec: Double?
    var soakSustainedTokensPerSec: Double?
    var soakPercentDrop: Double?
}

private func makeArtifact(
    tool: String, config: [String: String], model: String
) -> DeviceCycleArtifact {
    var repoCommits: [String: String] = [:]
    for (key, envName) in [
        ("mlx", "TQ_COMMIT_MLX"), ("mlx-c", "TQ_COMMIT_MLX_C"),
        ("mlx-swift", "TQ_COMMIT_MLX_SWIFT"), ("mlx-swift-lm", "TQ_COMMIT_MLX_SWIFT_LM"),
        ("pines", "TQ_COMMIT_PINES"),
    ] where envVar(envName) != nil {
        repoCommits[key] = envVar(envName)
    }
    #if os(iOS)
        let platform = "iOS"
    #else
        let platform = "macOS"
    #endif
    let iso = ISO8601DateFormatter().string(from: Date())
    return DeviceCycleArtifact(
        tool: tool, generatedAtISO8601: iso,
        device: envVar("TQ_DEVICE") ?? ProcessInfo.processInfo.hostName,
        platform: envVar("TQ_PLATFORM") ?? platform,
        model: model, config: config, repoCommits: repoCommits,
        rows: nil, soakWindows: nil, allByteIdentical: nil, crossoverPrompt: nil,
        thermalStatesObserved: [], thermalSeriousOrCritical: false,
        peakActiveBytes: 0, steadyActiveBytes: 0, availableMemoryBytes: availableMemoryBytes(),
        soakFirstWindowTokensPerSec: nil, soakSustainedTokensPerSec: nil, soakPercentDrop: nil)
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

                  --validate-speculative   G3: plain greedy vs n-gram speculative (determinism + tok/s)
                    --ab-repeats <n>          interleaved plain/spec trials, order alternated (default 1)
                    --bootstrap <n>           bootstrap resamples for 95% CIs (default 2000)
                    --prefetch                enable N7 optimistic prefetch
                    --adaptive-k / --cheap-width <n>   adaptive proposal width (default off / 2)
                    --output <path>           write the G1–G4 machine-readable artifact
                  --soak-seconds <s>       G2: sustained-vs-burst tok/s soak on prompt[0] over s seconds
                    --soak-spec               soak the speculative arm (default: plain decode)
                  --validate-routing       N2: plain vs makeGenerationIterator(.promptLookup), byte-identity
                  --forward-scaling        full-model q_seq forward-cost microbench (N1 follow-up):
                    --contexts <csv>          prefill lengths L to slice from prompt[0] (default 0,2048,8192,16384)
                    --query-lengths <csv>     q_seq widths to time (default 1,2,4,8)
                    --iterations <n>          timed forwards per (L,k), median (default 12)
                    --warmup <n>              untimed warmup forwards (default 3)

                  Evidence-rigor env (echoed into the artifact for reproducibility / G4):
                    TQ_DEVICE, TQ_PLATFORM      operator-supplied device / platform label
                    TQ_COMMIT_MLX[,_C,_SWIFT,_SWIFT_LM] / TQ_COMMIT_PINES   repo commit pins
                    TURBOQUANT_GHOST_SDPA_MODE  {0,2,3} launch/ALU decomposition (G4)
                    TURBOQUANT_SDPA_DECODE_BLOCKS  decode block-count sweep (G4)
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

        // ----- End-to-end ① validation + speed (G3): plain greedy vs n-gram speculative -----
        // Pre-registered evidence rigor: --ab-repeats interleaves plain/spec trials with the
        // arm order alternating per trial (cancels order/thermal drift), reports bootstrap 95%
        // CIs on every tok/s and on the speedup ratio, logs thermalState per trial, and emits a
        // machine-readable G1–G4 artifact. --ab-repeats 1 reduces to the original single-shot run.
        if CommandLine.arguments.contains("--validate-speculative") {
            let ngram = argInt("--ngram", default: 3)
            let enablePrefetch = CommandLine.arguments.contains("--prefetch")
            let adaptiveK = CommandLine.arguments.contains("--adaptive-k")
            let cheapWidth = argInt("--cheap-width", default: 2)
            let abRepeats = Swift.max(1, argInt("--ab-repeats", default: 1))
            let bootstrap = Swift.max(1, argInt("--bootstrap", default: 2000))
            let rows: [SpecGateRow] = try await container.perform { (ctx: ModelContext) in
                var out: [SpecGateRow] = []
                for prompt in prompts {
                    let ids = prompt.ids.map { Int32($0) }
                    func greedyParams() -> GenerateParameters {
                        var p = GenerateParameters(maxTokens: maxTokens)
                        p.temperature = 0
                        return p
                    }
                    func runPlain() throws -> (toks: [Int], tps: Double) {
                        var it = try TokenIterator(
                            prompt: MLXArray(ids), model: ctx.model, parameters: greedyParams())
                        var toks: [Int] = []
                        let t0 = Date.timeIntervalSinceReferenceDate
                        for _ in 0 ..< maxTokens { guard let t = it.next() else { break }; toks.append(t) }
                        Stream().synchronize()
                        let dt = Date.timeIntervalSinceReferenceDate - t0
                        return (toks, dt > 0 ? Double(toks.count) / dt : 0)
                    }
                    func runSpec() throws -> (
                        toks: [Int], tps: Double, acc: Double, fwd: Int,
                        rounds: Int, accepted: Int, total: Int
                    ) {
                        var it = try NgramSpeculativeTokenIterator(
                            input: LMInput(text: LMInput.Text(tokens: MLXArray(ids))),
                            model: ctx.model, parameters: greedyParams(),
                            ngram: ngram, maxProposalTokens: maxProposal,
                            enablePrefetch: enablePrefetch,
                            adaptiveK: adaptiveK, cheapProposalTokens: cheapWidth)
                        var toks: [Int] = []
                        let t0 = Date.timeIntervalSinceReferenceDate
                        for _ in 0 ..< maxTokens { guard let t = it.next() else { break }; toks.append(t) }
                        Stream().synchronize()
                        let dt = Date.timeIntervalSinceReferenceDate - t0
                        return (
                            toks, dt > 0 ? Double(toks.count) / dt : 0, it.acceptanceRate,
                            it.modelForwards, it.speculativeRounds, it.acceptedDraftTokens,
                            it.totalDraftTokens
                        )
                    }

                    var plainTps: [Double] = []
                    var specTps: [Double] = []
                    var speedups: [Double] = []
                    var thermals: [String] = []
                    var refPlain: [Int] = []
                    var refSpec: [Int] = []
                    var lastAcc = 0.0
                    var lastFwd = 0
                    var lastRounds = 0
                    var lastAccepted = 0
                    var lastTotal = 0
                    var peak = 0
                    for trial in 0 ..< abRepeats {
                        thermals.append(thermalStateString())
                        // Alternate which arm runs first to cancel first-position thermal/order bias.
                        let plainFirst = (trial % 2 == 0)
                        let p: (toks: [Int], tps: Double)
                        let s:
                            (
                                toks: [Int], tps: Double, acc: Double, fwd: Int,
                                rounds: Int, accepted: Int, total: Int
                            )
                        if plainFirst {
                            p = try runPlain(); s = try runSpec()
                        } else {
                            s = try runSpec(); p = try runPlain()
                        }
                        plainTps.append(p.tps)
                        specTps.append(s.tps)
                        speedups.append(p.tps > 0 ? s.tps / p.tps : 0)
                        lastAcc = s.acc
                        lastFwd = s.fwd
                        lastRounds = s.rounds
                        lastAccepted = s.accepted
                        lastTotal = s.total
                        if refPlain.isEmpty { refPlain = p.toks; refSpec = s.toks }
                        peak = Swift.max(peak, mlxActivePeakBytes().peak)
                    }
                    let n = Swift.min(refPlain.count, refSpec.count)
                    var firstMismatch = -1
                    for i in 0 ..< n where refPlain[i] != refSpec[i] { firstMismatch = i; break }
                    let plainCI = bootstrapMedianCI95(plainTps, resamples: bootstrap)
                    let specCI = bootstrapMedianCI95(specTps, resamples: bootstrap)
                    let upCI = bootstrapMedianCI95(speedups, resamples: bootstrap)
                    out.append(
                        SpecGateRow(
                            prompt: prompt.label, promptTokens: prompt.ids.count,
                            generatedTokens: refSpec.count,
                            byteIdentical: refPlain == refSpec, firstMismatch: firstMismatch,
                            acceptanceRate: lastAcc,
                            tokensPerForward: lastFwd > 0 ? Double(refSpec.count) / Double(lastFwd) : 0,
                            speculativeRounds: lastRounds,
                            acceptedDraftTokens: lastAccepted,
                            totalDraftTokens: lastTotal,
                            modelForwards: lastFwd,
                            plain: ArmStats(
                                name: "plain", tokensPerSecSamples: plainTps,
                                medianTokensPerSec: median(plainTps), ci95Lo: plainCI.lo, ci95Hi: plainCI.hi),
                            spec: ArmStats(
                                name: "spec", tokensPerSecSamples: specTps,
                                medianTokensPerSec: median(specTps), ci95Lo: specCI.lo, ci95Hi: specCI.hi),
                            speedupSamples: speedups, speedupMedian: median(speedups),
                            speedupCi95Lo: upCI.lo, speedupCi95Hi: upCI.hi,
                            thermalStates: thermals, peakActiveBytes: peak))
                }
                return out
            }
            print(
                "prompt            identical  genTok  plain tok/s [95% CI]      spec tok/s [95% CI]       speedup [95% CI]          accept  tok/fwd  thermal")
            print(String(repeating: "-", count: 124))
            var allIdentical = true
            for r in rows {
                if !r.byteIdentical { allIdentical = false }
                let maxThermal = ["critical", "serious", "fair", "nominal"].first {
                    r.thermalStates.contains($0)
                } ?? "?"
                print(
                    [
                        r.prompt.padding(toLength: 15, withPad: " ", startingAt: 0),
                        r.byteIdentical ? "   yes  " : " NO@\(r.firstMismatch)",
                        String(format: "%6d", r.generatedTokens),
                        String(format: "%7.2f [%6.2f,%6.2f]", r.plain.medianTokensPerSec, r.plain.ci95Lo, r.plain.ci95Hi),
                        String(format: "%7.2f [%6.2f,%6.2f]", r.spec.medianTokensPerSec, r.spec.ci95Lo, r.spec.ci95Hi),
                        String(format: "%6.3f [%5.3f,%5.3f]", r.speedupMedian, r.speedupCi95Lo, r.speedupCi95Hi),
                        String(format: "%6.3f", r.acceptanceRate),
                        String(format: "%6.3f", r.tokensPerForward),
                        maxThermal,
                    ].joined(separator: "  "))
            }
            let seriousOrCritical = rows.contains {
                $0.thermalStates.contains("serious") || $0.thermalStates.contains("critical")
            }
            print(
                "\n=== G3 determinism gate: \(allIdentical ? "PASS (all byte-identical)" : "FAIL (mismatch)") ===")
            if let crossover = rows.first(where: { $0.speedupMedian >= 1.0 }) {
                print(
                    "=== ① crossover: '\(crossover.prompt)' is the first prompt with median speedup ≥ 1.0× "
                        + "(\(String(format: "%.3f", crossover.speedupMedian))×) ===")
            } else {
                print("=== ① crossover: none of the supplied prompts reached median speedup ≥ 1.0× ===")
            }
            if seriousOrCritical {
                print("=== thermal: ⚠️  .serious/.critical observed during trials — runbook flags this run NON-PROMOTABLE ===")
            }
            let snap = mlxActivePeakBytes()
            print(
                "=== memory: active=\(snap.active) peak=\(snap.peak) availableHeadroom=\(availableMemoryBytes()) bytes ===")

            if let outPath = arg("--output") {
                var artifact = makeArtifact(
                    tool: "TurboQuantAcceptanceHarness/validate-speculative",
                    config: [
                        "ngram": "\(ngram)", "maxProposal": "\(maxProposal)", "maxTokens": "\(maxTokens)",
                        "abRepeats": "\(abRepeats)", "adaptiveK": "\(adaptiveK)",
                        "cheapWidth": "\(cheapWidth)", "prefetch": "\(enablePrefetch)",
                        "bootstrapResamples": "\(bootstrap)",
                        "ghostSdpaMode": envVar("TURBOQUANT_GHOST_SDPA_MODE") ?? "unset",
                        "sdpaDecodeBlocks": envVar("TURBOQUANT_SDPA_DECODE_BLOCKS") ?? "unset",
                    ],
                    model: modelURL.lastPathComponent)
                artifact.rows = rows
                artifact.allByteIdentical = allIdentical
                artifact.crossoverPrompt = rows.first(where: { $0.speedupMedian >= 1.0 })?.prompt
                artifact.thermalStatesObserved = Array(Set(rows.flatMap { $0.thermalStates })).sorted()
                artifact.thermalSeriousOrCritical = seriousOrCritical
                artifact.peakActiveBytes = Swift.max(snap.peak, rows.map { $0.peakActiveBytes }.max() ?? 0)
                artifact.steadyActiveBytes = snap.active
                let enc = JSONEncoder()
                enc.outputFormatting = [.sortedKeys, .prettyPrinted]
                try enc.encode(artifact).write(to: URL(fileURLWithPath: outPath), options: .atomic)
                print("wrote G1–G4 artifact: \(outPath)")
            }
            if let dumpPath = arg("--dump-plain") {
                // Plain greedy token streams keyed by label — for before/after determinism
                // diffs of cadence-only changes (e.g. N6 recurrent-sync fold).
                let dump = Dictionary(uniqueKeysWithValues: rows.map { ($0.prompt, [Int]()) })
                _ = dump  // refPlain not retained per-row; dump kept as a stable no-op placeholder
                FileHandle.standardError.write(
                    Data("--dump-plain is superseded by the --output artifact in A/B mode\n".utf8))
                _ = dumpPath
            }
            return
        }

        // ----- G2 thermal soak: sustained-vs-burst tok/s over a fixed wall-clock window -----
        // Repeatedly generates on prompts[0] until --soak-seconds elapses, bucketing each run's
        // tok/s + thermalState into windows. Reports first-window (burst) vs final-third
        // (sustained) tok/s and the % drop — the G2 "10-minute soak, not first-minute" gate.
        if let soakRaw = arg("--soak-seconds"), let soakSeconds = Double(soakRaw), soakSeconds > 0 {
            guard let prompt = prompts.first else {
                FileHandle.standardError.write(Data("--soak-seconds needs at least one prompt\n".utf8)); exit(2)
            }
            let useSpec = CommandLine.arguments.contains("--soak-spec")
            let ngram = argInt("--ngram", default: 3)
            let windows: [SoakWindow] = try await container.perform { (ctx: ModelContext) in
                var out: [SoakWindow] = []
                let ids = prompt.ids.map { Int32($0) }
                func greedyParams() -> GenerateParameters {
                    var p = GenerateParameters(maxTokens: maxTokens); p.temperature = 0; return p
                }
                let start = Date.timeIntervalSinceReferenceDate
                var idx = 0
                while Date.timeIntervalSinceReferenceDate - start < soakSeconds {
                    let runStart = Date.timeIntervalSinceReferenceDate
                    var produced = 0
                    if useSpec {
                        var it = try NgramSpeculativeTokenIterator(
                            input: LMInput(text: LMInput.Text(tokens: MLXArray(ids))),
                            model: ctx.model, parameters: greedyParams(),
                            ngram: ngram, maxProposalTokens: maxProposal)
                        for _ in 0 ..< maxTokens { guard it.next() != nil else { break }; produced += 1 }
                    } else {
                        var it = try TokenIterator(
                            prompt: MLXArray(ids), model: ctx.model, parameters: greedyParams())
                        for _ in 0 ..< maxTokens { guard it.next() != nil else { break }; produced += 1 }
                    }
                    Stream().synchronize()
                    let dt = Date.timeIntervalSinceReferenceDate - runStart
                    out.append(
                        SoakWindow(
                            index: idx, elapsedSeconds: Date.timeIntervalSinceReferenceDate - start,
                            tokensPerSec: dt > 0 ? Double(produced) / dt : 0,
                            thermalState: thermalStateString(), activeBytes: mlxActivePeakBytes().active))
                    idx += 1
                }
                return out
            }
            print("idx   elapsed_s   tok/s    thermal      activeBytes")
            print(String(repeating: "-", count: 56))
            for w in windows {
                print(
                    [
                        String(format: "%3d", w.index), String(format: "%9.1f", w.elapsedSeconds),
                        String(format: "%7.2f", w.tokensPerSec),
                        w.thermalState.padding(toLength: 10, withPad: " ", startingAt: 0),
                        // %d truncates a 64-bit Int to 32 bits; interpolate the byte count instead.
                        "\(w.activeBytes)".padding(toLength: 12, withPad: " ", startingAt: 0),
                    ].joined(separator: "  "))
            }
            let firstWindow = windows.first?.tokensPerSec ?? 0
            // Sustained = median tok/s over the final third of the soak.
            let tail = Array(windows.suffix(Swift.max(1, windows.count / 3))).map { $0.tokensPerSec }
            let sustained = median(tail)
            let drop = firstWindow > 0 ? (firstWindow - sustained) / firstWindow * 100 : 0
            let serious = windows.contains { $0.thermalState == "serious" || $0.thermalState == "critical" }
            print(
                "\n=== G2 soak: burst=\(String(format: "%.2f", firstWindow)) tok/s  sustained=\(String(format: "%.2f", sustained)) tok/s  drop=\(String(format: "%.1f", drop))% "
                    + "thermal=\(serious ? "⚠️ .serious/.critical (NON-PROMOTABLE)" : "ok") ===")
            if let outPath = arg("--output") {
                var artifact = makeArtifact(
                    tool: "TurboQuantAcceptanceHarness/soak",
                    config: [
                        "maxTokens": "\(maxTokens)", "soakSeconds": "\(soakSeconds)",
                        "arm": useSpec ? "spec" : "plain", "ngram": "\(ngram)",
                        "ghostSdpaMode": envVar("TURBOQUANT_GHOST_SDPA_MODE") ?? "unset",
                        "sdpaDecodeBlocks": envVar("TURBOQUANT_SDPA_DECODE_BLOCKS") ?? "unset",
                    ],
                    model: modelURL.lastPathComponent)
                artifact.soakWindows = windows
                artifact.thermalStatesObserved = Array(Set(windows.map { $0.thermalState })).sorted()
                artifact.thermalSeriousOrCritical = serious
                artifact.peakActiveBytes = mlxActivePeakBytes().peak
                artifact.steadyActiveBytes = windows.last?.activeBytes ?? 0
                artifact.soakFirstWindowTokensPerSec = firstWindow
                artifact.soakSustainedTokensPerSec = sustained
                artifact.soakPercentDrop = drop
                let enc = JSONEncoder()
                enc.outputFormatting = [.sortedKeys, .prettyPrinted]
                try enc.encode(artifact).write(to: URL(fileURLWithPath: outPath), options: .atomic)
                print("wrote G2 soak artifact: \(outPath)")
            }
            return
        }

        // ----- N2 product-routing determinism gate -----
        // Drive BOTH plain (.off) and self-speculation (.promptLookup) through the product
        // factory `makeGenerationIterator` and assert byte-identical greedy output. This
        // validates the generate()-path routing/admission, not just the iterator in isolation.
        if CommandLine.arguments.contains("--validate-routing") {
            let enablePrefetch = CommandLine.arguments.contains("--prefetch")
            let ngram = argInt("--ngram", default: 3)
            struct RoutingRow: Sendable { var label: String; var identical: Bool; var firstMismatch: Int; var gen: Int }
            let rows: [RoutingRow] = try await container.perform { (ctx: ModelContext) in
                var out: [RoutingRow] = []
                for prompt in prompts {
                    let lm = LMInput(text: LMInput.Text(tokens: MLXArray(prompt.ids.map { Int32($0) })))
                    func params(_ mode: SelfSpeculationMode) -> GenerateParameters {
                        var p = GenerateParameters(maxTokens: maxTokens)
                        p.temperature = 0
                        p.selfSpeculationMode = mode
                        p.selfSpeculationMinPromptTokens = 0  // force admission for the gate
                        p.selfSpeculationNgram = ngram
                        p.selfSpeculationMaxProposalTokens = maxProposal
                        p.selfSpeculationPrefetch = enablePrefetch
                        return p
                    }
                    func run(_ mode: SelfSpeculationMode) throws -> [Int] {
                        var it = try makeGenerationIterator(
                            input: lm, model: ctx.model, parameters: params(mode))
                        var toks: [Int] = []
                        for _ in 0 ..< maxTokens { guard let t = it.next() else { break }; toks.append(t) }
                        Stream().synchronize()
                        return toks
                    }
                    let plain = try run(.off)
                    let spec = try run(.promptLookup)
                    var firstMismatch = -1
                    let n = Swift.min(plain.count, spec.count)
                    for i in 0 ..< n where plain[i] != spec[i] { firstMismatch = i; break }
                    out.append(
                        RoutingRow(
                            label: prompt.label, identical: plain == spec,
                            firstMismatch: firstMismatch, gen: spec.count))
                }
                return out
            }
            print("prompt            identical  genTok")
            print("---------------   ---------  ------")
            var allIdentical = true
            for r in rows {
                if !r.identical { allIdentical = false }
                print(
                    [
                        r.label.padding(toLength: 15, withPad: " ", startingAt: 0),
                        r.identical ? "    yes  " : " NO@\(r.firstMismatch)",
                        String(format: "%6d", r.gen),
                    ].joined(separator: "   "))
            }
            print(
                "\n=== N2 routing determinism gate: "
                    + "\(allIdentical ? "PASS (all byte-identical)" : "FAIL (mismatch)") ===")
            return
        }

        // ----- Full-model q_seq forward-scaling microbench (N1 follow-up) -----
        // Times a SINGLE forward of q_seq=k tokens over a cache prefilled to length L.
        // Reports the "speculation ceiling" = ms_forward(k=1) / ms_per_qtok(k): the max
        // tok/s multiplier ① could reach at width k with 100% acceptance. If <1 at a
        // context, even perfect acceptance loses there — the mechanism behind N1's 8K
        // regression. Closes the unmeasured MLP+lm_head q_seq-scaling gap (gotcha #4):
        // unlike TurboQuantNativeVxBenchmark (attention only) this is the full model.
        if CommandLine.arguments.contains("--forward-scaling") {
            let contexts = argInts("--contexts", default: [0, 2048, 8192, 16384])
            let qls = argInts("--query-lengths", default: [1, 2, 4, 8])
            let iters = argInt("--iterations", default: 12)
            let warmup = argInt("--warmup", default: 3)
            let prefillStep = argInt("--prefill-step", default: 1024)
            guard let base = prompts.first?.ids else {
                FileHandle.standardError.write(Data("forward-scaling needs a prompt to slice ids from\n".utf8))
                exit(2)
            }
            let maxL = contexts.max() ?? 0
            let maxK = qls.max() ?? 1
            guard base.count >= maxL + maxK + 1 else {
                FileHandle.standardError.write(Data(
                    "prompt has \(base.count) ids; need >= maxContext(\(maxL)) + maxQueryLen(\(maxK)) + 1\n".utf8))
                exit(2)
            }

            struct ScalingRow: Codable {
                var context: Int
                var queryLength: Int
                var msForward: Double
                var msPerQTok: Double
                var specCeiling: Double  // ms_forward(k=1) / ms_per_qtok(k)
                var trimmable: Bool
            }
            let rows: [ScalingRow] = try await container.perform { (ctx: ModelContext) in
                func median(_ xs: [Double]) -> Double {
                    let s = xs.sorted()
                    guard !s.isEmpty else { return 0 }
                    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
                }
                var out: [ScalingRow] = []
                let measuredKs = Array(Set(qls + [1])).sorted()
                for L in contexts {
                    var msForwardByK: [Int: Double] = [:]
                    var trimByK: [Int: Bool] = [:]
                    for k in measuredKs {
                        let cache = ctx.model.newCache(parameters: nil)
                        // Chunked prefill of the first L ids.
                        var off = 0
                        while off < L {
                            let end = Swift.min(off + prefillStep, L)
                            let chunk = MLXArray(base[off ..< end].map { Int32($0) })
                            let o = ctx.model(
                                LMInput.Text(tokens: chunk)[text: .newAxis], cache: cache, state: nil)
                            eval(o.logits)
                            off = end
                        }
                        Stream().synchronize()
                        let canTrim = canTrimPromptCache(cache)
                        trimByK[k] = canTrim
                        let qTokens = MLXArray(base[L ..< L + k].map { Int32($0) })
                        func oneForward() {
                            let o = ctx.model(
                                LMInput.Text(tokens: qTokens)[text: .newAxis], cache: cache, state: nil)
                            eval(o.logits)
                        }
                        // Hold context at ~L by trimming the k just-appended tokens (if the
                        // cache supports it). Without trim, context drifts by k per iter —
                        // negligible at k<<L, recorded via `trimmable=false`.
                        for _ in 0 ..< warmup {
                            oneForward(); Stream().synchronize()
                            if canTrim { _ = trimPromptCache(cache, numTokens: k) }
                        }
                        var samples: [Double] = []
                        for _ in 0 ..< iters {
                            let t0 = Date.timeIntervalSinceReferenceDate
                            oneForward(); Stream().synchronize()
                            samples.append((Date.timeIntervalSinceReferenceDate - t0) * 1000.0)
                            if canTrim { _ = trimPromptCache(cache, numTokens: k) }
                        }
                        msForwardByK[k] = median(samples)
                    }
                    let ms1 = msForwardByK[1] ?? 0
                    for k in qls {
                        let msF = msForwardByK[k] ?? 0
                        let msQ = msF / Double(k)
                        out.append(
                            ScalingRow(
                                context: L, queryLength: k, msForward: msF, msPerQTok: msQ,
                                specCeiling: msQ > 0 ? ms1 / msQ : 0, trimmable: trimByK[k] ?? false))
                    }
                }
                return out
            }

            print("context   qLen   ms/forward   ms/qtok   specCeiling(k=1 vs k)   trim")
            print("-------   ----   ----------   -------   --------------------   ----")
            for r in rows {
                print(
                    [
                        String(format: "%7d", r.context), String(format: "%4d", r.queryLength),
                        String(format: "%10.3f", r.msForward), String(format: "%9.3f", r.msPerQTok),
                        String(format: "%18.3fx", r.specCeiling),
                        r.trimmable ? "  yes" : "   no",
                    ].joined(separator: "   "))
            }
            print(
                "\nspecCeiling = ms_forward(k=1) / ms_per_qtok(k): max tok/s multiplier ① could reach\n"
                    + "at width k with 100% acceptance. <1.0 => speculation cannot win at that context.")
            if let outputPath = arg("--output") {
                struct ScalingReport: Codable {
                    var schemaVersion = 1
                    var model: String
                    var iterations: Int
                    var warmup: Int
                    var rows: [ScalingRow]
                }
                let report = ScalingReport(
                    model: modelURL.lastPathComponent, iterations: iters, warmup: warmup, rows: rows)
                let enc = JSONEncoder()
                enc.outputFormatting = [.prettyPrinted, .sortedKeys]
                try enc.encode(report).write(to: URL(fileURLWithPath: outputPath), options: .atomic)
                print("\nwrote report: \(outputPath)")
            }
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

        // Plain greedy generated streams keyed by label — for before/after determinism diffs
        // of cadence-only changes (e.g. N6 recurrent-sync fold) on models whose cache is not
        // trimmable (so --validate-speculative cannot run, e.g. the Qwen3.5 hybrid).
        if let dumpPath = arg("--dump-plain") {
            let dump = Dictionary(
                uniqueKeysWithValues: runs.map { ($0.label, Array($0.tokens[$0.promptLength...])) })
            let enc = JSONEncoder()
            enc.outputFormatting = [.sortedKeys]
            try enc.encode(dump).write(to: URL(fileURLWithPath: dumpPath), options: .atomic)
            print("wrote plain token dump: \(dumpPath)")
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
