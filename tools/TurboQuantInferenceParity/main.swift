import Foundation
import IntegrationTestHelpers
import MLX
import MLXLLM
import MLXLMCommon

private struct IdentityTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.utf8.first.map(Int.init)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(UnicodeScalar(UInt8(truncatingIfNeeded: id)))
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            "\(message["role"] ?? ""): \(message["content"] ?? "")\n".utf8.map { Int($0) }
        }
    }
}

private struct IdentityTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        IdentityTokenizer()
    }
}

private func argumentString(_ name: String, default defaultValue: String? = nil) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    return arguments[index + 1]
}

private func argumentInt(_ name: String, default defaultValue: Int) -> Int {
    argumentString(name).flatMap(Int.init) ?? defaultValue
}

private func argumentFloat(_ name: String) -> Float? {
    argumentString(name).flatMap(Float.init)
}

private func argumentDouble(
    _ name: String,
    environmentName: String? = nil,
    default defaultValue: Double
) -> Double {
    if let value = argumentString(name).flatMap(Double.init) {
        return value
    }
    if let environmentName,
        let raw = ProcessInfo.processInfo.environment[environmentName],
        let value = Double(raw)
    {
        return value
    }
    return defaultValue
}

private func argumentInt(_ name: String) -> Int? {
    argumentString(name).flatMap(Int.init)
}

private func argumentUInt64(_ name: String, default defaultValue: UInt64) -> UInt64 {
    argumentString(name).flatMap(UInt64.init) ?? defaultValue
}

private func argumentInts(_ name: String, default defaultValue: [Int]) -> [Int] {
    guard let raw = argumentString(name) else { return defaultValue }
    let values = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return values.isEmpty ? defaultValue : values
}

private func argumentBool(_ name: String, environmentName: String? = nil) -> Bool {
    if CommandLine.arguments.contains(name) {
        return true
    }
    guard let environmentName else { return false }
    let raw = ProcessInfo.processInfo.environment[environmentName]?.lowercased()
    return raw == "1" || raw == "true" || raw == "yes"
}

private func printUsage() {
    let labels = InferenceParityBenchmark.defaultConfigLabels.joined(separator: ",")
    print(
        """
        TurboQuantInferenceParity

        Required:
          --model-dir <path>       MLX model directory, or set TQ_MODEL_DIR.

        Options:
          --contexts <csv>         Throughput contexts. Default: 4096,8192,16384
          --generate-tokens <n>    Decode tokens per throughput cell. Default: 32
          --throughput-repeats <n> Median-select N repeated throughput samples. Default: 1.
          --randomize-throughput-order
                                  Randomize context/config throughput sample order.
          --throughput-seed <n>    Deterministic throughput randomization seed.
          --throughput-cooldown <seconds>
                                  Cool down after each measured throughput cell. Default: 0.25.
          --quality-cooldown <seconds>
                                  Cool down after each quality candidate/reference pair. Default: 0.5.
          --configs <csv>          Config labels or aliases. Default: TQ_PARITY_CONFIGS or all defaults.
          --strict-configs         Fail on unknown --configs entries instead of warning and ignoring.
          --quality-gates          Run real-model logit quality gates after throughput.
          --quality-contexts <csv> Quality contexts. Default: min(max(contexts),4096)
          --quality-reference-config <label>
                                  Reference config for quality gates. Default: fp16.
          --quality-candidate-first
                                  Evaluate candidate before reference for long-context stress.
          --quality-logits-output <path>
                                  Dump CPU logits for one quality config/context and exit.
          --quality-logits-config <label>
                                  Config label for --quality-logits-output.
          --quality-logits-context <n>
                                  Context for --quality-logits-output.
          --quality-reference-logits <path>
          --quality-candidate-logits <path>
                                  Compare dumped quality logits and exit.
          --kv-layer-policy-json <path>
                                  Apply a calibrated KVLayerPolicy JSON to calibrated config labels.
          --emit-cache-policy-summary
                                  Include cache policy summaries in diagnostics JSON.
          --turboquant-timing     Include opt-in TurboQuant timing snapshots in diagnostics JSON.
          --diagnostics-output <path>
                                  Write throughput, quality, and per-layer attention diagnostics JSON.
          --diagnostics-samples-output <path>
                                  Append throughput/quality sample diagnostics JSONL as work completes.
          --sparse-v <mode>       Override Sparse-V for TurboQuant configs: off, threshold,
                                  topK, cumulativeMass, hybridCumulativeMassTopK,
                                  blockThreshold, pageTopK, candidateSparse.
          --sparse-v-threshold <f>
                                  Threshold mode value. Default: 1e-6.
          --sparse-v-top-k <n>    TopK mode value. Default: 256.
          --sparse-v-cumulative-mass <f>
                                  Cumulative mass fraction. Default: 0.995.
          --sparse-v-max-top-k <n>
                                  Hybrid TopK cap. Default: --sparse-v-top-k or 256.
          --sparse-v-recent-tokens <n>
                                  CandidateSparse recent-token window. Default: 256.
          --sparse-v-candidate-pages <n>
                                  CandidateSparse page-candidate count. Default: 4.
          --skip-throughput        Only run quality gates. Requires --quality-gates.
          --list-configs           Print available config labels and exit.
          --help                   Print this help.

        Available configs:
          \(labels)

        Useful aliases:
          k8v4, k8v3, k8v2, q8, affine_int4, turbo35, turbo4,
          polar_wht, polar_wht_reference, wht_reference_v3
        """
    )
}

private struct DiagnosticsOptions {
    var modelPath: String
    var contexts: [Int]
    var qualityContexts: [Int]
    var generateTokens: Int
    var configs: [InferenceParityBenchmark.CacheConfig]
    var runQualityGates: Bool
    var skipThroughput: Bool
    var throughputRepeats: Int
    var randomizeThroughputOrder: Bool
    var throughputSeed: UInt64
    var throughputCooldownSeconds: Double
    var qualityCooldownSeconds: Double
    var qualityReferenceLabel: String
    var qualityCandidateFirst: Bool
    var strictConfigs: Bool
    var sparseOverride: TurboQuantSparseValueSelection?
    var emitCachePolicySummary: Bool
    var turboQuantTimingEnabled: Bool
    var diagnosticsSamplesOutput: String?
    var memoryProfile: ModelMemoryProfile?
    var memoryProfileSource: String
}

private func gitCommit(_ relativePath: String) -> String {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = ["git", "-C", relativePath, "rev-parse", "HEAD"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = Pipe()
    do {
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return "unknown" }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty ?? "unknown"
    } catch {
        return "unknown"
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

private func writeStandardError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func uniqueSortedDiagnostics(_ values: [String]) -> [String] {
    Array(Set(values)).sorted()
}

private func diagnosticFallbackReasons(
    _ diagnostics: [TurboQuantAttentionDiagnostics]
) -> [String] {
    uniqueSortedDiagnostics(diagnostics.compactMap { $0.lastFallback?.reason })
}

private func diagnosticUnsupportedShapes(
    _ diagnostics: [TurboQuantAttentionDiagnostics]
) -> [String] {
    uniqueSortedDiagnostics(diagnostics.compactMap(\.lastUnsupportedShape))
}

private func diagnosticNativeFallbackReasons(
    _ diagnostics: [TurboQuantAttentionDiagnostics]
) -> [String] {
    uniqueSortedDiagnostics(diagnostics.compactMap(\.nativeFallbackReason))
}

private final class DiagnosticsJSONLWriter: @unchecked Sendable {
    private let fileHandle: FileHandle
    private let encoder = JSONEncoder()
    private let lock = NSLock()

    init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data().write(to: url, options: .atomic)
        self.fileHandle = try FileHandle(forWritingTo: url)
    }

    deinit {
        try? fileHandle.close()
    }

    func writeThroughputSample(
        _ sample: InferenceParityBenchmark.ThroughputSample
    ) throws {
        try append(ThroughputSampleEvent(sample: sample))
    }

    func writeQualityAttempt(
        _ attempt: InferenceParityBenchmark.QualityAttempt
    ) throws {
        try append(QualityAttemptEvent(attempt: attempt))
    }

    private func append<Event: Encodable>(_ event: Event) throws {
        var data = try encoder.encode(event)
        data.append(0x0A)
        lock.lock()
        defer { lock.unlock() }
        fileHandle.seekToEndOfFile()
        fileHandle.write(data)
        fileHandle.synchronizeFile()
    }
}

private struct ThroughputSampleEvent: Encodable {
    var schemaVersion: Int = 1
    var generatedAt: String = ISO8601DateFormatter().string(from: Date())
    var kind: String = "throughputSample"
    var context: Int
    var label: String
    var sampleIndex: Int
    var sampleCount: Int
    var status: String
    var error: String?
    var decodeTokensPerSecond: Double?
    var prefillTokensPerSecond: Double?
    var generationSeconds: Double?
    var generationLoopSeconds: Double?
    var generationSynchronizationSeconds: Double?
    var promptPrefillSeconds: Double?
    var generationTokenCount: Int?
    var promptPrefillTiming: TurboQuantTimingSnapshot?
    var generationTiming: TurboQuantTimingSnapshot?
    var estimatedRawKVBytes: Int?
    var estimatedConfigKVBytes: Int?
    var estimatedMemoryReductionRatio: Double?
    var memoryStart: Memory.Snapshot?
    var memoryEnd: Memory.Snapshot?
    var peakActiveMemoryBytes: Int?
    var codecCounts: [String: Int]?
    var boundaryProtectedLayerCount: Int?
    var residualCorrectionActive: Bool?
    var nativeKernelKinds: [Int]?
    var sparseSkippedTokens: Int?
    var sparseTotalTokens: Int?
    var sparseRecentTokenCount: Int?
    var sparseOlderTokenCount: Int?
    var sparsePageCandidateCount: Int?
    var sparseSkipRatio: Double?
    var sparseRequestedLayerCount: Int?
    var sparseActiveLayerCount: Int?
    var sparseRequestedButInactive: Bool?
    var sparseFallbackReason: String?
    var fallbackReasons: [String]?
    var unsupportedShapes: [String]?
    var nativeFallbackReasons: [String]?

    init(sample: InferenceParityBenchmark.ThroughputSample) {
        let measurement = sample.measurement
        let diagnostics = measurement?.attentionDiagnostics ?? []
        self.context = sample.context
        self.label = sample.label
        self.sampleIndex = sample.sampleIndex
        self.sampleCount = sample.sampleCount
        self.status = sample.status.rawValue
        self.error = sample.error
        self.decodeTokensPerSecond = measurement?.decodeTokensPerSecond
        self.prefillTokensPerSecond = measurement?.prefillTokensPerSecond
        self.generationSeconds = measurement?.generationSeconds
        self.generationLoopSeconds = measurement?.generationLoopSeconds
        self.generationSynchronizationSeconds = measurement?.generationSynchronizationSeconds
        self.promptPrefillSeconds = measurement?.promptPrefillSeconds
        self.generationTokenCount = measurement?.generationTokenCount
        self.promptPrefillTiming = measurement?.promptPrefillTiming
        self.generationTiming = measurement?.generationTiming
        self.estimatedRawKVBytes = measurement?.estimatedRawKVBytes
        self.estimatedConfigKVBytes = measurement?.estimatedConfigKVBytes
        self.estimatedMemoryReductionRatio = measurement?.estimatedMemoryReductionRatio
        self.memoryStart = measurement?.memoryStart
        self.memoryEnd = measurement?.memoryEnd
        self.peakActiveMemoryBytes = measurement?.peakActiveMemoryBytes
        self.codecCounts = measurement?.codecCounts
        self.boundaryProtectedLayerCount = measurement?.boundaryProtectedLayerCount
        self.residualCorrectionActive = measurement?.residualCorrectionActive
        self.nativeKernelKinds = measurement?.nativeKernelKinds
        self.sparseSkippedTokens = measurement?.sparseSkippedTokens
        self.sparseTotalTokens = measurement?.sparseTotalTokens
        self.sparseRecentTokenCount = measurement?.sparseRecentTokenCount
        self.sparseOlderTokenCount = measurement?.sparseOlderTokenCount
        self.sparsePageCandidateCount = measurement?.sparsePageCandidateCount
        self.sparseSkipRatio = measurement?.sparseSkipRatio
        self.sparseRequestedLayerCount = measurement?.sparseRequestedLayerCount
        self.sparseActiveLayerCount = measurement?.sparseActiveLayerCount
        self.sparseRequestedButInactive = measurement?.sparseRequestedButInactive
        self.sparseFallbackReason = diagnostics.compactMap {
            $0.lastFallback?.reason ?? $0.lastUnsupportedShape ?? $0.nativeFallbackReason
        }.first
        self.fallbackReasons = measurement == nil ? nil : diagnosticFallbackReasons(diagnostics)
        self.unsupportedShapes = measurement == nil ? nil : diagnosticUnsupportedShapes(diagnostics)
        self.nativeFallbackReasons =
            measurement == nil ? nil : diagnosticNativeFallbackReasons(diagnostics)
    }
}

private struct QualityAttemptEvent: Encodable {
    var schemaVersion: Int = 1
    var generatedAt: String = ISO8601DateFormatter().string(from: Date())
    var kind: String = "qualityAttempt"
    var context: Int
    var label: String
    var referenceLabel: String
    var candidateFirst: Bool
    var status: String
    var error: String?
    var deterministicTop1MatchRate: Double?
    var logitKLDivergenceMean: Double?
    var logitMaxAbsErrorP95: Double?
    var cosine: Double?
    var passed: Bool?
    var reason: String?
    var report: TurboQuantQualityGateReport?

    init(attempt: InferenceParityBenchmark.QualityAttempt) {
        let quality = attempt.measurement?.quality
        self.context = attempt.context
        self.label = attempt.label
        self.referenceLabel = attempt.referenceLabel
        self.candidateFirst = attempt.candidateFirst
        self.status = attempt.status.rawValue
        self.error = attempt.error
        self.deterministicTop1MatchRate = quality?.deterministicTop1MatchRate
        self.logitKLDivergenceMean = quality?.logitKLDivergenceMean
        self.logitMaxAbsErrorP95 = quality?.logitMaxAbsErrorP95
        self.cosine = quality?.attentionOutputCosineMean
        self.passed = quality?.passed
        self.reason = quality?.gateReason
        self.report = quality
    }
}

/// Benchmark memory guard. Long-context batteries wire enough unified GPU
/// memory (KV planes + MLX buffer cache held across 25-sample loops) to
/// freeze the whole machine on small-RAM hosts (observed on 18 GB). Caps are
/// applied identically to every config/arm, so A/B comparisons stay fair,
/// and the applied values are echoed to stderr so artifacts record them.
/// TQ_METAL_MEMORY_LIMIT_MB (default 60% of physical RAM, relaxed) and
/// TQ_METAL_CACHE_LIMIT_MB (default 1024). Set -1 to disable a guard.
func applyBenchmarkMemoryGuard() {
    let env = ProcessInfo.processInfo.environment
    func megabytes(_ name: String, default def: Int) -> Int {
        env[name].flatMap { Int($0) } ?? def
    }
    let physicalMB = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
    let memoryMB = megabytes("TQ_METAL_MEMORY_LIMIT_MB", default: physicalMB * 60 / 100)
    let cacheMB = megabytes("TQ_METAL_CACHE_LIMIT_MB", default: 1024)
    if memoryMB >= 0 { GPU.set(memoryLimit: memoryMB * 1024 * 1024, relaxed: true) }
    if cacheMB >= 0 { GPU.set(cacheLimit: cacheMB * 1024 * 1024) }
    FileHandle.standardError.write(
        Data(
            "memory guard: metal memoryLimit=\(memoryMB >= 0 ? "\(memoryMB)MB (relaxed)" : "off") cacheLimit=\(cacheMB >= 0 ? "\(cacheMB)MB" : "off") physical=\(physicalMB)MB\n"
                .utf8))
}

@main
struct TurboQuantInferenceParityCLI {
    static func main() async throws {
        let environment = ProcessInfo.processInfo.environment
        if CommandLine.arguments.contains("--help") {
            printUsage()
            return
        }
        applyBenchmarkMemoryGuard()
        if CommandLine.arguments.contains("--list-configs") {
            for label in InferenceParityBenchmark.defaultConfigLabels {
                print(label)
            }
            return
        }
        let contexts = argumentInts("--contexts", default: [4096, 8192, 16384])
        let generateTokens = argumentInt("--generate-tokens", default: 32)
        let throughputRepeats = max(1, argumentInt("--throughput-repeats", default: 1))
        let randomizeThroughputOrder = argumentBool(
            "--randomize-throughput-order",
            environmentName: "TQ_PARITY_RANDOMIZE_THROUGHPUT_ORDER"
        )
        let throughputSeed = argumentUInt64(
            "--throughput-seed",
            default: 0x5451_2026_0602
        )
        let throughputCooldownSeconds = max(
            0,
            argumentDouble(
                "--throughput-cooldown",
                environmentName: "TQ_PARITY_THROUGHPUT_COOLDOWN",
                default: 0.25
            )
        )
        let qualityCooldownSeconds = max(
            0,
            argumentDouble(
                "--quality-cooldown",
                environmentName: "TQ_PARITY_QUALITY_COOLDOWN",
                default: 0.5
            )
        )
        let qualityContexts = argumentInts(
            "--quality-contexts",
            default: [min(contexts.max() ?? 4096, 4096)]
        )
        let runQualityGates = argumentBool(
            "--quality-gates",
            environmentName: "TQ_PARITY_QUALITY_GATES"
        )
        let skipThroughput = argumentBool("--skip-throughput")
        let qualityCandidateFirst = argumentBool("--quality-candidate-first")
        let qualityLogitsOutput = argumentString("--quality-logits-output")
        let qualityLogitsConfigLabel = argumentString("--quality-logits-config")
        let qualityLogitsContext = argumentInt("--quality-logits-context")
        let qualityReferenceLogitsPath = argumentString("--quality-reference-logits")
        let qualityCandidateLogitsPath = argumentString("--quality-candidate-logits")
        let strictConfigs = argumentBool(
            "--strict-configs",
            environmentName: "TQ_PARITY_STRICT_CONFIGS"
        )
        let explicitConfigs = try argumentString("--configs").flatMap {
            try InferenceParityBenchmark.configs(fromCSV: $0, strict: strictConfigs)
        }
        let environmentConfigs = InferenceParityBenchmark.configsFromEnvironment(environment)
        let kvLayerPolicy = try argumentString("--kv-layer-policy-json").map {
            try loadKVLayerPolicy(path: $0)
        }
        let emitCachePolicySummary = argumentBool("--emit-cache-policy-summary")
        let turboQuantTimingEnabled = argumentBool(
            "--turboquant-timing",
            environmentName: "TQ_TURBOQUANT_TIMING"
        )
        let sparseOverrideMode = argumentString("--sparse-v")
            ?? argumentString("--sparse-v-mode")
        let sparseOverride = InferenceParityBenchmark.sparseValueSelection(
            mode: sparseOverrideMode,
            threshold: argumentFloat("--sparse-v-threshold"),
            topK: argumentInt("--sparse-v-top-k"),
            cumulativeMass: argumentFloat("--sparse-v-cumulative-mass")
                ?? argumentFloat("--sparse-v-mass"),
            maxTopK: argumentInt("--sparse-v-max-top-k")
                ?? argumentInt("--sparse-v-hybrid-top-k"),
            recentTokens: argumentInt("--sparse-v-recent-tokens"),
            candidatePages: argumentInt("--sparse-v-candidate-pages")
        )
        let benchmarkConfigs = InferenceParityBenchmark.applyingSparseValueSelectionOverride(
            sparseOverride,
            to: applyKVLayerPolicy(kvLayerPolicy, to: explicitConfigs ?? environmentConfigs)
        )
        let resolvedBenchmarkConfigs = benchmarkConfigs ?? InferenceParityBenchmark.defaultConfigs
        let qualityReference = try argumentString("--quality-reference-config").flatMap {
            try InferenceParityBenchmark.configs(fromCSV: $0, strict: strictConfigs)?.first
        }
        let resolvedQualityReference =
            qualityReference
            ?? InferenceParityBenchmark.CacheConfig(label: "fp16", strategy: .none, preset: nil)
        let diagnosticsOutput = argumentString("--diagnostics-output")
        let diagnosticsSamplesOutput = argumentString("--diagnostics-samples-output")
        let diagnosticsStream = try diagnosticsSamplesOutput.map { path in
            try DiagnosticsJSONLWriter(path: path)
        }

        if let qualityReferenceLogitsPath, let qualityCandidateLogitsPath {
            let measurement = try InferenceParityBenchmark.qualityMeasurement(
                referenceLogitsPath: qualityReferenceLogitsPath,
                candidateLogitsPath: qualityCandidateLogitsPath
            )
            printQualityMeasurement(measurement, referenceLabel: "dumped-reference")
            let attempt = InferenceParityBenchmark.QualityAttempt(
                context: measurement.context,
                label: measurement.label,
                referenceLabel: measurement.referenceLabel,
                candidateFirst: measurement.candidateFirst,
                status: .ok,
                measurement: measurement,
                error: nil
            )
            try diagnosticsStream?.writeQualityAttempt(attempt)
            if let diagnosticsOutput {
                let options = DiagnosticsOptions(
                    modelPath: "quality-logits",
                    contexts: [],
                    qualityContexts: [measurement.context],
                    generateTokens: 0,
                    configs: [],
                    runQualityGates: true,
                    skipThroughput: true,
                    throughputRepeats: 0,
                    randomizeThroughputOrder: false,
                    throughputSeed: 0,
                    throughputCooldownSeconds: 0,
                    qualityCooldownSeconds: 0,
                    qualityReferenceLabel: measurement.referenceLabel,
                    qualityCandidateFirst: measurement.candidateFirst,
                    strictConfigs: strictConfigs,
                    sparseOverride: sparseOverride,
                    emitCachePolicySummary: false,
                    turboQuantTimingEnabled: turboQuantTimingEnabled,
                    diagnosticsSamplesOutput: diagnosticsSamplesOutput,
                    memoryProfile: nil,
                    memoryProfileSource: "unavailable: dumped-logit comparison"
                )
                try writeDiagnosticsReport(
                    to: diagnosticsOutput,
                    options: options,
                    throughputRun: InferenceParityBenchmark.ThroughputRunResult(),
                    qualityRun: InferenceParityBenchmark.QualityRunResult(
                        measurements: [measurement],
                        attempts: [attempt]
                    )
                )
            }
            return
        }

        guard let modelPath = argumentString("--model-dir", default: environment["TQ_MODEL_DIR"]) else {
            throw IntegrationTestFailure(
                "missing model directory; pass --model-dir or set TQ_MODEL_DIR")
        }
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let memoryProfile: ModelMemoryProfile?
        let memoryProfileSource: String
        do {
            memoryProfile = try ModelMemoryProfile.profile(
                modelDirectory: modelURL,
                modelID: modelURL.lastPathComponent
            )
            memoryProfileSource = "ModelMemoryProfile.profile(modelDirectory:modelID:)"
        } catch {
            memoryProfile = nil
            memoryProfileSource = "unavailable: \(error)"
        }
        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: IdentityTokenizerLoader()
        )
        if let qualityLogitsOutput {
            let configLabel =
                qualityLogitsConfigLabel
                ?? explicitConfigs?.first?.label
                ?? qualityReference?.label
                ?? "fp16"
            guard let config = InferenceParityBenchmark.config(named: configLabel) else {
                throw IntegrationTestFailure("unknown quality logits config '\(configLabel)'")
            }
            let context = qualityLogitsContext ?? qualityContexts.first ?? min(contexts.max() ?? 4096, 4096)
            let artifact = try await InferenceParityBenchmark.writeQualityLogits(
                container: container,
                contextLength: context,
                config: config,
                outputPath: qualityLogitsOutput
            )
            print(
                "wrote quality logits: \(qualityLogitsOutput) "
                    + "(ctx=\(artifact.context), config=\(artifact.label), rowWidth=\(artifact.rowWidth))"
            )
            return
        }
        var throughputRun = InferenceParityBenchmark.ThroughputRunResult()
        var qualityRun = InferenceParityBenchmark.QualityRunResult()
        if !skipThroughput {
            throughputRun = try await InferenceParityBenchmark.runDetailed(
                container: container,
                contexts: contexts,
                generateTokens: generateTokens,
                configs: resolvedBenchmarkConfigs,
                throughputRepeats: throughputRepeats,
                randomizeOrder: randomizeThroughputOrder,
                randomSeed: throughputSeed,
                memoryProfile: memoryProfile,
                cooldownSeconds: throughputCooldownSeconds,
                turboQuantTimingEnabled: turboQuantTimingEnabled,
                sampleObserver: { sample in
                    do {
                        try diagnosticsStream?.writeThroughputSample(sample)
                    } catch {
                        writeStandardError("failed to write throughput sample diagnostics: \(error)")
                    }
                }
            )
        }
        if runQualityGates {
            qualityRun = try await InferenceParityBenchmark.runQualityGatesDetailed(
                container: container,
                contexts: qualityContexts,
                configs: resolvedBenchmarkConfigs,
                referenceConfig: resolvedQualityReference,
                candidateFirst: qualityCandidateFirst,
                cooldownSeconds: qualityCooldownSeconds,
                failFast: false,
                attemptObserver: { attempt in
                    do {
                        try diagnosticsStream?.writeQualityAttempt(attempt)
                    } catch {
                        writeStandardError("failed to write quality attempt diagnostics: \(error)")
                    }
                }
            )
        }
        if let diagnosticsOutput {
            let options = DiagnosticsOptions(
                modelPath: modelPath,
                contexts: contexts,
                qualityContexts: qualityContexts,
                generateTokens: generateTokens,
                configs: resolvedBenchmarkConfigs,
                runQualityGates: runQualityGates,
                skipThroughput: skipThroughput,
                throughputRepeats: throughputRepeats,
                randomizeThroughputOrder: randomizeThroughputOrder,
                throughputSeed: throughputSeed,
                throughputCooldownSeconds: throughputCooldownSeconds,
                qualityCooldownSeconds: qualityCooldownSeconds,
                qualityReferenceLabel: resolvedQualityReference.label,
                qualityCandidateFirst: qualityCandidateFirst,
                strictConfigs: strictConfigs,
                sparseOverride: sparseOverride,
                emitCachePolicySummary: emitCachePolicySummary,
                turboQuantTimingEnabled: turboQuantTimingEnabled,
                diagnosticsSamplesOutput: diagnosticsSamplesOutput,
                memoryProfile: memoryProfile,
                memoryProfileSource: memoryProfileSource
            )
            try writeDiagnosticsReport(
                to: diagnosticsOutput,
                options: options,
                throughputRun: throughputRun,
                qualityRun: qualityRun
            )
        }
        Stream().synchronize()
    }

    private static func printQualityMeasurement(
        _ measurement: InferenceParityBenchmark.QualityMeasurement,
        referenceLabel: String
    ) {
        let quality = measurement.quality
        print(
            "\n=== TurboQuant dumped-logit quality gate "
                + "(\(measurement.label) vs \(referenceLabel)) ==="
        )
        print("ctx       config          top1    kl_mean      p95_abs     cosine    passed")
        print("-------   -------------   -----   ----------   ---------   -------   ------")
        let top1 = String(format: "%.3f", quality.deterministicTop1MatchRate)
        let kl = String(format: "%.6f", quality.logitKLDivergenceMean)
        let p95 = String(format: "%.4f", quality.logitMaxAbsErrorP95)
        let cosine = quality.attentionOutputCosineMean.map {
            String(format: "%.4f", $0)
        } ?? "--"
        let passed = quality.passed ? "yes" : "no"
        print(
            [
                pad("\(measurement.context)", 7),
                pad(measurement.label, 13),
                pad(top1, 5),
                pad(kl, 10),
                pad(p95, 9),
                pad(cosine, 7),
                passed,
            ].joined(separator: "   "))
        if let reason = quality.gateReason {
            print("          reason: \(reason)")
        }
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        s + String(repeating: " ", count: max(0, width - s.count))
    }

    private static func loadKVLayerPolicy(path: String) throws -> KVLayerPolicy {
        let url = URL(fileURLWithPath: path)
        let policy = try JSONDecoder().decode(KVLayerPolicy.self, from: Data(contentsOf: url))
        try policy.validate()
        return policy
    }

    private static func applyKVLayerPolicy(
        _ policy: KVLayerPolicy?,
        to configs: [InferenceParityBenchmark.CacheConfig]?
    ) -> [InferenceParityBenchmark.CacheConfig]? {
        guard let policy else { return configs }
        let policyIsResidual = policyContainsResidual(policy)
        let source = configs ?? InferenceParityBenchmark.defaultConfigs
        return source.map { config in
            let label = config.label.lowercased()
            guard label.contains("calibrated") else { return config }
            let labelIsResidual = label.contains("residual")
            return labelIsResidual == policyIsResidual ? config.withKVLayerPolicy(policy) : config
        }
    }

    private static func policyContainsResidual(_ policy: KVLayerPolicy) -> Bool {
        if case .affineK8VxResidual = policy.defaultCodec {
            return true
        }
        return policy.rules.contains {
            if case .affineK8VxResidual = $0.codec {
                return true
            }
            return false
        }
    }

    private static func writeDiagnosticsReport(
        to path: String,
        options: DiagnosticsOptions,
        throughputRun: InferenceParityBenchmark.ThroughputRunResult,
        qualityRun: InferenceParityBenchmark.QualityRunResult
    ) throws {
        struct RunMetadata: Encodable {
            var commandLine: [String]
            var contexts: [Int]
            var qualityContexts: [Int]
            var configLabels: [String]
            var generateTokens: Int
            var runQualityGates: Bool
            var skipThroughput: Bool
            var throughputRepeats: Int
            var randomizeThroughputOrder: Bool
            var throughputSeed: UInt64
            var throughputCooldownSeconds: Double
            var qualityCooldownSeconds: Double
            var qualityReferenceLabel: String
            var qualityCandidateFirst: Bool
            var strictConfigs: Bool
            var sparseOverride: TurboQuantSparseValueSelection?
            var emitCachePolicySummary: Bool
            var turboQuantTimingEnabled: Bool
            var diagnosticsSamplesOutput: String?
        }
        struct WorkloadMetadata: Encodable {
            var throughputPromptKind: String
            var qualityPromptKind: String
            var promptTokenPattern: String
            var promptContentSemanticRole: String
            var tokenizer: String
            var decodeLoop: String
            var qualityDecodeTokens: Int
        }
        struct MemoryProfileMetadata: Encodable {
            var estimateSource: String
            var modelProfile: ModelMemoryProfile?
        }
        struct ConfigDescriptor: Encodable {
            var label: String
            var isFP16Baseline: Bool
            var strategy: String
            var preset: String?
            var kvBits: Int?
            var kvGroupSize: Int?
            var kvCodec: String?
            var turboQuantBackend: String?
            var valueBits: Int?
            var valueGroupSize: Int?
            var runtimeMode: String?
            var quantizedKVStart: Int?
            var precisionPolicy: TurboQuantKVPrecisionPolicy?
            var kvLayerPolicy: KVLayerPolicy?
            var kvLayerPolicySummary: String?
            var sparseValueSelection: TurboQuantSparseValueSelection

            init(_ config: InferenceParityBenchmark.CacheConfig) {
                self.label = config.label
                self.isFP16Baseline = config.strategy == .none
                self.strategy = config.strategy.rawValue
                self.preset = config.preset?.rawValue
                self.kvBits = config.kvBits
                self.kvGroupSize = config.kvGroupSize
                self.kvCodec = config.kvCodec?.rawValue
                self.turboQuantBackend = config.turboQuantBackend?.rawValue
                self.valueBits = config.valueBits
                self.valueGroupSize = config.valueGroupSize
                self.runtimeMode = config.runtimeMode?.rawValue
                self.quantizedKVStart = config.quantizedKVStart
                self.precisionPolicy = config.precisionPolicy
                self.kvLayerPolicy = config.kvLayerPolicy
                self.kvLayerPolicySummary = config.kvLayerPolicy?.summary()
                self.sparseValueSelection = config.sparseValueSelection
            }
        }
        struct Distribution: Encodable {
            var count: Int
            var min: Double
            var p50: Double
            var p95: Double
            var max: Double
            var mean: Double
            var standardDeviation: Double

            static func make(_ values: [Double]) -> Distribution? {
                let values = values.filter(\.isFinite)
                guard !values.isEmpty else { return nil }
                let sorted = values.sorted()
                let mean = sorted.reduce(0, +) / Double(sorted.count)
                let variance =
                    sorted.map { pow($0 - mean, 2) }.reduce(0, +) / Double(sorted.count)
                let p95Index = Swift.min(
                    sorted.count - 1,
                    Swift.max(0, Int((0.95 * Double(sorted.count)).rounded(.up)) - 1)
                )
                return Distribution(
                    count: sorted.count,
                    min: sorted[0],
                    p50: sorted[sorted.count / 2],
                    p95: sorted[p95Index],
                    max: sorted[sorted.count - 1],
                    mean: mean,
                    standardDeviation: sqrt(variance)
                )
            }
        }
        struct ThroughputCell: Encodable {
            var status: String
            var context: Int
            var label: String
            var sampleCount: Int
            var successfulSampleCount: Int
            var failedSampleCount: Int
            var selectedSampleIndex: Int
            var decodeTokensPerSecond: Double
            var prefillTokensPerSecond: Double
            var generationSeconds: Double
            var generationLoopSeconds: Double
            var generationSynchronizationSeconds: Double
            var promptPrefillSeconds: Double
            var decodeTokensPerSecondDistribution: Distribution?
            var prefillTokensPerSecondDistribution: Distribution?
            var generationSecondsDistribution: Distribution?
            var generationTokenCount: Int
            var speedRatioToFP16: Double?
            var speedRatioToAffineK8V4: Double?
            var estimatedRawKVBytes: Int?
            var estimatedConfigKVBytes: Int?
            var estimatedMemoryReductionRatio: Double?
            var residentKVCompressionRatio: Double?
            var estimatedMemoryReductionPercent: Double?
            var estimatedMemorySource: String
            var memoryStart: Memory.Snapshot
            var memoryEnd: Memory.Snapshot
            var peakActiveMemoryBytes: Int
            var steadyActiveMemoryBytes: Int
            var activeMemoryDeltaBytes: Int
            var cacheMemoryDeltaBytes: Int
            var cachePolicySummary: String?
            var promptPrefillTiming: TurboQuantTimingSnapshot?
            var generationTiming: TurboQuantTimingSnapshot?
            var requestedBackend: String?
            var selectedAttentionPaths: [String]
            var codecCounts: [String: Int]
            var boundaryProtectedLayerCount: Int
            var valueBits: Int?
            var valueGroupSize: Int?
            var residualCorrectionActive: Bool
            var nativeKernelKinds: [Int]
            // TEST-ONLY observability: Swift segmented/block-parallel dispatched-kernel
            // counts (captured around the timed decode loop). Proves whether the coop
            // kernel (`..._coop..`/`..._coopw..`) actually ran. The Measurement already
            // carries this; surface it in the report JSON for the coop A/B measurement.
            var swiftDispatchedKernels: [String: Int]?
            var sparseSkippedTokens: Int
            var sparseTotalTokens: Int
            var sparseRecentTokenCount: Int
            var sparseOlderTokenCount: Int
            var sparsePageCandidateCount: Int
            var sparseSkipRatio: Double?
            var sparseRequestedLayerCount: Int
            var sparseActiveLayerCount: Int
            var sparseRequestedButInactive: Bool
            var sparseFallbackReason: String?
            var fallbackReasons: [String]
            var unsupportedShapes: [String]
            var nativeFallbackReasons: [String]
            var qualityPassed: Bool?
            var qualityReason: String?
            var promotionEligible: Bool
            var promotionBlockReasons: [String]
            var promotionGate: InferenceParityBenchmark.PromotionGate
            var layers: [TurboQuantAttentionDiagnostics]
        }
        struct ThroughputSampleCell: Encodable {
            var context: Int
            var label: String
            var sampleIndex: Int
            var sampleCount: Int
            var status: String
            var error: String?
            var decodeTokensPerSecond: Double?
            var prefillTokensPerSecond: Double?
            var generationSeconds: Double?
            var generationLoopSeconds: Double?
            var generationSynchronizationSeconds: Double?
            var promptPrefillSeconds: Double?
            var generationTokenCount: Int?
            var promptPrefillTiming: TurboQuantTimingSnapshot?
            var generationTiming: TurboQuantTimingSnapshot?
            var estimatedRawKVBytes: Int?
            var estimatedConfigKVBytes: Int?
            var estimatedMemoryReductionRatio: Double?
            var memoryStart: Memory.Snapshot?
            var memoryEnd: Memory.Snapshot?
            var peakActiveMemoryBytes: Int?
            var codecCounts: [String: Int]?
            var nativeKernelKinds: [Int]?
            // TEST-ONLY observability: see ThroughputCell.swiftDispatchedKernels.
            var swiftDispatchedKernels: [String: Int]?
            var sparseSkipRatio: Double?
            var sparseRequestedLayerCount: Int?
            var sparseActiveLayerCount: Int?
            var sparseRequestedButInactive: Bool?
        }
        struct QualityCell: Encodable {
            var status: String
            var context: Int
            var label: String
            var referenceLabel: String
            var candidateFirst: Bool
            var deterministicTop1MatchRate: Double
            var logitKLDivergenceMean: Double
            var logitMaxAbsErrorP95: Double
            var cosine: Double?
            var passed: Bool
            var reason: String?
            var selectedAttentionPaths: [String]
            var codecCounts: [String: Int]
            var rawFallbackAllocated: Bool
            var fallbackReasons: [String]
            var report: TurboQuantQualityGateReport
        }
        struct QualityAttemptCell: Encodable {
            var context: Int
            var label: String
            var referenceLabel: String
            var candidateFirst: Bool
            var status: String
            var error: String?
            var selectedAttentionPaths: [String]?
            var codecCounts: [String: Int]?
            var report: TurboQuantQualityGateReport?
        }
        struct Report: Encodable {
            var schemaVersion: Int
            var modelPath: String
            var generatedAt: String
            var repoCommits: [String: String]
            var run: RunMetadata
            var workload: WorkloadMetadata
            var memory: MemoryProfileMetadata
            var configurations: [ConfigDescriptor]
            var throughput: [ThroughputCell]
            var throughputSamples: [ThroughputSampleCell]
            var quality: [QualityCell]
            var qualityAttempts: [QualityAttemptCell]
        }
        func key(context: Int, label: String) -> String {
            "\(context)\u{1F}\(label)"
        }
        func uniqueSorted(_ values: [String]) -> [String] {
            Array(Set(values)).sorted()
        }
        func fallbackReasons(_ diagnostics: [TurboQuantAttentionDiagnostics]) -> [String] {
            uniqueSorted(diagnostics.compactMap { $0.lastFallback?.reason })
        }
        func unsupportedShapes(_ diagnostics: [TurboQuantAttentionDiagnostics]) -> [String] {
            uniqueSorted(diagnostics.compactMap(\.lastUnsupportedShape))
        }
        func nativeFallbackReasons(_ diagnostics: [TurboQuantAttentionDiagnostics]) -> [String] {
            uniqueSorted(diagnostics.compactMap(\.nativeFallbackReason))
        }
        let throughputResults = throughputRun.measurements
        let qualityResults = qualityRun.measurements
        let configByLabel = Dictionary(uniqueKeysWithValues: options.configs.map {
            ($0.label, $0)
        })
        let qualityByCell = Dictionary(uniqueKeysWithValues: qualityResults.map {
            (key(context: $0.context, label: $0.label), $0)
        })
        let fp16ByContext = Dictionary(
            uniqueKeysWithValues: throughputResults
                .filter { $0.label == "fp16" }
                .map { ($0.context, $0.decodeTokensPerSecond) }
        )
        let fp16MeasurementByContext = Dictionary(
            uniqueKeysWithValues: throughputResults
                .filter { $0.label == "fp16" }
                .map { ($0.context, $0) }
        )
        let affineK8V4ByContext = Dictionary(
            uniqueKeysWithValues: throughputResults
                .filter { $0.label == "affineK8V4" }
                .map { ($0.context, $0.decodeTokensPerSecond) }
        )
        let affineK8V4MeasurementByContext = Dictionary(
            uniqueKeysWithValues: throughputResults
                .filter { $0.label == "affineK8V4" }
                .map { ($0.context, $0) }
        )
        let samplesByCell = Dictionary(grouping: throughputRun.samples) {
            key(context: $0.context, label: $0.label)
        }
        func speedRatio(_ measurement: InferenceParityBenchmark.Measurement, to baseline: Double?)
            -> Double?
        {
            guard let baseline, baseline > 0 else { return nil }
            return measurement.decodeTokensPerSecond / baseline
        }
        func memoryReductionPercent(_ measurement: InferenceParityBenchmark.Measurement) -> Double? {
            guard let ratio = measurement.estimatedMemoryReductionRatio, ratio > 0 else {
                return nil
            }
            return (1 - (1 / ratio)) * 100
        }

        let report = Report(
            schemaVersion: 3,
            modelPath: options.modelPath,
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            repoCommits: [
                "mlx-swift-lm": gitCommit("."),
                "mlx-swift": gitCommit(".build/checkouts/mlx-swift"),
            ],
            run: RunMetadata(
                commandLine: CommandLine.arguments,
                contexts: options.contexts,
                qualityContexts: options.qualityContexts,
                configLabels: options.configs.map(\.label),
                generateTokens: options.generateTokens,
                runQualityGates: options.runQualityGates,
                skipThroughput: options.skipThroughput,
                throughputRepeats: options.throughputRepeats,
                randomizeThroughputOrder: options.randomizeThroughputOrder,
                throughputSeed: options.throughputSeed,
                throughputCooldownSeconds: options.throughputCooldownSeconds,
                qualityCooldownSeconds: options.qualityCooldownSeconds,
                qualityReferenceLabel: options.qualityReferenceLabel,
                qualityCandidateFirst: options.qualityCandidateFirst,
                strictConfigs: options.strictConfigs,
                sparseOverride: options.sparseOverride,
                emitCachePolicySummary: options.emitCachePolicySummary,
                turboQuantTimingEnabled: options.turboQuantTimingEnabled,
                diagnosticsSamplesOutput: options.diagnosticsSamplesOutput
            ),
            workload: WorkloadMetadata(
                throughputPromptKind: "synthetic-in-vocab exact-length prompt",
                qualityPromptKind: "synthetic-in-vocab exact-length prompt",
                promptTokenPattern: "Int32(index % 1024 + 16)",
                promptContentSemanticRole:
                    "content is irrelevant for speed; context length determines KV depth",
                tokenizer: "IdentityTokenizer",
                decodeLoop: "TokenIterator greedy decode",
                qualityDecodeTokens: 1
            ),
            memory: MemoryProfileMetadata(
                estimateSource: options.memoryProfileSource,
                modelProfile: options.memoryProfile
            ),
            configurations: options.configs.map(ConfigDescriptor.init),
            throughput: throughputResults.map { measurement in
                let cellSamples =
                    samplesByCell[key(context: measurement.context, label: measurement.label)] ?? []
                let successfulMeasurements = cellSamples.compactMap(\.measurement)
                let config = configByLabel[measurement.label]
                let quality = qualityByCell[
                    key(context: measurement.context, label: measurement.label)
                ]
                let promotionGate = InferenceParityBenchmark.promotionGate(
                    measurement: measurement,
                    config: config,
                    quality: quality,
                    runQualityGates: options.runQualityGates,
                    fp16Baseline: fp16MeasurementByContext[measurement.context],
                    affineK8V4Baseline: affineK8V4MeasurementByContext[measurement.context]
                )
                return ThroughputCell(
                    status: InferenceParityBenchmark.RunCellStatus.ok.rawValue,
                    context: measurement.context,
                    label: measurement.label,
                    sampleCount: cellSamples.count,
                    successfulSampleCount: successfulMeasurements.count,
                    failedSampleCount: cellSamples.filter { $0.status == .failed }.count,
                    selectedSampleIndex: measurement.sampleIndex,
                    decodeTokensPerSecond: measurement.decodeTokensPerSecond,
                    prefillTokensPerSecond: measurement.prefillTokensPerSecond,
                    generationSeconds: measurement.generationSeconds,
                    generationLoopSeconds: measurement.generationLoopSeconds,
                    generationSynchronizationSeconds: measurement.generationSynchronizationSeconds,
                    promptPrefillSeconds: measurement.promptPrefillSeconds,
                    decodeTokensPerSecondDistribution: Distribution.make(
                        successfulMeasurements.map(\.decodeTokensPerSecond)
                    ),
                    prefillTokensPerSecondDistribution: Distribution.make(
                        successfulMeasurements.map(\.prefillTokensPerSecond)
                    ),
                    generationSecondsDistribution: Distribution.make(
                        successfulMeasurements.map(\.generationSeconds)
                    ),
                    generationTokenCount: measurement.generationTokenCount,
                    speedRatioToFP16: speedRatio(
                        measurement,
                        to: fp16ByContext[measurement.context]
                    ),
                    speedRatioToAffineK8V4: speedRatio(
                        measurement,
                        to: affineK8V4ByContext[measurement.context]
                    ),
                    estimatedRawKVBytes: measurement.estimatedRawKVBytes,
                    estimatedConfigKVBytes: measurement.estimatedConfigKVBytes,
                    estimatedMemoryReductionRatio: measurement.estimatedMemoryReductionRatio,
                    residentKVCompressionRatio: measurement.estimatedMemoryReductionRatio,
                    estimatedMemoryReductionPercent: memoryReductionPercent(measurement),
                    estimatedMemorySource: options.memoryProfileSource,
                    memoryStart: measurement.memoryStart,
                    memoryEnd: measurement.memoryEnd,
                    peakActiveMemoryBytes: measurement.peakActiveMemoryBytes,
                    steadyActiveMemoryBytes: measurement.memoryEnd.activeMemory,
                    activeMemoryDeltaBytes: measurement.memoryEnd.activeMemory
                        - measurement.memoryStart.activeMemory,
                    cacheMemoryDeltaBytes: measurement.memoryEnd.cacheMemory
                        - measurement.memoryStart.cacheMemory,
                    cachePolicySummary: options.emitCachePolicySummary
                        ? measurement.cachePolicySummary : nil,
                    promptPrefillTiming: options.turboQuantTimingEnabled
                        ? measurement.promptPrefillTiming : nil,
                    generationTiming: options.turboQuantTimingEnabled
                        ? measurement.generationTiming : nil,
                    requestedBackend: promotionGate.requestedBackend,
                    selectedAttentionPaths: promotionGate.selectedAttentionPaths,
                    codecCounts: measurement.codecCounts,
                    boundaryProtectedLayerCount: measurement.boundaryProtectedLayerCount,
                    valueBits: measurement.valueBits,
                    valueGroupSize: measurement.valueGroupSize,
                    residualCorrectionActive: measurement.residualCorrectionActive,
                    nativeKernelKinds: measurement.nativeKernelKinds,
                    swiftDispatchedKernels: measurement.dispatchedKernelCounts.isEmpty
                        ? nil : measurement.dispatchedKernelCounts,
                    sparseSkippedTokens: measurement.sparseSkippedTokens,
                    sparseTotalTokens: measurement.sparseTotalTokens,
                    sparseRecentTokenCount: measurement.sparseRecentTokenCount,
                    sparseOlderTokenCount: measurement.sparseOlderTokenCount,
                    sparsePageCandidateCount: measurement.sparsePageCandidateCount,
                    sparseSkipRatio: measurement.sparseSkipRatio,
                    sparseRequestedLayerCount: measurement.sparseRequestedLayerCount,
                    sparseActiveLayerCount: measurement.sparseActiveLayerCount,
                    sparseRequestedButInactive: measurement.sparseRequestedButInactive,
                    sparseFallbackReason: measurement.attentionDiagnostics.compactMap {
                        $0.lastFallback?.reason ?? $0.lastUnsupportedShape
                            ?? $0.nativeFallbackReason
                    }.first,
                    fallbackReasons: fallbackReasons(measurement.attentionDiagnostics),
                    unsupportedShapes: unsupportedShapes(measurement.attentionDiagnostics),
                    nativeFallbackReasons: nativeFallbackReasons(measurement.attentionDiagnostics),
                    qualityPassed: promotionGate.qualityPassed,
                    qualityReason: promotionGate.qualityReason,
                    promotionEligible: promotionGate.promotionEligible,
                    promotionBlockReasons: promotionGate.promotionBlockReasons,
                    promotionGate: promotionGate,
                    layers: measurement.attentionDiagnostics
                )
            },
            throughputSamples: throughputRun.samples.map { sample in
                let measurement = sample.measurement
                return ThroughputSampleCell(
                    context: sample.context,
                    label: sample.label,
                    sampleIndex: sample.sampleIndex,
                    sampleCount: sample.sampleCount,
                    status: sample.status.rawValue,
                    error: sample.error,
                    decodeTokensPerSecond: measurement?.decodeTokensPerSecond,
                    prefillTokensPerSecond: measurement?.prefillTokensPerSecond,
                    generationSeconds: measurement?.generationSeconds,
                    generationLoopSeconds: measurement?.generationLoopSeconds,
                    generationSynchronizationSeconds: measurement?.generationSynchronizationSeconds,
                    promptPrefillSeconds: measurement?.promptPrefillSeconds,
                    generationTokenCount: measurement?.generationTokenCount,
                    promptPrefillTiming: options.turboQuantTimingEnabled
                        ? measurement?.promptPrefillTiming : nil,
                    generationTiming: options.turboQuantTimingEnabled
                        ? measurement?.generationTiming : nil,
                    estimatedRawKVBytes: measurement?.estimatedRawKVBytes,
                    estimatedConfigKVBytes: measurement?.estimatedConfigKVBytes,
                    estimatedMemoryReductionRatio: measurement?.estimatedMemoryReductionRatio,
                    memoryStart: measurement?.memoryStart,
                    memoryEnd: measurement?.memoryEnd,
                    peakActiveMemoryBytes: measurement?.peakActiveMemoryBytes,
                    codecCounts: measurement?.codecCounts,
                    nativeKernelKinds: measurement?.nativeKernelKinds,
                    swiftDispatchedKernels: (measurement?.dispatchedKernelCounts).flatMap {
                        $0.isEmpty ? nil : $0
                    },
                    sparseSkipRatio: measurement?.sparseSkipRatio,
                    sparseRequestedLayerCount: measurement?.sparseRequestedLayerCount,
                    sparseActiveLayerCount: measurement?.sparseActiveLayerCount,
                    sparseRequestedButInactive: measurement?.sparseRequestedButInactive
                )
            },
            quality: qualityResults.map {
                QualityCell(
                    status: InferenceParityBenchmark.RunCellStatus.ok.rawValue,
                    context: $0.context,
                    label: $0.label,
                    referenceLabel: $0.referenceLabel,
                    candidateFirst: $0.candidateFirst,
                    deterministicTop1MatchRate: $0.quality.deterministicTop1MatchRate,
                    logitKLDivergenceMean: $0.quality.logitKLDivergenceMean,
                    logitMaxAbsErrorP95: $0.quality.logitMaxAbsErrorP95,
                    cosine: $0.quality.attentionOutputCosineMean,
                    passed: $0.quality.passed,
                    reason: $0.quality.gateReason,
                    selectedAttentionPaths: $0.selectedAttentionPaths,
                    codecCounts: $0.codecCounts,
                    rawFallbackAllocated: $0.rawFallbackAllocated,
                    fallbackReasons: $0.fallbackReasons,
                    report: $0.quality
                )
            },
            qualityAttempts: qualityRun.attempts.map { attempt in
                QualityAttemptCell(
                    context: attempt.context,
                    label: attempt.label,
                    referenceLabel: attempt.referenceLabel,
                    candidateFirst: attempt.candidateFirst,
                    status: attempt.status.rawValue,
                    error: attempt.error,
                    selectedAttentionPaths: attempt.measurement?.selectedAttentionPaths,
                    codecCounts: attempt.measurement?.codecCounts,
                    report: attempt.measurement?.quality
                )
            }
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
        print("wrote diagnostics: \(url.path)")
    }
}
