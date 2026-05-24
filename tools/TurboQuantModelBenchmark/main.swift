import Foundation
import MLX
import MLXLMCommon

struct BenchmarkModelConfig: Codable {
    var family: String
    var hiddenSize: Int
    var layerCount: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rope: TurboQuantRoPEFingerprint
    var slidingWindow: TurboQuantSlidingWindowFingerprint
    var cacheType: String
}

struct BenchmarkTurboQuantConfig: Codable {
    var layoutVersion: Int
    var path: String?
    var preset: String
    var keyBits: Double
    var valueBits: Int
    var groupSize: Int
    var fallbackPolicy: String
    var kernelProfile: String
    var actualBytesPerToken: Double?
}

struct BenchmarkMemoryMetrics: Codable {
    var compressedKeyBytes: Int? = nil
    var compressedValueBytes: Int? = nil
    var scratchBytes: Int? = nil
    var peakResidentBytes: Int? = nil
}

struct BenchmarkThroughputMetrics: Codable {
    var decodeTokensPerSecond: Double? = nil
    var prefillTokensPerSecond: Double? = nil
    var firstTokenLatencySeconds: Double? = nil
}

struct BenchmarkQualityMetrics: Codable {
    var logitKL: Double? = nil
    var top1MatchRate: Double? = nil
    var longContextRetrievalScore: Double? = nil
    var goldenTokenMismatch: Bool? = nil
}

struct BenchmarkCase: Codable {
    var id: String
    var headDim: Int
    var contextLength: Int
    var queryLength: Int
    var fallbackDType: String
    var mask: String
    var cacheLayout: String
}

struct BenchmarkResult: Codable {
    var id: String
    var status: String
    var benchmarkCase: BenchmarkCase
    var shape: [Int]
    var latencySeconds: Double?
    var memory: BenchmarkMemoryMetrics
    var throughput: BenchmarkThroughputMetrics
    var quality: BenchmarkQualityMetrics
    var selectedPath: TurboQuantAttentionPath?
    var fallbackReason: String?
    var error: String?
}

struct BenchmarkSummary: Codable {
    var ok: Int
    var skipped: Int
    var failed: Int
}

struct BenchmarkReport: Codable {
    var schemaVersion: Int
    var generatedAt: String
    var iterations: Int
    var repoCommits: [String: String]
    var device: TurboQuantDeviceCapabilities
    var supportsMetalCodec: Bool
    var supportsMetalAttention: Bool
    var modelConfig: BenchmarkModelConfig
    var turboQuant: BenchmarkTurboQuantConfig
    var coverageMatrix: [BenchmarkCase]
    var results: [BenchmarkResult]
    var summary: BenchmarkSummary
}

func argumentValue(_ name: String, default defaultValue: Int) -> Int {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1),
        let value = Int(arguments[index + 1])
    else {
        return defaultValue
    }
    return value
}

func argumentValues(_ name: String, default defaultValue: [Int]) -> [Int] {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    let values = arguments[index + 1].split(separator: ",").compactMap { Int($0) }
    return values.isEmpty ? defaultValue : values
}

func hasFlag(_ name: String) -> Bool {
    CommandLine.arguments.contains(name)
}

func environmentValue(_ key: String, default defaultValue: String = "unknown") -> String {
    ProcessInfo.processInfo.environment[key].flatMap { $0.isEmpty ? nil : $0 } ?? defaultValue
}

func gitCommit(_ relativePath: String) -> String {
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

extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

func values(count: Int, scale: Double, phase: Double = 0) -> [Float] {
    (0 ..< count).map { index in
        let position = Double(index)
        return Float(0.29 * sin(position * scale + phase) + 0.13 * cos(position * 0.041))
    }
}

func timed(iterations: Int, _ body: () throws -> MLXArray) throws -> (Double, MLXArray) {
    let warmup = try body()
    eval(warmup)
    let start = Date.timeIntervalSinceReferenceDate
    var last = warmup
    for _ in 0 ..< iterations {
        last = try body()
        eval(last)
    }
    return ((Date.timeIntervalSinceReferenceDate - start) / Double(max(1, iterations)), last)
}

func makeCases(
    releaseMatrix: Bool,
    headDims: [Int],
    contexts: [Int],
    queryLengths: [Int]
) -> [BenchmarkCase] {
    let fallbackDTypes = releaseMatrix ? ["fp16", "bf16", "fp32"] : ["fp16"]
    let masks = releaseMatrix ? ["causal", "additive", "bool"] : ["causal"]
    let cacheLayouts =
        releaseMatrix ? ["contiguous", "ring", "pinned_prefix"] : ["contiguous", "ring"]
    var cases = [BenchmarkCase]()
    for headDim in headDims {
        for context in contexts {
            for queryLength in queryLengths {
                for dtype in fallbackDTypes {
                    for mask in masks {
                        for cacheLayout in cacheLayouts {
                            cases.append(
                                BenchmarkCase(
                                    id:
                                        "hd\(headDim)_ctx\(context)_q\(queryLength)_\(dtype)_\(mask)_\(cacheLayout)",
                                    headDim: headDim,
                                    contextLength: context,
                                    queryLength: queryLength,
                                    fallbackDType: dtype,
                                    mask: mask,
                                    cacheLayout: cacheLayout
                                )
                            )
                        }
                    }
                }
            }
        }
    }
    return cases
}

func skipped(_ benchmarkCase: BenchmarkCase, reason: String) -> BenchmarkResult {
    BenchmarkResult(
        id: benchmarkCase.id,
        status: "skipped",
        benchmarkCase: benchmarkCase,
        shape: [],
        latencySeconds: nil,
        memory: BenchmarkMemoryMetrics(),
        throughput: BenchmarkThroughputMetrics(),
        quality: BenchmarkQualityMetrics(),
        selectedPath: nil,
        fallbackReason: reason,
        error: nil
    )
}

func failed(_ benchmarkCase: BenchmarkCase, error: String) -> BenchmarkResult {
    BenchmarkResult(
        id: benchmarkCase.id,
        status: "failed",
        benchmarkCase: benchmarkCase,
        shape: [],
        latencySeconds: nil,
        memory: BenchmarkMemoryMetrics(),
        throughput: BenchmarkThroughputMetrics(),
        quality: BenchmarkQualityMetrics(),
        selectedPath: nil,
        fallbackReason: nil,
        error: error
    )
}

func attentionMask(_ name: String) -> MLXFast.ScaledDotProductAttentionMaskMode {
    switch name {
    case "none":
        .none
    default:
        .causal
    }
}

func runCase(
    _ benchmarkCase: BenchmarkCase,
    iterations: Int,
    availability: TurboQuantKernelAvailability
) throws -> BenchmarkResult {
    guard availability.supportsMetalPolarQJLAttention else {
        return skipped(benchmarkCase, reason: "Metal attention unavailable or probe failed")
    }
    guard benchmarkCase.mask == "causal" || benchmarkCase.mask == "none" else {
        return skipped(
            benchmarkCase,
            reason:
                "materialized \(benchmarkCase.mask) mask is not used by this synthetic benchmark")
    }

    let batch = 1
    let kvHeads = 2
    let queryHeads = 4
    let headDim = benchmarkCase.headDim
    let context = benchmarkCase.contextLength
    let queryLength = benchmarkCase.queryLength
    let elementCount = batch * kvHeads * context * headDim
    let keys = MLXArray(
        values(count: elementCount, scale: 0.007), [batch, kvHeads, context, headDim])
    let valuesArray = MLXArray(
        values(count: elementCount, scale: 0.011, phase: 0.2),
        [batch, kvHeads, context, headDim]
    )
    let queries = MLXArray(
        values(count: batch * queryHeads * queryLength * headDim, scale: 0.019),
        [batch, queryHeads, queryLength, headDim]
    )
    let scale = 1 / sqrt(Float(headDim))
    let cache: any TurboQuantCompressedKVCacheProtocol
    switch benchmarkCase.cacheLayout {
    case "ring":
        cache = RotatingTurboQuantKVCache(
            maxSize: min(max(context / 2, 1), context),
            keep: min(16, context),
            preset: .turbo4v2,
            backend: .metalPolarQJL
        )
    case "pinned_prefix":
        cache = RotatingTurboQuantKVCache(
            maxSize: min(max(context / 2, 1), context),
            keep: min(64, context),
            preset: .turbo4v2,
            backend: .metalPolarQJL
        )
    default:
        cache = TurboQuantKVCache(
            preset: .turbo4v2,
            backend: .metalPolarQJL,
            optimizationPolicy: .preferThroughput
        )
    }

    let prefillStart = Date.timeIntervalSinceReferenceDate
    _ = try cache.updateCompressed(keys: keys, values: valuesArray)
    guard let compressed = cache.compressedState else {
        throw NSError(
            domain: "TurboQuantModelBenchmark",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "compressed cache state was not produced"]
        )
    }
    let prefillLatency = Date.timeIntervalSinceReferenceDate - prefillStart
    let (decodeLatency, output) = try timed(iterations: iterations) {
        try turboQuantMetalScaledDotProductAttention(
            queries: queries,
            keyCode: compressed.0,
            valueCode: compressed.1,
            scale: scale,
            mask: attentionMask(benchmarkCase.mask),
            preferOnlineFused: cache.prefersOnlineFusedAttention,
            kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
        )
    }

    return BenchmarkResult(
        id: benchmarkCase.id,
        status: "ok",
        benchmarkCase: benchmarkCase,
        shape: output.shape,
        latencySeconds: decodeLatency,
        memory: BenchmarkMemoryMetrics(
            compressedKeyBytes: compressed.0.storageByteCount,
            compressedValueBytes: compressed.1.storageByteCount
        ),
        throughput: BenchmarkThroughputMetrics(
            decodeTokensPerSecond: Double(queryLength) / max(decodeLatency, .leastNonzeroMagnitude),
            prefillTokensPerSecond: Double(context) / max(prefillLatency, .leastNonzeroMagnitude),
            firstTokenLatencySeconds: queryLength == 1 ? decodeLatency : nil
        ),
        quality: BenchmarkQualityMetrics(goldenTokenMismatch: nil),
        selectedPath: cache.attentionDiagnostics.activeAttentionPath,
        fallbackReason: cache.attentionDiagnostics.fallbackReason,
        error: nil
    )
}

let iterations = argumentValue("--iterations", default: 10)
let releaseMatrix = hasFlag("--release-matrix")
let headDims = argumentValues("--head-dims", default: [64, 80, 96, 128, 192, 256])
let contexts = argumentValues(
    "--contexts",
    default: releaseMatrix ? [1024, 4096, 8192, 16384, 32768, 65536] : [1024]
)
let queryLengths = argumentValues(
    "--query-lengths", default: releaseMatrix ? [1, 4, 16, 128] : [1])
let availability = TurboQuantKernelAvailability.current
let cases = makeCases(
    releaseMatrix: releaseMatrix,
    headDims: headDims,
    contexts: contexts,
    queryLengths: queryLengths
)

var results = [BenchmarkResult]()
for benchmarkCase in cases {
    do {
        results.append(
            try runCase(benchmarkCase, iterations: iterations, availability: availability))
    } catch {
        results.append(failed(benchmarkCase, error: "\(error)"))
    }
}

let okCount = results.filter { $0.status == "ok" }.count
let skippedCount = results.filter { $0.status == "skipped" }.count
let failedCount = results.filter { $0.status == "failed" }.count
let firstSelectedPath = results.compactMap(\.selectedPath).first
let firstActualBytesPerToken =
    results.first(where: { $0.status == "ok" }).flatMap { result -> Double? in
        guard let keyBytes = result.memory.compressedKeyBytes,
            let valueBytes = result.memory.compressedValueBytes
        else {
            return nil
        }
        return Double(keyBytes + valueBytes) / Double(max(1, result.benchmarkCase.contextLength))
    }

let modelConfig = BenchmarkModelConfig(
    family: environmentValue("TURBOQUANT_BENCHMARK_MODEL_FAMILY", default: "synthetic"),
    hiddenSize: argumentValue("--hidden-size", default: 4096),
    layerCount: argumentValue("--layers", default: 32),
    attentionHeads: argumentValue("--attention-heads", default: 32),
    kvHeads: argumentValue("--kv-heads", default: 8),
    headDim: headDims.first ?? 128,
    rope: TurboQuantRoPEFingerprint(
        type: environmentValue("TURBOQUANT_BENCHMARK_ROPE_TYPE", default: "llama"),
        theta: Double(argumentValue("--rope-theta", default: 1_000_000))
    ),
    slidingWindow: TurboQuantSlidingWindowFingerprint(
        enabled: hasFlag("--sliding-window"),
        size: hasFlag("--sliding-window")
            ? argumentValue("--sliding-window-size", default: 4096) : nil
    ),
    cacheType: environmentValue("TURBOQUANT_BENCHMARK_CACHE_TYPE", default: "standard")
)
let report = BenchmarkReport(
    schemaVersion: 2,
    generatedAt: ISO8601DateFormatter().string(from: Date()),
    iterations: iterations,
    repoCommits: [
        "mlx-swift-lm": environmentValue(
            "MLX_SWIFT_LM_COMMIT",
            default: gitCommit(".")
        ),
        "mlx-swift": environmentValue(
            "MLX_SWIFT_COMMIT",
            default: gitCommit(".build/checkouts/mlx-swift")
        ),
        "mlx": environmentValue(
            "MLX_COMMIT",
            default: gitCommit(".build/checkouts/mlx-swift/Source/Cmlx/mlx")
        ),
    ],
    device: TurboQuantDeviceCapabilities.current,
    supportsMetalCodec: availability.supportsMetalPolarQJLCodec,
    supportsMetalAttention: availability.supportsMetalPolarQJLAttention,
    modelConfig: modelConfig,
    turboQuant: BenchmarkTurboQuantConfig(
        layoutVersion: TurboQuantAttentionLayout.currentVersion,
        path: firstSelectedPath?.rawValue,
        preset: TurboQuantPreset.turbo4v2.rawValue,
        keyBits: Double(TurboQuantPreset.turbo4v2.targetMagnitudeBits),
        valueBits: TurboQuantPreset.turbo4v2.defaultValueBits,
        groupSize: 64,
        fallbackPolicy: "compressedDecodeAllowed",
        kernelProfile: availability.selectedKernelProfile.rawValue,
        actualBytesPerToken: firstActualBytesPerToken
    ),
    coverageMatrix: cases,
    results: results,
    summary: BenchmarkSummary(ok: okCount, skipped: skippedCount, failed: failedCount)
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
