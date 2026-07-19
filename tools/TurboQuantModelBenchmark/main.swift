import Foundation
import MLX
import MLXLMCommon

enum BenchmarkCodec: String {
    case polarQJL = "polar_qjl"
    case polarWHT = "polar_wht"
    case affineK8V4 = "affine_k8_v4"
    case affineK8Vx = "affine_k8_vx"
    case affineInt4 = "affine_int4"
    case raw
}

struct BenchmarkOptimizationConfig {
    var label: String
    var codec: BenchmarkCodec
    var preset: TurboQuantPreset?
    var valueBits: Int?
    var valueGroupSize: Int?
    var backend: TurboQuantBackend = .metalPolarQJL
    var precisionPolicy: TurboQuantKVPrecisionPolicy?
    var optimizationPolicy: TurboQuantOptimizationPolicy

    static let fp16 = BenchmarkOptimizationConfig(
        label: "fp16",
        codec: .raw,
        preset: nil,
        valueBits: 16,
        valueGroupSize: nil,
        optimizationPolicy: .auto
    )

    static let affineK8V4 = BenchmarkOptimizationConfig(
        label: "affineK8V4",
        codec: .affineK8V4,
        preset: nil,
        valueBits: TurboQuantKVCodec.affineK8V4ValueBits,
        valueGroupSize: TurboQuantKVCodec.affineK8V4ValueGroupSize,
        optimizationPolicy: .preferThroughput
    )

    static let affineK8V3 = BenchmarkOptimizationConfig(
        label: "affineK8V3",
        codec: .affineK8Vx,
        preset: nil,
        valueBits: 3,
        valueGroupSize: TurboQuantKVCodec.affineK8V4ValueGroupSize,
        optimizationPolicy: .preferThroughput
    )

    static let affineK8V2 = BenchmarkOptimizationConfig(
        label: "affineK8V2",
        codec: .affineK8Vx,
        preset: nil,
        valueBits: 2,
        valueGroupSize: TurboQuantKVCodec.affineK8V4ValueGroupSize,
        optimizationPolicy: .preferThroughput
    )

    static let affineInt4 = BenchmarkOptimizationConfig(
        label: "affineInt4",
        codec: .affineInt4,
        preset: nil,
        valueBits: TurboQuantKVCodec.affineInt4Bits,
        valueGroupSize: nil,
        optimizationPolicy: .preferThroughput
    )

    static func turboQuant(
        label: String,
        preset: TurboQuantPreset,
        optimizationPolicy: TurboQuantOptimizationPolicy
    ) -> BenchmarkOptimizationConfig {
        BenchmarkOptimizationConfig(
            label: label,
            codec: .polarQJL,
            preset: preset,
            valueBits: preset.defaultValueBits,
            valueGroupSize: nil,
            optimizationPolicy: optimizationPolicy
        )
    }

    static let polarWHTV3 = BenchmarkOptimizationConfig(
        label: "polarWHTV3",
        codec: .polarWHT,
        preset: .turbo4v2,
        valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
        valueGroupSize: nil,
        backend: .metalPolarWHT,
        optimizationPolicy: .preferThroughput
    )

    static let hybridK8PolarWHTV3 = BenchmarkOptimizationConfig(
        label: "hybridK8PolarWHTV3",
        codec: .polarWHT,
        preset: .turbo8,
        valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
        valueGroupSize: nil,
        backend: .metalPolarWHT,
        precisionPolicy: TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: TurboQuantKVCodec.polarWHTDefaultValueBits),
            boundary: .disabled
        ),
        optimizationPolicy: .preferThroughput
    )

    static let hybridK8PolarWHTV4 = BenchmarkOptimizationConfig(
        label: "hybridK8PolarWHTV4",
        codec: .polarWHT,
        preset: .turbo8,
        valueBits: 4,
        valueGroupSize: nil,
        backend: .metalPolarWHT,
        precisionPolicy: TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: 4),
            boundary: .disabled
        ),
        optimizationPolicy: .preferThroughput
    )
}

let validBenchmarkOptimizationConfigs: [BenchmarkOptimizationConfig] = [
    .fp16,
    .affineK8V4,
    .affineK8V3,
    .affineK8V2,
    .affineInt4,
    .turboQuant(label: "turbo8", preset: .turbo8, optimizationPolicy: .preferThroughput),
    .turboQuant(label: "turbo4v2", preset: .turbo4v2, optimizationPolicy: .preferThroughput),
    .turboQuant(label: "turbo3_5", preset: .turbo3_5, optimizationPolicy: .preferThroughput),
    .polarWHTV3,
    .hybridK8PolarWHTV3,
    .hybridK8PolarWHTV4,
]

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
    var label: String? = nil
    var codec: String
    var layoutVersion: Int
    var path: String?
    var backend: String
    var preset: String
    var keyBits: Double
    var valueBits: Int
    var groupSize: Int
    var scaleBiasBytes: Int?
    var fallbackPolicy: String
    var kernelProfile: String
    var optimizationPolicy: String? = nil
    var isFP16Baseline: Bool? = nil
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

struct BenchmarkFP16BaselineMetrics: Codable {
    var id: String
    var latencySeconds: Double?
    var memory: BenchmarkMemoryMetrics
    var throughput: BenchmarkThroughputMetrics
}

struct BenchmarkCase: Codable {
    var id: String
    var config: String
    var codec: String
    var headDim: Int
    var queryHeads: Int
    var kvHeads: Int
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
    var quality: TurboQuantQualityGateReport
    var selectedPath: TurboQuantAttentionPath?
    var fp16Baseline: BenchmarkFP16BaselineMetrics?
    var speedRatioToFP16: Double?
    var memoryRatioToFP16: Double?
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
    var optimizationConfigs: [BenchmarkTurboQuantConfig]
    var cooldownSeconds: Double
    var qualityGate: TurboQuantQualityGateReport
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

func argumentStrings(_ name: String, default defaultValue: [String]) -> [String] {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    let values = arguments[index + 1].split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    return values.isEmpty ? defaultValue : values
}

func argumentDouble(_ name: String, default defaultValue: Double) -> Double {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1),
        let value = Double(arguments[index + 1])
    else {
        return defaultValue
    }
    return value
}

func argumentString(_ name: String, default defaultValue: String) -> String {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    return arguments[index + 1]
}

func argumentCodec(_ name: String, default defaultValue: BenchmarkCodec) -> BenchmarkCodec {
    let raw = argumentString(name, default: defaultValue.rawValue)
    switch raw {
    case "raw":
        return .raw
    case "affine_k8_v4", "affineK8V4", "affine-k8-v4", "k8v4", "k8_v4":
        return .affineK8V4
    case "affine_k8_vx", "affineK8Vx", "affine-k8-vx", "k8vx", "k8_vx":
        return .affineK8Vx
    case "affine_int4", "affineInt4", "affine-int4":
        return .affineInt4
    case "polar_wht", "polarWHT", "polar-wht", "wht", "whtv3":
        return .polarWHT
    default:
        return .polarQJL
    }
}

func normalizedConfigLabel(_ label: String) -> String {
    label
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
        .replacingOccurrences(of: "-", with: "_")
        .replacingOccurrences(of: ".", with: "_")
}

func configs(for codec: BenchmarkCodec) -> [BenchmarkOptimizationConfig] {
    switch codec {
    case .raw:
        return [.fp16]
    case .affineK8V4:
        return [.affineK8V4]
    case .affineK8Vx:
        return [.affineK8V3, .affineK8V2]
    case .affineInt4:
        return [.affineInt4]
    case .polarQJL:
        return [.turboQuant(label: "turbo4v2", preset: .turbo4v2, optimizationPolicy: .preferThroughput)]
    case .polarWHT:
        return [.hybridK8PolarWHTV3, .hybridK8PolarWHTV4, .polarWHTV3]
    }
}

func argumentOptimizationConfigs(
    _ name: String,
    default defaultValue: [BenchmarkOptimizationConfig]
) throws -> [BenchmarkOptimizationConfig] {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }

    let requested = arguments[index + 1].split(separator: ",")
        .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }
    guard !requested.isEmpty else { return defaultValue }
    if requested.contains(where: { normalizedConfigLabel($0) == "all" }) {
        return validBenchmarkOptimizationConfigs
    }

    let configsByLabel = Dictionary(
        uniqueKeysWithValues: validBenchmarkOptimizationConfigs.map {
            (normalizedConfigLabel($0.label), $0)
        }
    )
    var selected = [BenchmarkOptimizationConfig]()
    var unknown = [String]()
    for label in requested {
        switch normalizedConfigLabel(label) {
        case "fp16", "raw", "baseline":
            selected.append(.fp16)
        case "affinek8v4", "affine_k8_v4", "k8v4", "k8_v4":
            selected.append(.affineK8V4)
        case "affinek8v3", "affine_k8_v3", "k8v3", "k8_v3":
            selected.append(.affineK8V3)
        case "affinek8v2", "affine_k8_v2", "k8v2", "k8_v2":
            selected.append(.affineK8V2)
        case "affinek8vx", "affine_k8_vx", "k8vx", "k8_vx":
            selected.append(.affineK8V3)
            selected.append(.affineK8V2)
        case "int4", "affineint4", "affine_int4":
            selected.append(.affineInt4)
        case "turbo4", "turbo4v2", "turbo_4_v2":
            selected.append(
                .turboQuant(label: "turbo4v2", preset: .turbo4v2, optimizationPolicy: .preferThroughput))
        case "turbo35", "turbo3_5", "turbo_3_5":
            selected.append(
                .turboQuant(label: "turbo3_5", preset: .turbo3_5, optimizationPolicy: .preferThroughput))
        case "turbo8", "turbo_8":
            selected.append(
                .turboQuant(label: "turbo8", preset: .turbo8, optimizationPolicy: .preferThroughput))
        case "polarwht", "polar_wht", "polarwhtv3", "polar_wht_v3", "whtv3":
            selected.append(.polarWHTV3)
        case "hybridk8polarwhtv3", "hybrid_k8_polar_wht_v3",
            "hybridk8polarwht", "hybrid_k8_polar_wht", "k8whtv3", "k8_wht_v3":
            selected.append(.hybridK8PolarWHTV3)
        case "hybridk8polarwhtv4", "hybrid_k8_polar_wht_v4",
            "k8whtv4", "k8_wht_v4":
            selected.append(.hybridK8PolarWHTV4)
        case let normalized where configsByLabel[normalized] != nil:
            selected.append(configsByLabel[normalized]!)
        default:
            unknown.append(label)
        }
    }
    if !unknown.isEmpty {
        let known = validBenchmarkOptimizationConfigs.map(\.label).joined(separator: ", ")
        throw NSError(
            domain: "TurboQuantModelBenchmark",
            code: 2,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "unknown --configs entries: \(unknown.joined(separator: ", ")); known configs: \(known)"
            ]
        )
    }

    var unique = [BenchmarkOptimizationConfig]()
    var seen = Set<String>()
    for config in selected where seen.insert(config.label).inserted {
        unique.append(config)
    }
    return unique.isEmpty ? defaultValue : unique
}

func hasFlag(_ name: String) -> Bool {
    CommandLine.arguments.contains(name)
}

func environmentValue(_ key: String, default defaultValue: String = "unknown") -> String {
    ProcessInfo.processInfo.environment[key].flatMap { $0.isEmpty ? nil : $0 } ?? defaultValue
}

func environmentDouble(_ key: String, default defaultValue: Double) -> Double {
    ProcessInfo.processInfo.environment[key].flatMap(Double.init) ?? defaultValue
}

func mlxDType(_ name: String) -> DType {
    switch name.lowercased() {
    case "fp16", "float16":
        return .float16
    case "bf16", "bfloat16":
        return .bfloat16
    case "fp32", "float32":
        return .float32
    default:
        return .float16
    }
}

func gitCommit(_ relativePath: String) -> String {
    #if !os(macOS)
    // Process (NSTask) is unavailable on iOS; provenance capture is a
    // macOS-host concern and the tool never runs on device.
    return "unknown"
    #else
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
    #endif
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
    configs: [BenchmarkOptimizationConfig],
    headDims: [Int],
    queryHeads: Int,
    kvHeads: Int,
    contexts: [Int],
    queryLengths: [Int],
    cacheLayouts: [String]
) -> [BenchmarkCase] {
    let fallbackDTypes = releaseMatrix ? ["fp16", "bf16", "fp32"] : ["fp16"]
    let masks = releaseMatrix ? ["causal", "additive", "bool"] : ["causal"]
    var cases = [BenchmarkCase]()
    for config in configs {
        for headDim in headDims {
            for context in contexts {
                for queryLength in queryLengths {
                    for dtype in fallbackDTypes {
                        for mask in masks {
                            for cacheLayout in cacheLayouts {
                                cases.append(
                                    BenchmarkCase(
                                        id:
                                            "\(config.label)_qh\(queryHeads)_kvh\(kvHeads)_hd\(headDim)_ctx\(context)_q\(queryLength)_\(dtype)_\(mask)_\(cacheLayout)",
                                        config: config.label,
                                        codec: config.codec.rawValue,
                                        headDim: headDim,
                                        queryHeads: queryHeads,
                                        kvHeads: kvHeads,
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
        quality: .failed(reason: reason),
        selectedPath: nil,
        fp16Baseline: nil,
        speedRatioToFP16: nil,
        memoryRatioToFP16: nil,
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
        quality: .failed(reason: error),
        selectedPath: nil,
        fp16Baseline: nil,
        speedRatioToFP16: nil,
        memoryRatioToFP16: nil,
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

func qualityGate(
    candidate: MLXArray,
    reference: MLXArray,
    prefillExact: Bool,
    codec: BenchmarkCodec = .polarQJL
) -> TurboQuantQualityGateReport {
    eval(candidate, reference)
    let candidateValues = candidate.asArray(Float.self)
    let referenceValues = reference.asArray(Float.self)
    guard candidateValues.count == referenceValues.count,
        let rowWidth = candidate.shape.last,
        rowWidth > 0,
        !candidateValues.isEmpty
    else {
        return .failed(reason: "candidate and reference quality shapes do not match")
    }

    let noNaNOrInf =
        candidateValues.allSatisfy(\.isFinite) && referenceValues.allSatisfy(\.isFinite)
    let top1MatchRate = rowTop1MatchRate(
        candidate: candidateValues,
        reference: referenceValues,
        rowWidth: rowWidth
    )
    let klDivergenceMean = meanKLDivergence(
        candidate: candidateValues,
        reference: referenceValues,
        rowWidth: rowWidth
    )
    let p95MaxAbsError = percentile(
        rowMaxAbsErrors(
            candidate: candidateValues,
            reference: referenceValues,
            rowWidth: rowWidth
        ),
        percentile: 0.95
    )
    let cosineMean = meanCosineSimilarity(
        candidate: candidateValues,
        reference: referenceValues,
        rowWidth: rowWidth
    )
    let fallbackEquivalent =
        noNaNOrInf
        && top1MatchRate >= 0.95
        && klDivergenceMean <= 0.05
        && p95MaxAbsError <= 0.5
    if codec == .affineInt4 {
        return .evaluatedAffineInt4(
            benchmarkSuiteID: .fallbackEquivalenceV1,
            deterministicTop1MatchRate: top1MatchRate,
            logitKLDivergenceMean: klDivergenceMean,
            logitMaxAbsErrorP95: p95MaxAbsError,
            attentionOutputCosineMean: cosineMean,
            noNaNOrInf: noNaNOrInf,
            snapshotRoundtripEquivalent: nil
        )
    }
    return .evaluated(
        benchmarkSuiteID: .fallbackEquivalenceV1,
        deterministicTop1MatchRate: top1MatchRate,
        logitKLDivergenceMean: klDivergenceMean,
        logitMaxAbsErrorP95: p95MaxAbsError,
        attentionOutputCosineMean: cosineMean,
        noNaNOrInf: noNaNOrInf,
        fallbackEquivalent: fallbackEquivalent,
        prefillExact: prefillExact,
        snapshotRoundtripEquivalent: nil
    )
}

func rowTop1MatchRate(candidate: [Float], reference: [Float], rowWidth: Int) -> Double {
    let rowCount = min(candidate.count, reference.count) / rowWidth
    guard rowCount > 0 else { return 0 }
    var matches = 0
    for row in 0 ..< rowCount {
        let start = row * rowWidth
        let end = start + rowWidth
        if argmax(Array(candidate[start ..< end])) == argmax(Array(reference[start ..< end])) {
            matches += 1
        }
    }
    return Double(matches) / Double(rowCount)
}

func meanKLDivergence(candidate: [Float], reference: [Float], rowWidth: Int) -> Double {
    let rowCount = min(candidate.count, reference.count) / rowWidth
    guard rowCount > 0 else { return Double.greatestFiniteMagnitude }
    var total = 0.0
    for row in 0 ..< rowCount {
        let start = row * rowWidth
        let end = start + rowWidth
        let candidateRow = Array(candidate[start ..< end]).map(Double.init)
        let referenceRow = Array(reference[start ..< end]).map(Double.init)
        total += klDivergence(candidateLogits: candidateRow, referenceLogits: referenceRow)
    }
    return total / Double(rowCount)
}

func rowMaxAbsErrors(candidate: [Float], reference: [Float], rowWidth: Int) -> [Double] {
    let rowCount = min(candidate.count, reference.count) / rowWidth
    guard rowCount > 0 else { return [] }
    var errors = [Double]()
    errors.reserveCapacity(rowCount)
    for row in 0 ..< rowCount {
        let start = row * rowWidth
        let end = start + rowWidth
        let rowError = zip(candidate[start ..< end], reference[start ..< end])
            .reduce(Double(0)) { partial, pair in
                max(partial, Double(abs(pair.0 - pair.1)))
            }
        errors.append(rowError)
    }
    return errors
}

func meanCosineSimilarity(candidate: [Float], reference: [Float], rowWidth: Int) -> Double {
    let rowCount = min(candidate.count, reference.count) / rowWidth
    guard rowCount > 0 else { return 0 }
    var total = 0.0
    for row in 0 ..< rowCount {
        let start = row * rowWidth
        let end = start + rowWidth
        var dot = 0.0
        var candidateNorm = 0.0
        var referenceNorm = 0.0
        for (candidateValue, referenceValue) in zip(candidate[start ..< end], reference[start ..< end]) {
            let c = Double(candidateValue)
            let r = Double(referenceValue)
            dot += c * r
            candidateNorm += c * c
            referenceNorm += r * r
        }
        let denominator = sqrt(candidateNorm) * sqrt(referenceNorm)
        total += denominator > 0 ? dot / denominator : 0
    }
    return total / Double(rowCount)
}

func klDivergence(candidateLogits: [Double], referenceLogits: [Double]) -> Double {
    guard candidateLogits.allSatisfy(\.isFinite), referenceLogits.allSatisfy(\.isFinite) else {
        return Double.greatestFiniteMagnitude
    }
    let candidateLogProbs = logSoftmax(candidateLogits)
    let referenceLogProbs = logSoftmax(referenceLogits)
    return zip(referenceLogProbs, candidateLogProbs).reduce(0.0) { partial, pair in
        let referenceProbability = exp(pair.0)
        return partial + referenceProbability * (pair.0 - pair.1)
    }
}

func logSoftmax(_ values: [Double]) -> [Double] {
    guard let maxValue = values.max(), maxValue.isFinite else {
        return Array(repeating: -Double.greatestFiniteMagnitude, count: values.count)
    }
    let shiftedExpSum = values.reduce(0.0) { partial, value in
        partial + exp(value - maxValue)
    }
    let logDenominator = maxValue + log(max(shiftedExpSum, Double.leastNonzeroMagnitude))
    return values.map { $0 - logDenominator }
}

func argmax(_ values: [Float]) -> Int {
    guard var bestValue = values.first else { return -1 }
    var bestIndex = 0
    for index in values.indices.dropFirst() where values[index] > bestValue {
        bestValue = values[index]
        bestIndex = index
    }
    return bestIndex
}

func percentile(_ values: [Double], percentile: Double) -> Double {
    guard !values.isEmpty else { return Double.greatestFiniteMagnitude }
    let sorted = values.sorted()
    let clamped = min(max(percentile, 0), 1)
    let index = min(
        sorted.count - 1,
        max(0, Int(ceil(clamped * Double(sorted.count))) - 1)
    )
    return sorted[index]
}

func aggregateQualityGate(_ results: [BenchmarkResult]) -> TurboQuantQualityGateReport {
    let successfulQuality = results
        .filter { $0.status == "ok" }
        .map(\.quality)
    guard !successfulQuality.isEmpty else {
        return .failed(reason: "no successful benchmark cases")
    }
    let count = Double(successfulQuality.count)
    let top1 = successfulQuality.reduce(0.0) {
        $0 + $1.deterministicTop1MatchRate
    } / count
    let kl = successfulQuality.reduce(0.0) {
        $0 + $1.logitKLDivergenceMean
    } / count
    let p95 = percentile(
        successfulQuality.map(\.logitMaxAbsErrorP95),
        percentile: 0.95
    )
    let finiteCosines = successfulQuality.compactMap(\.attentionOutputCosineMean)
    let cosineMean =
        finiteCosines.isEmpty ? nil : finiteCosines.reduce(0, +) / Double(finiteCosines.count)

    return .evaluated(
        benchmarkSuiteID: .fallbackEquivalenceV1,
        deterministicTop1MatchRate: top1,
        logitKLDivergenceMean: kl,
        logitMaxAbsErrorP95: p95,
        attentionOutputCosineMean: cosineMean,
        noNaNOrInf: successfulQuality.allSatisfy(\.noNaNOrInf),
        fallbackEquivalent: successfulQuality.allSatisfy(\.fallbackEquivalent),
        prefillExact: successfulQuality.allSatisfy(\.prefillExact),
        snapshotRoundtripEquivalent: nil
    )
}

func fp16BaselineKey(_ benchmarkCase: BenchmarkCase) -> String {
    [
        "qh\(benchmarkCase.queryHeads)",
        "kvh\(benchmarkCase.kvHeads)",
        "hd\(benchmarkCase.headDim)",
        "ctx\(benchmarkCase.contextLength)",
        "q\(benchmarkCase.queryLength)",
        benchmarkCase.mask,
        benchmarkCase.cacheLayout,
    ].joined(separator: "_")
}

func keyValueBytes(_ memory: BenchmarkMemoryMetrics) -> Int? {
    guard let keyBytes = memory.compressedKeyBytes,
        let valueBytes = memory.compressedValueBytes
    else {
        return nil
    }
    return keyBytes + valueBytes
}

func attachFP16Baselines(_ results: [BenchmarkResult]) -> [BenchmarkResult] {
    let baselines = Dictionary(
        uniqueKeysWithValues: results.compactMap { result -> (String, BenchmarkResult)? in
            guard result.status == "ok",
                result.benchmarkCase.config == "fp16",
                result.benchmarkCase.fallbackDType == "fp16"
            else {
                return nil
            }
            return (fp16BaselineKey(result.benchmarkCase), result)
        }
    )

    return results.map { result in
        guard let baseline = baselines[fp16BaselineKey(result.benchmarkCase)] else {
            return result
        }
        var annotated = result
        annotated.fp16Baseline = BenchmarkFP16BaselineMetrics(
            id: baseline.id,
            latencySeconds: baseline.latencySeconds,
            memory: baseline.memory,
            throughput: baseline.throughput
        )
        if result.status == "ok",
            let resultDecode = result.throughput.decodeTokensPerSecond,
            let baselineDecode = baseline.throughput.decodeTokensPerSecond,
            baselineDecode > 0
        {
            annotated.speedRatioToFP16 = resultDecode / baselineDecode
        }
        if result.status == "ok",
            let resultBytes = keyValueBytes(result.memory),
            let baselineBytes = keyValueBytes(baseline.memory),
            baselineBytes > 0
        {
            annotated.memoryRatioToFP16 = Double(resultBytes) / Double(baselineBytes)
        }
        return annotated
    }
}

func cooldown(seconds: Double) {
    guard seconds > 0 else { return }
    Stream.gpu.synchronize()
    Memory.clearCache()
    Thread.sleep(forTimeInterval: seconds)
}

func actualBytesPerToken(for label: String, results: [BenchmarkResult]) -> Double? {
    results.first(where: { $0.status == "ok" && $0.benchmarkCase.config == label }).flatMap {
        result -> Double? in
        guard let bytes = keyValueBytes(result.memory) else { return nil }
        return Double(bytes) / Double(max(1, result.benchmarkCase.contextLength))
    }
}

func scaleBiasBytes(
    for config: BenchmarkOptimizationConfig,
    sampleCase: BenchmarkCase?
) -> Int? {
    guard let sampleCase else { return nil }
    switch config.codec {
    case .affineInt4:
        let groups = max(1, sampleCase.headDim / TurboQuantKVCodec.affineInt4DefaultGroupSize)
        return 4 * 2 * 2 * sampleCase.kvHeads * sampleCase.contextLength * groups
    case .affineK8V4, .affineK8Vx:
        let keyGroups = max(1, sampleCase.headDim / TurboQuantKVCodec.affineK8V4KeyGroupSize)
        let valueGroups = max(
            1,
            sampleCase.headDim / (config.valueGroupSize ?? TurboQuantKVCodec.affineK8V4ValueGroupSize)
        )
        return 4 * 2 * sampleCase.kvHeads * sampleCase.contextLength * (keyGroups + valueGroups)
    case .polarQJL, .polarWHT, .raw:
        return nil
    }
}

func reportConfig(
    for config: BenchmarkOptimizationConfig,
    cases: [BenchmarkCase],
    results: [BenchmarkResult],
    availability: TurboQuantKernelAvailability
) -> BenchmarkTurboQuantConfig {
    let selectedPath = results
        .first { $0.status == "ok" && $0.benchmarkCase.config == config.label }?
        .selectedPath
    let sampleCase = cases.first { $0.config == config.label }
    let preset = config.preset ?? .turbo4v2
    let keyBits =
        config.precisionPolicy?.key == .affineQ8
            ? Double(TurboQuantKVCodec.affineK8V4KeyBits)
            : Double(preset.targetMagnitudeBits)

    switch config.codec {
    case .raw:
        return BenchmarkTurboQuantConfig(
            label: config.label,
            codec: config.codec.rawValue,
            layoutVersion: TurboQuantAttentionLayout.currentVersion,
            path: selectedPath?.rawValue ?? TurboQuantAttentionPath.baseline.rawValue,
            backend: "rawSDPA",
            preset: "fp16",
            keyBits: 16,
            valueBits: 16,
            groupSize: sampleCase?.headDim ?? 0,
            scaleBiasBytes: nil,
            fallbackPolicy: "none",
            kernelProfile: "baseline",
            optimizationPolicy: config.optimizationPolicy.rawValue,
            isFP16Baseline: true,
            actualBytesPerToken: actualBytesPerToken(for: config.label, results: results)
        )
    case .affineK8V4, .affineK8Vx:
        return BenchmarkTurboQuantConfig(
            label: config.label,
            codec: config.codec.rawValue,
            layoutVersion: TurboQuantAttentionLayout.currentVersion,
            path: selectedPath?.rawValue,
            backend: TurboQuantBackend.mlxPacked.rawValue,
            preset: config.label,
            keyBits: Double(TurboQuantKVCodec.affineK8V4KeyBits),
            valueBits: config.valueBits ?? TurboQuantKVCodec.affineK8V4ValueBits,
            groupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            scaleBiasBytes: scaleBiasBytes(for: config, sampleCase: sampleCase),
            fallbackPolicy: "compressedDecodeAllowed",
            kernelProfile: availability.selectedKernelProfile.rawValue,
            optimizationPolicy: config.optimizationPolicy.rawValue,
            isFP16Baseline: false,
            actualBytesPerToken: actualBytesPerToken(for: config.label, results: results)
        )
    case .affineInt4:
        return BenchmarkTurboQuantConfig(
            label: config.label,
            codec: config.codec.rawValue,
            layoutVersion: TurboQuantAttentionLayout.currentVersion,
            path: selectedPath?.rawValue,
            backend: TurboQuantBackend.mlxPacked.rawValue,
            preset: config.label,
            keyBits: Double(TurboQuantKVCodec.affineInt4Bits),
            valueBits: TurboQuantKVCodec.affineInt4Bits,
            groupSize: TurboQuantKVCodec.affineInt4DefaultGroupSize,
            scaleBiasBytes: scaleBiasBytes(for: config, sampleCase: sampleCase),
            fallbackPolicy: "compressedDecodeAllowed",
            kernelProfile: availability.selectedKernelProfile.rawValue,
            optimizationPolicy: config.optimizationPolicy.rawValue,
            isFP16Baseline: false,
            actualBytesPerToken: actualBytesPerToken(for: config.label, results: results)
        )
    case .polarQJL, .polarWHT:
        return BenchmarkTurboQuantConfig(
            label: config.label,
            codec: config.codec.rawValue,
            layoutVersion: TurboQuantAttentionLayout.currentVersion,
            path: selectedPath?.rawValue,
            backend: config.backend.rawValue,
            preset: preset.rawValue,
            keyBits: keyBits,
            valueBits: config.valueBits ?? preset.defaultValueBits,
            groupSize: 64,
            scaleBiasBytes: nil,
            fallbackPolicy: "compressedDecodeAllowed",
            kernelProfile: availability.selectedKernelProfile.rawValue,
            optimizationPolicy: config.optimizationPolicy.rawValue,
            isFP16Baseline: false,
            actualBytesPerToken: actualBytesPerToken(for: config.label, results: results)
        )
    }
}

func runCase(
    _ benchmarkCase: BenchmarkCase,
    iterations: Int,
    config: BenchmarkOptimizationConfig,
    availability: TurboQuantKernelAvailability
) throws -> BenchmarkResult {
    let codec = config.codec
    guard codec != .polarQJL || availability.supportsMetalPolarQJLAttention else {
        return skipped(benchmarkCase, reason: "Metal attention unavailable or probe failed")
    }
    guard codec != .polarWHT || availability.supportsMetalPolarWHTAttention else {
        return skipped(benchmarkCase, reason: "PolarWHT Metal attention unavailable or probe failed")
    }
    guard benchmarkCase.mask == "causal" || benchmarkCase.mask == "none" else {
        return skipped(
            benchmarkCase,
            reason:
                "materialized \(benchmarkCase.mask) mask is not used by this synthetic benchmark")
    }

    let batch = 1
    let kvHeads = benchmarkCase.kvHeads
    let queryHeads = benchmarkCase.queryHeads
    let headDim = benchmarkCase.headDim
    let context = benchmarkCase.contextLength
    let queryLength = benchmarkCase.queryLength
    let dtype = mlxDType(benchmarkCase.fallbackDType)
    let elementCount = batch * kvHeads * context * headDim
    let keys = MLXArray(
        values(count: elementCount, scale: 0.007), [batch, kvHeads, context, headDim]
    ).asType(dtype)
    let valuesArray = MLXArray(
        values(count: elementCount, scale: 0.011, phase: 0.2),
        [batch, kvHeads, context, headDim]
    ).asType(dtype)
    let queries = MLXArray(
        values(count: batch * queryHeads * queryLength * headDim, scale: 0.019),
        [batch, queryHeads, queryLength, headDim]
    ).asType(dtype)
    let scale = 1 / sqrt(Float(headDim))

    if codec == .raw {
        let (decodeLatency, output) = try timed(iterations: iterations) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: valuesArray,
                scale: scale,
                mask: attentionMask(benchmarkCase.mask)
            )
        }
        return BenchmarkResult(
            id: benchmarkCase.id,
            status: "ok",
            benchmarkCase: benchmarkCase,
            shape: output.shape,
            latencySeconds: decodeLatency,
            memory: BenchmarkMemoryMetrics(
                compressedKeyBytes: keys.nbytes,
                compressedValueBytes: valuesArray.nbytes
            ),
            throughput: BenchmarkThroughputMetrics(
                decodeTokensPerSecond: Double(queryLength) / max(decodeLatency, .leastNonzeroMagnitude),
                prefillTokensPerSecond: nil,
                firstTokenLatencySeconds: queryLength == 1 ? decodeLatency : nil
            ),
            quality: qualityGate(candidate: output, reference: output, prefillExact: true),
            selectedPath: .baseline,
            fp16Baseline: nil,
            speedRatioToFP16: nil,
            memoryRatioToFP16: nil,
            fallbackReason: nil,
            error: nil
        )
    }

    if codec == .affineInt4 {
        let groupSize = TurboQuantKVCodec.affineInt4DefaultGroupSize
        let qKeys = quantized(
            keys,
            groupSize: groupSize,
            bits: TurboQuantKVCodec.affineInt4Bits,
            mode: .affine
        )
        let qValues = quantized(
            valuesArray,
            groupSize: groupSize,
            bits: TurboQuantKVCodec.affineInt4Bits,
            mode: .affine
        )
        let keyTuple = (qKeys.wq, qKeys.scales, qKeys.biases)
        let valueTuple = (qValues.wq, qValues.scales, qValues.biases)
        guard supportsNativeAffineInt4ScaledDotProductAttention(
            queries: queries,
            quantizedKeys: keyTuple,
            quantizedValues: valueTuple,
            mask: attentionMask(benchmarkCase.mask),
            groupSize: groupSize
        ) else {
            return skipped(benchmarkCase, reason: "native affine int4 SDPA unsupported")
        }
        let prefillLatency = Date.timeIntervalSinceReferenceDate
        let (decodeLatency, output) = try timed(iterations: iterations) {
            try affineInt4NativeScaledDotProductAttention(
                queries: queries,
                quantizedKeys: keyTuple,
                quantizedValues: valueTuple,
                scale: scale,
                mask: attentionMask(benchmarkCase.mask),
                groupSize: groupSize
            )
        }
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: valuesArray,
            scale: scale,
            mask: attentionMask(benchmarkCase.mask)
        )
        let quality = qualityGate(
            candidate: output,
            reference: reference,
            prefillExact: true,
            codec: .affineInt4
        )
        return BenchmarkResult(
            id: benchmarkCase.id,
            status: "ok",
            benchmarkCase: benchmarkCase,
            shape: output.shape,
            latencySeconds: decodeLatency,
            memory: BenchmarkMemoryMetrics(
                compressedKeyBytes: qKeys.wq.nbytes + qKeys.scales.nbytes
                    + (qKeys.biases?.nbytes ?? 0),
                compressedValueBytes: qValues.wq.nbytes + qValues.scales.nbytes
                    + (qValues.biases?.nbytes ?? 0)
            ),
            throughput: BenchmarkThroughputMetrics(
                decodeTokensPerSecond: Double(queryLength) / max(decodeLatency, .leastNonzeroMagnitude),
                prefillTokensPerSecond: Double(context)
                    / max(Date.timeIntervalSinceReferenceDate - prefillLatency, .leastNonzeroMagnitude),
                firstTokenLatencySeconds: queryLength == 1 ? decodeLatency : nil
            ),
            quality: quality,
            selectedPath: .affineInt4Native,
            fp16Baseline: nil,
            speedRatioToFP16: nil,
            memoryRatioToFP16: nil,
            fallbackReason: nil,
            error: nil
        )
    }

    if codec == .affineK8V4 || codec == .affineK8Vx {
        let valueBits = config.valueBits ?? TurboQuantKVCodec.affineK8V4ValueBits
        let valueGroupSize = config.valueGroupSize ?? TurboQuantKVCodec.affineK8V4ValueGroupSize
        let qKeys = quantized(
            keys,
            groupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            bits: TurboQuantKVCodec.affineK8V4KeyBits,
            mode: .affine
        )
        let qValues = quantized(
            valuesArray,
            groupSize: valueGroupSize,
            bits: valueBits,
            mode: .affine
        )
        let keyTuple = (qKeys.wq, qKeys.scales, qKeys.biases)
        let valueTuple = (qValues.wq, qValues.scales, qValues.biases)
        let usesNativeMixedAttention = supportsNativeAffineK8V4ScaledDotProductAttention(
            queries: queries,
            quantizedKeys: keyTuple,
            quantizedValues: valueTuple,
            mask: attentionMask(benchmarkCase.mask),
            valueGroupSize: valueGroupSize,
            valueBits: valueBits
        )
        let prefillLatency = Date.timeIntervalSinceReferenceDate
        let (decodeLatency, output) = try timed(iterations: iterations) {
            try mixedAffineK8V4ScaledDotProductAttention(
                queries: queries,
                quantizedKeys: keyTuple,
                quantizedValues: valueTuple,
                scale: scale,
                mask: attentionMask(benchmarkCase.mask),
                valueGroupSize: valueGroupSize,
                valueBits: valueBits
            )
        }
        let reference = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: valuesArray,
            scale: scale,
            mask: attentionMask(benchmarkCase.mask)
        )
        let quality = qualityGate(
            candidate: output,
            reference: reference,
            prefillExact: true
        )
        return BenchmarkResult(
            id: benchmarkCase.id,
            status: "ok",
            benchmarkCase: benchmarkCase,
            shape: output.shape,
            latencySeconds: decodeLatency,
            memory: BenchmarkMemoryMetrics(
                compressedKeyBytes: qKeys.wq.nbytes + qKeys.scales.nbytes
                    + (qKeys.biases?.nbytes ?? 0),
                compressedValueBytes: qValues.wq.nbytes + qValues.scales.nbytes
                    + (qValues.biases?.nbytes ?? 0)
            ),
            throughput: BenchmarkThroughputMetrics(
                decodeTokensPerSecond: Double(queryLength) / max(decodeLatency, .leastNonzeroMagnitude),
                prefillTokensPerSecond: Double(context)
                    / max(Date.timeIntervalSinceReferenceDate - prefillLatency, .leastNonzeroMagnitude),
                firstTokenLatencySeconds: queryLength == 1 ? decodeLatency : nil
            ),
            quality: quality,
            selectedPath: usesNativeMixedAttention
                ? (valueBits == TurboQuantKVCodec.affineK8V4ValueBits
                    ? .affineK8V4Native : .affineK8VxNative)
                : .mlxPackedFallback,
            fp16Baseline: nil,
            speedRatioToFP16: nil,
            memoryRatioToFP16: nil,
            fallbackReason: usesNativeMixedAttention
                ? nil
                : "mixed affine K8/Vx uses quantizedMM QK + quantizedMM AV",
            error: nil
        )
    }

    let preset = config.preset ?? .turbo4v2
    let valueBits = config.valueBits ?? preset.defaultValueBits
    let cache: any TurboQuantCompressedKVCacheProtocol
    switch benchmarkCase.cacheLayout {
    case "ring":
        cache = RotatingTurboQuantKVCache(
            maxSize: min(max(context / 2, 1), context),
            keep: min(16, context),
            preset: preset,
            backend: config.backend,
            optimizationPolicy: config.optimizationPolicy,
            valueBits: valueBits,
            precisionPolicy: config.precisionPolicy
        )
    case "pinned_prefix":
        cache = RotatingTurboQuantKVCache(
            maxSize: min(max(context / 2, 1), context),
            keep: min(64, context),
            preset: preset,
            backend: config.backend,
            optimizationPolicy: config.optimizationPolicy,
            valueBits: valueBits,
            precisionPolicy: config.precisionPolicy
        )
    default:
        cache = TurboQuantKVCache(
            preset: preset,
            backend: config.backend,
            optimizationPolicy: config.optimizationPolicy,
            valueBits: valueBits,
            precisionPolicy: config.precisionPolicy
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
    let (decodedKeys, decodedValues) = try cache.decodedCompressedState(outputDType: .float32)
    let reference = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: decodedKeys,
        values: decodedValues,
        scale: scale,
        mask: attentionMask(benchmarkCase.mask)
    )
    let quality = qualityGate(
        candidate: output,
        reference: reference,
        prefillExact: false
    )

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
        quality: quality,
        selectedPath: cache.attentionDiagnostics.activeAttentionPath,
        fp16Baseline: nil,
        speedRatioToFP16: nil,
        memoryRatioToFP16: nil,
        fallbackReason: cache.attentionDiagnostics.fallbackReason,
        error: nil
    )
}

let iterations = argumentValue("--iterations", default: 10)
let codec = argumentCodec("--codec", default: .polarQJL)
let releaseMatrix = hasFlag("--release-matrix")
let explicitCodec = CommandLine.arguments.contains("--codec")
let defaultOptimizationConfigs =
    explicitCodec ? configs(for: codec)
        : (releaseMatrix ? validBenchmarkOptimizationConfigs : configs(for: codec))
let includeFP16Baseline = hasFlag("--include-fp16-baseline")
let cooldownSeconds = max(
    0,
    argumentDouble(
        "--cooldown-seconds",
        default: environmentDouble("TURBOQUANT_BENCHMARK_COOLDOWN_SECONDS", default: 0)
    )
)
var benchmarkConfigs = try argumentOptimizationConfigs("--configs", default: defaultOptimizationConfigs)
if includeFP16Baseline, !benchmarkConfigs.contains(where: { $0.label == "fp16" }) {
    benchmarkConfigs.insert(.fp16, at: 0)
}
let headDims = argumentValues("--head-dims", default: [64, 80, 96, 128, 192, 256])
let queryHeads = argumentValue("--query-heads", default: 4)
let kvHeads = argumentValue("--kv-heads", default: 2)
let contexts = argumentValues(
    "--contexts",
    default: releaseMatrix ? [1024, 4096, 8192, 16384, 32768, 65536] : [1024]
)
let queryLengths = argumentValues(
    "--query-lengths", default: releaseMatrix ? [1, 4, 16, 128] : [1])
let cacheLayouts = argumentStrings(
    "--cache-layouts",
    default: releaseMatrix ? ["contiguous", "ring", "pinned_prefix"] : ["contiguous"])
let availability = TurboQuantKernelAvailability.current
let cases = makeCases(
    releaseMatrix: releaseMatrix,
    configs: benchmarkConfigs,
    headDims: headDims,
    queryHeads: queryHeads,
    kvHeads: kvHeads,
    contexts: contexts,
    queryLengths: queryLengths,
    cacheLayouts: cacheLayouts
)

var results = [BenchmarkResult]()
let configsByLabel = Dictionary(uniqueKeysWithValues: benchmarkConfigs.map { ($0.label, $0) })
for (index, benchmarkCase) in cases.enumerated() {
    let result: BenchmarkResult
    do {
        guard let config = configsByLabel[benchmarkCase.config] else {
            throw NSError(
                domain: "TurboQuantModelBenchmark",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "missing optimization config for benchmark case \(benchmarkCase.id)"
                ]
            )
        }
        result = try runCase(
            benchmarkCase,
            iterations: iterations,
            config: config,
            availability: availability
        )
    } catch {
        result = failed(benchmarkCase, error: "\(error)")
    }
    results.append(result)
    if index < cases.count - 1, result.status != "skipped" {
        cooldown(seconds: cooldownSeconds)
    }
}

results = attachFP16Baselines(results)
let okCount = results.filter { $0.status == "ok" }.count
let skippedCount = results.filter { $0.status == "skipped" }.count
let failedCount = results.filter { $0.status == "failed" }.count
let aggregateQuality = aggregateQualityGate(results)
let optimizationConfigReports = benchmarkConfigs.map {
    reportConfig(for: $0, cases: cases, results: results, availability: availability)
}
let primaryConfigReport =
    optimizationConfigReports.first(where: { $0.isFP16Baseline != true })
    ?? optimizationConfigReports.first

let modelConfig = BenchmarkModelConfig(
    family: environmentValue("TURBOQUANT_BENCHMARK_MODEL_FAMILY", default: "synthetic"),
    hiddenSize: argumentValue("--hidden-size", default: 4096),
    layerCount: argumentValue("--layers", default: 32),
    attentionHeads: queryHeads,
    kvHeads: kvHeads,
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
    turboQuant: primaryConfigReport
        ?? BenchmarkTurboQuantConfig(
            codec: codec.rawValue,
            layoutVersion: TurboQuantAttentionLayout.currentVersion,
            path: nil,
            backend: "unknown",
            preset: "unknown",
            keyBits: 0,
            valueBits: 0,
            groupSize: 0,
            scaleBiasBytes: nil,
            fallbackPolicy: "unknown",
            kernelProfile: availability.selectedKernelProfile.rawValue,
            actualBytesPerToken: nil
        ),
    optimizationConfigs: optimizationConfigReports,
    cooldownSeconds: cooldownSeconds,
    qualityGate: aggregateQuality,
    coverageMatrix: cases,
    results: results,
    summary: BenchmarkSummary(ok: okCount, skipped: skippedCount, failed: failedCount)
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
