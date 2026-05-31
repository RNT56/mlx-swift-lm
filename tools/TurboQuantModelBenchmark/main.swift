import Foundation
import MLX
import MLXLMCommon

enum BenchmarkCodec: String {
    case polarQJL = "polar_qjl"
    case affineInt4 = "affine_int4"
    case raw
}

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
    var quality: TurboQuantQualityGateReport
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
    case "affine_int4", "affineInt4", "affine-int4":
        return .affineInt4
    default:
        return .polarQJL
    }
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
        quality: .failed(reason: reason),
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
        quality: .failed(reason: error),
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

func runCase(
    _ benchmarkCase: BenchmarkCase,
    iterations: Int,
    codec: BenchmarkCodec,
    availability: TurboQuantKernelAvailability
) throws -> BenchmarkResult {
    guard codec != .polarQJL || availability.supportsMetalPolarQJLAttention else {
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
            fallbackReason: nil,
            error: nil
        )
    }

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
        fallbackReason: cache.attentionDiagnostics.fallbackReason,
        error: nil
    )
}

let iterations = argumentValue("--iterations", default: 10)
let codec = argumentCodec("--codec", default: .polarQJL)
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
            try runCase(
                benchmarkCase,
                iterations: iterations,
                codec: codec,
                availability: availability
            ))
    } catch {
        results.append(failed(benchmarkCase, error: "\(error)"))
    }
}

let okCount = results.filter { $0.status == "ok" }.count
let skippedCount = results.filter { $0.status == "skipped" }.count
let failedCount = results.filter { $0.status == "failed" }.count
let aggregateQuality = aggregateQualityGate(results)
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
        codec: codec.rawValue,
        layoutVersion: TurboQuantAttentionLayout.currentVersion,
        path: firstSelectedPath?.rawValue,
        backend: codec == .affineInt4 ? TurboQuantBackend.mlxPacked.rawValue : "metalPolarQJL",
        preset: TurboQuantPreset.turbo4v2.rawValue,
        keyBits: Double(TurboQuantPreset.turbo4v2.targetMagnitudeBits),
        valueBits: codec == .affineInt4
            ? TurboQuantKVCodec.affineInt4Bits : TurboQuantPreset.turbo4v2.defaultValueBits,
        groupSize: codec == .affineInt4 ? TurboQuantKVCodec.affineInt4DefaultGroupSize : 64,
        scaleBiasBytes: codec == .affineInt4
            ? (cases.first.map {
                4 * 2 * 2 * $0.contextLength * max(1, $0.headDim / TurboQuantKVCodec.affineInt4DefaultGroupSize)
            })
            : nil,
        fallbackPolicy: "compressedDecodeAllowed",
        kernelProfile: availability.selectedKernelProfile.rawValue,
        actualBytesPerToken: firstActualBytesPerToken
    ),
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
