import Darwin
import Foundation
import MLX
import MLXLMCommon

struct BenchmarkReport: Codable {
    var schemaVersion: Int = 1
    var generatedAt: String
    var context: Int
    var appendIterations: Int
    var attentionIterations: Int
    var warmupIterations: Int
    var queryHeads: Int
    var kvHeads: Int
    var headDimension: Int
    var valueBits: Int
    var prefillUpdateMS: Double
    var appendUpdateMedianMS: Double
    var appendUpdateP95MS: Double
    var directPackedAttentionMedianMS: Double
    var directPackedAttentionP95MS: Double
    var cacheBackedAttentionMedianMS: Double
    var cacheBackedAttentionP95MS: Double
    var directValueEncodeMedianMS: Double
    var directValueEncodeP95MS: Double
    var directReferenceValueEncodeMedianMS: Double
    var directReferenceValueEncodeP95MS: Double
    var directAffineKeyQuantizeMedianMS: Double
    var directAffineKeyQuantizeP95MS: Double
    var directFusedHybridEncodeMedianMS: Double
    var directFusedHybridEncodeP95MS: Double
    var keyStateFetchMedianMS: Double
    var keyStateFetchP95MS: Double
    var valueStateFetchMedianMS: Double
    var valueStateFetchP95MS: Double
    var cacheLogicalLength: Int
    var cacheKeyBytes: Int
    var cacheValueBytes: Int
    var polarWHTValueBytes: Int
    var decodedActiveValueBytes: Int
    var cacheResidentBytes: Int
    var selectedPath: String
    var selectedReason: String?
    var outputChecksum: Float
    var directPackedOutputChecksum: Float
}

struct TimedSamples {
    var median: Double
    var p95: Double
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

private func argumentUInt64(_ name: String, default defaultValue: UInt64) -> UInt64 {
    argumentString(name).flatMap(UInt64.init) ?? defaultValue
}

private func printUsage() {
    print(
        """
        TurboQuantCacheUpdateBenchmark

        Measures LM cache update overhead for hybrid K8 + PolarWHT-V.

        Options:
          --context <n>              Default: 4096
          --append-iterations <n>    Default: 32
          --attention-iterations <n> Default: 32
          --warmup <n>               Default: 3
          --query-heads <n>          Default: 16
          --kv-heads <n>             Default: 8
          --head-dim <n>             Default: 128
          --value-bits <n>           Default: 4
          --seed <n>                 Default: 545120260605
          --json                     Emit JSON only.
          --output <path>            Write JSON report.
          --help                     Print this help.
        """
    )
}

private func percentileIndex(_ count: Int, percentile: Double) -> Int {
    min(count - 1, max(0, Int((percentile * Double(count)).rounded(.up)) - 1))
}

private func summarize(_ samples: [Double]) -> TimedSamples {
    let sorted = samples.sorted()
    guard let first = sorted.first else { return TimedSamples(median: 0, p95: 0) }
    let median = sorted[percentileIndex(sorted.count, percentile: 0.50)]
    let p95 = sorted[percentileIndex(sorted.count, percentile: 0.95)]
    return TimedSamples(median: median.isFinite ? median : first, p95: p95.isFinite ? p95 : first)
}

private func timedArray(iterations: Int, warmup: Int, _ body: () throws -> MLXArray) throws
    -> (TimedSamples, MLXArray)
{
    var last = MLXArray.zeros([1], dtype: .float32)
    for _ in 0 ..< max(0, warmup) {
        last = try body()
        eval(last)
        Stream().synchronize()
    }

    var samples: [Double] = []
    samples.reserveCapacity(max(1, iterations))
    for _ in 0 ..< max(1, iterations) {
        let start = DispatchTime.now().uptimeNanoseconds
        last = try body()
        eval(last)
        Stream().synchronize()
        let end = DispatchTime.now().uptimeNanoseconds
        samples.append(Double(end - start) / 1_000_000.0)
    }
    return (summarize(samples), last)
}

private func timedVoid(iterations: Int, warmup: Int, _ body: () throws -> Void) throws
    -> TimedSamples
{
    for _ in 0 ..< max(0, warmup) {
        try body()
        Stream().synchronize()
    }

    var samples: [Double] = []
    samples.reserveCapacity(max(1, iterations))
    for _ in 0 ..< max(1, iterations) {
        let start = DispatchTime.now().uptimeNanoseconds
        try body()
        Stream().synchronize()
        let end = DispatchTime.now().uptimeNanoseconds
        samples.append(Double(end - start) / 1_000_000.0)
    }
    return summarize(samples)
}

private func waveArray(shape: [Int], seed: UInt64, dtype: DType = .float32) -> MLXArray {
    let count = shape.reduce(1, *)
    let seedTerm = Double(seed % 997) / 997.0
    let values = (0 ..< count).map { index -> Float in
        let x = Double(index) + seedTerm
        return Float(sin(x * 0.013) * 0.17 + cos(x * 0.031) * 0.11)
    }
    return MLXArray(values, shape).asType(dtype)
}

private func makeCache(valueBits: Int) -> TurboQuantKVCache {
    TurboQuantKVCache(
        preset: .turbo8,
        backend: .metalPolarWHT,
        optimizationPolicy: .preferThroughput,
        fallbackPolicy: .fatalOnFailure,
        valueBits: valueBits,
        precisionPolicy: TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: valueBits),
            boundary: .disabled
        )
    )
}

if CommandLine.arguments.contains("--help") {
    printUsage()
    exit(0)
}

let context = argumentInt("--context", default: 4096)
let appendIterations = argumentInt("--append-iterations", default: 32)
let attentionIterations = argumentInt("--attention-iterations", default: 32)
let warmupIterations = argumentInt("--warmup", default: 3)
let queryHeads = argumentInt("--query-heads", default: 16)
let kvHeads = argumentInt("--kv-heads", default: 8)
let headDimension = argumentInt("--head-dim", default: 128)
let valueBits = argumentInt("--value-bits", default: 4)
let seed = argumentUInt64("--seed", default: 545_120_260_605)
let jsonOnly = CommandLine.arguments.contains("--json")
let outputPath = argumentString("--output")

let keys = waveArray(shape: [1, kvHeads, context, headDimension], seed: seed)
let values = waveArray(shape: [1, kvHeads, context, headDimension], seed: seed ^ 0xD1B5)
let appendKeys = waveArray(
    shape: [1, kvHeads, appendIterations + warmupIterations + 1, headDimension],
    seed: seed ^ 0xA11CE
)
let appendValues = waveArray(
    shape: [1, kvHeads, appendIterations + warmupIterations + 1, headDimension],
    seed: seed ^ 0xB0B
)
let queries = waveArray(shape: [1, queryHeads, 1, headDimension], seed: seed ^ 0x513)
let scale = 1 / sqrt(Float(headDimension))

let cache = makeCache(valueBits: valueBits)
let prefillStart = DispatchTime.now().uptimeNanoseconds
_ = try cache.updateCompressed(keys: keys, values: values)
Stream().synchronize()
let prefillEnd = DispatchTime.now().uptimeNanoseconds
let prefillUpdateMS = Double(prefillEnd - prefillStart) / 1_000_000.0

var appendIndex = 0
let appendStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    let token = appendIndex
    appendIndex += 1
    let range = token ..< (token + 1)
    _ = try cache.updateCompressed(
        keys: appendKeys[0..., 0..., range, 0...],
        values: appendValues[0..., 0..., range, 0...]
    )
}

let valueEncodeStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    let token = appendIndex % appendKeys.dim(2)
    appendIndex += 1
    let range = token ..< (token + 1)
    let code = try turboQuantMetalPolarWHTEncodeAttentionValues(
        appendValues[0..., 0..., range, 0...],
        bits: valueBits,
        seed: seed ^ 0xD1B5_4A32_D192_ED03,
        capacity: 1,
        logicalLength: 1
    )
    eval(code.packedIndices, code.norms)
}

let referenceValueEncodeStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    let token = appendIndex % appendKeys.dim(2)
    appendIndex += 1
    let range = token ..< (token + 1)
    let code = try turboQuantPolarWHTReferenceEncodeAttentionValues(
        appendValues[0..., 0..., range, 0...],
        bits: valueBits,
        seed: seed ^ 0xD1B5_4A32_D192_ED03,
        capacity: 1,
        logicalLength: 1
    )
    eval(code.packedIndices, code.norms)
}

let affineKeyQuantizeStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    let token = appendIndex % appendKeys.dim(2)
    appendIndex += 1
    let range = token ..< (token + 1)
    let q = quantized(
        appendKeys[0..., 0..., range, 0...],
        groupSize: 64,
        bits: 8,
        mode: .affine
    )
    eval([q.wq, q.scales] + [q.biases].compactMap { $0 })
}

let fusedHybridEncodeStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    let token = appendIndex % appendKeys.dim(2)
    appendIndex += 1
    let range = token ..< (token + 1)
    let fused = try turboQuantMetalHybridAffineK8PolarWHTValueEncode(
        keys: appendKeys[0..., 0..., range, 0...].contiguous(stream: .gpu),
        values: appendValues[0..., 0..., range, 0...].contiguous(stream: .gpu),
        keyGroupSize: 64,
        valueBits: valueBits,
        valueSeed: seed ^ 0xD1B5_4A32_D192_ED03,
        capacity: 1,
        logicalLength: 1
    )
    eval(
        [fused.key.weight, fused.key.scales, fused.value.packedIndices, fused.value.norms]
            + [fused.key.biases].compactMap { $0 }
    )
}

let keyStateFetchStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    guard let state = cache.hybridAffineKeyState else {
        throw NSError(
            domain: "TurboQuantCacheUpdateBenchmark",
            code: 3,
            userInfo: [NSLocalizedDescriptionKey: "hybrid affine key state unavailable"]
        )
    }
    eval([state.0, state.1] + [state.2].compactMap { $0 })
}

let valueStateFetchStats = try timedVoid(
    iterations: appendIterations,
    warmup: warmupIterations
) {
    guard let code = cache.polarWHTValueState else {
        throw NSError(
            domain: "TurboQuantCacheUpdateBenchmark",
            code: 4,
            userInfo: [NSLocalizedDescriptionKey: "PolarWHT value state unavailable"]
        )
    }
    eval(code.packedIndices, code.norms)
}

guard let keyState = cache.hybridAffineKeyStateForAttention,
    var valueCode = cache.polarWHTValueStateForAttention,
    let keyBiases = keyState.2
else {
    throw NSError(
        domain: "TurboQuantCacheUpdateBenchmark",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "hybrid affine K + PolarWHT-V state unavailable"]
    )
}
valueCode.packedIndices = valueCode.packedIndices.contiguous(stream: .gpu)
valueCode.norms = valueCode.norms.contiguous(stream: .gpu)

let (directPackedAttentionStats, directPackedOutput) = try timedArray(
    iterations: attentionIterations,
    warmup: warmupIterations
) {
    guard
        let fused = try turboQuantMetalHybridAffineK8PolarWHTValueScaledDotProductAttentionIfSupported(
            queries: queries.contiguous(stream: .gpu),
            keyWeight: keyState.0.contiguous(stream: .gpu),
            keyScales: keyState.1.contiguous(stream: .gpu),
            keyBiases: keyBiases.contiguous(stream: .gpu),
            keyGroupSize: 64,
            valueCode: valueCode,
            scale: scale,
            mask: .causal,
            outputDType: .float32
        )
    else {
        throw NSError(
            domain: "TurboQuantCacheUpdateBenchmark",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "native hybrid attention did not admit shape"]
        )
    }
    return fused
}

guard let compressedState = cache.compressedState else {
    throw NSError(
        domain: "TurboQuantCacheUpdateBenchmark",
        code: 5,
        userInfo: [NSLocalizedDescriptionKey: "compressed TurboQuant state unavailable"]
    )
}

let (cacheBackedAttentionStats, cacheBackedOutput) = try timedArray(
    iterations: attentionIterations,
    warmup: warmupIterations
) {
    try attentionWithKVStateThrowing(
        queries: queries.contiguous(stream: .gpu),
        state: .turboQuant(keys: compressedState.0, values: compressedState.1, cache: cache),
        scale: scale,
        mask: .causal
    )
}

let snapshot = cache.runtimeSnapshot()
let checksum = cacheBackedOutput.asType(.float32).sum().item(Float.self)
let directPackedChecksum = directPackedOutput.asType(.float32).sum().item(Float.self)
let report = BenchmarkReport(
    generatedAt: ISO8601DateFormatter().string(from: Date()),
    context: context,
    appendIterations: appendIterations,
    attentionIterations: attentionIterations,
    warmupIterations: warmupIterations,
    queryHeads: queryHeads,
    kvHeads: kvHeads,
    headDimension: headDimension,
    valueBits: valueBits,
    prefillUpdateMS: prefillUpdateMS,
    appendUpdateMedianMS: appendStats.median,
    appendUpdateP95MS: appendStats.p95,
    directPackedAttentionMedianMS: directPackedAttentionStats.median,
    directPackedAttentionP95MS: directPackedAttentionStats.p95,
    cacheBackedAttentionMedianMS: cacheBackedAttentionStats.median,
    cacheBackedAttentionP95MS: cacheBackedAttentionStats.p95,
    directValueEncodeMedianMS: valueEncodeStats.median,
    directValueEncodeP95MS: valueEncodeStats.p95,
    directReferenceValueEncodeMedianMS: referenceValueEncodeStats.median,
    directReferenceValueEncodeP95MS: referenceValueEncodeStats.p95,
    directAffineKeyQuantizeMedianMS: affineKeyQuantizeStats.median,
    directAffineKeyQuantizeP95MS: affineKeyQuantizeStats.p95,
    directFusedHybridEncodeMedianMS: fusedHybridEncodeStats.median,
    directFusedHybridEncodeP95MS: fusedHybridEncodeStats.p95,
    keyStateFetchMedianMS: keyStateFetchStats.median,
    keyStateFetchP95MS: keyStateFetchStats.p95,
    valueStateFetchMedianMS: valueStateFetchStats.median,
    valueStateFetchP95MS: valueStateFetchStats.p95,
    cacheLogicalLength: snapshot.logicalLength,
    cacheKeyBytes: snapshot.keyBytes,
    cacheValueBytes: snapshot.valueBytes,
    polarWHTValueBytes: snapshot.polarWHTValueBytes,
    decodedActiveValueBytes: snapshot.decodedActiveValueBytes,
    cacheResidentBytes: cache.cacheFootprint.residentBytes,
    selectedPath: cache.attentionDiagnostics.activeAttentionPath.rawValue,
    selectedReason: cache.fallbackResults.last?.reason,
    outputChecksum: checksum,
    directPackedOutputChecksum: directPackedChecksum
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let encoded = try encoder.encode(report)
if let outputPath {
    try encoded.write(to: URL(fileURLWithPath: outputPath))
}
if jsonOnly {
    print(String(decoding: encoded, as: UTF8.self))
} else {
    print(
        """
        context=\(context) qh=\(queryHeads) kvh=\(kvHeads) hd=\(headDimension) valueBits=\(valueBits)
        prefillUpdateMS=\(String(format: "%.3f", prefillUpdateMS))
        appendUpdateMedianMS=\(String(format: "%.3f", appendStats.median)) p95=\(String(format: "%.3f", appendStats.p95))
        directValueEncodeMedianMS=\(String(format: "%.3f", valueEncodeStats.median)) p95=\(String(format: "%.3f", valueEncodeStats.p95))
        directReferenceValueEncodeMedianMS=\(String(format: "%.3f", referenceValueEncodeStats.median)) p95=\(String(format: "%.3f", referenceValueEncodeStats.p95))
        directAffineKeyQuantizeMedianMS=\(String(format: "%.3f", affineKeyQuantizeStats.median)) p95=\(String(format: "%.3f", affineKeyQuantizeStats.p95))
        directFusedHybridEncodeMedianMS=\(String(format: "%.3f", fusedHybridEncodeStats.median)) p95=\(String(format: "%.3f", fusedHybridEncodeStats.p95))
        keyStateFetchMedianMS=\(String(format: "%.3f", keyStateFetchStats.median)) p95=\(String(format: "%.3f", keyStateFetchStats.p95))
        valueStateFetchMedianMS=\(String(format: "%.3f", valueStateFetchStats.median)) p95=\(String(format: "%.3f", valueStateFetchStats.p95))
        directPackedAttentionMedianMS=\(String(format: "%.3f", directPackedAttentionStats.median)) p95=\(String(format: "%.3f", directPackedAttentionStats.p95))
        cacheBackedAttentionMedianMS=\(String(format: "%.3f", cacheBackedAttentionStats.median)) p95=\(String(format: "%.3f", cacheBackedAttentionStats.p95))
        keyBytes=\(snapshot.keyBytes) valueBytes=\(snapshot.valueBytes) polarWHTValueBytes=\(snapshot.polarWHTValueBytes) decodedActiveValueBytes=\(snapshot.decodedActiveValueBytes) residentBytes=\(cache.cacheFootprint.residentBytes)
        selectedPath=\(cache.attentionDiagnostics.activeAttentionPath.rawValue) reason=\(cache.fallbackResults.last?.reason ?? "none")
        """
    )
}
