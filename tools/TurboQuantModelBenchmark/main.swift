import Foundation
import MLX
import MLXLMCommon

struct BenchmarkResult: Codable {
    var name: String
    var status: String
    var shape: [Int]
    var latencySeconds: Double?
    var error: String?
}

struct BenchmarkReport: Codable {
    var generatedAt: String
    var iterations: Int
    var device: TurboQuantDeviceCapabilities
    var supportsMetalCodec: Bool
    var supportsMetalAttention: Bool
    var activeAttentionPath: TurboQuantAttentionPath?
    var selectedKernelProfile: TurboQuantKernelProfile
    var fallbackReason: String?
    var results: [BenchmarkResult]
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

func skipped(_ name: String, reason: String) -> BenchmarkResult {
    BenchmarkResult(name: name, status: "skipped", shape: [], error: reason)
}

let iterations = argumentValue("--iterations", default: 10)
let availability = TurboQuantKernelAvailability.current
var results: [BenchmarkResult] = []
var activePath: TurboQuantAttentionPath?
var fallbackReason: String?

if availability.supportsMetalPolarQJLAttention {
    do {
        let cache = TurboQuantKVCache(
            preset: .turbo4v2,
            backend: .metalPolarQJL,
            optimizationPolicy: .preferThroughput
        )
        let keys = MLXArray(values(count: 1 * 2 * 256 * 128, scale: 0.007), [1, 2, 256, 128])
        let valuesArray = MLXArray(
            values(count: 1 * 2 * 256 * 128, scale: 0.011, phase: 0.2),
            [1, 2, 256, 128]
        )
        let queries = MLXArray(values(count: 1 * 4 * 1 * 128, scale: 0.019), [1, 4, 1, 128])
        let scale = 1 / sqrt(Float(queries.dim(-1)))
        _ = cache.supportsCompressedAttention(
            queries: queries,
            keys: keys,
            values: valuesArray,
            mask: .causal
        )

        let prefillStart = Date.timeIntervalSinceReferenceDate
        let (prefillKeys, prefillValues) = try cache.updateCompressed(
            keys: keys,
            values: valuesArray
        )
        let prefill = try turboQuantMetalDecodeAttention(prefillValues, outputDType: .float32)
            + MLXArray(Float(prefillKeys.layout.logicalLength))
        eval(prefill)
        let prefillLatency = Date.timeIntervalSinceReferenceDate - prefillStart
        results.append(
            BenchmarkResult(
                name: "cache.prefill_compressed",
                status: "ok",
                shape: prefill.shape,
                latencySeconds: prefillLatency
            ))

        guard let compressed = cache.compressedState else {
            throw NSError(
                domain: "TurboQuantModelBenchmark",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "compressed cache state was not produced"]
            )
        }
        let (attentionLatency, output) = try timed(iterations: iterations) {
            try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: compressed.0,
                valueCode: compressed.1,
                scale: scale,
                mask: .causal,
                preferOnlineFused: cache.prefersOnlineFusedAttention,
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
            )
        }
        activePath = cache.attentionDiagnostics.activeAttentionPath
        fallbackReason = cache.attentionDiagnostics.fallbackReason
        results.append(
            BenchmarkResult(
                name: "cache.decode_attention",
                status: "ok",
                shape: output.shape,
                latencySeconds: attentionLatency
            ))

        let rotating = RotatingTurboQuantKVCache(
            maxSize: 512,
            keep: 4,
            preset: .turbo4v2,
            backend: .metalPolarQJL
        )
        let longKeys = MLXArray(values(count: 1 * 2 * 768 * 128, scale: 0.005), [1, 2, 768, 128])
        let longValues = MLXArray(
            values(count: 1 * 2 * 768 * 128, scale: 0.006, phase: 0.4),
            [1, 2, 768, 128]
        )
        let growthStart = Date.timeIntervalSinceReferenceDate
        let rotatingCompressed = try rotating.updateCompressed(keys: longKeys, values: longValues)
        let decodedRotating = try turboQuantMetalDecodeAttention(
            rotatingCompressed.1,
            outputDType: .float32
        )
        eval(decodedRotating)
        let growthLatency = Date.timeIntervalSinceReferenceDate - growthStart
        results.append(
            BenchmarkResult(
                name: "rotating_cache.long_context_growth",
                status: "ok",
                shape: rotating.compressedState?.1.layout.logicalShape ?? [],
                latencySeconds: growthLatency
            ))
    } catch {
        results.append(
            BenchmarkResult(name: "turboquant_model_cache", status: "failed", shape: [], error: "\(error)"))
    }
} else {
    results.append(skipped("turboquant_model_cache", reason: "Metal attention unavailable or probe failed"))
}

let report = BenchmarkReport(
    generatedAt: ISO8601DateFormatter().string(from: Date()),
    iterations: iterations,
    device: TurboQuantDeviceCapabilities.current,
    supportsMetalCodec: availability.supportsMetalPolarQJLCodec,
    supportsMetalAttention: availability.supportsMetalPolarQJLAttention,
    activeAttentionPath: activePath,
    selectedKernelProfile: availability.selectedKernelProfile,
    fallbackReason: fallbackReason,
    results: results
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
