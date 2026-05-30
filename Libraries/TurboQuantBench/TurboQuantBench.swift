// Copyright © 2026 RNT56.
//
// On-device A-series benchmark harness for the TurboQuant compressed-KV attention
// kernels. Measures decode-time attention throughput (compressed vs plain FP16),
// reconstruction quality (cosine similarity vs the FP16 reference), and KV memory
// footprint at the production Qwen3.5-2B geometry across a context-length sweep.
//
// This is an additive validation surface built directly on the public MLX /
// MLXLMCommon kernel APIs (`turboQuantMetalEncodeAttention`,
// `turboQuantMetalScaledDotProductAttention`, `TurboQuantKVCache`). It drives the
// exact Metal kernels the production cache uses, so a unit test running it on a
// physical A-series device yields production-faithful numbers — the measurement
// gate the overhaul plan calls "highest value, unblocks everything" (it cannot be
// produced on a Mac or the iOS Simulator, both of which run the desktop GPU).
//
// The richer `TurboQuantQwenProof` CLI tool remains the macOS validation surface;
// this library deliberately re-expresses only the minimal measurement core over
// the same public kernels rather than coupling to that tool's executable target.

import Foundation
import MLX
import MLXLMCommon

/// One attention-shape benchmark configuration: model geometry + context + scheme.
public struct TurboQuantBenchCase: Sendable {
    public var label: String
    public var kvHeads: Int
    public var queryHeads: Int
    public var headDimension: Int
    public var contextLength: Int
    /// Query rows attended per measured iteration. Decode = 1 (the production hot path).
    public var queryLength: Int
    public var scheme: TurboQuantScheme
    public var dtype: DType

    public init(
        label: String,
        kvHeads: Int,
        queryHeads: Int,
        headDimension: Int,
        contextLength: Int,
        queryLength: Int = 1,
        scheme: TurboQuantScheme,
        dtype: DType = .float16
    ) {
        self.label = label
        self.kvHeads = kvHeads
        self.queryHeads = queryHeads
        self.headDimension = headDimension
        self.contextLength = contextLength
        self.queryLength = queryLength
        self.scheme = scheme
        self.dtype = dtype
    }

    /// Production Qwen3.5-2B attention geometry (kv=4, q=16, head_dim=256) at decode.
    public static func qwen35_2B(
        contextLength: Int,
        scheme: TurboQuantScheme,
        dtype: DType = .float16
    ) -> TurboQuantBenchCase {
        TurboQuantBenchCase(
            label: "qwen3.5-2b",
            kvHeads: 4,
            queryHeads: 16,
            headDimension: 256,
            contextLength: contextLength,
            queryLength: 1,
            scheme: scheme,
            dtype: dtype
        )
    }
}

/// Result of one benchmark case. `Codable` so callers can emit JSON to read off-device.
public struct TurboQuantBenchResult: Codable, Sendable {
    public enum Status: String, Codable, Sendable {
        case ok
        case skipped
        case failed
    }

    public var label: String
    public var scheme: String
    public var contextLength: Int
    public var status: Status
    /// Skip reason or error description when `status != .ok`.
    public var detail: String?

    // Throughput — decode rows per second at the measured query length.
    public var compressedTokensPerSecond: Double
    public var plainTokensPerSecond: Double
    /// compressed ÷ plain. ≥ 1 ⟹ compressed at least as fast (the "close to FP16" target).
    public var speedRatioToPlain: Double

    // Quality vs the plain FP16 reference output.
    public var cosineSimilarity: Double
    public var maxAbsErrorP95: Double
    public var finite: Bool

    // Memory.
    public var compressedKVBytes: Int
    public var plainKVBytes: Int
    /// plain ÷ compressed. > 1 ⟹ compressed smaller (the context-unlock metric).
    public var memoryReductionRatio: Double

    fileprivate static func skipped(_ c: TurboQuantBenchCase, _ detail: String) -> TurboQuantBenchResult {
        TurboQuantBenchResult(
            label: c.label, scheme: c.scheme.rawValue, contextLength: c.contextLength,
            status: .skipped, detail: detail,
            compressedTokensPerSecond: 0, plainTokensPerSecond: 0, speedRatioToPlain: 0,
            cosineSimilarity: 0, maxAbsErrorP95: .greatestFiniteMagnitude, finite: false,
            compressedKVBytes: 0, plainKVBytes: 0, memoryReductionRatio: 0
        )
    }

    fileprivate static func failed(_ c: TurboQuantBenchCase, _ detail: String) -> TurboQuantBenchResult {
        var result = skipped(c, detail)
        result.status = .failed
        return result
    }
}

public enum TurboQuantBench {
    /// Measure one case against the supplied production profile.
    ///
    /// Mirrors the QwenProof harness's measurement core (deterministic synthetic
    /// K/V/Q at the case geometry → Metal encode → time plain vs compressed SDPA →
    /// row-wise cosine + KV byte counts) using the same public kernels the
    /// production `TurboQuantKVCache` selects, so the numbers are device-faithful.
    /// Never throws: shape/precision rejections surface as `.skipped`, runtime
    /// kernel errors as `.failed`, so a sweep always returns a full result row set.
    public static func measure(
        profile: TurboQuantProfile,
        _ benchCase: TurboQuantBenchCase,
        iterations: Int = 16,
        warmupIterations: Int = 4
    ) -> TurboQuantBenchResult {
        guard let precision = profile.applyingPrecisionCandidate(benchCase.scheme) else {
            return .skipped(
                benchCase, "scheme is not a valid precision candidate for profile \(profile.id)")
        }

        let preset = precision.recommendedScheme.preset
        let kvHeads = benchCase.kvHeads
        let queryHeads = benchCase.queryHeads
        let headDim = benchCase.headDimension
        let ctx = benchCase.contextLength
        let qLen = benchCase.queryLength
        let dtype = benchCase.dtype

        let keys = MLXArray(
            deterministicValues(count: kvHeads * ctx * headDim, scale: 0.0037, phase: 0.11),
            [1, kvHeads, ctx, headDim]
        ).asType(dtype)
        let values = MLXArray(
            deterministicValues(count: kvHeads * ctx * headDim, scale: 0.0041, phase: 0.29),
            [1, kvHeads, ctx, headDim]
        ).asType(dtype)
        let queries = MLXArray(
            deterministicValues(count: queryHeads * qLen * headDim, scale: 0.0061, phase: 0.43),
            [1, queryHeads, qLen, headDim]
        ).asType(dtype)
        let scale = 1 / Float(headDim).squareRoot()

        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: precision.groupSize,
            backend: precision.backend,
            optimizationPolicy: precision.optimizationPolicy,
            fallbackPolicy: precision.turboQuant.fallbackPolicy,
            valueBits: precision.valueBits
        )
        guard
            cache.supportsCompressedAttention(
                queries: queries, keys: keys, values: values, mask: .causal)
        else {
            return .skipped(
                benchCase,
                cache.attentionDiagnostics.lastUnsupportedShape
                    ?? "compressed attention unsupported for this shape")
        }

        let keyConfiguration = TurboQuantConfiguration(
            preset: preset, role: .key, groupSize: precision.groupSize, backend: precision.backend)
        let valueConfiguration = TurboQuantConfiguration(
            preset: preset, role: .value, groupSize: precision.groupSize,
            backend: precision.backend,
            seed: 0x9E37_79B9_7F4A_7C15 ^ 0xD1B5_4A32_D192_ED03, valueBits: precision.valueBits)

        do {
            let compressedKeys = try turboQuantMetalEncodeAttention(
                keys, configuration: keyConfiguration, capacity: ctx, logicalLength: ctx)
            let compressedValues = try turboQuantMetalEncodeAttention(
                values, configuration: valueConfiguration, capacity: ctx, logicalLength: ctx)
            let preferOnline = cache.prefersOnlineFusedAttention
            let kernelProfile = cache.attentionDiagnostics.selectedKernelProfile

            let plain = try timedMedianSeconds(iterations: iterations, warmup: warmupIterations) {
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
            }
            let compressed = try timedMedianSeconds(
                iterations: iterations, warmup: warmupIterations
            ) {
                try turboQuantMetalScaledDotProductAttention(
                    queries: queries, keyCode: compressedKeys, valueCode: compressedValues,
                    scale: scale, mask: .causal, preferOnlineFused: preferOnline,
                    kernelProfile: kernelProfile, blockParallelTokenBlockSize: nil)
            }

            let quality = reconstructionQuality(
                candidate: compressed.output, reference: plain.output)
            let compressedBytes = compressedKeys.storageByteCount + compressedValues.storageByteCount
            let plainBytes = keys.nbytes + values.nbytes
            let rows = Double(qLen)
            let compressedTPS = rows / Swift.max(compressed.median, Double.leastNonzeroMagnitude)
            let plainTPS = rows / Swift.max(plain.median, Double.leastNonzeroMagnitude)

            return TurboQuantBenchResult(
                label: benchCase.label,
                scheme: benchCase.scheme.rawValue,
                contextLength: ctx,
                status: .ok,
                detail: nil,
                compressedTokensPerSecond: compressedTPS,
                plainTokensPerSecond: plainTPS,
                speedRatioToPlain: compressedTPS
                    / Swift.max(plainTPS, Double.leastNonzeroMagnitude),
                cosineSimilarity: quality.cosine,
                maxAbsErrorP95: quality.maxAbsP95,
                finite: quality.finite,
                compressedKVBytes: compressedBytes,
                plainKVBytes: plainBytes,
                memoryReductionRatio: Double(plainBytes) / Double(Swift.max(1, compressedBytes))
            )
        } catch {
            return .failed(benchCase, String(describing: error))
        }
    }

    /// Run `measure` over every supplied case, returning one result row per case.
    public static func sweep(
        profile: TurboQuantProfile,
        cases: [TurboQuantBenchCase],
        iterations: Int = 16,
        warmupIterations: Int = 4
    ) -> [TurboQuantBenchResult] {
        cases.map {
            measure(
                profile: profile, $0, iterations: iterations, warmupIterations: warmupIterations)
        }
    }

    /// Render a fixed-width table of results for printing to a test/console log.
    public static func renderTable(_ results: [TurboQuantBenchResult]) -> String {
        var lines = [
            "scheme    ctx      status   comp tok/s  plain tok/s  ratio   cosine     mem×",
            "-------   ------   ------   ----------  -----------  -----   --------   -----",
        ]
        for r in results {
            let ctxLabel =
                r.contextLength >= 1024
                ? "\(r.contextLength / 1024)K" : "\(r.contextLength)"
            lines.append(
                pad(r.scheme, 8) + "  " + pad(ctxLabel, 6) + "   " + pad(r.status.rawValue, 7)
                    + "  " + pad(fmt(r.compressedTokensPerSecond, 1), 10) + "  "
                    + pad(fmt(r.plainTokensPerSecond, 1), 11) + "  " + pad(fmt(r.speedRatioToPlain, 2), 6)
                    + "  " + pad(fmt(r.cosineSimilarity, 6), 9) + "  "
                    + pad(fmt(r.memoryReductionRatio, 2), 5))
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Measurement helpers (compact, deterministic; mirror the QwenProof core).

private enum TurboQuantBenchError: Error { case noMeasuredIteration }

private struct TimedMedian {
    var median: Double
    var output: MLXArray
}

/// Warm up, then time `iterations` evaluated runs and return the median wall-clock
/// seconds plus the final output (for the quality comparison).
private func timedMedianSeconds(
    iterations: Int,
    warmup: Int,
    _ body: () throws -> MLXArray
) throws -> TimedMedian {
    var last: MLXArray?
    for _ in 0 ..< Swift.max(0, warmup) {
        let warmRun = try body()
        eval(warmRun)
        last = warmRun
    }
    let measured = Swift.max(1, iterations)
    var samples = [Double]()
    samples.reserveCapacity(measured)
    for _ in 0 ..< measured {
        let start = Date.timeIntervalSinceReferenceDate
        let output = try body()
        eval(output)
        last = output
        samples.append(Date.timeIntervalSinceReferenceDate - start)
    }
    guard let output = last else { throw TurboQuantBenchError.noMeasuredIteration }
    samples.sort()
    return TimedMedian(median: samples[samples.count / 2], output: output)
}

private func deterministicValues(count: Int, scale: Double, phase: Double) -> [Float] {
    (0 ..< count).map { index in
        let position = Double(index)
        return Float(0.31 * sin(position * scale + phase) + 0.17 * cos(position * 0.037))
    }
}

/// Row-wise mean cosine similarity + p95 max-abs error of `candidate` vs `reference`.
private func reconstructionQuality(
    candidate: MLXArray,
    reference: MLXArray
) -> (cosine: Double, maxAbsP95: Double, finite: Bool) {
    eval(candidate, reference)
    let candidateValues = candidate.asArray(Float.self)
    let referenceValues = reference.asArray(Float.self)
    guard candidateValues.count == referenceValues.count,
        let rowWidth = candidate.shape.last, rowWidth > 0, !candidateValues.isEmpty
    else {
        return (0, .greatestFiniteMagnitude, false)
    }

    let rowCount = candidateValues.count / rowWidth
    var cosineTotal = 0.0
    var maxErrors = [Double]()
    maxErrors.reserveCapacity(rowCount)
    for row in 0 ..< rowCount {
        let start = row * rowWidth
        let end = start + rowWidth
        var dot = 0.0
        var candidateNorm = 0.0
        var referenceNorm = 0.0
        var maxError = 0.0
        for index in start ..< end {
            let c = Double(candidateValues[index])
            let r = Double(referenceValues[index])
            dot += c * r
            candidateNorm += c * c
            referenceNorm += r * r
            maxError = Swift.max(maxError, abs(c - r))
        }
        let denominator = candidateNorm.squareRoot() * referenceNorm.squareRoot()
        cosineTotal += denominator > 0 ? dot / denominator : 0
        maxErrors.append(maxError)
    }

    let cosine = cosineTotal / Double(Swift.max(1, rowCount))
    maxErrors.sort()
    let p95Index = Swift.min(
        maxErrors.count - 1,
        Swift.max(0, Int((0.95 * Double(maxErrors.count)).rounded(.up)) - 1))
    let finite =
        candidateValues.allSatisfy(\.isFinite) && referenceValues.allSatisfy(\.isFinite)
    return (cosine, maxErrors.isEmpty ? .greatestFiniteMagnitude : maxErrors[p95Index], finite)
}

private func pad(_ string: String, _ width: Int) -> String {
    string.count >= width
        ? string : string + String(repeating: " ", count: width - string.count)
}

private func fmt(_ value: Double, _ decimals: Int) -> String {
    guard value.isFinite else { return "n/a" }
    return String(format: "%.\(decimals)f", value)
}
