import Darwin
import Foundation
import MLX

private struct BenchmarkRow: Codable {
    var label: String
    var context: Int
    var batch: Int
    var queryHeads: Int
    var kvHeads: Int
    var queryLength: Int
    var headDimension: Int
    var dtype: String
    var keyBits: Int
    var keyGroupSize: Int
    var valueBits: Int
    var valueGroupSize: Int
    var iterations: Int
    var warmupIterations: Int
    var cooldownMilliseconds: Int
    var fp16MedianSeconds: Double
    var fp16P95Seconds: Double
    var fp16TokensPerSecond: Double
    var fp16P95TokensPerSecond: Double
    var speedRatioToFP16: Double
    var speedRatioToFP16P95: Double
    var medianSeconds: Double
    var p95Seconds: Double
    var tokensPerSecond: Double
    var p95TokensPerSecond: Double
    var keyBytes: Int
    var valueBytes: Int
    var keyScaleBytes: Int
    var valueScaleBytes: Int
    var keyBiasBytes: Int
    var valueBiasBytes: Int
    var compressedBytes: Int
    var denseKVBytes: Int
    var memoryBytesSavedVsFP16: Int
    var memoryReductionRatioToFP16: Double
    var compressionRatioToDense: Double
    var outputChecksum: Float
    var maxAbsVsK8V4: Double?
    var cosineVsK8V4: Double?
    // Roofline / amortization instrumentation (P0-a, P0-b).
    // gbPerSecond is the achieved KV-read bandwidth: compressed cache bytes read
    // once per decode call / median seconds. msPerQueryToken exposes speculative
    // amortization: if it falls as queryLength rises, one KV scan serves many
    // query rows (the fused kernel amortizes); if it stays flat, KV is re-scanned
    // per query row (no amortization).
    var gbPerSecond: Double
    var fp16GBPerSecond: Double
    var bandwidthRatioToFP16: Double
    var msPerQueryToken: Double
    var fp16MsPerQueryToken: Double
}

private struct BenchmarkReport: Codable {
    var schemaVersion: Int = 3
    var generatedAt: String
    var cooldownMilliseconds: Int
    var rows: [BenchmarkRow]
}

private struct TimedResult {
    var median: Double
    var p95: Double
    var output: MLXArray
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

private func argumentInts(_ name: String, default defaultValue: [Int]) -> [Int] {
    guard let raw = argumentString(name) else { return defaultValue }
    let parsed = raw.split(separator: ",").compactMap { part in
        Int(part.trimmingCharacters(in: .whitespaces))
    }
    return parsed.isEmpty ? defaultValue : parsed
}

private func printUsage() {
    print(
        """
        TurboQuantNativeVxBenchmark

        Synthetic native K8/Vx decode benchmark for MLX mixed quantized attention.

        Options:
          --contexts <csv>      Default: 20480,32768,65536,131072
          --value-bits <csv>    Default: 4,3,2
          --query-lengths <csv> Default: 1  (e.g. 1,4 to probe speculative KV amortization)
          --iterations <n>      Default: 7
          --warmup <n>          Default: 2
          --cooldown-ms <n>     Default: 25
          --query-heads <n>     Default: 16
          --kv-heads <n>        Default: 4
          --head-dim <n>        Default: 256
          --seed <n>            Default: 545120260602
          --output <path>       Write JSON report.
          --help                Print this help.
        """
    )
}

private func cooldown(milliseconds: Int) {
    guard milliseconds > 0 else { return }
    Stream().synchronize()
    Memory.clearCache()
    usleep(useconds_t(milliseconds * 1000))
}

private func percentileIndex(_ count: Int, percentile: Double) -> Int {
    min(count - 1, max(0, Int((percentile * Double(count)).rounded(.up)) - 1))
}

private func timed(
    iterations: Int,
    warmup: Int,
    cooldownMilliseconds: Int,
    _ body: () throws -> MLXArray
) throws -> TimedResult {
    var last: MLXArray?
    for _ in 0 ..< max(0, warmup) {
        let output = try body()
        eval(output)
        Stream().synchronize()
        last = output
        cooldown(milliseconds: cooldownMilliseconds)
    }

    let measured = max(1, iterations)
    var samples = [Double]()
    samples.reserveCapacity(measured)
    for _ in 0 ..< measured {
        let start = DispatchTime.now().uptimeNanoseconds
        let output = try body()
        eval(output)
        Stream().synchronize()
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
        samples.append(elapsed)
        last = output
        cooldown(milliseconds: cooldownMilliseconds)
    }
    samples.sort()
    return TimedResult(
        median: samples[samples.count / 2],
        p95: samples[percentileIndex(samples.count, percentile: 0.95)],
        output: last!
    )
}

private func outputQuality(candidate: MLXArray, reference: MLXArray) -> (maxAbs: Double, cosine: Double) {
    eval(candidate, reference)
    let c = candidate.asArray(Float.self)
    let r = reference.asArray(Float.self)
    guard c.count == r.count, !c.isEmpty else {
        return (.greatestFiniteMagnitude, 0)
    }
    var maxAbs = 0.0
    var dot = 0.0
    var cn = 0.0
    var rn = 0.0
    for index in c.indices {
        let cv = Double(c[index])
        let rv = Double(r[index])
        maxAbs = max(maxAbs, abs(cv - rv))
        dot += cv * rv
        cn += cv * cv
        rn += rv * rv
    }
    let denom = sqrt(cn) * sqrt(rn)
    return (maxAbs, denom == 0 ? 0 : dot / denom)
}

private func rowLabel(valueBits: Int) -> String {
    "affineK8V\(valueBits)"
}

private func pad(_ value: String, _ width: Int) -> String {
    value + String(repeating: " ", count: max(0, width - value.count))
}

private func runContext(
    context: Int,
    queryLength: Int,
    valueBitsList: [Int],
    keyGroupSize: Int,
    valueGroupSize: Int,
    iterations: Int,
    warmup: Int,
    queryHeads: Int,
    kvHeads: Int,
    headDimension: Int,
    seed: UInt64,
    cooldownMilliseconds: Int
) throws -> [BenchmarkRow] {
    let batch = 1
    let keyBits = 8
    let dtype = DType.float16
    let scale = Float(1 / sqrt(Double(headDimension)))

    MLXRandom.seed(seed ^ UInt64(context))
    let queries = (MLXRandom.normal([batch, queryHeads, queryLength, headDimension]) * 0.1)
        .asType(dtype)
    let denseKeys = (MLXRandom.normal([batch, kvHeads, context, headDimension]) * 0.1)
        .asType(dtype)
    let denseValues = (MLXRandom.normal([batch, kvHeads, context, headDimension]) * 0.1)
        .asType(dtype)
    eval(queries, denseKeys, denseValues)
    Stream().synchronize()

    let fp16TimedResult = try timed(
        iterations: iterations,
        warmup: warmup,
        cooldownMilliseconds: cooldownMilliseconds
    ) {
        MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: denseKeys,
            values: denseValues,
            scale: scale,
            mask: .causal
        )
    }
    let fp16TokensPerSecond = Double(queryLength)
        / max(fp16TimedResult.median, .leastNonzeroMagnitude)
    let fp16P95TokensPerSecond = Double(queryLength)
        / max(fp16TimedResult.p95, .leastNonzeroMagnitude)
    let denseBytes = denseKeys.nbytes + denseValues.nbytes
    let fp16GBPerSecond = Double(denseBytes)
        / max(fp16TimedResult.median, .leastNonzeroMagnitude) / 1_000_000_000
    let fp16MsPerQueryToken = fp16TimedResult.median * 1_000 / Double(max(1, queryLength))

    let (quantizedKeys, keyScales, keyBiasesOptional) = quantized(
        denseKeys,
        groupSize: keyGroupSize,
        bits: keyBits,
        mode: .affine
    )
    guard let keyBiases = keyBiasesOptional else {
        throw NSError(domain: "TurboQuantNativeVxBenchmark", code: 1)
    }
    eval(quantizedKeys, keyScales, keyBiases)
    Stream().synchronize()

    var referenceOutput: MLXArray?
    var rows = [BenchmarkRow]()
    for valueBits in valueBitsList {
        let (quantizedValues, valueScales, valueBiasesOptional) = quantized(
            denseValues,
            groupSize: valueGroupSize,
            bits: valueBits,
            mode: .affine
        )
        guard let valueBiases = valueBiasesOptional else {
            throw NSError(domain: "TurboQuantNativeVxBenchmark", code: 2)
        }
        eval(quantizedValues, valueScales, valueBiases)
        Stream().synchronize()

        let timedResult = try timed(
            iterations: iterations,
            warmup: warmup,
            cooldownMilliseconds: cooldownMilliseconds
        ) {
            MLXFast.mixedQuantizedScaledDotProductAttention(
                queries: queries,
                keys: quantizedKeys,
                keyScales: keyScales,
                values: quantizedValues,
                valueScales: valueScales,
                scale: scale,
                keyBiases: keyBiases,
                valueBiases: valueBiases,
                mask: .causal,
                keyGroupSize: keyGroupSize,
                keyBits: keyBits,
                valueGroupSize: valueGroupSize,
                valueBits: valueBits,
                stream: .gpu
            )
        }
        let checksum = timedResult.output.sum().item(Float.self)
        let quality: (maxAbs: Double, cosine: Double)?
        if valueBits == 4 {
            referenceOutput = timedResult.output
            quality = (0, 1)
        } else if let referenceOutput {
            quality = outputQuality(candidate: timedResult.output, reference: referenceOutput)
        } else {
            quality = nil
        }

        let compressedBytes =
            quantizedKeys.nbytes + quantizedValues.nbytes
            + keyScales.nbytes + valueScales.nbytes
            + keyBiases.nbytes + valueBiases.nbytes
        let tokensPerSecond = Double(queryLength) / max(timedResult.median, .leastNonzeroMagnitude)
        let p95TokensPerSecond = Double(queryLength) / max(timedResult.p95, .leastNonzeroMagnitude)
        let gbPerSecond = Double(compressedBytes)
            / max(timedResult.median, .leastNonzeroMagnitude) / 1_000_000_000
        let msPerQueryToken = timedResult.median * 1_000 / Double(max(1, queryLength))
        rows.append(
            BenchmarkRow(
                label: rowLabel(valueBits: valueBits),
                context: context,
                batch: batch,
                queryHeads: queryHeads,
                kvHeads: kvHeads,
                queryLength: queryLength,
                headDimension: headDimension,
                dtype: "\(dtype)",
                keyBits: keyBits,
                keyGroupSize: keyGroupSize,
                valueBits: valueBits,
                valueGroupSize: valueGroupSize,
                iterations: iterations,
                warmupIterations: warmup,
                cooldownMilliseconds: cooldownMilliseconds,
                fp16MedianSeconds: fp16TimedResult.median,
                fp16P95Seconds: fp16TimedResult.p95,
                fp16TokensPerSecond: fp16TokensPerSecond,
                fp16P95TokensPerSecond: fp16P95TokensPerSecond,
                speedRatioToFP16: tokensPerSecond / max(fp16TokensPerSecond, .leastNonzeroMagnitude),
                speedRatioToFP16P95: p95TokensPerSecond
                    / max(fp16P95TokensPerSecond, .leastNonzeroMagnitude),
                medianSeconds: timedResult.median,
                p95Seconds: timedResult.p95,
                tokensPerSecond: tokensPerSecond,
                p95TokensPerSecond: p95TokensPerSecond,
                keyBytes: quantizedKeys.nbytes,
                valueBytes: quantizedValues.nbytes,
                keyScaleBytes: keyScales.nbytes,
                valueScaleBytes: valueScales.nbytes,
                keyBiasBytes: keyBiases.nbytes,
                valueBiasBytes: valueBiases.nbytes,
                compressedBytes: compressedBytes,
                denseKVBytes: denseBytes,
                memoryBytesSavedVsFP16: max(0, denseBytes - compressedBytes),
                memoryReductionRatioToFP16: Double(denseBytes) / Double(max(1, compressedBytes)),
                compressionRatioToDense: Double(denseBytes) / Double(max(1, compressedBytes)),
                outputChecksum: checksum,
                maxAbsVsK8V4: quality?.maxAbs,
                cosineVsK8V4: quality?.cosine,
                gbPerSecond: gbPerSecond,
                fp16GBPerSecond: fp16GBPerSecond,
                bandwidthRatioToFP16: gbPerSecond / max(fp16GBPerSecond, .leastNonzeroMagnitude),
                msPerQueryToken: msPerQueryToken,
                fp16MsPerQueryToken: fp16MsPerQueryToken
            )
        )
    }
    return rows
}

@main
struct TurboQuantNativeVxBenchmarkCLI {
    static func main() throws {
        if CommandLine.arguments.contains("--help") {
            printUsage()
            return
        }

        let contexts = argumentInts("--contexts", default: [20_480, 32_768, 65_536, 131_072])
        let valueBits = argumentInts("--value-bits", default: [4, 3, 2])
        let queryLengths = argumentInts("--query-lengths", default: [1])
        let keyGroupSize = argumentInt("--key-group-size", default: 64)
        let valueGroupSize = argumentInt("--value-group-size", default: 32)
        let iterations = argumentInt("--iterations", default: 7)
        let warmup = argumentInt("--warmup", default: 2)
        let cooldownMilliseconds = argumentInt("--cooldown-ms", default: 25)
        let queryHeads = argumentInt("--query-heads", default: 16)
        let kvHeads = argumentInt("--kv-heads", default: 4)
        let headDimension = argumentInt("--head-dim", default: 256)
        let seed = argumentUInt64("--seed", default: 54_512_026_0602)
        let outputPath = argumentString("--output")

        guard Device.defaultDevice().deviceType == .gpu else {
            throw NSError(
                domain: "TurboQuantNativeVxBenchmark",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "GPU default device is required"]
            )
        }

        var allRows = [BenchmarkRow]()
        print("ctx       config       qL   median ms   ms/qtok   fp16 ms/qtok   GB/s      fp16 GB/s   BWx     fp16x     cosine")
        print("-------   ----------   --   ---------   -------   ------------   -------   ---------   -----   -------   -------")
        for context in contexts {
            for queryLength in queryLengths {
                let rows = try runContext(
                    context: context,
                    queryLength: queryLength,
                    valueBitsList: valueBits,
                    keyGroupSize: keyGroupSize,
                    valueGroupSize: valueGroupSize,
                    iterations: iterations,
                    warmup: warmup,
                    queryHeads: queryHeads,
                    kvHeads: kvHeads,
                    headDimension: headDimension,
                    seed: seed,
                    cooldownMilliseconds: cooldownMilliseconds
                )
                for row in rows {
                    allRows.append(row)
                    let cosine = row.cosineVsK8V4.map { String(format: "%.5f", $0) } ?? "--"
                    print(
                        [
                            String(format: "%7d", row.context),
                            pad(row.label, 10),
                            String(format: "%2d", row.queryLength),
                            String(format: "%9.3f", row.medianSeconds * 1_000),
                            String(format: "%7.3f", row.msPerQueryToken),
                            String(format: "%12.3f", row.fp16MsPerQueryToken),
                            String(format: "%7.1f", row.gbPerSecond),
                            String(format: "%9.1f", row.fp16GBPerSecond),
                            String(format: "%5.2f", row.bandwidthRatioToFP16),
                            String(format: "%7.3f", row.speedRatioToFP16),
                            pad(cosine, 7),
                        ].joined(separator: "   ")
                    )
                }
                Memory.clearCache()
            }
        }

        if let outputPath {
            let report = BenchmarkReport(
                generatedAt: ISO8601DateFormatter().string(from: Date()),
                cooldownMilliseconds: cooldownMilliseconds,
                rows: allRows
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(report)
            try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
            print("wrote report: \(outputPath)")
        }
    }
}
