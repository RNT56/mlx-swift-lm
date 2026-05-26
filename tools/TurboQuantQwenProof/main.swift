import Darwin
import Foundation
import MLX
import MLXLMCommon

struct QwenProofProfileCoverage: Codable {
    var profileID: String
    var architecture: String?
    var modelTypes: [String]
    var safeContextLength: Int?
    var preferredScheme: String
    var precisionCandidates: [String]
}

struct QwenProofCase: Codable {
    var id: String
    var profileID: String
    var scheme: String
    var dtype: String
    var headDimension: Int
    var kvHeads: Int
    var queryHeads: Int
    var contextLength: Int
    var queryLength: Int
}

struct QwenProofQuality: Codable {
    var top1MatchRate: Double
    var klDivergenceMean: Double
    var maxAbsErrorP95: Double
    var cosineSimilarityMean: Double
    var noNaNOrInf: Bool
    var passed: Bool
}

struct QwenProofThroughput: Codable {
    var compressedSeconds: Double
    var plainSeconds: Double
    var compressedTokensPerSecond: Double
    var plainTokensPerSecond: Double
    var speedRatioToPlain: Double
    var minExtendedTokensPerSecond: Double
    var passedParityGate: Bool
    var passedProductionGate: Bool
}

struct QwenProofMemory: Codable {
    var compressedKeyBytes: Int
    var compressedValueBytes: Int
    var compressedBytesPerToken: Double
    var plainKVBytes: Int
    var compressionRatioToPlain: Double
}

struct QwenProofResult: Codable {
    var id: String
    var status: String
    var benchmarkCase: QwenProofCase
    var selectedAttentionPath: String?
    var precisionStatus: String
    var quality: QwenProofQuality?
    var throughput: QwenProofThroughput?
    var memory: QwenProofMemory?
    var fallbackReason: String?
    var error: String?
}

struct QwenProofSummary: Codable {
    var ok: Int
    var skipped: Int
    var failed: Int
    var strictPassed: Bool
}

struct QwenProofReport: Codable {
    var schemaVersion: Int
    var generatedAt: String
    var iterations: Int
    var dtype: String
    var speedParityRatio: Double
    var minExtendedTokensPerSecond: Double
    var shortContextPlainKVThreshold: Int
    var device: TurboQuantDeviceCapabilities
    var supportsMetalCodec: Bool
    var supportsMetalAttention: Bool
    var profileCoverage: [QwenProofProfileCoverage]
    var results: [QwenProofResult]
    var summary: QwenProofSummary
}

enum QwenProofAttentionPath: String, CaseIterable {
    case auto
    case twoStage
    case fused

    init?(normalizing value: String) {
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "auto":
            self = .auto
        case "two-stage", "twostage", "two-stage-compressed", "twostagecompressed":
            self = .twoStage
        case "fused", "online-fused", "tiled-online-fused", "onlinefused", "tiledonlinefused":
            self = .fused
        default:
            return nil
        }
    }

    var idSuffix: String {
        switch self {
        case .auto:
            return ""
        case .twoStage:
            return "_pathTwoStage"
        case .fused:
            return "_pathFused"
        }
    }

    func preferOnlineFused(default defaultValue: Bool) -> Bool {
        switch self {
        case .auto:
            return defaultValue
        case .twoStage:
            return false
        case .fused:
            return true
        }
    }
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

func argumentValue(_ name: String, default defaultValue: Double) -> Double {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1),
        let value = Double(arguments[index + 1])
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
    let values = arguments[index + 1].split(separator: ",").map { String($0) }
    return values.isEmpty ? defaultValue : values
}

func argumentSchemes(_ name: String, default defaultValue: [TurboQuantScheme]) -> [TurboQuantScheme]
{
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    let values = arguments[index + 1].split(separator: ",").compactMap {
        TurboQuantScheme(normalizing: String($0))
    }
    return values.isEmpty ? defaultValue : values
}

func argumentAttentionPaths(
    _ name: String,
    default defaultValue: [QwenProofAttentionPath]
) -> [QwenProofAttentionPath] {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    let values = arguments[index + 1].split(separator: ",").compactMap {
        QwenProofAttentionPath(normalizing: String($0))
    }
    return values.isEmpty ? defaultValue : values
}

func argumentDType(_ name: String, default defaultValue: DType) -> DType {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    switch arguments[index + 1].lowercased() {
    case "float16", "fp16":
        return .float16
    case "bfloat16", "bf16":
        return .bfloat16
    case "float32", "fp32":
        return .float32
    default:
        return defaultValue
    }
}

func dtypeName(_ dtype: DType) -> String {
    switch dtype {
    case .float16:
        return "float16"
    case .bfloat16:
        return "bfloat16"
    case .float32:
        return "float32"
    default:
        return String(describing: dtype)
    }
}

func hasFlag(_ name: String) -> Bool {
    CommandLine.arguments.contains(name)
}

func deterministicValues(count: Int, scale: Double, phase: Double) -> [Float] {
    (0 ..< count).map { index in
        let position = Double(index)
        return Float(0.31 * sin(position * scale + phase) + 0.17 * cos(position * 0.037))
    }
}

func timed(iterations: Int, _ body: () throws -> MLXArray) throws -> (Double, MLXArray) {
    let warmup = try body()
    eval(warmup)
    var last = warmup
    let start = Date.timeIntervalSinceReferenceDate
    for _ in 0 ..< max(1, iterations) {
        last = try body()
        eval(last)
    }
    return ((Date.timeIntervalSinceReferenceDate - start) / Double(max(1, iterations)), last)
}

func argmax(_ values: ArraySlice<Float>) -> Int {
    guard var bestValue = values.first else { return -1 }
    var bestOffset = 0
    for (offset, value) in values.enumerated().dropFirst() where value > bestValue {
        bestValue = value
        bestOffset = offset
    }
    return bestOffset
}

func logSoftmax(_ values: ArraySlice<Float>) -> [Double] {
    let doubleValues = values.map(Double.init)
    guard let maxValue = doubleValues.max(), maxValue.isFinite else {
        return Array(repeating: -Double.greatestFiniteMagnitude, count: values.count)
    }
    let denominator =
        maxValue
        + log(
            max(
                doubleValues.reduce(0.0) { $0 + exp($1 - maxValue) },
                Double.leastNonzeroMagnitude
            )
        )
    return doubleValues.map { $0 - denominator }
}

func percentile(_ values: [Double], percentile: Double) -> Double {
    guard !values.isEmpty else { return Double.greatestFiniteMagnitude }
    let sorted = values.sorted()
    let index = min(
        sorted.count - 1,
        max(0, Int(ceil(min(max(percentile, 0), 1) * Double(sorted.count))) - 1)
    )
    return sorted[index]
}

func qualityGate(
    candidate: MLXArray,
    reference: MLXArray,
    scheme: TurboQuantScheme
) -> QwenProofQuality {
    eval(candidate, reference)
    let candidateValues = candidate.asArray(Float.self)
    let referenceValues = reference.asArray(Float.self)
    guard candidateValues.count == referenceValues.count,
        let rowWidth = candidate.shape.last,
        rowWidth > 0,
        !candidateValues.isEmpty
    else {
        return QwenProofQuality(
            top1MatchRate: 0,
            klDivergenceMean: Double.greatestFiniteMagnitude,
            maxAbsErrorP95: Double.greatestFiniteMagnitude,
            cosineSimilarityMean: 0,
            noNaNOrInf: false,
            passed: false
        )
    }

    let rowCount = candidateValues.count / rowWidth
    var top1Matches = 0
    var klTotal = 0.0
    var cosineTotal = 0.0
    var maxErrors = [Double]()
    maxErrors.reserveCapacity(rowCount)
    for row in 0 ..< rowCount {
        let start = row * rowWidth
        let end = start + rowWidth
        let candidateRow = candidateValues[start ..< end]
        let referenceRow = referenceValues[start ..< end]
        if argmax(candidateRow) == argmax(referenceRow) {
            top1Matches += 1
        }

        let candidateLogProb = logSoftmax(candidateRow)
        let referenceLogProb = logSoftmax(referenceRow)
        klTotal += zip(referenceLogProb, candidateLogProb).reduce(0.0) { partial, pair in
            partial + exp(pair.0) * (pair.0 - pair.1)
        }

        var dot = 0.0
        var candidateNorm = 0.0
        var referenceNorm = 0.0
        var maxError = 0.0
        for (candidateValue, referenceValue) in zip(candidateRow, referenceRow) {
            let c = Double(candidateValue)
            let r = Double(referenceValue)
            dot += c * r
            candidateNorm += c * c
            referenceNorm += r * r
            maxError = max(maxError, abs(c - r))
        }
        let denominator = sqrt(candidateNorm) * sqrt(referenceNorm)
        cosineTotal += denominator > 0 ? dot / denominator : 0
        maxErrors.append(maxError)
    }

    let top1 = Double(top1Matches) / Double(max(1, rowCount))
    let kl = klTotal / Double(max(1, rowCount))
    let cosine = cosineTotal / Double(max(1, rowCount))
    let p95 = percentile(maxErrors, percentile: 0.95)
    let finite =
        candidateValues.allSatisfy(\.isFinite) && referenceValues.allSatisfy(\.isFinite)
    let passed: Bool
    switch scheme {
    case .turbo8:
        passed = finite && kl <= 0.02 && p95 <= 0.20 && cosine >= 0.995
    case .turbo4v2, .turbo3_5:
        passed = finite && kl <= 0.05 && p95 <= 0.50 && cosine >= 0.990
    case .disabled, .turbo3, .turbo2_5:
        passed = false
    }
    return QwenProofQuality(
        top1MatchRate: top1,
        klDivergenceMean: kl,
        maxAbsErrorP95: p95,
        cosineSimilarityMean: cosine,
        noNaNOrInf: finite,
        passed: passed
    )
}

func compressedAttentionPathName(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    preferOnlineFused: Bool,
    availability: TurboQuantKernelAvailability
) -> String {
    do {
        let decision = try turboQuantAttentionDecision(
            request: TurboQuantAttentionRequest(
                queryShape: queries.shape,
                keyLayout: keyCode.layout,
                valueLayout: valueCode.layout,
                queryDType: queries.dtype,
                outputDType: queries.dtype,
                maskKind: .causal,
                hasSinks: false,
                preferOnlineFused: preferOnlineFused,
                memoryBudgetBytes: nil,
                fallbackState: .none
            ),
            capabilities: availability.attentionCapabilities
        )
        return decision.selectedPath.rawValue
    } catch {
        return "unavailable"
    }
}

func runCase(
    profile: TurboQuantProfile,
    scheme: TurboQuantScheme,
    contextLength: Int,
    queryLength: Int,
    dtype: DType,
    iterations: Int,
    speedParityRatio: Double,
    minExtendedTokensPerSecond: Double,
    shortContextPlainKVThreshold: Int,
    availability: TurboQuantKernelAvailability,
    attentionPath: QwenProofAttentionPath = .auto
) -> QwenProofResult {
    let caseID =
        "\(profile.id)_\(scheme.rawValue)_ctx\(contextLength)_q\(queryLength)\(attentionPath.idSuffix)"
    guard let precisionProfile = profile.applyingPrecisionCandidate(scheme) else {
        let benchmarkCase = QwenProofCase(
            id: caseID,
            profileID: profile.id,
            scheme: scheme.rawValue,
            dtype: dtypeName(dtype),
            headDimension: 256,
            kvHeads: 4,
            queryHeads: 16,
            contextLength: contextLength,
            queryLength: queryLength
        )
        return QwenProofResult(
            id: benchmarkCase.id,
            status: "skipped",
            benchmarkCase: benchmarkCase,
            selectedAttentionPath: nil,
            precisionStatus: "unsupported",
            quality: nil,
            throughput: nil,
            memory: nil,
            fallbackReason: "precision is not valid for this Qwen profile",
            error: nil
        )
    }

    let candidate = precisionProfile.precisionCandidates.first { $0.scheme == scheme }
    let kvHeads = 4
    let queryHeads = 16
    let headDimension = 256
    let benchmarkCase = QwenProofCase(
        id: caseID,
        profileID: profile.id,
        scheme: scheme.rawValue,
        dtype: dtypeName(dtype),
        headDimension: headDimension,
        kvHeads: kvHeads,
        queryHeads: queryHeads,
        contextLength: contextLength,
        queryLength: queryLength
    )

    guard availability.supportsMetalPolarQJLAttention else {
        return QwenProofResult(
            id: benchmarkCase.id,
            status: "skipped",
            benchmarkCase: benchmarkCase,
            selectedAttentionPath: nil,
            precisionStatus: candidate?.status.rawValue ?? "unknown",
            quality: nil,
            throughput: nil,
            memory: nil,
            fallbackReason:
                "Metal TurboQuant attention unavailable: \(availability.selfTestFailureReason ?? "self-test unavailable")",
            error: nil
        )
    }

    do {
        let keyCount = kvHeads * contextLength * headDimension
        let queryCount = queryHeads * queryLength * headDimension
        let keys = MLXArray(
            deterministicValues(count: keyCount, scale: 0.0037, phase: 0.11),
            [1, kvHeads, contextLength, headDimension]
        ).asType(dtype)
        let values = MLXArray(
            deterministicValues(count: keyCount, scale: 0.0041, phase: 0.29),
            [1, kvHeads, contextLength, headDimension]
        ).asType(dtype)
        let queries = MLXArray(
            deterministicValues(count: queryCount, scale: 0.0061, phase: 0.43),
            [1, queryHeads, queryLength, headDimension]
        ).asType(dtype)
        let scale = 1 / sqrt(Float(headDimension))
        let cache = TurboQuantKVCache(
            preset: precisionProfile.recommendedScheme.preset,
            groupSize: precisionProfile.groupSize,
            backend: precisionProfile.backend,
            optimizationPolicy: precisionProfile.optimizationPolicy,
            fallbackPolicy: precisionProfile.turboQuant.fallbackPolicy,
            valueBits: precisionProfile.valueBits
        )
        guard
            cache.supportsCompressedAttention(
                queries: queries,
                keys: keys,
                values: values,
                mask: .causal
            )
        else {
            return QwenProofResult(
                id: benchmarkCase.id,
                status: "skipped",
                benchmarkCase: benchmarkCase,
                selectedAttentionPath: cache.attentionDiagnostics.activeAttentionPath.rawValue,
                precisionStatus: candidate?.status.rawValue ?? "unknown",
                quality: nil,
                throughput: nil,
                memory: nil,
                fallbackReason: cache.attentionDiagnostics.lastUnsupportedShape,
                error: nil
            )
        }

        let (compressedKeys, compressedValues) = try cache.updateCompressed(
            keys: keys,
            values: values
        )
        let preferOnline = attentionPath.preferOnlineFused(
            default: cache.prefersOnlineFusedAttention
        )
        let selectedAttentionPath = compressedAttentionPathName(
            queries: queries,
            keyCode: compressedKeys,
            valueCode: compressedValues,
            preferOnlineFused: preferOnline,
            availability: availability
        )
        let (plainSeconds, plainOutput) = try timed(iterations: iterations) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .causal
            )
        }
        let (compressedSeconds, compressedOutput) = try timed(iterations: iterations) {
            try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                scale: scale,
                mask: .causal,
                preferOnlineFused: preferOnline,
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile
            )
        }
        let quality = qualityGate(
            candidate: compressedOutput,
            reference: plainOutput,
            scheme: scheme
        )
        let compressedTPS =
            Double(queryLength) / max(compressedSeconds, Double.leastNonzeroMagnitude)
        let plainTPS = Double(queryLength) / max(plainSeconds, Double.leastNonzeroMagnitude)
        let speedRatio = compressedTPS / max(plainTPS, Double.leastNonzeroMagnitude)
        let usesPlainProductionRoute = contextLength <= shortContextPlainKVThreshold
        let passedParityGate = usesPlainProductionRoute || speedRatio >= speedParityRatio
        let passedProductionGate =
            usesPlainProductionRoute || compressedTPS >= minExtendedTokensPerSecond
        let throughput = QwenProofThroughput(
            compressedSeconds: compressedSeconds,
            plainSeconds: plainSeconds,
            compressedTokensPerSecond: compressedTPS,
            plainTokensPerSecond: plainTPS,
            speedRatioToPlain: speedRatio,
            minExtendedTokensPerSecond: minExtendedTokensPerSecond,
            passedParityGate: passedParityGate,
            passedProductionGate: passedProductionGate
        )
        let compressedBytes = compressedKeys.storageByteCount + compressedValues.storageByteCount
        let plainBytes = keys.nbytes + values.nbytes
        let memory = QwenProofMemory(
            compressedKeyBytes: compressedKeys.storageByteCount,
            compressedValueBytes: compressedValues.storageByteCount,
            compressedBytesPerToken: Double(compressedBytes) / Double(max(1, contextLength)),
            plainKVBytes: plainBytes,
            compressionRatioToPlain: Double(compressedBytes) / Double(max(1, plainBytes))
        )
        let fallbackReason: String?
        if usesPlainProductionRoute {
            fallbackReason =
                "speed parity gate waived because production routing uses plain KV at or below \(shortContextPlainKVThreshold) tokens"
        } else if attentionPath != .auto {
            fallbackReason =
                "attention path forced to \(attentionPath.rawValue) for path comparison; selected \(selectedAttentionPath)"
        } else {
            fallbackReason =
                cache.attentionDiagnostics.fallbackReason
                ?? cache.attentionDiagnostics.lastUnsupportedShape
        }
        let status = quality.passed && throughput.passedProductionGate ? "ok" : "failed"
        return QwenProofResult(
            id: benchmarkCase.id,
            status: status,
            benchmarkCase: benchmarkCase,
            selectedAttentionPath: selectedAttentionPath,
            precisionStatus: candidate?.status.rawValue ?? "unknown",
            quality: quality,
            throughput: throughput,
            memory: memory,
            fallbackReason: fallbackReason,
            error: nil
        )
    } catch {
        return QwenProofResult(
            id: benchmarkCase.id,
            status: "failed",
            benchmarkCase: benchmarkCase,
            selectedAttentionPath: nil,
            precisionStatus: candidate?.status.rawValue ?? "unknown",
            quality: nil,
            throughput: nil,
            memory: nil,
            fallbackReason: nil,
            error: String(describing: error)
        )
    }
}

let iterations = argumentValue("--iterations", default: 5)
let releaseMatrix = hasFlag("--release-matrix")
let pathCompare = hasFlag("--path-compare")
let strict = hasFlag("--strict")
let proofDType = argumentDType("--dtype", default: .float16)
let speedParityRatio = argumentValue("--speed-parity-ratio", default: 1.0)
let minExtendedTokensPerSecond = argumentValue("--min-extended-tokens-per-second", default: 20.0)
let shortContextPlainKVThreshold = argumentValue("--plain-route-threshold", default: 4096)
let cooldownMilliseconds = argumentValue("--cooldown-ms", default: 0)
let contexts = argumentValues(
    "--contexts",
    default: releaseMatrix ? [8192, 16384] : [8192]
)
let queryLengths = argumentValues(
    "--query-lengths",
    default: releaseMatrix ? [1] : [1, 4]
)
let schemes = argumentSchemes(
    "--schemes",
    default: pathCompare ? [.turbo8] : [.turbo8, .turbo4v2, .turbo3_5]
)
let attentionPaths = argumentAttentionPaths(
    "--attention-paths",
    default: pathCompare ? [.twoStage, .fused] : [.auto]
)
let defaultProfileIDs = [
    "qwen3.5-0.8b",
    "qwen3.5-2b",
    "qwen3.6-27b",
    "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b",
]
let requestedProfileIDs = argumentStrings(
    "--profiles",
    default: pathCompare ? ["qwen3.5-2b"] : defaultProfileIDs
)
let availability = TurboQuantKernelAvailability.current
let qwenProfiles = TurboQuantProfileRegistry.bundled.profiles
    .filter(\.isQwen35Or36Family)
    .sorted { $0.id < $1.id }
let profileCoverage = qwenProfiles.map { profile in
    QwenProofProfileCoverage(
        profileID: profile.id,
        architecture: profile.architecture,
        modelTypes: profile.modelTypes,
        safeContextLength: profile.safeContextLength,
        preferredScheme: profile.recommendedScheme.rawValue,
        precisionCandidates: profile.precisionCandidates.map {
            "\($0.scheme.rawValue):\($0.status.rawValue)"
        }
    )
}

var results = [QwenProofResult]()
let requestedProfileIDSet = Set(requestedProfileIDs)
let representativeProfiles = qwenProfiles.filter { requestedProfileIDSet.contains($0.id) }
for profile in representativeProfiles {
    for scheme in schemes {
        for context in contexts {
            guard context <= (profile.safeContextLength ?? context) else { continue }
            for queryLength in queryLengths {
                for attentionPath in attentionPaths {
                    results.append(
                        runCase(
                            profile: profile,
                            scheme: scheme,
                            contextLength: context,
                            queryLength: queryLength,
                            dtype: proofDType,
                            iterations: iterations,
                            speedParityRatio: speedParityRatio,
                            minExtendedTokensPerSecond: minExtendedTokensPerSecond,
                            shortContextPlainKVThreshold: shortContextPlainKVThreshold,
                            availability: availability,
                            attentionPath: attentionPath
                        )
                    )
                    Memory.clearCache()
                    if cooldownMilliseconds > 0 {
                        usleep(useconds_t(cooldownMilliseconds * 1000))
                    }
                }
            }
        }
    }
}

let okCount = results.filter { $0.status == "ok" }.count
let skippedCount = results.filter { $0.status == "skipped" }.count
let failedCount = results.filter { $0.status == "failed" }.count
let strictPassed = failedCount == 0 && skippedCount == 0 && !results.isEmpty
let report = QwenProofReport(
    schemaVersion: 1,
    generatedAt: ISO8601DateFormatter().string(from: Date()),
    iterations: iterations,
    dtype: dtypeName(proofDType),
    speedParityRatio: speedParityRatio,
    minExtendedTokensPerSecond: minExtendedTokensPerSecond,
    shortContextPlainKVThreshold: shortContextPlainKVThreshold,
    device: TurboQuantDeviceCapabilities.current,
    supportsMetalCodec: availability.supportsMetalPolarQJLCodec,
    supportsMetalAttention: availability.supportsMetalPolarQJLAttention,
    profileCoverage: profileCoverage,
    results: results,
    summary: QwenProofSummary(
        ok: okCount,
        skipped: skippedCount,
        failed: failedCount,
        strictPassed: strictPassed
    )
)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(report)
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))

if strict && !strictPassed {
    FileHandle.standardError.write(
        Data("TurboQuantQwenProof failed strict gates.\n".utf8)
    )
    exit(1)
}
