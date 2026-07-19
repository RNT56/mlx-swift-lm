import Darwin
import Foundation
import MLX
import MLXLMCommon

struct QwenProofProfileCoverage: Codable {
    var profileID: String
    var architecture: String?
    var modelTypes: [String]
    var supportedContextLengths: [Int]
    var safeContextLength: Int?
    var preferredScheme: String
    var precisionCandidates: [String]
}

enum QwenProofGateScope: String, Codable {
    case production
    case largeContextExperiment
}

enum QwenProofCodec: String, Codable {
    case polarQJL = "polar_qjl"
    case affineInt4 = "affine_int4"
    case raw
}

struct QwenProofCase: Codable {
    var id: String
    var profileID: String
    var scheme: String
    var codec: String = QwenProofCodec.polarQJL.rawValue
    var requestedRuntimeMode: String = "auto"
    var resolvedRuntimeMode: String = "capacityTurboQuant"
    var keyPrecision: String = "turbo8"
    var valuePrecision: String = "turbo4v2"
    var backend: String = TurboQuantBackend.metalPolarQJL.rawValue
    var sparseVEnabled: Bool = false
    var sparseVSelectionMode: String = "off"
    var sparseVThreshold: Float? = nil
    var sparseVTopK: Int? = nil
    var sparseVCumulativeMassPercent: Double? = nil
    var sparseVMaxTopK: Int? = nil
    var sparseVSkipRatio: Double = 0
    var boundaryProtectedLayerCount: Int = 0
    var boundaryProtectionReason: String? = nil
    var dtype: String
    var gateScope: QwenProofGateScope
    var headDimension: Int
    var kvHeads: Int
    var queryHeads: Int
    var contextLength: Int
    var reservedCapacityLength: Int
    var blockParallelTokenBlockSize: Int?
    var recommendedBlockParallelTokenBlockSize: Int?
    var effectiveBlockParallelTokenBlockSize: Int?
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
    var compressedSecondsP50: Double
    var compressedSecondsP95: Double
    var plainSeconds: Double
    var plainSecondsP50: Double
    var plainSecondsP95: Double
    var compressedTokensPerSecond: Double
    var compressedTokensPerSecondP50: Double
    var compressedTokensPerSecondP95: Double
    var plainTokensPerSecond: Double
    var plainTokensPerSecondP50: Double
    var plainTokensPerSecondP95: Double
    var speedRatioToPlain: Double
    var speedRatioToPlainP50: Double
    var speedRatioToPlainP95: Double
    var productionTokensPerSecondP50: Double
    var productionTokensPerSecondP95: Double
    var productionThroughputSource: String
    var minExtendedTokensPerSecond: Double
    var passedParityGate: Bool
    var passedProductionGate: Bool
}

struct QwenProofMemory: Codable {
    var compressedKeyBytes: Int
    var compressedValueBytes: Int
    var scaleBiasBytes: Int? = nil
    var compressedBytesPerToken: Double
    var compressedBytesPerLogicalToken: Double
    var compressedBytesPerReservedToken: Double
    var plainKVBytes: Int
    var plainKVBytesAtReservedCapacity: Int
    var compressionRatioToPlain: Double
    var compressionRatioToPlainReservedCapacity: Double
    var reservedCapacityMultiplier: Double
}

struct QwenProofResult: Codable {
    var id: String
    var status: String
    var benchmarkCase: QwenProofCase
    var gateScope: QwenProofGateScope
    var strictGateRequired: Bool
    var certificationStatus: String
    var selectedAttentionPath: String?
    var productionKVRoute: String?
    var requestedRuntimeMode: String? = nil
    var resolvedRuntimeMode: String? = nil
    var keyPrecision: String? = nil
    var valuePrecision: String? = nil
    var codec: String? = nil
    var backend: String? = nil
    var sparseVEnabled: Bool = false
    var sparseVSelectionMode: String = "off"
    var sparseVThreshold: Float? = nil
    var sparseVTopK: Int? = nil
    var sparseVCumulativeMassPercent: Double? = nil
    var sparseVMaxTopK: Int? = nil
    var sparseVSkipRatio: Double = 0
    var sparseVSkippedValueTokens: Int? = nil
    var sparseVTotalValueTokens: Int? = nil
    var sparseVRetainedMass: Double? = nil
    var sparseVDenseCosineSimilarity: Double? = nil
    var sparseVDenseMaxAbsErrorP95: Double? = nil
    var sparseVHeadDiagnostics: [QwenProofSparseHeadDiagnostics]? = nil
    var lowerVAndSparseV: QwenProofLowerVAndSparseVReport? = nil
    var nativeAttentionDiagnostics: [Int]? = nil
    var boundaryProtectedLayerCount: Int = 0
    var boundaryProtectionReason: String? = nil
    var decodedActiveKVBytes: Int? = nil
    var precisionStatus: String
    var quality: QwenProofQuality?
    var throughput: QwenProofThroughput?
    var memory: QwenProofMemory?
    var fallbackReason: String?
    var error: String?
}

struct QwenProofLowerVAndSparseVReport: Codable {
    var referenceConfig: String
    var candidateConfig: String
    var valueBits: Int?
    var valueBitPolicy: String?
    var sparseVMode: String?
    var sparseVTopK: Int?
    var sparseVCumulativeMass: Double?
    var sparseVMaxTopK: Int?
    var selectionLatencyMS: Double?
    var qkMS: Double?
    var softmaxMS: Double?
    var maskOrCompactionMS: Double?
    var avLatencyMS: Double?
    var totalMS: Double?
    var denseK8V4ReferenceMS: Double?
    var skippedValueTokens: Int?
    var consideredValueTokens: Int?
    var retainedMass: Double?
    var skipRatio: Double?
    var fallbackCount: Int
    var fallbackReason: String?
    var actualMixedBitsPerValue: Double?
    var layerIndex: Int?
    var headIndex: Int?
}

struct QwenProofSparseHeadDiagnostics: Codable {
    var layerIndex: Int
    var headIndex: Int
    var skippedValueTokens: Int
    var totalValueTokens: Int
    var retainedMass: Double?
    var maxOutputError: Double?
    var cosineSimilarity: Double?
}

struct QwenProofSummary: Codable {
    var ok: Int
    var skipped: Int
    var failed: Int
    var productionOk: Int
    var productionSkipped: Int
    var productionFailed: Int
    var largeContextExperimentOk: Int
    var largeContextExperimentSkipped: Int
    var largeContextExperimentFailed: Int
    var strictPassed: Bool
}

struct QwenProofReport: Codable {
    var schemaVersion: Int
    var generatedAt: String
    var iterations: Int
    var warmupIterations: Int
    var dtype: String
    var speedParityRatio: Double
    var minExtendedTokensPerSecond: Double
    var shortContextPlainKVThreshold: Int
    var requestedReservedCapacityLength: Int?
    var requestedBlockParallelTokenBlockSize: Int?
    var blockParallelTokenBlockPolicy: String
    var requestedRuntimeMode: String
    var sparseValuePolicy: String
    var sparseValueThreshold: Float
    var sparseValueSelectionMode: String
    var sparseValueTopK: Int?
    var sparseValueCumulativeMassPercent: Double?
    var sparseValueMaxTopK: Int?
    var codec: String
    var productionContexts: [Int]
    var largeContextExperimentContexts: [Int]
    var requireLargeContextExperimentGates: Bool
    var device: TurboQuantDeviceCapabilities
    var supportsMetalCodec: Bool
    var supportsMetalAttention: Bool
    var profileCoverage: [QwenProofProfileCoverage]
    var results: [QwenProofResult]
    var summary: QwenProofSummary
    /// Provenance guardrails: this proof times synthetic (deterministic sinusoid)
    /// attention shapes with no checkpoint loaded, so it is not real-model and not
    /// production-certified. Emitted so consumers cannot mistake it for real-model
    /// evidence. Defaulted so the synthesized memberwise init is unchanged.
    var synthetic = true
    var realModel = false
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

enum QwenProofSparseSelectionMode: String, Codable, Equatable {
    case off
    case threshold
    case topK = "top_k"
    case cumulativeMass = "cumulative_mass"
    case hybridCumulativeMassTopK = "hybrid_cumulative_mass_top_k"

    init?(normalizing value: String) {
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "off", "false", "none", "0":
            self = .off
        case "threshold", "weight-threshold", "softmax-threshold":
            self = .threshold
        case "top-k", "topk":
            self = .topK
        case "cumulative", "cumulative-mass", "mass":
            self = .cumulativeMass
        case "hybrid", "hybrid-cumulative", "hybrid-cumulative-mass-top-k":
            self = .hybridCumulativeMassTopK
        default:
            return nil
        }
    }

    var nativeMode: TurboQuantSparseValueNativeSelectionMode {
        switch self {
        case .off:
            return .off
        case .threshold:
            return .threshold
        case .topK:
            return .topK
        case .cumulativeMass:
            return .cumulativeMass
        case .hybridCumulativeMassTopK:
            return .hybridCumulativeMassTopK
        }
    }

    var reportName: String {
        switch self {
        case .off:
            return "off"
        case .threshold:
            return "threshold"
        case .topK:
            return "topK"
        case .cumulativeMass:
            return "cumulativeMass"
        case .hybridCumulativeMassTopK:
            return "hybridCumulativeMassTopK"
        }
    }
}

struct QwenProofSparseSelectionConfig: Codable, Equatable {
    var mode: QwenProofSparseSelectionMode
    var threshold: Float?
    var topK: Int?
    /// Percent mass to retain. For example, 99.5 means retain 99.5% of softmax mass.
    var cumulativeMassPercent: Double?
    var maxTopK: Int?

    static let off = QwenProofSparseSelectionConfig(mode: .off)

    var isEnabled: Bool {
        mode != .off
    }

    var idSuffix: String {
        switch mode {
        case .off:
            return ""
        case .threshold:
            return "_sparseThreshold"
        case .topK:
            return "_sparseTopK\(topK ?? 0)"
        case .cumulativeMass:
            let mass = cumulativeMassPercent.map { String(format: "%.3g", $0) } ?? "0"
            return "_sparseMass\(mass)"
        case .hybridCumulativeMassTopK:
            let mass = cumulativeMassPercent.map { String(format: "%.3g", $0) } ?? "0"
            return "_sparseHybrid\(mass)_topK\(maxTopK ?? 0)"
        }
    }

    var nativeTopK: Int {
        switch mode {
        case .topK:
            return max(0, topK ?? 0)
        case .hybridCumulativeMassTopK:
            return max(0, maxTopK ?? topK ?? 0)
        case .off, .threshold, .cumulativeMass:
            return 0
        }
    }

    var nativeCumulativeMass: Float {
        switch mode {
        case .cumulativeMass, .hybridCumulativeMassTopK:
            return Float(min(1, max(0, (cumulativeMassPercent ?? 0) / 100)))
        case .off, .threshold, .topK:
            return 0
        }
    }

    var nativeMaxTopK: Int {
        switch mode {
        case .hybridCumulativeMassTopK:
            return max(0, maxTopK ?? topK ?? 0)
        case .off, .threshold, .topK, .cumulativeMass:
            return 0
        }
    }

    func enabledFor(runtimeMode: TurboQuantRuntimeMode, queryLength: Int) -> Bool {
        guard isEnabled, queryLength == 1 else { return false }
        return runtimeMode == .capacityTurboQuant
    }

    func resolvedThreshold(
        policy: TurboQuantSparseValuePolicy,
        runtimeMode: TurboQuantRuntimeMode,
        contextLength: Int
    ) -> Float? {
        guard mode == .threshold else { return nil }
        return threshold
            ?? policy.resolvedThreshold(runtimeMode: runtimeMode, contextLength: contextLength)
    }

    func nativeOptions(
        scale: Float,
        causal: Bool,
        threshold: Float?,
        diagnostics: Bool,
        backendVersion: Int?
    ) -> TurboQuantNativeAttentionOptions {
        TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: causal,
            sparseVThreshold: threshold ?? 0,
            sparseVSelectionMode: mode.nativeMode,
            sparseVTopK: nativeTopK,
            sparseVCumulativeMass: nativeCumulativeMass,
            sparseVMaxTopK: nativeMaxTopK,
            diagnostics: diagnostics,
            backendVersion: backendVersion ?? TurboQuantNativeAttentionOptions.backendVersion
        )
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

func optionalArgumentString(_ name: String) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return nil
    }
    return arguments[index + 1]
}

func optionalArgumentValue(_ name: String) -> Double? {
    guard let raw = optionalArgumentString(name) else { return nil }
    return Double(raw)
}

func argumentValues(_ name: String, default defaultValue: [Int]) -> [Int] {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    let values = arguments[index + 1].split(separator: ",").compactMap { Int($0) }
    return values.isEmpty ? defaultValue : values
}

func optionalPositiveArgumentValue(_ name: String) -> Int? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1),
        let value = Int(arguments[index + 1]),
        value > 0
    else {
        return nil
    }
    return value
}

func normalizedPercentArgument(_ name: String) -> Double? {
    guard let value = optionalArgumentValue(name), value.isFinite, value > 0 else {
        return nil
    }
    return min(100, value <= 1 ? value * 100 : value)
}

func uniqueSorted(_ values: [Int]) -> [Int] {
    Array(Set(values)).sorted()
}

func roundedReservedCapacityLength(_ requiredLength: Int) -> Int {
    let compressedCapacityStep = 256
    return ((compressedCapacityStep + requiredLength - 1) / compressedCapacityStep)
        * compressedCapacityStep
}

func roundedBlockParallelTokenBlockSize(_ requested: Int, headDimension: Int) -> Int {
    let target = max(1, min(512, max(requested, headDimension)))
    var width = 1
    while width < target {
        width <<= 1
    }
    return width
}

func blockParallelTokenPlan(
    contextLength: Int,
    headDimension: Int,
    queryLength: Int,
    availability: TurboQuantKernelAvailability,
    requestedBlockParallelTokenBlockSize: Int?
) -> (recommended: Int?, effective: Int?) {
    let recommended = turboQuantRecommendedBlockParallelTokenBlockSize(
        logicalLength: contextLength,
        headDimension: headDimension,
        queryLength: queryLength,
        kernelProfile: availability.selectedKernelProfile
    )
    if let requestedBlockParallelTokenBlockSize {
        return (
            recommended,
            roundedBlockParallelTokenBlockSize(
                requestedBlockParallelTokenBlockSize,
                headDimension: headDimension
            )
        )
    }
    return (recommended, recommended)
}

func argumentStrings(_ name: String, default defaultValue: [String]) -> [String] {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    let values = arguments[index + 1].split(separator: ",").map { String($0) }
    return values.isEmpty ? defaultValue : values
}

func argumentCodec(_ name: String, default defaultValue: QwenProofCodec) -> QwenProofCodec {
    let value = argumentStrings(name, default: [defaultValue.rawValue]).first ?? defaultValue.rawValue
    switch value.lowercased().replacingOccurrences(of: "-", with: "_") {
    case "affine_int4", "affineint4":
        return .affineInt4
    case "raw":
        return .raw
    default:
        return .polarQJL
    }
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

func argumentRuntimeMode(
    _ name: String,
    default defaultValue: TurboQuantRuntimeMode
) -> TurboQuantRuntimeMode {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    switch arguments[index + 1].lowercased().replacingOccurrences(of: "_", with: "-") {
    case "auto":
        return .auto
    case "raw", "raw-preferred", "rawpreferred":
        return .rawPreferred
    case "throughput", "throughput-turboquant", "throughputturboquant":
        return .throughputTurboQuant
    case "capacity", "capacity-turboquant", "capacityturboquant":
        return .capacityTurboQuant
    default:
        return defaultValue
    }
}

func argumentSparseValuePolicy(
    _ name: String,
    threshold: Float
) -> TurboQuantSparseValuePolicy {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return .off
    }
    switch arguments[index + 1].lowercased().replacingOccurrences(of: "_", with: "-") {
    case "off", "false", "none", "0":
        return .off
    case "auto", "on", "true", "1":
        return .auto(threshold: threshold)
    case "force", "forced":
        return .force(threshold: threshold)
    default:
        return .off
    }
}

func sparseValuePolicyName(_ policy: TurboQuantSparseValuePolicy) -> String {
    switch policy {
    case .off:
        return "off"
    case .auto:
        return "auto"
    case .force:
        return "force"
    }
}

func valueBitPolicyName(valueBits: Int?) -> String? {
    guard let valueBits else { return nil }
    switch valueBits {
    case 4...:
        return "denseV4"
    case 3:
        return "calibratedV3"
    default:
        return "calibratedV2"
    }
}

func argumentSparseSelectionConfig(
    policy: TurboQuantSparseValuePolicy,
    threshold: Float
) -> QwenProofSparseSelectionConfig {
    let explicitMode = optionalArgumentString("--sparse-v-mode")
        .flatMap(QwenProofSparseSelectionMode.init(normalizing:))
    let topK = optionalPositiveArgumentValue("--sparse-v-top-k")
    let maxTopK =
        optionalPositiveArgumentValue("--sparse-v-max-top-k")
        ?? optionalPositiveArgumentValue("--sparse-v-hybrid-top-k")
    let cumulativeMass =
        normalizedPercentArgument("--sparse-v-cumulative-mass")
        ?? normalizedPercentArgument("--sparse-v-mass")
    let hybridMass =
        normalizedPercentArgument("--sparse-v-hybrid-mass")
        ?? normalizedPercentArgument("--sparse-v-cumulative-floor")

    let mode: QwenProofSparseSelectionMode
    if let explicitMode {
        mode = explicitMode
    } else if hybridMass != nil || maxTopK != nil && cumulativeMass != nil {
        mode = .hybridCumulativeMassTopK
    } else if topK != nil {
        mode = .topK
    } else if cumulativeMass != nil {
        mode = .cumulativeMass
    } else if policy != .off {
        mode = .threshold
    } else {
        mode = .off
    }

    switch mode {
    case .off:
        return .off
    case .threshold:
        return QwenProofSparseSelectionConfig(
            mode: .threshold,
            threshold: policy.threshold ?? threshold
        )
    case .topK:
        return QwenProofSparseSelectionConfig(
            mode: .topK,
            topK: topK ?? maxTopK ?? 256
        )
    case .cumulativeMass:
        return QwenProofSparseSelectionConfig(
            mode: .cumulativeMass,
            cumulativeMassPercent: cumulativeMass ?? hybridMass ?? 99.5
        )
    case .hybridCumulativeMassTopK:
        return QwenProofSparseSelectionConfig(
            mode: .hybridCumulativeMassTopK,
            topK: topK,
            cumulativeMassPercent: hybridMass ?? cumulativeMass ?? 99.5,
            maxTopK: maxTopK ?? topK ?? 256
        )
    }
}

func resolvedProofRuntimeMode(
    requestedRuntimeMode: TurboQuantRuntimeMode,
    contextLength: Int,
    shortContextPlainKVThreshold: Int
) -> TurboQuantRuntimeMode {
    switch requestedRuntimeMode {
    case .auto:
        return contextLength <= shortContextPlainKVThreshold ? .rawPreferred : .throughputTurboQuant
    case .rawPreferred, .throughputTurboQuant, .capacityTurboQuant:
        return requestedRuntimeMode
    }
}

func runtimeRouteName(_ runtimeMode: TurboQuantRuntimeMode) -> String {
    switch runtimeMode {
    case .auto:
        return "auto"
    case .rawPreferred:
        return TurboQuantRuntimeRoute.rawSDPA.rawValue
    case .throughputTurboQuant:
        return TurboQuantRuntimeRoute.throughputTurboQuantNativeSDPA.rawValue
    case .capacityTurboQuant:
        return TurboQuantRuntimeRoute.capacityTurboQuantCompressed.rawValue
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

func certificationStatus(for gateScope: QwenProofGateScope) -> String {
    switch gateScope {
    case .production:
        return "synthetic-microbench-not-production-certified"
    case .largeContextExperiment:
        return "experiment-only-not-production-certified"
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

struct QwenProofTiming {
    var averageSeconds: Double
    var p50Seconds: Double
    var p95Seconds: Double
    var output: MLXArray
}

func timed(
    iterations: Int,
    warmupIterations: Int,
    _ body: () throws -> MLXArray
) throws -> QwenProofTiming {
    var last: MLXArray?
    for _ in 0 ..< max(0, warmupIterations) {
        let warmup = try body()
        eval(warmup)
        last = warmup
    }
    let measuredIterations = max(1, iterations)
    var samples = [Double]()
    samples.reserveCapacity(measuredIterations)
    for _ in 0 ..< measuredIterations {
        let start = Date.timeIntervalSinceReferenceDate
        let output = try body()
        eval(output)
        last = output
        samples.append(Date.timeIntervalSinceReferenceDate - start)
    }
    let averageSeconds = samples.reduce(0, +) / Double(measuredIterations)
    return QwenProofTiming(
        averageSeconds: averageSeconds,
        p50Seconds: percentile(samples, percentile: 0.50),
        p95Seconds: percentile(samples, percentile: 0.95),
        output: last!
    )
}

func timedSamples(
    iterations: Int,
    warmupIterations: Int,
    _ body: () -> MLXArray
) -> QwenProofTiming {
    try! timed(iterations: iterations, warmupIterations: warmupIterations) {
        body()
    }
}

func timedSamplesThrowing(
    iterations: Int,
    warmupIterations: Int,
    _ body: () throws -> MLXArray
) throws -> QwenProofTiming {
    try timed(iterations: iterations, warmupIterations: warmupIterations, body)
}

func throughputMetrics(
    compressed: QwenProofTiming,
    plain: QwenProofTiming,
    queryLength: Int,
    speedParityRatio: Double,
    minExtendedTokensPerSecond: Double,
    productionSource: String
) -> QwenProofThroughput {
    let compressedTPS = Double(queryLength)
        / max(compressed.averageSeconds, Double.leastNonzeroMagnitude)
    let compressedP50TPS = Double(queryLength)
        / max(compressed.p50Seconds, Double.leastNonzeroMagnitude)
    let compressedP95TPS = Double(queryLength)
        / max(compressed.p95Seconds, Double.leastNonzeroMagnitude)
    let plainTPS = Double(queryLength) / max(plain.averageSeconds, Double.leastNonzeroMagnitude)
    let plainP50TPS = Double(queryLength) / max(plain.p50Seconds, Double.leastNonzeroMagnitude)
    let plainP95TPS = Double(queryLength) / max(plain.p95Seconds, Double.leastNonzeroMagnitude)
    let ratio = compressedTPS / max(plainTPS, Double.leastNonzeroMagnitude)
    let ratioP50 = compressedP50TPS / max(plainP50TPS, Double.leastNonzeroMagnitude)
    let ratioP95 = compressedP95TPS / max(plainP95TPS, Double.leastNonzeroMagnitude)
    return QwenProofThroughput(
        compressedSeconds: compressed.averageSeconds,
        compressedSecondsP50: compressed.p50Seconds,
        compressedSecondsP95: compressed.p95Seconds,
        plainSeconds: plain.averageSeconds,
        plainSecondsP50: plain.p50Seconds,
        plainSecondsP95: plain.p95Seconds,
        compressedTokensPerSecond: compressedTPS,
        compressedTokensPerSecondP50: compressedP50TPS,
        compressedTokensPerSecondP95: compressedP95TPS,
        plainTokensPerSecond: plainTPS,
        plainTokensPerSecondP50: plainP50TPS,
        plainTokensPerSecondP95: plainP95TPS,
        speedRatioToPlain: ratio,
        speedRatioToPlainP50: ratioP50,
        speedRatioToPlainP95: ratioP95,
        productionTokensPerSecondP50: compressedP50TPS,
        productionTokensPerSecondP95: compressedP95TPS,
        productionThroughputSource: productionSource,
        minExtendedTokensPerSecond: minExtendedTokensPerSecond,
        passedParityGate: ratioP50 >= speedParityRatio,
        passedProductionGate: compressedP50TPS >= minExtendedTokensPerSecond
    )
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
        let rowCosine = denominator.isFinite && denominator > 0 ? dot / denominator : 0
        cosineTotal += rowCosine.isFinite ? rowCosine : 0
        maxErrors.append(maxError)
    }

    let top1 = Double(top1Matches) / Double(max(1, rowCount))
    let rawKL = klTotal / Double(max(1, rowCount))
    let kl = rawKL.isFinite ? rawKL : Double.greatestFiniteMagnitude
    let rawCosine = cosineTotal / Double(max(1, rowCount))
    let cosine = rawCosine.isFinite ? rawCosine : 0
    let rawP95 = percentile(maxErrors, percentile: 0.95)
    let p95 = rawP95.isFinite ? rawP95 : Double.greatestFiniteMagnitude
    let finite =
        candidateValues.allSatisfy(\.isFinite) && referenceValues.allSatisfy(\.isFinite)
    let passed: Bool
    switch scheme {
    case .turbo8:
        passed = finite && kl <= 0.02 && p95 <= 0.20 && cosine >= 0.995
    case .turbo4v2, .turbo3_5:
        passed = finite && kl <= 0.05 && p95 <= 0.50 && cosine >= 0.990
    case .disabled, .turbo2_5:
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

func sortedWeightIndexes(_ weights: [Double], limit: Int) -> [Int] {
    guard limit > 0 else { return [] }
    return weights.indices.sorted {
        let lhs = weights[$0]
        let rhs = weights[$1]
        return lhs == rhs ? $0 < $1 : lhs > rhs
    }.prefix(limit).map { $0 }
}

func cumulativeWeightIndexes(
    _ weights: [Double],
    target: Double,
    limit: Int?
) -> [Int] {
    let capped = limit.map { max(0, $0) } ?? weights.count
    guard capped > 0, target > 0 else { return [] }
    let sorted = weights.indices.sorted {
        let lhs = weights[$0]
        let rhs = weights[$1]
        return lhs == rhs ? $0 < $1 : lhs > rhs
    }
    var retained: [Int] = []
    retained.reserveCapacity(min(capped, sorted.count))
    var mass = 0.0
    for index in sorted.prefix(capped) {
        retained.append(index)
        mass += weights[index]
        if mass >= target {
            break
        }
    }
    return retained
}

func sparseSelectedIndexes(
    weights: [Double],
    config: QwenProofSparseSelectionConfig,
    threshold: Float?
) -> [Int] {
    func indexes(atOrAbove cutoff: Double) -> [Int] {
        guard cutoff.isFinite else { return [] }
        return weights.indices.filter { weights[$0] >= cutoff }
    }

    func rankedCutoff(limit: Int, cumulativeTarget: Double?) -> Double {
        let uniqueWeights = Array(Set(weights.filter { $0 > 0 })).sorted(by: >)
        let capped = min(max(0, limit), uniqueWeights.count)
        guard capped > 0 else { return Double.infinity }
        var cutoff = Double.infinity
        var retainedMass = 0.0
        for weight in uniqueWeights.prefix(capped) {
            cutoff = weight
            retainedMass += weight
            if let cumulativeTarget, retainedMass >= cumulativeTarget {
                break
            }
        }
        return cutoff
    }

    func cumulativeCutoff(target: Double) -> Double {
        if target <= 0 {
            return Double.infinity
        }
        if target >= 1 {
            return 0
        }
        var low = 0.0
        var high = weights.max() ?? 0
        for _ in 0 ..< 24 {
            let mid = 0.5 * (low + high)
            let mass = weights.reduce(0.0) { $0 + ($1 >= mid ? $1 : 0) }
            if mass >= target {
                low = mid
            } else {
                high = mid
            }
        }
        return low
    }

    switch config.mode {
    case .off:
        return Array(weights.indices)
    case .threshold:
        let cutoff = Double(max(0, threshold ?? config.threshold ?? 0))
        return indexes(atOrAbove: cutoff)
    case .topK:
        return indexes(atOrAbove: rankedCutoff(limit: config.topK ?? 0, cumulativeTarget: nil))
    case .cumulativeMass:
        let target = min(1, max(0, (config.cumulativeMassPercent ?? 0) / 100))
        return indexes(atOrAbove: cumulativeCutoff(target: target))
    case .hybridCumulativeMassTopK:
        let target = min(1, max(0, (config.cumulativeMassPercent ?? 0) / 100))
        let limit = min(max(0, config.maxTopK ?? config.topK ?? 0), weights.count)
        return indexes(atOrAbove: rankedCutoff(limit: limit, cumulativeTarget: target))
    }
}

func sparseSelectionDiagnostics(
    weights: MLXArray,
    config: QwenProofSparseSelectionConfig,
    threshold: Float?,
    selectedOutput: MLXArray?,
    denseOutput: MLXArray?,
    queryHeadCount: Int,
    queryLength: Int,
    headDimension: Int
) -> (TurboQuantSparseValueDiagnostics, [QwenProofSparseHeadDiagnostics]) {
    eval(weights)
    if let selectedOutput {
        eval(selectedOutput)
    }
    if let denseOutput {
        eval(denseOutput)
    }

    let columns = max(1, weights.dim(-1))
    let weightValues = weights.asArray(Float.self).map(Double.init)
    let rows = max(1, weightValues.count / columns)
    let selectedValues = selectedOutput?.asArray(Float.self).map(Double.init)
    let denseValues = denseOutput?.asArray(Float.self).map(Double.init)

    struct HeadAccumulator {
        var skipped = 0
        var considered = 0
        var retainedMass = 0.0
        var rows = 0
        var maxOutputError = 0.0
        var cosineTotal = 0.0
        var qualityRows = 0
    }

    var skipped = 0
    var considered = 0
    var retainedMassTotal = 0.0
    var heads = Array(repeating: HeadAccumulator(), count: max(1, queryHeadCount))

    for row in 0 ..< rows {
        let start = row * columns
        let end = min(weightValues.count, start + columns)
        guard start < end else { continue }
        let rowWeights = Array(weightValues[start ..< end])
        let retainedIndexes = sparseSelectedIndexes(
            weights: rowWeights,
            config: config,
            threshold: threshold
        )
        let retainedMass = retainedIndexes.reduce(0.0) { $0 + rowWeights[$1] }
        let rowSkipped = rowWeights.count - retainedIndexes.count
        let headIndex = min(heads.count - 1, (row / max(1, queryLength)) % max(1, queryHeadCount))

        skipped += rowSkipped
        considered += rowWeights.count
        retainedMassTotal += retainedMass
        heads[headIndex].skipped += rowSkipped
        heads[headIndex].considered += rowWeights.count
        heads[headIndex].retainedMass += retainedMass
        heads[headIndex].rows += 1

        guard let selectedValues, let denseValues else { continue }
        let outputStart = row * headDimension
        let outputEnd = min(selectedValues.count, outputStart + headDimension)
        guard outputStart < outputEnd, outputEnd <= denseValues.count else { continue }
        var dot = 0.0
        var selectedNorm = 0.0
        var denseNorm = 0.0
        var maxError = 0.0
        for index in outputStart ..< outputEnd {
            let candidate = selectedValues[index]
            let reference = denseValues[index]
            dot += candidate * reference
            selectedNorm += candidate * candidate
            denseNorm += reference * reference
            maxError = max(maxError, abs(candidate - reference))
        }
        let denominator = sqrt(selectedNorm) * sqrt(denseNorm)
        let cosine = denominator.isFinite && denominator > 0 ? dot / denominator : 0
        heads[headIndex].maxOutputError = max(heads[headIndex].maxOutputError, maxError)
        heads[headIndex].cosineTotal += cosine.isFinite ? cosine : 0
        heads[headIndex].qualityRows += 1
    }

    let diagnostics = TurboQuantSparseValueDiagnostics(
        enabled: config.isEnabled,
        threshold: threshold ?? config.threshold,
        skipped: skipped,
        considered: considered,
        retainedMass: retainedMassTotal / Double(max(1, rows))
    )
    let headDiagnostics = heads.enumerated().map { index, head in
        QwenProofSparseHeadDiagnostics(
            layerIndex: 0,
            headIndex: index,
            skippedValueTokens: head.skipped,
            totalValueTokens: head.considered,
            retainedMass: head.rows > 0 ? head.retainedMass / Double(head.rows) : nil,
            maxOutputError: head.qualityRows > 0 ? head.maxOutputError : nil,
            cosineSimilarity: head.qualityRows > 0
                ? head.cosineTotal / Double(head.qualityRows)
                : nil
        )
    }
    return (diagnostics, headDiagnostics)
}

func compressedAttentionPathName(
    queries: MLXArray,
    keyCode: TurboQuantAttentionCode,
    valueCode: TurboQuantAttentionCode,
    preferOnlineFused: Bool,
    sparseVThreshold: Float?,
    availability: TurboQuantKernelAvailability
) -> String {
    do {
        let resolvedSparseVThreshold: Float?
        if let sparseVThreshold, sparseVThreshold > 0,
            queries.dim(2) == 1,
            keyCode.layout.headDimension == valueCode.layout.headDimension,
            keyCode.layout.logicalLength == valueCode.layout.logicalLength
        {
            resolvedSparseVThreshold = sparseVThreshold
        } else {
            resolvedSparseVThreshold = nil
        }
        let decision = try turboQuantAttentionDecision(
            request: TurboQuantAttentionRequest(
                queryShape: queries.shape,
                keyLayout: keyCode.layout,
                valueLayout: valueCode.layout,
                queryDType: queries.dtype,
                outputDType: queries.dtype,
                maskKind: .causal,
                hasSinks: false,
                preferOnlineFused: preferOnlineFused && resolvedSparseVThreshold == nil,
                memoryBudgetBytes: nil,
                fallbackState: .none,
                sparseVThreshold: resolvedSparseVThreshold
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
    reservedCapacityLength: Int,
    queryLength: Int,
    dtype: DType,
    gateScope: QwenProofGateScope,
    strictGateRequired: Bool,
    iterations: Int,
    warmupIterations: Int,
    speedParityRatio: Double,
    minExtendedTokensPerSecond: Double,
    shortContextPlainKVThreshold: Int,
    availability: TurboQuantKernelAvailability,
    blockParallelTokenBlockSize: Int? = nil,
    attentionPath: QwenProofAttentionPath = .auto,
    requestedRuntimeMode: TurboQuantRuntimeMode = .auto,
    sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
    sparseSelectionConfig: QwenProofSparseSelectionConfig = .off,
    codec: QwenProofCodec = .polarQJL
) -> QwenProofResult {
    let caseID =
        "\(profile.id)_\(scheme.rawValue)_\(codec.rawValue)_ctx\(contextLength)_cap\(reservedCapacityLength)_q\(queryLength)\(attentionPath.idSuffix)\(sparseSelectionConfig.idSuffix)"
    guard let precisionProfile = profile.applyingPrecisionCandidate(scheme) else {
        let blockPlan = blockParallelTokenPlan(
            contextLength: contextLength,
            headDimension: 256,
            queryLength: queryLength,
            availability: availability,
            requestedBlockParallelTokenBlockSize: blockParallelTokenBlockSize
        )
        let benchmarkCase = QwenProofCase(
            id: caseID,
            profileID: profile.id,
            scheme: scheme.rawValue,
            codec: codec.rawValue,
            requestedRuntimeMode: requestedRuntimeMode.rawValue,
            resolvedRuntimeMode: resolvedProofRuntimeMode(
                requestedRuntimeMode: requestedRuntimeMode,
                contextLength: contextLength,
                shortContextPlainKVThreshold: shortContextPlainKVThreshold
            ).rawValue,
            dtype: dtypeName(dtype),
            gateScope: gateScope,
            headDimension: 256,
            kvHeads: 4,
            queryHeads: 16,
            contextLength: contextLength,
            reservedCapacityLength: reservedCapacityLength,
            blockParallelTokenBlockSize: blockParallelTokenBlockSize,
            recommendedBlockParallelTokenBlockSize: blockPlan.recommended,
            effectiveBlockParallelTokenBlockSize: blockPlan.effective,
            queryLength: queryLength
        )
        return QwenProofResult(
            id: benchmarkCase.id,
            status: "skipped",
            benchmarkCase: benchmarkCase,
            gateScope: gateScope,
            strictGateRequired: strictGateRequired,
            certificationStatus: certificationStatus(for: gateScope),
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
    let precisionPolicy =
        precisionProfile.turboQuant.precisionPolicy
        ?? TurboQuantKVPrecisionPolicy.legacy(
            preset: precisionProfile.recommendedScheme.preset,
            valueBits: precisionProfile.valueBits
        )
    let resolvedRuntimeMode = resolvedProofRuntimeMode(
        requestedRuntimeMode: requestedRuntimeMode,
        contextLength: contextLength,
        shortContextPlainKVThreshold: shortContextPlainKVThreshold
    )
    let kvHeads = 4
    let queryHeads = 16
    let headDimension = 256
    let sparseVThreshold = sparseSelectionConfig.resolvedThreshold(
        policy: sparseValuePolicy,
        runtimeMode: resolvedRuntimeMode,
        contextLength: contextLength
    )
    let sparseSelectionEnabled = sparseSelectionConfig.enabledFor(
        runtimeMode: resolvedRuntimeMode,
        queryLength: queryLength
    ) && (sparseSelectionConfig.mode != .threshold || sparseVThreshold != nil)
    let layerCount = precisionProfile.modelFingerprint?.layerCount
    let boundaryProtectedLayerCount =
        layerCount.map { precisionPolicy.protectedBoundaryLayerIndexes(layerCount: $0).count } ?? 0
    let boundaryProtectionReason: String? =
        boundaryProtectedLayerCount > 0
        ? "K8/V4 boundary protection for low-bit K/V policy"
        : (precisionPolicy.requiresBoundaryProtection
            ? "K8/V4 boundary protection requested but profile layer count unavailable"
            : nil)
    let blockPlan = blockParallelTokenPlan(
        contextLength: contextLength,
        headDimension: headDimension,
        queryLength: queryLength,
        availability: availability,
        requestedBlockParallelTokenBlockSize: blockParallelTokenBlockSize
    )
    let benchmarkCase = QwenProofCase(
        id: caseID,
        profileID: profile.id,
        scheme: scheme.rawValue,
        codec: codec.rawValue,
        requestedRuntimeMode: requestedRuntimeMode.rawValue,
        resolvedRuntimeMode: resolvedRuntimeMode.rawValue,
        keyPrecision: precisionPolicy.key.rawValue,
        valuePrecision: precisionPolicy.value.rawValue,
        backend: precisionProfile.backend.rawValue,
        sparseVEnabled: sparseSelectionEnabled,
        sparseVSelectionMode: sparseSelectionEnabled
            ? sparseSelectionConfig.mode.reportName
            : QwenProofSparseSelectionMode.off.reportName,
        sparseVThreshold: sparseVThreshold,
        sparseVTopK: sparseSelectionEnabled ? sparseSelectionConfig.topK : nil,
        sparseVCumulativeMassPercent: sparseSelectionEnabled
            ? sparseSelectionConfig.cumulativeMassPercent
            : nil,
        sparseVMaxTopK: sparseSelectionEnabled ? sparseSelectionConfig.maxTopK : nil,
        boundaryProtectedLayerCount: boundaryProtectedLayerCount,
        boundaryProtectionReason: boundaryProtectionReason,
        dtype: dtypeName(dtype),
        gateScope: gateScope,
        headDimension: headDimension,
        kvHeads: kvHeads,
        queryHeads: queryHeads,
        contextLength: contextLength,
        reservedCapacityLength: reservedCapacityLength,
        blockParallelTokenBlockSize: blockParallelTokenBlockSize,
        recommendedBlockParallelTokenBlockSize: blockPlan.recommended,
        effectiveBlockParallelTokenBlockSize: blockPlan.effective,
        queryLength: queryLength
    )

    guard codec != .polarQJL || availability.supportsMetalPolarQJLAttention else {
        return QwenProofResult(
            id: benchmarkCase.id,
            status: "skipped",
            benchmarkCase: benchmarkCase,
            gateScope: gateScope,
            strictGateRequired: strictGateRequired,
            certificationStatus: certificationStatus(for: gateScope),
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
        if codec == .raw {
            let compressedTiming = try timed(iterations: iterations, warmupIterations: warmupIterations) {
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .causal
                )
            }
            let output = compressedTiming.output
            let quality = qualityGate(candidate: output, reference: output, scheme: scheme)
            let plainBytes = keys.nbytes + values.nbytes
            return QwenProofResult(
                id: benchmarkCase.id,
                status: "ok",
                benchmarkCase: benchmarkCase,
                gateScope: gateScope,
                strictGateRequired: strictGateRequired,
                certificationStatus: certificationStatus(for: gateScope),
                selectedAttentionPath: TurboQuantAttentionPath.baseline.rawValue,
                productionKVRoute: "raw",
                requestedRuntimeMode: requestedRuntimeMode.rawValue,
                resolvedRuntimeMode: resolvedRuntimeMode.rawValue,
                keyPrecision: "raw",
                valuePrecision: "raw",
                codec: codec.rawValue,
                backend: "rawSDPA",
                precisionStatus: candidate?.status.rawValue ?? "unknown",
                quality: quality,
                throughput: throughputMetrics(
                    compressed: compressedTiming,
                    plain: compressedTiming,
                    queryLength: queryLength,
                    speedParityRatio: speedParityRatio,
                    minExtendedTokensPerSecond: minExtendedTokensPerSecond,
                    productionSource: "raw"
                ),
                memory: QwenProofMemory(
                    compressedKeyBytes: keys.nbytes,
                    compressedValueBytes: values.nbytes,
                    compressedBytesPerToken: Double(plainBytes) / Double(max(1, contextLength)),
                    compressedBytesPerLogicalToken: Double(plainBytes) / Double(max(1, contextLength)),
                    compressedBytesPerReservedToken: Double(plainBytes)
                        / Double(max(1, reservedCapacityLength)),
                    plainKVBytes: plainBytes,
                    plainKVBytesAtReservedCapacity: plainBytes,
                    compressionRatioToPlain: 1,
                    compressionRatioToPlainReservedCapacity: 1,
                    reservedCapacityMultiplier: Double(reservedCapacityLength)
                        / Double(max(1, contextLength))
                ),
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
                values,
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
                mask: .causal,
                groupSize: groupSize
            ) else {
                return QwenProofResult(
                    id: benchmarkCase.id,
                    status: "skipped",
                    benchmarkCase: benchmarkCase,
                    gateScope: gateScope,
                    strictGateRequired: strictGateRequired,
                    certificationStatus: certificationStatus(for: gateScope),
                    selectedAttentionPath: nil,
                    precisionStatus: candidate?.status.rawValue ?? "unknown",
                    quality: nil,
                    throughput: nil,
                    memory: nil,
                    fallbackReason: "native affine int4 SDPA unsupported",
                    error: nil
                )
            }
            let compressedTiming = try timed(
                iterations: iterations,
                warmupIterations: warmupIterations
            ) {
                try affineInt4NativeScaledDotProductAttention(
                    queries: queries,
                    quantizedKeys: keyTuple,
                    quantizedValues: valueTuple,
                    scale: scale,
                    mask: .causal,
                    groupSize: groupSize
                )
            }
            let plainTiming = try timed(iterations: iterations, warmupIterations: warmupIterations) {
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: .causal
                )
            }
            let quality = qualityGate(
                candidate: compressedTiming.output,
                reference: plainTiming.output,
                scheme: scheme
            )
            let compressedKeyBytes = qKeys.wq.nbytes + qKeys.scales.nbytes
                + (qKeys.biases?.nbytes ?? 0)
            let compressedValueBytes = qValues.wq.nbytes + qValues.scales.nbytes
                + (qValues.biases?.nbytes ?? 0)
            let compressedBytes = compressedKeyBytes + compressedValueBytes
            let plainBytes = keys.nbytes + values.nbytes
            let scaleBiasBytes = qKeys.scales.nbytes + (qKeys.biases?.nbytes ?? 0)
                + qValues.scales.nbytes + (qValues.biases?.nbytes ?? 0)
            return QwenProofResult(
                id: benchmarkCase.id,
                status: "ok",
                benchmarkCase: benchmarkCase,
                gateScope: gateScope,
                strictGateRequired: strictGateRequired,
                certificationStatus: certificationStatus(for: gateScope),
                selectedAttentionPath: TurboQuantAttentionPath.affineInt4Native.rawValue,
                productionKVRoute: "affine_int4_native",
                requestedRuntimeMode: requestedRuntimeMode.rawValue,
                resolvedRuntimeMode: "nativeAffineInt4",
                keyPrecision: "affineInt4",
                valuePrecision: "affineInt4",
                codec: codec.rawValue,
                backend: TurboQuantBackend.mlxPacked.rawValue,
                precisionStatus: candidate?.status.rawValue ?? "unknown",
                quality: quality,
                throughput: throughputMetrics(
                    compressed: compressedTiming,
                    plain: plainTiming,
                    queryLength: queryLength,
                    speedParityRatio: speedParityRatio,
                    minExtendedTokensPerSecond: minExtendedTokensPerSecond,
                    productionSource: "affineInt4Native"
                ),
                memory: QwenProofMemory(
                    compressedKeyBytes: compressedKeyBytes,
                    compressedValueBytes: compressedValueBytes,
                    scaleBiasBytes: scaleBiasBytes,
                    compressedBytesPerToken: Double(compressedBytes)
                        / Double(max(1, contextLength)),
                    compressedBytesPerLogicalToken: Double(compressedBytes)
                        / Double(max(1, contextLength)),
                    compressedBytesPerReservedToken: Double(compressedBytes)
                        / Double(max(1, reservedCapacityLength)),
                    plainKVBytes: plainBytes,
                    plainKVBytesAtReservedCapacity: plainBytes,
                    compressionRatioToPlain: Double(plainBytes) / Double(max(1, compressedBytes)),
                    compressionRatioToPlainReservedCapacity: Double(plainBytes)
                        / Double(max(1, compressedBytes)),
                    reservedCapacityMultiplier: Double(reservedCapacityLength)
                        / Double(max(1, contextLength))
                ),
                fallbackReason: nil,
                error: nil
            )
        }
        let cache = TurboQuantKVCache(
            preset: precisionPolicy.compressedKeyPreset,
            groupSize: precisionProfile.groupSize,
            backend: precisionProfile.backend,
            optimizationPolicy: precisionProfile.optimizationPolicy,
            fallbackPolicy: precisionProfile.turboQuant.fallbackPolicy,
            valueBits: precisionPolicy.resolvedValueBits ?? precisionProfile.valueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: requestedRuntimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode
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
                gateScope: gateScope,
                strictGateRequired: strictGateRequired,
                certificationStatus: certificationStatus(for: gateScope),
                selectedAttentionPath: cache.attentionDiagnostics.activeAttentionPath.rawValue,
                precisionStatus: candidate?.status.rawValue ?? "unknown",
                quality: nil,
                throughput: nil,
                memory: nil,
                fallbackReason: cache.attentionDiagnostics.lastUnsupportedShape,
                error: nil
            )
        }

        let keyConfiguration = TurboQuantConfiguration(
            preset: precisionPolicy.compressedKeyPreset,
            role: .key,
            groupSize: precisionProfile.groupSize,
            backend: precisionProfile.backend
        )
        let valueConfiguration = TurboQuantConfiguration(
            preset: precisionPolicy.compressedKeyPreset,
            role: .value,
            groupSize: precisionProfile.groupSize,
            backend: precisionProfile.backend,
            seed: 0x9E37_79B9_7F4A_7C15 ^ 0xD1B5_4A32_D192_ED03,
            valueBits: precisionPolicy.resolvedValueBits ?? precisionProfile.valueBits
        )
        let (compressedKeys, compressedValues) = (
            try turboQuantMetalEncodeAttention(
                keys,
                configuration: keyConfiguration,
                capacity: reservedCapacityLength,
                logicalLength: contextLength
            ),
            try turboQuantMetalEncodeAttention(
                values,
                configuration: valueConfiguration,
                capacity: reservedCapacityLength,
                logicalLength: contextLength
            )
        )
        let preferOnline = attentionPath.preferOnlineFused(
            default: cache.prefersOnlineFusedAttention
        )
        let nativeSparseSelectionAvailable =
            availability.attentionCapabilities.nativeCompressedAttention == true
            && availability.attentionCapabilities.nativeSparseVSupport == true
            && queries.dim(2) == 1
            && compressedKeys.layout.headDimension == compressedValues.layout.headDimension
            && compressedKeys.layout.logicalLength == compressedValues.layout.logicalLength
        let useNativeSparseSelection = sparseSelectionEnabled && nativeSparseSelectionAvailable
        let denseSelectedAttentionPath = compressedAttentionPathName(
            queries: queries,
            keyCode: compressedKeys,
            valueCode: compressedValues,
            preferOnlineFused: preferOnline,
            sparseVThreshold: nil,
            availability: availability
        )
        let selectedAttentionPath =
            useNativeSparseSelection
            ? TurboQuantAttentionPath.nativeMLXCompressed.rawValue
            : denseSelectedAttentionPath
        let plainTiming = try timed(iterations: iterations, warmupIterations: warmupIterations) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .causal
            )
        }
        let compressedTiming = try timed(
            iterations: iterations,
            warmupIterations: warmupIterations
        ) {
            if useNativeSparseSelection {
                return try turboQuantNativeScaledDotProductAttention(
                    queries: queries,
                    keyCode: compressedKeys,
                    valueCode: compressedValues,
                    options: sparseSelectionConfig.nativeOptions(
                        scale: scale,
                        causal: true,
                        threshold: sparseVThreshold,
                        diagnostics: false,
                        backendVersion: availability.attentionCapabilities.nativeBackendVersion
                    )
                )
            }
            return try turboQuantMetalScaledDotProductAttention(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                scale: scale,
                mask: .causal,
                preferOnlineFused: preferOnline,
                kernelProfile: cache.attentionDiagnostics.selectedKernelProfile,
                blockParallelTokenBlockSize: blockParallelTokenBlockSize,
                sparseVThreshold: nil
            )
        }
        let denseCompressedReference: MLXArray?
        if useNativeSparseSelection {
            denseCompressedReference = try turboQuantNativeScaledDotProductAttention(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                options: TurboQuantNativeAttentionOptions(
                    scale: scale,
                    causal: true,
                    backendVersion: availability.attentionCapabilities.nativeBackendVersion
                        ?? TurboQuantNativeAttentionOptions.backendVersion
                )
            )
        } else {
            denseCompressedReference = nil
        }
        let sparseDiagnostics: TurboQuantSparseValueDiagnostics
        let sparseHeadDiagnostics: [QwenProofSparseHeadDiagnostics]?
        if useNativeSparseSelection {
            let scores = try turboQuantMetalQK(
                queries: queries,
                keyCode: compressedKeys,
                scale: scale,
                mask: .causal,
            )
            let diagnosticResult = sparseSelectionDiagnostics(
                weights: softmax(scores.asType(.float32), axis: -1),
                config: sparseSelectionConfig,
                threshold: sparseVThreshold,
                selectedOutput: compressedTiming.output,
                denseOutput: denseCompressedReference,
                queryHeadCount: queryHeads,
                queryLength: queryLength,
                headDimension: headDimension
            )
            sparseDiagnostics = diagnosticResult.0
            sparseHeadDiagnostics = diagnosticResult.1
        } else {
            sparseDiagnostics = TurboQuantSparseValueDiagnostics(enabled: false)
            sparseHeadDiagnostics = nil
        }
        let sparseDenseQuality: QwenProofQuality?
        if let denseCompressedReference {
            sparseDenseQuality = qualityGate(
                candidate: compressedTiming.output,
                reference: denseCompressedReference,
                scheme: scheme
            )
        } else {
            sparseDenseQuality = nil
        }
        let nativeAttentionDiagnostics: [Int]?
        if useNativeSparseSelection,
            let diagnostics = try? turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                queries: queries,
                keyCode: compressedKeys,
                valueCode: compressedValues,
                options: sparseSelectionConfig.nativeOptions(
                    scale: scale,
                    causal: true,
                    threshold: sparseVThreshold,
                    diagnostics: true,
                    backendVersion: availability.attentionCapabilities.nativeBackendVersion
                )
            ).diagnostics
        {
            nativeAttentionDiagnostics = [
                diagnostics.backendVersion,
                diagnostics.kernelKind,
                diagnostics.activeBlocks,
                diagnostics.blockTokens,
                diagnostics.sparseSkippedTokens,
                diagnostics.sparseTotalTokens,
                diagnostics.fallbackCode,
                diagnostics.flags,
            ]
        } else {
            nativeAttentionDiagnostics = nil
        }
        let quality = qualityGate(
            candidate: compressedTiming.output,
            reference: plainTiming.output,
            scheme: scheme
        )
        let compressedTPS =
            Double(queryLength) / max(compressedTiming.averageSeconds, Double.leastNonzeroMagnitude)
        let compressedTPSP50 =
            Double(queryLength) / max(compressedTiming.p50Seconds, Double.leastNonzeroMagnitude)
        let compressedTPSP95 =
            Double(queryLength) / max(compressedTiming.p95Seconds, Double.leastNonzeroMagnitude)
        let plainTPS =
            Double(queryLength) / max(plainTiming.averageSeconds, Double.leastNonzeroMagnitude)
        let plainTPSP50 =
            Double(queryLength) / max(plainTiming.p50Seconds, Double.leastNonzeroMagnitude)
        let plainTPSP95 =
            Double(queryLength) / max(plainTiming.p95Seconds, Double.leastNonzeroMagnitude)
        let speedRatio = compressedTPS / max(plainTPS, Double.leastNonzeroMagnitude)
        let speedRatioP50 = compressedTPSP50 / max(plainTPSP50, Double.leastNonzeroMagnitude)
        let speedRatioP95 = compressedTPSP95 / max(plainTPSP95, Double.leastNonzeroMagnitude)
        let usesPlainProductionRoute = resolvedRuntimeMode == .rawPreferred
        let productionKVRoute = runtimeRouteName(resolvedRuntimeMode)
        let productionTPSP50 =
            resolvedRuntimeMode == .capacityTurboQuant ? compressedTPSP50 : plainTPSP50
        let productionTPSP95 =
            resolvedRuntimeMode == .capacityTurboQuant ? compressedTPSP95 : plainTPSP95
        let productionThroughputSource: String
        switch resolvedRuntimeMode {
        case .rawPreferred:
            productionThroughputSource = "plainRawSDPA"
        case .throughputTurboQuant:
            productionThroughputSource = "throughputNativeSDPA"
        case .capacityTurboQuant:
            productionThroughputSource = "capacityCompressedAttention"
        case .auto:
            productionThroughputSource = "auto"
        }
        let passedParityGate = usesPlainProductionRoute || productionKVRoute == "throughputTurboQuantNativeSDPA" || speedRatioP95 >= speedParityRatio
        let passedProductionGate =
            productionTPSP95 >= minExtendedTokensPerSecond
        let throughput = QwenProofThroughput(
            compressedSeconds: compressedTiming.averageSeconds,
            compressedSecondsP50: compressedTiming.p50Seconds,
            compressedSecondsP95: compressedTiming.p95Seconds,
            plainSeconds: plainTiming.averageSeconds,
            plainSecondsP50: plainTiming.p50Seconds,
            plainSecondsP95: plainTiming.p95Seconds,
            compressedTokensPerSecond: compressedTPS,
            compressedTokensPerSecondP50: compressedTPSP50,
            compressedTokensPerSecondP95: compressedTPSP95,
            plainTokensPerSecond: plainTPS,
            plainTokensPerSecondP50: plainTPSP50,
            plainTokensPerSecondP95: plainTPSP95,
            speedRatioToPlain: speedRatio,
            speedRatioToPlainP50: speedRatioP50,
            speedRatioToPlainP95: speedRatioP95,
            productionTokensPerSecondP50: productionTPSP50,
            productionTokensPerSecondP95: productionTPSP95,
            productionThroughputSource: productionThroughputSource,
            minExtendedTokensPerSecond: minExtendedTokensPerSecond,
            passedParityGate: passedParityGate,
            passedProductionGate: passedProductionGate
        )
        let compressedBytes = compressedKeys.storageByteCount + compressedValues.storageByteCount
        let plainBytes = keys.nbytes + values.nbytes
        let plainBytesAtReservedCapacity =
            plainBytes * reservedCapacityLength / max(1, contextLength)
        let memory = QwenProofMemory(
            compressedKeyBytes: compressedKeys.storageByteCount,
            compressedValueBytes: compressedValues.storageByteCount,
            compressedBytesPerToken: Double(compressedBytes) / Double(max(1, contextLength)),
            compressedBytesPerLogicalToken: Double(compressedBytes) / Double(max(1, contextLength)),
            compressedBytesPerReservedToken: Double(compressedBytes)
                / Double(max(1, reservedCapacityLength)),
            plainKVBytes: plainBytes,
            plainKVBytesAtReservedCapacity: plainBytesAtReservedCapacity,
            compressionRatioToPlain: Double(compressedBytes) / Double(max(1, plainBytes)),
            compressionRatioToPlainReservedCapacity: Double(compressedBytes)
                / Double(max(1, plainBytesAtReservedCapacity)),
            reservedCapacityMultiplier: Double(reservedCapacityLength)
                / Double(max(1, contextLength))
        )
        let fallbackReason: String?
        if usesPlainProductionRoute {
            fallbackReason =
                "speed parity gate waived because production routing uses plain KV at or below \(shortContextPlainKVThreshold) tokens"
        } else if attentionPath != .auto {
            fallbackReason =
                "attention path forced to \(attentionPath.rawValue) for path comparison; selected \(selectedAttentionPath)"
        } else if sparseSelectionEnabled
            && selectedAttentionPath != TurboQuantAttentionPath.nativeMLXCompressed.rawValue
        {
            fallbackReason =
                "Sparse-V \(sparseSelectionConfig.mode.reportName) requested but selected dense \(selectedAttentionPath); nativeSparseVSupport=\(availability.attentionCapabilities.nativeSparseVSupport == true)"
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
            gateScope: gateScope,
            strictGateRequired: strictGateRequired,
            certificationStatus: certificationStatus(for: gateScope),
            selectedAttentionPath: selectedAttentionPath,
            productionKVRoute: productionKVRoute,
            requestedRuntimeMode: requestedRuntimeMode.rawValue,
            resolvedRuntimeMode: resolvedRuntimeMode.rawValue,
            keyPrecision: precisionPolicy.key.rawValue,
            valuePrecision: precisionPolicy.value.rawValue,
            codec: codec.rawValue,
            backend: precisionProfile.backend.rawValue,
            sparseVEnabled: sparseDiagnostics.enabled,
            sparseVSelectionMode: sparseDiagnostics.enabled
                ? sparseSelectionConfig.mode.reportName
                : QwenProofSparseSelectionMode.off.reportName,
            sparseVThreshold: sparseDiagnostics.threshold,
            sparseVTopK: sparseDiagnostics.enabled ? sparseSelectionConfig.topK : nil,
            sparseVCumulativeMassPercent: sparseDiagnostics.enabled
                ? sparseSelectionConfig.cumulativeMassPercent
                : nil,
            sparseVMaxTopK: sparseDiagnostics.enabled ? sparseSelectionConfig.maxTopK : nil,
            sparseVSkipRatio: sparseDiagnostics.skipRatio,
            sparseVSkippedValueTokens: sparseDiagnostics.enabled ? sparseDiagnostics.skipped : nil,
            sparseVTotalValueTokens: sparseDiagnostics.enabled ? sparseDiagnostics.considered : nil,
            sparseVRetainedMass: sparseDiagnostics.retainedMass,
            sparseVDenseCosineSimilarity: sparseDenseQuality?.cosineSimilarityMean,
            sparseVDenseMaxAbsErrorP95: sparseDenseQuality?.maxAbsErrorP95,
            sparseVHeadDiagnostics: sparseHeadDiagnostics,
            lowerVAndSparseV: QwenProofLowerVAndSparseVReport(
                referenceConfig: "dense K8/V4",
                candidateConfig: "\(codec.rawValue)-\(precisionPolicy.value.rawValue)"
                    + (sparseDiagnostics.enabled ? "-sparse-\(sparseSelectionConfig.mode.reportName)" : ""),
                valueBits: precisionPolicy.resolvedValueBits,
                valueBitPolicy: valueBitPolicyName(valueBits: precisionPolicy.resolvedValueBits),
                sparseVMode: sparseDiagnostics.enabled ? sparseSelectionConfig.mode.reportName : nil,
                sparseVTopK: sparseDiagnostics.enabled ? sparseSelectionConfig.topK : nil,
                sparseVCumulativeMass: sparseDiagnostics.enabled
                    ? sparseSelectionConfig.cumulativeMassPercent.map { $0 / 100 }
                    : nil,
                sparseVMaxTopK: sparseDiagnostics.enabled ? sparseSelectionConfig.maxTopK : nil,
                selectionLatencyMS: nil,
                qkMS: nil,
                softmaxMS: nil,
                maskOrCompactionMS: nil,
                avLatencyMS: nil,
                totalMS: nil,
                denseK8V4ReferenceMS: nil,
                skippedValueTokens: sparseDiagnostics.enabled ? sparseDiagnostics.skipped : nil,
                consideredValueTokens: sparseDiagnostics.enabled ? sparseDiagnostics.considered : nil,
                retainedMass: sparseDiagnostics.retainedMass,
                skipRatio: sparseDiagnostics.enabled ? sparseDiagnostics.skipRatio : nil,
                fallbackCount: sparseDiagnostics.enabled && fallbackReason != nil ? 1 : 0,
                fallbackReason: sparseDiagnostics.enabled ? fallbackReason : nil,
                actualMixedBitsPerValue: precisionPolicy.resolvedValueBits.map(Double.init),
                layerIndex: nil,
                headIndex: nil
            ),
            nativeAttentionDiagnostics: nativeAttentionDiagnostics,
            boundaryProtectedLayerCount: boundaryProtectedLayerCount,
            boundaryProtectionReason: boundaryProtectionReason,
            decodedActiveKVBytes: usesPlainProductionRoute ? nil : plainBytes,
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
            gateScope: gateScope,
            strictGateRequired: strictGateRequired,
            certificationStatus: certificationStatus(for: gateScope),
            selectedAttentionPath: nil,
            productionKVRoute: nil,
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
let warmupIterations = argumentValue("--warmup", default: 1)
let releaseMatrix = hasFlag("--release-matrix")
let pathCompare = hasFlag("--path-compare")
let strict = hasFlag("--strict")
let proofDType = argumentDType("--dtype", default: .float16)
let codec = argumentCodec("--codec", default: .polarQJL)
let requestedRuntimeMode = argumentRuntimeMode("--runtime-mode", default: .auto)
let sparseValueThreshold = Float(
    argumentValue(
        "--sparse-v-threshold",
        default: Double(TurboQuantSparseValuePolicy.defaultAutoThreshold)
    )
)
let sparseValuePolicy = argumentSparseValuePolicy(
    "--sparse-v",
    threshold: sparseValueThreshold
)
let sparseSelectionConfig = argumentSparseSelectionConfig(
    policy: sparseValuePolicy,
    threshold: sparseValueThreshold
)
let speedParityRatio = argumentValue("--speed-parity-ratio", default: 1.0)
let minExtendedTokensPerSecond = argumentValue("--min-extended-tokens-per-second", default: 20.0)
let shortContextPlainKVThreshold = argumentValue("--plain-route-threshold", default: 4096)
let reservedCapacityOverride = optionalPositiveArgumentValue("--reserved-capacity")
let blockParallelTokenBlockSize = optionalPositiveArgumentValue("--block-tokens")
let cooldownMilliseconds = argumentValue("--cooldown-ms", default: 0)
let contexts = argumentValues(
    "--contexts",
    default: releaseMatrix ? [8192, 16384] : [8192]
)
let largeContextExperimentContexts = argumentValues(
    "--experimental-contexts",
    default: []
)
let requireLargeContextExperimentGates = hasFlag("--require-experimental-gates")
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
        supportedContextLengths: profile.supportedContextLengths,
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
let contextMatrix =
    uniqueSorted(contexts).map { (context: $0, gateScope: QwenProofGateScope.production) }
    + uniqueSorted(largeContextExperimentContexts)
    .filter { !contexts.contains($0) }
    .map { (context: $0, gateScope: QwenProofGateScope.largeContextExperiment) }
for profile in representativeProfiles {
    for scheme in schemes {
        for entry in contextMatrix {
            let context = entry.context
            guard context <= (profile.safeContextLength ?? context) else { continue }
            let reservedCapacityLength = roundedReservedCapacityLength(
                max(context, reservedCapacityOverride ?? context)
            )
            for queryLength in queryLengths {
                for attentionPath in attentionPaths {
                    results.append(
                        runCase(
                            profile: profile,
                            scheme: scheme,
                            contextLength: context,
                            reservedCapacityLength: reservedCapacityLength,
                            queryLength: queryLength,
                            dtype: proofDType,
                            gateScope: entry.gateScope,
                            strictGateRequired: entry.gateScope == .production
                                || requireLargeContextExperimentGates,
                            iterations: iterations,
                            warmupIterations: warmupIterations,
                            speedParityRatio: speedParityRatio,
                            minExtendedTokensPerSecond: minExtendedTokensPerSecond,
                            shortContextPlainKVThreshold: shortContextPlainKVThreshold,
                            availability: availability,
                            blockParallelTokenBlockSize: blockParallelTokenBlockSize,
                            attentionPath: attentionPath,
                            requestedRuntimeMode: requestedRuntimeMode,
                            sparseValuePolicy: sparseValuePolicy,
                            sparseSelectionConfig: sparseSelectionConfig,
                            codec: codec
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
let productionResults = results.filter { $0.gateScope == .production }
let largeContextExperimentResults = results.filter { $0.gateScope == .largeContextExperiment }
let productionOkCount = productionResults.filter { $0.status == "ok" }.count
let productionSkippedCount = productionResults.filter { $0.status == "skipped" }.count
let productionFailedCount = productionResults.filter { $0.status == "failed" }.count
let largeContextExperimentOkCount =
    largeContextExperimentResults.filter { $0.status == "ok" }.count
let largeContextExperimentSkippedCount =
    largeContextExperimentResults.filter { $0.status == "skipped" }.count
let largeContextExperimentFailedCount =
    largeContextExperimentResults.filter { $0.status == "failed" }.count
let strictGateResults =
    requireLargeContextExperimentGates ? results : productionResults
let strictPassed =
    !strictGateResults.isEmpty
    && strictGateResults.allSatisfy { $0.status == "ok" }
let report = QwenProofReport(
    schemaVersion: 10,
    generatedAt: ISO8601DateFormatter().string(from: Date()),
    iterations: iterations,
    warmupIterations: warmupIterations,
    dtype: dtypeName(proofDType),
    speedParityRatio: speedParityRatio,
    minExtendedTokensPerSecond: minExtendedTokensPerSecond,
    shortContextPlainKVThreshold: shortContextPlainKVThreshold,
    requestedReservedCapacityLength: reservedCapacityOverride,
    requestedBlockParallelTokenBlockSize: blockParallelTokenBlockSize,
    blockParallelTokenBlockPolicy:
        "auto uses MLX.turboQuantRecommendedBlockParallelTokenBlockSize; --block-tokens overrides and is rounded to the fused kernel power-of-two threadgroup width",
    requestedRuntimeMode: requestedRuntimeMode.rawValue,
    sparseValuePolicy: sparseValuePolicyName(sparseValuePolicy),
    sparseValueThreshold: sparseValueThreshold,
    sparseValueSelectionMode: sparseSelectionConfig.mode.reportName,
    sparseValueTopK: sparseSelectionConfig.topK,
    sparseValueCumulativeMassPercent: sparseSelectionConfig.cumulativeMassPercent,
    sparseValueMaxTopK: sparseSelectionConfig.maxTopK,
    codec: codec.rawValue,
    productionContexts: uniqueSorted(contexts),
    largeContextExperimentContexts: uniqueSorted(largeContextExperimentContexts),
    requireLargeContextExperimentGates: requireLargeContextExperimentGates,
    device: TurboQuantDeviceCapabilities.current,
    supportsMetalCodec: availability.supportsMetalPolarQJLCodec,
    supportsMetalAttention: availability.supportsMetalPolarQJLAttention,
    profileCoverage: profileCoverage,
    results: results,
    summary: QwenProofSummary(
        ok: okCount,
        skipped: skippedCount,
        failed: failedCount,
        productionOk: productionOkCount,
        productionSkipped: productionSkippedCount,
        productionFailed: productionFailedCount,
        largeContextExperimentOk: largeContextExperimentOkCount,
        largeContextExperimentSkipped: largeContextExperimentSkippedCount,
        largeContextExperimentFailed: largeContextExperimentFailedCount,
        strictPassed: strictPassed
    )
)
FileHandle.standardError.write(Data(
    "SYNTHETIC KERNEL MICROBENCH — NOT real-model, NOT promotable (sinusoid K/V/Q, no checkpoint loaded)\n".utf8))
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
