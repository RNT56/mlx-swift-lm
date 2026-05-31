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
    public var codec: TurboQuantKVCodec
    public var runtimeMode: TurboQuantRuntimeMode
    public var precisionPolicy: TurboQuantKVPrecisionPolicy?
    public var sparseValuePolicy: TurboQuantSparseValuePolicy
    public var dtype: DType

    public init(
        label: String,
        kvHeads: Int,
        queryHeads: Int,
        headDimension: Int,
        contextLength: Int,
        queryLength: Int = 1,
        scheme: TurboQuantScheme,
        codec: TurboQuantKVCodec = .polarQJL,
        runtimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
        dtype: DType = .float16
    ) {
        self.label = label
        self.kvHeads = kvHeads
        self.queryHeads = queryHeads
        self.headDimension = headDimension
        self.contextLength = contextLength
        self.queryLength = queryLength
        self.scheme = scheme
        self.codec = codec
        self.runtimeMode = runtimeMode
        self.precisionPolicy = precisionPolicy
        self.sparseValuePolicy = sparseValuePolicy
        self.dtype = dtype
    }

    /// Production Qwen3.5-2B attention geometry (kv=4, q=16, head_dim=256) at decode.
    public static func qwen35_2B(
        contextLength: Int,
        scheme: TurboQuantScheme,
        codec: TurboQuantKVCodec = .polarQJL,
        runtimeMode: TurboQuantRuntimeMode = .capacityTurboQuant,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy = .off,
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
            codec: codec,
            runtimeMode: runtimeMode,
            precisionPolicy: precisionPolicy,
            sparseValuePolicy: sparseValuePolicy,
            dtype: dtype
        )
    }
}

public struct TurboQuantHybridBenchCase: Sendable {
    public var label: String
    public var contextLength: Int
    public var hotWindowTokens: Int
    public var coldBlockTokens: Int
    public var coldBudgetTokens: Int
    public var maxColdBudgetTokens: Int
    public var selectorPolicy: TurboQuantColdSelectorPolicy
    public var selectorHints: [TurboQuantColdSelectorHint]

    public init(
        label: String = "hybrid-selector",
        contextLength: Int,
        hotWindowTokens: Int = 8192,
        coldBlockTokens: Int = 1024,
        coldBudgetTokens: Int = 4096,
        maxColdBudgetTokens: Int = 8192,
        selectorPolicy: TurboQuantColdSelectorPolicy = .automatic,
        selectorHints: [TurboQuantColdSelectorHint] = []
    ) {
        self.label = label
        self.contextLength = max(1, contextLength)
        self.hotWindowTokens = max(1, hotWindowTokens)
        self.coldBlockTokens = max(1, coldBlockTokens)
        self.coldBudgetTokens = max(0, coldBudgetTokens)
        self.maxColdBudgetTokens = max(self.coldBudgetTokens, maxColdBudgetTokens)
        self.selectorPolicy = selectorPolicy
        self.selectorHints = selectorHints.filter { !$0.isEmpty }
    }
}

/// Result of one benchmark case. `Codable` so callers can emit JSON to read off-device.
public struct TurboQuantBenchKernelFlags: Codable, Sendable {
    public var tqCoopEnabled: Bool
    public var blockTokenSize: Int?
    public var gqaSpecialization: String?
    public var outputDType: String

    public init(
        tqCoopEnabled: Bool,
        blockTokenSize: Int? = nil,
        gqaSpecialization: String? = nil,
        outputDType: String
    ) {
        self.tqCoopEnabled = tqCoopEnabled
        self.blockTokenSize = blockTokenSize
        self.gqaSpecialization = gqaSpecialization
        self.outputDType = outputDType
    }
}

public struct TurboQuantBenchResult: Codable, Sendable {
    public enum Status: String, Codable, Sendable {
        case ok
        case skipped
        case failed
    }

    public var label: String
    public var scheme: String
    public var codec: String
    public var contextLength: Int
    public var status: Status
    /// Skip reason or error description when `status != .ok`.
    public var detail: String?
    public var route: String
    public var selectedPath: String
    public var runtimeMode: String
    public var requestedRuntimeMode: String
    public var resolvedRuntimeMode: String
    public var keyPrecision: String
    public var valuePrecision: String
    public var precisionPolicy: TurboQuantKVPrecisionPolicy?
    public var backend: String
    public var groupSize: Int
    public var scaleBiasBytes: Int
    public var nativeDiagnostics: [Int]?
    public var fallbackReason: String?
    public var swiftMetalCompressedTokensPerSecond: Double?
    public var swiftMetalCompressedP95TokensPerSecond: Double?
    /// Native MLX compressed attention throughput divided by the Swift Metal compressed path.
    public var nativeSpeedRatioToSwiftMetal: Double?
    public var nativePerfGateMinimumContextLength: Int?
    public var nativePerfGateRequiredSpeedup: Double?
    public var nativePerfGatePassed: Bool?
    public var kernelFlags: TurboQuantBenchKernelFlags?
    public var sparseVEnabled: Bool
    public var sparseVThreshold: Float?
    public var sparseVSkipRatio: Double
    public var boundaryProtectedLayerCount: Int
    public var boundaryProtectionReason: String?
    public var hotTokens: Int?
    public var coldBlockCount: Int?
    public var selectedColdTokens: Int?
    public var selectedBudgetedColdTokens: Int?
    public var anchorColdTokens: Int?
    public var anchorOverflowTokens: Int?
    public var coldBudgetTokens: Int?
    public var maxColdBudgetTokens: Int?
    public var selectorConfidence: Double?
    public var selectorInitialConfidence: Double?
    public var selectorFinalConfidence: Double?
    public var selectorEscalation: String?
    public var selectorReasonFlags: [String]?
    public var fullScanFallbackCount: Int?

    // Throughput — decode rows per second at the measured query length.
    public var compressedTokensPerSecond: Double
    public var plainTokensPerSecond: Double
    public var compressedP95TokensPerSecond: Double
    public var plainP95TokensPerSecond: Double
    /// compressed ÷ plain. ≥ 1 ⟹ compressed at least as fast (the "close to FP16" target).
    public var speedRatioToPlain: Double

    // Quality vs the plain FP16 reference output.
    public var cosineSimilarity: Double
    public var maxAbsErrorP95: Double
    public var finite: Bool

    // Memory.
    public var compressedKVBytes: Int
    public var compressedKeyBytes: Int
    public var compressedValueBytes: Int
    public var decodedActiveKVBytes: Int
    public var plainKVBytes: Int
    /// plain ÷ compressed. > 1 ⟹ compressed smaller (the context-unlock metric).
    public var memoryReductionRatio: Double

    public init(
        label: String,
        scheme: String,
        codec: String = TurboQuantKVCodec.polarQJL.rawValue,
        contextLength: Int,
        status: Status,
        detail: String?,
        route: String = "unavailable",
        selectedPath: String? = nil,
        runtimeMode: String = "capacityTurboQuant",
        requestedRuntimeMode: String? = nil,
        resolvedRuntimeMode: String? = nil,
        keyPrecision: String = "turbo8",
        valuePrecision: String = "turbo4v2",
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        backend: String = "unavailable",
        groupSize: Int = 64,
        scaleBiasBytes: Int = 0,
        nativeDiagnostics: [Int]? = nil,
        fallbackReason: String? = nil,
        swiftMetalCompressedTokensPerSecond: Double? = nil,
        swiftMetalCompressedP95TokensPerSecond: Double? = nil,
        nativeSpeedRatioToSwiftMetal: Double? = nil,
        nativePerfGateMinimumContextLength: Int? = nil,
        nativePerfGateRequiredSpeedup: Double? = nil,
        nativePerfGatePassed: Bool? = nil,
        kernelFlags: TurboQuantBenchKernelFlags? = nil,
        sparseVEnabled: Bool = false,
        sparseVThreshold: Float? = nil,
        sparseVSkipRatio: Double = 0,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        hotTokens: Int? = nil,
        coldBlockCount: Int? = nil,
        selectedColdTokens: Int? = nil,
        selectedBudgetedColdTokens: Int? = nil,
        anchorColdTokens: Int? = nil,
        anchorOverflowTokens: Int? = nil,
        coldBudgetTokens: Int? = nil,
        maxColdBudgetTokens: Int? = nil,
        selectorConfidence: Double? = nil,
        selectorInitialConfidence: Double? = nil,
        selectorFinalConfidence: Double? = nil,
        selectorEscalation: String? = nil,
        selectorReasonFlags: [String]? = nil,
        fullScanFallbackCount: Int? = nil,
        compressedTokensPerSecond: Double,
        plainTokensPerSecond: Double,
        compressedP95TokensPerSecond: Double? = nil,
        plainP95TokensPerSecond: Double? = nil,
        speedRatioToPlain: Double,
        cosineSimilarity: Double,
        maxAbsErrorP95: Double,
        finite: Bool,
        compressedKVBytes: Int,
        compressedKeyBytes: Int? = nil,
        compressedValueBytes: Int? = nil,
        decodedActiveKVBytes: Int = 0,
        plainKVBytes: Int,
        memoryReductionRatio: Double
    ) {
        self.label = label
        self.scheme = scheme
        self.codec = codec
        self.contextLength = contextLength
        self.status = status
        self.detail = detail
        self.route = route
        self.selectedPath = selectedPath ?? route
        self.runtimeMode = runtimeMode
        self.requestedRuntimeMode = requestedRuntimeMode ?? runtimeMode
        self.resolvedRuntimeMode = resolvedRuntimeMode ?? runtimeMode
        self.keyPrecision = keyPrecision
        self.valuePrecision = valuePrecision
        self.precisionPolicy = precisionPolicy
        self.backend = backend
        self.groupSize = max(1, groupSize)
        self.scaleBiasBytes = max(0, scaleBiasBytes)
        self.nativeDiagnostics = nativeDiagnostics
        self.fallbackReason = fallbackReason
        self.swiftMetalCompressedTokensPerSecond = swiftMetalCompressedTokensPerSecond
        self.swiftMetalCompressedP95TokensPerSecond = swiftMetalCompressedP95TokensPerSecond
        self.nativeSpeedRatioToSwiftMetal = nativeSpeedRatioToSwiftMetal
        self.nativePerfGateMinimumContextLength = nativePerfGateMinimumContextLength
        self.nativePerfGateRequiredSpeedup = nativePerfGateRequiredSpeedup
        self.nativePerfGatePassed = nativePerfGatePassed
        self.kernelFlags = kernelFlags
        self.sparseVEnabled = sparseVEnabled
        self.sparseVThreshold = sparseVThreshold
        self.sparseVSkipRatio = max(0, min(1, sparseVSkipRatio))
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.hotTokens = hotTokens
        self.coldBlockCount = coldBlockCount
        self.selectedColdTokens = selectedColdTokens
        self.selectedBudgetedColdTokens = selectedBudgetedColdTokens
        self.anchorColdTokens = anchorColdTokens
        self.anchorOverflowTokens = anchorOverflowTokens
        self.coldBudgetTokens = coldBudgetTokens
        self.maxColdBudgetTokens = maxColdBudgetTokens
        self.selectorConfidence = selectorConfidence
        self.selectorInitialConfidence = selectorInitialConfidence
        self.selectorFinalConfidence = selectorFinalConfidence
        self.selectorEscalation = selectorEscalation
        self.selectorReasonFlags = selectorReasonFlags
        self.fullScanFallbackCount = fullScanFallbackCount
        self.compressedTokensPerSecond = compressedTokensPerSecond
        self.plainTokensPerSecond = plainTokensPerSecond
        self.compressedP95TokensPerSecond = compressedP95TokensPerSecond ?? compressedTokensPerSecond
        self.plainP95TokensPerSecond = plainP95TokensPerSecond ?? plainTokensPerSecond
        self.speedRatioToPlain = speedRatioToPlain
        self.cosineSimilarity = cosineSimilarity
        self.maxAbsErrorP95 = maxAbsErrorP95
        self.finite = finite
        self.compressedKVBytes = compressedKVBytes
        self.compressedKeyBytes = compressedKeyBytes ?? compressedKVBytes / 2
        self.compressedValueBytes = compressedValueBytes ?? compressedKVBytes - self.compressedKeyBytes
        self.decodedActiveKVBytes = max(0, decodedActiveKVBytes)
        self.plainKVBytes = plainKVBytes
        self.memoryReductionRatio = memoryReductionRatio
    }

    private enum CodingKeys: String, CodingKey {
        case label
        case scheme
        case codec
        case contextLength
        case status
        case detail
        case route
        case selectedPath
        case runtimeMode
        case requestedRuntimeMode
        case resolvedRuntimeMode
        case keyPrecision
        case valuePrecision
        case precisionPolicy
        case backend
        case groupSize
        case scaleBiasBytes
        case nativeDiagnostics
        case fallbackReason
        case swiftMetalCompressedTokensPerSecond
        case swiftMetalCompressedP95TokensPerSecond
        case nativeSpeedRatioToSwiftMetal
        case nativePerfGateMinimumContextLength
        case nativePerfGateRequiredSpeedup
        case nativePerfGatePassed
        case kernelFlags
        case sparseVEnabled
        case sparseVThreshold
        case sparseVSkipRatio
        case boundaryProtectedLayerCount
        case boundaryProtectionReason
        case hotTokens
        case coldBlockCount
        case selectedColdTokens
        case selectedBudgetedColdTokens
        case anchorColdTokens
        case anchorOverflowTokens
        case coldBudgetTokens
        case maxColdBudgetTokens
        case selectorConfidence
        case selectorInitialConfidence
        case selectorFinalConfidence
        case selectorEscalation
        case selectorReasonFlags
        case fullScanFallbackCount
        case compressedTokensPerSecond
        case plainTokensPerSecond
        case compressedP95TokensPerSecond
        case plainP95TokensPerSecond
        case speedRatioToPlain
        case cosineSimilarity
        case maxAbsErrorP95
        case finite
        case compressedKVBytes
        case compressedKeyBytes
        case compressedValueBytes
        case decodedActiveKVBytes
        case plainKVBytes
        case memoryReductionRatio
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let status = try container.decode(Status.self, forKey: .status)
        self.init(
            label: try container.decode(String.self, forKey: .label),
            scheme: try container.decode(String.self, forKey: .scheme),
            codec: try container.decodeIfPresent(String.self, forKey: .codec)
                ?? TurboQuantKVCodec.polarQJL.rawValue,
            contextLength: try container.decode(Int.self, forKey: .contextLength),
            status: status,
            detail: try container.decodeIfPresent(String.self, forKey: .detail),
            route: try container.decodeIfPresent(String.self, forKey: .route)
                ?? (status == .ok ? "compressedFused" : "unavailable"),
            selectedPath: try container.decodeIfPresent(String.self, forKey: .selectedPath),
            runtimeMode: try container.decodeIfPresent(String.self, forKey: .runtimeMode)
                ?? "capacityTurboQuant",
            requestedRuntimeMode: try container.decodeIfPresent(
                String.self,
                forKey: .requestedRuntimeMode
            ),
            resolvedRuntimeMode: try container.decodeIfPresent(
                String.self,
                forKey: .resolvedRuntimeMode
            ),
            keyPrecision: try container.decodeIfPresent(String.self, forKey: .keyPrecision)
                ?? "turbo8",
            valuePrecision: try container.decodeIfPresent(String.self, forKey: .valuePrecision)
                ?? "turbo4v2",
            precisionPolicy: try container.decodeIfPresent(
                TurboQuantKVPrecisionPolicy.self,
                forKey: .precisionPolicy
            ),
            backend: try container.decodeIfPresent(String.self, forKey: .backend)
                ?? (status == .ok ? "swiftMetalKernel" : "unavailable"),
            groupSize: try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64,
            scaleBiasBytes: try container.decodeIfPresent(Int.self, forKey: .scaleBiasBytes) ?? 0,
            nativeDiagnostics: try container.decodeIfPresent(
                [Int].self,
                forKey: .nativeDiagnostics
            ),
            fallbackReason: try container.decodeIfPresent(String.self, forKey: .fallbackReason),
            swiftMetalCompressedTokensPerSecond: try container.decodeIfPresent(
                Double.self,
                forKey: .swiftMetalCompressedTokensPerSecond
            ),
            swiftMetalCompressedP95TokensPerSecond: try container.decodeIfPresent(
                Double.self,
                forKey: .swiftMetalCompressedP95TokensPerSecond
            ),
            nativeSpeedRatioToSwiftMetal: try container.decodeIfPresent(
                Double.self,
                forKey: .nativeSpeedRatioToSwiftMetal
            ),
            nativePerfGateMinimumContextLength: try container.decodeIfPresent(
                Int.self,
                forKey: .nativePerfGateMinimumContextLength
            ),
            nativePerfGateRequiredSpeedup: try container.decodeIfPresent(
                Double.self,
                forKey: .nativePerfGateRequiredSpeedup
            ),
            nativePerfGatePassed: try container.decodeIfPresent(
                Bool.self,
                forKey: .nativePerfGatePassed
            ),
            kernelFlags: try container.decodeIfPresent(
                TurboQuantBenchKernelFlags.self,
                forKey: .kernelFlags
            ),
            sparseVEnabled: try container.decodeIfPresent(Bool.self, forKey: .sparseVEnabled)
                ?? false,
            sparseVThreshold: try container.decodeIfPresent(
                Float.self,
                forKey: .sparseVThreshold
            ),
            sparseVSkipRatio: try container.decodeIfPresent(
                Double.self,
                forKey: .sparseVSkipRatio
            ) ?? 0,
            boundaryProtectedLayerCount: try container.decodeIfPresent(
                Int.self,
                forKey: .boundaryProtectedLayerCount
            ) ?? 0,
            boundaryProtectionReason: try container.decodeIfPresent(
                String.self,
                forKey: .boundaryProtectionReason
            ),
            hotTokens: try container.decodeIfPresent(Int.self, forKey: .hotTokens),
            coldBlockCount: try container.decodeIfPresent(Int.self, forKey: .coldBlockCount),
            selectedColdTokens: try container.decodeIfPresent(
                Int.self,
                forKey: .selectedColdTokens
            ),
            selectedBudgetedColdTokens: try container.decodeIfPresent(
                Int.self,
                forKey: .selectedBudgetedColdTokens
            ),
            anchorColdTokens: try container.decodeIfPresent(Int.self, forKey: .anchorColdTokens),
            anchorOverflowTokens: try container.decodeIfPresent(
                Int.self,
                forKey: .anchorOverflowTokens
            ),
            coldBudgetTokens: try container.decodeIfPresent(Int.self, forKey: .coldBudgetTokens),
            maxColdBudgetTokens: try container.decodeIfPresent(
                Int.self,
                forKey: .maxColdBudgetTokens
            ),
            selectorConfidence: try container.decodeIfPresent(
                Double.self,
                forKey: .selectorConfidence
            ),
            selectorInitialConfidence: try container.decodeIfPresent(
                Double.self,
                forKey: .selectorInitialConfidence
            ),
            selectorFinalConfidence: try container.decodeIfPresent(
                Double.self,
                forKey: .selectorFinalConfidence
            ),
            selectorEscalation: try container.decodeIfPresent(
                String.self,
                forKey: .selectorEscalation
            ),
            selectorReasonFlags: try container.decodeIfPresent(
                [String].self,
                forKey: .selectorReasonFlags
            ),
            fullScanFallbackCount: try container.decodeIfPresent(
                Int.self,
                forKey: .fullScanFallbackCount
            ),
            compressedTokensPerSecond: try container.decode(
                Double.self,
                forKey: .compressedTokensPerSecond
            ),
            plainTokensPerSecond: try container.decode(Double.self, forKey: .plainTokensPerSecond),
            compressedP95TokensPerSecond: try container.decodeIfPresent(
                Double.self,
                forKey: .compressedP95TokensPerSecond
            ),
            plainP95TokensPerSecond: try container.decodeIfPresent(
                Double.self,
                forKey: .plainP95TokensPerSecond
            ),
            speedRatioToPlain: try container.decode(Double.self, forKey: .speedRatioToPlain),
            cosineSimilarity: try container.decode(Double.self, forKey: .cosineSimilarity),
            maxAbsErrorP95: try container.decode(Double.self, forKey: .maxAbsErrorP95),
            finite: try container.decode(Bool.self, forKey: .finite),
            compressedKVBytes: try container.decode(Int.self, forKey: .compressedKVBytes),
            compressedKeyBytes: try container.decodeIfPresent(
                Int.self,
                forKey: .compressedKeyBytes
            ),
            compressedValueBytes: try container.decodeIfPresent(
                Int.self,
                forKey: .compressedValueBytes
            ),
            decodedActiveKVBytes: try container.decodeIfPresent(
                Int.self,
                forKey: .decodedActiveKVBytes
            ) ?? 0,
            plainKVBytes: try container.decode(Int.self, forKey: .plainKVBytes),
            memoryReductionRatio: try container.decode(Double.self, forKey: .memoryReductionRatio)
        )
    }

    fileprivate static func skipped(_ c: TurboQuantBenchCase, _ detail: String) -> TurboQuantBenchResult {
        TurboQuantBenchResult(
            label: c.label, scheme: c.scheme.rawValue, codec: c.codec.rawValue,
            contextLength: c.contextLength,
            status: .skipped, detail: detail,
            route: "unavailable", runtimeMode: c.runtimeMode.rawValue,
            requestedRuntimeMode: c.runtimeMode.rawValue,
            resolvedRuntimeMode: "unavailable",
            keyPrecision: c.precisionPolicy?.key.rawValue ?? "unavailable",
            valuePrecision: c.precisionPolicy?.value.rawValue ?? "unavailable",
            precisionPolicy: c.precisionPolicy,
            backend: "unavailable",
            groupSize: c.codec == .affineInt4
                ? TurboQuantKVCodec.affineInt4DefaultGroupSize : 64,
            kernelFlags: nil,
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
    public static let nativePerfGateMinimumContextLength = 32_768
    public static let nativePerfGateRequiredSpeedup = 2.0

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

        let precisionPolicy =
            benchCase.precisionPolicy
            ?? profile.turboQuant.precisionPolicy
            ?? TurboQuantKVPrecisionPolicy.legacy(
                preset: precision.recommendedScheme.preset,
                valueBits: precision.valueBits
            )
        let preset = precisionPolicy.compressedKeyPreset
        let valueBits = precisionPolicy.resolvedValueBits ?? precision.valueBits
        let resolvedRuntimeMode: TurboQuantRuntimeMode =
            switch benchCase.runtimeMode {
            case .auto:
                benchCase.contextLength <= 16_384 ? .rawPreferred : .throughputTurboQuant
            case .rawPreferred:
                .rawPreferred
            case .throughputTurboQuant:
                .throughputTurboQuant
            case .capacityTurboQuant:
                .capacityTurboQuant
            }
        let sparseVThreshold = benchCase.sparseValuePolicy.resolvedThreshold(
            runtimeMode: resolvedRuntimeMode,
            contextLength: benchCase.contextLength
        )
        let boundaryProtectedLayerCount = precisionPolicy
            .protectedBoundaryLayerIndexes(layerCount: profile.modelFingerprint?.layerCount ?? 0)
            .count
        let boundaryProtectionReason =
            boundaryProtectedLayerCount > 0
            ? "rawKV boundary protection for low-bit V or compressed K policy"
            : nil
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

        if benchCase.codec == .affineInt4 {
            let groupSize =
                profile.affineInt4?.groupSize ?? TurboQuantKVCodec.affineInt4DefaultGroupSize
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
                return .skipped(
                    benchCase,
                    "native affine int4 SDPA unsupported for this shape")
            }
            do {
                let plain = try timedMedianSeconds(
                    iterations: iterations,
                    warmup: warmupIterations
                ) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries,
                        keys: keys,
                        values: values,
                        scale: scale,
                        mask: .causal
                    )
                }
                let measured = try timedMedianSeconds(
                    iterations: iterations,
                    warmup: warmupIterations
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
                let quality = reconstructionQuality(
                    candidate: measured.output,
                    reference: plain.output
                )
                let compressedKeyBytes =
                    qKeys.wq.nbytes + qKeys.scales.nbytes + (qKeys.biases?.nbytes ?? 0)
                let compressedValueBytes =
                    qValues.wq.nbytes + qValues.scales.nbytes + (qValues.biases?.nbytes ?? 0)
                let scaleBiasBytes =
                    qKeys.scales.nbytes + (qKeys.biases?.nbytes ?? 0)
                    + qValues.scales.nbytes + (qValues.biases?.nbytes ?? 0)
                let compressedBytes = compressedKeyBytes + compressedValueBytes
                let plainBytes = keys.nbytes + values.nbytes
                let rows = Double(qLen)
                let affineTPS = rows / Swift.max(measured.median, Double.leastNonzeroMagnitude)
                let plainTPS = rows / Swift.max(plain.median, Double.leastNonzeroMagnitude)
                let affineP95TPS = rows / Swift.max(measured.p95, Double.leastNonzeroMagnitude)
                let plainP95TPS = rows / Swift.max(plain.p95, Double.leastNonzeroMagnitude)
                return TurboQuantBenchResult(
                    label: benchCase.label,
                    scheme: benchCase.scheme.rawValue,
                    codec: benchCase.codec.rawValue,
                    contextLength: ctx,
                    status: .ok,
                    detail: nil,
                    route: TurboQuantAttentionPath.affineInt4Native.rawValue,
                    selectedPath: TurboQuantAttentionPath.affineInt4Native.rawValue,
                    runtimeMode: "nativeAffineInt4",
                    requestedRuntimeMode: benchCase.runtimeMode.rawValue,
                    resolvedRuntimeMode: "nativeAffineInt4",
                    keyPrecision: "affineInt4",
                    valuePrecision: "affineInt4",
                    precisionPolicy: precisionPolicy,
                    backend: TurboQuantBackend.mlxPacked.rawValue,
                    groupSize: groupSize,
                    scaleBiasBytes: scaleBiasBytes,
                    nativeDiagnostics: [groupSize, TurboQuantKVCodec.affineInt4Bits],
                    fallbackReason: nil,
                    kernelFlags: TurboQuantBenchKernelFlags(
                        tqCoopEnabled: false,
                        blockTokenSize: nil,
                        gqaSpecialization: queryHeads % kvHeads == 0 && queryHeads / kvHeads > 1
                            ? "gqa\(queryHeads / kvHeads)" : nil,
                        outputDType: "\(measured.output.dtype)"
                    ),
                    compressedTokensPerSecond: affineTPS,
                    plainTokensPerSecond: plainTPS,
                    compressedP95TokensPerSecond: affineP95TPS,
                    plainP95TokensPerSecond: plainP95TPS,
                    speedRatioToPlain: affineTPS / Swift.max(plainTPS, Double.leastNonzeroMagnitude),
                    cosineSimilarity: quality.cosine,
                    maxAbsErrorP95: quality.maxAbsP95,
                    finite: quality.finite,
                    compressedKVBytes: compressedBytes,
                    compressedKeyBytes: compressedKeyBytes,
                    compressedValueBytes: compressedValueBytes,
                    decodedActiveKVBytes: 0,
                    plainKVBytes: plainBytes,
                    memoryReductionRatio: Double(plainBytes) / Double(Swift.max(1, compressedBytes))
                )
            } catch {
                return .failed(benchCase, String(describing: error))
            }
        }

        let cache = TurboQuantKVCache(
            preset: preset,
            groupSize: precision.groupSize,
            backend: precision.backend,
            optimizationPolicy: precision.optimizationPolicy,
            fallbackPolicy: precision.turboQuant.fallbackPolicy,
            valueBits: valueBits,
            precisionPolicy: precisionPolicy,
            requestedRuntimeMode: benchCase.runtimeMode,
            resolvedRuntimeMode: resolvedRuntimeMode
        )
        if resolvedRuntimeMode == .capacityTurboQuant,
            !cache.supportsCompressedAttention(
                queries: queries, keys: keys, values: values, mask: .causal)
        {
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
            seed: 0x9E37_79B9_7F4A_7C15 ^ 0xD1B5_4A32_D192_ED03, valueBits: valueBits)

        do {
            let compressedKeys = try turboQuantMetalEncodeAttention(
                keys, configuration: keyConfiguration, capacity: ctx, logicalLength: ctx)
            let compressedValues = try turboQuantMetalEncodeAttention(
                values, configuration: valueConfiguration, capacity: ctx, logicalLength: ctx)
            let preferOnline = cache.prefersOnlineFusedAttention
            let kernelProfile = cache.attentionDiagnostics.selectedKernelProfile
            let nativeCapabilities = TurboQuantKernelAvailability.current.attentionCapabilities
            let compressedRuntimeMode: Bool =
                switch resolvedRuntimeMode {
                case .capacityTurboQuant, .auto:
                    true
                case .rawPreferred, .throughputTurboQuant:
                    false
                }
            let useNativeCompressed =
                compressedRuntimeMode && nativeCapabilities.nativeCompressedAttention == true
                && (sparseVThreshold == nil || nativeCapabilities.nativeSparseVSupport == true)

            let plain = try timedMedianSeconds(iterations: iterations, warmup: warmupIterations) {
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: .causal)
            }
            let measured = try timedMedianSeconds(iterations: iterations, warmup: warmupIterations) {
                switch resolvedRuntimeMode {
                case .rawPreferred:
                    return MLXFast.scaledDotProductAttention(
                        queries: queries,
                        keys: keys,
                        values: values,
                        scale: scale,
                        mask: .causal
                    )
                case .throughputTurboQuant:
                    return MLXFast.scaledDotProductAttention(
                        queries: queries,
                        keys: keys,
                        values: values,
                        scale: scale,
                        mask: .causal
                    )
                case .capacityTurboQuant, .auto:
                    if useNativeCompressed {
                        return try turboQuantNativeScaledDotProductAttention(
                            queries: queries,
                            keyCode: compressedKeys,
                            valueCode: compressedValues,
                            options: TurboQuantNativeAttentionOptions(
                                scale: scale,
                                causal: true,
                                sparseVThreshold: sparseVThreshold ?? 0,
                                backendVersion: nativeCapabilities.nativeBackendVersion
                                    ?? TurboQuantNativeAttentionOptions.backendVersion
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
                        kernelProfile: kernelProfile,
                        blockParallelTokenBlockSize: nil,
                        sparseVThreshold: sparseVThreshold
                    )
                }
            }
            let swiftMetalComparison =
                useNativeCompressed
                ? try timedMedianSeconds(iterations: iterations, warmup: warmupIterations) {
                    try turboQuantMetalScaledDotProductAttention(
                        queries: queries,
                        keyCode: compressedKeys,
                        valueCode: compressedValues,
                        scale: scale,
                        mask: .causal,
                        preferOnlineFused: preferOnline,
                        kernelProfile: kernelProfile,
                        blockParallelTokenBlockSize: nil,
                        sparseVThreshold: sparseVThreshold
                    )
                } : nil
            let sparseDiagnostics =
                try? turboQuantMetalScaledDotProductAttentionWithDiagnostics(
                    queries: queries,
                    keyCode: compressedKeys,
                    valueCode: compressedValues,
                    scale: scale,
                    mask: .causal,
                    preferOnlineFused: preferOnline,
                    kernelProfile: kernelProfile,
                    blockParallelTokenBlockSize: nil,
                    sparseVThreshold: sparseVThreshold
                ).sparseValueDiagnostics
            var nativeDiagnostics: [Int]?
            if nativeCapabilities.nativeCompressedAttention == true,
                sparseVThreshold == nil || nativeCapabilities.nativeSparseVSupport == true,
                let diagnostics = try? turboQuantNativeScaledDotProductAttentionWithDiagnostics(
                    queries: queries,
                    keyCode: compressedKeys,
                    valueCode: compressedValues,
                    options: TurboQuantNativeAttentionOptions(
                        scale: scale,
                        causal: true,
                        sparseVThreshold: sparseVThreshold ?? 0,
                        diagnostics: nativeCapabilities.nativeDiagnosticsSupport == true,
                        backendVersion: nativeCapabilities.nativeBackendVersion
                            ?? TurboQuantNativeAttentionOptions.backendVersion
                    )
                ).diagnostics
            {
                nativeDiagnostics = [
                    diagnostics.backendVersion,
                    diagnostics.kernelKind,
                    diagnostics.activeBlocks,
                    diagnostics.blockTokens,
                    diagnostics.sparseSkippedTokens,
                    diagnostics.sparseTotalTokens,
                    diagnostics.fallbackCode,
                    diagnostics.flags,
                ]
            }

            let quality = reconstructionQuality(
                candidate: measured.output, reference: plain.output)
            let compressedKeyBytes = compressedKeys.storageByteCount
            let compressedValueBytes = compressedValues.storageByteCount
            let compressedBytes = compressedKeyBytes + compressedValueBytes
            let plainBytes = keys.nbytes + values.nbytes
            let rows = Double(qLen)
            let compressedTPS = rows / Swift.max(measured.median, Double.leastNonzeroMagnitude)
            let plainTPS = rows / Swift.max(plain.median, Double.leastNonzeroMagnitude)
            let compressedP95TPS = rows / Swift.max(measured.p95, Double.leastNonzeroMagnitude)
            let plainP95TPS = rows / Swift.max(plain.p95, Double.leastNonzeroMagnitude)
            let swiftMetalCompressedTPS = swiftMetalComparison.map {
                rows / Swift.max($0.median, Double.leastNonzeroMagnitude)
            }
            let swiftMetalCompressedP95TPS = swiftMetalComparison.map {
                rows / Swift.max($0.p95, Double.leastNonzeroMagnitude)
            }
            let nativeSpeedRatioToSwiftMetal = swiftMetalComparison.map {
                $0.median / Swift.max(measured.median, Double.leastNonzeroMagnitude)
            }
            let nativePerfGateApplies =
                useNativeCompressed && ctx >= Self.nativePerfGateMinimumContextLength
            let nativePerfGatePassed = nativePerfGateApplies
                ? (nativeSpeedRatioToSwiftMetal ?? 0) >= Self.nativePerfGateRequiredSpeedup
                : nil
            let route =
                switch resolvedRuntimeMode {
                case .rawPreferred:
                    "rawSDPA"
                case .throughputTurboQuant:
                    "throughputTurboQuantNativeSDPA"
                case .capacityTurboQuant, .auto:
                    "capacityTurboQuantCompressed"
                }
            let benchmarkBackend =
                switch resolvedRuntimeMode {
                case .rawPreferred, .throughputTurboQuant:
                    "rawSDPA"
                case .capacityTurboQuant, .auto:
                    useNativeCompressed ? "nativeMLX" : "swiftMetal"
                }

            return TurboQuantBenchResult(
                label: benchCase.label,
                scheme: benchCase.scheme.rawValue,
                codec: benchCase.codec.rawValue,
                contextLength: ctx,
                status: .ok,
                detail: nil,
                route: route,
                selectedPath: route,
                runtimeMode: resolvedRuntimeMode.rawValue,
                requestedRuntimeMode: benchCase.runtimeMode.rawValue,
                resolvedRuntimeMode: resolvedRuntimeMode.rawValue,
                keyPrecision: precisionPolicy.key.rawValue,
                valuePrecision: precisionPolicy.value.rawValue,
                precisionPolicy: precisionPolicy,
                backend: benchmarkBackend,
                groupSize: precision.groupSize,
                scaleBiasBytes: 0,
                nativeDiagnostics: nativeDiagnostics,
                fallbackReason: nativeCapabilities.nativeFallbackReason,
                swiftMetalCompressedTokensPerSecond: swiftMetalCompressedTPS,
                swiftMetalCompressedP95TokensPerSecond: swiftMetalCompressedP95TPS,
                nativeSpeedRatioToSwiftMetal: nativeSpeedRatioToSwiftMetal,
                nativePerfGateMinimumContextLength: useNativeCompressed
                    ? Self.nativePerfGateMinimumContextLength : nil,
                nativePerfGateRequiredSpeedup: useNativeCompressed
                    ? Self.nativePerfGateRequiredSpeedup : nil,
                nativePerfGatePassed: nativePerfGatePassed,
                kernelFlags: TurboQuantBenchKernelFlags(
                    tqCoopEnabled: ProcessInfo.processInfo.environment["TQ_COOP"] == "1",
                    blockTokenSize: nil,
                    gqaSpecialization: queryHeads % kvHeads == 0 && queryHeads / kvHeads > 1
                        ? "gqa\(queryHeads / kvHeads)" : nil,
                    outputDType: "\(measured.output.dtype)"
                ),
                sparseVEnabled: sparseDiagnostics?.enabled ?? false,
                sparseVThreshold: sparseDiagnostics?.threshold,
                sparseVSkipRatio: sparseDiagnostics?.skipRatio ?? 0,
                boundaryProtectedLayerCount: boundaryProtectedLayerCount,
                boundaryProtectionReason: boundaryProtectionReason,
                compressedTokensPerSecond: compressedTPS,
                plainTokensPerSecond: plainTPS,
                compressedP95TokensPerSecond: compressedP95TPS,
                plainP95TokensPerSecond: plainP95TPS,
                speedRatioToPlain: compressedTPS
                    / Swift.max(plainTPS, Double.leastNonzeroMagnitude),
                cosineSimilarity: quality.cosine,
                maxAbsErrorP95: quality.maxAbsP95,
                finite: quality.finite,
                compressedKVBytes: compressedBytes,
                compressedKeyBytes: compressedKeyBytes,
                compressedValueBytes: compressedValueBytes,
                decodedActiveKVBytes: resolvedRuntimeMode == .throughputTurboQuant ? plainBytes : 0,
                plainKVBytes: plainBytes,
                memoryReductionRatio: Double(plainBytes) / Double(Swift.max(1, compressedBytes))
            )
        } catch {
            return .failed(benchCase, String(describing: error))
        }
    }

    public static func measureHybridSelector(
        _ benchCase: TurboQuantHybridBenchCase,
        iterations: Int = 16,
        warmupIterations: Int = 4
    ) -> TurboQuantBenchResult {
        let descriptors = hybridSelectorDescriptors(for: benchCase)
        var lastSelection = TurboQuantColdSelection.empty

        func runSelection() -> TurboQuantColdSelection {
            HybridTurboQuantKVCache.selectColdBlocks(
                descriptors: descriptors,
                mode: .selected,
                budgetTokens: benchCase.coldBudgetTokens,
                maxBudgetTokens: benchCase.maxColdBudgetTokens,
                selectorPolicy: benchCase.selectorPolicy
            )
        }

        for _ in 0 ..< max(0, warmupIterations) {
            lastSelection = runSelection()
        }

        let measured = max(1, iterations)
        var samples = [Double]()
        samples.reserveCapacity(measured)
        for _ in 0 ..< measured {
            let start = Date.timeIntervalSinceReferenceDate
            lastSelection = runSelection()
            samples.append(Date.timeIntervalSinceReferenceDate - start)
        }
        samples.sort()
        let median = samples[samples.count / 2]
        let p95Index = Swift.min(
            samples.count - 1,
            Swift.max(0, Int((0.95 * Double(samples.count)).rounded(.up)) - 1)
        )
        let p95 = samples[p95Index]
        let selectedTokens = max(1, lastSelection.selectedTokenCount)
        let selectorTPS = Double(selectedTokens) / Swift.max(median, Double.leastNonzeroMagnitude)
        let selectorP95TPS = Double(selectedTokens) / Swift.max(p95, Double.leastNonzeroMagnitude)
        let coldTokens = max(0, benchCase.contextLength - benchCase.hotWindowTokens)
        let compressedBytes = coldTokens * 2

        return TurboQuantBenchResult(
            label: benchCase.label,
            scheme: "hybridSelector",
            codec: TurboQuantKVCodec.polarQJL.rawValue,
            contextLength: benchCase.contextLength,
            status: .ok,
            detail: nil,
            route: "hybridTurboQuant",
            selectedPath: "hybrid_selected_cold",
            runtimeMode: "capacityTurboQuant",
            requestedRuntimeMode: "capacityTurboQuant",
            resolvedRuntimeMode: "capacityTurboQuant",
            keyPrecision: "selectorOnly",
            valuePrecision: "selectorOnly",
            precisionPolicy: nil,
            backend: "selectorOnly",
            groupSize: 64,
            hotTokens: min(benchCase.hotWindowTokens, benchCase.contextLength),
            coldBlockCount: descriptors.count,
            selectedColdTokens: lastSelection.selectedTokenCount,
            selectedBudgetedColdTokens: lastSelection.budgetedTokenCount,
            anchorColdTokens: lastSelection.anchorTokenCount,
            anchorOverflowTokens: lastSelection.anchorOverflowTokens,
            coldBudgetTokens: lastSelection.selectedTokenBudget,
            maxColdBudgetTokens: benchCase.maxColdBudgetTokens,
            selectorConfidence: Double(lastSelection.confidence),
            selectorInitialConfidence: Double(lastSelection.initialConfidence),
            selectorFinalConfidence: Double(lastSelection.finalConfidence),
            selectorEscalation: lastSelection.selectorEscalation.rawValue,
            selectorReasonFlags: lastSelection.reasonFlags,
            fullScanFallbackCount: lastSelection.selectorEscalation == .exhaustive ? 1 : 0,
            compressedTokensPerSecond: selectorTPS,
            plainTokensPerSecond: selectorTPS,
            compressedP95TokensPerSecond: selectorP95TPS,
            plainP95TokensPerSecond: selectorP95TPS,
            speedRatioToPlain: 1,
            cosineSimilarity: 1,
            maxAbsErrorP95: 0,
            finite: true,
            compressedKVBytes: compressedBytes,
            compressedKeyBytes: compressedBytes / 2,
            compressedValueBytes: compressedBytes - compressedBytes / 2,
            decodedActiveKVBytes: 0,
            plainKVBytes: max(1, coldTokens * 4),
            memoryReductionRatio: 2
        )
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
            "codec       scheme    ctx      status   comp tok/s  plain tok/s  ratio   mlx/swift×  gate   cosine     mem×",
            "----------  -------   ------   ------   ----------  -----------  -----   ----------  -----  --------   -----",
        ]
        for r in results {
            let ctxLabel =
                r.contextLength >= 1024
                ? "\(r.contextLength / 1024)K" : "\(r.contextLength)"
            let nativeSpeed = r.nativeSpeedRatioToSwiftMetal.map { fmt($0, 2) } ?? "-"
            let nativeGate = r.nativePerfGatePassed.map { $0 ? "pass" : "fail" } ?? "-"
            lines.append(
                pad(r.codec, 10) + "  " + pad(r.scheme, 8) + "  " + pad(ctxLabel, 6)
                    + "   " + pad(r.status.rawValue, 7) + "  "
                    + pad(fmt(r.compressedTokensPerSecond, 1), 10) + "  "
                    + pad(fmt(r.plainTokensPerSecond, 1), 11) + "  " + pad(fmt(r.speedRatioToPlain, 2), 6)
                    + "  " + pad(nativeSpeed, 10) + "  " + pad(nativeGate, 5)
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
    var p95: Double
    var output: MLXArray
}

private func hybridSelectorDescriptors(
    for benchCase: TurboQuantHybridBenchCase
) -> [TurboQuantColdBlockDescriptor] {
    let coldTokens = max(0, benchCase.contextLength - benchCase.hotWindowTokens)
    guard coldTokens > 0 else { return [] }
    let blockCount = Int(ceil(Double(coldTokens) / Double(benchCase.coldBlockTokens)))
    return (0 ..< blockCount).map { index in
        let start = index * benchCase.coldBlockTokens
        let end = min(coldTokens, start + benchCase.coldBlockTokens)
        var anchorFlags: TurboQuantColdBlockAnchorFlags = []
        var lexicalScore: Float = 0
        var semanticScore: Float = 0
        for hint in benchCase.selectorHints {
            let overlapStart = max(start, hint.startToken)
            let overlapEnd = min(end, hint.endToken)
            guard overlapStart < overlapEnd, end > start else { continue }
            let overlap = Float(overlapEnd - overlapStart) / Float(end - start)
            lexicalScore += hint.lexicalScore * overlap
            semanticScore += hint.semanticScore * overlap
            anchorFlags.formUnion(hint.anchorFlags)
        }
        let keyNorm = 1 + Float((index * 17) % 11) / 10
        return TurboQuantColdBlockDescriptor(
            blockID: index,
            startToken: start,
            endToken: end,
            compressedSlotStart: 0,
            compressedSlotEnd: end - start,
            logicalTokenCount: end - start,
            recencyRank: blockCount - index,
            anchorFlags: anchorFlags,
            maxKeyNormEstimate: keyNorm,
            lexicalScore: lexicalScore,
            semanticScore: semanticScore,
            relevanceScore: Float(end)
        )
    }
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
    let p95Index = Swift.min(
        samples.count - 1,
        Swift.max(0, Int((0.95 * Double(samples.count)).rounded(.up)) - 1)
    )
    return TimedMedian(median: samples[samples.count / 2], p95: samples[p95Index], output: output)
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
