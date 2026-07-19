// Copyright © 2026 RNT56.

import Foundation
import MLXLMCommon

/// Sparse-V selection policies used by proof and benchmark runs.
///
/// These policies operate on normalized softmax weights. They are intentionally
/// separate from the production cache policy so top-k/cumulative/hybrid Sparse-V
/// stays opt-in while the native decode kernels collect quality evidence.
public enum TurboQuantSparseVSelectionMode: Hashable, Codable, Sendable {
    case off
    case threshold(Float)
    case topK(Int)
    case cumulativeMass(Double)
    case hybrid(cumulativeMass: Double, maxTopK: Int)
    case candidateSparse(recentTokens: Int, candidatePages: Int, olderTokenBudget: Int)

    private enum CodingKeys: String, CodingKey {
        case kind
        case threshold
        case topK
        case cumulativeMass
        case maxTopK
        case recentTokens
        case candidatePages
    }

    private enum Kind: String, Codable {
        case off
        case threshold
        case topK
        case cumulativeMass
        case hybrid = "hybridCumulativeMassTopK"
        case candidateSparse
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawKind = try container.decode(String.self, forKey: .kind)
        let kind: Kind
        switch rawKind.replacingOccurrences(of: "_", with: "-").lowercased() {
        case "off":
            kind = .off
        case "threshold":
            kind = .threshold
        case "topk", "top-k":
            kind = .topK
        case "cumulativemass", "cumulative-mass", "mass":
            kind = .cumulativeMass
        case "hybrid", "hybrid-cumulative", "hybrid-cumulative-mass-top-k",
             "hybridcumulativemasstopk":
            kind = .hybrid
        case "candidatesparse", "candidate-sparse":
            kind = .candidateSparse
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .kind,
                in: container,
                debugDescription: "Unknown Sparse-V selection mode '\(rawKind)'."
            )
        }
        switch kind {
        case .off:
            self = .off
        case .threshold:
            self = .threshold(
                try container.decodeIfPresent(Float.self, forKey: .threshold) ?? 0
            )
        case .topK:
            self = .topK(try container.decodeIfPresent(Int.self, forKey: .topK) ?? 0)
        case .cumulativeMass:
            self = .cumulativeMass(
                try container.decodeIfPresent(Double.self, forKey: .cumulativeMass) ?? 1
            )
        case .hybrid:
            self = .hybrid(
                cumulativeMass: try container.decodeIfPresent(
                    Double.self,
                    forKey: .cumulativeMass
                ) ?? 1,
                maxTopK: try container.decodeIfPresent(Int.self, forKey: .maxTopK) ?? 0
            )
        case .candidateSparse:
            self = .candidateSparse(
                recentTokens: try container.decodeIfPresent(
                    Int.self,
                    forKey: .recentTokens
                ) ?? 0,
                candidatePages: try container.decodeIfPresent(
                    Int.self,
                    forKey: .candidatePages
                ) ?? 0,
                olderTokenBudget: try container.decodeIfPresent(Int.self, forKey: .topK) ?? 0
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .off:
            try container.encode(Kind.off, forKey: .kind)
        case .threshold(let threshold):
            try container.encode(Kind.threshold, forKey: .kind)
            try container.encode(max(0, threshold), forKey: .threshold)
        case .topK(let topK):
            try container.encode(Kind.topK, forKey: .kind)
            try container.encode(max(0, topK), forKey: .topK)
        case .cumulativeMass(let cumulativeMass):
            try container.encode(Kind.cumulativeMass, forKey: .kind)
            try container.encode(Self.clampedMass(cumulativeMass), forKey: .cumulativeMass)
        case .hybrid(let cumulativeMass, let maxTopK):
            try container.encode(Kind.hybrid, forKey: .kind)
            try container.encode(Self.clampedMass(cumulativeMass), forKey: .cumulativeMass)
            try container.encode(max(0, maxTopK), forKey: .maxTopK)
        case .candidateSparse(let recentTokens, let candidatePages, let olderTokenBudget):
            try container.encode(Kind.candidateSparse, forKey: .kind)
            try container.encode(max(0, recentTokens), forKey: .recentTokens)
            try container.encode(max(0, candidatePages), forKey: .candidatePages)
            try container.encode(max(0, olderTokenBudget), forKey: .topK)
        }
    }

    public var label: String {
        switch self {
        case .off:
            return "dense"
        case .threshold(let threshold):
            return "threshold-\(Self.formatScientific(threshold))"
        case .topK(let topK):
            return "topk-\(max(0, topK))"
        case .cumulativeMass(let cumulativeMass):
            return "mass-\(Self.formatPercent(cumulativeMass))"
        case .hybrid(let cumulativeMass, let maxTopK):
            return "mass-\(Self.formatPercent(cumulativeMass))-topk-\(max(0, maxTopK))"
        case .candidateSparse(let recentTokens, let candidatePages, let olderTokenBudget):
            return "candidate-sparse-r\(max(0, recentTokens))-p\(max(0, candidatePages))-older\(max(0, olderTokenBudget))"
        }
    }

    public var canonicalKind: String {
        switch self {
        case .off:
            return "off"
        case .threshold:
            return "threshold"
        case .topK:
            return "topK"
        case .cumulativeMass:
            return "cumulativeMass"
        case .hybrid:
            return "hybridCumulativeMassTopK"
        case .candidateSparse:
            return "candidateSparse"
        }
    }

    public var thresholdForNativeKernel: Float? {
        if case .threshold(let threshold) = self, threshold > 0 {
            return threshold
        }
        return nil
    }

    public var requiresProofFallback: Bool {
        switch self {
        case .off, .threshold, .topK, .cumulativeMass, .hybrid, .candidateSparse:
            false
        }
    }

    public static let suggestedThresholds: [TurboQuantSparseVSelectionMode] = [
        .threshold(1e-4), .threshold(5e-5), .threshold(1e-5),
    ]

    public static let suggestedTopKCaps: [TurboQuantSparseVSelectionMode] = [
        .topK(128), .topK(256), .topK(512),
    ]

    public static let suggestedCumulativeMasses: [TurboQuantSparseVSelectionMode] = [
        .cumulativeMass(0.990), .cumulativeMass(0.995), .cumulativeMass(0.999),
    ]

    public static let suggestedHybrids: [TurboQuantSparseVSelectionMode] = [
        .hybrid(cumulativeMass: 0.995, maxTopK: 256),
        .hybrid(cumulativeMass: 0.999, maxTopK: 512),
    ]

    public static let suggestedCandidateSparse: [TurboQuantSparseVSelectionMode] = [
        .candidateSparse(recentTokens: 256, candidatePages: 4, olderTokenBudget: 128)
    ]

    public static let suggestedProofModes: [TurboQuantSparseVSelectionMode] =
        suggestedThresholds + suggestedTopKCaps + suggestedCumulativeMasses + suggestedHybrids
        + suggestedCandidateSparse

    public init(_ config: TurboQuantSparseSelectionConfig?) {
        guard let config else {
            self = .off
            return
        }
        switch config.mode {
        case .threshold:
            self = .threshold(config.threshold ?? 0)
        case .topK:
            self = .topK(config.topK ?? 0)
        case .cumulativeMass:
            self = .cumulativeMass((config.cumulativeMassPercent ?? 0) / 100)
        case .hybridCumulativeFloorMaxTopK:
            self = .hybrid(
                cumulativeMass: (config.cumulativeFloorPercent ?? 0) / 100,
                maxTopK: config.maxTopK ?? 0
            )
        case .candidateSparse:
            self = .candidateSparse(
                recentTokens: config.recentTokens ?? 0,
                candidatePages: config.candidatePages ?? 0,
                olderTokenBudget: config.topK ?? 0
            )
        }
    }

    private static func clampedMass(_ mass: Double) -> Double {
        max(0, min(1, mass))
    }

    private static func formatPercent(_ mass: Double) -> String {
        let percent = clampedMass(mass) * 100
        return String(format: "%.1f", percent)
    }

    private static func formatScientific(_ value: Float) -> String {
        String(format: "%.0e", max(0, value))
    }
}

public struct TurboQuantSparseVSelectionResult: Equatable, Sendable {
    public var retainedIndexes: [Int]
    public var retainedAttentionMass: Double
    public var consideredValueTokens: Int
    public var skippedValueTokens: Int
    public var recentTokenCount: Int?
    public var olderTokenCount: Int?
    public var pageCandidateCount: Int?
    public var fallbackReason: String?

    public init(
        retainedIndexes: [Int],
        retainedAttentionMass: Double,
        consideredValueTokens: Int,
        skippedValueTokens: Int,
        recentTokenCount: Int? = nil,
        olderTokenCount: Int? = nil,
        pageCandidateCount: Int? = nil,
        fallbackReason: String? = nil
    ) {
        self.retainedIndexes = retainedIndexes.sorted()
        self.retainedAttentionMass = max(0, min(1, retainedAttentionMass))
        self.consideredValueTokens = max(0, consideredValueTokens)
        self.skippedValueTokens = max(0, skippedValueTokens)
        self.recentTokenCount = recentTokenCount.map { max(0, $0) }
        self.olderTokenCount = olderTokenCount.map { max(0, $0) }
        self.pageCandidateCount = pageCandidateCount.map { max(0, $0) }
        self.fallbackReason = fallbackReason
    }

    public var skipRatio: Double {
        guard consideredValueTokens > 0 else { return 0 }
        return Double(skippedValueTokens) / Double(consideredValueTokens)
    }
}

public struct TurboQuantSparseVLayerHeadDiagnostics: Equatable, Codable, Sendable {
    public var layerIndex: Int
    public var headIndex: Int
    public var skippedValueTokens: Int
    public var consideredValueTokens: Int
    public var recentTokenCount: Int?
    public var olderTokenCount: Int?
    public var pageCandidateCount: Int?
    public var retainedAttentionMass: Double
    public var maxOutputError: Double?
    public var cosineSimilarity: Double?
    public var fallbackReason: String?
    public var selectionLatencyMS: Double?
    public var avLatencyMS: Double?
    public var denseReferenceLatencyMS: Double?

    public init(
        layerIndex: Int,
        headIndex: Int,
        skippedValueTokens: Int,
        consideredValueTokens: Int,
        recentTokenCount: Int? = nil,
        olderTokenCount: Int? = nil,
        pageCandidateCount: Int? = nil,
        retainedAttentionMass: Double,
        maxOutputError: Double? = nil,
        cosineSimilarity: Double? = nil,
        fallbackReason: String? = nil,
        selectionLatencyMS: Double? = nil,
        avLatencyMS: Double? = nil,
        denseReferenceLatencyMS: Double? = nil
    ) {
        self.layerIndex = max(0, layerIndex)
        self.headIndex = max(0, headIndex)
        self.skippedValueTokens = max(0, skippedValueTokens)
        self.consideredValueTokens = max(0, consideredValueTokens)
        self.recentTokenCount = recentTokenCount.map { max(0, $0) }
        self.olderTokenCount = olderTokenCount.map { max(0, $0) }
        self.pageCandidateCount = pageCandidateCount.map { max(0, $0) }
        self.retainedAttentionMass = max(0, min(1, retainedAttentionMass))
        self.maxOutputError = maxOutputError
        self.cosineSimilarity = cosineSimilarity
        self.fallbackReason = fallbackReason
        self.selectionLatencyMS = selectionLatencyMS.map { max(0, $0) }
        self.avLatencyMS = avLatencyMS.map { max(0, $0) }
        self.denseReferenceLatencyMS = denseReferenceLatencyMS.map { max(0, $0) }
    }
}

public struct TurboQuantSparseVProofDiagnostics: Equatable, Codable, Sendable {
    public var selectionMode: TurboQuantSparseVSelectionMode
    public var skippedValueTokens: Int
    public var consideredValueTokens: Int
    public var recentTokenCount: Int?
    public var olderTokenCount: Int?
    public var pageCandidateCount: Int?
    public var retainedAttentionMass: Double
    public var maxOutputError: Double?
    public var cosineSimilarity: Double?
    public var fallbackReason: String?
    public var perLayerHeadDiagnostics: [TurboQuantSparseVLayerHeadDiagnostics]
    public var qkMS: Double?
    public var softmaxMS: Double?
    public var selectionMS: Double?
    public var maskOrCompactionMS: Double?
    public var avMS: Double?
    public var totalMS: Double?
    public var denseK8V4ReferenceMS: Double?

    public init(
        selectionMode: TurboQuantSparseVSelectionMode,
        skippedValueTokens: Int,
        consideredValueTokens: Int,
        recentTokenCount: Int? = nil,
        olderTokenCount: Int? = nil,
        pageCandidateCount: Int? = nil,
        retainedAttentionMass: Double,
        maxOutputError: Double? = nil,
        cosineSimilarity: Double? = nil,
        fallbackReason: String? = nil,
        perLayerHeadDiagnostics: [TurboQuantSparseVLayerHeadDiagnostics] = [],
        qkMS: Double? = nil,
        softmaxMS: Double? = nil,
        selectionMS: Double? = nil,
        maskOrCompactionMS: Double? = nil,
        avMS: Double? = nil,
        totalMS: Double? = nil,
        denseK8V4ReferenceMS: Double? = nil
    ) {
        self.selectionMode = selectionMode
        self.skippedValueTokens = max(0, skippedValueTokens)
        self.consideredValueTokens = max(0, consideredValueTokens)
        self.recentTokenCount = recentTokenCount.map { max(0, $0) }
        self.olderTokenCount = olderTokenCount.map { max(0, $0) }
        self.pageCandidateCount = pageCandidateCount.map { max(0, $0) }
        self.retainedAttentionMass = max(0, min(1, retainedAttentionMass))
        self.maxOutputError = maxOutputError
        self.cosineSimilarity = cosineSimilarity
        self.fallbackReason = fallbackReason
        self.perLayerHeadDiagnostics = perLayerHeadDiagnostics
        self.qkMS = qkMS.map { max(0, $0) }
        self.softmaxMS = softmaxMS.map { max(0, $0) }
        self.selectionMS = selectionMS.map { max(0, $0) }
        self.maskOrCompactionMS = maskOrCompactionMS.map { max(0, $0) }
        self.avMS = avMS.map { max(0, $0) }
        self.totalMS = totalMS.map { max(0, $0) }
        self.denseK8V4ReferenceMS = denseK8V4ReferenceMS.map { max(0, $0) }
    }

    public var skipRatio: Double {
        guard consideredValueTokens > 0 else { return 0 }
        return Double(skippedValueTokens) / Double(consideredValueTokens)
    }
}

public enum TurboQuantValueBitPolicy: String, Hashable, Codable, Sendable, Equatable {
    case denseV4
    case calibratedV3
    case calibratedV2
    case residualVx

    public static func defaultPolicy(valueBits: Int) -> TurboQuantValueBitPolicy {
        switch max(2, min(4, valueBits)) {
        case 4:
            .denseV4
        case 3:
            .calibratedV3
        default:
            .calibratedV2
        }
    }
}

public struct TurboQuantLowerVCalibrationSummary: Equatable, Codable, Sendable {
    public var referenceConfig: String
    public var candidateConfig: String
    public var valueBitPolicy: TurboQuantValueBitPolicy
    public var valueBits: Int
    public var layerIndex: Int?
    public var headIndex: Int?
    public var attentionMass: Double?
    public var attentionEntropy: Double?
    public var valueReconstructionError: Double?
    public var outputCosine: Double?
    public var klDivergence: Double?
    public var maxLogitErrorP95: Double?
    public var promotionAllowed: Bool
    public var failureReason: String?

    public init(
        referenceConfig: String = "dense K8/V4",
        candidateConfig: String,
        valueBitPolicy: TurboQuantValueBitPolicy,
        valueBits: Int,
        layerIndex: Int? = nil,
        headIndex: Int? = nil,
        attentionMass: Double? = nil,
        attentionEntropy: Double? = nil,
        valueReconstructionError: Double? = nil,
        outputCosine: Double? = nil,
        klDivergence: Double? = nil,
        maxLogitErrorP95: Double? = nil,
        promotionAllowed: Bool = false,
        failureReason: String? = nil
    ) {
        self.referenceConfig = referenceConfig
        self.candidateConfig = candidateConfig
        self.valueBitPolicy = valueBitPolicy
        self.valueBits = max(2, min(4, valueBits))
        self.layerIndex = layerIndex.map { max(0, $0) }
        self.headIndex = headIndex.map { max(0, $0) }
        self.attentionMass = attentionMass.map { max(0, min(1, $0)) }
        self.attentionEntropy = attentionEntropy.map { max(0, $0) }
        self.valueReconstructionError = valueReconstructionError.map { max(0, $0) }
        self.outputCosine = outputCosine.map { max(-1, min(1, $0)) }
        self.klDivergence = klDivergence.map { max(0, $0) }
        self.maxLogitErrorP95 = maxLogitErrorP95.map { max(0, $0) }
        self.promotionAllowed = promotionAllowed
        self.failureReason = failureReason
    }
}

public struct TurboQuantSparseLowerVBenchmarkCandidate: Hashable, Codable, Sendable {
    public var label: String
    public var keyBits: Int
    public var valueBits: Int
    public var sparseVSelection: TurboQuantSparseVSelectionMode
    public var boundaryPolicy: TurboQuantBoundaryPolicy
    public var boundaryCachePrecision: TurboQuantBoundaryCachePrecision
    public var valueBitPolicy: TurboQuantValueBitPolicy
    public var productionDefault: Bool
    public var requiresRealModelGate: Bool

    public init(
        label: String,
        keyBits: Int = 8,
        valueBits: Int,
        sparseVSelection: TurboQuantSparseVSelectionMode = .off,
        boundaryPolicy: TurboQuantBoundaryPolicy = .disabled,
        boundaryCachePrecision: TurboQuantBoundaryCachePrecision = .affineK8V4,
        valueBitPolicy: TurboQuantValueBitPolicy? = nil,
        productionDefault: Bool = false,
        requiresRealModelGate: Bool = true
    ) {
        self.label = label
        self.keyBits = keyBits
        self.valueBits = max(2, min(4, valueBits))
        self.sparseVSelection = sparseVSelection
        self.boundaryPolicy = boundaryPolicy
        self.boundaryCachePrecision = boundaryCachePrecision
        self.valueBitPolicy = valueBitPolicy ?? TurboQuantValueBitPolicy.defaultPolicy(
            valueBits: self.valueBits
        )
        self.productionDefault = productionDefault
        self.requiresRealModelGate = requiresRealModelGate
    }

    public var precisionPolicy: TurboQuantKVPrecisionPolicy {
        TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: valueBits),
            boundary: boundaryPolicy,
            boundaryCachePrecision: boundaryCachePrecision
        )
    }

    public var sparseValuePolicy: TurboQuantSparseValuePolicy {
        guard let threshold = sparseVSelection.thresholdForNativeKernel else {
            return .off
        }
        return .force(threshold: threshold)
    }

    public var sparseSelectionConfig: TurboQuantSparseSelectionConfig? {
        switch sparseVSelection {
        case .off:
            return nil
        case .threshold(let threshold):
            return .threshold(threshold)
        case .topK(let topK):
            return .topK(topK)
        case .cumulativeMass(let cumulativeMass):
            return .cumulativeMass(cumulativeMass * 100)
        case .hybrid(let cumulativeMass, let maxTopK):
            return .hybrid(cumulativeFloorPercent: cumulativeMass * 100, maxTopK: maxTopK)
        case .candidateSparse(let recentTokens, let candidatePages, let olderTokenBudget):
            return .candidateSparse(
                recentTokens: recentTokens,
                candidatePages: candidatePages,
                olderTokenBudget: olderTokenBudget
            )
        }
    }

    public var fallbackReason: String? {
        nil
    }
}

public struct TurboQuantSparseLowerVBenchmarkRow: Equatable, Codable, Sendable {
    public enum Status: String, Codable, Sendable {
        case planned
        case measured
        case failed
        case blocked
    }

    public var candidate: TurboQuantSparseLowerVBenchmarkCandidate
    public var contextLength: Int
    public var status: Status
    public var baselineLabel: String
    public var decodeTokensPerSecond: Double?
    public var baselineDecodeTokensPerSecond: Double?
    public var speedRatioToDenseK8V4: Double?
    public var plainDecodeTokensPerSecond: Double?
    public var speedRatioToPlainFP16: Double?
    public var peakMemoryBytes: Int?
    public var kvMemoryBytes: Int?
    public var baselineKVMemoryBytes: Int?
    public var plainKVBytes: Int?
    public var memoryReductionRatioToDenseK8V4: Double?
    public var memoryReductionRatioToPlain: Double?
    public var outputCosineToDenseK8V4: Double?
    public var maxAbsErrorToDenseK8V4: Double?
    public var retrievalGate: String?
    public var jsonToolCallGate: String?
    public var selectionLatencyMS: Double?
    public var avLatencyMS: Double?
    public var denseK8V4ReferenceMS: Double?
    public var actualMixedBitsPerValue: Double?
    public var fallbackCount: Int
    public var fallbackReason: String?
    public var promotionEligible: Bool?
    public var promotionBlockReason: String?
    public var sparseVDiagnostics: TurboQuantSparseVProofDiagnostics?
    public var calibrationSummary: TurboQuantLowerVCalibrationSummary?

    public init(
        candidate: TurboQuantSparseLowerVBenchmarkCandidate,
        contextLength: Int,
        status: Status = .planned,
        baselineLabel: String = "dense K8/V4",
        decodeTokensPerSecond: Double? = nil,
        baselineDecodeTokensPerSecond: Double? = nil,
        speedRatioToDenseK8V4: Double? = nil,
        plainDecodeTokensPerSecond: Double? = nil,
        speedRatioToPlainFP16: Double? = nil,
        peakMemoryBytes: Int? = nil,
        kvMemoryBytes: Int? = nil,
        baselineKVMemoryBytes: Int? = nil,
        plainKVBytes: Int? = nil,
        memoryReductionRatioToDenseK8V4: Double? = nil,
        memoryReductionRatioToPlain: Double? = nil,
        outputCosineToDenseK8V4: Double? = nil,
        maxAbsErrorToDenseK8V4: Double? = nil,
        retrievalGate: String? = nil,
        jsonToolCallGate: String? = nil,
        selectionLatencyMS: Double? = nil,
        avLatencyMS: Double? = nil,
        denseK8V4ReferenceMS: Double? = nil,
        actualMixedBitsPerValue: Double? = nil,
        fallbackCount: Int = 0,
        fallbackReason: String? = nil,
        promotionEligible: Bool? = nil,
        promotionBlockReason: String? = nil,
        sparseVDiagnostics: TurboQuantSparseVProofDiagnostics? = nil,
        calibrationSummary: TurboQuantLowerVCalibrationSummary? = nil
    ) {
        self.candidate = candidate
        self.contextLength = max(1, contextLength)
        self.status = status
        self.baselineLabel = baselineLabel
        self.decodeTokensPerSecond = decodeTokensPerSecond
        self.baselineDecodeTokensPerSecond = baselineDecodeTokensPerSecond
        if let speedRatioToDenseK8V4 {
            self.speedRatioToDenseK8V4 = speedRatioToDenseK8V4
        } else if let decodeTokensPerSecond, let baselineDecodeTokensPerSecond {
            self.speedRatioToDenseK8V4 =
                decodeTokensPerSecond / max(baselineDecodeTokensPerSecond, .leastNonzeroMagnitude)
        } else {
            self.speedRatioToDenseK8V4 = nil
        }
        self.plainDecodeTokensPerSecond = plainDecodeTokensPerSecond
        if let speedRatioToPlainFP16 {
            self.speedRatioToPlainFP16 = speedRatioToPlainFP16
        } else if let decodeTokensPerSecond, let plainDecodeTokensPerSecond {
            self.speedRatioToPlainFP16 =
                decodeTokensPerSecond / max(plainDecodeTokensPerSecond, .leastNonzeroMagnitude)
        } else {
            self.speedRatioToPlainFP16 = nil
        }
        self.peakMemoryBytes = peakMemoryBytes
        self.kvMemoryBytes = kvMemoryBytes
        self.baselineKVMemoryBytes = baselineKVMemoryBytes
        self.plainKVBytes = plainKVBytes
        if let memoryReductionRatioToDenseK8V4 {
            self.memoryReductionRatioToDenseK8V4 = memoryReductionRatioToDenseK8V4
        } else if let baselineKVMemoryBytes, let kvMemoryBytes {
            self.memoryReductionRatioToDenseK8V4 =
                Double(baselineKVMemoryBytes) / Double(max(1, kvMemoryBytes))
        } else {
            self.memoryReductionRatioToDenseK8V4 = nil
        }
        if let memoryReductionRatioToPlain {
            self.memoryReductionRatioToPlain = memoryReductionRatioToPlain
        } else if let plainKVBytes, let kvMemoryBytes {
            self.memoryReductionRatioToPlain =
                Double(plainKVBytes) / Double(max(1, kvMemoryBytes))
        } else {
            self.memoryReductionRatioToPlain = nil
        }
        self.outputCosineToDenseK8V4 = outputCosineToDenseK8V4
        self.maxAbsErrorToDenseK8V4 = maxAbsErrorToDenseK8V4
        self.retrievalGate = retrievalGate
        self.jsonToolCallGate = jsonToolCallGate
        self.selectionLatencyMS = selectionLatencyMS.map { max(0, $0) }
        self.avLatencyMS = avLatencyMS.map { max(0, $0) }
        self.denseK8V4ReferenceMS = denseK8V4ReferenceMS.map { max(0, $0) }
        self.actualMixedBitsPerValue = actualMixedBitsPerValue.map { max(0, $0) }
        self.fallbackCount = max(0, fallbackCount)
        self.fallbackReason = fallbackReason ?? candidate.fallbackReason
        let inferredPromotionEligible =
            status == .measured && self.fallbackCount == 0 && self.fallbackReason == nil
        let effectivePromotionEligible = promotionEligible ?? inferredPromotionEligible
        self.promotionEligible = effectivePromotionEligible
        self.promotionBlockReason =
            promotionBlockReason
            ?? (effectivePromotionEligible ? nil : self.fallbackReason)
        self.sparseVDiagnostics = sparseVDiagnostics
        self.calibrationSummary = calibrationSummary
    }
}

public enum TurboQuantSparseVProof {
    public static func select(
        normalizedWeights: [Double],
        mode: TurboQuantSparseVSelectionMode
    ) -> TurboQuantSparseVSelectionResult {
        let weights = normalizedWeights.map { $0.isFinite ? max(0, $0) : 0 }
        let considered = weights.count
        guard considered > 0 else {
            return TurboQuantSparseVSelectionResult(
                retainedIndexes: [],
                retainedAttentionMass: 0,
                consideredValueTokens: 0,
                skippedValueTokens: 0
            )
        }

        let retained: [Int]
        var fallbackReason: String?
        var recentTokenCount: Int?
        var olderTokenCount: Int?
        var pageCandidateCount: Int?
        switch mode {
        case .off:
            retained = Array(0 ..< considered)
        case .threshold(let threshold):
            retained = weights.indices.filter { weights[$0] >= Double(max(0, threshold)) }
        case .topK(let topK):
            retained = topWeightIndexes(weights, limit: topK)
        case .cumulativeMass(let cumulativeMass):
            retained = cumulativeWeightIndexes(weights, mass: cumulativeMass)
        case .hybrid(let cumulativeMass, let maxTopK):
            retained = cumulativeWeightIndexes(weights, mass: cumulativeMass, limit: maxTopK)
            let retainedMass = retained.reduce(0) { $0 + weights[$1] }
            if retainedMass + 1e-12 < max(0, min(1, cumulativeMass)) {
                fallbackReason =
                    "retained mass \(retainedMass) is below requested floor \(cumulativeMass) under top-k cap \(max(0, maxTopK))"
            }
        case .candidateSparse(let recentTokens, let candidatePages, let olderTokenBudget):
            let recent = min(max(0, recentTokens), considered)
            let olderCount = max(0, considered - recent)
            let olderWeights = Array(weights.prefix(olderCount))
            let olderIndexes = topWeightIndexes(olderWeights, limit: olderTokenBudget)
            let recentIndexes = weights.indices.suffix(recent)
            retained = Array(Set(olderIndexes).union(recentIndexes)).sorted()
            recentTokenCount = recent
            olderTokenCount = olderIndexes.count
            pageCandidateCount = max(0, candidatePages)
        }

        let retainedMass = retained.reduce(0) { $0 + weights[$1] }
        return TurboQuantSparseVSelectionResult(
            retainedIndexes: retained,
            retainedAttentionMass: retainedMass,
            consideredValueTokens: considered,
            skippedValueTokens: considered - retained.count,
            recentTokenCount: recentTokenCount,
            olderTokenCount: olderTokenCount,
            pageCandidateCount: pageCandidateCount,
            fallbackReason: fallbackReason
        )
    }

    public static func diagnostics(
        normalizedWeights: [Double],
        mode: TurboQuantSparseVSelectionMode,
        maxOutputError: Double? = nil,
        cosineSimilarity: Double? = nil,
        fallbackReason: String? = nil,
        perLayerHeadDiagnostics: [TurboQuantSparseVLayerHeadDiagnostics] = [],
        qkMS: Double? = nil,
        softmaxMS: Double? = nil,
        selectionMS: Double? = nil,
        maskOrCompactionMS: Double? = nil,
        avMS: Double? = nil,
        totalMS: Double? = nil,
        denseK8V4ReferenceMS: Double? = nil
    ) -> TurboQuantSparseVProofDiagnostics {
        let selection = select(normalizedWeights: normalizedWeights, mode: mode)
        return TurboQuantSparseVProofDiagnostics(
            selectionMode: mode,
            skippedValueTokens: selection.skippedValueTokens,
            consideredValueTokens: selection.consideredValueTokens,
            recentTokenCount: selection.recentTokenCount,
            olderTokenCount: selection.olderTokenCount,
            pageCandidateCount: selection.pageCandidateCount,
            retainedAttentionMass: selection.retainedAttentionMass,
            maxOutputError: maxOutputError,
            cosineSimilarity: cosineSimilarity,
            fallbackReason: fallbackReason ?? selection.fallbackReason,
            perLayerHeadDiagnostics: perLayerHeadDiagnostics,
            qkMS: qkMS,
            softmaxMS: softmaxMS,
            selectionMS: selectionMS,
            maskOrCompactionMS: maskOrCompactionMS,
            avMS: avMS,
            totalMS: totalMS,
            denseK8V4ReferenceMS: denseK8V4ReferenceMS
        )
    }

    private static func topWeightIndexes(_ weights: [Double], limit: Int) -> [Int] {
        guard limit > 0 else { return [] }
        return weights.indices.sorted {
            if weights[$0] == weights[$1] { return $0 < $1 }
            return weights[$0] > weights[$1]
        }.prefix(min(limit, weights.count)).sorted()
    }

    private static func cumulativeWeightIndexes(
        _ weights: [Double],
        mass: Double,
        limit: Int? = nil
    ) -> [Int] {
        let target = max(0, min(1, mass))
        let maxCount = max(0, min(limit ?? weights.count, weights.count))
        guard target > 0, maxCount > 0 else { return [] }
        let sorted = weights.indices.sorted {
            if weights[$0] == weights[$1] { return $0 < $1 }
            return weights[$0] > weights[$1]
        }
        var retained: [Int] = []
        retained.reserveCapacity(maxCount)
        var cumulative = 0.0
        for index in sorted.prefix(maxCount) {
            retained.append(index)
            cumulative += weights[index]
            if cumulative >= target { break }
        }
        return retained.sorted()
    }
}

public enum TurboQuantSparseLowerVBenchmarkMatrix {
    public static let acceptanceContexts = [32_768, 65_536, 131_072]

    public static var defaultCandidates: [TurboQuantSparseLowerVBenchmarkCandidate] {
        var candidates = [
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "K8/V4 dense baseline",
                valueBits: 4,
                productionDefault: true,
                requiresRealModelGate: false
            ),
            TurboQuantSparseLowerVBenchmarkCandidate(label: "K8/V3", valueBits: 3),
            TurboQuantSparseLowerVBenchmarkCandidate(label: "K8/V2", valueBits: 2),
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "K8/V3 residual outliers",
                valueBits: 3,
                valueBitPolicy: .residualVx
            ),
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "K8/V2 residual outliers",
                valueBits: 2,
                valueBitPolicy: .residualVx
            ),
        ]
        candidates += TurboQuantSparseVSelectionMode.suggestedProofModes.map {
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "K8/V4 + Sparse-V \($0.label)",
                valueBits: 4,
                sparseVSelection: $0
            )
        }
        candidates += [
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "K8/V3 + Sparse-V threshold 1e-5",
                valueBits: 3,
                sparseVSelection: .threshold(1e-5)
            ),
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "protected K8/V4 edge5 + middle V3",
                valueBits: 3,
                boundaryPolicy: .protectedEdges(
                    first: TurboQuantProfile.guardedAffineK8V3ProtectedEdgeLayers,
                    last: TurboQuantProfile.guardedAffineK8V3ProtectedEdgeLayers
                ),
                boundaryCachePrecision: .affineK8V4
            ),
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "protected K8/V4 edges + middle V2",
                valueBits: 2,
                boundaryPolicy: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .affineK8V4
            ),
            TurboQuantSparseLowerVBenchmarkCandidate(
                label: "raw edges + middle V3",
                valueBits: 3,
                boundaryPolicy: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .raw
            ),
        ]
        return candidates
    }

    public static func plannedRows(
        contexts: [Int] = acceptanceContexts,
        candidates: [TurboQuantSparseLowerVBenchmarkCandidate] = defaultCandidates
    ) -> [TurboQuantSparseLowerVBenchmarkRow] {
        contexts.flatMap { context in
            candidates.map {
                TurboQuantSparseLowerVBenchmarkRow(
                    candidate: $0,
                    contextLength: context,
                    status: .planned,
                    fallbackCount: $0.fallbackReason == nil ? 0 : 1
                )
            }
        }
    }

    public static func comparisonRows(
        from results: [TurboQuantBenchResult],
        baselineVariantLabel: String = "dense-k8-v4"
    ) -> [TurboQuantSparseLowerVBenchmarkRow] {
        var baselines: [Int: TurboQuantBenchResult] = [:]
        for result in results {
            let isLabeledBaseline = result.variantLabel == baselineVariantLabel
            let isLegacyDenseBaseline =
                result.anchorLabel == nil && result.variantLabel == nil
                && result.sparseVEnabled == false
                && result.valuePrecision == TurboQuantValuePrecision.turbo4v2.rawValue
            if result.status == .ok, isLabeledBaseline || isLegacyDenseBaseline {
                baselines[result.contextLength] = result
            }
        }

        return results.map { result in
            let baseline = baselines[result.contextLength]
            let status: TurboQuantSparseLowerVBenchmarkRow.Status =
                switch result.status {
                case .ok:
                    .measured
                case .failed:
                    .failed
                case .skipped:
                    .blocked
                }
            let sparseMode = TurboQuantSparseVSelectionMode(result.sparseVSelectionConfig)
            let valueBits = result.precisionPolicy?.resolvedValueBits
                ?? TurboQuantValuePrecision(rawValue: result.valuePrecision)?.valueBits
                ?? 4
            let candidate = TurboQuantSparseLowerVBenchmarkCandidate(
                label: result.variantLabel ?? result.label,
                valueBits: valueBits,
                sparseVSelection: sparseMode,
                boundaryPolicy: result.precisionPolicy?.boundary ?? .disabled,
                boundaryCachePrecision: result.precisionPolicy?.boundaryCachePrecision ?? .affineK8V4,
                valueBitPolicy: TurboQuantValueBitPolicy.defaultPolicy(valueBits: valueBits),
                productionDefault: result.variantLabel == baselineVariantLabel,
                requiresRealModelGate: result.variantLabel != baselineVariantLabel
            )
            let fallbackReason = result.fallbackReason ?? result.detail ?? candidate.fallbackReason
            var promotionBlockReasons: [String] = []
            if result.status != .ok {
                promotionBlockReasons.append(result.detail ?? "row status \(result.status.rawValue)")
            }
            if !candidate.productionDefault && baseline == nil {
                promotionBlockReasons.append("missing dense K8/V4 baseline")
            }
            if result.sparseVSelectionConfig != nil && !result.sparseVEnabled {
                promotionBlockReasons.append("Sparse-V requested but inactive")
            }
            if let fallbackReason, result.status == .ok {
                promotionBlockReasons.append("fallback: \(fallbackReason)")
            }
            let promotionBlockReason =
                promotionBlockReasons.isEmpty
                ? nil
                : promotionBlockReasons.joined(separator: "; ")
            let timingComponents = [
                result.qkMS,
                result.softmaxMS,
                result.selectionMS,
                result.maskOrCompactionMS,
                result.avMS,
            ].compactMap { $0 }
            let sparseTotalMS = timingComponents.isEmpty
                ? nil
                : timingComponents.reduce(0, +)
            let sparseDiagnostics =
                result.sparseVEnabled || result.sparseVSelectionConfig != nil
                ? TurboQuantSparseVProofDiagnostics(
                    selectionMode: sparseMode,
                    skippedValueTokens: result.sparseVSkippedValueTokens ?? 0,
                    consideredValueTokens: result.sparseVTotalValueTokens ?? 0,
                    recentTokenCount: result.sparseVRecentTokenCount,
                    olderTokenCount: result.sparseVOlderTokenCount,
                    pageCandidateCount: result.sparseVPageCandidateCount,
                    retainedAttentionMass: result.sparseVRetainedMass ?? 0,
                    maxOutputError: result.maxOutputError ?? result.maxAbsErrorP95,
                    cosineSimilarity: result.cosineSimilarity,
                    fallbackReason: fallbackReason,
                    perLayerHeadDiagnostics: (result.layerHeadDiagnostics ?? []).map {
                        TurboQuantSparseVLayerHeadDiagnostics(
                            layerIndex: $0.layerIndex,
                            headIndex: $0.headIndex,
                            skippedValueTokens: $0.skippedValueTokens,
                            consideredValueTokens: $0.totalValueTokens,
                            recentTokenCount: $0.recentTokenCount,
                            olderTokenCount: $0.olderTokenCount,
                            pageCandidateCount: $0.pageCandidateCount,
                            retainedAttentionMass: $0.retainedMass ?? 0,
                            maxOutputError: $0.maxOutputError,
                            cosineSimilarity: $0.cosineSimilarity,
                            fallbackReason: $0.fallbackReason
                        )
                    },
                    qkMS: result.qkMS,
                    softmaxMS: result.softmaxMS,
                    selectionMS: result.selectionMS,
                    maskOrCompactionMS: result.maskOrCompactionMS,
                    avMS: result.avMS,
                    totalMS: sparseTotalMS,
                    denseK8V4ReferenceMS: result.denseK8V4ReferenceMS
                ) : nil

            return TurboQuantSparseLowerVBenchmarkRow(
                candidate: candidate,
                contextLength: result.contextLength,
                status: status,
                decodeTokensPerSecond: result.compressedTokensPerSecond,
                baselineDecodeTokensPerSecond: baseline?.compressedTokensPerSecond,
                plainDecodeTokensPerSecond: result.plainTokensPerSecond,
                speedRatioToPlainFP16: result.speedRatioToPlain,
                peakMemoryBytes: nil,
                kvMemoryBytes: result.compressedKVBytes,
                baselineKVMemoryBytes: baseline?.compressedKVBytes,
                plainKVBytes: result.plainKVBytes,
                memoryReductionRatioToPlain: result.memoryReductionRatio,
                outputCosineToDenseK8V4: result.cosineSimilarity,
                maxAbsErrorToDenseK8V4: result.maxOutputError ?? result.maxAbsErrorP95,
                selectionLatencyMS: result.selectionMS,
                avLatencyMS: result.avMS,
                denseK8V4ReferenceMS: result.denseK8V4ReferenceMS,
                actualMixedBitsPerValue: Double(valueBits),
                fallbackCount: fallbackReason == nil ? 0 : 1,
                fallbackReason: fallbackReason,
                promotionEligible: promotionBlockReason == nil,
                promotionBlockReason: promotionBlockReason,
                sparseVDiagnostics: sparseDiagnostics,
                calibrationSummary: TurboQuantLowerVCalibrationSummary(
                    candidateConfig: result.variantLabel ?? result.label,
                    valueBitPolicy: candidate.valueBitPolicy,
                    valueBits: valueBits,
                    outputCosine: result.cosineSimilarity,
                    maxLogitErrorP95: result.maxOutputError ?? result.maxAbsErrorP95,
                    promotionAllowed: candidate.productionDefault,
                    failureReason: candidate.productionDefault ? nil : "requires real-model gate"
                )
            )
        }
    }

    public static func benchmarkCases(
        contexts: [Int] = acceptanceContexts,
        candidates: [TurboQuantSparseLowerVBenchmarkCandidate] = defaultCandidates
    ) -> [TurboQuantBenchCase] {
        let anchor = "dense-k8-v4"
        return contexts.flatMap { context in
            candidates.map { candidate in
                let codec: TurboQuantKVCodec =
                    candidate.valueBits == TurboQuantKVCodec.affineK8V4ValueBits
                    ? .affineK8V4
                    : .affineK8Vx
                var benchCase = TurboQuantBenchCase.qwen35_2B(
                    contextLength: context,
                    scheme: .turbo8,
                    codec: codec,
                    runtimeMode: .capacityTurboQuant,
                    precisionPolicy: candidate.precisionPolicy,
                    sparseValuePolicy: candidate.sparseValuePolicy,
                    sparseSelectionConfig: candidate.sparseSelectionConfig
                )
                benchCase.anchorLabel = anchor
                benchCase.variantLabel = variantLabel(for: candidate, baselineVariantLabel: anchor)
                benchCase.label = "\(benchCase.label)-\(benchCase.variantLabel ?? candidate.label)"
                return benchCase
            }
        }
    }

    private static func variantLabel(
        for candidate: TurboQuantSparseLowerVBenchmarkCandidate,
        baselineVariantLabel: String
    ) -> String {
        guard !candidate.productionDefault else { return baselineVariantLabel }
        let label = candidate.label.lowercased()
            .replacingOccurrences(of: "[^a-z0-9]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return label.isEmpty ? "variant" : label
    }

    public static func renderTable(_ rows: [TurboQuantSparseLowerVBenchmarkRow]) -> String {
        var lines = [
            "candidate                                ctx      status    sparse-v                    dense speedx  fp16 speedx  memx fp16  cosine     max err    fallback",
            "---------------------------------------  -------  --------  --------------------------  ------------  -----------  ---------  ---------  ---------  --------",
        ]
        for row in rows {
            let ctx = row.contextLength >= 1024
                ? "\(row.contextLength / 1024)K" : "\(row.contextLength)"
            let denseRatio = row.speedRatioToDenseK8V4.map { fmt($0, 2) } ?? "-"
            let fp16Ratio = row.speedRatioToPlainFP16.map { fmt($0, 2) } ?? "-"
            let memoryRatio = row.memoryReductionRatioToPlain.map { fmt($0, 2) } ?? "-"
            let cosine = row.outputCosineToDenseK8V4.map { fmt($0, 6) } ?? "-"
            let maxError = row.maxAbsErrorToDenseK8V4.map { fmt($0, 6) } ?? "-"
            let fallback = row.fallbackReason == nil ? "-" : "\(row.fallbackCount)"
            lines.append(
                pad(row.candidate.label, 39) + "  "
                    + pad(ctx, 7) + "  "
                    + pad(row.status.rawValue, 8) + "  "
                    + pad(row.candidate.sparseVSelection.label, 26) + "  "
                    + pad(denseRatio, 12) + "  "
                    + pad(fp16Ratio, 11) + "  "
                    + pad(memoryRatio, 9) + "  "
                    + pad(cosine, 9) + "  "
                    + pad(maxError, 9) + "  "
                    + fallback
            )
        }
        return lines.joined(separator: "\n")
    }
}

private func pad(_ string: String, _ width: Int) -> String {
    string.count >= width
        ? String(string.prefix(width))
        : string + String(repeating: " ", count: width - string.count)
}

private func fmt(_ value: Double, _ decimals: Int) -> String {
    guard value.isFinite else { return "n/a" }
    return String(format: "%.\(decimals)f", value)
}
