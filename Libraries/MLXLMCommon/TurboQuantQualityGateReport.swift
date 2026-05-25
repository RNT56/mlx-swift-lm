import Foundation

public enum TurboQuantBenchmarkSuiteID: String, Codable, Sendable, CaseIterable {
    case tinyDeterministicLogitsV1 = "tiny-deterministic-logits-v1"
    case prefillExactnessV1 = "prefill-exactness-v1"
    case fallbackEquivalenceV1 = "fallback-equivalence-v1"
    case longContextNeedleV1 = "long-context-needle-v1"
    case snapshotRoundtripV1 = "snapshot-roundtrip-v1"
    case mobileMemoryAcceptanceV1 = "mobile-memory-acceptance-v1"
}

public struct TurboQuantQualityGateReport: Hashable, Codable, Sendable {
    public static let schemaVersion = 1
    public static let gateVersion = 1

    public var schemaVersion: Int
    public var gateVersion: Int
    public var benchmarkSuiteID: String
    public var deterministicTop1MatchRate: Double
    public var logitKLDivergenceMean: Double
    public var logitMaxAbsErrorP95: Double
    public var perplexityDeltaPercent: Double?
    public var retrievalNeedlePassRate: Double?
    public var taskEvalDeltaPercent: Double?
    public var attentionOutputCosineMean: Double?
    public var noNaNOrInf: Bool
    public var fallbackEquivalent: Bool
    public var prefillExact: Bool
    public var snapshotRoundtripEquivalent: Bool?
    public var profileQualityThresholdOverride: String?
    public var gateReason: String?
    public var passed: Bool

    public init(
        schemaVersion: Int = Self.schemaVersion,
        gateVersion: Int = Self.gateVersion,
        benchmarkSuiteID: String,
        deterministicTop1MatchRate: Double,
        logitKLDivergenceMean: Double,
        logitMaxAbsErrorP95: Double,
        perplexityDeltaPercent: Double? = nil,
        retrievalNeedlePassRate: Double? = nil,
        taskEvalDeltaPercent: Double? = nil,
        attentionOutputCosineMean: Double? = nil,
        noNaNOrInf: Bool,
        fallbackEquivalent: Bool,
        prefillExact: Bool,
        snapshotRoundtripEquivalent: Bool? = nil,
        profileQualityThresholdOverride: String? = nil,
        gateReason: String? = nil,
        passed: Bool
    ) {
        self.schemaVersion = schemaVersion
        self.gateVersion = gateVersion
        self.benchmarkSuiteID = benchmarkSuiteID
        self.deterministicTop1MatchRate = deterministicTop1MatchRate
        self.logitKLDivergenceMean = logitKLDivergenceMean
        self.logitMaxAbsErrorP95 = logitMaxAbsErrorP95
        self.perplexityDeltaPercent = perplexityDeltaPercent
        self.retrievalNeedlePassRate = retrievalNeedlePassRate
        self.taskEvalDeltaPercent = taskEvalDeltaPercent
        self.attentionOutputCosineMean = attentionOutputCosineMean
        self.noNaNOrInf = noNaNOrInf
        self.fallbackEquivalent = fallbackEquivalent
        self.prefillExact = prefillExact
        self.snapshotRoundtripEquivalent = snapshotRoundtripEquivalent
        self.profileQualityThresholdOverride = profileQualityThresholdOverride
        self.gateReason = gateReason
        self.passed = passed
    }

    public init(
        schemaVersion: Int = Self.schemaVersion,
        gateVersion: Int = Self.gateVersion,
        benchmarkSuiteID: TurboQuantBenchmarkSuiteID,
        deterministicTop1MatchRate: Double,
        logitKLDivergenceMean: Double,
        logitMaxAbsErrorP95: Double,
        perplexityDeltaPercent: Double? = nil,
        retrievalNeedlePassRate: Double? = nil,
        taskEvalDeltaPercent: Double? = nil,
        attentionOutputCosineMean: Double? = nil,
        noNaNOrInf: Bool,
        fallbackEquivalent: Bool,
        prefillExact: Bool,
        snapshotRoundtripEquivalent: Bool? = nil,
        profileQualityThresholdOverride: String? = nil,
        gateReason: String? = nil,
        passed: Bool
    ) {
        self.init(
            schemaVersion: schemaVersion,
            gateVersion: gateVersion,
            benchmarkSuiteID: benchmarkSuiteID.rawValue,
            deterministicTop1MatchRate: deterministicTop1MatchRate,
            logitKLDivergenceMean: logitKLDivergenceMean,
            logitMaxAbsErrorP95: logitMaxAbsErrorP95,
            perplexityDeltaPercent: perplexityDeltaPercent,
            retrievalNeedlePassRate: retrievalNeedlePassRate,
            taskEvalDeltaPercent: taskEvalDeltaPercent,
            attentionOutputCosineMean: attentionOutputCosineMean,
            noNaNOrInf: noNaNOrInf,
            fallbackEquivalent: fallbackEquivalent,
            prefillExact: prefillExact,
            snapshotRoundtripEquivalent: snapshotRoundtripEquivalent,
            profileQualityThresholdOverride: profileQualityThresholdOverride,
            gateReason: gateReason,
            passed: passed
        )
    }

    public static func evaluated(
        benchmarkSuiteID: TurboQuantBenchmarkSuiteID = .tinyDeterministicLogitsV1,
        deterministicTop1MatchRate: Double,
        logitKLDivergenceMean: Double,
        logitMaxAbsErrorP95: Double,
        perplexityDeltaPercent: Double? = nil,
        retrievalNeedlePassRate: Double? = nil,
        taskEvalDeltaPercent: Double? = nil,
        attentionOutputCosineMean: Double? = nil,
        noNaNOrInf: Bool,
        fallbackEquivalent: Bool,
        prefillExact: Bool,
        snapshotRoundtripEquivalent: Bool? = nil,
        profileQualityThresholdOverride: String? = nil,
        top1Threshold: Double = 0.95,
        klThreshold: Double = 0.05,
        p95MaxAbsErrorThreshold: Double = 0.5
    ) -> TurboQuantQualityGateReport {
        let top1 = finiteOrDefault(deterministicTop1MatchRate, default: 0)
        let kl = finiteOrDefault(logitKLDivergenceMean, default: Double.greatestFiniteMagnitude)
        let p95 = finiteOrDefault(logitMaxAbsErrorP95, default: Double.greatestFiniteMagnitude)
        var reasons = [String]()
        if !noNaNOrInf { reasons.append("NaN or Inf detected") }
        if !prefillExact { reasons.append("prefill exactness failed") }
        if !fallbackEquivalent { reasons.append("fallback equivalence failed") }
        if snapshotRoundtripEquivalent == false { reasons.append("snapshot roundtrip failed") }
        if top1 < top1Threshold {
            reasons.append("deterministic top-1 match \(top1) below \(top1Threshold)")
        }
        if kl > klThreshold {
            reasons.append("mean logit KL \(kl) above \(klThreshold)")
        }
        if p95 > p95MaxAbsErrorThreshold {
            reasons.append("p95 max abs error \(p95) above \(p95MaxAbsErrorThreshold)")
        }

        return TurboQuantQualityGateReport(
            benchmarkSuiteID: benchmarkSuiteID,
            deterministicTop1MatchRate: top1,
            logitKLDivergenceMean: kl,
            logitMaxAbsErrorP95: p95,
            perplexityDeltaPercent: finiteOptional(perplexityDeltaPercent),
            retrievalNeedlePassRate: finiteOptional(retrievalNeedlePassRate),
            taskEvalDeltaPercent: finiteOptional(taskEvalDeltaPercent),
            attentionOutputCosineMean: finiteOptional(attentionOutputCosineMean),
            noNaNOrInf: noNaNOrInf,
            fallbackEquivalent: fallbackEquivalent,
            prefillExact: prefillExact,
            snapshotRoundtripEquivalent: snapshotRoundtripEquivalent,
            profileQualityThresholdOverride: profileQualityThresholdOverride,
            gateReason: reasons.isEmpty ? nil : reasons.joined(separator: "; "),
            passed: reasons.isEmpty
        )
    }

    public static func failed(
        benchmarkSuiteID: TurboQuantBenchmarkSuiteID = .tinyDeterministicLogitsV1,
        reason: String
    ) -> TurboQuantQualityGateReport {
        TurboQuantQualityGateReport(
            benchmarkSuiteID: benchmarkSuiteID,
            deterministicTop1MatchRate: 0,
            logitKLDivergenceMean: Double.greatestFiniteMagnitude,
            logitMaxAbsErrorP95: Double.greatestFiniteMagnitude,
            noNaNOrInf: false,
            fallbackEquivalent: false,
            prefillExact: false,
            snapshotRoundtripEquivalent: nil,
            gateReason: reason,
            passed: false
        )
    }

    private static func finiteOrDefault(_ value: Double, default defaultValue: Double) -> Double {
        value.isFinite ? value : defaultValue
    }

    private static func finiteOptional(_ value: Double?) -> Double? {
        guard let value, value.isFinite else { return nil }
        return value
    }
}
