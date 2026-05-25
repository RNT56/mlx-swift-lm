import Foundation
import Testing

@testable import MLXLMCommon

@Suite("TurboQuant Wave 7 platform contracts")
struct TurboQuantWave7PlatformTests {
    @Test func adaptivePrecisionPolicyIsCodableAndDisabledByDefault() throws {
        let policy = TurboQuantAdaptivePrecisionPolicy.disabled

        let data = try JSONEncoder().encode(policy)
        let decoded = try JSONDecoder().decode(TurboQuantAdaptivePrecisionPolicy.self, from: data)

        try decoded.validate()
        #expect(decoded == policy)
        #expect(decoded.gate.feature == .adaptivePrecision)
        #expect(decoded.gate.state == .disabled)
        #expect(decoded.segments.isEmpty)
    }

    @Test func activeAdaptivePrecisionFailsClosedWithoutEvidence() {
        let policy = TurboQuantAdaptivePrecisionPolicy(
            gate: TurboQuantPlatformFeatureGate(feature: .adaptivePrecision, state: .active),
            segments: [
                TurboQuantPrecisionSegment(
                    role: .key,
                    layerRange: 0 ..< 4,
                    magnitudeBits: 5
                )
            ]
        )

        #expect(throws: TurboQuantPlatformPolicyError.self) {
            try policy.validate()
        }
    }

    @Test func activeAdaptivePrecisionAllowsVerifiedEvidenceContract() throws {
        let policy = TurboQuantAdaptivePrecisionPolicy(
            gate: TurboQuantPlatformFeatureGate(
                feature: .adaptivePrecision,
                state: .active,
                evidence: TurboQuantPlatformEvidence(
                    evidenceID: "evidence-1",
                    compatibilityPairID: "pair-1",
                    benchmarkSuiteID: TurboQuantBenchmarkSuiteID.prefillExactnessV1.rawValue,
                    qualityGatePassed: true,
                    generatedAt: Date(timeIntervalSinceReferenceDate: 10)
                )
            ),
            segments: [
                TurboQuantPrecisionSegment(
                    role: .value,
                    layerRange: 2 ..< 8,
                    tokenRange: 128 ..< 1024,
                    magnitudeBits: 4,
                    residualBits: 1,
                    evidenceTag: "evidence-1"
                )
            ]
        )

        try policy.validate()
    }

    @Test func openKVDescriptorRoundTripsAndRejectsIdentityMismatch() throws {
        let identity = Self.openKVIdentity()
        let descriptor = TurboQuantOpenKVFormatDescriptor(
            identity: identity,
            layoutVersion: 4,
            keyEncoding: "polar-qjl",
            valueEncoding: "packed-4bit",
            groupSize: 64,
            valueBits: 4,
            tensors: [
                TurboQuantOpenKVTensorDescriptor(
                    name: "key.packedMagnitudes",
                    dtype: "uint32",
                    shape: [1, 2, 16, 2],
                    byteCount: 256,
                    role: .key
                )
            ]
        )

        let data = try JSONEncoder().encode(descriptor)
        let decoded = try JSONDecoder().decode(TurboQuantOpenKVFormatDescriptor.self, from: data)
        try decoded.validate(expectedIdentity: identity)

        var mismatched = identity
        mismatched.tokenPrefixHash = "other-prefix"
        #expect(throws: TurboQuantPlatformPolicyError.self) {
            try decoded.validate(expectedIdentity: mismatched)
        }
    }

    @Test func platformCapabilityReportFailsClosedForIdentityAndActiveGateEvidence() throws {
        let identity = TurboQuantPlatformIdentity(
            platformName: "iOS",
            osVersion: "19.0",
            deviceModel: "iPhone-test",
            mlxSwiftRevision: "mlx-1",
            mlxSwiftLMRevision: "lm-1",
            metalFeatureSet: "apple9"
        )
        let disabled = TurboQuantPlatformCapabilityReport.disabled(
            identity: identity,
            generatedAt: Date(timeIntervalSinceReferenceDate: 20)
        )
        try disabled.validate(expectedIdentity: identity)
        #expect(!disabled.supports(.adaptivePrecision))

        var wrongIdentity = identity
        wrongIdentity.mlxSwiftLMRevision = "lm-2"
        #expect(throws: TurboQuantPlatformPolicyError.self) {
            try disabled.validate(expectedIdentity: wrongIdentity)
        }

        let activeWithoutEvidence = TurboQuantPlatformCapabilityReport(
            identity: identity,
            gates: [
                TurboQuantPlatformFeatureGate(feature: .openKVFormat, state: .active)
            ]
        )
        #expect(throws: TurboQuantPlatformPolicyError.self) {
            try activeWithoutEvidence.validate(expectedIdentity: identity)
        }
    }

    private static func openKVIdentity() -> TurboQuantOpenKVFormatIdentity {
        TurboQuantOpenKVFormatIdentity(
            modelID: "model-a",
            modelRevision: "rev-a",
            tokenizerHash: "tokenizer-a",
            profileHash: "profile-a",
            ropeConfigHash: "rope-a",
            tokenPrefixHash: "prefix-a",
            fallbackContractHash: "fallback-a"
        )
    }
}
