// Copyright (c) 2026 RNT56.

import Foundation

public enum TurboQuantPlatformPolicyError: Error, Equatable, CustomStringConvertible {
    case invalidSchemaVersion(Int)
    case activeWithoutEvidence(feature: String)
    case identityMismatch(field: String, expected: String, actual: String)
    case invalidPrecisionSegment(String)
    case invalidDescriptor(String)

    public var description: String {
        switch self {
        case .invalidSchemaVersion(let version):
            "unsupported TurboQuant platform policy schema version \(version)"
        case .activeWithoutEvidence(let feature):
            "TurboQuant platform feature \(feature) cannot be active without verified evidence"
        case .identityMismatch(let field, let expected, let actual):
            "TurboQuant platform identity mismatch for \(field): expected \(expected), got \(actual)"
        case .invalidPrecisionSegment(let message):
            "invalid TurboQuant adaptive precision segment: \(message)"
        case .invalidDescriptor(let message):
            "invalid TurboQuant open KV descriptor: \(message)"
        }
    }
}

public enum TurboQuantPlatformFeature: String, Codable, Sendable, CaseIterable {
    case adaptivePrecision = "adaptive_precision"
    case openKVFormat = "open_kv_format"
    case platformCapabilityReporting = "platform_capability_reporting"
}

public enum TurboQuantPlatformFeatureGateState: String, Codable, Sendable, CaseIterable {
    case disabled
    case reportOnly = "report_only"
    case active
}

public struct TurboQuantPlatformEvidence: Hashable, Codable, Sendable {
    public var evidenceID: String
    public var compatibilityPairID: String
    public var benchmarkSuiteID: String
    public var qualityGatePassed: Bool
    public var generatedAt: Date

    public init(
        evidenceID: String,
        compatibilityPairID: String,
        benchmarkSuiteID: String,
        qualityGatePassed: Bool,
        generatedAt: Date = Date()
    ) {
        self.evidenceID = evidenceID
        self.compatibilityPairID = compatibilityPairID
        self.benchmarkSuiteID = benchmarkSuiteID
        self.qualityGatePassed = qualityGatePassed
        self.generatedAt = generatedAt
    }

    public var permitsActivation: Bool {
        !evidenceID.isEmpty && !compatibilityPairID.isEmpty
            && !benchmarkSuiteID.isEmpty && qualityGatePassed
    }
}

public struct TurboQuantPlatformFeatureGate: Hashable, Codable, Sendable {
    public var feature: TurboQuantPlatformFeature
    public var state: TurboQuantPlatformFeatureGateState
    public var evidence: TurboQuantPlatformEvidence?
    public var reason: String?

    public init(
        feature: TurboQuantPlatformFeature,
        state: TurboQuantPlatformFeatureGateState = .disabled,
        evidence: TurboQuantPlatformEvidence? = nil,
        reason: String? = nil
    ) {
        self.feature = feature
        self.state = state
        self.evidence = evidence
        self.reason = reason
    }

    public static func disabled(
        _ feature: TurboQuantPlatformFeature,
        reason: String? = nil
    ) -> TurboQuantPlatformFeatureGate {
        TurboQuantPlatformFeatureGate(feature: feature, state: .disabled, reason: reason)
    }

    public func validate() throws {
        guard state == .active else { return }
        guard evidence?.permitsActivation == true else {
            throw TurboQuantPlatformPolicyError.activeWithoutEvidence(feature: feature.rawValue)
        }
    }
}

public struct TurboQuantPlatformIdentity: Hashable, Codable, Sendable {
    public var platformName: String
    public var osVersion: String
    public var deviceModel: String
    public var mlxSwiftRevision: String?
    public var mlxSwiftLMRevision: String?
    public var metalFeatureSet: String?

    public init(
        platformName: String,
        osVersion: String,
        deviceModel: String,
        mlxSwiftRevision: String? = nil,
        mlxSwiftLMRevision: String? = nil,
        metalFeatureSet: String? = nil
    ) {
        self.platformName = platformName
        self.osVersion = osVersion
        self.deviceModel = deviceModel
        self.mlxSwiftRevision = mlxSwiftRevision
        self.mlxSwiftLMRevision = mlxSwiftLMRevision
        self.metalFeatureSet = metalFeatureSet
    }

    public func validate(matches expected: TurboQuantPlatformIdentity) throws {
        try requireEqual("platformName", expected.platformName, platformName)
        try requireEqual("osVersion", expected.osVersion, osVersion)
        try requireEqual("deviceModel", expected.deviceModel, deviceModel)
        try requireEqual(
            "mlxSwiftRevision", expected.mlxSwiftRevision ?? "", mlxSwiftRevision ?? "")
        try requireEqual(
            "mlxSwiftLMRevision",
            expected.mlxSwiftLMRevision ?? "",
            mlxSwiftLMRevision ?? ""
        )
        try requireEqual("metalFeatureSet", expected.metalFeatureSet ?? "", metalFeatureSet ?? "")
    }

    private func requireEqual(_ field: String, _ expected: String, _ actual: String) throws {
        guard expected == actual else {
            throw TurboQuantPlatformPolicyError.identityMismatch(
                field: field,
                expected: expected,
                actual: actual
            )
        }
    }
}

public enum TurboQuantPrecisionTensorRole: String, Codable, Sendable, CaseIterable {
    case key
    case value
    case attentionOutput = "attention_output"
}

public struct TurboQuantPrecisionSegment: Hashable, Codable, Sendable {
    public var role: TurboQuantPrecisionTensorRole
    public var layerRange: Range<Int>
    public var tokenRange: Range<Int>?
    public var magnitudeBits: Int
    public var residualBits: Int
    public var evidenceTag: String?

    public init(
        role: TurboQuantPrecisionTensorRole,
        layerRange: Range<Int>,
        tokenRange: Range<Int>? = nil,
        magnitudeBits: Int,
        residualBits: Int = 1,
        evidenceTag: String? = nil
    ) {
        self.role = role
        self.layerRange = layerRange
        self.tokenRange = tokenRange
        self.magnitudeBits = magnitudeBits
        self.residualBits = residualBits
        self.evidenceTag = evidenceTag
    }

    public func validate() throws {
        guard !layerRange.isEmpty, layerRange.lowerBound >= 0 else {
            throw TurboQuantPlatformPolicyError.invalidPrecisionSegment(
                "layer range must be non-empty and non-negative"
            )
        }
        if let tokenRange {
            guard !tokenRange.isEmpty, tokenRange.lowerBound >= 0 else {
                throw TurboQuantPlatformPolicyError.invalidPrecisionSegment(
                    "token range must be non-empty and non-negative"
                )
            }
        }
        guard (1 ... 8).contains(magnitudeBits), (0 ... 8).contains(residualBits) else {
            throw TurboQuantPlatformPolicyError.invalidPrecisionSegment(
                "precision bits must fit in the supported 0...8 bit contract"
            )
        }
    }
}

public struct TurboQuantAdaptivePrecisionPolicy: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var gate: TurboQuantPlatformFeatureGate
    public var defaultMagnitudeBits: Int
    public var defaultResidualBits: Int
    public var segments: [TurboQuantPrecisionSegment]

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        gate: TurboQuantPlatformFeatureGate = .disabled(.adaptivePrecision),
        defaultMagnitudeBits: Int = 4,
        defaultResidualBits: Int = 1,
        segments: [TurboQuantPrecisionSegment] = []
    ) {
        self.schemaVersion = schemaVersion
        self.gate = gate
        self.defaultMagnitudeBits = defaultMagnitudeBits
        self.defaultResidualBits = defaultResidualBits
        self.segments = segments
    }

    public static let disabled = TurboQuantAdaptivePrecisionPolicy()

    public func validate() throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw TurboQuantPlatformPolicyError.invalidSchemaVersion(schemaVersion)
        }
        guard gate.feature == .adaptivePrecision else {
            throw TurboQuantPlatformPolicyError.invalidDescriptor(
                "adaptive precision policy must use the adaptive precision feature gate"
            )
        }
        guard (1 ... 8).contains(defaultMagnitudeBits), (0 ... 8).contains(defaultResidualBits)
        else {
            throw TurboQuantPlatformPolicyError.invalidPrecisionSegment(
                "default precision bits must fit in the supported 0...8 bit contract"
            )
        }
        try gate.validate()
        try segments.forEach { try $0.validate() }
    }
}

public struct TurboQuantOpenKVFormatIdentity: Hashable, Codable, Sendable {
    public var formatName: String
    public var formatVersion: Int
    public var modelID: String
    public var modelRevision: String?
    public var tokenizerHash: String
    public var profileHash: String
    public var ropeConfigHash: String
    public var tokenPrefixHash: String
    public var fallbackContractHash: String?

    public init(
        formatName: String = "turboquant-open-kv",
        formatVersion: Int = 1,
        modelID: String,
        modelRevision: String? = nil,
        tokenizerHash: String,
        profileHash: String,
        ropeConfigHash: String,
        tokenPrefixHash: String,
        fallbackContractHash: String? = nil
    ) {
        self.formatName = formatName
        self.formatVersion = formatVersion
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.tokenizerHash = tokenizerHash
        self.profileHash = profileHash
        self.ropeConfigHash = ropeConfigHash
        self.tokenPrefixHash = tokenPrefixHash
        self.fallbackContractHash = fallbackContractHash
    }

    public init(snapshotIdentity: TurboQuantKVSnapshotIdentity) {
        self.init(
            modelID: snapshotIdentity.modelID,
            modelRevision: snapshotIdentity.modelRevision,
            tokenizerHash: snapshotIdentity.tokenizerHash,
            profileHash: snapshotIdentity.profileHash,
            ropeConfigHash: snapshotIdentity.ropeConfigHash,
            tokenPrefixHash: snapshotIdentity.tokenPrefixHash,
            fallbackContractHash: snapshotIdentity.fallbackContractHash
        )
    }

    public func validate(matches expected: TurboQuantOpenKVFormatIdentity) throws {
        try requireEqual("formatName", expected.formatName, formatName)
        try requireEqual("formatVersion", "\(expected.formatVersion)", "\(formatVersion)")
        try requireEqual("modelID", expected.modelID, modelID)
        try requireEqual("modelRevision", expected.modelRevision ?? "", modelRevision ?? "")
        try requireEqual("tokenizerHash", expected.tokenizerHash, tokenizerHash)
        try requireEqual("profileHash", expected.profileHash, profileHash)
        try requireEqual("ropeConfigHash", expected.ropeConfigHash, ropeConfigHash)
        try requireEqual("tokenPrefixHash", expected.tokenPrefixHash, tokenPrefixHash)
        try requireEqual(
            "fallbackContractHash",
            expected.fallbackContractHash ?? "",
            fallbackContractHash ?? ""
        )
    }

    private func requireEqual(_ field: String, _ expected: String, _ actual: String) throws {
        guard expected == actual else {
            throw TurboQuantPlatformPolicyError.identityMismatch(
                field: field,
                expected: expected,
                actual: actual
            )
        }
    }
}

public struct TurboQuantOpenKVTensorDescriptor: Hashable, Codable, Sendable {
    public var name: String
    public var dtype: String
    public var shape: [Int]
    public var byteCount: Int64
    public var role: TurboQuantPrecisionTensorRole

    public init(
        name: String,
        dtype: String,
        shape: [Int],
        byteCount: Int64,
        role: TurboQuantPrecisionTensorRole
    ) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.byteCount = byteCount
        self.role = role
    }
}

public struct TurboQuantOpenKVFormatDescriptor: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var identity: TurboQuantOpenKVFormatIdentity
    public var gate: TurboQuantPlatformFeatureGate
    public var layoutVersion: Int
    public var keyEncoding: String
    public var valueEncoding: String
    public var groupSize: Int
    public var valueBits: Int
    public var tensors: [TurboQuantOpenKVTensorDescriptor]

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        identity: TurboQuantOpenKVFormatIdentity,
        gate: TurboQuantPlatformFeatureGate = .disabled(.openKVFormat),
        layoutVersion: Int,
        keyEncoding: String,
        valueEncoding: String,
        groupSize: Int,
        valueBits: Int,
        tensors: [TurboQuantOpenKVTensorDescriptor] = []
    ) {
        self.schemaVersion = schemaVersion
        self.identity = identity
        self.gate = gate
        self.layoutVersion = layoutVersion
        self.keyEncoding = keyEncoding
        self.valueEncoding = valueEncoding
        self.groupSize = groupSize
        self.valueBits = valueBits
        self.tensors = tensors
    }

    public func validate(expectedIdentity: TurboQuantOpenKVFormatIdentity) throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw TurboQuantPlatformPolicyError.invalidSchemaVersion(schemaVersion)
        }
        guard gate.feature == .openKVFormat else {
            throw TurboQuantPlatformPolicyError.invalidDescriptor(
                "open KV descriptor must use the open KV feature gate"
            )
        }
        try gate.validate()
        try identity.validate(matches: expectedIdentity)
        guard layoutVersion > 0, groupSize > 0, (1 ... 8).contains(valueBits) else {
            throw TurboQuantPlatformPolicyError.invalidDescriptor(
                "layout version, group size, and value bits must be positive"
            )
        }
        guard !keyEncoding.isEmpty, !valueEncoding.isEmpty else {
            throw TurboQuantPlatformPolicyError.invalidDescriptor(
                "key and value encodings must be present"
            )
        }
        for tensor in tensors {
            guard !tensor.name.isEmpty, !tensor.dtype.isEmpty, tensor.byteCount >= 0,
                tensor.shape.allSatisfy({ $0 > 0 })
            else {
                throw TurboQuantPlatformPolicyError.invalidDescriptor(
                    "tensor descriptors must include name, dtype, positive shape, and byte count"
                )
            }
        }
    }
}

public struct TurboQuantPlatformCapability: Hashable, Codable, Sendable {
    public var feature: TurboQuantPlatformFeature
    public var supported: Bool
    public var reason: String?

    public init(
        feature: TurboQuantPlatformFeature,
        supported: Bool,
        reason: String? = nil
    ) {
        self.feature = feature
        self.supported = supported
        self.reason = reason
    }
}

public struct TurboQuantPlatformCapabilityReport: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var identity: TurboQuantPlatformIdentity
    public var generatedAt: Date
    public var capabilities: [TurboQuantPlatformCapability]
    public var gates: [TurboQuantPlatformFeatureGate]

    public init(
        schemaVersion: Int = Self.currentSchemaVersion,
        identity: TurboQuantPlatformIdentity,
        generatedAt: Date = Date(),
        capabilities: [TurboQuantPlatformCapability] = [],
        gates: [TurboQuantPlatformFeatureGate] = []
    ) {
        self.schemaVersion = schemaVersion
        self.identity = identity
        self.generatedAt = generatedAt
        self.capabilities = capabilities
        self.gates = gates
    }

    public static func disabled(
        identity: TurboQuantPlatformIdentity,
        generatedAt: Date = Date()
    ) -> TurboQuantPlatformCapabilityReport {
        TurboQuantPlatformCapabilityReport(
            identity: identity,
            generatedAt: generatedAt,
            capabilities: TurboQuantPlatformFeature.allCases.map {
                TurboQuantPlatformCapability(feature: $0, supported: false)
            },
            gates: TurboQuantPlatformFeature.allCases.map {
                TurboQuantPlatformFeatureGate.disabled($0)
            }
        )
    }

    public func validate(expectedIdentity: TurboQuantPlatformIdentity) throws {
        guard schemaVersion == Self.currentSchemaVersion else {
            throw TurboQuantPlatformPolicyError.invalidSchemaVersion(schemaVersion)
        }
        try identity.validate(matches: expectedIdentity)
        try gates.forEach { try $0.validate() }
    }

    public func supports(_ feature: TurboQuantPlatformFeature) -> Bool {
        (try? validate(expectedIdentity: identity)) != nil
            && capabilities.contains { $0.feature == feature && $0.supported }
            && gates.contains { $0.feature == feature && $0.state == .active }
    }
}
