// Copyright (c) 2026 RNT56.

import Foundation
import MLX

public enum TurboQuantCheckpointMetadataIssueKind: String, Codable, Sendable {
    case missingField
    case invalidField
    case unsupportedSchemaVersion
    case unsupportedLayoutVersion
    case mismatchedProfile
}

public struct TurboQuantCheckpointMetadataIssue: Codable, Equatable, Sendable {
    public var field: String
    public var kind: TurboQuantCheckpointMetadataIssueKind
    public var expected: String?
    public var actual: String?
    public var reason: String

    public init(
        field: String,
        kind: TurboQuantCheckpointMetadataIssueKind,
        expected: String? = nil,
        actual: String? = nil,
        reason: String
    ) {
        self.field = field
        self.kind = kind
        self.expected = expected
        self.actual = actual
        self.reason = reason
    }
}

public enum TurboQuantCheckpointMetadataValidationError: Error, LocalizedError, Equatable {
    case invalidMetadata([TurboQuantCheckpointMetadataIssue])

    public var issues: [TurboQuantCheckpointMetadataIssue] {
        switch self {
        case .invalidMetadata(let issues):
            issues
        }
    }

    public var errorDescription: String? {
        switch self {
        case .invalidMetadata(let issues):
            let fields = issues.map(\.field).joined(separator: ", ")
            return "Invalid TurboQuant checkpoint metadata: \(fields)"
        }
    }
}

public struct TurboQuantCheckpointMetadataPolicy: Codable, Equatable, Sendable {
    public var supportedMetadataSchemaVersions: [Int]
    public var supportedLayoutVersions: [Int]
    public var expectedProfileName: String?

    public init(
        supportedMetadataSchemaVersions: [Int] = [2],
        supportedLayoutVersions: [Int] = [TurboQuantAttentionLayout.currentVersion],
        expectedProfileName: String? = nil
    ) {
        self.supportedMetadataSchemaVersions = supportedMetadataSchemaVersions
        self.supportedLayoutVersions = supportedLayoutVersions
        self.expectedProfileName = expectedProfileName
    }
}

public enum TurboQuantCheckpointMetadataValidator {
    public static let metadataSchemaVersionKey = "turboquant_metadata_schema_version"
    public static let layoutVersionKey = "turboquant_layout_version"

    private static let requiredStringFields = [
        "turboquant_seed_policy",
        "turboquant_key_format",
        "turboquant_value_format",
        "turboquant_linear_format",
        "turboquant_codebook_version",
        "turboquant_rotation_version",
        "turboquant_mlx_commit",
        "turboquant_mlx_swift_commit",
        "turboquant_mlx_swift_lm_commit",
        "turboquant_converter_version",
        "turboquant_conversion_device",
        "turboquant_conversion_date",
        "turboquant_converted_tensor_hash",
        "turboquant_profile_name",
    ]

    private static let requiredIntegerFields = [
        "turboquant_group_size",
        "turboquant_bits",
        "turboquant_value_bits",
        "turboquant_converted_tensors",
    ]

    public static func isTurboQuantCheckpoint(_ metadata: [String: String]) -> Bool {
        metadata["quant_method"]?.lowercased() == "turboquant"
            || metadata["linear_class"]?.lowercased() == "turboquantlinear"
    }

    public static func validate(
        _ metadata: [String: String],
        policy: TurboQuantCheckpointMetadataPolicy = TurboQuantCheckpointMetadataPolicy()
    ) throws {
        let issues = validationIssues(metadata, policy: policy)
        guard issues.isEmpty else {
            throw TurboQuantCheckpointMetadataValidationError.invalidMetadata(issues)
        }
    }

    public static func validationIssues(
        _ metadata: [String: String],
        policy: TurboQuantCheckpointMetadataPolicy = TurboQuantCheckpointMetadataPolicy()
    ) -> [TurboQuantCheckpointMetadataIssue] {
        guard isTurboQuantCheckpoint(metadata) else {
            return []
        }

        var issues = [TurboQuantCheckpointMetadataIssue]()
        requireString(
            "quant_method",
            in: metadata,
            expected: "turboquant",
            issues: &issues
        )
        requireString(
            "linear_class",
            in: metadata,
            expected: "TurboQuantLinear",
            caseSensitive: false,
            issues: &issues
        )
        requireString("turboquant_format", in: metadata, expected: "mlx_packed", issues: &issues)
        requireString("turboquant_preset", in: metadata, issues: &issues)
        requireString("turboquant_mode", in: metadata, issues: &issues)
        requireString("turboquant_seed", in: metadata, issues: &issues)

        validateInteger(
            metadataSchemaVersionKey,
            in: metadata,
            issues: &issues,
            kind: .unsupportedSchemaVersion
        ) { value in
            policy.supportedMetadataSchemaVersions.contains(value)
        }
        validateInteger(
            layoutVersionKey,
            in: metadata,
            issues: &issues,
            kind: .unsupportedLayoutVersion
        ) { value in
            policy.supportedLayoutVersions.contains(value)
        }

        for field in requiredStringFields {
            requireString(field, in: metadata, issues: &issues)
        }
        for field in requiredIntegerFields {
            validateInteger(field, in: metadata, issues: &issues) { $0 >= 0 }
        }

        if let expectedProfileName = policy.expectedProfileName,
            metadata["turboquant_profile_name"] != expectedProfileName
        {
            issues.append(
                TurboQuantCheckpointMetadataIssue(
                    field: "turboquant_profile_name",
                    kind: .mismatchedProfile,
                    expected: expectedProfileName,
                    actual: metadata["turboquant_profile_name"],
                    reason: "checkpoint profile name does not match the selected TurboQuant profile"
                )
            )
        }

        return issues
    }

    public static func requiredMetadata(
        profileName: String,
        convertedTensorHash: String,
        metadataSchemaVersion: Int = 2,
        layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
        seedPolicy: String = "deterministic",
        keyFormat: String = "turboquant_prod",
        valueFormat: String = "affine_value",
        linearFormat: String = "mlx_packed",
        codebookVersion: String = "none",
        rotationVersion: String = "none",
        mlxCommit: String = "unknown",
        mlxSwiftCommit: String = "unknown",
        mlxSwiftLMCommit: String = "unknown",
        converterVersion: String = "unknown",
        conversionDevice: String = "unknown",
        conversionDate: String = "unknown"
    ) -> [String: String] {
        [
            metadataSchemaVersionKey: String(metadataSchemaVersion),
            layoutVersionKey: String(layoutVersion),
            "turboquant_seed_policy": seedPolicy,
            "turboquant_key_format": keyFormat,
            "turboquant_value_format": valueFormat,
            "turboquant_linear_format": linearFormat,
            "turboquant_codebook_version": codebookVersion,
            "turboquant_rotation_version": rotationVersion,
            "turboquant_mlx_commit": mlxCommit,
            "turboquant_mlx_swift_commit": mlxSwiftCommit,
            "turboquant_mlx_swift_lm_commit": mlxSwiftLMCommit,
            "turboquant_converter_version": converterVersion,
            "turboquant_conversion_device": conversionDevice,
            "turboquant_conversion_date": conversionDate,
            "turboquant_converted_tensor_hash": convertedTensorHash,
            "turboquant_profile_name": profileName,
        ]
    }

    private static func requireString(
        _ field: String,
        in metadata: [String: String],
        expected: String? = nil,
        caseSensitive: Bool = true,
        issues: inout [TurboQuantCheckpointMetadataIssue]
    ) {
        guard let value = metadata[field], !value.isEmpty else {
            issues.append(
                TurboQuantCheckpointMetadataIssue(
                    field: field,
                    kind: .missingField,
                    expected: expected,
                    actual: metadata[field],
                    reason: "required metadata field is missing"
                )
            )
            return
        }
        guard let expected else { return }
        let lhs = caseSensitive ? value : value.lowercased()
        let rhs = caseSensitive ? expected : expected.lowercased()
        if lhs != rhs {
            issues.append(
                TurboQuantCheckpointMetadataIssue(
                    field: field,
                    kind: .invalidField,
                    expected: expected,
                    actual: value,
                    reason: "metadata field has an unsupported value"
                )
            )
        }
    }

    private static func validateInteger(
        _ field: String,
        in metadata: [String: String],
        issues: inout [TurboQuantCheckpointMetadataIssue],
        kind: TurboQuantCheckpointMetadataIssueKind = .invalidField,
        isSupported: (Int) -> Bool
    ) {
        guard let raw = metadata[field], !raw.isEmpty else {
            issues.append(
                TurboQuantCheckpointMetadataIssue(
                    field: field,
                    kind: .missingField,
                    actual: metadata[field],
                    reason: "required integer metadata field is missing"
                )
            )
            return
        }
        guard let value = Int(raw) else {
            issues.append(
                TurboQuantCheckpointMetadataIssue(
                    field: field,
                    kind: .invalidField,
                    actual: raw,
                    reason: "metadata field must be an integer"
                )
            )
            return
        }
        if !isSupported(value) {
            issues.append(
                TurboQuantCheckpointMetadataIssue(
                    field: field,
                    kind: kind,
                    actual: raw,
                    reason: "metadata field is outside the supported range"
                )
            )
        }
    }
}
