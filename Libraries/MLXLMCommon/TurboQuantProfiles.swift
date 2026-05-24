import Foundation
import MLX

/// Public aliases for TurboQuant KV-cache schemes used by server and profile layers.
///
/// The runtime implementation is still expressed in terms of ``TurboQuantPreset``.
/// These aliases preserve the scheme vocabulary used by SwiftLM/vLLM-style
/// configuration while mapping to the concrete MLX-Swift cache backends.
public enum TurboQuantScheme: String, Codable, Sendable, CaseIterable {
    case disabled
    case turbo4v2
    case turbo3_5
    case turbo3
    case turbo2_5

    public init?(normalizing value: String) {
        let normalized =
            value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_")

        switch normalized {
        case "none", "off", "disabled", "false":
            self = .disabled
        case "turbo4", "turbo4v2", "turbo_4_v2":
            self = .turbo4v2
        case "turbo3_5", "turbo35", "turbo_3_5":
            self = .turbo3_5
        case "turbo3", "turbo_3":
            self = .turbo3
        case "turbo2_5", "turbo25", "turbo_2_5":
            self = .turbo2_5
        default:
            return nil
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        guard let scheme = TurboQuantScheme(normalizing: value) else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unknown TurboQuant scheme: \(value)"
            )
        }
        self = scheme
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    public var kvCacheStrategy: KVCacheStrategy {
        self == .disabled ? .mlxAffine : .turboQuant
    }

    public var preset: TurboQuantPreset {
        switch self {
        case .disabled, .turbo4v2:
            .turbo4v2
        case .turbo3_5:
            .turbo3_5
        case .turbo3, .turbo2_5:
            .turbo2_5
        }
    }

    public var defaultValueBits: Int? {
        switch self {
        case .disabled:
            nil
        case .turbo4v2, .turbo3_5:
            4
        case .turbo3, .turbo2_5:
            2
        }
    }

    public var defaultOptimizationPolicy: TurboQuantOptimizationPolicy {
        switch self {
        case .disabled, .turbo4v2, .turbo3_5:
            .auto
        case .turbo3, .turbo2_5:
            .preferMemory
        }
    }
}

public enum TurboQuantMaskMode: String, Codable, Sendable, CaseIterable {
    case none
    case causal
    case array
    case arrays
}

public enum TurboQuantQualityProfile: String, Codable, Sendable, CaseIterable {
    case conservative
    case balanced
    case memory
    case experimental
}

public enum TurboQuantProfileStatus: String, Codable, Sendable, CaseIterable {
    case experimental
    case validated
    case deprecated
}

public enum TurboQuantModelModality: String, Codable, Sendable, CaseIterable {
    case text
    case visionText
    case videoText
}

public struct TurboQuantRoPEFingerprint: Codable, Equatable, Sendable {
    public var type: String?
    public var theta: Double?
    public var scalingType: String?
    public var scalingFactor: Double?
    public var originalMaxPositionEmbeddings: Int?

    public init(
        type: String? = nil,
        theta: Double? = nil,
        scalingType: String? = nil,
        scalingFactor: Double? = nil,
        originalMaxPositionEmbeddings: Int? = nil
    ) {
        self.type = type
        self.theta = theta
        self.scalingType = scalingType
        self.scalingFactor = scalingFactor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
    }

    public var missingRequiredFields: [String] {
        var fields = [String]()
        if type == nil { fields.append("model_fingerprint.rope.type") }
        if theta == nil { fields.append("model_fingerprint.rope.theta") }
        return fields
    }
}

public struct TurboQuantSlidingWindowFingerprint: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var size: Int?

    public init(enabled: Bool, size: Int? = nil) {
        self.enabled = enabled
        self.size = size
    }

    public var missingRequiredFields: [String] {
        enabled && size == nil ? ["model_fingerprint.sliding_window.size"] : []
    }
}

public struct TurboQuantModelFingerprint: Codable, Equatable, Sendable {
    public var family: String?
    public var hiddenSize: Int?
    public var layerCount: Int?
    public var attentionHeads: Int?
    public var kvHeads: Int?
    public var headDim: Int?
    public var rope: TurboQuantRoPEFingerprint?
    public var slidingWindow: TurboQuantSlidingWindowFingerprint?
    public var cacheType: String?

    public init(
        family: String? = nil,
        hiddenSize: Int? = nil,
        layerCount: Int? = nil,
        attentionHeads: Int? = nil,
        kvHeads: Int? = nil,
        headDim: Int? = nil,
        rope: TurboQuantRoPEFingerprint? = nil,
        slidingWindow: TurboQuantSlidingWindowFingerprint? = nil,
        cacheType: String? = nil
    ) {
        self.family = family
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.rope = rope
        self.slidingWindow = slidingWindow
        self.cacheType = cacheType
    }

    public var missingRequiredFields: [String] {
        var fields = [String]()
        if family == nil { fields.append("model_fingerprint.family") }
        if hiddenSize == nil { fields.append("model_fingerprint.hidden_size") }
        if layerCount == nil { fields.append("model_fingerprint.layer_count") }
        if attentionHeads == nil { fields.append("model_fingerprint.attention_heads") }
        if kvHeads == nil { fields.append("model_fingerprint.kv_heads") }
        if headDim == nil { fields.append("model_fingerprint.head_dim") }
        if let rope {
            fields.append(contentsOf: rope.missingRequiredFields)
        } else {
            fields.append("model_fingerprint.rope")
        }
        if slidingWindow == nil { fields.append("model_fingerprint.sliding_window") }
        if cacheType == nil { fields.append("model_fingerprint.cache_type") }
        return fields
    }

    public func mismatchIssues(
        comparedTo actual: TurboQuantModelFingerprint,
        profileID: String
    ) -> [TurboQuantProfileManifestIssue] {
        var issues = [TurboQuantProfileManifestIssue]()
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.family",
            expected: family.map(Self.normalizedIdentifier),
            actual: actual.family.map(Self.normalizedIdentifier)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.hidden_size",
            expected: hiddenSize.map(String.init),
            actual: actual.hiddenSize.map(String.init)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.layer_count",
            expected: layerCount.map(String.init),
            actual: actual.layerCount.map(String.init)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.attention_heads",
            expected: attentionHeads.map(String.init),
            actual: actual.attentionHeads.map(String.init)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.kv_heads",
            expected: kvHeads.map(String.init),
            actual: actual.kvHeads.map(String.init)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.head_dim",
            expected: headDim.map(String.init),
            actual: actual.headDim.map(String.init)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.rope.type",
            expected: rope?.type.map(Self.normalizedIdentifier),
            actual: actual.rope?.type.map(Self.normalizedIdentifier)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.rope.theta",
            expected: rope?.theta.map(Self.stableString),
            actual: actual.rope?.theta.map(Self.stableString)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.rope.scaling_type",
            expected: rope?.scalingType.map(Self.normalizedIdentifier),
            actual: actual.rope?.scalingType.map(Self.normalizedIdentifier)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.rope.scaling_factor",
            expected: rope?.scalingFactor.map(Self.stableString),
            actual: actual.rope?.scalingFactor.map(Self.stableString)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.sliding_window.enabled",
            expected: slidingWindow.map { String($0.enabled) },
            actual: actual.slidingWindow.map { String($0.enabled) }
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.sliding_window.size",
            expected: slidingWindow?.size.map(String.init),
            actual: actual.slidingWindow?.size.map(String.init)
        )
        Self.appendMismatch(
            &issues,
            profileID: profileID,
            field: "model_fingerprint.cache_type",
            expected: cacheType.map(Self.normalizedIdentifier),
            actual: actual.cacheType.map(Self.normalizedIdentifier)
        )
        return issues
    }

    private static func appendMismatch(
        _ issues: inout [TurboQuantProfileManifestIssue],
        profileID: String,
        field: String,
        expected: String?,
        actual: String?
    ) {
        guard let expected else { return }
        guard expected != actual else { return }
        issues.append(
            TurboQuantProfileManifestIssue(
                profileID: profileID,
                field: field,
                kind: .mismatch,
                expected: expected,
                actual: actual,
                reason: "model fingerprint field does not match"
            )
        )
    }

    private static func normalizedIdentifier(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_")
    }

    private static func stableString(_ value: Double) -> String {
        if value.rounded() == value {
            return String(Int(value))
        }
        return String(value)
    }
}

public struct TurboQuantProfileTurboQuantManifest: Codable, Equatable, Sendable {
    public var layoutVersion: Int
    public var keyPreset: TurboQuantScheme
    public var valueBits: Int
    public var groupSize: Int
    public var preferredPaths: [TurboQuantAttentionPath]
    public var fallbackPolicy: TurboQuantFallbackPolicy

    public init(
        layoutVersion: Int = TurboQuantAttentionLayout.currentVersion,
        keyPreset: TurboQuantScheme,
        valueBits: Int,
        groupSize: Int,
        preferredPaths: [TurboQuantAttentionPath] = [
            .onlineFused,
            .tiledOnlineFused,
            .twoStageCompressed,
            .mlxPackedFallback,
            .baseline,
        ],
        fallbackPolicy: TurboQuantFallbackPolicy = .compressedDecodeAllowed
    ) {
        self.layoutVersion = layoutVersion
        self.keyPreset = keyPreset
        self.valueBits = valueBits
        self.groupSize = groupSize
        self.preferredPaths = preferredPaths
        self.fallbackPolicy = fallbackPolicy
    }
}

public struct TurboQuantProfileMemoryMeasurement: Codable, Equatable, Sendable {
    public var compressedKVBytes: Int?
    public var fallbackBytes: Int?
    public var scratchBytes: Int?
    public var peakResidentBytes: Int?

    public init(
        compressedKVBytes: Int? = nil,
        fallbackBytes: Int? = nil,
        scratchBytes: Int? = nil,
        peakResidentBytes: Int? = nil
    ) {
        self.compressedKVBytes = compressedKVBytes
        self.fallbackBytes = fallbackBytes
        self.scratchBytes = scratchBytes
        self.peakResidentBytes = peakResidentBytes
    }
}

public struct TurboQuantMeasuredOutcome: Codable, Equatable, Sendable {
    public var deviceClass: String
    public var osVersion: String
    public var maxContextByMode: [String: Int]
    public var actualBytesPerToken: Double
    public var decodeP50Seconds: Double
    public var decodeP95Seconds: Double
    public var prefillP50Seconds: Double
    public var memory: TurboQuantProfileMemoryMeasurement
    public var logitKL: Double?
    public var top1MatchRate: Double?
    public var longContextRetrievalScore: Double?
    public var sourceBenchmarkID: String?
    public var measuredAt: String?

    public init(
        deviceClass: String,
        osVersion: String,
        maxContextByMode: [String: Int],
        actualBytesPerToken: Double,
        decodeP50Seconds: Double,
        decodeP95Seconds: Double,
        prefillP50Seconds: Double,
        memory: TurboQuantProfileMemoryMeasurement = TurboQuantProfileMemoryMeasurement(),
        logitKL: Double? = nil,
        top1MatchRate: Double? = nil,
        longContextRetrievalScore: Double? = nil,
        sourceBenchmarkID: String? = nil,
        measuredAt: String? = nil
    ) {
        self.deviceClass = deviceClass
        self.osVersion = osVersion
        self.maxContextByMode = maxContextByMode
        self.actualBytesPerToken = actualBytesPerToken
        self.decodeP50Seconds = decodeP50Seconds
        self.decodeP95Seconds = decodeP95Seconds
        self.prefillP50Seconds = prefillP50Seconds
        self.memory = memory
        self.logitKL = logitKL
        self.top1MatchRate = top1MatchRate
        self.longContextRetrievalScore = longContextRetrievalScore
        self.sourceBenchmarkID = sourceBenchmarkID
        self.measuredAt = measuredAt
    }
}

public enum TurboQuantProfileManifestIssueKind: String, Codable, Sendable {
    case missingField
    case mismatch
    case unsupportedSchemaVersion
    case inconsistentTurboQuantField
    case missingMeasuredOutcome
}

public struct TurboQuantProfileManifestIssue: Codable, Equatable, Sendable {
    public var profileID: String
    public var field: String
    public var kind: TurboQuantProfileManifestIssueKind
    public var expected: String?
    public var actual: String?
    public var reason: String

    public init(
        profileID: String,
        field: String,
        kind: TurboQuantProfileManifestIssueKind,
        expected: String? = nil,
        actual: String? = nil,
        reason: String
    ) {
        self.profileID = profileID
        self.field = field
        self.kind = kind
        self.expected = expected
        self.actual = actual
        self.reason = reason
    }
}

public struct TurboQuantProfileManifestValidation: Codable, Equatable, Sendable {
    public var profileID: String
    public var issues: [TurboQuantProfileManifestIssue]

    public init(profileID: String, issues: [TurboQuantProfileManifestIssue]) {
        self.profileID = profileID
        self.issues = issues
    }

    public var isValid: Bool { issues.isEmpty }
}

public struct TurboQuantProfileMeasurements: Codable, Equatable, Sendable {
    public var perplexityDelta: Double?
    public var promptTokensPerSecond: Double?
    public var generationTokensPerSecond: Double?
    public var maxResidentMemoryGB: Double?
    public var measuredContextLength: Int?
    public var measuredOn: String?
    public var source: String?
    public var notes: String?

    public init(
        perplexityDelta: Double? = nil,
        promptTokensPerSecond: Double? = nil,
        generationTokensPerSecond: Double? = nil,
        maxResidentMemoryGB: Double? = nil,
        measuredContextLength: Int? = nil,
        measuredOn: String? = nil,
        source: String? = nil,
        notes: String? = nil
    ) {
        self.perplexityDelta = perplexityDelta
        self.promptTokensPerSecond = promptTokensPerSecond
        self.generationTokensPerSecond = generationTokensPerSecond
        self.maxResidentMemoryGB = maxResidentMemoryGB
        self.measuredContextLength = measuredContextLength
        self.measuredOn = measuredOn
        self.source = source
        self.notes = notes
    }
}

public struct TurboQuantModelDescriptor: Equatable, Sendable {
    public var modelID: String
    public var modelType: String?
    public var textConfigModelType: String?
    public var modality: TurboQuantModelModality?
    public var parameterCountB: Double?
    public var routedExperts: Int?
    public var expertsPerToken: Int?
    public var fingerprint: TurboQuantModelFingerprint?

    public init(
        modelID: String,
        modelType: String? = nil,
        textConfigModelType: String? = nil,
        modality: TurboQuantModelModality? = nil,
        parameterCountB: Double? = nil,
        routedExperts: Int? = nil,
        expertsPerToken: Int? = nil,
        fingerprint: TurboQuantModelFingerprint? = nil
    ) {
        self.modelID = modelID
        self.modelType = modelType
        self.textConfigModelType = textConfigModelType
        self.modality = modality
        self.parameterCountB =
            parameterCountB ?? Self.inferParameterCountB(from: modelID)
        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.fingerprint = fingerprint
    }

    public static func inferParameterCountB(from modelID: String) -> Double? {
        let normalized =
            modelID
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
        let range = NSRange(normalized.startIndex ..< normalized.endIndex, in: normalized)
        let billionValues = Self.inferModelSizeValues(
            from: normalized,
            pattern: #"(?<![a-z0-9])(?:[ea])?([0-9]+(?:\.[0-9]+)?)b(?!it|its|yte|[a-z0-9])"#,
            range: range,
            multiplier: 1
        )
        let millionValues = Self.inferModelSizeValues(
            from: normalized,
            pattern: #"(?<![a-z0-9])([0-9]+(?:\.[0-9]+)?)m(?![a-z0-9])"#,
            range: range,
            multiplier: 0.001
        )
        return (billionValues + millionValues).max()
    }

    private static func inferModelSizeValues(
        from normalizedModelID: String,
        pattern: String,
        range: NSRange,
        multiplier: Double
    ) -> [Double] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }
        return regex.matches(in: normalizedModelID, range: range).compactMap { match -> Double? in
            guard let valueRange = Range(match.range(at: 1), in: normalizedModelID),
                let value = Double(normalizedModelID[valueRange])
            else {
                return nil
            }
            return value * multiplier
        }
    }
}

public struct TurboQuantProfile: Codable, Equatable, Identifiable, Sendable {
    public var schemaVersion: Int
    public var id: String
    public var exactModelIDs: [String]
    public var modelPatterns: [String]
    public var includePatterns: [String]
    public var excludePatterns: [String]
    public var architecture: String?
    public var modelTypes: [String]
    public var textConfigModelTypes: [String]
    public var modalities: [TurboQuantModelModality]
    public var minParametersB: Double?
    public var maxParametersB: Double?
    public var requiresModelType: Bool
    public var requiresTextConfigModelType: Bool
    public var requiresHeadDimensions: Bool
    public var minRoutedExperts: Int?
    public var maxRoutedExperts: Int?
    public var supportedExpertsPerToken: [Int]
    public var supportedKeyHeadDimensions: [Int]
    public var supportedValueHeadDimensions: [Int]
    public var recommendedScheme: TurboQuantScheme
    public var fallbackScheme: TurboQuantScheme?
    public var keyBits: Double
    public var valueBits: Int
    public var groupSize: Int
    public var safeMaskModes: [TurboQuantMaskMode]
    public var supportedContextLengths: [Int]
    public var safeContextLength: Int?
    public var qualityProfile: TurboQuantQualityProfile
    public var backend: TurboQuantBackend
    public var optimizationPolicy: TurboQuantOptimizationPolicy
    public var requiresMetalSelfTest: Bool
    public var requiresFusedAttentionSelfTest: Bool
    public var status: TurboQuantProfileStatus
    public var source: String?
    public var validatedOn: String?
    public var validatedBy: String?
    public var confidence: Double?
    public var measured: TurboQuantProfileMeasurements
    public var modelFingerprint: TurboQuantModelFingerprint?
    public var turboQuant: TurboQuantProfileTurboQuantManifest
    public var measuredOutcomes: [TurboQuantMeasuredOutcome]
    public var notes: [String]

    public init(
        schemaVersion: Int = 2,
        id: String,
        exactModelIDs: [String] = [],
        modelPatterns: [String],
        includePatterns: [String] = [],
        excludePatterns: [String] = [],
        architecture: String? = nil,
        modelTypes: [String] = [],
        textConfigModelTypes: [String] = [],
        modalities: [TurboQuantModelModality] = [.text],
        minParametersB: Double? = nil,
        maxParametersB: Double? = nil,
        requiresModelType: Bool = false,
        requiresTextConfigModelType: Bool = false,
        requiresHeadDimensions: Bool = false,
        minRoutedExperts: Int? = nil,
        maxRoutedExperts: Int? = nil,
        supportedExpertsPerToken: [Int] = [],
        supportedKeyHeadDimensions: [Int],
        supportedValueHeadDimensions: [Int]? = nil,
        recommendedScheme: TurboQuantScheme = .turbo4v2,
        fallbackScheme: TurboQuantScheme? = .turbo3_5,
        keyBits: Double = 3.5,
        valueBits: Int = 4,
        groupSize: Int = 64,
        safeMaskModes: [TurboQuantMaskMode] = [.none, .causal],
        supportedContextLengths: [Int] = [4096, 8192, 16384, 32768, 65536],
        safeContextLength: Int? = 65536,
        qualityProfile: TurboQuantQualityProfile = .balanced,
        backend: TurboQuantBackend = .metalPolarQJL,
        optimizationPolicy: TurboQuantOptimizationPolicy = .auto,
        requiresMetalSelfTest: Bool = true,
        requiresFusedAttentionSelfTest: Bool = false,
        status: TurboQuantProfileStatus = .experimental,
        source: String? = nil,
        validatedOn: String? = nil,
        validatedBy: String? = nil,
        confidence: Double? = nil,
        measured: TurboQuantProfileMeasurements = TurboQuantProfileMeasurements(),
        modelFingerprint: TurboQuantModelFingerprint? = nil,
        turboQuant: TurboQuantProfileTurboQuantManifest? = nil,
        measuredOutcomes: [TurboQuantMeasuredOutcome] = [],
        notes: [String] = []
    ) {
        self.schemaVersion = schemaVersion
        self.id = id
        self.exactModelIDs = exactModelIDs
        self.modelPatterns = modelPatterns
        self.includePatterns = includePatterns
        self.excludePatterns = excludePatterns
        self.architecture = architecture
        self.modelTypes = modelTypes.isEmpty ? [architecture].compactMap { $0 } : modelTypes
        self.textConfigModelTypes = textConfigModelTypes
        self.modalities = modalities
        self.minParametersB = minParametersB
        self.maxParametersB = maxParametersB
        self.requiresModelType = requiresModelType
        self.requiresTextConfigModelType = requiresTextConfigModelType
        self.requiresHeadDimensions = requiresHeadDimensions
        self.minRoutedExperts = minRoutedExperts
        self.maxRoutedExperts = maxRoutedExperts
        self.supportedExpertsPerToken = supportedExpertsPerToken
        self.supportedKeyHeadDimensions = supportedKeyHeadDimensions
        self.supportedValueHeadDimensions =
            supportedValueHeadDimensions ?? supportedKeyHeadDimensions
        self.recommendedScheme = recommendedScheme
        self.fallbackScheme = fallbackScheme
        self.keyBits = keyBits
        self.valueBits = valueBits
        self.groupSize = groupSize
        self.safeMaskModes = safeMaskModes
        self.supportedContextLengths = supportedContextLengths
        self.safeContextLength = safeContextLength
        self.qualityProfile = qualityProfile
        self.backend = backend
        self.optimizationPolicy = optimizationPolicy
        self.requiresMetalSelfTest = requiresMetalSelfTest
        self.requiresFusedAttentionSelfTest = requiresFusedAttentionSelfTest
        self.status = status
        self.source = source
        self.validatedOn = validatedOn
        self.validatedBy = validatedBy
        self.confidence = confidence
        self.measured = measured
        self.modelFingerprint = modelFingerprint
        self.turboQuant =
            turboQuant
            ?? TurboQuantProfileTurboQuantManifest(
                keyPreset: recommendedScheme,
                valueBits: valueBits,
                groupSize: groupSize
            )
        self.measuredOutcomes = measuredOutcomes
        self.notes = notes
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion
        case id
        case exactModelIDs
        case modelPatterns
        case includePatterns
        case excludePatterns
        case architecture
        case modelTypes
        case textConfigModelTypes
        case modalities
        case minParametersB
        case maxParametersB
        case requiresModelType
        case requiresTextConfigModelType
        case requiresHeadDimensions
        case minRoutedExperts
        case maxRoutedExperts
        case supportedExpertsPerToken
        case supportedKeyHeadDimensions
        case supportedValueHeadDimensions
        case recommendedScheme
        case fallbackScheme
        case keyBits
        case valueBits
        case groupSize
        case safeMaskModes
        case supportedContextLengths
        case safeContextLength
        case qualityProfile
        case backend
        case optimizationPolicy
        case requiresMetalSelfTest
        case requiresFusedAttentionSelfTest
        case status
        case source
        case validatedOn
        case validatedBy
        case confidence
        case measured
        case modelFingerprint
        case turboQuant
        case measuredOutcomes
        case notes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let id = try container.decode(String.self, forKey: .id)
        let architecture = try container.decodeIfPresent(String.self, forKey: .architecture)
        let supportedKeyHeadDimensions = try container.decode(
            [Int].self, forKey: .supportedKeyHeadDimensions)
        let supportedValueHeadDimensions =
            try container.decodeIfPresent([Int].self, forKey: .supportedValueHeadDimensions)

        self.init(
            schemaVersion: try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1,
            id: id,
            exactModelIDs: try container.decodeIfPresent(
                [String].self, forKey: .exactModelIDs) ?? [],
            modelPatterns: try container.decodeIfPresent(
                [String].self, forKey: .modelPatterns) ?? [],
            includePatterns: try container.decodeIfPresent(
                [String].self, forKey: .includePatterns) ?? [],
            excludePatterns: try container.decodeIfPresent(
                [String].self, forKey: .excludePatterns) ?? [],
            architecture: architecture,
            modelTypes: try container.decodeIfPresent([String].self, forKey: .modelTypes) ?? [],
            textConfigModelTypes: try container.decodeIfPresent(
                [String].self, forKey: .textConfigModelTypes) ?? [],
            modalities: try container.decodeIfPresent(
                [TurboQuantModelModality].self, forKey: .modalities) ?? [.text],
            minParametersB: try container.decodeIfPresent(Double.self, forKey: .minParametersB),
            maxParametersB: try container.decodeIfPresent(Double.self, forKey: .maxParametersB),
            requiresModelType: try container.decodeIfPresent(
                Bool.self, forKey: .requiresModelType) ?? false,
            requiresTextConfigModelType: try container.decodeIfPresent(
                Bool.self, forKey: .requiresTextConfigModelType) ?? false,
            requiresHeadDimensions: try container.decodeIfPresent(
                Bool.self, forKey: .requiresHeadDimensions) ?? false,
            minRoutedExperts: try container.decodeIfPresent(Int.self, forKey: .minRoutedExperts),
            maxRoutedExperts: try container.decodeIfPresent(Int.self, forKey: .maxRoutedExperts),
            supportedExpertsPerToken: try container.decodeIfPresent(
                [Int].self, forKey: .supportedExpertsPerToken) ?? [],
            supportedKeyHeadDimensions: supportedKeyHeadDimensions,
            supportedValueHeadDimensions: supportedValueHeadDimensions,
            recommendedScheme: try container.decodeIfPresent(
                TurboQuantScheme.self, forKey: .recommendedScheme) ?? .turbo4v2,
            fallbackScheme: try container.decodeIfPresent(
                TurboQuantScheme.self, forKey: .fallbackScheme),
            keyBits: try container.decodeIfPresent(Double.self, forKey: .keyBits) ?? 3.5,
            valueBits: try container.decodeIfPresent(Int.self, forKey: .valueBits) ?? 4,
            groupSize: try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64,
            safeMaskModes: try container.decodeIfPresent(
                [TurboQuantMaskMode].self, forKey: .safeMaskModes) ?? [.none, .causal],
            supportedContextLengths: try container.decodeIfPresent(
                [Int].self, forKey: .supportedContextLengths)
                ?? [4096, 8192, 16384, 32768, 65536],
            safeContextLength: try container.decodeIfPresent(Int.self, forKey: .safeContextLength),
            qualityProfile: try container.decodeIfPresent(
                TurboQuantQualityProfile.self, forKey: .qualityProfile) ?? .balanced,
            backend: try container.decodeIfPresent(TurboQuantBackend.self, forKey: .backend)
                ?? .metalPolarQJL,
            optimizationPolicy: try container.decodeIfPresent(
                TurboQuantOptimizationPolicy.self, forKey: .optimizationPolicy) ?? .auto,
            requiresMetalSelfTest: try container.decodeIfPresent(
                Bool.self, forKey: .requiresMetalSelfTest) ?? true,
            requiresFusedAttentionSelfTest: try container.decodeIfPresent(
                Bool.self, forKey: .requiresFusedAttentionSelfTest) ?? false,
            status: try container.decodeIfPresent(TurboQuantProfileStatus.self, forKey: .status)
                ?? .experimental,
            source: try container.decodeIfPresent(String.self, forKey: .source),
            validatedOn: try container.decodeIfPresent(String.self, forKey: .validatedOn),
            validatedBy: try container.decodeIfPresent(String.self, forKey: .validatedBy),
            confidence: try container.decodeIfPresent(Double.self, forKey: .confidence),
            measured: try container.decodeIfPresent(
                TurboQuantProfileMeasurements.self, forKey: .measured)
                ?? TurboQuantProfileMeasurements(),
            modelFingerprint: try container.decodeIfPresent(
                TurboQuantModelFingerprint.self, forKey: .modelFingerprint),
            turboQuant: try container.decodeIfPresent(
                TurboQuantProfileTurboQuantManifest.self, forKey: .turboQuant),
            measuredOutcomes: try container.decodeIfPresent(
                [TurboQuantMeasuredOutcome].self, forKey: .measuredOutcomes) ?? [],
            notes: try container.decodeIfPresent([String].self, forKey: .notes) ?? []
        )
    }

    public func supports(
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil
    ) -> Bool {
        if requiresHeadDimensions, keyHeadDimension == nil {
            return false
        }
        if requiresHeadDimensions, valueHeadDimension == nil {
            return false
        }
        if let keyHeadDimension,
            !supportedKeyHeadDimensions.contains(keyHeadDimension)
        {
            return false
        }
        if let valueHeadDimension,
            !supportedValueHeadDimensions.contains(valueHeadDimension)
        {
            return false
        }
        if !safeMaskModes.contains(maskMode) {
            return false
        }
        if let contextLength, let safeContextLength, contextLength > safeContextLength {
            return false
        }
        return true
    }

    public func applying(to parameters: GenerateParameters) -> GenerateParameters {
        var resolved = parameters
        resolved.kvCacheStrategy = recommendedScheme.kvCacheStrategy
        resolved.kvGroupSize = groupSize
        resolved.turboQuantPreset = recommendedScheme.preset
        resolved.turboQuantBackend = backend
        resolved.turboQuantOptimizationPolicy =
            optimizationPolicy == .auto
            ? recommendedScheme.defaultOptimizationPolicy
            : optimizationPolicy
        resolved.turboQuantValueBits = valueBits
        return resolved
    }

    public func productManifestValidation(
        actualFingerprint: TurboQuantModelFingerprint? = nil,
        requireMeasuredOutcomes: Bool = true
    ) -> TurboQuantProfileManifestValidation {
        var issues = [TurboQuantProfileManifestIssue]()
        if schemaVersion != 2 {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "schema_version",
                    kind: .unsupportedSchemaVersion,
                    expected: "2",
                    actual: String(schemaVersion),
                    reason: "TurboQuant product manifests require schema version 2"
                )
            )
        }

        if let modelFingerprint {
            for field in modelFingerprint.missingRequiredFields {
                issues.append(
                    TurboQuantProfileManifestIssue(
                        profileID: id,
                        field: field,
                        kind: .missingField,
                        reason: "required model fingerprint field is missing"
                    )
                )
            }
            if let actualFingerprint {
                issues.append(
                    contentsOf: modelFingerprint.mismatchIssues(
                        comparedTo: actualFingerprint,
                        profileID: id
                    )
                )
            }
        } else {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "model_fingerprint",
                    kind: .missingField,
                    reason: "required model fingerprint is missing"
                )
            )
        }

        if turboQuant.layoutVersion != TurboQuantAttentionLayout.currentVersion {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "turbo_quant.layout_version",
                    kind: .unsupportedSchemaVersion,
                    expected: String(TurboQuantAttentionLayout.currentVersion),
                    actual: String(turboQuant.layoutVersion),
                    reason: "TurboQuant compressed-cache layout version is not supported"
                )
            )
        }
        if turboQuant.keyPreset != recommendedScheme {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "turbo_quant.key_preset",
                    kind: .inconsistentTurboQuantField,
                    expected: recommendedScheme.rawValue,
                    actual: turboQuant.keyPreset.rawValue,
                    reason: "manifest key preset must match the selected profile scheme"
                )
            )
        }
        if turboQuant.valueBits != valueBits {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "turbo_quant.value_bits",
                    kind: .inconsistentTurboQuantField,
                    expected: String(valueBits),
                    actual: String(turboQuant.valueBits),
                    reason: "manifest value bits must match the selected profile value bits"
                )
            )
        }
        if turboQuant.groupSize != groupSize {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "turbo_quant.group_size",
                    kind: .inconsistentTurboQuantField,
                    expected: String(groupSize),
                    actual: String(turboQuant.groupSize),
                    reason: "manifest group size must match the selected profile group size"
                )
            )
        }
        if turboQuant.preferredPaths.isEmpty {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "turbo_quant.preferred_paths",
                    kind: .missingField,
                    reason: "at least one preferred runtime path is required"
                )
            )
        }
        if requireMeasuredOutcomes && measuredOutcomes.isEmpty {
            issues.append(
                TurboQuantProfileManifestIssue(
                    profileID: id,
                    field: "measured_outcomes",
                    kind: .missingMeasuredOutcome,
                    reason: "measured product profiles require device and OS outcomes"
                )
            )
        }
        for (index, outcome) in measuredOutcomes.enumerated() {
            let prefix = "measured_outcomes[\(index)]"
            if outcome.deviceClass.isEmpty {
                issues.append(
                    TurboQuantProfileManifestIssue(
                        profileID: id,
                        field: "\(prefix).device_class",
                        kind: .missingField,
                        reason: "measured outcome device class is required"
                    )
                )
            }
            if outcome.osVersion.isEmpty {
                issues.append(
                    TurboQuantProfileManifestIssue(
                        profileID: id,
                        field: "\(prefix).os_version",
                        kind: .missingField,
                        reason: "measured outcome OS version is required"
                    )
                )
            }
            if outcome.maxContextByMode.isEmpty {
                issues.append(
                    TurboQuantProfileManifestIssue(
                        profileID: id,
                        field: "\(prefix).max_context_by_mode",
                        kind: .missingField,
                        reason: "measured outcome max context by mode is required"
                    )
                )
            }
        }

        return TurboQuantProfileManifestValidation(profileID: id, issues: issues)
    }
}

public struct TurboQuantProfileRegistry: Sendable {
    public var profiles: [TurboQuantProfile]

    public init(profiles: [TurboQuantProfile]) {
        self.profiles = profiles
    }

    public static let bundled = TurboQuantProfileRegistry(profiles: bundledProfiles)

    public func profile(
        for modelID: String,
        modelType: String? = nil,
        textConfigModelType: String? = nil,
        modality: TurboQuantModelModality? = nil,
        parameterCountB: Double? = nil,
        routedExperts: Int? = nil,
        expertsPerToken: Int? = nil,
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil,
        fingerprint: TurboQuantModelFingerprint? = nil,
        requireFingerprint: Bool = false
    ) -> TurboQuantProfile? {
        let descriptor = TurboQuantModelDescriptor(
            modelID: modelID,
            modelType: modelType,
            textConfigModelType: textConfigModelType,
            modality: modality,
            parameterCountB: parameterCountB,
            routedExperts: routedExperts,
            expertsPerToken: expertsPerToken,
            fingerprint: fingerprint)
        return selection(
            for: descriptor,
            keyHeadDimension: keyHeadDimension,
            valueHeadDimension: valueHeadDimension,
            maskMode: maskMode,
            contextLength: contextLength,
            requireFingerprint: requireFingerprint
        ).profile
    }

    public func selection(
        for descriptor: TurboQuantModelDescriptor,
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil,
        requireFingerprint: Bool = false
    ) -> TurboQuantProfileSelection {
        var diagnostics: [TurboQuantProfileDiagnostic] = []
        for profile in profiles {
            let reasons = profile.rejectionReasons(
                descriptor: descriptor,
                keyHeadDimension: keyHeadDimension,
                valueHeadDimension: valueHeadDimension,
                maskMode: maskMode,
                contextLength: contextLength,
                requireFingerprint: requireFingerprint)
            diagnostics.append(
                TurboQuantProfileDiagnostic(
                    profileID: profile.id,
                    accepted: reasons.isEmpty,
                    reasons: reasons))
            if reasons.isEmpty {
                return TurboQuantProfileSelection(
                    descriptor: descriptor,
                    profile: profile,
                    diagnostics: diagnostics)
            }
        }
        return TurboQuantProfileSelection(
            descriptor: descriptor,
            profile: nil,
            diagnostics: diagnostics)
    }

    public static func loadJSONProfiles(from directory: URL) throws -> [TurboQuantProfile] {
        let fileManager = FileManager.default
        let urls = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return
            try urls
            .filter { $0.pathExtension.lowercased() == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { url in
                let data = try Data(contentsOf: url)
                return try decoder.decode(TurboQuantProfile.self, from: data)
            }
    }
}

public struct TurboQuantProfileDiagnostic: Equatable, Sendable {
    public var profileID: String
    public var accepted: Bool
    public var reasons: [String]
}

public struct TurboQuantProfileSelection: Equatable, Sendable {
    public var descriptor: TurboQuantModelDescriptor
    public var profile: TurboQuantProfile?
    public var diagnostics: [TurboQuantProfileDiagnostic]

    public var rejectionReasons: [String] {
        diagnostics.flatMap { diagnostic in
            diagnostic.reasons.map { "\(diagnostic.profileID): \($0)" }
        }
    }
}

extension GenerateParameters {
    public init(
        turboQuantProfile profile: TurboQuantProfile,
        base parameters: GenerateParameters = GenerateParameters()
    ) {
        self = profile.applying(to: parameters)
    }

    public init(
        turboQuantModelID modelID: String,
        registry: TurboQuantProfileRegistry = .bundled,
        modelType: String? = nil,
        textConfigModelType: String? = nil,
        modality: TurboQuantModelModality? = nil,
        parameterCountB: Double? = nil,
        routedExperts: Int? = nil,
        expertsPerToken: Int? = nil,
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil,
        fingerprint: TurboQuantModelFingerprint? = nil,
        requireFingerprint: Bool = false,
        base parameters: GenerateParameters = GenerateParameters()
    ) {
        if let profile = registry.profile(
            for: modelID,
            modelType: modelType,
            textConfigModelType: textConfigModelType,
            modality: modality,
            parameterCountB: parameterCountB,
            routedExperts: routedExperts,
            expertsPerToken: expertsPerToken,
            keyHeadDimension: keyHeadDimension,
            valueHeadDimension: valueHeadDimension,
            maskMode: maskMode,
            contextLength: contextLength,
            fingerprint: fingerprint,
            requireFingerprint: requireFingerprint
        ) {
            self = profile.applying(to: parameters)
        } else {
            self = parameters
        }
    }

    public mutating func applyTurboQuantProfile(_ profile: TurboQuantProfile) {
        self = profile.applying(to: self)
    }
}

extension TurboQuantProfile {
    fileprivate func rejectionReasons(
        descriptor: TurboQuantModelDescriptor,
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil,
        requireFingerprint: Bool = false
    ) -> [String] {
        var reasons = [String]()
        if status == .deprecated {
            reasons.append("profile is deprecated")
        }

        if requireFingerprint, modelFingerprint == nil {
            reasons.append("model fingerprint metadata is required")
        }
        if requireFingerprint, let modelFingerprint {
            for field in modelFingerprint.missingRequiredFields {
                reasons.append("\(field) is required")
            }
        }
        if let actualFingerprint = descriptor.fingerprint, let modelFingerprint {
            let issues = modelFingerprint.mismatchIssues(
                comparedTo: actualFingerprint,
                profileID: id
            )
            for issue in issues {
                let expected = issue.expected ?? "nil"
                let actual = issue.actual ?? "nil"
                reasons.append("\(issue.field) expected \(expected), got \(actual)")
            }
        }

        let normalizedID = Self.normalized(descriptor.modelID)
        let normalizedExactIDs = Set(exactModelIDs.map(Self.normalized))
        let exactIDMatched = normalizedExactIDs.contains(normalizedID)
        let includeSet = includePatterns.isEmpty ? modelPatterns : includePatterns
        let includeMatched =
            exactIDMatched
            || includeSet.contains { pattern in
                Self.matches(pattern: Self.normalized(pattern), modelID: normalizedID)
            }
        if !includeMatched {
            reasons.append("model id does not match exact ids or include patterns")
        }

        if excludePatterns.contains(where: { pattern in
            Self.matches(pattern: Self.normalized(pattern), modelID: normalizedID)
        }) {
            reasons.append("model id matches an exclude pattern")
        }

        if requiresModelType, descriptor.modelType == nil {
            reasons.append("model type metadata is required")
        }
        if let modelType = descriptor.modelType {
            let normalizedModelType = Self.normalizedModelType(modelType)
            let allowedTypes = Set(
                (modelTypes.isEmpty ? [architecture].compactMap { $0 } : modelTypes)
                    .map(Self.normalizedModelType))
            if !allowedTypes.isEmpty, !allowedTypes.contains(normalizedModelType) {
                reasons.append("model type '\(modelType)' is not supported")
            }
        }
        if requiresTextConfigModelType, descriptor.textConfigModelType == nil {
            reasons.append("text_config model type metadata is required")
        }
        if let textConfigModelType = descriptor.textConfigModelType {
            let normalizedTextConfigModelType = Self.normalizedModelType(textConfigModelType)
            let allowedTextTypes = Set(textConfigModelTypes.map(Self.normalizedModelType))
            if !allowedTextTypes.isEmpty, !allowedTextTypes.contains(normalizedTextConfigModelType)
            {
                reasons.append("text_config model type '\(textConfigModelType)' is not supported")
            }
        }

        if let modality = descriptor.modality, !modalities.contains(modality) {
            reasons.append("modality '\(modality.rawValue)' is not supported")
        }

        let implicitProfileSizeB = TurboQuantModelDescriptor.inferParameterCountB(from: id)
        let effectiveMinParametersB =
            minParametersB ?? implicitProfileSizeB.map { max($0 - 0.05, 0) }
        let effectiveMaxParametersB = maxParametersB ?? implicitProfileSizeB.map { $0 + 0.05 }
        if (effectiveMinParametersB != nil || effectiveMaxParametersB != nil)
            && descriptor.parameterCountB == nil
        {
            reasons.append("model size could not be inferred from id")
        }
        if let effectiveMinParametersB, let parameterCountB = descriptor.parameterCountB,
            parameterCountB < effectiveMinParametersB
        {
            reasons.append(
                "model size \(parameterCountB)B is below minimum \(effectiveMinParametersB)B")
        }
        if let effectiveMaxParametersB, let parameterCountB = descriptor.parameterCountB,
            parameterCountB > effectiveMaxParametersB
        {
            reasons.append(
                "model size \(parameterCountB)B exceeds maximum \(effectiveMaxParametersB)B")
        }
        if minRoutedExperts != nil || maxRoutedExperts != nil, descriptor.routedExperts == nil {
            reasons.append("routed expert count metadata is required")
        }
        if let minRoutedExperts, let routedExperts = descriptor.routedExperts,
            routedExperts < minRoutedExperts
        {
            reasons.append(
                "routed expert count \(routedExperts) is below minimum \(minRoutedExperts)")
        }
        if let maxRoutedExperts, let routedExperts = descriptor.routedExperts,
            routedExperts > maxRoutedExperts
        {
            reasons.append(
                "routed expert count \(routedExperts) exceeds maximum \(maxRoutedExperts)")
        }
        if !supportedExpertsPerToken.isEmpty, descriptor.expertsPerToken == nil {
            reasons.append("experts-per-token metadata is required")
        }
        if let expertsPerToken = descriptor.expertsPerToken,
            !supportedExpertsPerToken.isEmpty,
            !supportedExpertsPerToken.contains(expertsPerToken)
        {
            reasons.append("experts per token \(expertsPerToken) is not supported")
        }

        if requiresHeadDimensions, keyHeadDimension == nil {
            reasons.append("key head dimension metadata is required")
        }
        if requiresHeadDimensions, valueHeadDimension == nil {
            reasons.append("value head dimension metadata is required")
        }
        if let keyHeadDimension,
            !supportedKeyHeadDimensions.contains(keyHeadDimension)
        {
            reasons.append("key head dimension \(keyHeadDimension) is not supported")
        }
        if let valueHeadDimension,
            !supportedValueHeadDimensions.contains(valueHeadDimension)
        {
            reasons.append("value head dimension \(valueHeadDimension) is not supported")
        }
        if !safeMaskModes.contains(maskMode) {
            reasons.append("mask mode '\(maskMode.rawValue)' is not supported")
        }
        if let contextLength, let safeContextLength, contextLength > safeContextLength {
            reasons.append(
                "context length \(contextLength) exceeds safe length \(safeContextLength)")
        }

        return reasons
    }

    fileprivate static func normalized(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .replacingOccurrences(of: " ", with: "-")
    }

    fileprivate static func normalizedModelType(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: ".", with: "_")
            .replacingOccurrences(of: "-", with: "_")
    }

    fileprivate static func matches(pattern: String, modelID: String) -> Bool {
        if pattern == modelID || modelID.contains(pattern) {
            return true
        }
        guard pattern.contains("*") else {
            return false
        }

        let regexPattern =
            "^"
            + NSRegularExpression.escapedPattern(for: pattern)
            .replacingOccurrences(of: "\\*", with: ".*")
            + "$"
        guard let regex = try? NSRegularExpression(pattern: regexPattern) else {
            return false
        }
        let range = NSRange(modelID.startIndex ..< modelID.endIndex, in: modelID)
        return regex.firstMatch(in: modelID, range: range) != nil
    }
}

private let turboQuantProfileNotes = [
    "Uses ordinary MLX model weights; TurboQuant applies only to runtime KV-cache state.",
    "Measured quality and throughput fields are intentionally empty until reproduced on target hardware.",
    "Runtime shape checks and Metal self-tests remain authoritative.",
]

private let commonSafeMasks: [TurboQuantMaskMode] = [.none, .causal]

private let bundledProfileDefaultContextLengths = [4096, 8192, 16384, 32768, 65536]
private let bundledProfile8KContextLengths = [4096, 8192]
private let bundledProfile32KContextLengths = [4096, 8192, 16384, 32768]
private let bundledProfile40KContextLengths = [4096, 8192, 16384, 32768, 40960]
private let bundledProfile128KContextLengths =
    bundledProfileDefaultContextLengths + [131072]
private let bundledProfile256KContextLengths =
    bundledProfile128KContextLengths + [262144]
private let bundledProfile384KContextLengths =
    bundledProfile256KContextLengths + [393216]
private let bundledProfile1MContextLengths =
    bundledProfile256KContextLengths + [524288, 1_048_576]
private let bundledProfileMistralNemoContextLengths =
    bundledProfile128KContextLengths

private func bundledProfile(
    id: String,
    patterns: [String],
    includePatterns: [String]? = nil,
    excludePatterns: [String] = [],
    architecture: String? = nil,
    modelTypes: [String] = [],
    textConfigModelTypes: [String] = [],
    modalities: [TurboQuantModelModality] = [.text],
    minParametersB: Double? = nil,
    maxParametersB: Double? = nil,
    requiresModelType: Bool = false,
    requiresTextConfigModelType: Bool = false,
    requiresHeadDimensions: Bool = false,
    minRoutedExperts: Int? = nil,
    maxRoutedExperts: Int? = nil,
    supportedExpertsPerToken: [Int] = [],
    supportedKeyHeadDimensions: [Int],
    supportedValueHeadDimensions: [Int]? = nil,
    supportedContextLengths: [Int] = bundledProfileDefaultContextLengths,
    safeContextLength: Int? = 65536,
    source: String? = "profile-audit-2026-05-23",
    confidence: Double? = nil,
    extraNotes: [String] = []
) -> TurboQuantProfile {
    TurboQuantProfile(
        schemaVersion: 2,
        id: id,
        modelPatterns: patterns,
        includePatterns: includePatterns ?? patterns,
        excludePatterns: excludePatterns,
        architecture: architecture,
        modelTypes: modelTypes,
        textConfigModelTypes: textConfigModelTypes,
        modalities: modalities,
        minParametersB: minParametersB,
        maxParametersB: maxParametersB,
        requiresModelType: requiresModelType,
        requiresTextConfigModelType: requiresTextConfigModelType,
        requiresHeadDimensions: requiresHeadDimensions,
        minRoutedExperts: minRoutedExperts,
        maxRoutedExperts: maxRoutedExperts,
        supportedExpertsPerToken: supportedExpertsPerToken,
        supportedKeyHeadDimensions: supportedKeyHeadDimensions,
        supportedValueHeadDimensions: supportedValueHeadDimensions,
        safeMaskModes: commonSafeMasks,
        supportedContextLengths: supportedContextLengths,
        safeContextLength: safeContextLength,
        status: .experimental,
        source: source,
        confidence: confidence,
        notes: turboQuantProfileNotes + extraNotes
    )
}

private let bundledProfileContextLengthOverrides: [String: [Int]] = [
    "gemma-3-270m": bundledProfile32KContextLengths,
    "gemma-3-1b": bundledProfile32KContextLengths,
    "gemma-3-4b": bundledProfile128KContextLengths,
    "gemma-3-12b": bundledProfile128KContextLengths,
    "gemma-3-27b": bundledProfile128KContextLengths,
    "gemma-3n-e2b": bundledProfile32KContextLengths,
    "gemma-3n-e4b": bundledProfile32KContextLengths,
    "gemma-4-e2b": bundledProfile128KContextLengths,
    "gemma-4-e4b": bundledProfile128KContextLengths,
    "gemma-4-26b-a4b": bundledProfile256KContextLengths,
    "gemma-4-31b": bundledProfile256KContextLengths,
    "llama-3-3b": bundledProfile8KContextLengths,
    "llama-3-8b": bundledProfile8KContextLengths,
    "llama-3-16b": bundledProfile8KContextLengths,
    "llama-3-70b": bundledProfile8KContextLengths,
    "llama-3-120b": bundledProfile8KContextLengths,
    "llama-3.1-4b": bundledProfile128KContextLengths,
    "llama-3.1-8b": bundledProfile128KContextLengths,
    "llama-3.1-16b": bundledProfile128KContextLengths,
    "llama-3.1-70b": bundledProfile128KContextLengths,
    "llama-3.1-120b": bundledProfile128KContextLengths,
    "llama-3.1-405b": bundledProfile128KContextLengths,
    "llama-3.2-1b": bundledProfile128KContextLengths,
    "llama-3.2-3b": bundledProfile128KContextLengths,
    "llama-3.3-3b": bundledProfile128KContextLengths,
    "llama-3.3-70b": bundledProfile128KContextLengths,
    "mistral-7b": bundledProfile32KContextLengths,
    "mistral-nemo-12b": bundledProfileMistralNemoContextLengths,
    "ministral-8b-2410": bundledProfile32KContextLengths,
    "codestral-22b": bundledProfile32KContextLengths,
    "mistral-small-22b": bundledProfile128KContextLengths,
    "mistral-small-24b": bundledProfile128KContextLengths,
    "mistral-large-2407": bundledProfile128KContextLengths,
    "devstral-small-24b": bundledProfile128KContextLengths,
    "magistral-small-24b": bundledProfile32KContextLengths,
    "ministral3-3b": bundledProfile256KContextLengths,
    "ministral3-8b": bundledProfile256KContextLengths,
    "ministral3-14b": bundledProfile256KContextLengths,
    "mistral-small-3.1-24b": bundledProfile128KContextLengths,
    "mistral-small-3.2-24b": bundledProfile128KContextLengths,
    "devstral-small-2-24b": bundledProfile384KContextLengths,
    "mistral-medium-3.5-128b": bundledProfile256KContextLengths,
    "devstral-2-123b": bundledProfile256KContextLengths,
    "mistral-small-4-119b-a6b": bundledProfile1MContextLengths,
    "pixtral-12b": bundledProfile128KContextLengths,
    "qwen3-0.6b": bundledProfile40KContextLengths,
    "qwen3-1.7b": bundledProfile40KContextLengths,
    "qwen3-4b": bundledProfile40KContextLengths,
    "qwen3-8b": bundledProfile40KContextLengths,
    "qwen3.5-0.8b": bundledProfile256KContextLengths,
    "qwen3.5-2b": bundledProfile256KContextLengths,
    "qwen3.5-4b": bundledProfile256KContextLengths,
    "qwen3.5-9b": bundledProfile256KContextLengths,
    "qwen3.5-27b": bundledProfile256KContextLengths,
    "qwen3.6-27b": bundledProfile256KContextLengths,
    "qwen3.5-40b": bundledProfile256KContextLengths,
    "qwen3.6-40b": bundledProfile256KContextLengths,
    "qwen3.5-35b-a3b": bundledProfile256KContextLengths,
    "qwen3.6-35b-a3b": bundledProfile256KContextLengths,
    "qwen3.5-97b-a10b": bundledProfile256KContextLengths,
    "qwen3.5-122b-a10b": bundledProfile256KContextLengths,
    "qwen3.5-397b-a17b": bundledProfile256KContextLengths,
]

private let bundledConservativeOptimizationProfileIDs: Set<String> = [
    "glm4-moe-lite"
]

private let bundledThroughputOptimizationProfileIDs: Set<String> = [
    "phi-2",
    "phi-3-mini",
    "phi-3.5-mini",
    "phi-4-mini",
]

private func defaultBundledOptimizationPolicy(
    for profile: TurboQuantProfile,
    safeContextLength: Int?
) -> TurboQuantOptimizationPolicy {
    if bundledConservativeOptimizationProfileIDs.contains(profile.id) {
        return .conservative
    }
    if bundledThroughputOptimizationProfileIDs.contains(profile.id) {
        return .preferThroughput
    }
    if profile.modalities.contains(where: { $0 != .text }) {
        return .preferMemory
    }
    if let architecture = profile.architecture?.lowercased(),
        architecture.contains("moe")
    {
        return .preferMemory
    }
    if profile.minRoutedExperts != nil || profile.maxRoutedExperts != nil {
        return .preferMemory
    }
    if let safeContextLength, safeContextLength > 65536 {
        return .preferMemory
    }
    if let maxParametersB = profile.maxParametersB, maxParametersB <= 8.5 {
        return .preferThroughput
    }
    if let minParametersB = profile.minParametersB, minParametersB >= 8.5 {
        return .preferMemory
    }
    if let maxParametersB = profile.maxParametersB, maxParametersB >= 9 {
        return .preferMemory
    }
    return .auto
}

private func applyingBundledProfileOptimizations(_ profile: TurboQuantProfile)
    -> TurboQuantProfile
{
    var profile = profile
    if let supportedContextLengths = bundledProfileContextLengthOverrides[profile.id] {
        profile.supportedContextLengths = supportedContextLengths
        profile.safeContextLength = supportedContextLengths.last
    }
    profile.optimizationPolicy = defaultBundledOptimizationPolicy(
        for: profile,
        safeContextLength: profile.safeContextLength
    )
    return profile
}

private let commonNonTextExcludePatterns = [
    "*embedding*",
    "*embed*",
    "*reranker*",
    "*reward*",
    "*classifier*",
    "*vl*",
    "*vision*",
    "*video*",
    "*audio*",
    "*omni*",
    "*llava*",
    "*bunny*",
    "*paligemma*",
    "*pixtral*",
    "*molmo*",
    "*moondream*",
    "*idefics*",
]

private let qwen3ExcludePatterns =
    commonNonTextExcludePatterns + [
        "*qwen3.5*",
        "*qwen-3.5*",
        "*qwen3-5*",
        "*qwen-3-5*",
        "*qwen3.6*",
        "*qwen-3.6*",
        "*qwen3-6*",
        "*qwen-3-6*",
        "*qwen3.7*",
        "*qwen-3.7*",
        "*qwen3-7*",
        "*qwen-3-7*",
        "*moe*",
        "*a3b*",
        "*a22b*",
        "*coder*",
        "*next*",
        "*deepseek*",
    ]

private func modelIDPatterns(_ bases: [String]) -> [String] {
    bases.flatMap { ["*\($0)", "*\($0)-*"] }
}

private let qwen25SmallPatterns = modelIDPatterns([
    "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
    "qwen2.5-coder-0.5b", "qwen2.5-coder-1.5b", "qwen2.5-coder-3b",
    "qwen2.5-coder-7b", "qwen2.5.1-coder-7b",
    "qwen2-5-0.5b", "qwen2-5-1.5b", "qwen2-5-3b", "qwen2-5-7b",
    "qwen2-5-coder-0.5b", "qwen2-5-coder-1.5b", "qwen2-5-coder-3b",
    "qwen2-5-coder-7b", "qwen2-5-1-coder-7b",
    "qwen-2.5-0.5b", "qwen-2.5-1.5b", "qwen-2.5-3b", "qwen-2.5-7b",
    "qwen-2.5-coder-0.5b", "qwen-2.5-coder-1.5b", "qwen-2.5-coder-3b",
    "qwen-2.5-coder-7b", "qwen-2.5.1-coder-7b",
    "qwen-2-5-0.5b", "qwen-2-5-1.5b", "qwen-2-5-3b", "qwen-2-5-7b",
    "qwen-2-5-coder-0.5b", "qwen-2-5-coder-1.5b", "qwen-2-5-coder-3b",
    "qwen-2-5-coder-7b", "qwen-2-5-1-coder-7b",
])

private func qwen35FamilyPatterns(_ version: String, size: String) -> [String] {
    let normalizedVersion = version.replacingOccurrences(of: ".", with: "-")
    let compactVersion = version.filter(\.isNumber)
    let names = [
        "qwen\(version)-\(size)",
        "qwen\(normalizedVersion)-\(size)",
        "qwen\(compactVersion)-\(size)",
        "qwen-\(version)-\(size)",
        "qwen-\(normalizedVersion)-\(size)",
    ]
    return modelIDPatterns(names)
}

private func qwen35MoEPatterns(_ version: String, totalSize: String, activeSize: String) -> [String]
{
    let normalizedVersion = version.replacingOccurrences(of: ".", with: "-")
    let compactVersion = version.filter(\.isNumber)
    let names = [
        "qwen\(version)-\(totalSize)-\(activeSize)",
        "qwen\(normalizedVersion)-\(totalSize)-\(activeSize)",
        "qwen\(compactVersion)-\(totalSize)-\(activeSize)",
        "qwen-\(version)-\(totalSize)-\(activeSize)",
        "qwen-\(normalizedVersion)-\(totalSize)-\(activeSize)",
    ]
    return modelIDPatterns(names)
}

private let qwen35DenseExcludePatterns =
    commonNonTextExcludePatterns + [
        "*qwen2*", "*qwen-2*", "*qwen2.5*", "*qwen-2.5*", "*qwen2-5*", "*qwen-2-5*",
        "*qwen3-next*", "*qwen3next*", "*qwen-3-next*",
        "*qwen3.7*", "*qwen-3.7*", "*qwen3-7*", "*qwen-3-7*",
        "*moe*", "*a3b*", "*a10b*", "*a17b*", "*a22b*",
    ]

private let qwen35MoEExcludePatterns =
    commonNonTextExcludePatterns + [
        "*qwen2*", "*qwen-2*", "*qwen2.5*", "*qwen-2.5*", "*qwen2-5*", "*qwen-2-5*",
        "*qwen3-next*", "*qwen3next*", "*qwen-3-next*",
        "*qwen3.7*", "*qwen-3.7*", "*qwen3-7*", "*qwen-3-7*",
    ]

private let qwen35ModelTypes = ["qwen3_5", "qwen3_5_text"]
private let qwen35MoEModelTypes = ["qwen3_5_moe", "qwen3_5_moe_text"]
private let qwen35Modalities: [TurboQuantModelModality] = [.text, .visionText]
private let qwen35ProfileNotes = [
    "Qwen3.5 and Qwen3.6 profiles are config-backed with verified 256-dimensional key and value heads."
]

private let gemmaTextExcludePatterns =
    commonNonTextExcludePatterns + [
        "*embeddinggemma*",
        "*paligemma*",
    ]

private let gemmaVisionTextExcludePatterns = [
    "*embedding*",
    "*embed*",
    "*embeddinggemma*",
    "*reranker*",
    "*reward*",
    "*classifier*",
    "*video*",
    "*audio*",
    "*omni*",
    "*llava*",
    "*bunny*",
    "*paligemma*",
    "*pixtral*",
    "*molmo*",
    "*moondream*",
    "*idefics*",
]

private let gemma3ExcludePatterns =
    gemmaVisionTextExcludePatterns + [
        "*gemma-3n*",
        "*gemma3n*",
    ]

private let gemma2ModelTypes = ["gemma2"]
private let gemma3ModelTypes = ["gemma3", "gemma3_text"]
private let gemma3nModelTypes = ["gemma3n", "gemma3n_text"]
private let gemma4ModelTypes = ["gemma4", "gemma4_text", "gemma4_assistant"]
private let gemmaProfileNotes = [
    "Gemma 2/3/3n/4 profiles are config-backed experimental profiles; measured quality and throughput validation is still pending."
]

private let llamaExcludePatterns =
    commonNonTextExcludePatterns + [
        "*mllama*",
        "*llama-3.2-vision*",
        "*llama3.2-vision*",
        "*llama-4-*",
        "*llama4-*",
        "*llama-4-scout*",
        "*llama4-scout*",
        "*llama-4-maverick*",
        "*llama4-maverick*",
        "*mixtral*",
        "*mamba-codestral*",
        "*mamba*codestral*",
        "*outetts*",
    ]

private let mistralTextExcludePatterns =
    commonNonTextExcludePatterns + [
        "*mixtral*",
        "*pixtral*",
        "*mamba-codestral*",
        "*mamba*codestral*",
        "*mllama*",
        "*llama*",
    ]

private let pixtralExcludePatterns = [
    "*embedding*",
    "*embed*",
    "*reranker*",
    "*reward*",
    "*classifier*",
    "*video*",
    "*audio*",
    "*omni*",
    "*llava*",
    "*bunny*",
    "*paligemma*",
    "*molmo*",
    "*moondream*",
    "*idefics*",
    "*mixtral*",
]

private let llamaProfileNotes = [
    "Llama profiles are config-backed experimental profiles gated on model_type=llama and explicit KV head dimensions."
]
private let mistralProfileNotes = [
    "Mistral profiles are config-backed experimental profiles gated on model_type, nested text_config.model_type where required, and explicit KV head dimensions."
]

private func llamaFamilyPatterns(_ bases: [String]) -> [String] {
    modelIDPatterns(bases)
}

private func mistralFamilyPatterns(_ bases: [String]) -> [String] {
    modelIDPatterns(bases)
}

private let bundledProfiles: [TurboQuantProfile] = [
    bundledProfile(
        id: "exaone-small",
        patterns: [
            "*exaone-3.0-2.4b", "*exaone-3.0-2.4b-*", "*exaone-3.5-2.4b",
            "*exaone-3.5-2.4b-*", "*exaone-4.0-1.2b", "*exaone-4.0-1.2b-*",
            "*exaone-4.0-4b", "*exaone-4.0-4b-*", "*exaone-3-0-2.4b",
            "*exaone-3-0-2.4b-*", "*exaone-3-5-2.4b", "*exaone-3-5-2.4b-*",
            "*exaone-4-0-1.2b", "*exaone-4-0-1.2b-*", "*exaone-4-0-4b",
            "*exaone-4-0-4b-*",
        ],
        excludePatterns: commonNonTextExcludePatterns + ["*moe*"],
        architecture: "exaone",
        modelTypes: ["exaone", "exaone4"],
        minParametersB: 1,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 80, 128],
        confidence: 0.65,
        extraNotes: ["Covers EXAONE dense <=8B variants with runtime head-dimension checks."]
    ),
    bundledProfile(
        id: "gemma-2-2b",
        patterns: modelIDPatterns([
            "gemma-2-2b",
            "gemma2-2b",
            "gemma-2-baku-2b",
            "dolphin-gemma2-2b",
        ]),
        excludePatterns: gemmaTextExcludePatterns,
        architecture: "gemma2",
        modelTypes: gemma2ModelTypes,
        minParametersB: 1.5,
        maxParametersB: 2.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        supportedContextLengths: [4096, 8192],
        safeContextLength: 8192,
        confidence: 0.8,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-2-9b",
        patterns: modelIDPatterns([
            "gemma-2-9b",
            "gemma2-9b",
            "gemma-sea-lion-v3-9b",
            "tiger-gemma-9b",
        ]),
        excludePatterns: gemmaTextExcludePatterns,
        architecture: "gemma2",
        modelTypes: gemma2ModelTypes,
        minParametersB: 8.5,
        maxParametersB: 9.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        supportedContextLengths: [4096, 8192],
        safeContextLength: 8192,
        confidence: 0.8,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-2-27b",
        patterns: modelIDPatterns([
            "gemma-2-27b",
            "gemma2-27b",
            "big-tiger-gemma-27b",
        ]),
        excludePatterns: gemmaTextExcludePatterns,
        architecture: "gemma2",
        modelTypes: gemma2ModelTypes,
        minParametersB: 25,
        maxParametersB: 29,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        supportedContextLengths: [4096, 8192],
        safeContextLength: 8192,
        confidence: 0.8,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-2b",
        patterns: modelIDPatterns([
            "gemma-2b",
            "gemma-1.1-2b",
            "gemma1.1-2b",
            "gemma-1-1-2b",
        ]),
        excludePatterns: gemmaTextExcludePatterns + [
            "*gemma-2-2b*", "*gemma2-2b*", "*embeddinggemma*",
        ],
        architecture: "gemma",
        modelTypes: ["gemma"],
        minParametersB: 1.5,
        maxParametersB: 2.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        supportedContextLengths: [4096, 8192],
        safeContextLength: 8192,
        confidence: 0.75
    ),
    bundledProfile(
        id: "gemma-3-270m",
        patterns: modelIDPatterns([
            "gemma-3-270m",
            "gemma3-270m",
            "huihui-gemma-3-270m",
        ]),
        excludePatterns: gemma3ExcludePatterns,
        architecture: "gemma3",
        modelTypes: gemma3ModelTypes,
        modalities: [.text],
        minParametersB: 0.2,
        maxParametersB: 0.35,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.75,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-3-1b",
        patterns: modelIDPatterns([
            "gemma-3-1b",
            "gemma3-1b",
            "swahili-gemma-1b",
        ]),
        excludePatterns: gemma3ExcludePatterns,
        architecture: "gemma3",
        modelTypes: gemma3ModelTypes,
        modalities: [.text],
        minParametersB: 0.8,
        maxParametersB: 1.3,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.75,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-3-4b",
        patterns: modelIDPatterns([
            "gemma-3-4b",
            "gemma3-4b",
            "gemma-3-text-4b",
            "gemma3-text-4b",
            "text-to-cypher-gemma-3-4b",
            "gemma-sea-lion-v4-4b",
        ]),
        excludePatterns: gemma3ExcludePatterns,
        architecture: "gemma3",
        modelTypes: gemma3ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256, 320],
        confidence: 0.85,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-3-12b",
        patterns: modelIDPatterns([
            "gemma-3-12b",
            "gemma3-12b",
            "gemma-3-text-12b",
            "gemma3-text-12b",
            "gemma-3-glitter-12b",
        ]),
        excludePatterns: gemma3ExcludePatterns,
        architecture: "gemma3",
        modelTypes: gemma3ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 11,
        maxParametersB: 13.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [240, 256],
        confidence: 0.8,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-3-27b",
        patterns: modelIDPatterns([
            "gemma-3-27b",
            "gemma3-27b",
            "gemma-3-text-27b",
            "gemma3-text-27b",
            "gemma-sea-lion-v4-27b",
        ]),
        excludePatterns: gemma3ExcludePatterns,
        architecture: "gemma3",
        modelTypes: gemma3ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 25,
        maxParametersB: 29,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.8,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-3n-e2b",
        patterns: modelIDPatterns([
            "gemma-3n-e2b",
            "gemma3n-e2b",
        ]),
        excludePatterns: gemmaVisionTextExcludePatterns,
        architecture: "gemma3n",
        modelTypes: gemma3nModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 1.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.85,
        extraNotes: gemmaProfileNotes + [
            "Shared-KV layers use AttentionKVState so compressed state can be reused without raw KV materialization."
        ]
    ),
    bundledProfile(
        id: "gemma-3n-e4b",
        patterns: modelIDPatterns([
            "gemma-3n-e4b",
            "gemma3n-e4b",
            "huihui-gemma-3n-e4b",
        ]),
        excludePatterns: gemmaVisionTextExcludePatterns,
        architecture: "gemma3n",
        modelTypes: gemma3nModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.85,
        extraNotes: gemmaProfileNotes + [
            "Shared-KV layers use AttentionKVState so compressed state can be reused without raw KV materialization."
        ]
    ),
    bundledProfile(
        id: "gemma-4-e2b",
        patterns: modelIDPatterns([
            "gemma-4-e2b",
            "gemma4-e2b",
        ]),
        excludePatterns: gemmaVisionTextExcludePatterns,
        architecture: "gemma4",
        modelTypes: gemma4ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 1.5,
        maxParametersB: 2.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256, 512],
        confidence: 0.85,
        extraNotes: gemmaProfileNotes + [
            "Shared-KV layers use AttentionKVState so compressed state can be reused without raw KV materialization."
        ]
    ),
    bundledProfile(
        id: "gemma-4-e4b",
        patterns: modelIDPatterns([
            "gemma-4-e4b",
            "gemma4-e4b",
        ]),
        excludePatterns: gemmaVisionTextExcludePatterns,
        architecture: "gemma4",
        modelTypes: gemma4ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256, 512],
        confidence: 0.85,
        extraNotes: gemmaProfileNotes + [
            "Shared-KV layers use AttentionKVState so compressed state can be reused without raw KV materialization."
        ]
    ),
    bundledProfile(
        id: "gemma-4-26b-a4b",
        patterns: modelIDPatterns([
            "gemma-4-26b",
            "gemma4-26b",
            "gemma-4-26b-a4b",
            "gemma4-26b-a4b",
            "gemma-4-text-26b-a4b",
            "gemma4-text-26b-a4b",
        ]),
        excludePatterns: gemmaVisionTextExcludePatterns,
        architecture: "gemma4",
        modelTypes: gemma4ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 24,
        maxParametersB: 28,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256, 512],
        confidence: 0.8,
        extraNotes: gemmaProfileNotes + [
            "Gemma 4 MoE active-parameter routing remains gated by config model_type and 256/512-dimensional KV heads."
        ]
    ),
    bundledProfile(
        id: "gemma-4-31b",
        patterns: modelIDPatterns([
            "gemma-4-31b",
            "gemma4-31b",
        ]),
        excludePatterns: gemmaVisionTextExcludePatterns,
        architecture: "gemma4",
        modelTypes: gemma4ModelTypes,
        modalities: [.text, .visionText],
        minParametersB: 29,
        maxParametersB: 33,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256, 512],
        confidence: 0.8,
        extraNotes: gemmaProfileNotes
    ),
    bundledProfile(
        id: "gemma-7b",
        patterns: modelIDPatterns([
            "gemma-7b",
            "gemma-1.1-7b",
            "gemma1.1-7b",
            "gemma-1-1-7b",
            "zephyr-7b-gemma",
        ]),
        excludePatterns: gemmaTextExcludePatterns + [
            "*gemma-2-7b*", "*gemma2-7b*", "*embeddinggemma*",
        ],
        architecture: "gemma",
        modelTypes: ["gemma"],
        minParametersB: 6.5,
        maxParametersB: 7.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        supportedContextLengths: [4096, 8192],
        safeContextLength: 8192,
        confidence: 0.75
    ),
    bundledProfile(
        id: "glm4-moe-lite",
        patterns: ["*glm4*moe*lite*", "*glm-4*moe*lite*", "*glm-4.7*flash*"],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "glm4_moe_lite",
        modelTypes: ["glm4_moe_lite"],
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 96, 128, 192],
        supportedValueHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: [
            "Latent attention may have different key and value dimensions; use two-stage compressed attention when they differ."
        ]
    ),
    bundledProfile(
        id: "mistral-7b",
        patterns: mistralFamilyPatterns([
            "mistral-7b",
            "mistral7b",
            "mistral-ft-optimized-1227",
        ]),
        excludePatterns: mistralTextExcludePatterns + ["*e5-*"],
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 6.5,
        maxParametersB: 7.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-nemo-12b",
        patterns: mistralFamilyPatterns(["mistral-nemo", "mistral-nemo-12b", "nemo-12b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 11,
        maxParametersB: 13,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.8,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "ministral-8b-2410",
        patterns: mistralFamilyPatterns(["ministral-8b", "ministral-8b-2410"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.8,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "codestral-22b",
        patterns: mistralFamilyPatterns(["codestral-22b", "codestral-22b-v0.1"]),
        excludePatterns: mistralTextExcludePatterns + ["*mamba-codestral*", "*mamba*codestral*"],
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 21,
        maxParametersB: 23,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-small-22b",
        patterns: mistralFamilyPatterns([
            "mistral-small-22b", "mistral-small-instruct-22b",
            "mistral-small-instruct-2409",
        ]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 21,
        maxParametersB: 23,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-small-24b",
        patterns: mistralFamilyPatterns([
            "mistral-small", "mistral-small-24b", "mistral-small-instruct",
        ]),
        excludePatterns: mistralTextExcludePatterns + [
            "*3.1*", "*3-1*", "*3.2*", "*3-2*", "*small-4*",
        ],
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-compatible-8b",
        patterns: mistralFamilyPatterns([
            "mistral-nemo-minitron-8b",
            "mistralai-8b-diagnosis-qa",
        ]),
        excludePatterns: mistralTextExcludePatterns + ["*e5-*", "*pairrm*"],
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: mistralProfileNotes + [
            "Covers config-compatible Mistral 8B derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "mistral-compatible-22b",
        patterns: mistralFamilyPatterns(["mistral-22b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 21,
        maxParametersB: 23,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: mistralProfileNotes + [
            "Covers config-compatible Mistral 22B derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "mistral-compatible-24b",
        patterns: mistralFamilyPatterns([
            "deephermes-3-mistral-24b",
            "dolphin-mistral-24b",
            "dolphin3.0-r1-mistral-24b",
            "dolphin3-0-r1-mistral-24b",
        ]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: mistralProfileNotes + [
            "Covers config-compatible Mistral 24B derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "mistral-large-2407",
        patterns: mistralFamilyPatterns(["mistral-large-instruct-2407"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 120,
        maxParametersB: 130,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.6,
        extraNotes: mistralProfileNotes + [
            "Covers config-compatible Mistral Large 2407 derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "devstral-small-24b",
        patterns: mistralFamilyPatterns(["devstral-small", "devstral-small-24b", "devstral-samll"]),
        excludePatterns: mistralTextExcludePatterns + [
            "*devstral-small-2-24b*", "*devstral-small-2.0*",
        ],
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "magistral-small-24b",
        patterns: mistralFamilyPatterns(["magistral-small", "magistral-small-24b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral",
        modelTypes: ["mistral"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "ministral3-3b",
        patterns: mistralFamilyPatterns(["ministral-3-3b", "ministral3-3b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3", "ministral3"],
        textConfigModelTypes: ["ministral3"],
        minParametersB: 2.5,
        maxParametersB: 3.5,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.8,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "ministral3-8b",
        patterns: mistralFamilyPatterns(["ministral-3-8b", "ministral3-8b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3", "ministral3"],
        textConfigModelTypes: ["ministral3"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.8,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "ministral3-14b",
        patterns: mistralFamilyPatterns(["ministral-3-14b", "ministral3-14b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3", "ministral3"],
        textConfigModelTypes: ["ministral3"],
        minParametersB: 13,
        maxParametersB: 15,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-small-3.1-24b",
        patterns: mistralFamilyPatterns([
            "mistral-small-3.1-24b",
            "mistral-small-3-1-24b",
            "mistral-small-3.1-text-24b",
            "mistral-small-3-1-text-24b",
        ]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3", "mistral"],
        textConfigModelTypes: ["mistral"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-small-3.2-24b",
        patterns: mistralFamilyPatterns(["mistral-small-3.2-24b", "mistral-small-3-2-24b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3"],
        textConfigModelTypes: ["mistral"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "devstral-small-2-24b",
        patterns: mistralFamilyPatterns(["devstral-small-2", "devstral-small-2-24b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3"],
        textConfigModelTypes: ["mistral", "ministral3"],
        minParametersB: 23,
        maxParametersB: 25.5,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "mistral-medium-3.5-128b",
        patterns: mistralFamilyPatterns(["mistral-medium-3.5-128b", "mistral-medium-3-5-128b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["mistral3"],
        textConfigModelTypes: ["mistral", "ministral3"],
        minParametersB: 120,
        maxParametersB: 135,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "devstral-2-123b",
        patterns: mistralFamilyPatterns(["devstral-2-123b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral3",
        modelTypes: ["ministral3"],
        minParametersB: 120,
        maxParametersB: 130,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.6,
        extraNotes: mistralProfileNotes + [
            "Covers config-compatible Devstral 2 123B variants that expose top-level model_type=ministral3."
        ]
    ),
    bundledProfile(
        id: "mistral-small-4-119b-a6b",
        patterns: mistralFamilyPatterns(["mistral-small-4-119b", "mistral-small-4-119b-a6b"]),
        excludePatterns: mistralTextExcludePatterns,
        architecture: "mistral4",
        modelTypes: ["mistral3"],
        textConfigModelTypes: ["mistral4"],
        minParametersB: 110,
        maxParametersB: 125,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        minRoutedExperts: 128,
        maxRoutedExperts: 128,
        supportedExpertsPerToken: [4],
        supportedKeyHeadDimensions: [128],
        supportedValueHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: mistralProfileNotes + [
            "Mistral Small 4 MoE is gated on nested text_config.model_type=mistral4 plus routed expert metadata."
        ]
    ),
    bundledProfile(
        id: "pixtral-12b",
        patterns: mistralFamilyPatterns(["pixtral-12b", "pixtral-12b-2409"]),
        excludePatterns: pixtralExcludePatterns,
        architecture: "pixtral",
        modelTypes: ["pixtral"],
        textConfigModelTypes: ["mistral"],
        modalities: [.visionText],
        minParametersB: 11,
        maxParametersB: 13,
        requiresModelType: true,
        requiresTextConfigModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: mistralProfileNotes
    ),
    bundledProfile(
        id: "granite-small",
        patterns: [
            "*granite-3.0-2b", "*granite-3.0-2b-*", "*granite-3.1-2b",
            "*granite-3.1-2b-*", "*granite-3.2-2b", "*granite-3.2-2b-*",
            "*granite-3.3-2b", "*granite-3.3-2b-*", "*granite-3.0-8b",
            "*granite-3.0-8b-*", "*granite-3.1-8b", "*granite-3.1-8b-*",
            "*granite-3.2-8b", "*granite-3.2-8b-*", "*granite-3.3-8b",
            "*granite-3.3-8b-*",
        ],
        excludePatterns: commonNonTextExcludePatterns + ["*moe*"],
        architecture: "granite",
        modelTypes: ["granite"],
        minParametersB: 1.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 80, 128],
        confidence: 0.65,
        extraNotes: ["Covers Granite dense <=8B variants with runtime head-dimension checks."]
    ),
    bundledProfile(
        id: "lfm2-small",
        patterns: [
            "*lfm2-350m", "*lfm2-350m-*", "*lfm2-700m", "*lfm2-700m-*",
            "*lfm2-1.2b", "*lfm2-1.2b-*", "*lfm-2-350m", "*lfm-2-350m-*",
            "*lfm-2-700m", "*lfm-2-700m-*", "*lfm-2-1.2b", "*lfm-2-1.2b-*",
        ],
        excludePatterns: commonNonTextExcludePatterns + ["*moe*"],
        architecture: "lfm2",
        modelTypes: ["lfm2"],
        minParametersB: 0.2,
        maxParametersB: 1.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: ["Covers LFM2 small dense variants with runtime head-dimension checks."]
    ),
    bundledProfile(
        id: "llama-2-7b",
        patterns: llamaFamilyPatterns(["llama-2-7b", "llama2-7b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 6.5,
        maxParametersB: 7.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        supportedContextLengths: [2048, 4096, 8192],
        safeContextLength: 8192,
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-2-13b",
        patterns: llamaFamilyPatterns(["llama-2-13b", "llama2-13b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 12,
        maxParametersB: 14,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        supportedContextLengths: [2048, 4096, 8192],
        safeContextLength: 8192,
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-2-70b",
        patterns: llamaFamilyPatterns(["llama-2-70b", "llama2-70b", "llama2-70"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 65,
        maxParametersB: 75,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        supportedContextLengths: [2048, 4096, 8192],
        safeContextLength: 8192,
        confidence: 0.7,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3-8b",
        patterns: llamaFamilyPatterns(["llama-3-8b", "llama3-8b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-8b",
        patterns: llamaFamilyPatterns([
            "llama-8b",
            "llama3-*8b",
            "llama-3-*8b",
            "llama3.1-*8b",
            "llama-3.1-*8b",
            "llama-guard-2-8b",
            "llama-pro-8b",
            "deepseek-r1-distill-llama-8b",
            "arrowcanaria-llama-8b",
            "llama-3.1-tulu-3-8b",
            "llama-3-elyza-jp-8b",
            "llama-3-groq-8b",
            "llama-3-smaug-8b",
            "llama-3-karamaru",
            "llama-3-refueled",
            "llama-3-swallow-8b",
            "llama-3.1-swallow-8b",
            "llama-3.1-nemotron-8b",
            "llama-3.1-supernova-lite",
            "llama-sea-lion-v3-8b",
            "llama-sea-lion-v3.5-8b",
        ]),
        excludePatterns: llamaExcludePatterns + [
            "*llama-3.1-8b*",
            "*llama3.1-8b*",
        ],
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes + [
            "Covers config-compatible Llama 8B derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "llama-3-3b",
        patterns: llamaFamilyPatterns(["llama-3-3b", "llama3-3b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 2.5,
        maxParametersB: 3.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.7,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3-16b",
        patterns: llamaFamilyPatterns(["llama-3-16b", "llama3-16b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 15,
        maxParametersB: 17,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3-70b",
        patterns: llamaFamilyPatterns(["llama-3-70b", "llama3-70b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 65,
        maxParametersB: 75,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.7,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-70b",
        patterns: llamaFamilyPatterns([
            "deepseek-r1-distill-llama-70b",
            "llama-3-groq-70b",
            "llama-3-swallow-70b",
            "llama-3.1-swallow-70b",
            "r1-1776-distill-llama-70b",
            "wayfarer-large-70b-llama-3.3",
            "deepcogito-cogito-v1-preview-llama-70b",
            "cogito-v2-preview-llama-70b",
            "llama3-*70b",
            "llama-3-*70b",
            "llama3.1-*70b",
            "llama-3.1-*70b",
            "llama3.3-*70b",
            "llama-3.3-*70b",
        ]),
        excludePatterns: llamaExcludePatterns + [
            "*llama-3.1-70b*",
            "*llama3.1-70b*",
            "*llama-3.1-nemotron-70b*",
            "*llama3.1-nemotron-70b*",
            "*llama-3.3-70b*",
            "*llama3.3-70b*",
        ],
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 65,
        maxParametersB: 75,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.6,
        extraNotes: llamaProfileNotes + [
            "Covers config-compatible Llama 70B derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "llama-3.1-8b",
        patterns: llamaFamilyPatterns(["llama-3.1-8b", "llama-3-1-8b", "llama3.1-8b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.85,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.1-4b",
        patterns: llamaFamilyPatterns(["llama-3.1-4b", "llama-3-1-4b", "llama3.1-4b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.1-16b",
        patterns: llamaFamilyPatterns(["llama-3.1-16b", "llama-3-1-16b", "llama3.1-16b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 15,
        maxParametersB: 17,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.7,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.1-70b",
        patterns: llamaFamilyPatterns([
            "llama-3.1-70b", "llama-3-1-70b", "llama3.1-70b",
            "llama-3.1-nemotron-70b",
        ]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 65,
        maxParametersB: 75,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.1-120b",
        patterns: llamaFamilyPatterns(["llama-3.1-120b", "llama-3-1-120b", "llama3.1-120b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 110,
        maxParametersB: 130,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3-120b",
        patterns: llamaFamilyPatterns([
            "llama-3-120b",
            "llama3-120b",
            "meta-llama-3-120b",
        ]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 110,
        maxParametersB: 130,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.6,
        extraNotes: llamaProfileNotes + [
            "Covers config-compatible Llama 3 120B derivatives discovered in mlx-community."
        ]
    ),
    bundledProfile(
        id: "llama-3.1-405b",
        patterns: llamaFamilyPatterns(["llama-3.1-405b", "llama-3-1-405b", "llama3.1-405b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 390,
        maxParametersB: 420,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.2-1b",
        patterns: llamaFamilyPatterns(["llama-3.2-1b", "llama-3-2-1b", "llama3.2-1b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 0.9,
        maxParametersB: 1.2,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64],
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.2-3b",
        patterns: llamaFamilyPatterns(["llama-3.2-3b", "llama-3-2-3b", "llama3.2-3b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 2.5,
        maxParametersB: 3.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.85,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.3-3b",
        patterns: llamaFamilyPatterns(["llama-3.3-3b", "llama-3-3-3b", "llama3.3-3b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 2.5,
        maxParametersB: 3.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.7,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-3.3-70b",
        patterns: llamaFamilyPatterns(["llama-3.3-70b", "llama-3-3-70b", "llama3.3-70b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 65,
        maxParametersB: 75,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-135m",
        patterns: llamaFamilyPatterns(["amd-llama-135m", "llama-135m"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 0.1,
        maxParametersB: 0.16,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-160m",
        patterns: llamaFamilyPatterns(["llama-160m", "llama3-160m"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 0.15,
        maxParametersB: 0.2,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-1b",
        patterns: llamaFamilyPatterns(["llama-1b", "llama3-1b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 0.8,
        maxParametersB: 1.3,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-2b",
        patterns: llamaFamilyPatterns([
            "llama-2b",
            "llama3-2b",
            "minicpm-2b-sft",
            "minicpm-2b-sft-*llama-format",
        ]),
        excludePatterns: llamaExcludePatterns + ["*llama-2-*", "*llama2-*"],
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 1.5,
        maxParametersB: 2.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-3b",
        patterns: llamaFamilyPatterns(["llama-3b", "llama3-3b", "llama_3b", "impish_llama_3b"]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 2.5,
        maxParametersB: 3.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "llama-compatible-30b",
        patterns: llamaFamilyPatterns([
            "llama-30b",
            "llama2-30b",
            "yayi2-30b",
        ]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 28,
        maxParametersB: 32,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [112, 128],
        confidence: 0.55,
        extraNotes: llamaProfileNotes + [
            "Covers config-compatible Llama 30B derivatives with nonstandard 112/128 KV head dimensions."
        ]
    ),
    bundledProfile(
        id: "llama-compatible-4b",
        patterns: llamaFamilyPatterns([
            "llama-4b",
            "llama3-4b",
            "llama-3.1-nemotron-nano-4b",
        ]),
        excludePatterns: llamaExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama"],
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: llamaProfileNotes
    ),
    bundledProfile(
        id: "phi-2",
        patterns: ["*phi-2", "*phi-2-*"],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "phi",
        modelTypes: ["phi"],
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [80],
        supportedContextLengths: [2048, 4096],
        safeContextLength: 4096,
        confidence: 0.75
    ),
    bundledProfile(
        id: "phi-3-mini",
        patterns: ["*phi-3-mini", "*phi-3-mini-*", "*phi3-mini", "*phi3-mini-*"],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "phi3",
        modelTypes: ["phi3"],
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [96],
        confidence: 0.75
    ),
    bundledProfile(
        id: "phi-3.5-mini",
        patterns: [
            "*phi-3.5-mini", "*phi-3.5-mini-*", "*phi-3-5-mini",
            "*phi-3-5-mini-*", "*phi3.5-mini", "*phi3.5-mini-*",
        ],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "phi3",
        modelTypes: ["phi3"],
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [96],
        confidence: 0.75
    ),
    bundledProfile(
        id: "phi-4-mini",
        patterns: ["*phi-4-mini", "*phi-4-mini-*", "*phi4-mini", "*phi4-mini-*"],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "phi3",
        modelTypes: ["phi3"],
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75
    ),
    bundledProfile(
        id: "qwen2.5-small",
        patterns: qwen25SmallPatterns,
        excludePatterns: commonNonTextExcludePatterns + [
            "*qwen3*", "*qwen-3*", "*moe*", "*a3b*", "*a22b*", "*next*", "*deepseek*",
        ],
        architecture: "qwen2",
        modelTypes: ["qwen2"],
        minParametersB: 0.4,
        maxParametersB: 7.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: ["Covers Qwen2.5 dense <=7B variants with runtime head-dimension checks."]
    ),
    bundledProfile(
        id: "qwen3-0.6b",
        patterns: ["*qwen3-0.6b", "*qwen3-0.6b-*", "*qwen-3-0.6b", "*qwen-3-0.6b-*"],
        excludePatterns: qwen3ExcludePatterns,
        architecture: "qwen3",
        modelTypes: ["qwen3"],
        minParametersB: 0.5,
        maxParametersB: 0.7,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75
    ),
    bundledProfile(
        id: "qwen3-1.7b",
        patterns: ["*qwen3-1.7b", "*qwen3-1.7b-*", "*qwen-3-1.7b", "*qwen-3-1.7b-*"],
        excludePatterns: qwen3ExcludePatterns,
        architecture: "qwen3",
        modelTypes: ["qwen3"],
        minParametersB: 1.5,
        maxParametersB: 1.9,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75
    ),
    bundledProfile(
        id: "qwen3-4b",
        patterns: ["*qwen3-4b", "*qwen3-4b-*", "*qwen-3-4b", "*qwen-3-4b-*"],
        excludePatterns: qwen3ExcludePatterns,
        architecture: "qwen3",
        modelTypes: ["qwen3"],
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.85
    ),
    bundledProfile(
        id: "qwen3-8b",
        patterns: ["*qwen3-8b", "*qwen3-8b-*", "*qwen-3-8b", "*qwen-3-8b-*"],
        excludePatterns: qwen3ExcludePatterns,
        architecture: "qwen3",
        modelTypes: ["qwen3"],
        minParametersB: 7.5,
        maxParametersB: 8.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [128],
        confidence: 0.75
    ),
    bundledProfile(
        id: "qwen3.5-2b",
        patterns: qwen35FamilyPatterns("3.5", size: "2b"),
        excludePatterns: qwen35DenseExcludePatterns + ["*qwen3-2b*", "*qwen-3-2b*"],
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 1.5,
        maxParametersB: 2.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.85,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-0.8b",
        patterns: qwen35FamilyPatterns("3.5", size: "0.8b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 0.7,
        maxParametersB: 1,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.85,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-4b",
        patterns: qwen35FamilyPatterns("3.5", size: "4b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 3.5,
        maxParametersB: 4.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.85,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-9b",
        patterns: qwen35FamilyPatterns("3.5", size: "9b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 8,
        maxParametersB: 10.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.85,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-27b",
        patterns: qwen35FamilyPatterns("3.5", size: "27b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 24,
        maxParametersB: 30,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.8,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.6-27b",
        patterns: qwen35FamilyPatterns("3.6", size: "27b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 24,
        maxParametersB: 30,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.8,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-40b",
        patterns: qwen35FamilyPatterns("3.5", size: "40b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 38,
        maxParametersB: 42,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.65,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.6-40b",
        patterns: qwen35FamilyPatterns("3.6", size: "40b"),
        excludePatterns: qwen35DenseExcludePatterns,
        architecture: "qwen3_5",
        modelTypes: qwen35ModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 38,
        maxParametersB: 42,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.65,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-35b-a3b",
        patterns: qwen35MoEPatterns("3.5", totalSize: "35b", activeSize: "a3b"),
        excludePatterns: qwen35MoEExcludePatterns + ["*a10b*", "*a17b*", "*a22b*"],
        architecture: "qwen3_5_moe",
        modelTypes: qwen35MoEModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 34,
        maxParametersB: 37,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.75,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.6-35b-a3b",
        patterns: qwen35MoEPatterns("3.6", totalSize: "35b", activeSize: "a3b"),
        excludePatterns: qwen35MoEExcludePatterns + ["*a10b*", "*a17b*", "*a22b*"],
        architecture: "qwen3_5_moe",
        modelTypes: qwen35MoEModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 34,
        maxParametersB: 37,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.75,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-97b-a10b",
        patterns: qwen35MoEPatterns("3.5", totalSize: "97b", activeSize: "a10b")
            + ["*qwen3.5*97b-a10b*", "*qwen-3.5*97b-a10b*", "*qwen3-5*97b-a10b*"],
        excludePatterns: qwen35MoEExcludePatterns + ["*a3b*", "*a17b*", "*a22b*"],
        architecture: "qwen3_5_moe",
        modelTypes: qwen35MoEModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 90,
        maxParametersB: 105,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.65,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-122b-a10b",
        patterns: qwen35MoEPatterns("3.5", totalSize: "122b", activeSize: "a10b"),
        excludePatterns: qwen35MoEExcludePatterns + ["*a3b*", "*a17b*", "*a22b*"],
        architecture: "qwen3_5_moe",
        modelTypes: qwen35MoEModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 115,
        maxParametersB: 130,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.7,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "qwen3.5-397b-a17b",
        patterns: qwen35MoEPatterns("3.5", totalSize: "397b", activeSize: "a17b"),
        excludePatterns: qwen35MoEExcludePatterns + ["*a3b*", "*a10b*", "*a22b*"],
        architecture: "qwen3_5_moe",
        modelTypes: qwen35MoEModelTypes,
        modalities: qwen35Modalities,
        minParametersB: 380,
        maxParametersB: 410,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [256],
        confidence: 0.65,
        extraNotes: qwen35ProfileNotes
    ),
    bundledProfile(
        id: "smollm-small",
        patterns: [
            "*smollm-135m", "*smollm-135m-*", "*smollm-360m", "*smollm-360m-*",
            "*smollm-1.7b", "*smollm-1.7b-*", "*smollm2-135m", "*smollm2-135m-*",
            "*smollm2-360m", "*smollm2-360m-*", "*smollm2-1.7b", "*smollm2-1.7b-*",
        ],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama", "smollm", "smollm2"],
        minParametersB: 0.1,
        maxParametersB: 2,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.65,
        extraNotes: [
            "Covers SmolLM and SmolLM2 small dense variants with runtime head-dimension checks."
        ]
    ),
    bundledProfile(
        id: "smollm3-3b",
        patterns: ["*smollm3-3b", "*smollm3-3b-*", "*smollm-3-3b", "*smollm-3-3b-*"],
        excludePatterns: commonNonTextExcludePatterns,
        architecture: "llama",
        modelTypes: ["llama", "smollm3"],
        minParametersB: 2.5,
        maxParametersB: 3.5,
        requiresModelType: true,
        requiresHeadDimensions: true,
        supportedKeyHeadDimensions: [64, 128],
        confidence: 0.75
    ),
].map(applyingBundledProfileOptimizations)
