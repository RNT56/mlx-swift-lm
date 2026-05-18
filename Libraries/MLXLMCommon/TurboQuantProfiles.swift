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

public struct TurboQuantProfile: Codable, Equatable, Identifiable, Sendable {
    public var id: String
    public var modelPatterns: [String]
    public var architecture: String?
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
    public var measured: TurboQuantProfileMeasurements
    public var notes: [String]

    public init(
        id: String,
        modelPatterns: [String],
        architecture: String? = nil,
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
        measured: TurboQuantProfileMeasurements = TurboQuantProfileMeasurements(),
        notes: [String] = []
    ) {
        self.id = id
        self.modelPatterns = modelPatterns
        self.architecture = architecture
        self.supportedKeyHeadDimensions = supportedKeyHeadDimensions
        self.supportedValueHeadDimensions = supportedValueHeadDimensions ?? supportedKeyHeadDimensions
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
        self.measured = measured
        self.notes = notes
    }

    public func supports(
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil
    ) -> Bool {
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
}

public struct TurboQuantProfileRegistry: Sendable {
    public var profiles: [TurboQuantProfile]

    public init(profiles: [TurboQuantProfile]) {
        self.profiles = profiles
    }

    public static let bundled = TurboQuantProfileRegistry(profiles: bundledProfiles)

    public func profile(
        for modelID: String,
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil
    ) -> TurboQuantProfile? {
        profiles.first { profile in
            profile.matches(modelID: modelID)
                && profile.supports(
                    keyHeadDimension: keyHeadDimension,
                    valueHeadDimension: valueHeadDimension,
                    maskMode: maskMode,
                    contextLength: contextLength
                )
        }
    }

    public static func loadJSONProfiles(from directory: URL) throws -> [TurboQuantProfile] {
        let fileManager = FileManager.default
        let urls = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        )
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try urls
            .filter { $0.pathExtension.lowercased() == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .map { url in
                let data = try Data(contentsOf: url)
                return try decoder.decode(TurboQuantProfile.self, from: data)
            }
    }
}

public extension GenerateParameters {
    init(
        turboQuantProfile profile: TurboQuantProfile,
        base parameters: GenerateParameters = GenerateParameters()
    ) {
        self = profile.applying(to: parameters)
    }

    init(
        turboQuantModelID modelID: String,
        registry: TurboQuantProfileRegistry = .bundled,
        keyHeadDimension: Int? = nil,
        valueHeadDimension: Int? = nil,
        maskMode: TurboQuantMaskMode = .causal,
        contextLength: Int? = nil,
        base parameters: GenerateParameters = GenerateParameters()
    ) {
        if let profile = registry.profile(
            for: modelID,
            keyHeadDimension: keyHeadDimension,
            valueHeadDimension: valueHeadDimension,
            maskMode: maskMode,
            contextLength: contextLength
        ) {
            self = profile.applying(to: parameters)
        } else {
            self = parameters
        }
    }

    mutating func applyTurboQuantProfile(_ profile: TurboQuantProfile) {
        self = profile.applying(to: self)
    }
}

private extension TurboQuantProfile {
    func matches(modelID: String) -> Bool {
        let normalizedID = Self.normalized(modelID)
        return modelPatterns.contains { pattern in
            Self.matches(pattern: Self.normalized(pattern), modelID: normalizedID)
        }
    }

    static func normalized(_ value: String) -> String {
        value
            .lowercased()
            .replacingOccurrences(of: "_", with: "-")
            .replacingOccurrences(of: " ", with: "-")
    }

    static func matches(pattern: String, modelID: String) -> Bool {
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

private let bundledProfiles: [TurboQuantProfile] = [
    TurboQuantProfile(
        id: "llama-3.2-3b",
        modelPatterns: ["*llama-3.2*3b*", "*llama-3-2*3b*"],
        architecture: "llama",
        supportedKeyHeadDimensions: [128],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes
    ),
    TurboQuantProfile(
        id: "llama-3.1-8b",
        modelPatterns: ["*llama-3.1*8b*", "*llama-3-1*8b*"],
        architecture: "llama",
        supportedKeyHeadDimensions: [128],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes
    ),
    TurboQuantProfile(
        id: "mistral-7b",
        modelPatterns: ["*mistral*7b*", "*mistral-7b*"],
        architecture: "mistral",
        supportedKeyHeadDimensions: [128],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes
    ),
    TurboQuantProfile(
        id: "gemma-3-4b",
        modelPatterns: ["*gemma-3*4b*", "*gemma3*4b*"],
        architecture: "gemma3",
        supportedKeyHeadDimensions: [128, 256],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes
    ),
    TurboQuantProfile(
        id: "gemma-4",
        modelPatterns: ["*gemma-4*", "*gemma4*"],
        architecture: "gemma4",
        supportedKeyHeadDimensions: [64, 128, 256],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes + [
            "Shared-KV layers use AttentionKVState so compressed state can be reused without raw KV materialization."
        ]
    ),
    TurboQuantProfile(
        id: "gemma-3n",
        modelPatterns: ["*gemma-3n*", "*gemma3n*"],
        architecture: "gemma3n",
        supportedKeyHeadDimensions: [64, 128, 256],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes + [
            "Shared-KV layers use AttentionKVState so compressed state can be reused without raw KV materialization."
        ]
    ),
    TurboQuantProfile(
        id: "qwen3-4b",
        modelPatterns: ["*qwen3*4b*", "*qwen-3*4b*"],
        architecture: "qwen3",
        supportedKeyHeadDimensions: [128],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes
    ),
    TurboQuantProfile(
        id: "qwen3.5-2b",
        modelPatterns: ["*qwen3.5*2b*", "*qwen3-5*2b*", "*qwen-3.5*2b*"],
        architecture: "qwen3.5",
        supportedKeyHeadDimensions: [128],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes
    ),
    TurboQuantProfile(
        id: "glm4-moe-lite",
        modelPatterns: ["*glm4*moe*lite*", "*glm-4*moe*lite*"],
        architecture: "glm4_moe_lite",
        supportedKeyHeadDimensions: [64, 96, 128, 192],
        supportedValueHeadDimensions: [64, 128],
        safeMaskModes: commonSafeMasks,
        notes: turboQuantProfileNotes + [
            "Latent attention may have different key and value dimensions; use two-stage compressed attention when they differ."
        ]
    ),
]
