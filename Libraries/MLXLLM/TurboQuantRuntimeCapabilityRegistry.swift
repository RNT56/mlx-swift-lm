// Copyright © 2026 Schtack

import Foundation

/// Cache topology supported by a concrete MLX LLM model type for TurboQuant generation.
public enum MLXTurboQuantCacheTopology: String, Codable, Hashable, Sendable {
    case standardAttentionKV
    case hybridAttentionKVAndNativeState
    case gatedVLMOrDualModel
    case unsupported
}

/// Runtime capability exported by MLXLLM so host apps do not infer TurboQuant support from repository names.
public struct MLXTurboQuantRuntimeModelCapability: Codable, Hashable, Sendable {
    public var modelType: String
    public var supportsThrowingTurboQuantAttention: Bool
    public var cacheTopology: MLXTurboQuantCacheTopology
    public var note: String?

    public init(
        modelType: String,
        supportsThrowingTurboQuantAttention: Bool,
        cacheTopology: MLXTurboQuantCacheTopology,
        note: String? = nil
    ) {
        self.modelType = modelType
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
        self.supportsThrowingTurboQuantAttention = supportsThrowingTurboQuantAttention
        self.cacheTopology = cacheTopology
        self.note = note
    }
}

/// Exported capability registry for MLXLLM text runtimes.
public enum MLXTurboQuantRuntimeCapabilityRegistry {
    public static let capabilities: [MLXTurboQuantRuntimeModelCapability] = [
        .standard("llama"),
        .standard("mistral"),
        .standard("ministral3"),
        .standard("mistral3"),
        .standard("mistral4"),
        .standard("gemma"),
        .standard("gemma2"),
        .standard("gemma3"),
        .standard("gemma3_text"),
        .standard("gemma3n"),
        .standard("gemma3n_text"),
        .standard("gemma4"),
        .standard("gemma4_text"),
        .gated(
            "gemma4_assistant",
            note: "Gemma4 assistant is draft-only MTP and requires explicit dual-model orchestration."
        ),
        .standard("qwen2"),
        .standard("qwen3"),
        .standard("qwen3_moe"),
        .hybrid("qwen3_5"),
        .hybrid("qwen3_5_text"),
        .hybrid("qwen3_5_moe"),
        .hybrid("qwen3_5_moe_text"),
        .standard("acereason"),
        .standard("phi"),
        .standard("phi3"),
        .standard("granite"),
        .standard("exaone4"),
        .standard("smollm3"),
        .hybrid("lfm2"),
        .standard("glm4_moe_lite"),
    ]

    public static let capabilitiesByModelType: [String: MLXTurboQuantRuntimeModelCapability] =
        Dictionary(uniqueKeysWithValues: capabilities.map { ($0.modelType, $0) })

    public static let throwingTurboQuantAttentionModelTypes: Set<String> = Set(
        capabilities
            .filter(\.supportsThrowingTurboQuantAttention)
            .map(\.modelType)
    )

    public static func capability(
        for modelType: String?
    ) -> MLXTurboQuantRuntimeModelCapability? {
        guard let modelType else { return nil }
        return capabilitiesByModelType[normalize(modelType)]
    }

    public static func supportsThrowingTurboQuantAttention(modelType: String?) -> Bool {
        capability(for: modelType)?.supportsThrowingTurboQuantAttention == true
    }

    private static func normalize(_ value: String) -> String {
        value
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
    }
}

private extension MLXTurboQuantRuntimeModelCapability {
    static func standard(_ modelType: String) -> Self {
        Self(
            modelType: modelType,
            supportsThrowingTurboQuantAttention: true,
            cacheTopology: .standardAttentionKV
        )
    }

    static func hybrid(_ modelType: String) -> Self {
        Self(
            modelType: modelType,
            supportsThrowingTurboQuantAttention: true,
            cacheTopology: .hybridAttentionKVAndNativeState
        )
    }

    static func gated(_ modelType: String, note: String) -> Self {
        Self(
            modelType: modelType,
            supportsThrowingTurboQuantAttention: false,
            cacheTopology: .gatedVLMOrDualModel,
            note: note
        )
    }
}
