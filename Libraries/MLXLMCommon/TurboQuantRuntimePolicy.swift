// Copyright © 2026 RNT56.

import Foundation
import MLX

public enum TurboQuantRuntimeMode: String, Codable, Sendable, CaseIterable, Equatable {
    case auto
    case rawPreferred
    case throughputTurboQuant
    case capacityTurboQuant
}

public enum TurboQuantRuntimeRoute: String, Codable, Sendable, CaseIterable, Equatable {
    case rawSDPA
    case throughputTurboQuantNativeSDPA
    case capacityTurboQuantCompressed
}

public enum TurboQuantSparseValuePolicy: Hashable, Codable, Sendable {
    case off
    case auto(threshold: Float)
    case force(threshold: Float)

    private enum CodingKeys: String, CodingKey {
        case kind
        case threshold
    }

    private enum Kind: String, Codable {
        case off
        case auto
        case force
    }

    public static let defaultAutoThreshold: Float = 1e-6
    public static let profileDefault = TurboQuantSparseValuePolicy.off

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(Kind.self, forKey: .kind)
        let threshold =
            try container.decodeIfPresent(Float.self, forKey: .threshold)
            ?? Self.defaultAutoThreshold
        switch kind {
        case .off:
            self = .off
        case .auto:
            self = .auto(threshold: threshold)
        case .force:
            self = .force(threshold: threshold)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .off:
            try container.encode(Kind.off, forKey: .kind)
        case .auto(let threshold):
            try container.encode(Kind.auto, forKey: .kind)
            try container.encode(max(0, threshold), forKey: .threshold)
        case .force(let threshold):
            try container.encode(Kind.force, forKey: .kind)
            try container.encode(max(0, threshold), forKey: .threshold)
        }
    }

    public var threshold: Float? {
        switch self {
        case .off:
            nil
        case .auto(let threshold), .force(let threshold):
            max(0, threshold)
        }
    }

    public var isForced: Bool {
        if case .force = self { return true }
        return false
    }

    public func resolvedThreshold(
        runtimeMode: TurboQuantRuntimeMode,
        contextLength: Int,
        minimumAutoContextLength: Int = 16_384
    ) -> Float? {
        switch self {
        case .off:
            nil
        case .auto:
            nil
        case .force(let threshold):
            runtimeMode == .capacityTurboQuant ? max(0, threshold) : nil
        }
    }
}

public enum TurboQuantSparseValueSelectionMode: String, Hashable, Codable, Sendable, CaseIterable {
    case off
    case threshold
    case topK
    case cumulativeMass
    case hybridCumulativeMassTopK
    case blockThreshold
    case pageTopK
    case candidateSparse

    public var nativeMode: TurboQuantSparseValueNativeSelectionMode {
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
        case .blockThreshold:
            return .blockThreshold
        case .pageTopK:
            return .pageTopK
        case .candidateSparse:
            return .candidateSparse
        }
    }
}

public struct TurboQuantSparseValueSelection: Hashable, Codable, Sendable {
    public var mode: TurboQuantSparseValueSelectionMode
    public var threshold: Float?
    public var topK: Int?
    /// Fraction of softmax mass to retain, e.g. 0.995 for 99.5%.
    public var cumulativeMass: Float?
    public var maxTopK: Int?
    public var recentTokens: Int?
    public var candidatePages: Int?

    public init(
        mode: TurboQuantSparseValueSelectionMode = .off,
        threshold: Float? = nil,
        topK: Int? = nil,
        cumulativeMass: Float? = nil,
        maxTopK: Int? = nil,
        recentTokens: Int? = nil,
        candidatePages: Int? = nil
    ) {
        self.mode = mode
        self.threshold = threshold.map { max(0, $0) }
        self.topK = topK.map { max(1, $0) }
        self.cumulativeMass = cumulativeMass.map { min(1, max(0, $0)) }
        self.maxTopK = maxTopK.map { max(1, $0) }
        self.recentTokens = recentTokens.map { max(0, $0) }
        self.candidatePages = candidatePages.map { max(0, $0) }
    }

    public static let off = TurboQuantSparseValueSelection()

    public static func threshold(_ threshold: Float) -> Self {
        Self(mode: .threshold, threshold: threshold)
    }

    public static func blockThreshold(_ threshold: Float) -> Self {
        Self(mode: .blockThreshold, threshold: threshold)
    }

    public static func topK(_ topK: Int) -> Self {
        Self(mode: .topK, topK: topK)
    }

    public static func pageTopK(_ topK: Int) -> Self {
        Self(mode: .pageTopK, topK: topK)
    }

    public static func candidateSparse(
        recentTokens: Int,
        candidatePages: Int,
        olderTokenBudget: Int
    ) -> Self {
        Self(
            mode: .candidateSparse,
            topK: olderTokenBudget,
            recentTokens: recentTokens,
            candidatePages: candidatePages
        )
    }

    public static func cumulativeMass(_ mass: Float) -> Self {
        Self(mode: .cumulativeMass, cumulativeMass: mass)
    }

    public static func hybrid(cumulativeMass: Float, maxTopK: Int) -> Self {
        Self(
            mode: .hybridCumulativeMassTopK,
            cumulativeMass: cumulativeMass,
            maxTopK: maxTopK
        )
    }

    public static func thresholdPolicy(_ policy: TurboQuantSparseValuePolicy) -> Self {
        switch policy {
        case .off, .auto:
            return .off
        case .force(let threshold):
            return .threshold(threshold)
        }
    }

    public var isEnabled: Bool {
        switch mode {
        case .off:
            return false
        case .threshold, .blockThreshold:
            return (threshold ?? 0) > 0
        case .topK, .pageTopK:
            return (topK ?? 0) > 0
        case .candidateSparse:
            return (recentTokens ?? 0) > 0 || (candidatePages ?? 0) > 0 || (topK ?? 0) > 0
        case .cumulativeMass:
            return (cumulativeMass ?? 0) > 0
        case .hybridCumulativeMassTopK:
            return (cumulativeMass ?? 0) > 0 && (maxTopK ?? topK ?? 0) > 0
        }
    }

    public func resolved(
        runtimeMode: TurboQuantRuntimeMode,
        contextLength: Int,
        policy: TurboQuantSparseValuePolicy = .off,
        minimumAutoContextLength: Int = 16_384
    ) -> Self {
        guard runtimeMode == .capacityTurboQuant else { return .off }
        if mode != .off {
            return isEnabled ? self : .off
        }
        guard let threshold = policy.resolvedThreshold(
            runtimeMode: runtimeMode,
            contextLength: contextLength,
            minimumAutoContextLength: minimumAutoContextLength
        ) else {
            return .off
        }
        return .threshold(threshold)
    }

    public var resolvedThreshold: Float? {
        mode == .threshold || mode == .blockThreshold ? threshold : nil
    }

    public func nativeOptions(
        scale: Float,
        causal: Bool,
        diagnostics: Bool,
        backendVersion: Int?
    ) -> TurboQuantNativeAttentionOptions {
        TurboQuantNativeAttentionOptions(
            scale: scale,
            causal: causal,
            sparseVThreshold: resolvedThreshold ?? 0,
            sparseVSelectionMode: mode.nativeMode,
            sparseVTopK: topK ?? 0,
            sparseVCumulativeMass: cumulativeMass ?? 0,
            sparseVMaxTopK: maxTopK ?? 0,
            sparseVRecentTokens: recentTokens ?? 0,
            sparseVCandidatePages: candidatePages ?? 0,
            diagnostics: diagnostics,
            backendVersion: backendVersion ?? TurboQuantNativeAttentionOptions.backendVersion
        )
    }
}

public enum TurboQuantKeyPrecision: String, Codable, Sendable, CaseIterable, Equatable {
    case fp16OrQ8
    case fp16
    case affineQ8
    case turbo8
    case turbo4v2
    case turbo3_5
    case turbo2_5

    public var compressedPreset: TurboQuantPreset {
        switch self {
        case .fp16OrQ8, .fp16, .affineQ8, .turbo8:
            .turbo8
        case .turbo4v2:
            .turbo4v2
        case .turbo3_5:
            .turbo3_5
        case .turbo2_5:
            .turbo2_5
        }
    }

    public var isHighPrecision: Bool {
        switch self {
        case .fp16OrQ8, .fp16, .affineQ8, .turbo8:
            true
        case .turbo4v2, .turbo3_5, .turbo2_5:
            false
        }
    }

    public var diagnosticLabel: String {
        rawValue
    }
}

public enum TurboQuantValuePrecision: String, Codable, Sendable, CaseIterable, Equatable {
    case fp16
    case turbo8
    case turbo4v2
    case turbo3_5
    case turbo2_5

    public var valueBits: Int? {
        switch self {
        case .fp16:
            nil
        case .turbo8:
            8
        case .turbo4v2:
            4
        case .turbo3_5:
            3
        case .turbo2_5:
            2
        }
    }

    public static func compressed(bits: Int) -> TurboQuantValuePrecision {
        switch min(8, max(2, bits)) {
        case 8:
            return .turbo8
        case 4...7:
            return .turbo4v2
        case 3:
            return .turbo3_5
        default:
            return .turbo2_5
        }
    }
}

public enum TurboQuantBoundaryCachePrecision: String, Codable, Sendable, CaseIterable, Hashable {
    case affineK8V4
    case raw
}

public enum TurboQuantBoundaryPolicy: Hashable, Codable, Sendable {
    case profileDefault
    case disabled
    case protectedEdges(first: Int, last: Int)
    case custom([Int])

    private enum CodingKeys: String, CodingKey {
        case kind
        case first
        case last
        case layers
    }

    private enum Kind: String, Codable {
        case profileDefault
        case disabled
        case protectedEdges
        case custom
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(Kind.self, forKey: .kind)
        switch kind {
        case .profileDefault:
            self = .profileDefault
        case .disabled:
            self = .disabled
        case .protectedEdges:
            self = .protectedEdges(
                first: try container.decodeIfPresent(Int.self, forKey: .first) ?? 0,
                last: try container.decodeIfPresent(Int.self, forKey: .last) ?? 0
            )
        case .custom:
            self = .custom(try container.decodeIfPresent([Int].self, forKey: .layers) ?? [])
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .profileDefault:
            try container.encode(Kind.profileDefault, forKey: .kind)
        case .disabled:
            try container.encode(Kind.disabled, forKey: .kind)
        case .protectedEdges(let first, let last):
            try container.encode(Kind.protectedEdges, forKey: .kind)
            try container.encode(max(0, first), forKey: .first)
            try container.encode(max(0, last), forKey: .last)
        case .custom(let layers):
            try container.encode(Kind.custom, forKey: .kind)
            try container.encode(layers.map { max(0, $0) }, forKey: .layers)
        }
    }

    public func resolved(
        layerCount: Int,
        requiresProtection: Bool,
        defaultLeading: Int = 2,
        defaultTrailing: Int = 2
    ) -> TurboQuantBoundaryPolicy {
        switch self {
        case .profileDefault:
            requiresProtection
                ? .protectedEdges(first: defaultLeading, last: defaultTrailing)
                : .disabled
        case .disabled, .protectedEdges, .custom:
            self
        }
    }

    public func protectedLayerIndexes(
        layerCount: Int,
        requiresProtection: Bool = true,
        defaultLeading: Int = 2,
        defaultTrailing: Int = 2
    ) -> Set<Int> {
        let layerCount = max(0, layerCount)
        guard layerCount > 0 else { return [] }
        let policy = resolved(
            layerCount: layerCount,
            requiresProtection: requiresProtection,
            defaultLeading: defaultLeading,
            defaultTrailing: defaultTrailing
        )
        switch policy {
        case .profileDefault:
            return []
        case .disabled:
            return []
        case .protectedEdges(let first, let last):
            let leading = Set(0 ..< min(max(0, first), layerCount))
            let trailingStart = max(0, layerCount - max(0, last))
            let trailing = Set(trailingStart ..< layerCount)
            return leading.union(trailing)
        case .custom(let layers):
            return Set(layers.filter { $0 >= 0 && $0 < layerCount })
        }
    }

    public func protects(
        layerIndex: Int,
        layerCount: Int,
        requiresProtection: Bool = true,
        defaultLeading: Int = 2,
        defaultTrailing: Int = 2
    ) -> Bool {
        protectedLayerIndexes(
            layerCount: layerCount,
            requiresProtection: requiresProtection,
            defaultLeading: defaultLeading,
            defaultTrailing: defaultTrailing
        ).contains(layerIndex)
    }
}

public struct TurboQuantKVPrecisionPolicy: Hashable, Codable, Sendable {
    public var key: TurboQuantKeyPrecision
    public var value: TurboQuantValuePrecision
    public var boundary: TurboQuantBoundaryPolicy
    public var boundaryCachePrecision: TurboQuantBoundaryCachePrecision

    public init(
        key: TurboQuantKeyPrecision,
        value: TurboQuantValuePrecision,
        boundary: TurboQuantBoundaryPolicy = .profileDefault,
        boundaryCachePrecision: TurboQuantBoundaryCachePrecision = .affineK8V4
    ) {
        self.key = key
        self.value = value
        self.boundary = boundary
        self.boundaryCachePrecision = boundaryCachePrecision
    }

    public static let qwenQ4Default = TurboQuantKVPrecisionPolicy(
        key: .fp16OrQ8,
        value: .turbo4v2,
        boundary: .protectedEdges(first: 1, last: 1),
        boundaryCachePrecision: .affineK8V4
    )

    private enum CodingKeys: String, CodingKey {
        case key
        case value
        case boundary
        case boundaryCachePrecision
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            key: try container.decode(TurboQuantKeyPrecision.self, forKey: .key),
            value: try container.decode(TurboQuantValuePrecision.self, forKey: .value),
            boundary: try container.decodeIfPresent(TurboQuantBoundaryPolicy.self, forKey: .boundary)
                ?? .profileDefault,
            boundaryCachePrecision: try container.decodeIfPresent(
                TurboQuantBoundaryCachePrecision.self,
                forKey: .boundaryCachePrecision
            ) ?? .affineK8V4
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(key, forKey: .key)
        try container.encode(value, forKey: .value)
        try container.encode(boundary, forKey: .boundary)
        try container.encode(boundaryCachePrecision, forKey: .boundaryCachePrecision)
    }

    public static func legacy(
        preset: TurboQuantPreset,
        valueBits: Int?,
        boundary: TurboQuantBoundaryPolicy = .profileDefault,
        boundaryCachePrecision: TurboQuantBoundaryCachePrecision = .affineK8V4
    ) -> TurboQuantKVPrecisionPolicy {
        let key: TurboQuantKeyPrecision =
            switch preset {
            case .turbo8:
                .turbo8
            case .turbo4, .turbo4v2:
                .turbo4v2
            case .turbo3_5:
                .turbo3_5
            case .turbo2_5:
                .turbo2_5
            }
        return TurboQuantKVPrecisionPolicy(
            key: key,
            value: .compressed(bits: valueBits ?? preset.defaultValueBits),
            boundary: boundary,
            boundaryCachePrecision: boundaryCachePrecision
        )
    }

    public var compressedKeyPreset: TurboQuantPreset {
        key.compressedPreset
    }

    public var resolvedValueBits: Int? {
        value.valueBits
    }

    public var usesLowPrecisionKey: Bool {
        !key.isHighPrecision
    }

    public var usesLowPrecisionValue: Bool {
        guard let valueBits = value.valueBits else { return false }
        return valueBits < 8
    }

    public var requiresBoundaryProtection: Bool {
        usesLowPrecisionKey || usesLowPrecisionValue
    }

    public var requiresRawBoundaryProtection: Bool {
        requiresBoundaryProtection
    }

    public func resolvedBoundaryPolicy(layerCount: Int) -> TurboQuantBoundaryPolicy {
        boundary.resolved(
            layerCount: layerCount,
            requiresProtection: requiresBoundaryProtection
        )
    }

    public func protectedBoundaryLayerIndexes(layerCount: Int) -> Set<Int> {
        boundary.protectedLayerIndexes(
            layerCount: layerCount,
            requiresProtection: requiresBoundaryProtection
        )
    }

    public func protectsBoundaryLayer(layerIndex: Int, layerCount: Int) -> Bool {
        boundary.protects(
            layerIndex: layerIndex,
            layerCount: layerCount,
            requiresProtection: requiresBoundaryProtection
        )
    }

    public var diagnostics: [TurboQuantDiagnosticEvent] {
        guard usesLowPrecisionKey else { return [] }
        return [
            TurboQuantDiagnosticEvent(
                category: "precision-policy",
                message: "TurboQuant key precision \(key.diagnosticLabel) requires exact profile evidence before verified/certified promotion.",
                metadata: ["keyPrecision": key.rawValue]
            )
        ]
    }
}
