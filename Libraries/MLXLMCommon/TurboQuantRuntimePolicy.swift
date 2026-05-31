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
    public static let profileDefault = TurboQuantSparseValuePolicy.auto(
        threshold: defaultAutoThreshold
    )

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
        case .auto(let threshold):
            runtimeMode == .capacityTurboQuant && contextLength >= minimumAutoContextLength
                ? max(0, threshold)
                : nil
        case .force(let threshold):
            runtimeMode == .capacityTurboQuant ? max(0, threshold) : nil
        }
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
        case .turbo4v2, .turbo3_5:
            4
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

    public init(
        key: TurboQuantKeyPrecision,
        value: TurboQuantValuePrecision,
        boundary: TurboQuantBoundaryPolicy = .profileDefault
    ) {
        self.key = key
        self.value = value
        self.boundary = boundary
    }

    public static let qwenQ4Default = TurboQuantKVPrecisionPolicy(
        key: .fp16OrQ8,
        value: .turbo4v2,
        boundary: .profileDefault
    )

    public static func legacy(
        preset: TurboQuantPreset,
        valueBits: Int?,
        boundary: TurboQuantBoundaryPolicy = .disabled
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
            boundary: boundary
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

    public var usesCompressedKeyPolicy: Bool {
        key != .fp16
    }

    public var usesLowPrecisionValue: Bool {
        value != .fp16
    }

    public var requiresRawBoundaryProtection: Bool {
        usesCompressedKeyPolicy || usesLowPrecisionValue
    }

    public func resolvedBoundaryPolicy(layerCount: Int) -> TurboQuantBoundaryPolicy {
        boundary.resolved(
            layerCount: layerCount,
            requiresProtection: requiresRawBoundaryProtection
        )
    }

    public func protectedBoundaryLayerIndexes(layerCount: Int) -> Set<Int> {
        boundary.protectedLayerIndexes(
            layerCount: layerCount,
            requiresProtection: requiresRawBoundaryProtection
        )
    }

    public func protectsBoundaryLayer(layerIndex: Int, layerCount: Int) -> Bool {
        boundary.protects(
            layerIndex: layerIndex,
            layerCount: layerCount,
            requiresProtection: requiresRawBoundaryProtection
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
