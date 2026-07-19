// Copyright © 2026 RNT56.

import Foundation

public enum KVLayerCodec: Hashable, Codable, Sendable {
    case inherit
    case rawFP16
    case mlxAffine(bits: Int, groupSize: Int)
    case affineK8V4
    case affineK8Vx(valueBits: Int)
    case affineK8VxResidual(valueBits: Int, residualsPerGroup: Int)
    case affineInt4
    case turboQuant(
        preset: TurboQuantPreset,
        valueBits: Int?,
        groupSize: Int,
        backend: TurboQuantBackend
    )

    private enum CodingKeys: String, CodingKey {
        case kind
        case bits
        case groupSize
        case preset
        case valueBits
        case residualsPerGroup
        case backend
    }

    private enum Kind: String, Codable {
        case inherit
        case rawFP16
        case mlxAffine
        case affineK8V4
        case affineK8Vx
        case affineK8VxResidual
        case affineInt4
        case turboQuant
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .inherit:
            self = .inherit
        case .rawFP16:
            self = .rawFP16
        case .mlxAffine:
            self = .mlxAffine(
                bits: try container.decode(Int.self, forKey: .bits),
                groupSize: try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
            )
        case .affineK8V4:
            self = .affineK8V4
        case .affineK8Vx:
            self = .affineK8Vx(
                valueBits: try container.decodeIfPresent(Int.self, forKey: .valueBits)
                    ?? TurboQuantKVCodec.affineK8V4ValueBits
            )
        case .affineK8VxResidual:
            self = .affineK8VxResidual(
                valueBits: try container.decodeIfPresent(Int.self, forKey: .valueBits) ?? 2,
                residualsPerGroup: try container.decodeIfPresent(
                    Int.self,
                    forKey: .residualsPerGroup
                ) ?? 1
            )
        case .affineInt4:
            self = .affineInt4
        case .turboQuant:
            self = .turboQuant(
                preset: try container.decode(TurboQuantPreset.self, forKey: .preset),
                valueBits: try container.decodeIfPresent(Int.self, forKey: .valueBits),
                groupSize: try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64,
                backend: try container.decodeIfPresent(TurboQuantBackend.self, forKey: .backend)
                    ?? .metalPolarQJL
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .inherit:
            try container.encode(Kind.inherit, forKey: .kind)
        case .rawFP16:
            try container.encode(Kind.rawFP16, forKey: .kind)
        case .mlxAffine(let bits, let groupSize):
            try container.encode(Kind.mlxAffine, forKey: .kind)
            try container.encode(bits, forKey: .bits)
            try container.encode(groupSize, forKey: .groupSize)
        case .affineK8V4:
            try container.encode(Kind.affineK8V4, forKey: .kind)
        case .affineK8Vx(let valueBits):
            try container.encode(Kind.affineK8Vx, forKey: .kind)
            try container.encode(valueBits, forKey: .valueBits)
        case .affineK8VxResidual(let valueBits, let residualsPerGroup):
            try container.encode(Kind.affineK8VxResidual, forKey: .kind)
            try container.encode(valueBits, forKey: .valueBits)
            try container.encode(residualsPerGroup, forKey: .residualsPerGroup)
        case .affineInt4:
            try container.encode(Kind.affineInt4, forKey: .kind)
        case .turboQuant(let preset, let valueBits, let groupSize, let backend):
            try container.encode(Kind.turboQuant, forKey: .kind)
            try container.encode(preset, forKey: .preset)
            try container.encodeIfPresent(valueBits, forKey: .valueBits)
            try container.encode(groupSize, forKey: .groupSize)
            try container.encode(backend, forKey: .backend)
        }
    }

    public var requiresThrowingAttention: Bool {
        if case .turboQuant = self { return true }
        return false
    }

    public var summary: String {
        switch self {
        case .inherit:
            "inherit"
        case .rawFP16:
            "rawFP16"
        case .mlxAffine(let bits, let groupSize):
            "mlxAffine\(bits)g\(groupSize)"
        case .affineK8V4:
            "affineK8V4"
        case .affineK8Vx(let valueBits):
            "affineK8V\(valueBits)"
        case .affineK8VxResidual(let valueBits, let residualsPerGroup):
            "affineK8V\(valueBits)ResidualR\(residualsPerGroup)"
        case .affineInt4:
            "affineInt4"
        case .turboQuant(let preset, let valueBits, let groupSize, let backend):
            "turboQuant(\(preset.rawValue),v\(valueBits.map(String.init) ?? "default"),g\(groupSize),\(backend.rawValue))"
        }
    }

    fileprivate var canonicalString: String {
        switch self {
        case .inherit:
            "inherit"
        case .rawFP16:
            "rawFP16"
        case .mlxAffine(let bits, let groupSize):
            "mlxAffine(bits:\(bits),groupSize:\(groupSize))"
        case .affineK8V4:
            "affineK8V4"
        case .affineK8Vx(let valueBits):
            "affineK8Vx(valueBits:\(valueBits))"
        case .affineK8VxResidual(let valueBits, let residualsPerGroup):
            "affineK8VxResidual(valueBits:\(valueBits),residualsPerGroup:\(residualsPerGroup))"
        case .affineInt4:
            "affineInt4"
        case .turboQuant(let preset, let valueBits, let groupSize, let backend):
            "turboQuant(preset:\(preset.rawValue),valueBits:\(valueBits.map(String.init) ?? "nil"),groupSize:\(groupSize),backend:\(backend.rawValue))"
        }
    }
}

public struct KVLayerRule: Hashable, Codable, Sendable {
    public var layerIndex: Int
    public var codec: KVLayerCodec

    public init(layerIndex: Int, codec: KVLayerCodec) {
        self.layerIndex = layerIndex
        self.codec = codec
    }
}

public struct KVLayerPolicy: Hashable, Codable, Sendable {
    public var defaultCodec: KVLayerCodec?
    public var rules: [KVLayerRule]

    public init(
        defaultCodec: KVLayerCodec? = nil,
        rules: [KVLayerRule] = []
    ) {
        self.defaultCodec = defaultCodec
        self.rules = rules
    }

    public func codec(forLayerIndex layerIndex: Int) -> KVLayerCodec {
        rules.last { $0.layerIndex == layerIndex }?.codec ?? defaultCodec ?? .inherit
    }

    public var requiresThrowingAttention: Bool {
        if defaultCodec?.requiresThrowingAttention == true {
            return true
        }
        return rules.contains { $0.codec.requiresThrowingAttention }
    }

    public func validationErrors(layerCount: Int? = nil) -> [String] {
        var errors: [String] = []
        var seen = Set<Int>()
        func validateCodec(_ codec: KVLayerCodec, context: String) {
            if case .affineK8Vx(let valueBits) = codec,
                !TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits)
            {
                errors.append("\(context) affine K8/Vx value bits must be 2, 3, or 4")
            }
            if case .affineK8VxResidual(let valueBits, let residualsPerGroup) = codec {
                if valueBits != 2 {
                    errors.append("\(context) residual affine K8/Vx currently supports only V2")
                }
                if residualsPerGroup != 1 {
                    errors.append(
                        "\(context) residual affine K8/Vx currently supports residualsPerGroup == 1"
                    )
                }
            }
        }
        if let defaultCodec {
            validateCodec(defaultCodec, context: "default codec")
        }
        for rule in rules {
            if rule.layerIndex < 0 {
                errors.append("layer \(rule.layerIndex) is negative")
            }
            if let layerCount, rule.layerIndex >= layerCount {
                errors.append("layer \(rule.layerIndex) is out of range for \(layerCount) layers")
            }
            if !seen.insert(rule.layerIndex).inserted {
                errors.append("layer \(rule.layerIndex) has duplicate KV policy rules")
            }
            validateCodec(rule.codec, context: "layer \(rule.layerIndex)")
        }
        return errors
    }

    public func validate(layerCount: Int? = nil) throws {
        let errors = validationErrors(layerCount: layerCount)
        guard errors.isEmpty else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "Invalid KV layer policy: \(errors.joined(separator: "; "))"
                )
            )
        }
    }

    public var stableHash: String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in canonicalString.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x0000_0100_0000_01b3
        }
        return String(format: "%016llx", hash)
    }

    public func summary(maxRules: Int = 8) -> String {
        let defaultSummary = defaultCodec?.summary ?? "inherit"
        let sortedRules = rules.sorted { lhs, rhs in
            lhs.layerIndex == rhs.layerIndex
                ? lhs.codec.canonicalString < rhs.codec.canonicalString
                : lhs.layerIndex < rhs.layerIndex
        }
        let visible = sortedRules.prefix(max(0, maxRules)).map {
            "\($0.layerIndex):\($0.codec.summary)"
        }
        let suffix = sortedRules.count > visible.count ? ",+\(sortedRules.count - visible.count)" : ""
        return "default=\(defaultSummary);layers=[\(visible.joined(separator: ","))\(suffix)]"
    }

    private var canonicalString: String {
        var parts = ["default:\(defaultCodec?.canonicalString ?? "nil")"]
        for rule in rules.sorted(by: { lhs, rhs in
            lhs.layerIndex == rhs.layerIndex
                ? lhs.codec.canonicalString < rhs.codec.canonicalString
                : lhs.layerIndex < rhs.layerIndex
        }) {
            parts.append("\(rule.layerIndex):\(rule.codec.canonicalString)")
        }
        return parts.joined(separator: "|")
    }

    public static func affineK8VxProtectedEdges(
        layerCount: Int,
        valueBits: Int,
        boundaryCachePrecision: TurboQuantBoundaryCachePrecision = .affineK8V4,
        first: Int = 2,
        last: Int = 2
    ) -> KVLayerPolicy {
        affineK8VxProtectedLayers(
            layerCount: layerCount,
            valueBits: valueBits,
            boundaryPolicy: .protectedEdges(first: first, last: last),
            boundaryCachePrecision: boundaryCachePrecision
        )
    }

    public static func affineK8VxProtectedLayers(
        layerCount: Int,
        valueBits: Int,
        boundaryPolicy: TurboQuantBoundaryPolicy,
        boundaryCachePrecision: TurboQuantBoundaryCachePrecision = .affineK8V4
    ) -> KVLayerPolicy {
        let layerCount = max(0, layerCount)
        let resolvedValueBits = TurboQuantKVCodec.affineK8VxSupportedValueBits.contains(valueBits)
            ? valueBits : TurboQuantKVCodec.affineK8V4ValueBits
        let boundaryCodec: KVLayerCodec =
            switch boundaryCachePrecision {
            case .affineK8V4:
                .affineK8V4
            case .raw:
                .rawFP16
            }
        let boundaryLayers = boundaryPolicy.protectedLayerIndexes(layerCount: layerCount)
        let rules = boundaryLayers.sorted().map {
            KVLayerRule(layerIndex: $0, codec: boundaryCodec)
        }
        return KVLayerPolicy(
            defaultCodec: .affineK8Vx(valueBits: resolvedValueBits),
            rules: rules
        )
    }

    public static func optiQKVConfig(
        data: Data,
        defaultCodec: KVLayerCodec? = nil
    ) throws -> KVLayerPolicy {
        let entries = try JSONDecoder().decode([OptIQKVConfigEntry].self, from: data)
        let rules = try entries.map { entry in
            KVLayerRule(layerIndex: entry.layerIndex, codec: try entry.codec())
        }
        let policy = KVLayerPolicy(defaultCodec: defaultCodec, rules: rules)
        try policy.validate()
        return policy
    }

    public static func loadOptIQKVConfig(
        at url: URL,
        defaultCodec: KVLayerCodec? = nil
    ) throws -> KVLayerPolicy {
        try optiQKVConfig(data: Data(contentsOf: url), defaultCodec: defaultCodec)
    }

    private struct OptIQKVConfigEntry: Decodable {
        var layerIndex: Int
        var bits: Int
        var groupSize: Int

        private enum CodingKeys: String, CodingKey {
            case layerIndex = "layer_idx"
            case bits
            case groupSize = "group_size"
        }

        func codec() throws -> KVLayerCodec {
            switch bits {
            case 8:
                .affineK8V4
            case 4:
                .turboQuant(
                    preset: .turbo4v2,
                    valueBits: 4,
                    groupSize: groupSize,
                    backend: .metalPolarQJL
                )
            default:
                throw DecodingError.dataCorrupted(
                    DecodingError.Context(
                        codingPath: [],
                        debugDescription: "Unsupported OptIQ KV bit width \(bits) for layer \(layerIndex)"
                    )
                )
            }
        }
    }
}

public struct KVLayerTopology: Hashable, Sendable {
    public var kvHeads: [Int]

    public init(kvHeads: [Int]) {
        self.kvHeads = kvHeads
    }

    public var modelLayerCount: Int {
        kvHeads.count
    }

    public var attentionLayerCount: Int {
        kvHeads.filter { $0 > 0 }.count
    }

    public var admissionLayerCount: Int {
        attentionLayerCount > 0 ? attentionLayerCount : modelLayerCount
    }

    public func isAttentionLayer(_ layerIndex: Int) -> Bool {
        kvHeads.indices.contains(layerIndex) && kvHeads[layerIndex] > 0
    }
}
