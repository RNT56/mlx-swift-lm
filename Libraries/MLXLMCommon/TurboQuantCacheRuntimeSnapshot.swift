// Copyright © 2026 RNT56.

import Foundation

public struct TurboQuantCacheRuntimeSnapshot: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 2

    public var schemaVersion: Int
    public var lifecycleDescription: String
    public var logicalLength: Int
    public var capacity: Int
    public var pinnedPrefixLength: Int
    public var ringOffset: Int
    public var keyBytes: Int
    public var valueBytes: Int
    public var rawShadowAllocated: Bool
    public var packedFallbackAllocated: Bool
    public var lastAttentionPath: String?
    public var lastFailure: String?
    public var kvCodec: TurboQuantKVCodec
    public var quantizationMode: String?
    public var keyBits: Int?
    public var groupSize: Int?
    public var selectedPath: String?
    public var fallbackReason: String?
    public var hybridDiagnostics: TurboQuantHybridDiagnostics?
    public var requestedRuntimeMode: TurboQuantRuntimeMode?
    public var resolvedRuntimeMode: TurboQuantRuntimeMode?
    public var precisionPolicy: TurboQuantKVPrecisionPolicy?
    public var sparseValuePolicy: TurboQuantSparseValuePolicy?
    public var boundaryPolicy: TurboQuantBoundaryPolicy?
    public var boundaryProtectedLayerCount: Int
    public var boundaryProtectionReason: String?
    public var runtimeFallbackReason: String?
    public var decodedActiveKeyBytes: Int
    public var decodedActiveValueBytes: Int
    public var activeCacheAllocated: Bool

    public init(
        schemaVersion: Int = TurboQuantCacheRuntimeSnapshot.currentSchemaVersion,
        lifecycleDescription: String,
        logicalLength: Int,
        capacity: Int,
        pinnedPrefixLength: Int,
        ringOffset: Int,
        keyBytes: Int,
        valueBytes: Int,
        rawShadowAllocated: Bool,
        packedFallbackAllocated: Bool,
        lastAttentionPath: String?,
        lastFailure: String?,
        kvCodec: TurboQuantKVCodec = .polarQJL,
        quantizationMode: String? = nil,
        keyBits: Int? = nil,
        groupSize: Int? = nil,
        selectedPath: String? = nil,
        fallbackReason: String? = nil,
        hybridDiagnostics: TurboQuantHybridDiagnostics? = nil,
        requestedRuntimeMode: TurboQuantRuntimeMode? = nil,
        resolvedRuntimeMode: TurboQuantRuntimeMode? = nil,
        precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
        sparseValuePolicy: TurboQuantSparseValuePolicy? = nil,
        boundaryPolicy: TurboQuantBoundaryPolicy? = nil,
        boundaryProtectedLayerCount: Int = 0,
        boundaryProtectionReason: String? = nil,
        runtimeFallbackReason: String? = nil,
        decodedActiveKeyBytes: Int = 0,
        decodedActiveValueBytes: Int = 0,
        activeCacheAllocated: Bool = false
    ) {
        self.schemaVersion = schemaVersion
        self.lifecycleDescription = lifecycleDescription
        self.logicalLength = logicalLength
        self.capacity = capacity
        self.pinnedPrefixLength = pinnedPrefixLength
        self.ringOffset = ringOffset
        self.keyBytes = keyBytes
        self.valueBytes = valueBytes
        self.rawShadowAllocated = rawShadowAllocated
        self.packedFallbackAllocated = packedFallbackAllocated
        self.lastAttentionPath = lastAttentionPath
        self.lastFailure = lastFailure
        self.kvCodec = kvCodec
        self.quantizationMode = quantizationMode
        self.keyBits = keyBits
        self.groupSize = groupSize
        self.selectedPath = selectedPath ?? lastAttentionPath
        self.fallbackReason = fallbackReason ?? lastFailure
        self.hybridDiagnostics = hybridDiagnostics
        self.requestedRuntimeMode = requestedRuntimeMode
        self.resolvedRuntimeMode = resolvedRuntimeMode
        self.precisionPolicy = precisionPolicy
        self.sparseValuePolicy = sparseValuePolicy
        self.boundaryPolicy = boundaryPolicy
        self.boundaryProtectedLayerCount = max(0, boundaryProtectedLayerCount)
        self.boundaryProtectionReason = boundaryProtectionReason
        self.runtimeFallbackReason = runtimeFallbackReason
        self.decodedActiveKeyBytes = max(0, decodedActiveKeyBytes)
        self.decodedActiveValueBytes = max(0, decodedActiveValueBytes)
        self.activeCacheAllocated = activeCacheAllocated
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case lifecycleDescription
        case logicalLength
        case capacity
        case pinnedPrefixLength
        case ringOffset
        case keyBytes
        case valueBytes
        case rawShadowAllocated
        case packedFallbackAllocated
        case lastAttentionPath
        case lastFailure
        case kvCodec
        case quantizationMode
        case keyBits
        case groupSize
        case selectedPath
        case fallbackReason
        case hybridDiagnostics
        case requestedRuntimeMode
        case resolvedRuntimeMode
        case precisionPolicy
        case sparseValuePolicy
        case boundaryPolicy
        case boundaryProtectedLayerCount
        case boundaryProtectionReason
        case runtimeFallbackReason
        case decodedActiveKeyBytes
        case decodedActiveValueBytes
        case activeCacheAllocated
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            schemaVersion: try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1,
            lifecycleDescription: try container.decode(
                String.self,
                forKey: .lifecycleDescription
            ),
            logicalLength: try container.decode(Int.self, forKey: .logicalLength),
            capacity: try container.decode(Int.self, forKey: .capacity),
            pinnedPrefixLength: try container.decode(Int.self, forKey: .pinnedPrefixLength),
            ringOffset: try container.decode(Int.self, forKey: .ringOffset),
            keyBytes: try container.decode(Int.self, forKey: .keyBytes),
            valueBytes: try container.decode(Int.self, forKey: .valueBytes),
            rawShadowAllocated: try container.decode(Bool.self, forKey: .rawShadowAllocated),
            packedFallbackAllocated: try container.decode(
                Bool.self,
                forKey: .packedFallbackAllocated
            ),
            lastAttentionPath: try container.decodeIfPresent(
                String.self,
                forKey: .lastAttentionPath
            ),
            lastFailure: try container.decodeIfPresent(String.self, forKey: .lastFailure),
            kvCodec: try container.decodeIfPresent(TurboQuantKVCodec.self, forKey: .kvCodec)
                ?? .polarQJL,
            quantizationMode: try container.decodeIfPresent(
                String.self,
                forKey: .quantizationMode
            ),
            keyBits: try container.decodeIfPresent(Int.self, forKey: .keyBits),
            groupSize: try container.decodeIfPresent(Int.self, forKey: .groupSize),
            selectedPath: try container.decodeIfPresent(String.self, forKey: .selectedPath),
            fallbackReason: try container.decodeIfPresent(String.self, forKey: .fallbackReason),
            hybridDiagnostics: try container.decodeIfPresent(
                TurboQuantHybridDiagnostics.self,
                forKey: .hybridDiagnostics
            ),
            requestedRuntimeMode: try container.decodeIfPresent(
                TurboQuantRuntimeMode.self,
                forKey: .requestedRuntimeMode
            ),
            resolvedRuntimeMode: try container.decodeIfPresent(
                TurboQuantRuntimeMode.self,
                forKey: .resolvedRuntimeMode
            ),
            precisionPolicy: try container.decodeIfPresent(
                TurboQuantKVPrecisionPolicy.self,
                forKey: .precisionPolicy
            ),
            sparseValuePolicy: try container.decodeIfPresent(
                TurboQuantSparseValuePolicy.self,
                forKey: .sparseValuePolicy
            ),
            boundaryPolicy: try container.decodeIfPresent(
                TurboQuantBoundaryPolicy.self,
                forKey: .boundaryPolicy
            ),
            boundaryProtectedLayerCount: try container.decodeIfPresent(
                Int.self,
                forKey: .boundaryProtectedLayerCount
            ) ?? 0,
            boundaryProtectionReason: try container.decodeIfPresent(
                String.self,
                forKey: .boundaryProtectionReason
            ),
            runtimeFallbackReason: try container.decodeIfPresent(
                String.self,
                forKey: .runtimeFallbackReason
            ),
            decodedActiveKeyBytes: try container.decodeIfPresent(
                Int.self,
                forKey: .decodedActiveKeyBytes
            ) ?? 0,
            decodedActiveValueBytes: try container.decodeIfPresent(
                Int.self,
                forKey: .decodedActiveValueBytes
            ) ?? 0,
            activeCacheAllocated: try container.decodeIfPresent(
                Bool.self,
                forKey: .activeCacheAllocated
            ) ?? false
        )
    }
}

extension TurboQuantCacheLifecycle {
    var turboQuantRuntimeDescription: String {
        switch self {
        case .empty:
            "empty"
        case .rawPrefillChunkOpen:
            "rawPrefillChunkOpen"
        case .compressingChunk(let start, let count):
            "compressingChunk(start:\(start),count:\(count))"
        case .compressedCommitted(let logicalLength, let capacity):
            "compressedCommitted(logicalLength:\(logicalLength),capacity:\(capacity))"
        case .decodeCompressed:
            "decodeCompressed"
        case .degradedPackedFallback(let reason):
            "degradedPackedFallback(reason:\(reason))"
        case .degradedDecodedFallback(let reason):
            "degradedDecodedFallback(reason:\(reason))"
        case .failed(let reason):
            "failed(reason:\(reason))"
        }
    }

    var turboQuantRuntimeFailureReason: String? {
        switch self {
        case .degradedPackedFallback(let reason),
            .degradedDecodedFallback(let reason),
            .failed(let reason):
            reason
        case .empty,
            .rawPrefillChunkOpen,
            .compressingChunk,
            .compressedCommitted,
            .decodeCompressed:
            nil
        }
    }
}
