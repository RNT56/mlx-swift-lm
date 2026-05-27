// Copyright © 2026 RNT56.

import Foundation

public struct TurboQuantCacheRuntimeSnapshot: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 1

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
    public var hybridDiagnostics: TurboQuantHybridDiagnostics?

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
        hybridDiagnostics: TurboQuantHybridDiagnostics? = nil
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
        self.hybridDiagnostics = hybridDiagnostics
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
