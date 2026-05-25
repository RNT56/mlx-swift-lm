// Copyright © 2026 RNT56.

import Foundation
import MLX

public enum TurboQuantSpeculativeCacheRole: String, Codable, Sendable, Equatable {
    case target
    case draft
}

public struct TurboQuantSpeculativeCacheCheckpointMetadata: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var cacheIndex: Int
    public var cacheKind: String
    public var role: TurboQuantSpeculativeCacheRole
    public var logicalLength: Int
    public var capacity: Int
    public var pinnedPrefixLength: Int
    public var ringOffset: Int
    public var keyBytes: Int
    public var valueBytes: Int
    public var lifecycleDescription: String

    public init(
        schemaVersion: Int = TurboQuantSpeculativeCacheCheckpointMetadata.currentSchemaVersion,
        cacheIndex: Int,
        cacheKind: String,
        role: TurboQuantSpeculativeCacheRole,
        logicalLength: Int,
        capacity: Int,
        pinnedPrefixLength: Int,
        ringOffset: Int,
        keyBytes: Int,
        valueBytes: Int,
        lifecycleDescription: String
    ) {
        self.schemaVersion = schemaVersion
        self.cacheIndex = cacheIndex
        self.cacheKind = cacheKind
        self.role = role
        self.logicalLength = logicalLength
        self.capacity = capacity
        self.pinnedPrefixLength = pinnedPrefixLength
        self.ringOffset = ringOffset
        self.keyBytes = keyBytes
        self.valueBytes = valueBytes
        self.lifecycleDescription = lifecycleDescription
    }
}

private struct TurboQuantSpeculativeCacheCheckpointEntry {
    var metadata: TurboQuantSpeculativeCacheCheckpointMetadata
    var offset: Int
    var state: [MLXArray]
    var metaState: [String]
}

public struct TurboQuantSpeculativeCacheCheckpointSet {
    fileprivate var entries: [TurboQuantSpeculativeCacheCheckpointEntry]
    public var role: TurboQuantSpeculativeCacheRole

    fileprivate init(
        role: TurboQuantSpeculativeCacheRole,
        entries: [TurboQuantSpeculativeCacheCheckpointEntry]
    ) {
        self.role = role
        self.entries = entries
    }

    public var metadata: [TurboQuantSpeculativeCacheCheckpointMetadata] {
        entries.map(\.metadata)
    }

    public var turboQuantCacheCount: Int {
        entries.count
    }

    public var isEmpty: Bool {
        entries.isEmpty
    }
}

public struct TurboQuantSpeculativeCacheTrimResult: Hashable, Codable, Sendable {
    public var role: TurboQuantSpeculativeCacheRole
    public var requestedTrimTokens: Int
    public var actualTrimTokens: Int
    public var expectedLogicalLengthDelta: Int
    public var turboQuantCacheCount: Int
    public var resultingLogicalLengths: [Int]
    public var resultingRingOffsets: [Int]
    public var resultingPinnedPrefixLengths: [Int]

    public init(
        role: TurboQuantSpeculativeCacheRole,
        requestedTrimTokens: Int,
        actualTrimTokens: Int,
        expectedLogicalLengthDelta: Int,
        turboQuantCacheCount: Int,
        resultingLogicalLengths: [Int],
        resultingRingOffsets: [Int],
        resultingPinnedPrefixLengths: [Int]
    ) {
        self.role = role
        self.requestedTrimTokens = requestedTrimTokens
        self.actualTrimTokens = actualTrimTokens
        self.expectedLogicalLengthDelta = expectedLogicalLengthDelta
        self.turboQuantCacheCount = turboQuantCacheCount
        self.resultingLogicalLengths = resultingLogicalLengths
        self.resultingRingOffsets = resultingRingOffsets
        self.resultingPinnedPrefixLengths = resultingPinnedPrefixLengths
    }
}

public struct TurboQuantSpeculativeVerificationResult: Hashable, Codable, Sendable {
    public var draftedTokens: [Int]
    public var targetTokens: [Int]
    public var acceptedDraftTokens: Int
    public var correctionToken: Int

    public init(
        draftedTokens: [Int],
        targetTokens: [Int],
        acceptedDraftTokens: Int,
        correctionToken: Int
    ) {
        self.draftedTokens = draftedTokens
        self.targetTokens = targetTokens
        self.acceptedDraftTokens = acceptedDraftTokens
        self.correctionToken = correctionToken
    }

    public var rejectedDraftTokens: Int {
        max(0, draftedTokens.count - acceptedDraftTokens)
    }

    public var emittedTokens: [Int] {
        Array(draftedTokens.prefix(acceptedDraftTokens)) + [correctionToken]
    }
}

public struct TurboQuantSpeculativeAcceptanceMetrics: Hashable, Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var rounds: Int
    public var fullyAcceptedRounds: Int
    public var rejectedRounds: Int
    public var firstTokenRejectedRounds: Int
    public var acceptedDraftTokens: Int
    public var rejectedDraftTokens: Int
    public var proposedDraftTokens: Int
    public var emittedTokens: Int
    public var targetCacheRollbackCount: Int
    public var draftCacheRollbackCount: Int
    public var targetTurboQuantCacheCount: Int
    public var draftTurboQuantCacheCount: Int

    public init(
        schemaVersion: Int = TurboQuantSpeculativeAcceptanceMetrics.currentSchemaVersion,
        rounds: Int = 0,
        fullyAcceptedRounds: Int = 0,
        rejectedRounds: Int = 0,
        firstTokenRejectedRounds: Int = 0,
        acceptedDraftTokens: Int = 0,
        rejectedDraftTokens: Int = 0,
        proposedDraftTokens: Int = 0,
        emittedTokens: Int = 0,
        targetCacheRollbackCount: Int = 0,
        draftCacheRollbackCount: Int = 0,
        targetTurboQuantCacheCount: Int = 0,
        draftTurboQuantCacheCount: Int = 0
    ) {
        self.schemaVersion = schemaVersion
        self.rounds = rounds
        self.fullyAcceptedRounds = fullyAcceptedRounds
        self.rejectedRounds = rejectedRounds
        self.firstTokenRejectedRounds = firstTokenRejectedRounds
        self.acceptedDraftTokens = acceptedDraftTokens
        self.rejectedDraftTokens = rejectedDraftTokens
        self.proposedDraftTokens = proposedDraftTokens
        self.emittedTokens = emittedTokens
        self.targetCacheRollbackCount = targetCacheRollbackCount
        self.draftCacheRollbackCount = draftCacheRollbackCount
        self.targetTurboQuantCacheCount = targetTurboQuantCacheCount
        self.draftTurboQuantCacheCount = draftTurboQuantCacheCount
    }

    public var acceptanceRate: Double {
        guard proposedDraftTokens > 0 else { return 0 }
        return Double(acceptedDraftTokens) / Double(proposedDraftTokens)
    }

    public var rejectionRate: Double {
        guard proposedDraftTokens > 0 else { return 0 }
        return Double(rejectedDraftTokens) / Double(proposedDraftTokens)
    }

    public func shouldDisableSpeculation(
        minObservedDraftTokens: Int = 32,
        minimumAcceptanceRate: Double = 0.25
    ) -> Bool {
        proposedDraftTokens >= minObservedDraftTokens && acceptanceRate < minimumAcceptanceRate
    }

    public mutating func recordRound(
        draftedTokens: Int,
        acceptedDraftTokens: Int,
        emittedTokens: Int,
        targetRollback: TurboQuantSpeculativeCacheTrimResult,
        draftRollback: TurboQuantSpeculativeCacheTrimResult
    ) {
        let rejected = max(0, draftedTokens - acceptedDraftTokens)
        rounds += 1
        proposedDraftTokens += draftedTokens
        self.acceptedDraftTokens += acceptedDraftTokens
        rejectedDraftTokens += rejected
        self.emittedTokens += emittedTokens
        if rejected == 0 {
            fullyAcceptedRounds += 1
        } else {
            rejectedRounds += 1
        }
        if draftedTokens > 0 && acceptedDraftTokens == 0 {
            firstTokenRejectedRounds += 1
        }
        if targetRollback.requestedTrimTokens > 0 {
            targetCacheRollbackCount += 1
        }
        if draftRollback.requestedTrimTokens > 0 {
            draftCacheRollbackCount += 1
        }
        targetTurboQuantCacheCount = max(
            targetTurboQuantCacheCount,
            targetRollback.turboQuantCacheCount
        )
        draftTurboQuantCacheCount = max(
            draftTurboQuantCacheCount,
            draftRollback.turboQuantCacheCount
        )
    }
}

public enum TurboQuantSpeculativeTokenizerCompatibilityStatus: String, Codable, Sendable,
    Equatable
{
    case compatible
    case incompatible
    case unknown
}

public struct TurboQuantSpeculativeTokenizerFingerprint: Hashable, Codable, Sendable {
    public var typeName: String
    public var bosToken: String?
    public var eosToken: String?
    public var unknownToken: String?

    public init(
        typeName: String,
        bosToken: String?,
        eosToken: String?,
        unknownToken: String?
    ) {
        self.typeName = typeName
        self.bosToken = bosToken
        self.eosToken = eosToken
        self.unknownToken = unknownToken
    }

    public init(tokenizer: any Tokenizer) {
        self.init(
            typeName: String(reflecting: type(of: tokenizer)),
            bosToken: tokenizer.bosToken,
            eosToken: tokenizer.eosToken,
            unknownToken: tokenizer.unknownToken
        )
    }
}

public struct TurboQuantSpeculativeTokenizerCompatibility: Hashable, Codable, Sendable {
    public var status: TurboQuantSpeculativeTokenizerCompatibilityStatus
    public var reason: String
    public var target: TurboQuantSpeculativeTokenizerFingerprint
    public var draft: TurboQuantSpeculativeTokenizerFingerprint

    public init(
        status: TurboQuantSpeculativeTokenizerCompatibilityStatus,
        reason: String,
        target: TurboQuantSpeculativeTokenizerFingerprint,
        draft: TurboQuantSpeculativeTokenizerFingerprint
    ) {
        self.status = status
        self.reason = reason
        self.target = target
        self.draft = draft
    }

    public static func evaluate(
        target targetTokenizer: any Tokenizer,
        draft draftTokenizer: any Tokenizer
    ) -> TurboQuantSpeculativeTokenizerCompatibility {
        let target = TurboQuantSpeculativeTokenizerFingerprint(tokenizer: targetTokenizer)
        let draft = TurboQuantSpeculativeTokenizerFingerprint(tokenizer: draftTokenizer)

        guard target.typeName == draft.typeName else {
            return TurboQuantSpeculativeTokenizerCompatibility(
                status: .incompatible,
                reason: "tokenizer concrete types differ",
                target: target,
                draft: draft
            )
        }
        guard target.bosToken == draft.bosToken,
            target.eosToken == draft.eosToken,
            target.unknownToken == draft.unknownToken
        else {
            return TurboQuantSpeculativeTokenizerCompatibility(
                status: .incompatible,
                reason: "special token surfaces differ",
                target: target,
                draft: draft
            )
        }
        return TurboQuantSpeculativeTokenizerCompatibility(
            status: .compatible,
            reason: "tokenizer type and special token surfaces match",
            target: target,
            draft: draft
        )
    }
}

public enum TurboQuantSpeculativeTargetVerifier {
    public static func verifyGreedy(
        targetTokens: [Int],
        draftedTokens: [Int]
    ) throws -> TurboQuantSpeculativeVerificationResult {
        guard targetTokens.count >= draftedTokens.count + 1 else {
            throw TurboQuantRuntimeFailure.cacheLifecycleInvalid(
                "speculative target verifier requires one target token for each draft plus a correction token"
            )
        }

        var accepted = 0
        for index in draftedTokens.indices {
            guard targetTokens[index] == draftedTokens[index] else {
                break
            }
            accepted += 1
        }

        return TurboQuantSpeculativeVerificationResult(
            draftedTokens: draftedTokens,
            targetTokens: targetTokens,
            acceptedDraftTokens: accepted,
            correctionToken: targetTokens[accepted]
        )
    }
}

public enum TurboQuantSpeculativeVerifier {
    public static func checkpoint(
        _ caches: [KVCache],
        role: TurboQuantSpeculativeCacheRole
    ) throws -> TurboQuantSpeculativeCacheCheckpointSet {
        var entries = [TurboQuantSpeculativeCacheCheckpointEntry]()
        for (index, cache) in caches.enumerated() {
            guard let turboQuantCache = cache as? any TurboQuantCompressedKVCacheProtocol else {
                continue
            }
            try turboQuantCache.validateCompressedState(
                context: "\(role.rawValue) speculative checkpoint"
            )
            let snapshot = turboQuantCache.runtimeSnapshot()
            let metadata = TurboQuantSpeculativeCacheCheckpointMetadata(
                cacheIndex: index,
                cacheKind: String(describing: type(of: cache)),
                role: role,
                logicalLength: snapshot.logicalLength,
                capacity: snapshot.capacity,
                pinnedPrefixLength: snapshot.pinnedPrefixLength,
                ringOffset: snapshot.ringOffset,
                keyBytes: snapshot.keyBytes,
                valueBytes: snapshot.valueBytes,
                lifecycleDescription: snapshot.lifecycleDescription
            )
            entries.append(
                TurboQuantSpeculativeCacheCheckpointEntry(
                    metadata: metadata,
                    offset: turboQuantCache.offset,
                    state: turboQuantCache.state.map { $0[.ellipsis] },
                    metaState: turboQuantCache.metaState
                )
            )
        }
        return TurboQuantSpeculativeCacheCheckpointSet(role: role, entries: entries)
    }

    public static func checkpointTargetCache(
        _ caches: [KVCache]
    ) throws -> TurboQuantSpeculativeCacheCheckpointSet {
        try checkpoint(caches, role: .target)
    }

    public static func checkpointDraftCache(
        _ caches: [KVCache]
    ) throws -> TurboQuantSpeculativeCacheCheckpointSet {
        try checkpoint(caches, role: .draft)
    }

    @discardableResult
    public static func restore(
        _ caches: [KVCache],
        to checkpoint: TurboQuantSpeculativeCacheCheckpointSet
    ) throws -> Int {
        var restored = 0
        for entry in checkpoint.entries {
            guard caches.indices.contains(entry.metadata.cacheIndex) else {
                throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                    "speculative \(checkpoint.role.rawValue) cache checkpoint index \(entry.metadata.cacheIndex) is outside cache count \(caches.count)"
                )
            }
            let cache = caches[entry.metadata.cacheIndex]
            guard String(describing: type(of: cache)) == entry.metadata.cacheKind else {
                throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                    "speculative \(checkpoint.role.rawValue) cache kind changed from \(entry.metadata.cacheKind) to \(String(describing: type(of: cache)))"
                )
            }
            guard var turboQuantCache = cache as? any TurboQuantCompressedKVCacheProtocol else {
                throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                    "speculative \(checkpoint.role.rawValue) cache at index \(entry.metadata.cacheIndex) no longer supports TurboQuant rollback"
                )
            }

            turboQuantCache.metaState = entry.metaState
            turboQuantCache.state = entry.state.map { $0[.ellipsis] }
            if let baseCache = turboQuantCache as? BaseKVCache {
                baseCache.offset = entry.offset
            }
            turboQuantCache.metaState = entry.metaState
            try turboQuantCache.validateCompressedState(
                context: "\(checkpoint.role.rawValue) speculative checkpoint restore"
            )

            let snapshot = turboQuantCache.runtimeSnapshot()
            guard snapshot.logicalLength == entry.metadata.logicalLength,
                snapshot.ringOffset == entry.metadata.ringOffset,
                snapshot.pinnedPrefixLength == entry.metadata.pinnedPrefixLength
            else {
                throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                    "speculative \(checkpoint.role.rawValue) restore produced logicalLength=\(snapshot.logicalLength), ringOffset=\(snapshot.ringOffset), pinnedPrefix=\(snapshot.pinnedPrefixLength); expected logicalLength=\(entry.metadata.logicalLength), ringOffset=\(entry.metadata.ringOffset), pinnedPrefix=\(entry.metadata.pinnedPrefixLength)"
                )
            }
            restored += 1
        }
        return restored
    }

    @discardableResult
    public static func trimAfterVerification(
        _ caches: [KVCache],
        checkpoint: TurboQuantSpeculativeCacheCheckpointSet,
        trimTokenCount: Int,
        expectedLogicalLengthDelta: Int
    ) throws -> TurboQuantSpeculativeCacheTrimResult {
        let requestedTrim = max(0, trimTokenCount)
        let actualTrim =
            requestedTrim > 0
            ? caches.map { $0.trim(requestedTrim) }.max() ?? 0
            : 0
        let expectedDelta = max(0, expectedLogicalLengthDelta)

        var resultingLogicalLengths = [Int]()
        var resultingRingOffsets = [Int]()
        var resultingPinnedPrefixLengths = [Int]()

        for entry in checkpoint.entries {
            guard caches.indices.contains(entry.metadata.cacheIndex),
                let turboQuantCache =
                    caches[entry.metadata.cacheIndex] as? any TurboQuantCompressedKVCacheProtocol
            else {
                throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                    "speculative \(checkpoint.role.rawValue) cache missing at checkpoint index \(entry.metadata.cacheIndex)"
                )
            }
            try turboQuantCache.validateCompressedState(
                context: "\(checkpoint.role.rawValue) speculative rejected-token rollback"
            )
            let snapshot = turboQuantCache.runtimeSnapshot()
            let expectedLogicalLength = expectedLength(
                from: entry.metadata,
                delta: expectedDelta
            )
            guard snapshot.logicalLength == expectedLogicalLength else {
                throw TurboQuantRuntimeFailure.cacheLayoutInvalid(
                    "speculative \(checkpoint.role.rawValue) rollback expected logical length \(expectedLogicalLength), got \(snapshot.logicalLength)"
                )
            }
            resultingLogicalLengths.append(snapshot.logicalLength)
            resultingRingOffsets.append(snapshot.ringOffset)
            resultingPinnedPrefixLengths.append(snapshot.pinnedPrefixLength)
        }

        return TurboQuantSpeculativeCacheTrimResult(
            role: checkpoint.role,
            requestedTrimTokens: requestedTrim,
            actualTrimTokens: actualTrim,
            expectedLogicalLengthDelta: expectedDelta,
            turboQuantCacheCount: checkpoint.turboQuantCacheCount,
            resultingLogicalLengths: resultingLogicalLengths,
            resultingRingOffsets: resultingRingOffsets,
            resultingPinnedPrefixLengths: resultingPinnedPrefixLengths
        )
    }

    private static func expectedLength(
        from metadata: TurboQuantSpeculativeCacheCheckpointMetadata,
        delta: Int
    ) -> Int {
        let proposed = metadata.logicalLength + delta
        if metadata.cacheKind == "RotatingTurboQuantKVCache", metadata.capacity > 0 {
            return min(metadata.capacity, proposed)
        }
        return proposed
    }
}
