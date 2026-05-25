import Foundation
import MLXLMCommon
import Testing

@Suite("TurboQuant cache runtime snapshot")
struct TurboQuantCacheRuntimeSnapshotTests {
    @Test func emptyCacheSnapshotIsStableAndCodable() throws {
        let cache = TurboQuantKVCache()

        let snapshot = cache.runtimeSnapshot()
        let data = try JSONEncoder().encode(snapshot)
        let decoded = try JSONDecoder().decode(TurboQuantCacheRuntimeSnapshot.self, from: data)

        #expect(decoded == snapshot)
        #expect(snapshot.schemaVersion == 1)
        #expect(snapshot.lifecycleDescription == "empty")
        #expect(snapshot.logicalLength == 0)
        #expect(snapshot.capacity == 0)
        #expect(snapshot.keyBytes == 0)
        #expect(snapshot.valueBytes == 0)
        #expect(!snapshot.rawShadowAllocated)
        #expect(!snapshot.packedFallbackAllocated)
    }

    @Test func failureSnapshotRecordsReasonAndPath() {
        let cache = TurboQuantKVCache()

        cache.recordCompressedAttentionFailure("forced failure")

        let snapshot = cache.runtimeSnapshot()
        #expect(snapshot.lifecycleDescription == "failed(reason:forced failure)")
        #expect(snapshot.lastAttentionPath == TurboQuantAttentionPath.mlxPackedFallback.rawValue)
        #expect(snapshot.lastFailure == "forced failure")
    }

    @Test func fallbackSnapshotRecordsLifecycleReason() {
        let cache = TurboQuantKVCache()

        cache.recordFallback(
            TurboQuantFallbackResult(
                fromPath: .twoStageCompressed,
                toPath: .mlxPackedFallback,
                policy: .packedAllowed,
                reason: "qk unavailable",
                isSemanticallyExact: true
            )
        )

        let snapshot = cache.runtimeSnapshot()
        #expect(snapshot.lifecycleDescription == "degradedPackedFallback(reason:qk unavailable)")
        #expect(snapshot.lastAttentionPath == TurboQuantAttentionPath.mlxPackedFallback.rawValue)
        #expect(snapshot.lastFailure == "qk unavailable")
    }

    @Test func rotatingSnapshotReportsCapacityAndPinnedPrefix() {
        let cache = RotatingTurboQuantKVCache(maxSize: 32, keep: 4)

        let snapshot = cache.runtimeSnapshot()

        #expect(snapshot.capacity == 32)
        #expect(snapshot.logicalLength == 0)
        #expect(snapshot.pinnedPrefixLength == 0)
        #expect(snapshot.ringOffset == 0)
        #expect(!snapshot.rawShadowAllocated)
    }
}
