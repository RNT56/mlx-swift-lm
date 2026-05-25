import MLX
import MLXLMCommon
import Testing

@Suite("TurboQuant speculative verifier")
struct TurboQuantSpeculativeTests {
    @Test func targetVerifierRecordsPartialAcceptanceAndCorrection() throws {
        let verification = try TurboQuantSpeculativeTargetVerifier.verifyGreedy(
            targetTokens: [10, 44, 99],
            draftedTokens: [10, 11]
        )

        #expect(verification.acceptedDraftTokens == 1)
        #expect(verification.rejectedDraftTokens == 1)
        #expect(verification.correctionToken == 44)
        #expect(verification.emittedTokens == [10, 44])
    }

    @Test func poorAcceptanceMetricsDisableSpeculationAfterSampleWindow() {
        var metrics = TurboQuantSpeculativeAcceptanceMetrics()
        let targetRollback = TurboQuantSpeculativeCacheTrimResult(
            role: .target,
            requestedTrimTokens: 3,
            actualTrimTokens: 3,
            expectedLogicalLengthDelta: 1,
            turboQuantCacheCount: 2,
            resultingLogicalLengths: [5, 5],
            resultingRingOffsets: [0, 0],
            resultingPinnedPrefixLengths: [0, 0]
        )
        let draftRollback = TurboQuantSpeculativeCacheTrimResult(
            role: .draft,
            requestedTrimTokens: 2,
            actualTrimTokens: 2,
            expectedLogicalLengthDelta: 1,
            turboQuantCacheCount: 2,
            resultingLogicalLengths: [5, 5],
            resultingRingOffsets: [0, 0],
            resultingPinnedPrefixLengths: [0, 0]
        )

        metrics.recordRound(
            draftedTokens: 8,
            acceptedDraftTokens: 1,
            emittedTokens: 2,
            targetRollback: targetRollback,
            draftRollback: draftRollback
        )
        metrics.recordRound(
            draftedTokens: 8,
            acceptedDraftTokens: 1,
            emittedTokens: 2,
            targetRollback: targetRollback,
            draftRollback: draftRollback
        )

        #expect(metrics.acceptedDraftTokens == 2)
        #expect(metrics.rejectedDraftTokens == 14)
        #expect(metrics.acceptanceRate == 0.125)
        #expect(metrics.targetCacheRollbackCount == 2)
        #expect(metrics.draftCacheRollbackCount == 2)
        #expect(
            metrics.shouldDisableSpeculation(
                minObservedDraftTokens: 16,
                minimumAcceptanceRate: 0.25
            ))
    }

    @Test func nonRotatingCompressedCacheTrimKeepsRollbackLayoutConsistent() throws {
        guard TurboQuantKernelAvailability.current.supportsMetalPolarQJLAttention else {
            return
        }

        let cache = TurboQuantKVCache(backend: .metalPolarQJL)
        let promptKeys = MLXArray.ones([1, 2, 4, 64], dtype: .float32)
        let promptValues = promptKeys + 0.25
        _ = try cache.updateCompressed(keys: promptKeys, values: promptValues)

        let checkpoint = try TurboQuantSpeculativeVerifier.checkpointTargetCache([cache])
        let verifiedKeys = MLXArray.ones([1, 2, 3, 64], dtype: .float32) * 2.0
        let verifiedValues = verifiedKeys + 0.5
        _ = try cache.updateCompressed(keys: verifiedKeys, values: verifiedValues)

        let rollback = try TurboQuantSpeculativeVerifier.trimAfterVerification(
            [cache],
            checkpoint: checkpoint,
            trimTokenCount: 2,
            expectedLogicalLengthDelta: 1
        )
        let snapshot = cache.runtimeSnapshot()
        let compressed = try #require(cache.compressedState)

        #expect(rollback.requestedTrimTokens == 2)
        #expect(rollback.actualTrimTokens == 2)
        #expect(rollback.resultingLogicalLengths == [5])
        #expect(snapshot.logicalLength == 5)
        #expect(snapshot.ringOffset == 0)
        #expect(snapshot.pinnedPrefixLength == 0)
        #expect(compressed.0.layout.logicalLength == 5)
        #expect(compressed.1.layout.logicalLength == 5)
    }
}
