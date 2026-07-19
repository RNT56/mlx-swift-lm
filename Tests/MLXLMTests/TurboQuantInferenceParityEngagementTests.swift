import Foundation
import MLX
import Testing

@testable import IntegrationTestHelpers
@testable import MLXLMCommon

/// Guardrail tests for the benchmark engagement fixes:
///  - FIX 1: a compressed turboQuant config with a nil runtimeMode must resolve to
///    `.capacityTurboQuant` (so the segmented compressed-decode kernel dispatches
///    instead of the throughput single-pass bypass); affine / fp16 rows unaffected.
///  - FIX 2C: `promotionGate` must block a candidate that silently ran the
///    throughput single-pass path, or (with TQ_COOP=1 and context >= 32768) that
///    did not dispatch the coop kernel.
///  - FIX 4: `promotionGate` must block a candidate whose greedy decode collapsed
///    (entropy / repetition), and pass a varied one.
@Suite("TurboQuant inference parity engagement guards")
struct TurboQuantInferenceParityEngagementTests {

    // MARK: - FIX 1: capacity-mode pin

    @Test func nilRuntimeModeTurboQuantResolvesToCapacity() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        #expect(config.strategy == .turboQuant)
        #expect(config.runtimeMode == nil)

        let params = InferenceParityBenchmark.configuredParameters(
            contextLength: 16_384,
            generateTokens: 32,
            config: config
        )
        #expect(params.turboQuantRuntimeMode == .capacityTurboQuant)
    }

    @Test func explicitCapacityRowIsCapacityAndDense() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2Capacity"))
        #expect(config.strategy == .turboQuant)
        #expect(config.runtimeMode == .capacityTurboQuant)
        #expect(config.sparseValueSelection == .off)

        let params = InferenceParityBenchmark.configuredParameters(
            contextLength: 16_384,
            generateTokens: 32,
            config: config
        )
        #expect(params.turboQuantRuntimeMode == .capacityTurboQuant)
    }

    @Test func affineK8V4RowIsUnaffectedByCapacityPin() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "affineK8V4"))
        #expect(config.strategy == .affineK8V4)
        // The guard only pins .turboQuant / .hybridTurboQuant. An affine row with a
        // nil runtimeMode must NOT be forced to capacity by this benchmark helper.
        if config.runtimeMode == nil {
            let params = InferenceParityBenchmark.configuredParameters(
                contextLength: 16_384,
                generateTokens: 32,
                config: config
            )
            #expect(params.turboQuantRuntimeMode != .capacityTurboQuant)
        }
    }

    @Test func fp16BaselineRowIsUnaffectedByCapacityPin() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "fp16"))
        #expect(config.strategy == .none)
        let params = InferenceParityBenchmark.configuredParameters(
            contextLength: 16_384,
            generateTokens: 32,
            config: config
        )
        #expect(params.turboQuantRuntimeMode != .capacityTurboQuant)
    }

    // MARK: - fixtures

    // `Memory.Snapshot` is Codable but its memberwise init is not public across the
    // module boundary, so build the fixture by decoding a tiny JSON object.
    private static func snapshot(active: Int, peak: Int) -> Memory.Snapshot {
        let json = #"{"activeMemory":\#(active),"cacheMemory":0,"peakMemory":\#(peak)}"#
        // Decoding is infallible for this fixed shape; force-unwrap keeps the helper
        // non-throwing so the fixture builders stay simple.
        return try! JSONDecoder().decode(Memory.Snapshot.self, from: Data(json.utf8))
    }

    /// A minimal non-degenerate `Measurement` for `promotionGate`. `generationTiming`
    /// is left nil so the compressedAttentionCalls==0 block does not fire, isolating
    /// the dispatch-engagement / entropy assertions under test.
    /// A `nativeMLXCompressed` layer diagnostic — the native C++ compressed decode
    /// kernel (the default on Apple GPUs). It is a legitimate compressed path even
    /// though it dispatches NO Swift kernel, so it must never trip the
    /// "compressed decode kernel never dispatched" / coop-did-not-engage blocks.
    private static func nativeCompressedDiagnostic() -> TurboQuantAttentionDiagnostics {
        TurboQuantAttentionDiagnostics(
            metalAttentionAvailable: true,
            activeAttentionPath: .nativeMLXCompressed,
            selectedKernelProfile: .macAppleSilicon,
            selfTestStatus: .passed,
            selfTestFailureReason: nil,
            optimizationPolicy: .preferThroughput,
            fallbackReason: nil,
            lastUnsupportedShape: nil,
            rawFallbackAllocated: false
        )
    }

    private static func measurement(
        context: Int,
        dispatchedKernelCounts: [String: Int],
        attentionDiagnostics: [TurboQuantAttentionDiagnostics] = [],
        distinctTokenRatio: Double = 1,
        maxTokenRunLength: Int = 1
    ) -> InferenceParityBenchmark.Measurement {
        InferenceParityBenchmark.Measurement(
            context: context,
            label: "turbo4v2",
            sampleIndex: 0,
            sampleCount: 1,
            decodeTokensPerSecond: 100,
            prefillTokensPerSecond: 1000,
            generationSeconds: 1,
            promptPrefillSeconds: 1,
            generationTokenCount: 32,
            attentionDiagnostics: attentionDiagnostics,
            cachePolicySummary: nil,
            valueBits: 4,
            valueGroupSize: 64,
            estimatedRawKVBytes: 2_000,
            estimatedConfigKVBytes: 1_000,
            memoryStart: snapshot(active: 1_000, peak: 1_000),
            memoryEnd: snapshot(active: 1_000, peak: 1_000),
            peakActiveMemoryBytes: 1_000,
            dispatchedKernelCounts: dispatchedKernelCounts,
            distinctTokenRatio: distinctTokenRatio,
            maxTokenRunLength: maxTokenRunLength
        )
    }

    private static func gate(
        _ measurement: InferenceParityBenchmark.Measurement,
        config: InferenceParityBenchmark.CacheConfig
    ) -> InferenceParityBenchmark.PromotionGate {
        InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: nil,
            runQualityGates: false
        )
    }

    // MARK: - FIX 2C: no-silent-baseline

    @Test func throughputSinglePassIsBlocked() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        let m = Self.measurement(
            context: 16_384,
            dispatchedKernelCounts: ["singlePass:turboquant_attention_fused_decode_runtime_layout_s2": 10]
        )
        let result = Self.gate(m, config: config)
        #expect(
            result.promotionBlockReasons.contains { $0.contains("throughput-mode single-pass") },
            "throughput single-pass must be blocked; got \(result.promotionBlockReasons)"
        )
    }

    @Test func noKernelDispatchedIsBlocked() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        let m = Self.measurement(context: 16_384, dispatchedKernelCounts: [:])
        let result = Self.gate(m, config: config)
        #expect(
            result.promotionBlockReasons.contains {
                $0.contains("compressed decode kernel never dispatched")
            },
            "empty dispatch must be blocked; got \(result.promotionBlockReasons)"
        )
    }

    @Test func segmentedDispatchClearsTheThroughputBlock() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        let m = Self.measurement(
            context: 16_384,
            dispatchedKernelCounts: [
                "segmented:turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1": 32
            ]
        )
        let result = Self.gate(m, config: config)
        #expect(
            !result.promotionBlockReasons.contains { $0.contains("throughput-mode single-pass") }
        )
        #expect(
            !result.promotionBlockReasons.contains {
                $0.contains("compressed decode kernel never dispatched")
            }
        )
    }

    // MARK: - FIX 2C: coop-must-engage (context >= 32768 under TQ_COOP=1)

    @Test func coopDispatchedAt32KUnderTQCoopHasNoCoopBlock() throws {
        setenv("TQ_COOP", "1", 1)
        defer { unsetenv("TQ_COOP") }
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        let m = Self.measurement(
            context: 32_768,
            dispatchedKernelCounts: [
                "segmented:turboquant_attention_fused_gqa_block_partials_coop_runtime_layout_rtu1_s2_rf1": 32
            ]
        )
        let result = Self.gate(m, config: config)
        #expect(
            !result.promotionBlockReasons.contains { $0.contains("coop kernel did not dispatch") },
            "coop dispatched -> no coop block; got \(result.promotionBlockReasons)"
        )
    }

    @Test func stridedOnlyAt32KUnderTQCoopIsBlockedForCoop() throws {
        setenv("TQ_COOP", "1", 1)
        defer { unsetenv("TQ_COOP") }
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        // Segmented (non-coop) dispatched: clears the throughput block, but coop was
        // requested + eligible and did not run, so the coop block must fire.
        let m = Self.measurement(
            context: 32_768,
            dispatchedKernelCounts: [
                "segmented:turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1": 32
            ]
        )
        let result = Self.gate(m, config: config)
        #expect(
            result.promotionBlockReasons.contains { $0.contains("coop kernel did not dispatch") },
            "coop-eligible but strided-only must be blocked; got \(result.promotionBlockReasons)"
        )
    }

    // MARK: - FIX 2C: native C++ compressed path is legitimate (regression guard)

    @Test func nativeCompressedPathIsNotBlockedDespiteEmptySwiftDispatch() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2Capacity"))
        // The native C++ kernel dispatches no Swift kernel, so dispatchedKernelCounts is
        // empty — but activeAttentionPath is .nativeMLXCompressed, a real compressed path.
        let m = Self.measurement(
            context: 16_384,
            dispatchedKernelCounts: [:],
            attentionDiagnostics: [Self.nativeCompressedDiagnostic()]
        )
        let result = Self.gate(m, config: config)
        #expect(
            !result.promotionBlockReasons.contains {
                $0.contains("never dispatched") || $0.contains("throughput-mode single-pass")
            },
            "native compressed path must not be blocked as un-dispatched; got \(result.promotionBlockReasons)"
        )
    }

    @Test func nativePathUnderTQCoopIsInformationalNotBlocked() throws {
        setenv("TQ_COOP", "1", 1)
        defer { unsetenv("TQ_COOP") }
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2Capacity"))
        // Default native path taken with coop requested: coop is inert BY DESIGN here
        // (native attention on), so this is informational, never a coop block.
        let m = Self.measurement(
            context: 32_768,
            dispatchedKernelCounts: [:],
            attentionDiagnostics: [Self.nativeCompressedDiagnostic()]
        )
        let result = Self.gate(m, config: config)
        #expect(
            !result.promotionBlockReasons.contains { $0.contains("coop kernel did not dispatch") },
            "native path must not be coop-blocked; got \(result.promotionBlockReasons)"
        )
        #expect(result.coopEngagement?.contains("native MLX attention path") == true)
    }

    // MARK: - FIX 4: entropy / repetition collapse

    @Test func degenerateSequenceIsBlocked() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        // Segmented dispatched so only the entropy assertion can fail.
        let m = Self.measurement(
            context: 16_384,
            dispatchedKernelCounts: [
                "segmented:turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1": 32
            ],
            distinctTokenRatio: 0.02,
            maxTokenRunLength: 64
        )
        let result = Self.gate(m, config: config)
        #expect(
            result.promotionBlockReasons.contains {
                $0.contains("generated text degenerate")
            },
            "degenerate sequence must be blocked; got \(result.promotionBlockReasons)"
        )
    }

    @Test func variedSequenceHasNoDegeneracyBlock() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "turbo4v2"))
        let m = Self.measurement(
            context: 16_384,
            dispatchedKernelCounts: [
                "segmented:turboquant_attention_fused_gqa_block_partials_runtime_layout_rtu1_s2_rf1": 32
            ],
            distinctTokenRatio: 0.9,
            maxTokenRunLength: 2
        )
        let result = Self.gate(m, config: config)
        #expect(
            !result.promotionBlockReasons.contains { $0.contains("generated text degenerate") }
        )
    }
}
