// Copyright © 2026 RNT56.
//
// End-to-end inference-parity benchmark.
//
// Measures REAL full-model decode throughput (tokens/sec) of a TurboQuant-compressed KV
// cache vs a plain FP16 cache, across context depths, by running the actual
// `MLXLMCommon.generate` loop on a loaded model — not the attention operator in isolation.
//
// Why this exists: `TurboQuantBench` / `TurboQuantQwenProof` time only the attention op,
// which on M2 Pro reports 0.07–0.18x of FP16. But attention is a *slice* of per-token decode
// cost (MLP / projections / norms are identical FP16-vs-compressed), so the kernel-only ratio
// massively understates end-to-end parity. This harness produces the apples-to-apples number
// comparable to community end-to-end reports (arozanov ~0.72x, Open-TQ-Metal ~0.82x).
//
// The prompt is synthetic: content is irrelevant to a *speed* measurement — only the KV-cache
// depth (= context length) matters. A separate greedy-decode-identical pass (real text, FP16
// vs codec) covers quality.

import Foundation
import Darwin
import MLX
import MLXLLM
import MLXLMCommon

public enum InferenceParityBenchmark {

    /// One KV-cache configuration raced against FP16.
    public struct CacheConfig: Sendable {
        public let label: String
        public let strategy: KVCacheStrategy
        public let preset: TurboQuantPreset?
        public let kvBits: Int?
        public let kvGroupSize: Int?
        public let kvCodec: TurboQuantKVCodec?
        public let turboQuantBackend: TurboQuantBackend?
        public let valueBits: Int?
        public let valueGroupSize: Int?
        public let runtimeMode: TurboQuantRuntimeMode?
        public let quantizedKVStart: Int?
        public let precisionPolicy: TurboQuantKVPrecisionPolicy?
        public let kvLayerPolicy: KVLayerPolicy?
        public let sparseValueSelection: TurboQuantSparseValueSelection
        public let exactPrefill: Bool

        public init(
            label: String,
            strategy: KVCacheStrategy,
            preset: TurboQuantPreset?,
            kvBits: Int? = nil,
            kvGroupSize: Int? = nil,
            kvCodec: TurboQuantKVCodec? = nil,
            turboQuantBackend: TurboQuantBackend? = nil,
            valueBits: Int? = nil,
            valueGroupSize: Int? = nil,
            runtimeMode: TurboQuantRuntimeMode? = nil,
            quantizedKVStart: Int? = nil,
            precisionPolicy: TurboQuantKVPrecisionPolicy? = nil,
            kvLayerPolicy: KVLayerPolicy? = nil,
            sparseValueSelection: TurboQuantSparseValueSelection = .off,
            exactPrefill: Bool = false
        ) {
            self.label = label
            self.strategy = strategy
            self.preset = preset
            self.kvBits = kvBits
            self.kvGroupSize = kvGroupSize
            self.kvCodec = kvCodec
            self.turboQuantBackend = turboQuantBackend
            self.valueBits = valueBits
            self.valueGroupSize = valueGroupSize
            self.runtimeMode = runtimeMode
            self.quantizedKVStart = quantizedKVStart
            self.precisionPolicy = precisionPolicy
            self.kvLayerPolicy = kvLayerPolicy
            self.sparseValueSelection = sparseValueSelection
            self.exactPrefill = exactPrefill
        }

        public func withSparseValueSelection(
            _ selection: TurboQuantSparseValueSelection,
            label explicitLabel: String? = nil
        ) -> CacheConfig {
            CacheConfig(
                label: explicitLabel ?? label,
                strategy: strategy,
                preset: preset,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                kvCodec: kvCodec,
                turboQuantBackend: turboQuantBackend,
                valueBits: valueBits,
                valueGroupSize: valueGroupSize,
                runtimeMode: runtimeMode ?? (selection.isEnabled ? .capacityTurboQuant : nil),
                quantizedKVStart: quantizedKVStart,
                precisionPolicy: precisionPolicy,
                kvLayerPolicy: kvLayerPolicy,
                sparseValueSelection: selection,
                exactPrefill: exactPrefill
            )
        }

        public func withKVLayerPolicy(_ policy: KVLayerPolicy?) -> CacheConfig {
            CacheConfig(
                label: label,
                strategy: strategy,
                preset: preset,
                kvBits: kvBits,
                kvGroupSize: kvGroupSize,
                kvCodec: kvCodec,
                turboQuantBackend: turboQuantBackend,
                valueBits: valueBits,
                valueGroupSize: valueGroupSize,
                runtimeMode: runtimeMode,
                quantizedKVStart: quantizedKVStart,
                precisionPolicy: precisionPolicy,
                kvLayerPolicy: policy ?? kvLayerPolicy,
                sparseValueSelection: sparseValueSelection
            )
        }
    }

    /// One measured (context × config) cell.
    public struct Measurement: Sendable {
        public let context: Int
        public let label: String
        public let sampleIndex: Int
        public let sampleCount: Int
        public let decodeTokensPerSecond: Double
        public let prefillTokensPerSecond: Double
        public let generationSeconds: Double
        public let generationLoopSeconds: Double
        public let generationSynchronizationSeconds: Double
        public let promptPrefillSeconds: Double
        public let generationTokenCount: Int
        public let promptPrefillTiming: TurboQuantTimingSnapshot?
        public let generationTiming: TurboQuantTimingSnapshot?
        public let attentionDiagnostics: [TurboQuantAttentionDiagnostics]
        public let cachePolicySummary: String?
        public let valueBits: Int?
        public let valueGroupSize: Int?
        public let estimatedRawKVBytes: Int?
        public let estimatedConfigKVBytes: Int?
        public let memoryStart: Memory.Snapshot
        public let memoryEnd: Memory.Snapshot
        public let peakActiveMemoryBytes: Int
        /// Native (mlx-swift) dispatched-kernel counts captured across the timed
        /// decode loop via `TurboQuantKernelDispatchTelemetry`. Proves which
        /// TurboQuant kernel family actually ran (segmented/blockParallel compressed
        /// vs throughput single-pass vs coop) so promotion cannot pass on a silent
        /// throughput-bypass. Empty when telemetry was not captured.
        public let dispatchedKernelCounts: [String: Int]
        /// Fraction of distinct token ids over the generated sequence. A collapsed
        /// (degenerate) decode approaches 0. Defaults to 1 when not measured.
        public let distinctTokenRatio: Double
        /// Longest run of identical consecutive token ids in the generated sequence.
        /// A repetition-collapsed decode has a large run. Defaults to 0 when not measured.
        public let maxTokenRunLength: Int

        public init(
            context: Int,
            label: String,
            sampleIndex: Int,
            sampleCount: Int,
            decodeTokensPerSecond: Double,
            prefillTokensPerSecond: Double,
            generationSeconds: Double,
            generationLoopSeconds: Double = 0,
            generationSynchronizationSeconds: Double = 0,
            promptPrefillSeconds: Double,
            generationTokenCount: Int,
            promptPrefillTiming: TurboQuantTimingSnapshot? = nil,
            generationTiming: TurboQuantTimingSnapshot? = nil,
            attentionDiagnostics: [TurboQuantAttentionDiagnostics],
            cachePolicySummary: String?,
            valueBits: Int?,
            valueGroupSize: Int?,
            estimatedRawKVBytes: Int?,
            estimatedConfigKVBytes: Int?,
            memoryStart: Memory.Snapshot,
            memoryEnd: Memory.Snapshot,
            peakActiveMemoryBytes: Int,
            dispatchedKernelCounts: [String: Int] = [:],
            distinctTokenRatio: Double = 1,
            maxTokenRunLength: Int = 0
        ) {
            self.context = context
            self.label = label
            self.sampleIndex = sampleIndex
            self.sampleCount = sampleCount
            self.decodeTokensPerSecond = decodeTokensPerSecond
            self.prefillTokensPerSecond = prefillTokensPerSecond
            self.generationSeconds = generationSeconds
            self.generationLoopSeconds = generationLoopSeconds
            self.generationSynchronizationSeconds = generationSynchronizationSeconds
            self.promptPrefillSeconds = promptPrefillSeconds
            self.generationTokenCount = generationTokenCount
            self.promptPrefillTiming = promptPrefillTiming
            self.generationTiming = generationTiming
            self.attentionDiagnostics = attentionDiagnostics
            self.cachePolicySummary = cachePolicySummary
            self.valueBits = valueBits
            self.valueGroupSize = valueGroupSize
            self.estimatedRawKVBytes = estimatedRawKVBytes
            self.estimatedConfigKVBytes = estimatedConfigKVBytes
            self.memoryStart = memoryStart
            self.memoryEnd = memoryEnd
            self.peakActiveMemoryBytes = peakActiveMemoryBytes
            self.dispatchedKernelCounts = dispatchedKernelCounts
            self.distinctTokenRatio = distinctTokenRatio
            self.maxTokenRunLength = maxTokenRunLength
        }

        public var estimatedMemoryReductionRatio: Double? {
            guard let raw = estimatedRawKVBytes,
                let config = estimatedConfigKVBytes,
                raw > 0,
                config > 0
            else {
                return nil
            }
            return Double(raw) / Double(config)
        }

        public var sparseSkippedTokens: Int {
            attentionDiagnostics.compactMap(\.sparseVSkippedTokens).reduce(0, +)
        }

        public var sparseTotalTokens: Int {
            attentionDiagnostics.compactMap(\.sparseVTotalTokens).reduce(0, +)
        }

        public var sparseRecentTokenCount: Int {
            attentionDiagnostics.compactMap(\.sparseVRecentTokenCount).reduce(0, +)
        }

        public var sparseOlderTokenCount: Int {
            attentionDiagnostics.compactMap(\.sparseVOlderTokenCount).reduce(0, +)
        }

        public var sparsePageCandidateCount: Int {
            attentionDiagnostics.compactMap(\.sparseVPageCandidateCount).reduce(0, +)
        }

        public var sparseSkipRatio: Double? {
            let total = sparseTotalTokens
            guard total > 0 else { return nil }
            return Double(sparseSkippedTokens) / Double(total)
        }

        public var sparseRequestedLayerCount: Int {
            attentionDiagnostics.filter { $0.sparseVEnabled }.count
        }

        public var sparseActiveLayerCount: Int {
            attentionDiagnostics.filter { $0.sparseVActive == true }.count
        }

        public var sparseRequestedButInactive: Bool {
            sparseRequestedLayerCount > 0 && sparseActiveLayerCount == 0
        }

        public var nativeKernelKinds: [Int] {
            Array(Set(attentionDiagnostics.compactMap(\.nativeKernelKind))).sorted()
        }

        public var codecCounts: [String: Int] {
            attentionDiagnostics.reduce(into: [:]) { counts, diagnostics in
                counts[diagnostics.activeAttentionPath.rawValue, default: 0] += 1
            }
        }

        public var boundaryProtectedLayerCount: Int {
            attentionDiagnostics.map(\.boundaryProtectedLayerCount).max() ?? 0
        }

        public var residualCorrectionActive: Bool {
            attentionDiagnostics.contains { $0.activeAttentionPath == .affineK8VxResidual }
        }
    }

    public struct QualityMeasurement: Sendable {
        public let context: Int
        public let label: String
        public let referenceLabel: String
        public let candidateFirst: Bool
        public let quality: TurboQuantQualityGateReport
        public let attentionDiagnostics: [TurboQuantAttentionDiagnostics]

        public init(
            context: Int,
            label: String,
            referenceLabel: String,
            candidateFirst: Bool,
            quality: TurboQuantQualityGateReport,
            attentionDiagnostics: [TurboQuantAttentionDiagnostics] = []
        ) {
            self.context = context
            self.label = label
            self.referenceLabel = referenceLabel
            self.candidateFirst = candidateFirst
            self.quality = quality
            self.attentionDiagnostics = attentionDiagnostics
        }

        public var selectedAttentionPaths: [String] {
            InferenceParityBenchmark.uniqueSortedStrings(
                attentionDiagnostics.map { $0.activeAttentionPath.rawValue }
            )
        }

        public var codecCounts: [String: Int] {
            attentionDiagnostics.reduce(into: [:]) { counts, diagnostics in
                counts[diagnostics.activeAttentionPath.rawValue, default: 0] += 1
            }
        }

        public var rawFallbackAllocated: Bool {
            attentionDiagnostics.contains { $0.rawFallbackAllocated }
        }

        public var fallbackReasons: [String] {
            InferenceParityBenchmark.diagnosticReasons(attentionDiagnostics)
        }
    }

    public struct PromotionGate: Equatable, Codable, Sendable {
        public let promotionEligible: Bool
        public let promotionBlockReasons: [String]
        public let requestedBackend: String?
        public let selectedAttentionPaths: [String]
        public let requiresMetalPolarWHT: Bool
        public let metalPolarWHTAvailable: Bool?
        public let rawFallbackAllocated: Bool
        public let decodedFallbackPathActive: Bool
        public let sparseRequestedButInactive: Bool
        public let qualityRequired: Bool
        public let qualityPassed: Bool?
        public let qualityReason: String?
        public let qualitySelectedAttentionPaths: [String]
        public let speedRatioToFP16: Double?
        public let speedRatioToAffineK8V4: Double?
        public let steadyActiveMemoryRatioToFP16: Double?
        public let peakActiveMemoryRatioToFP16: Double?
        public let fallbackReasons: [String]
        /// Whether a `TQ_COOP=1` request actually engaged the coop kernel, and if not
        /// why (native path taken / strided on the Swift path / inert below 32768).
        /// Informational — does not by itself gate promotion. Nil when coop not requested.
        public let coopEngagement: String?
    }

    public static func promotionGate(
        measurement: Measurement,
        config: CacheConfig?,
        quality: QualityMeasurement?,
        runQualityGates: Bool,
        fp16Baseline: Measurement? = nil,
        affineK8V4Baseline: Measurement? = nil
    ) -> PromotionGate {
        let diagnostics = measurement.attentionDiagnostics
        let requestedBackend = config?.turboQuantBackend?.rawValue
        let requiresMetalPolarWHT = config?.turboQuantBackend == .metalPolarWHT
        let metalPolarWHTAvailable = requiresMetalPolarWHT
            ? diagnostics.contains { $0.metalAttentionAvailable } : nil
        let rawFallbackAllocated = diagnostics.contains { $0.rawFallbackAllocated }
        let decodedFallbackPathActive = diagnostics.contains {
            switch $0.activeAttentionPath {
            case .baseline, .mlxPackedFallback, .unavailable:
                return true
            default:
                return false
            }
        }
        let selectedAttentionPaths = uniqueSortedStrings(
            diagnostics.map { $0.activeAttentionPath.rawValue }
        )
        let fallbackReasons = diagnosticReasons(diagnostics)
        let qualityDiagnostics = quality?.attentionDiagnostics ?? []
        let qualitySelectedAttentionPaths = uniqueSortedStrings(
            qualityDiagnostics.map { $0.activeAttentionPath.rawValue }
        )
        let isBaselineConfig = config?.strategy == KVCacheStrategy.none
        let isHybridK8PolarWHTConfig =
            config?.label == "hybridK8PolarWHTV3" || config?.label == "hybridK8PolarWHTV4"
        let polarWHTHybridPromotionEnabled =
            ProcessInfo.processInfo.environment["TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION"]
            == "1"
        let isFullPolarWHTDiagnosticConfig =
            config?.label == "polarWHTV3" || config?.label == "polarWHTReferenceV3"
        let isHybridReferenceDiagnosticConfig = config?.label == "hybridK8PolarWHTV3Reference"
        let isAffineCompressedConfig: Bool = {
            switch config?.strategy {
            case .mlxAffine, .affineK8V4, .affineK8Vx, .affineInt4:
                true
            default:
                false
            }
        }()
        let affineCompressedNativePathSelected = diagnostics.contains {
            switch $0.activeAttentionPath {
            case .nativeMLXCompressed, .affineK8V4Native, .affineK8VxNative,
                .affineK8VxResidual, .affineInt4Native:
                return true
            default:
                return false
            }
        }
        let qualityAffineCompressedNativePathSelected = qualityDiagnostics.contains {
            switch $0.activeAttentionPath {
            case .nativeMLXCompressed, .affineK8V4Native, .affineK8VxNative,
                .affineK8VxResidual, .affineInt4Native:
                return true
            default:
                return false
            }
        }
        let isPromotableCandidate =
            !isBaselineConfig && !isFullPolarWHTDiagnosticConfig
                && !isHybridReferenceDiagnosticConfig

        func speedRatio(to baseline: Measurement?) -> Double? {
            guard let baseline, baseline.decodeTokensPerSecond > 0 else { return nil }
            return measurement.decodeTokensPerSecond / baseline.decodeTokensPerSecond
        }

        func activeMemoryRatio(to baseline: Measurement?) -> Double? {
            guard let baseline else { return nil }
            guard baseline.memoryEnd.activeMemory > 0 else {
                return measurement.memoryEnd.activeMemory <= 0 ? 1 : nil
            }
            return Double(measurement.memoryEnd.activeMemory)
                / Double(baseline.memoryEnd.activeMemory)
        }

        func peakMemoryRatio(to baseline: Measurement?) -> Double? {
            guard let baseline else { return nil }
            guard baseline.peakActiveMemoryBytes > 0 else {
                return measurement.peakActiveMemoryBytes <= 0 ? 1 : nil
            }
            return Double(measurement.peakActiveMemoryBytes)
                / Double(baseline.peakActiveMemoryBytes)
        }

        let speedRatioToFP16 = speedRatio(to: fp16Baseline)
        let speedRatioToAffineK8V4 = speedRatio(to: affineK8V4Baseline)
        let steadyActiveMemoryRatioToFP16 = activeMemoryRatio(to: fp16Baseline)
        let peakActiveMemoryRatioToFP16 = peakMemoryRatio(to: fp16Baseline)

        var blockReasons: [String] = []
        if isBaselineConfig {
            blockReasons.append("baseline config; promotion gate applies to compressed candidates")
        }
        if isFullPolarWHTDiagnosticConfig {
            blockReasons.append("full PolarWHT K/V path is diagnostic only")
        }
        if isHybridK8PolarWHTConfig && !polarWHTHybridPromotionEnabled {
            blockReasons.append(
                "hybrid K8+PolarWHT-V is experimental; upstream published K8+V4 speed uses affine V, set TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION=1 only for deliberate promotion runs"
            )
        }
        if config?.turboQuantBackend == .polarWHTReference {
            blockReasons.append(
                "polarWHTReference is a measured reference path, not a native metalPolarWHT promotion"
            )
        }
        if requiresMetalPolarWHT && metalPolarWHTAvailable != true {
            blockReasons.append("metalPolarWHT unavailable")
        }
        if requiresMetalPolarWHT
            && diagnostics.contains(where: { $0.activeAttentionPath == .polarWHTReferenceHybrid })
        {
            blockReasons.append("metalPolarWHT request used the PolarWHT reference path")
        }
        if isHybridK8PolarWHTConfig,
            !diagnostics.contains(where: { $0.activeAttentionPath == .metalHybridK8PolarWHTValue })
        {
            blockReasons.append("hybrid K8+PolarWHT-V native path was not selected")
        }
        if isAffineCompressedConfig && !affineCompressedNativePathSelected {
            blockReasons.append("affine compressed native path was not selected")
        }
        if rawFallbackAllocated {
            blockReasons.append("raw fallback allocated")
        }
        if decodedFallbackPathActive {
            blockReasons.append("decoded or unavailable fallback path active")
        }
        if measurement.sparseRequestedButInactive {
            blockReasons.append("Sparse-V requested but inactive")
        }
        // No-silent-baseline engagement gate. A requested compressed turboQuant scheme
        // MUST run a compressed decode kernel, not the throughput single-pass bypass.
        // There are TWO legitimate compressed kernels: the native C++ kernel
        // (activeAttentionPath == .nativeMLXCompressed, the default on Apple GPUs) and
        // the Swift segmented path (only when MLX_TURBOQUANT_NATIVE_ATTENTION=0, where
        // coop lives). Only a baseline/throughput bypass is illegitimate — and that is
        // already blocked above via decodedFallbackPathActive/rawFallbackAllocated; this
        // is the belt-and-suspenders check that also catches a throughput token.
        let isCompressedTurboQuant =
            config?.strategy == .turboQuant || config?.strategy == .adaptiveTurboQuant
            || config?.strategy == .hybridTurboQuant
        var coopEngagement: String? = nil
        if isCompressedTurboQuant, !isAffineCompressedConfig {
            let dispatched = measurement.dispatchedKernelCounts
            let swiftSegmentedDispatched = dispatched.contains {
                ($0.key.hasPrefix("segmented:") || $0.key.hasPrefix("blockParallel:")) && $0.value > 0
            }
            let nativeCompressedSelected = diagnostics.contains {
                $0.activeAttentionPath == .nativeMLXCompressed
            }
            let compressedEngaged = swiftSegmentedDispatched || nativeCompressedSelected
            let ranThroughput =
                dispatched.contains { $0.key.contains("fused_decode") && $0.value > 0 }
            if !compressedEngaged {
                blockReasons.append(
                    ranThroughput
                        ? "requested compressed scheme ran throughput-mode single-pass (no compressed segmented kernel dispatched)"
                        : "compressed decode kernel never dispatched (no segmented/blockParallel kernel recorded)"
                )
            }
            // Coop engagement. Coop is an optional speed optimization on the Swift path,
            // NOT a promotion requirement (the path is promotable on the native or
            // strided kernel). Only a coop REGRESSION on the Swift path — strided ran
            // where coop was requested + eligible — is a hard block. On the default
            // native path coop is inert by design; record that so a native/strided
            // result is never misattributed to coop, but do not block.
            let coopRequested = ProcessInfo.processInfo.environment["TQ_COOP"] == "1"
            // Mirror the kernel gate's TQ_COOP_MIN_CONTEXT test override (mlx-swift
            // turboQuantCooperativeQuadDecodeActive) so this check and the informational
            // coopEngagement label agree with where coop can actually engage.
            let coopMinContext =
                ProcessInfo.processInfo.environment["TQ_COOP_MIN_CONTEXT"].flatMap { Int($0) }
                ?? 32_768
            if coopRequested && measurement.context >= coopMinContext {
                let coopDispatched = dispatched.contains { $0.key.contains("coop") && $0.value > 0 }
                if coopDispatched {
                    coopEngagement = "engaged (coop kernel dispatched)"
                } else if swiftSegmentedDispatched {
                    coopEngagement = "requested but strided ran on the Swift path"
                    blockReasons.append(
                        "TQ_COOP=1 and coop-eligible (context>=32768) but coop kernel did not dispatch (strided ran on the Swift path)"
                    )
                } else if nativeCompressedSelected {
                    coopEngagement =
                        "requested but native MLX attention path taken; coop only runs with MLX_TURBOQUANT_NATIVE_ATTENTION=0"
                }
            } else if coopRequested {
                coopEngagement = "requested but inert (context<\(coopMinContext))"
            }
        }
        if isPromotableCandidate {
            // Assertion 3 (entropy): a compressed candidate whose greedy decode
            // collapsed into a near-single-token or long-repeat sequence is not a
            // valid quality signal, regardless of tok/s.
            if measurement.distinctTokenRatio < 0.1 || measurement.maxTokenRunLength > 32 {
                blockReasons.append("generated text degenerate (entropy/repetition collapse)")
            }
            // Assertion 1 (compressed calls): when generation timing is on, a
            // compressed candidate that recorded zero compressed attention calls
            // never actually ran the compressed decode. Gated on timing != nil so
            // it does not false-positive when timing is disabled.
            if isCompressedTurboQuant,
                let timing = measurement.generationTiming,
                timing.compressedAttentionCalls == 0
            {
                blockReasons.append(
                    "compressed decode kernel never dispatched (compressedAttentionCalls==0)"
                )
            }
            if let memoryReduction = measurement.estimatedMemoryReductionRatio {
                if memoryReduction <= 1 {
                    blockReasons.append("resident KV compression ratio is not above 1.0x")
                }
            } else {
                blockReasons.append("resident KV compression ratio missing")
            }
            if let speedRatioToFP16 {
                if speedRatioToFP16 < 0.70 {
                    blockReasons.append(
                        String(
                            format: "decode speed %.3fx FP16 is below 0.70x promotion floor",
                            speedRatioToFP16
                        )
                    )
                }
            } else {
                blockReasons.append("FP16 throughput baseline missing")
            }
            if isHybridK8PolarWHTConfig {
                if let speedRatioToAffineK8V4 {
                    if speedRatioToAffineK8V4 < 0.70 {
                        blockReasons.append(
                            String(
                                format:
                                    "decode speed %.3fx affineK8V4 is below 0.70x hybrid floor",
                                speedRatioToAffineK8V4
                            )
                        )
                    }
                } else {
                    blockReasons.append("affineK8V4 throughput baseline missing")
                }
            }
            if let steadyActiveMemoryRatioToFP16 {
                if steadyActiveMemoryRatioToFP16 > 1.05 {
                    blockReasons.append(
                        String(
                            format:
                                "steady active memory %.3fx FP16 exceeds 1.05x promotion ceiling",
                            steadyActiveMemoryRatioToFP16
                        )
                    )
                }
            } else {
                blockReasons.append("FP16 steady active memory baseline missing")
            }
            if let peakActiveMemoryRatioToFP16, peakActiveMemoryRatioToFP16 > 1.10 {
                blockReasons.append(
                    String(
                        format: "peak active memory %.3fx FP16 exceeds 1.10x promotion ceiling",
                        peakActiveMemoryRatioToFP16
                    )
                )
            }
        }
        if runQualityGates {
            if let quality {
                if !quality.quality.passed {
                    let suffix = quality.quality.gateReason.map { ": \($0)" } ?? ""
                    blockReasons.append("quality gate failed" + suffix)
                } else if isPromotableCandidate {
                    if isAffineCompressedConfig && !qualityAffineCompressedNativePathSelected {
                        blockReasons.append(
                            "quality gate did not select affine compressed native path"
                        )
                    }
                    if isHybridK8PolarWHTConfig
                        && !qualityDiagnostics.contains(where: {
                            $0.activeAttentionPath == .metalHybridK8PolarWHTValue
                        })
                    {
                        blockReasons.append(
                            "quality gate did not select hybrid K8+PolarWHT-V native path"
                        )
                    }
                }
            } else if !isBaselineConfig {
                blockReasons.append("quality gate missing")
            }
        } else if !isBaselineConfig {
            blockReasons.append("quality gate not run")
        }

        let uniqueBlockReasons = uniqueSortedStrings(blockReasons)
        return PromotionGate(
            promotionEligible: uniqueBlockReasons.isEmpty,
            promotionBlockReasons: uniqueBlockReasons,
            requestedBackend: requestedBackend,
            selectedAttentionPaths: selectedAttentionPaths,
            requiresMetalPolarWHT: requiresMetalPolarWHT,
            metalPolarWHTAvailable: metalPolarWHTAvailable,
            rawFallbackAllocated: rawFallbackAllocated,
            decodedFallbackPathActive: decodedFallbackPathActive,
            sparseRequestedButInactive: measurement.sparseRequestedButInactive,
            qualityRequired: !isBaselineConfig,
            qualityPassed: quality?.quality.passed,
            qualityReason: quality?.quality.gateReason,
            qualitySelectedAttentionPaths: qualitySelectedAttentionPaths,
            speedRatioToFP16: speedRatioToFP16,
            speedRatioToAffineK8V4: speedRatioToAffineK8V4,
            steadyActiveMemoryRatioToFP16: steadyActiveMemoryRatioToFP16,
            peakActiveMemoryRatioToFP16: peakActiveMemoryRatioToFP16,
            fallbackReasons: fallbackReasons,
            coopEngagement: coopEngagement
        )
    }

    private static func uniqueSortedStrings(_ values: [String]) -> [String] {
        Array(Set(values)).sorted()
    }

    private static func diagnosticReasons(
        _ diagnostics: [TurboQuantAttentionDiagnostics]
    ) -> [String] {
        uniqueSortedStrings(
            diagnostics.flatMap { diagnostic -> [String] in
                [
                    diagnostic.fallbackReason,
                    diagnostic.lastFallback?.reason,
                    diagnostic.lastUnsupportedShape,
                    diagnostic.nativeFallbackReason,
                    diagnostic.selfTestFailureReason,
                ].compactMap { $0 }
            }
        )
    }

    public enum RunCellStatus: String, Codable, Sendable {
        case ok
        case failed
        case skipped
    }

    public struct ThroughputSample: Sendable {
        public let context: Int
        public let label: String
        public let sampleIndex: Int
        public let sampleCount: Int
        public let status: RunCellStatus
        public let measurement: Measurement?
        public let error: String?
    }

    public struct ThroughputRunResult: Sendable {
        public let measurements: [Measurement]
        public let samples: [ThroughputSample]

        public init(measurements: [Measurement] = [], samples: [ThroughputSample] = []) {
            self.measurements = measurements
            self.samples = samples
        }
    }

    public struct QualityAttempt: Sendable {
        public let context: Int
        public let label: String
        public let referenceLabel: String
        public let candidateFirst: Bool
        public let status: RunCellStatus
        public let measurement: QualityMeasurement?
        public let error: String?

        public init(
            context: Int,
            label: String,
            referenceLabel: String,
            candidateFirst: Bool,
            status: RunCellStatus,
            measurement: QualityMeasurement?,
            error: String?
        ) {
            self.context = context
            self.label = label
            self.referenceLabel = referenceLabel
            self.candidateFirst = candidateFirst
            self.status = status
            self.measurement = measurement
            self.error = error
        }
    }

    public struct QualityRunResult: Sendable {
        public let measurements: [QualityMeasurement]
        public let attempts: [QualityAttempt]

        public init(measurements: [QualityMeasurement] = [], attempts: [QualityAttempt] = []) {
            self.measurements = measurements
            self.attempts = attempts
        }
    }

    public struct QualityLogits: Codable, Sendable {
        public var values: [Float]
        public var rowWidth: Int

        public init(values: [Float], rowWidth: Int) {
            self.values = values
            self.rowWidth = rowWidth
        }
    }

    public struct QualityLogitArtifact: Codable, Sendable {
        public var version: Int
        public var benchmarkSuiteID: String
        public var generatedAt: String
        public var context: Int
        public var label: String
        public var rowWidth: Int
        public var values: [Float]

        public init(context: Int, label: String, logits: QualityLogits) {
            self.version = 1
            self.benchmarkSuiteID = TurboQuantBenchmarkSuiteID.realModelInferenceV1.rawValue
            self.generatedAt = ISO8601DateFormatter().string(from: Date())
            self.context = context
            self.label = label
            self.rowWidth = logits.rowWidth
            self.values = logits.values
        }

        public var logits: QualityLogits {
            QualityLogits(values: values, rowWidth: rowWidth)
        }
    }

    public enum QualityLogitComparisonError: Error, Equatable, LocalizedError, Sendable {
        case artifactMismatch(String)

        public var errorDescription: String? {
            switch self {
            case .artifactMismatch(let reason):
                reason
            }
        }
    }

    /// FP16 baseline plus the production-relevant compressed routes.
    ///
    /// `affineK8V4`, `mlxAffine-q8`, and `affineInt4` exercise the native MLX affine family used
    /// by the fastest community ports as the practical throughput route. `turbo3_5`/`turbo4v2`/
    /// `turbo8` keep the current TurboQuant capacity/quality routes visible in the same table.
    public static let affineThroughputQuantizedKVStart = 16_384

    private static func denseTurboPolicy(
        preset: TurboQuantPreset,
        valueBits: Int? = nil
    ) -> TurboQuantKVPrecisionPolicy {
        TurboQuantKVPrecisionPolicy.legacy(
            preset: preset,
            valueBits: valueBits,
            boundary: .disabled
        )
    }

    private static func hybridK8PolarWHTValuePolicy(
        valueBits: Int = TurboQuantKVCodec.polarWHTDefaultValueBits
    ) -> TurboQuantKVPrecisionPolicy {
        TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: valueBits),
            boundary: .disabled
        )
    }

    private static func affineK8VxPolicy(
        valueBits: Int,
        protectedEdges: Int? = nil,
        boundaryCachePrecision: TurboQuantBoundaryCachePrecision = .affineK8V4
    ) -> TurboQuantKVPrecisionPolicy {
        let edgeLayers =
            protectedEdges
            ?? TurboQuantProfile.defaultAffineK8VxProtectedEdgeLayers(valueBits: valueBits)
        return TurboQuantKVPrecisionPolicy(
            key: .affineQ8,
            value: .compressed(bits: valueBits),
            boundary: .protectedEdges(first: edgeLayers, last: edgeLayers),
            boundaryCachePrecision: boundaryCachePrecision
        )
    }

    private static func residualV2LayerPolicy() -> KVLayerPolicy {
        KVLayerPolicy(
            defaultCodec: .affineK8VxResidual(valueBits: 2, residualsPerGroup: 1)
        )
    }

    public static let defaultConfigs: [CacheConfig] = [
        CacheConfig(label: "fp16", strategy: .none, preset: nil),
        CacheConfig(
            label: "affineK8V4",
            strategy: .affineK8V4,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8V4,
            valueBits: TurboQuantKVCodec.affineK8V4ValueBits,
            quantizedKVStart: affineThroughputQuantizedKVStart
        ),
        CacheConfig(
            label: "affineK8V3",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .disabled
            )
        ),
        CacheConfig(
            label: "affineK8V2",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .disabled
            )
        ),
        CacheConfig(
            label: "affineK8V3-protectedK8V4",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V3-protectedK8V4-edge4",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .protectedEdges(first: 4, last: 4),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V3-protectedK8V4-edge5",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .protectedEdges(first: 5, last: 5),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V3-optimized",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: affineK8VxPolicy(valueBits: 3)
        ),
        CacheConfig(
            label: "affineK8V3-last2",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .custom([19, 23]),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V3-first1-last2",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .custom([3, 19, 23]),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V3-first1-penultimate",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .custom([3, 19]),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V3-optimized-vgs64",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            valueGroupSize: 64,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: affineK8VxPolicy(valueBits: 3)
        ),
        CacheConfig(
            label: "affineK8V3-protectedK8V4-edge6",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .protectedEdges(first: 6, last: 6),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V2-protectedK8V4",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V2-protectedK8V4-edge4",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 4, last: 4),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V2-protectedK8V4-edge6",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 6, last: 6),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V2-protectedK8V4-edge8",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 8, last: 8),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V2-calibrated",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 8, last: 8),
                boundaryCachePrecision: .affineK8V4
            )
        ),
        CacheConfig(
            label: "affineK8V2-residual-r1",
            strategy: .none,
            preset: nil,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            kvLayerPolicy: residualV2LayerPolicy()
        ),
        CacheConfig(
            label: "affineK8V2-calibrated-residual-r1",
            strategy: .none,
            preset: nil,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            kvLayerPolicy: residualV2LayerPolicy()
        ),
        CacheConfig(
            label: "affineK8V3-protectedRaw",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 3,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 3),
                boundary: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .raw
            )
        ),
        CacheConfig(
            label: "affineK8V2-protectedRaw",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 2, last: 2),
                boundaryCachePrecision: .raw
            )
        ),
        CacheConfig(
            label: "affineK8V2-protectedRaw-edge4",
            strategy: .affineK8Vx,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8Vx,
            valueBits: 2,
            quantizedKVStart: affineThroughputQuantizedKVStart,
            precisionPolicy: TurboQuantKVPrecisionPolicy(
                key: .affineQ8,
                value: .compressed(bits: 2),
                boundary: .protectedEdges(first: 4, last: 4),
                boundaryCachePrecision: .raw
            )
        ),
        CacheConfig(
            label: "mlxAffine-q8",
            strategy: .mlxAffine,
            preset: nil,
            kvBits: 8,
            kvGroupSize: 64,
            quantizedKVStart: affineThroughputQuantizedKVStart
        ),
        CacheConfig(
            label: "affineInt4",
            strategy: .affineInt4,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineInt4Bits,
            kvGroupSize: TurboQuantKVCodec.affineInt4DefaultGroupSize,
            kvCodec: .affineInt4,
            quantizedKVStart: affineThroughputQuantizedKVStart
        ),
        CacheConfig(
            label: "turbo3_5",
            strategy: .turboQuant,
            preset: .turbo3_5,
            precisionPolicy: denseTurboPolicy(preset: .turbo3_5)
        ),
        CacheConfig(
            label: "turbo4v2",
            strategy: .turboQuant,
            preset: .turbo4v2,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2)
        ),
        CacheConfig(
            label: "turbo4v2Capacity",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2)
        ),
        CacheConfig(
            label: "polarWHTV3",
            strategy: .turboQuant,
            preset: .turbo4v2,
            kvCodec: .polarWHT,
            turboQuantBackend: .metalPolarWHT,
            valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2)
        ),
        CacheConfig(
            label: "polarWHTReferenceV3",
            strategy: .turboQuant,
            preset: .turbo4v2,
            kvCodec: .polarWHT,
            turboQuantBackend: .polarWHTReference,
            valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2)
        ),
        CacheConfig(
            label: "hybridK8PolarWHTV3",
            strategy: .adaptiveTurboQuant,
            preset: .turbo8,
            kvCodec: .polarWHT,
            turboQuantBackend: .metalPolarWHT,
            valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: hybridK8PolarWHTValuePolicy(),
            exactPrefill: true
        ),
        CacheConfig(
            label: "hybridK8PolarWHTV4",
            strategy: .adaptiveTurboQuant,
            preset: .turbo8,
            kvCodec: .polarWHT,
            turboQuantBackend: .metalPolarWHT,
            valueBits: 4,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: hybridK8PolarWHTValuePolicy(valueBits: 4),
            exactPrefill: true
        ),
        CacheConfig(
            label: "hybridK8PolarWHTV3Reference",
            strategy: .adaptiveTurboQuant,
            preset: .turbo8,
            kvCodec: .polarWHT,
            turboQuantBackend: .polarWHTReference,
            valueBits: TurboQuantKVCodec.polarWHTDefaultValueBits,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: hybridK8PolarWHTValuePolicy(),
            exactPrefill: true
        ),
        CacheConfig(
            label: "turbo4v2SparseThreshold1e-4",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .threshold(1e-4)
        ),
        CacheConfig(
            label: "turbo4v2SparseThreshold5e-5",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .threshold(5e-5)
        ),
        CacheConfig(
            label: "turbo4v2SparseThreshold1e-5",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .threshold(1e-5)
        ),
        CacheConfig(
            label: "turbo4v2SparseTopK128",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .topK(128)
        ),
        CacheConfig(
            label: "turbo4v2SparseTopK256",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .topK(256)
        ),
        CacheConfig(
            label: "turbo4v2SparseTopK512",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .topK(512)
        ),
        CacheConfig(
            label: "turbo4v2SparseMass990",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .cumulativeMass(0.990)
        ),
        CacheConfig(
            label: "turbo4v2SparseMass995",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .cumulativeMass(0.995)
        ),
        CacheConfig(
            label: "turbo4v2SparseMass999",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .cumulativeMass(0.999)
        ),
        CacheConfig(
            label: "turbo4v2SparseHybrid995TopK128",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .hybrid(cumulativeMass: 0.995, maxTopK: 128)
        ),
        CacheConfig(
            label: "turbo4v2SparseHybrid995TopK256",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .hybrid(cumulativeMass: 0.995, maxTopK: 256)
        ),
        CacheConfig(
            label: "turbo4v2SparseHybrid995TopK512",
            strategy: .turboQuant,
            preset: .turbo4v2,
            runtimeMode: .capacityTurboQuant,
            precisionPolicy: denseTurboPolicy(preset: .turbo4v2),
            sparseValueSelection: .hybrid(cumulativeMass: 0.995, maxTopK: 512)
        ),
        CacheConfig(
            label: "turbo8",
            strategy: .turboQuant,
            preset: .turbo8,
            precisionPolicy: denseTurboPolicy(preset: .turbo8)
        ),
    ]

    public static var defaultConfigLabels: [String] {
        defaultConfigs.map(\.label)
    }

    public enum ConfigParseError: Error, Equatable, LocalizedError, Sendable {
        case unknownConfigs([String], known: [String])

        public var errorDescription: String? {
            switch self {
            case .unknownConfigs(let labels, let known):
                "Unknown TurboQuant config entries: \(labels.joined(separator: ", ")). "
                    + "Known configs: \(known.joined(separator: ", "))."
            }
        }
    }

    public static func configs(fromCSV raw: String) -> [CacheConfig]? {
        try? configs(fromCSV: raw, strict: false)
    }

    public static func configs(fromCSV raw: String, strict: Bool) throws -> [CacheConfig]? {
        let requested = raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !requested.isEmpty else { return nil }
        if requested.contains(where: { $0.lowercased() == "all" }) {
            return defaultConfigs
        }

        let known = Dictionary(uniqueKeysWithValues: defaultConfigs.map { ($0.label, $0) })
        var selected: [CacheConfig] = []
        var unknown: [String] = []
        for label in requested {
            if let config = known[label] {
                selected.append(config)
            } else {
                switch label.lowercased().replacingOccurrences(of: "-", with: "_") {
                case "mlxaffine", "mlx_affine", "q8", "affine_q8":
                    selected.append(known["mlxAffine-q8"]!)
                case "affinek8v4", "affine_k8_v4", "k8v4", "k8_v4":
                    selected.append(known["affineK8V4"]!)
                case "affinek8v3", "affine_k8_v3", "k8v3", "k8_v3":
                    selected.append(known["affineK8V3"]!)
                case "affinek8v3_optimized", "affine_k8_v3_optimized",
                    "affinek8v3optimized", "k8v3_optimized", "k8_v3_optimized",
                    "k8v3optimized":
                    selected.append(known["affineK8V3-optimized"]!)
                case "affinek8v3_last2", "affine_k8_v3_last2",
                    "affinek8v3last2", "k8v3_last2", "k8_v3_last2":
                    selected.append(known["affineK8V3-last2"]!)
                case "affinek8v3_first1_last2", "affine_k8_v3_first1_last2",
                    "affinek8v3first1last2", "k8v3_first1_last2",
                    "k8_v3_first1_last2":
                    selected.append(known["affineK8V3-first1-last2"]!)
                case "affinek8v3_first1_penultimate", "affine_k8_v3_first1_penultimate",
                    "affinek8v3first1penultimate", "k8v3_first1_penultimate",
                    "k8_v3_first1_penultimate":
                    selected.append(known["affineK8V3-first1-penultimate"]!)
                case "affinek8v3_optimized_vgs64", "affine_k8_v3_optimized_vgs64",
                    "affinek8v3optimizedvgs64", "k8v3_optimized_vgs64",
                    "k8_v3_optimized_vgs64", "k8v3vgs64":
                    selected.append(known["affineK8V3-optimized-vgs64"]!)
                case "affinek8v2", "affine_k8_v2", "k8v2", "k8_v2":
                    selected.append(known["affineK8V2"]!)
                case "affinek8v2_calibrated", "affine_k8_v2_calibrated",
                    "k8v2_calibrated", "k8_v2_calibrated":
                    selected.append(known["affineK8V2-calibrated"]!)
                case "affinek8v2_residual_r1", "affine_k8_v2_residual_r1",
                    "k8v2_residual_r1", "k8_v2_residual_r1":
                    selected.append(known["affineK8V2-residual-r1"]!)
                case "affinek8v2_calibrated_residual_r1",
                    "affine_k8_v2_calibrated_residual_r1",
                    "k8v2_calibrated_residual_r1", "k8_v2_calibrated_residual_r1":
                    selected.append(known["affineK8V2-calibrated-residual-r1"]!)
                case "int4", "affine_int4", "affineint4":
                    selected.append(known["affineInt4"]!)
                case "turbo35", "turbo_3_5":
                    selected.append(known["turbo3_5"]!)
                case "turbo4", "turbo_4_v2":
                    selected.append(known["turbo4v2"]!)
                case "polarwht", "polar_wht", "polarwhtv3", "polar_wht_v3", "whtv3":
                    selected.append(known["polarWHTV3"]!)
                case "polarwhtreference", "polar_wht_reference",
                    "polarwhtreferencev3", "polar_wht_reference_v3",
                    "whtreference", "wht_reference", "whtreferencev3",
                    "wht_reference_v3", "polarwhtrefv3", "polar_wht_ref_v3":
                    selected.append(known["polarWHTReferenceV3"]!)
                case "hybridk8polarwhtv3reference",
                    "hybrid_k8_polar_wht_v3_reference",
                    "hybridk8polarwhtreference",
                    "hybrid_k8_polar_wht_reference",
                    "k8_polar_wht_v3_reference",
                    "k8polarwhtv3reference",
                    "k8_wht_v3_reference",
                    "k8whtv3reference":
                    selected.append(known["hybridK8PolarWHTV3Reference"]!)
                case "hybridk8polarwhtv3",
                    "hybrid_k8_polar_wht_v3",
                    "hybridk8polarwht",
                    "hybrid_k8_polar_wht",
                    "k8_polar_wht_v3",
                    "k8polarwhtv3",
                    "k8_wht_v3",
                    "k8whtv3":
                    selected.append(known["hybridK8PolarWHTV3"]!)
                case "hybridk8polarwhtv4",
                    "hybrid_k8_polar_wht_v4",
                    "k8_polar_wht_v4",
                    "k8polarwhtv4",
                    "k8_wht_v4",
                    "k8whtv4":
                    selected.append(known["hybridK8PolarWHTV4"]!)
                default:
                    if strict {
                        unknown.append(label)
                    } else {
                        print("warning: ignoring unknown TurboQuant config entry '\(label)'")
                    }
                }
            }
        }
        if strict, !unknown.isEmpty {
            throw ConfigParseError.unknownConfigs(unknown, known: defaultConfigLabels)
        }
        return selected.isEmpty ? nil : selected
    }

    public static func configsFromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [CacheConfig]? {
        guard let raw = environment["TQ_PARITY_CONFIGS"] else { return nil }
        return configs(fromCSV: raw)
    }

    public static func config(named label: String) -> CacheConfig? {
        configs(fromCSV: label)?.first
    }

    public static func sparseValueSelection(
        mode rawMode: String?,
        threshold: Float? = nil,
        topK: Int? = nil,
        cumulativeMass: Float? = nil,
        maxTopK: Int? = nil,
        recentTokens: Int? = nil,
        candidatePages: Int? = nil
    ) -> TurboQuantSparseValueSelection? {
        guard let rawMode else { return nil }
        switch normalizedSparseSelectionMode(rawMode) {
        case "off", "false", "none", "0":
            return .off
        case "threshold":
            return .threshold(threshold ?? TurboQuantSparseValuePolicy.defaultAutoThreshold)
        case "blockthreshold", "blockmass":
            return .blockThreshold(threshold ?? TurboQuantSparseValuePolicy.defaultAutoThreshold)
        case "topk":
            return .topK(topK ?? maxTopK ?? 256)
        case "pagetopk", "page":
            return .pageTopK(topK ?? maxTopK ?? 4)
        case "candidatesparse":
            return .candidateSparse(
                recentTokens: recentTokens ?? 256,
                candidatePages: candidatePages ?? 4,
                olderTokenBudget: topK ?? maxTopK ?? 256
            )
        case "cumulativemass", "mass":
            return .cumulativeMass(cumulativeMass ?? 0.995)
        case "hybrid", "hybridcumulativemasstopk":
            return .hybrid(
                cumulativeMass: cumulativeMass ?? 0.995,
                maxTopK: maxTopK ?? topK ?? 256
            )
        default:
            return nil
        }
    }

    public static func applyingSparseValueSelectionOverride(
        _ selection: TurboQuantSparseValueSelection?,
        to configs: [CacheConfig]?
    ) -> [CacheConfig]? {
        guard let selection, let configs else { return configs }
        return configs.map { config in
            guard config.strategy == .turboQuant else { return config }
            return config.withSparseValueSelection(
                selection,
                label: sparseOverrideLabel(base: config.label, selection: selection)
            )
        }
    }

    public static func sparseOverrideLabel(
        base: String,
        selection: TurboQuantSparseValueSelection
    ) -> String {
        let denseBase = base.components(separatedBy: "Sparse").first ?? base
        switch selection.mode {
        case .off:
            return denseBase
        case .threshold:
            let threshold = selection.threshold ?? TurboQuantSparseValuePolicy.defaultAutoThreshold
            return denseBase + "SparseThreshold" + sparseFloatLabel(Double(threshold))
        case .blockThreshold:
            let threshold = selection.threshold ?? TurboQuantSparseValuePolicy.defaultAutoThreshold
            return denseBase + "SparseBlockThreshold" + sparseFloatLabel(Double(threshold))
        case .topK:
            return denseBase + "SparseTopK\(selection.topK ?? 256)"
        case .pageTopK:
            return denseBase + "SparsePageTopK\(selection.topK ?? 4)"
        case .candidateSparse:
            return denseBase + "CandidateSparseR\(selection.recentTokens ?? 256)"
                + "P\(selection.candidatePages ?? 4)"
                + "Older\(selection.topK ?? 256)"
        case .cumulativeMass:
            return denseBase + "SparseMass" + sparseMassLabel(selection.cumulativeMass ?? 0.995)
        case .hybridCumulativeMassTopK:
            return denseBase + "SparseHybrid"
                + sparseMassLabel(selection.cumulativeMass ?? 0.995)
                + "TopK\(selection.maxTopK ?? selection.topK ?? 256)"
        }
    }

    private struct SeededGenerator: RandomNumberGenerator {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private static func throughputPlan(
        configs: [CacheConfig],
        repeats: Int,
        randomizeOrder: Bool,
        randomSeed: UInt64,
        context: Int
    ) -> [CacheConfig] {
        let repeatCount = max(1, repeats)
        var plan = Array(repeating: configs, count: repeatCount).flatMap { $0 }
        guard randomizeOrder else { return plan }
        var generator = SeededGenerator(seed: randomSeed ^ UInt64(bitPattern: Int64(context)))
        plan.shuffle(using: &generator)
        return plan
    }

    private static func medianDecodeSample(_ samples: [Measurement]) -> Measurement? {
        guard !samples.isEmpty else { return nil }
        return samples.sorted {
            if $0.decodeTokensPerSecond == $1.decodeTokensPerSecond {
                return $0.prefillTokensPerSecond < $1.prefillTokensPerSecond
            }
            return $0.decodeTokensPerSecond < $1.decodeTokensPerSecond
        }[samples.count / 2]
    }

    /// Run the full sweep. For each context, races every config against FP16 and prints a
    /// table plus the decode-throughput ratio — the headline inference-parity number.
    @discardableResult
    public static func run(
        container: LLModelContainer,
        contexts: [Int],
        generateTokens: Int = 64,
        configs: [CacheConfig]? = nil,
        throughputRepeats: Int = 1,
        randomizeOrder: Bool = false,
        randomSeed: UInt64 = 0x5451_2026_0602,
        memoryProfile: ModelMemoryProfile? = nil,
        cooldownSeconds: Double = 0.25,
        turboQuantTimingEnabled: Bool = false
    ) async throws -> [Measurement] {
        try await runDetailed(
            container: container,
            contexts: contexts,
            generateTokens: generateTokens,
            configs: configs,
            throughputRepeats: throughputRepeats,
            randomizeOrder: randomizeOrder,
            randomSeed: randomSeed,
            memoryProfile: memoryProfile,
            cooldownSeconds: cooldownSeconds,
            turboQuantTimingEnabled: turboQuantTimingEnabled
        ).measurements
    }

    @discardableResult
    public static func runDetailed(
        container: LLModelContainer,
        contexts: [Int],
        generateTokens: Int = 64,
        configs: [CacheConfig]? = nil,
        throughputRepeats: Int = 1,
        randomizeOrder: Bool = false,
        randomSeed: UInt64 = 0x5451_2026_0602,
        memoryProfile: ModelMemoryProfile? = nil,
        cooldownSeconds: Double = 0.25,
        turboQuantTimingEnabled: Bool = false,
        sampleObserver: (@Sendable (ThroughputSample) -> Void)? = nil
    ) async throws -> ThroughputRunResult {
        let configs = configs ?? configsFromEnvironment() ?? defaultConfigs
        let repeatCount = max(1, throughputRepeats)
        let cooldownSeconds = max(0, cooldownSeconds)
        var results: [Measurement] = []
        var samples: [ThroughputSample] = []

        // Untimed warmup: realize weights + compile Metal kernels so the first timed cell is
        // not charged for one-time setup.
        _ = try? await measureOne(
            container: container, contextLength: 512, generateTokens: 4,
            config: CacheConfig(label: "warmup", strategy: .none, preset: nil))

        print("=== TurboQuant end-to-end inference parity (full decode loop, codec vs FP16) ===")
        print("ctx       config                         decode tok/s   prefill tok/s   gen   vs fp16   vs k8v4   KV memx   sparse skip")
        print("-------   ----------------------------   ------------   -------------   ---   -------   -------   -------   -----------")
        fflush(stdout)

        for ctx in contexts {
            var samplesByLabel: [String: [Measurement]] = [:]
            var attemptsByLabel: [String: Int] = [:]
            for cfg in throughputPlan(
                configs: configs,
                repeats: repeatCount,
                randomizeOrder: randomizeOrder,
                randomSeed: randomSeed,
                context: ctx
            ) {
                let sampleIndex = (attemptsByLabel[cfg.label] ?? 0) + 1
                attemptsByLabel[cfg.label] = sampleIndex
                do {
                    print(
                        "running ctx=\(ctx) config=\(cfg.label) gen=\(generateTokens) sample=\(sampleIndex)/\(repeatCount)"
                    )
                    fflush(stdout)
                    let m = try await measureOne(
                        container: container, contextLength: ctx,
                        generateTokens: generateTokens, config: cfg,
                        sampleIndex: sampleIndex,
                        sampleCount: repeatCount,
                        memoryProfile: memoryProfile,
                        turboQuantTimingEnabled: turboQuantTimingEnabled)
                    samplesByLabel[cfg.label, default: []].append(m)
                    let sample = ThroughputSample(
                        context: ctx,
                        label: cfg.label,
                        sampleIndex: sampleIndex,
                        sampleCount: repeatCount,
                        status: .ok,
                        measurement: m,
                        error: nil
                    )
                    samples.append(sample)
                    sampleObserver?(sample)
                    cooldown(seconds: cooldownSeconds)
                } catch {
                    print("\(ctx)   \(cfg.label)   FAILED: \(error)")
                    fflush(stdout)
                    let sample = ThroughputSample(
                        context: ctx,
                        label: cfg.label,
                        sampleIndex: sampleIndex,
                        sampleCount: repeatCount,
                        status: .failed,
                        measurement: nil,
                        error: String(describing: error)
                    )
                    samples.append(sample)
                    sampleObserver?(sample)
                    cooldown(seconds: cooldownSeconds)
                }
            }
            let aggregateByLabel = samplesByLabel.compactMapValues(medianDecodeSample)
            let fp16Decode = aggregateByLabel["fp16"]?.decodeTokensPerSecond
            let k8v4Decode = aggregateByLabel["affineK8V4"]?.decodeTokensPerSecond
            for cfg in configs {
                guard let m = aggregateByLabel[cfg.label] else { continue }
                results.append(m)

                let fp16Ratio: String
                if let base = fp16Decode, base > 0, cfg.label != "fp16" {
                    fp16Ratio = String(format: "%.3f", m.decodeTokensPerSecond / base)
                } else {
                    fp16Ratio = "--"
                }
                let k8v4Ratio = k8v4Decode.map {
                    String(format: "%.3f", m.decodeTokensPerSecond / $0)
                } ?? "--"
                let sparseRatio = m.sparseSkipRatio.map {
                    String(format: "%.2f%%", $0 * 100)
                } ?? "--"
                let memoryRatio = m.estimatedMemoryReductionRatio.map {
                    String(format: "%.2f", $0)
                } ?? "--"
                print(
                    pad("\(ctx)", 7) + "   " + pad(cfg.label, 28) + "   "
                        + pad(round2(m.decodeTokensPerSecond), 12) + "   "
                        + pad(round1(m.prefillTokensPerSecond), 13) + "   "
                        + pad("\(m.generationTokenCount)", 3) + "   "
                        + pad(fp16Ratio, 7) + "   " + pad(k8v4Ratio, 7) + "   "
                        + pad(memoryRatio, 7) + "   "
                        + sparseRatio)
                if repeatCount > 1 {
                    print("          samples: \(samplesByLabel[cfg.label]?.count ?? 0) median-selected")
                }
                if let summary = diagnosticSummary(m.attentionDiagnostics) {
                    print("          diagnostics: \(summary)")
                }
                fflush(stdout)
            }
        }
        return ThroughputRunResult(measurements: results, samples: samples)
    }

    @discardableResult
    public static func runQualityGates(
        container: LLModelContainer,
        contexts: [Int],
        configs: [CacheConfig]? = nil,
        referenceConfig: CacheConfig? = nil,
        candidateFirst: Bool = false,
        cooldownSeconds: Double = 0.5
    ) async throws -> [QualityMeasurement] {
        try await runQualityGatesDetailed(
            container: container,
            contexts: contexts,
            configs: configs,
            referenceConfig: referenceConfig,
            candidateFirst: candidateFirst,
            cooldownSeconds: cooldownSeconds,
            failFast: true
        ).measurements
    }

    @discardableResult
    public static func runQualityGatesDetailed(
        container: LLModelContainer,
        contexts: [Int],
        configs: [CacheConfig]? = nil,
        referenceConfig: CacheConfig? = nil,
        candidateFirst: Bool = false,
        cooldownSeconds: Double = 0.5,
        failFast: Bool = false,
        attemptObserver: (@Sendable (QualityAttempt) -> Void)? = nil
    ) async throws -> QualityRunResult {
        let configs = configs ?? configsFromEnvironment() ?? defaultConfigs
        let reference = referenceConfig ?? CacheConfig(label: "fp16", strategy: .none, preset: nil)
        let candidates = configs.filter { $0.label != reference.label }
        let cooldownSeconds = max(0, cooldownSeconds)
        guard !candidates.isEmpty else { return QualityRunResult() }

        print(
            "\n=== TurboQuant real-model logit quality gates "
                + "(candidate vs \(reference.label)) ==="
        )
        print("ctx       config          top1    kl_mean      p95_abs     cosine    passed")
        print("-------   -------------   -----   ----------   ---------   -------   ------")
        fflush(stdout)

        var results: [QualityMeasurement] = []
        var attempts: [QualityAttempt] = []
        for ctx in contexts {
            for cfg in candidates {
                do {
                    let evaluation = try await qualityOne(
                        container: container,
                        contextLength: ctx,
                        referenceConfig: reference,
                        candidateConfig: cfg,
                        candidateFirst: candidateFirst
                    )
                    cooldown(seconds: cooldownSeconds)
                    let measurement = QualityMeasurement(
                        context: ctx,
                        label: cfg.label,
                        referenceLabel: reference.label,
                        candidateFirst: candidateFirst,
                        quality: evaluation.report,
                        attentionDiagnostics: evaluation.candidateDiagnostics
                    )
                    results.append(measurement)
                    let attempt = QualityAttempt(
                        context: ctx,
                        label: cfg.label,
                        referenceLabel: reference.label,
                        candidateFirst: candidateFirst,
                        status: .ok,
                        measurement: measurement,
                        error: nil
                    )
                    attempts.append(attempt)
                    attemptObserver?(attempt)
                    let top1 = String(
                        format: "%.3f",
                        evaluation.report.deterministicTop1MatchRate
                    )
                    let kl = String(format: "%.6f", evaluation.report.logitKLDivergenceMean)
                    let p95 = String(format: "%.4f", evaluation.report.logitMaxAbsErrorP95)
                    let cosine = evaluation.report.attentionOutputCosineMean.map {
                        String(format: "%.4f", $0)
                    } ?? "--"
                    let passed = evaluation.report.passed ? "yes" : "no"
                    print(
                        [
                            pad("\(ctx)", 7),
                            pad(cfg.label, 13),
                            pad(top1, 5),
                            pad(kl, 10),
                            pad(p95, 9),
                            pad(cosine, 7),
                            passed,
                        ].joined(separator: "   "))
                    if let reason = evaluation.report.gateReason {
                        print("          reason: \(reason)")
                    }
                    fflush(stdout)
                } catch {
                    cooldown(seconds: cooldownSeconds)
                    let attempt = QualityAttempt(
                        context: ctx,
                        label: cfg.label,
                        referenceLabel: reference.label,
                        candidateFirst: candidateFirst,
                        status: .failed,
                        measurement: nil,
                        error: String(describing: error)
                    )
                    attempts.append(attempt)
                    attemptObserver?(attempt)
                    print("\(ctx)   \(cfg.label)   FAILED: \(error)")
                    fflush(stdout)
                    if failFast {
                        throw error
                    }
                }
            }
        }
        return QualityRunResult(measurements: results, attempts: attempts)
    }

    @discardableResult
    public static func writeQualityLogits(
        container: LLModelContainer,
        contextLength: Int,
        config: CacheConfig,
        outputPath: String
    ) async throws -> QualityLogitArtifact {
        let promptIds = qualityPromptIds(contextLength: contextLength)
        let artifact = try await container.perform { context in
            let logits = try logitsForQuality(
                model: context.model,
                promptIds: promptIds,
                contextLength: contextLength,
                config: config
            )
            clearRuntimeMemory()
            return QualityLogitArtifact(
                context: contextLength,
                label: config.label,
                logits: logits.logits
            )
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(artifact)
        let url = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
        return artifact
    }

    public static func qualityMeasurement(
        referenceLogitsPath: String,
        candidateLogitsPath: String
    ) throws -> QualityMeasurement {
        let decoder = JSONDecoder()
        let reference = try decoder.decode(
            QualityLogitArtifact.self,
            from: Data(contentsOf: URL(fileURLWithPath: referenceLogitsPath))
        )
        let candidate = try decoder.decode(
            QualityLogitArtifact.self,
            from: Data(contentsOf: URL(fileURLWithPath: candidateLogitsPath))
        )
        try validateQualityLogitArtifacts(reference: reference, candidate: candidate)
        return QualityMeasurement(
            context: candidate.context,
            label: candidate.label,
            referenceLabel: reference.label,
            candidateFirst: true,
            quality: qualityReport(candidate: candidate.logits, reference: reference.logits)
        )
    }

    private static func validateQualityLogitArtifacts(
        reference: QualityLogitArtifact,
        candidate: QualityLogitArtifact
    ) throws {
        var mismatches: [String] = []
        if reference.version != candidate.version {
            mismatches.append("version \(reference.version) != \(candidate.version)")
        }
        if reference.benchmarkSuiteID != candidate.benchmarkSuiteID {
            mismatches.append(
                "benchmarkSuiteID \(reference.benchmarkSuiteID) != \(candidate.benchmarkSuiteID)"
            )
        }
        if reference.context != candidate.context {
            mismatches.append("context \(reference.context) != \(candidate.context)")
        }
        if reference.rowWidth != candidate.rowWidth {
            mismatches.append("rowWidth \(reference.rowWidth) != \(candidate.rowWidth)")
        }
        if reference.values.count != candidate.values.count {
            mismatches.append("value count \(reference.values.count) != \(candidate.values.count)")
        }
        if reference.rowWidth <= 0 || candidate.rowWidth <= 0 {
            mismatches.append("rowWidth must be positive")
        }
        if reference.values.isEmpty || candidate.values.isEmpty {
            mismatches.append("logit values must be nonempty")
        }
        guard mismatches.isEmpty else {
            throw QualityLogitComparisonError.artifactMismatch(
                "Quality logit artifacts do not match: \(mismatches.joined(separator: "; "))."
            )
        }
    }

    private struct QualityLogitsResult {
        var logits: QualityLogits
        var attentionDiagnostics: [TurboQuantAttentionDiagnostics]
    }

    private struct QualityEvaluation {
        var report: TurboQuantQualityGateReport
        var candidateDiagnostics: [TurboQuantAttentionDiagnostics]
    }

    private static func qualityOne(
        container: LLModelContainer,
        contextLength: Int,
        referenceConfig: CacheConfig,
        candidateConfig: CacheConfig,
        candidateFirst: Bool
    ) async throws -> QualityEvaluation {
        let promptIds = qualityPromptIds(contextLength: contextLength)

        return try await container.perform { context in
            let candidate: QualityLogitsResult
            let reference: QualityLogitsResult
            if candidateFirst {
                candidate = try logitsForQuality(
                    model: context.model,
                    promptIds: promptIds,
                    contextLength: contextLength,
                    config: candidateConfig
                )
                clearRuntimeMemory()
                reference = try logitsForQuality(
                    model: context.model,
                    promptIds: promptIds,
                    contextLength: contextLength,
                    config: referenceConfig
                )
                clearRuntimeMemory()
            } else {
                reference = try logitsForQuality(
                    model: context.model,
                    promptIds: promptIds,
                    contextLength: contextLength,
                    config: referenceConfig
                )
                clearRuntimeMemory()
                candidate = try logitsForQuality(
                    model: context.model,
                    promptIds: promptIds,
                    contextLength: contextLength,
                    config: candidateConfig
                )
                clearRuntimeMemory()
            }
            return QualityEvaluation(
                report: qualityReport(candidate: candidate.logits, reference: reference.logits),
                candidateDiagnostics: candidate.attentionDiagnostics
            )
        }
    }

    private static func qualityPromptIds(contextLength: Int) -> [Int32] {
        (0 ..< contextLength).map { Int32($0 % 1024 + 16) }
    }

    private static func compactArrayFingerprint(_ array: MLXArray) -> String {
        let shape = array.shape.map(String.init).joined(separator: "x")
        let floatArray = array.asType(.float32)
        let sumValue = floatArray.sum().item(Float.self)
        let maxAbsValue = abs(floatArray).max().item(Float.self)
        return
            "shape=\(shape) dtype=\(array.dtype) nbytes=\(array.nbytes) sum=\(String(format: "%.6g", sumValue)) maxAbs=\(String(format: "%.6g", maxAbsValue))"
    }

    private static func printTurboQuantCacheFingerprints(
        _ cache: [KVCache],
        configLabel: String
    ) {
        for (index, item) in cache.enumerated() {
            guard let compressed = item as? any TurboQuantCompressedKVCacheProtocol else {
                continue
            }
            let layer = compressed.layerIndex ?? index
            if let key = compressed.hybridAffineKeyState {
                let biasSummary = key.2.map(compactArrayFingerprint) ?? "nil"
                print(
                    "  layer \(layer) hybridAffineKey weight[\(compactArrayFingerprint(key.0))] scales[\(compactArrayFingerprint(key.1))] biases[\(biasSummary)]"
                )
            } else {
                print("  layer \(layer) hybridAffineKey nil")
            }
            if let value = compressed.polarWHTValueState {
                let layout = value.layout
                print(
                    "  layer \(layer) polarWHTValue bits=\(value.bits) logical=\(layout.logicalLength) capacity=\(layout.capacity) ring=\(layout.ringOffset) pinned=\(layout.pinnedPrefixLength) packed[\(compactArrayFingerprint(value.packedIndices))] norms[\(compactArrayFingerprint(value.norms))]"
                )
            } else {
                print("  layer \(layer) polarWHTValue nil")
            }
        }
        fflush(stdout)
    }

    private static func logitsForQuality(
        model: any LanguageModel,
        promptIds: [Int32],
        contextLength: Int,
        config: CacheConfig
    ) throws -> QualityLogitsResult {
        let tokens = MLXArray(promptIds)
        let input = LMInput(tokens: tokens)
        let params = configuredParameters(
            contextLength: contextLength,
            generateTokens: 1,
            config: config
        )
        let runtimeParams = try resolvedParameters(model: model, parameters: params)
        var cache = model.newCache(parameters: runtimeParams)
        let prefillLogits: MLXArray
        switch try model.prepare(input, cache: cache, windowSize: runtimeParams.prefillStepSize) {
        case .tokens(let tokens):
            prefillLogits = try callModel(model, text: tokens, cache: cache).logits
        case .logits(let output):
            prefillLogits = output.logits
        }

        let nextToken = argMax(prefillLogits[0..., -1, 0...], axis: -1)
        maybeQuantizeKVCache(cache: &cache, parameters: runtimeParams)
        if ProcessInfo.processInfo.environment["TQ_QUALITY_PRINT_CACHE_FINGERPRINTS"] == "1" {
            print("quality cache fingerprints \(config.label):")
            printTurboQuantCacheFingerprints(cache, configLabel: config.label)
        }
        let result = try callModel(
            model,
            text: LMInput.Text(tokens: nextToken),
            cache: cache
        )
        let diagnostics = qualityCacheDiagnostics(cache)
        if ProcessInfo.processInfo.environment["TQ_QUALITY_PRINT_CACHE_DIAGNOSTICS"] == "1" {
            let summary = diagnosticSummary(diagnostics) ?? "no TurboQuant diagnostics"
            print("quality diagnostics \(config.label): \(summary)")
            for (index, diagnostic) in diagnostics.prefix(8).enumerated() {
                print(
                    "  layer \(diagnostic.layerIndex ?? index): path=\(diagnostic.activeAttentionPath.rawValue) keyWHT=\(diagnostic.polarWHTKeyPayloadAllocated) valueWHT=\(diagnostic.polarWHTValuePayloadAllocated) rawFallback=\(diagnostic.rawFallbackAllocated) reason=\(diagnostic.lastFallback?.reason ?? diagnostic.lastUnsupportedShape ?? diagnostic.fallbackReason ?? "nil")"
                )
            }
            fflush(stdout)
        }
        let logits = result.logits[0..., -1, 0...].asType(.float32)
        eval(logits)
        Stream.gpu.synchronize()
        let rowWidth = logits.shape.last ?? 0
        let values = logits.asArray(Float.self)
        clearRuntimeMemory()
        return QualityLogitsResult(
            logits: QualityLogits(values: values, rowWidth: rowWidth),
            attentionDiagnostics: diagnostics
        )
    }

    private static func qualityCacheDiagnostics(_ cache: [KVCache]) -> [TurboQuantAttentionDiagnostics] {
        cache.compactMap { item -> TurboQuantAttentionDiagnostics? in
            if let compressed = item as? any TurboQuantCompressedKVCacheProtocol {
                return compressed.attentionDiagnostics
            }
            if let affine = item as? any NativeAffineK8V4KVCacheProtocol {
                return affine.attentionDiagnostics
            }
            return nil
        }
    }

    private static func resolvedParameters(
        model: any LanguageModel,
        parameters: GenerateParameters
    ) throws -> GenerateParameters {
        let topology = (model as? any KVCacheDimensionProvider).map {
            KVLayerTopology(kvHeads: $0.kvHeads)
        }
        let resolved = try parameters.resolvedForTurboQuantRuntime(
            layerCount: topology?.admissionLayerCount,
            kvLayerTopology: topology
        )
        if let policy = resolved.kvLayerPolicy {
            let errors = policy.validationErrors(layerCount: topology?.modelLayerCount)
            if !errors.isEmpty {
                throw IntegrationTestFailure(
                    "invalid KV layer policy: \(errors.joined(separator: "; "))"
                )
            }
        }
        return resolved
    }

    private static func callModel(
        _ model: any LanguageModel,
        text: LMInput.Text,
        cache: [KVCache]
    ) throws -> LMOutput {
        if let throwingModel = model as? any ThrowingLanguageModel {
            return try throwingModel.callAsFunctionThrowing(
                text[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
        }
        return model(
            text[text: .newAxis],
            cache: cache.isEmpty ? nil : cache,
            state: nil
        )
    }

    private static func maybeQuantizeKVCache(
        cache: inout [KVCache],
        parameters: GenerateParameters
    ) {
        MLXLMCommon.maybeQuantizeKVCache(
            cache: &cache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart,
            kvCacheStrategy: parameters.kvCacheStrategy,
            kvCodec: parameters.kvCodec,
            turboQuantPreset: parameters.turboQuantPreset,
            turboQuantBackend: parameters.turboQuantBackend,
            turboQuantOptimizationPolicy: parameters.turboQuantOptimizationPolicy,
            turboQuantFallbackPolicy: parameters.turboQuantFallbackPolicy,
            turboQuantSeed: parameters.turboQuantSeed,
            turboQuantValueBits: parameters.turboQuantValueBits,
            turboQuantPrecisionPolicy: parameters.effectiveTurboQuantPrecisionPolicy,
            turboQuantValueGroupSize: parameters.turboQuantValueGroupSize,
            turboQuantSparseValuePolicy: parameters.effectiveTurboQuantSparseValuePolicy,
            turboQuantSparseValueSelection: parameters.effectiveTurboQuantSparseValueSelection,
            turboQuantResidentBudgetBytes: parameters.turboQuantPerCacheResidentBudgetBytes,
            spillMemoryWatermarkBytes: parameters.spillMemoryWatermarkBytes,
            kvLayerPolicy: parameters.kvLayerPolicy
        )
    }

    private static func measureOne(
        container: LLModelContainer,
        contextLength: Int,
        generateTokens: Int,
        config: CacheConfig,
        sampleIndex: Int = 1,
        sampleCount: Int = 1,
        memoryProfile: ModelMemoryProfile? = nil,
        turboQuantTimingEnabled: Bool = false
    ) async throws -> Measurement {
        // Synthetic in-vocab prompt of exact length. Content is irrelevant for a speed
        // measurement; only KV-cache depth (= contextLength) matters.
        let promptIds = (0 ..< contextLength).map { Int32($0 % 1024 + 16) }

        return try await container.perform { context in
            let previousTimingOverride =
                turboQuantTimingEnabled
                ? TurboQuantTiming.setEnabledOverride(true) : nil
            defer {
                if turboQuantTimingEnabled {
                    TurboQuantTiming.setEnabledOverride(previousTimingOverride)
                }
            }
            let collectTurboQuantTiming = TurboQuantTiming.isEnabled
            if collectTurboQuantTiming {
                TurboQuantTiming.reset()
            }

            // LMInput expects a 1-D [seq] token array (it adds the batch dim internally);
            // see the reference generate(promptTokens:) path. Passing [1, seq] corrupts shapes.
            let tokens = MLXArray(promptIds)  // [contextLength]
            let input = LMInput(tokens: tokens)

            let params = configuredParameters(
                contextLength: contextLength,
                generateTokens: generateTokens,
                config: config
            )

            Stream().synchronize()
            Memory.peakMemory = 0
            let memoryStart = Memory.snapshot()
            var iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: params
            )
            let promptPrefillTiming =
                collectTurboQuantTiming ? TurboQuantTiming.snapshot() : nil
            if collectTurboQuantTiming {
                TurboQuantTiming.reset()
            }
            // Reset the always-on native dispatch counters immediately before the
            // timed decode loop so the post-loop snapshot is exactly the kernels this
            // run dispatched. This region MUST stay serial (a single synchronous
            // `while iterator.next()` loop) — the counter delta is process-global, so
            // a concurrent measureOne would contaminate the attribution.
            TurboQuantKernelDispatchTelemetry.reset()
            let generationStart = Date.timeIntervalSinceReferenceDate
            var generated = 0
            var generatedTokenIds: [Int] = []
            while let tokenId = iterator.next() {
                generated += 1
                generatedTokenIds.append(tokenId)
            }
            let generationLoopEnd = Date.timeIntervalSinceReferenceDate
            let dispatchedKernels = TurboQuantKernelDispatchTelemetry.snapshot()
            let synchronizationStart = generationLoopEnd
            Stream().synchronize()
            let synchronizationEnd = Date.timeIntervalSinceReferenceDate
            let generationTiming =
                collectTurboQuantTiming ? TurboQuantTiming.snapshot() : nil
            let memoryEnd = Memory.snapshot()
            // Assertion 3 support: sequence-degeneracy signals (entropy / repetition
            // collapse). Cheap host-side pass over the captured ids.
            let distinctTokenRatio =
                Double(Set(generatedTokenIds).count)
                / Double(Swift.max(1, generatedTokenIds.count))
            var maxTokenRunLength = 0
            var currentRunLength = 0
            var previousTokenId: Int? = nil
            for tokenId in generatedTokenIds {
                if tokenId == previousTokenId {
                    currentRunLength += 1
                } else {
                    currentRunLength = 1
                    previousTokenId = tokenId
                }
                maxTokenRunLength = Swift.max(maxTokenRunLength, currentRunLength)
            }

            guard generated > 0 else {
                throw IntegrationTestFailure(
                    "no generated tokens (ctx=\(contextLength), \(config.label))")
            }
            let generationTime = synchronizationEnd - generationStart
            let promptTime = max(iterator.promptPrefillTime, Double.leastNonzeroMagnitude)
            let kvEstimate = memoryProfile.map {
                estimatedKVMemory(
                    profile: $0,
                    contextLength: contextLength,
                    config: config
                )
            }
            return Measurement(
                context: contextLength,
                label: config.label,
                sampleIndex: sampleIndex,
                sampleCount: max(1, sampleCount),
                decodeTokensPerSecond: Double(generated)
                    / max(generationTime, Double.leastNonzeroMagnitude),
                prefillTokensPerSecond: Double(contextLength) / promptTime,
                generationSeconds: generationTime,
                generationLoopSeconds: generationLoopEnd - generationStart,
                generationSynchronizationSeconds: synchronizationEnd - synchronizationStart,
                promptPrefillSeconds: promptTime,
                generationTokenCount: generated,
                promptPrefillTiming: promptPrefillTiming,
                generationTiming: generationTiming,
                attentionDiagnostics: iterator.turboQuantAttentionDiagnostics,
                cachePolicySummary: config.kvLayerPolicy?.summary()
                    ?? config.precisionPolicy.map {
                        "precisionPolicy(valueBits:\($0.resolvedValueBits.map(String.init) ?? "default"),boundary:\($0.boundary))"
                },
                valueBits: config.valueBits,
                valueGroupSize: config.valueGroupSize ?? TurboQuantKVCodec.affineK8V4ValueGroupSize,
                estimatedRawKVBytes: kvEstimate?.rawKVBytes,
                estimatedConfigKVBytes: kvEstimate?.configKVBytes,
                memoryStart: memoryStart,
                memoryEnd: memoryEnd,
                peakActiveMemoryBytes: max(memoryEnd.peakMemory, memoryEnd.activeMemory),
                dispatchedKernelCounts: dispatchedKernels,
                distinctTokenRatio: distinctTokenRatio,
                maxTokenRunLength: maxTokenRunLength
            )
        }
    }

    public struct KVMemoryEstimate: Equatable, Sendable {
        public var rawKVBytes: Int
        public var configKVBytes: Int

        public init(rawKVBytes: Int, configKVBytes: Int) {
            self.rawKVBytes = rawKVBytes
            self.configKVBytes = configKVBytes
        }
    }

    public static func estimatedKVMemoryForDiagnostics(
        profile: ModelMemoryProfile,
        contextLength: Int,
        config: CacheConfig
    ) -> KVMemoryEstimate {
        estimatedKVMemory(profile: profile, contextLength: contextLength, config: config)
    }

    private static func estimatedKVMemory(
        profile: ModelMemoryProfile,
        contextLength: Int,
        config: CacheConfig
    ) -> KVMemoryEstimate {
        let contextLength = max(0, contextLength)
        let rawKVBytes = profile.kvCacheBytes(contextLength: contextLength)
        let layerCount = max(0, profile.layerCount)
        guard layerCount > 0, contextLength > 0 else {
            return KVMemoryEstimate(rawKVBytes: rawKVBytes, configKVBytes: rawKVBytes)
        }

        let rawBytesPerTokenPerLayer = rawKVBytes / max(1, contextLength * layerCount)
        let baseCodec = estimatedBaseCodec(for: config)
        let configBytesPerTokenPerLayer: Int
        let quantizedKVStart = effectiveQuantizedKVStart(
            for: config,
            contextLength: contextLength
        )

        if let quantizedKVStart, contextLength < quantizedKVStart {
            return KVMemoryEstimate(rawKVBytes: rawKVBytes, configKVBytes: rawKVBytes)
        }

        if let kvLayerPolicy = config.kvLayerPolicy {
            configBytesPerTokenPerLayer = (0 ..< layerCount).reduce(0) { total, layerIndex in
                let codec = kvLayerPolicy.codec(forLayerIndex: layerIndex)
                return total + estimatedBytesPerTokenPerLayer(
                    codec: codec == .inherit ? baseCodec : codec,
                    profile: profile,
                    rawBytesPerTokenPerLayer: rawBytesPerTokenPerLayer
                )
            }
        } else if let precisionPolicy = config.precisionPolicy,
            config.strategy.createsAffineK8VxCacheImmediately
        {
            let protectedLayers = precisionPolicy.protectedBoundaryLayerIndexes(
                layerCount: layerCount
            )
            configBytesPerTokenPerLayer = (0 ..< layerCount).reduce(0) { total, layerIndex in
                if protectedLayers.contains(layerIndex) {
                    switch precisionPolicy.boundaryCachePrecision {
                    case .affineK8V4:
                        return total + estimatedBytesPerTokenPerLayer(
                            codec: .affineK8V4,
                            profile: profile,
                            rawBytesPerTokenPerLayer: rawBytesPerTokenPerLayer
                        )
                    case .raw:
                        return total + rawBytesPerTokenPerLayer
                    }
                }
                return total + estimatedBytesPerTokenPerLayer(
                    codec: baseCodec,
                    profile: profile,
                    rawBytesPerTokenPerLayer: rawBytesPerTokenPerLayer
                )
            }
        } else {
            configBytesPerTokenPerLayer =
                estimatedBytesPerTokenPerLayer(
                    codec: baseCodec,
                    profile: profile,
                    rawBytesPerTokenPerLayer: rawBytesPerTokenPerLayer
                ) * layerCount
        }

        return KVMemoryEstimate(
            rawKVBytes: rawKVBytes,
            configKVBytes: max(0, configBytesPerTokenPerLayer * contextLength)
        )
    }

    private static func effectiveQuantizedKVStart(
        for config: CacheConfig,
        contextLength: Int
    ) -> Int? {
        if config.strategy.createsAffineK8VxCacheImmediately,
            let override = affineQuantizedKVStartOverride(for: contextLength)
        {
            return override
        }
        return config.quantizedKVStart
    }

    private static func estimatedBaseCodec(for config: CacheConfig) -> KVLayerCodec {
        switch config.strategy {
        case .none:
            return .rawFP16
        case .mlxAffine:
            return .mlxAffine(bits: config.kvBits ?? 8, groupSize: config.kvGroupSize ?? 64)
        case .affineK8V4:
            return .affineK8V4
        case .affineK8Vx:
            return .affineK8Vx(
                valueBits: config.valueBits ?? TurboQuantKVCodec.affineK8V4ValueBits)
        case .affineInt4:
            return .affineInt4
        case .adaptiveTurboQuant, .hybridTurboQuant, .turboQuant:
            let backend =
                config.turboQuantBackend
                ?? (config.kvCodec == .polarWHT ? .metalPolarWHT : .metalPolarQJL)
            return .turboQuant(
                preset: config.preset ?? .turbo3_5,
                valueBits: config.valueBits,
                groupSize: config.kvGroupSize ?? 64,
                backend: backend
            )
        }
    }

    private static func estimatedBytesPerTokenPerLayer(
        codec: KVLayerCodec,
        profile: ModelMemoryProfile,
        rawBytesPerTokenPerLayer: Int
    ) -> Int {
        switch codec {
        case .inherit, .rawFP16:
            return rawBytesPerTokenPerLayer
        case .mlxAffine(let bits, let groupSize):
            return affineBytesPerTokenPerLayer(
                profile: profile,
                keyBits: bits,
                keyGroupSize: groupSize,
                valueBits: bits,
                valueGroupSize: groupSize
            )
        case .affineK8V4:
            return profile.affineK8V4BytesPerTokenPerLayer()
        case .affineK8Vx(let valueBits), .affineK8VxResidual(let valueBits, _):
            return profile.affineK8VxBytesPerTokenPerLayer(valueBits: valueBits)
        case .affineInt4:
            return affineBytesPerTokenPerLayer(
                profile: profile,
                keyBits: TurboQuantKVCodec.affineInt4Bits,
                keyGroupSize: TurboQuantKVCodec.affineInt4DefaultGroupSize,
                valueBits: TurboQuantKVCodec.affineInt4Bits,
                valueGroupSize: TurboQuantKVCodec.affineInt4DefaultGroupSize
            )
        case .turboQuant(let preset, let valueBits, let groupSize, let backend):
            if backend == .polarWHTReference || backend == .metalPolarWHT {
                return profile.polarWHTLayerCacheBytesPerTokenPerLayer(
                    preset: preset,
                    valueBits: valueBits,
                    groupSize: groupSize
                )
            }
            return profile.turboQuantLayerCacheFootprint(
                preset: preset,
                valueBits: valueBits,
                groupSize: groupSize
            ).bytesPerTokenPerLayer
        }
    }

    private static func affineBytesPerTokenPerLayer(
        profile: ModelMemoryProfile,
        keyBits: Int,
        keyGroupSize: Int,
        valueBits: Int,
        valueGroupSize: Int
    ) -> Int {
        func packedAffineBytesPerHead(groupSize: Int, bits: Int) -> Int {
            let groupSize = max(1, groupSize)
            let bits = max(1, bits)
            let groups = (max(1, profile.headDimension) + groupSize - 1) / groupSize
            let packedWords = (groupSize * bits + 31) / 32
            return groups * (
                packedWords * MemoryLayout<UInt32>.stride
                    + 2 * MemoryLayout<Float>.stride
            )
        }

        let keyBytes = packedAffineBytesPerHead(groupSize: keyGroupSize, bits: keyBits)
        let valueBytes = packedAffineBytesPerHead(groupSize: valueGroupSize, bits: valueBits)
        return max(1, profile.kvHeadCount) * (keyBytes + valueBytes)
    }

    private static func clearRuntimeMemory() {
        Stream.gpu.synchronize()
        Memory.clearCache()
    }

    private static func cooldown(seconds: Double) {
        clearRuntimeMemory()
        guard seconds > 0 else { return }
        let microseconds = min(seconds * 1_000_000, Double(UInt32.max))
        usleep(UInt32(microseconds.rounded()))
    }

    private static func prefillStepSize(for contextLength: Int) -> Int {
        let environment = ProcessInfo.processInfo.environment
        for key in [
            "TQ_INFERENCE_PARITY_PREFILL_STEP_SIZE",
            "TQ_V3_PREFILL_STEP_SIZE",
            "TQ_PREFILL_STEP_SIZE",
            "TURBOQUANT_PREFILL_STEP_SIZE",
        ] {
            if let raw = environment[key]?.trimmingCharacters(in: .whitespaces),
                let value = Int(raw),
                value > 0
            {
                return value
            }
        }

        return contextLength >= 32_768 ? 128 : 512
    }

    private static func affineQuantizedKVStartOverride(for contextLength: Int) -> Int? {
        let environment = ProcessInfo.processInfo.environment
        for key in [
            "TQ_INFERENCE_PARITY_AFFINE_QUANTIZED_KV_START",
            "TQ_V3_QUANTIZED_KV_START",
            "TQ_AFFINE_QUANTIZED_KV_START",
            "TURBOQUANT_AFFINE_QUANTIZED_KV_START",
        ] {
            if let raw = environment[key]?.trimmingCharacters(in: .whitespaces),
                let value = Int(raw),
                value >= 0
            {
                return value
            }
        }

        return nil
    }

    // Exposed at internal visibility (was `private`) so engagement/runtime-mode
    // tests can assert the config -> GenerateParameters mapping directly.
    static func configuredParameters(
        contextLength: Int,
        generateTokens: Int,
        config: CacheConfig
    ) -> GenerateParameters {
        let cacheContextLength = contextLength + max(generateTokens, 0) + 16
        var params = GenerateParameters(
            maxTokens: generateTokens,
            maxKVSize: cacheContextLength,
            kvCacheStrategy: config.strategy)
        if config.exactPrefill {
            params.kvCacheStrategy = .adaptiveTurboQuant
            params.turboQuantExactPrefill = true
        }
        // Chunk prefill so no single command buffer trips the macOS GPU watchdog
        // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) at long context.
        params.prefillStepSize = prefillStepSize(for: contextLength)
        params.turboQuantRequestedContextLength = cacheContextLength
        params.turboQuantPromptTokenCount = contextLength
        if let kvBits = config.kvBits {
            params.kvBits = kvBits
        }
        if let kvGroupSize = config.kvGroupSize {
            params.kvGroupSize = kvGroupSize
        }
        if let kvCodec = config.kvCodec {
            params.kvCodec = kvCodec
        }
        if let turboQuantBackend = config.turboQuantBackend {
            params.turboQuantBackend = turboQuantBackend
        }
        if let valueBits = config.valueBits {
            params.turboQuantValueBits = valueBits
        }
        if let valueGroupSize = config.valueGroupSize {
            params.turboQuantValueGroupSize = valueGroupSize
        }
        if let runtimeMode = config.runtimeMode {
            params.turboQuantRuntimeMode = runtimeMode
        } else if config.strategy == .turboQuant || config.strategy == .hybridTurboQuant {
            // Compressed turboQuant schemes must exercise the segmented compressed-decode
            // path, not the throughput single-pass bypass. nil runtimeMode -> .auto ->
            // (throughputFits on a Mac) -> ThroughputTurboQuantKVCache (baseline + rawFallback),
            // so the compressed segmented kernel (where coop lives) never dispatches. Pin
            // capacity so the path-under-test actually runs.
            params.turboQuantRuntimeMode = .capacityTurboQuant
        }
        if let quantizedKVStart = config.quantizedKVStart {
            params.quantizedKVStart = quantizedKVStart
        }
        if config.strategy.createsAffineK8VxCacheImmediately,
            let quantizedKVStart = affineQuantizedKVStartOverride(for: contextLength)
        {
            params.quantizedKVStart = quantizedKVStart
        }
        if let precisionPolicy = config.precisionPolicy {
            params.turboQuantPrecisionPolicy = precisionPolicy
        }
        if let kvLayerPolicy = config.kvLayerPolicy {
            params.kvLayerPolicy = kvLayerPolicy
        }
        params.turboQuantSparseValueSelection = config.sparseValueSelection
        if let threshold = config.sparseValueSelection.resolvedThreshold {
            params.turboQuantSparseValuePolicy = .force(threshold: threshold)
        } else if config.sparseValueSelection.isEnabled {
            params.turboQuantSparseValuePolicy = .off
        }
        if let preset = config.preset {
            params.turboQuantPreset = preset
        }
        return params
    }

    private static func qualityReport(
        candidate: QualityLogits,
        reference: QualityLogits
    ) -> TurboQuantQualityGateReport {
        let rowWidth = candidate.rowWidth
        guard candidate.values.count == reference.values.count,
            candidate.rowWidth == reference.rowWidth,
            rowWidth > 0,
            !candidate.values.isEmpty
        else {
            return .failed(
                benchmarkSuiteID: .realModelInferenceV1,
                reason: "candidate and reference quality shapes do not match")
        }

        let noNaNOrInf =
            candidate.values.allSatisfy(\.isFinite) && reference.values.allSatisfy(\.isFinite)
        let top1 = rowTop1MatchRate(
            candidate: candidate.values,
            reference: reference.values,
            rowWidth: rowWidth
        )
        let kl = meanKLDivergence(
            candidate: candidate.values,
            reference: reference.values,
            rowWidth: rowWidth
        )
        let p95 = percentile(
            rowMaxAbsErrors(
                candidate: candidate.values,
                reference: reference.values,
                rowWidth: rowWidth
            ),
            percentile: 0.95
        )
        let cosine = meanCosineSimilarity(
            candidate: candidate.values,
            reference: reference.values,
            rowWidth: rowWidth
        )
        return .evaluated(
            benchmarkSuiteID: .realModelInferenceV1,
            deterministicTop1MatchRate: top1,
            logitKLDivergenceMean: kl,
            logitMaxAbsErrorP95: p95,
            attentionOutputCosineMean: cosine,
            noNaNOrInf: noNaNOrInf,
            fallbackEquivalent: true,
            prefillExact: true,
            snapshotRoundtripEquivalent: nil,
            top1Threshold: 1.0,
            klThreshold: 0.10,
            p95MaxAbsErrorThreshold: 2.0
        )
    }

    private static func rowTop1MatchRate(
        candidate: [Float],
        reference: [Float],
        rowWidth: Int
    ) -> Double {
        let rowCount = min(candidate.count, reference.count) / rowWidth
        guard rowCount > 0 else { return 0 }
        var matches = 0
        for row in 0 ..< rowCount {
            let start = row * rowWidth
            let end = start + rowWidth
            if argmax(candidate[start ..< end]) == argmax(reference[start ..< end]) {
                matches += 1
            }
        }
        return Double(matches) / Double(rowCount)
    }

    private static func rowMaxAbsErrors(
        candidate: [Float],
        reference: [Float],
        rowWidth: Int
    ) -> [Double] {
        let rowCount = min(candidate.count, reference.count) / rowWidth
        guard rowCount > 0 else { return [] }
        return (0 ..< rowCount).map { row in
            let start = row * rowWidth
            let end = start + rowWidth
            return zip(candidate[start ..< end], reference[start ..< end]).reduce(0.0) {
                max($0, abs(Double($1.0) - Double($1.1)))
            }
        }
    }

    private static func meanKLDivergence(
        candidate: [Float],
        reference: [Float],
        rowWidth: Int
    ) -> Double {
        let rowCount = min(candidate.count, reference.count) / rowWidth
        guard rowCount > 0 else { return Double.greatestFiniteMagnitude }
        var total = 0.0
        for row in 0 ..< rowCount {
            let start = row * rowWidth
            let end = start + rowWidth
            let candidateLogProb = logSoftmax(candidate[start ..< end])
            let referenceLogProb = logSoftmax(reference[start ..< end])
            total += zip(referenceLogProb, candidateLogProb).reduce(0.0) {
                $0 + exp($1.0) * ($1.0 - $1.1)
            }
        }
        return total / Double(rowCount)
    }

    private static func meanCosineSimilarity(
        candidate: [Float],
        reference: [Float],
        rowWidth: Int
    ) -> Double {
        let rowCount = min(candidate.count, reference.count) / rowWidth
        guard rowCount > 0 else { return 0 }
        var total = 0.0
        for row in 0 ..< rowCount {
            let start = row * rowWidth
            let end = start + rowWidth
            var dot = 0.0
            var candidateNorm = 0.0
            var referenceNorm = 0.0
            for (candidateValue, referenceValue) in zip(candidate[start ..< end], reference[start ..< end]) {
                let c = Double(candidateValue)
                let r = Double(referenceValue)
                dot += c * r
                candidateNorm += c * c
                referenceNorm += r * r
            }
            let denominator = sqrt(candidateNorm) * sqrt(referenceNorm)
            total += denominator > 0 ? dot / denominator : 0
        }
        return total / Double(rowCount)
    }

    private static func logSoftmax(_ row: ArraySlice<Float>) -> [Double] {
        guard let maxValue = row.max() else { return [] }
        let shifted = row.map { exp(Double($0 - maxValue)) }
        let denominator = log(shifted.reduce(0.0, +)) + Double(maxValue)
        return row.map { Double($0) - denominator }
    }

    private static func percentile(_ values: [Double], percentile: Double) -> Double {
        guard !values.isEmpty else { return Double.greatestFiniteMagnitude }
        let sorted = values.sorted()
        let index = min(
            sorted.count - 1,
            max(0, Int(ceil(min(max(percentile, 0), 1) * Double(sorted.count))) - 1)
        )
        return sorted[index]
    }

    private static func argmax(_ values: ArraySlice<Float>) -> Int {
        var bestIndex = values.startIndex
        var bestValue = -Float.greatestFiniteMagnitude
        for index in values.indices where values[index] > bestValue {
            bestValue = values[index]
            bestIndex = index
        }
        return bestIndex - values.startIndex
    }

    // MARK: - Tiny formatting helpers (avoid String(format:) %@ width quirks)

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
    private static func round1(_ v: Double) -> String { String(format: "%.1f", v) }
    private static func round2(_ v: Double) -> String { String(format: "%.2f", v) }

    private static func normalizedSparseSelectionMode(_ raw: String) -> String {
        raw.lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
    }

    private static func sparseFloatLabel(_ value: Double) -> String {
        String(format: "%.0e", value)
            .replacingOccurrences(of: ".", with: "")
            .replacingOccurrences(of: "+", with: "")
            .replacingOccurrences(of: "e-0", with: "e-")
            .replacingOccurrences(of: "e0", with: "e")
    }

    private static func sparseMassLabel(_ mass: Float) -> String {
        let percent = Double(mass) * 1000
        return "\(Int(percent.rounded()))"
    }

    private static func diagnosticSummary(
        _ diagnostics: [TurboQuantAttentionDiagnostics]
    ) -> String? {
        guard !diagnostics.isEmpty else { return nil }
        let pathCounts = Dictionary(grouping: diagnostics, by: \.activeAttentionPath.rawValue)
            .map { "\($0.key)=\($0.value.count)" }
            .sorted()
        let skipped = diagnostics.compactMap(\.sparseVSkippedTokens).reduce(0, +)
        let total = diagnostics.compactMap(\.sparseVTotalTokens).reduce(0, +)
        var parts = ["paths[\(pathCounts.joined(separator: ","))]"]
        let requestedLayers = diagnostics.filter { $0.sparseVEnabled }.count
        let activeLayers = diagnostics.filter { $0.sparseVActive == true }.count
        if requestedLayers > 0 {
            parts.append("sparseLayers=\(activeLayers)/\(requestedLayers) active")
        }
        if total > 0 {
            parts.append(String(format: "sparseSkipped=%d/%d %.2f%%", skipped, total, Double(skipped) / Double(total) * 100))
        }
        let fallbackReasons = diagnostics.compactMap { diag -> String? in
            diag.lastFallback?.reason ?? diag.lastUnsupportedShape ?? diag.nativeFallbackReason
        }
        if let first = fallbackReasons.first {
            parts.append("fallback=\(first)")
        }
        let sparseLayers = diagnostics.filter { $0.sparseVEnabled }.compactMap { diag -> String? in
            guard let layer = diag.layerIndex else { return nil }
            let ratio = diag.sparseVSkipRatio.map { String(format: "%.1f%%", $0 * 100) } ?? "--"
            return "L\(layer):\(ratio)"
        }
        if !sparseLayers.isEmpty {
            parts.append("layers[\(sparseLayers.prefix(8).joined(separator: ","))\(sparseLayers.count > 8 ? ",+\(sparseLayers.count - 8)" : "")]")
        }
        return parts.joined(separator: " ")
    }
}
