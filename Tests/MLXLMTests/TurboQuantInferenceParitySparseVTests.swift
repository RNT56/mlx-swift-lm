import Foundation
import MLX
import Testing
@testable import IntegrationTestHelpers
@testable import MLXLMCommon

@Suite("TurboQuant inference parity Sparse-V overrides")
struct TurboQuantInferenceParitySparseVTests {
    @Test func defaultConfigCatalogCoversBenchmarkableOptimizationFamilies() {
        let labels = Set(InferenceParityBenchmark.defaultConfigLabels)

        for required in [
            "fp16",
            "affineK8V4",
            "affineK8V3",
            "affineK8V2",
            "affineK8V3-optimized",
            "affineK8V2-residual-r1",
            "mlxAffine-q8",
            "affineInt4",
            "turbo3_5",
            "turbo4v2",
            "polarWHTV3",
            "polarWHTReferenceV3",
            "hybridK8PolarWHTV3",
            "hybridK8PolarWHTV4",
            "hybridK8PolarWHTV3Reference",
            "turbo8",
            "turbo4v2SparseThreshold1e-4",
            "turbo4v2SparseTopK128",
            "turbo4v2SparseMass995",
            "turbo4v2SparseHybrid995TopK256",
        ] {
            #expect(labels.contains(required), "missing benchmark config \(required)")
        }
    }

    @Test func kvMemoryEstimateRespectsAffineQuantizedStart() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "affineK8V4"))
        let profile = ModelMemoryProfile(
            modelID: "test-qwen",
            modelType: "qwen3",
            layerCount: 28,
            hiddenSize: 1024,
            attentionHeadCount: 16,
            kvHeadCount: 8,
            headDimension: 128
        )

        let belowThreshold = InferenceParityBenchmark.estimatedKVMemoryForDiagnostics(
            profile: profile,
            contextLength: 4_096,
            config: config
        )
        #expect(belowThreshold.configKVBytes == belowThreshold.rawKVBytes)

        let atThreshold = InferenceParityBenchmark.estimatedKVMemoryForDiagnostics(
            profile: profile,
            contextLength: InferenceParityBenchmark.affineThroughputQuantizedKVStart,
            config: config
        )
        #expect(atThreshold.configKVBytes < atThreshold.rawKVBytes)
    }

    @Test func topKOverrideMatchesPredefinedSparseTopKCandidate() throws {
        let configs = try #require(
            InferenceParityBenchmark.configs(fromCSV: "affineK8V4,turbo4v2")
        )
        let override = try #require(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "topK",
                topK: 128
            )
        )
        let overridden = try #require(
            InferenceParityBenchmark.applyingSparseValueSelectionOverride(
                override,
                to: configs
            )
        )
        let predefined = try #require(
            InferenceParityBenchmark.config(named: "turbo4v2SparseTopK128")
        )

        #expect(overridden.count == 2)
        #expect(overridden[0].label == "affineK8V4")
        #expect(overridden[0].sparseValueSelection == .off)
        #expect(overridden[1].label == "turbo4v2SparseTopK128")
        #expect(overridden[1].strategy == predefined.strategy)
        #expect(overridden[1].preset == predefined.preset)
        #expect(overridden[1].runtimeMode == predefined.runtimeMode)
        #expect(overridden[1].precisionPolicy == predefined.precisionPolicy)
        #expect(overridden[1].sparseValueSelection == predefined.sparseValueSelection)
    }

    @Test func sparseOverrideParserAcceptsPlanModes() throws {
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "threshold",
                threshold: 1e-5
            ) == .threshold(1e-5)
        )
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "cumulativeMass",
                cumulativeMass: 0.995
            ) == .cumulativeMass(0.995)
        )
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "hybridCumulativeMassTopK",
                cumulativeMass: 0.995,
                maxTopK: 512
            ) == .hybrid(cumulativeMass: 0.995, maxTopK: 512)
        )
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "blockThreshold",
                threshold: 0.04
            ) == .blockThreshold(0.04)
        )
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "pageTopK",
                topK: 4
            ) == .pageTopK(4)
        )
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "candidateSparse",
                topK: 128,
                recentTokens: 256,
                candidatePages: 4
            ) == .candidateSparse(
                recentTokens: 256,
                candidatePages: 4,
                olderTokenBudget: 128
            )
        )
        #expect(
            InferenceParityBenchmark.sparseValueSelection(
                mode: "candidate-sparse",
                topK: 64,
                recentTokens: 128,
                candidatePages: 2
            ) == .candidateSparse(
                recentTokens: 128,
                candidatePages: 2,
                olderTokenBudget: 64
            )
        )
    }

    @Test func sparseRequestedButInactiveSeparatesFallbackFromMeasuredSparse() {
        let inactive = measurementWithSparseDiagnostics(sparseVActive: false)
        let active = measurementWithSparseDiagnostics(sparseVActive: true)

        #expect(inactive.sparseRequestedLayerCount == 1)
        #expect(inactive.sparseActiveLayerCount == 0)
        #expect(inactive.sparseRequestedButInactive)
        #expect(active.sparseRequestedLayerCount == 1)
        #expect(active.sparseActiveLayerCount == 1)
        #expect(!active.sparseRequestedButInactive)
    }

    @Test func promotionGateBlocksUnavailableMetalPolarWHT() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "hybridK8PolarWHTV3"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "hybridK8PolarWHTV3",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: false,
                    path: .mlxPackedFallback,
                    fallbackReason: "PolarWHT Metal kernels unavailable"
                )
            ]
        )

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: nil,
            runQualityGates: false
        )

        #expect(!gate.promotionEligible)
        #expect(gate.requiresMetalPolarWHT)
        #expect(gate.metalPolarWHTAvailable == false)
        #expect(gate.selectedAttentionPaths == [TurboQuantAttentionPath.mlxPackedFallback.rawValue])
        #expect(gate.promotionBlockReasons.contains("metalPolarWHT unavailable"))
        #expect(
            gate.promotionBlockReasons.contains {
                $0.contains("hybrid K8+PolarWHT-V is experimental")
            }
        )
        #expect(gate.promotionBlockReasons.contains("hybrid K8+PolarWHT-V native path was not selected"))
        #expect(gate.promotionBlockReasons.contains("decoded or unavailable fallback path active"))
        #expect(gate.promotionBlockReasons.contains("quality gate not run"))
        #expect(gate.fallbackReasons.contains("PolarWHT Metal kernels unavailable"))
    }

    @Test func promotionGateTreatsFullPolarWHTAsDiagnosticOnly() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "polarWHTV3"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "polarWHTV3",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: true,
                    path: .metalPolarWHTHybrid
                )
            ]
        )
        let quality = passingQuality(label: "polarWHTV3")

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: quality,
            runQualityGates: true,
            fp16Baseline: measurement
        )

        #expect(!gate.promotionEligible)
        #expect(gate.qualityPassed == true)
        #expect(gate.promotionBlockReasons.contains("full PolarWHT K/V path is diagnostic only"))
    }

    @Test func promotionGateTreatsPolarWHTReferenceAsMeasuredOnly() throws {
        let config = try #require(
            InferenceParityBenchmark.config(named: "polarWHTReferenceV3")
        )
        let measurement = measurementWithAttentionDiagnostics(
            label: "polarWHTReferenceV3",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: false,
                    path: .polarWHTReferenceHybrid
                )
            ]
        )
        let quality = passingQuality(label: "polarWHTReferenceV3")

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: quality,
            runQualityGates: true,
            fp16Baseline: measurement,
            affineK8V4Baseline: measurement
        )

        #expect(!gate.promotionEligible)
        #expect(gate.qualityPassed == true)
        #expect(gate.metalPolarWHTAvailable == nil)
        #expect(gate.selectedAttentionPaths == [
            TurboQuantAttentionPath.polarWHTReferenceHybrid.rawValue
        ])
        #expect(
            gate.promotionBlockReasons.contains(
                "polarWHTReference is a measured reference path, not a native metalPolarWHT promotion"
            )
        )
        #expect(!gate.promotionBlockReasons.contains("quality gate missing"))
    }

    @Test func promotionGateAllowsNativeCompressedPathWithPassingQuality() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "affineK8V4"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "affineK8V4",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: true,
                    path: .affineK8V4Native
                )
            ]
        )
        let quality = passingQuality(label: "affineK8V4")

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: quality,
            runQualityGates: true,
            fp16Baseline: measurement
        )

        #expect(gate.promotionEligible)
        #expect(gate.promotionBlockReasons.isEmpty)
        #expect(gate.qualityPassed == true)
        #expect(gate.rawFallbackAllocated == false)
        #expect(gate.decodedFallbackPathActive == false)
    }

    @Test func promotionGateBlocksAffineWhenQualityDidNotUseNativeCompressedPath() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "affineK8V4"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "affineK8V4",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: true,
                    path: .affineK8V4Native
                )
            ]
        )
        let rawQuality = InferenceParityBenchmark.QualityMeasurement(
            context: 16_384,
            label: "affineK8V4",
            referenceLabel: "fp16",
            candidateFirst: false,
            quality: TurboQuantQualityGateReport.evaluated(
                benchmarkSuiteID: .realModelInferenceV1,
                deterministicTop1MatchRate: 1,
                logitKLDivergenceMean: 0,
                logitMaxAbsErrorP95: 0,
                noNaNOrInf: true,
                fallbackEquivalent: true,
                prefillExact: true
            )
        )

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: rawQuality,
            runQualityGates: true,
            fp16Baseline: measurement
        )

        #expect(!gate.promotionEligible)
        #expect(
            gate.promotionBlockReasons.contains(
                "quality gate did not select affine compressed native path"
            )
        )
        #expect(gate.qualityPassed == true)
        #expect(gate.qualitySelectedAttentionPaths.isEmpty)
    }

    @Test func promotionGateBlocksAffineWhenNativeCompressedPathIsNotSelected() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "affineK8V4"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "affineK8V4",
            diagnostics: []
        )
        let quality = passingQuality(label: "affineK8V4")

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: quality,
            runQualityGates: true,
            fp16Baseline: measurement
        )

        #expect(!gate.promotionEligible)
        #expect(
            gate.promotionBlockReasons.contains("affine compressed native path was not selected")
        )
        #expect(gate.qualityPassed == true)
    }

    @Test func promotionGateTreatsHybridK8PolarWHTValueAsExperimentalByDefault() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "hybridK8PolarWHTV3"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "hybridK8PolarWHTV3",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: true,
                    path: .metalHybridK8PolarWHTValue
                )
            ]
        )
        let quality = passingQuality(label: "hybridK8PolarWHTV3")

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: quality,
            runQualityGates: true,
            fp16Baseline: measurement,
            affineK8V4Baseline: measurement
        )

        #expect(!gate.promotionEligible)
        #expect(
            gate.promotionBlockReasons.contains {
                $0.contains("hybrid K8+PolarWHT-V is experimental")
            }
        )
        #expect(gate.selectedAttentionPaths == [
            TurboQuantAttentionPath.metalHybridK8PolarWHTValue.rawValue
        ])
        #expect(gate.qualityPassed == true)
    }

    @Test func promotionGateBlocksHybridWhenSpeedRegressesAgainstBaselines() throws {
        let config = try #require(InferenceParityBenchmark.config(named: "hybridK8PolarWHTV3"))
        let measurement = measurementWithAttentionDiagnostics(
            label: "hybridK8PolarWHTV3",
            diagnostics: [
                attentionDiagnostics(
                    metalAvailable: true,
                    path: .metalHybridK8PolarWHTValue
                )
            ],
            decodeTokensPerSecond: 1
        )
        let fp16 = measurementWithAttentionDiagnostics(
            label: "fp16",
            diagnostics: [attentionDiagnostics(metalAvailable: true, path: .baseline)],
            decodeTokensPerSecond: 10
        )
        let affine = measurementWithAttentionDiagnostics(
            label: "affineK8V4",
            diagnostics: [attentionDiagnostics(metalAvailable: true, path: .affineK8V4Native)],
            decodeTokensPerSecond: 10
        )
        let quality = passingQuality(label: "hybridK8PolarWHTV3")

        let gate = InferenceParityBenchmark.promotionGate(
            measurement: measurement,
            config: config,
            quality: quality,
            runQualityGates: true,
            fp16Baseline: fp16,
            affineK8V4Baseline: affine
        )

        #expect(!gate.promotionEligible)
        #expect(gate.speedRatioToFP16 == 0.1)
        #expect(gate.speedRatioToAffineK8V4 == 0.1)
        #expect(
            gate.promotionBlockReasons.contains {
                $0.contains("decode speed 0.100x FP16 is below 0.70x promotion floor")
            }
        )
        #expect(
            gate.promotionBlockReasons.contains {
                $0.contains("decode speed 0.100x affineK8V4 is below 0.70x hybrid floor")
            }
        )
    }

    private func measurementWithSparseDiagnostics(
        sparseVActive: Bool
    ) -> InferenceParityBenchmark.Measurement {
        let snapshot = Memory.snapshot()
        return InferenceParityBenchmark.Measurement(
            context: 16_384,
            label: "turbo4v2SparseTopK256",
            sampleIndex: 1,
            sampleCount: 1,
            decodeTokensPerSecond: 10,
            prefillTokensPerSecond: 70,
            generationSeconds: 1.6,
            promptPrefillSeconds: 230,
            generationTokenCount: 16,
            attentionDiagnostics: [
                TurboQuantAttentionDiagnostics(
                    metalAttentionAvailable: true,
                    activeAttentionPath: .nativeMLXCompressed,
                    selectedKernelProfile: .macAppleSilicon,
                    selfTestStatus: .passed,
                    selfTestFailureReason: nil,
                    optimizationPolicy: .preferMemory,
                    fallbackReason: sparseVActive ? nil : "dense compressed fallback",
                    lastUnsupportedShape: nil,
                    rawFallbackAllocated: false,
                    sparseVEnabled: true,
                    sparseVSelectionMode: .topK,
                    sparseVTopK: 256,
                    sparseVSkippedTokens: sparseVActive ? 1024 : 0,
                    sparseVTotalTokens: sparseVActive ? 2048 : 0,
                    sparseVActive: sparseVActive,
                    sparseVSkipRatio: sparseVActive ? 0.5 : nil
                )
            ],
            cachePolicySummary: nil,
            valueBits: 4,
            valueGroupSize: 64,
            estimatedRawKVBytes: 1_024,
            estimatedConfigKVBytes: 512,
            memoryStart: snapshot,
            memoryEnd: snapshot,
            peakActiveMemoryBytes: snapshot.peakMemory
        )
    }

    private func measurementWithAttentionDiagnostics(
        label: String,
        diagnostics: [TurboQuantAttentionDiagnostics],
        decodeTokensPerSecond: Double = 10
    ) -> InferenceParityBenchmark.Measurement {
        let snapshot = Memory.snapshot()
        return InferenceParityBenchmark.Measurement(
            context: 16_384,
            label: label,
            sampleIndex: 1,
            sampleCount: 1,
            decodeTokensPerSecond: decodeTokensPerSecond,
            prefillTokensPerSecond: 70,
            generationSeconds: 1.6,
            promptPrefillSeconds: 230,
            generationTokenCount: 16,
            attentionDiagnostics: diagnostics,
            cachePolicySummary: nil,
            valueBits: 4,
            valueGroupSize: 64,
            estimatedRawKVBytes: 1_024,
            estimatedConfigKVBytes: 512,
            memoryStart: snapshot,
            memoryEnd: snapshot,
            peakActiveMemoryBytes: snapshot.peakMemory
        )
    }

    private func attentionDiagnostics(
        metalAvailable: Bool,
        path: TurboQuantAttentionPath,
        fallbackReason: String? = nil,
        rawFallbackAllocated: Bool = false
    ) -> TurboQuantAttentionDiagnostics {
        TurboQuantAttentionDiagnostics(
            metalAttentionAvailable: metalAvailable,
            activeAttentionPath: path,
            selectedKernelProfile: .macAppleSilicon,
            selfTestStatus: .passed,
            selfTestFailureReason: nil,
            optimizationPolicy: .preferMemory,
            fallbackReason: fallbackReason,
            lastUnsupportedShape: nil,
            rawFallbackAllocated: rawFallbackAllocated
        )
    }

    private func passingQuality(
        label: String,
        context: Int = 16_384
    ) -> InferenceParityBenchmark.QualityMeasurement {
        let path: TurboQuantAttentionPath? = {
            switch label {
            case "affineK8V4":
                .affineK8V4Native
            case "hybridK8PolarWHTV3", "hybridK8PolarWHTV4":
                .metalHybridK8PolarWHTValue
            case "polarWHTV3":
                .metalPolarWHTHybrid
            case "polarWHTReferenceV3":
                .polarWHTReferenceHybrid
            default:
                nil
            }
        }()
        return InferenceParityBenchmark.QualityMeasurement(
            context: context,
            label: label,
            referenceLabel: "fp16",
            candidateFirst: false,
            quality: TurboQuantQualityGateReport.evaluated(
                benchmarkSuiteID: .realModelInferenceV1,
                deterministicTop1MatchRate: 1,
                logitKLDivergenceMean: 0,
                logitMaxAbsErrorP95: 0,
                noNaNOrInf: true,
                fallbackEquivalent: true,
                prefillExact: true
            ),
            attentionDiagnostics: path.map {
                [
                    attentionDiagnostics(
                        metalAvailable: true,
                        path: $0
                    )
                ]
            } ?? []
        )
    }

    @Test func lowerV3FrontierConfigAliasesResolve() throws {
        let configs = try #require(
            InferenceParityBenchmark.configs(
                fromCSV: "k8v3_last2,k8v3_first1_last2,k8v3_first1_penultimate"
            )
        )

        #expect(configs.map(\.label) == [
            "affineK8V3-last2",
            "affineK8V3-first1-last2",
            "affineK8V3-first1-penultimate",
        ])
        #expect(configs.allSatisfy { $0.valueBits == 3 })
    }

    @Test func strictConfigParsingRejectsUnknownEntries() throws {
        #expect(
            throws: InferenceParityBenchmark.ConfigParseError.unknownConfigs(
                ["not-a-config"],
                known: InferenceParityBenchmark.defaultConfigLabels
            )
        ) {
            _ = try InferenceParityBenchmark.configs(
                fromCSV: "affineK8V4,not-a-config",
                strict: true
            )
        }

        let tolerant = InferenceParityBenchmark.configs(fromCSV: "affineK8V4,not-a-config")
        #expect(tolerant?.map(\.label) == ["affineK8V4"])
    }

    @Test func dumpedQualityLogitsCanBeComparedWithoutModel() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("turboquant-quality-logits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let referenceURL = directory.appendingPathComponent("reference.json")
        let candidateURL = directory.appendingPathComponent("candidate.json")
        let reference = InferenceParityBenchmark.QualityLogitArtifact(
            context: 4,
            label: "affineK8V4",
            logits: InferenceParityBenchmark.QualityLogits(
                values: [1.0, 2.0, 3.0, 4.0],
                rowWidth: 4
            )
        )
        let candidate = InferenceParityBenchmark.QualityLogitArtifact(
            context: 4,
            label: "affineK8V3-optimized",
            logits: InferenceParityBenchmark.QualityLogits(
                values: [1.0, 2.0, 3.0, 4.0],
                rowWidth: 4
            )
        )

        let encoder = JSONEncoder()
        try encoder.encode(reference).write(to: referenceURL, options: .atomic)
        try encoder.encode(candidate).write(to: candidateURL, options: .atomic)

        let measurement = try InferenceParityBenchmark.qualityMeasurement(
            referenceLogitsPath: referenceURL.path,
            candidateLogitsPath: candidateURL.path
        )

        #expect(measurement.context == 4)
        #expect(measurement.label == "affineK8V3-optimized")
        #expect(measurement.quality.passed)
        #expect(measurement.quality.deterministicTop1MatchRate == 1.0)
        #expect(measurement.quality.logitMaxAbsErrorP95 == 0.0)
    }

    @Test func dumpedQualityLogitsRejectMismatchedArtifactIdentity() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("turboquant-quality-logits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let referenceURL = directory.appendingPathComponent("reference.json")
        let candidateURL = directory.appendingPathComponent("candidate.json")
        let reference = InferenceParityBenchmark.QualityLogitArtifact(
            context: 4,
            label: "affineK8V4",
            logits: InferenceParityBenchmark.QualityLogits(
                values: [1.0, 2.0, 3.0, 4.0],
                rowWidth: 4
            )
        )
        let candidate = InferenceParityBenchmark.QualityLogitArtifact(
            context: 5,
            label: "affineK8V3-optimized",
            logits: InferenceParityBenchmark.QualityLogits(
                values: [1.0, 2.0, 3.0, 4.0],
                rowWidth: 4
            )
        )

        let encoder = JSONEncoder()
        try encoder.encode(reference).write(to: referenceURL, options: .atomic)
        try encoder.encode(candidate).write(to: candidateURL, options: .atomic)

        #expect(
            throws: InferenceParityBenchmark.QualityLogitComparisonError.artifactMismatch(
                "Quality logit artifacts do not match: context 4 != 5."
            )
        ) {
            _ = try InferenceParityBenchmark.qualityMeasurement(
                referenceLogitsPath: referenceURL.path,
                candidateLogitsPath: candidateURL.path
            )
        }
    }
}
