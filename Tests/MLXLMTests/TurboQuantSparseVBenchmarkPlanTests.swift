import Foundation
import Testing
import TurboQuantBench

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {
    @Suite struct TurboQuantSparseVBenchmarkPlanSuite {
        @Test func sparseSelectionThresholdUsesNormalizedWeights() {
            let selection = TurboQuantSparseVProof.select(
                normalizedWeights: [0.70, 0.20, 0.09, 0.01],
                mode: .threshold(0.10)
            )

            #expect(selection.retainedIndexes == [0, 1])
            #expect(selection.consideredValueTokens == 4)
            #expect(selection.skippedValueTokens == 2)
            #expect(abs(selection.retainedAttentionMass - 0.90) < 1e-9)
        }

        @Test func sparseSelectionTopKAndCumulativeMassUseLargestSoftmaxWeights() {
            let weights = [0.04, 0.50, 0.16, 0.30]

            let topK = TurboQuantSparseVProof.select(
                normalizedWeights: weights,
                mode: .topK(2)
            )
            #expect(topK.retainedIndexes == [1, 3])
            #expect(abs(topK.retainedAttentionMass - 0.80) < 1e-9)

            let mass = TurboQuantSparseVProof.select(
                normalizedWeights: weights,
                mode: .cumulativeMass(0.75)
            )
            #expect(mass.retainedIndexes == [1, 3])
            #expect(abs(mass.retainedAttentionMass - 0.80) < 1e-9)
        }

        @Test func candidateSparseSelectionReportsRecentOlderAndPageCounts() {
            let selection = TurboQuantSparseVProof.select(
                normalizedWeights: [0.10, 0.30, 0.20, 0.40, 0.0],
                mode: .candidateSparse(
                    recentTokens: 2,
                    candidatePages: 3,
                    olderTokenBudget: 1
                )
            )

            #expect(selection.retainedIndexes == [1, 3, 4])
            #expect(selection.recentTokenCount == 2)
            #expect(selection.olderTokenCount == 1)
            #expect(selection.pageCandidateCount == 3)
            #expect(selection.skippedValueTokens == 2)
        }

        @Test func sparseSelectionHybridReportsCapFallbackWhenMassFloorCannotBeMet() {
            let diagnostics = TurboQuantSparseVProof.diagnostics(
                normalizedWeights: [0.40, 0.30, 0.20, 0.10],
                mode: .hybrid(cumulativeMass: 0.95, maxTopK: 2),
                maxOutputError: 0.012,
                cosineSimilarity: 0.999
            )

            #expect(diagnostics.skippedValueTokens == 2)
            #expect(abs(diagnostics.retainedAttentionMass - 0.70) < 1e-9)
            #expect(diagnostics.maxOutputError == 0.012)
            #expect(diagnostics.cosineSimilarity == 0.999)
            #expect(diagnostics.fallbackReason?.contains("below requested floor") == true)
        }

        @Test func sparseProofDiagnosticsRoundTripPerLayerHeadRecords() throws {
            let diagnostics = TurboQuantSparseVProofDiagnostics(
                selectionMode: .cumulativeMass(0.995),
                skippedValueTokens: 6,
                consideredValueTokens: 512,
                retainedAttentionMass: 0.996,
                maxOutputError: 0.001,
                cosineSimilarity: 0.99999,
                fallbackReason: "native cumulative-mass Sparse-V unavailable",
                perLayerHeadDiagnostics: [
                    TurboQuantSparseVLayerHeadDiagnostics(
                        layerIndex: 3,
                        headIndex: 7,
                        skippedValueTokens: 6,
                        consideredValueTokens: 512,
                        recentTokenCount: 256,
                        olderTokenCount: 128,
                        pageCandidateCount: 4,
                        retainedAttentionMass: 0.996,
                        maxOutputError: 0.001,
                        cosineSimilarity: 0.99999
                    )
                ]
            )

            let decoded = try JSONDecoder().decode(
                TurboQuantSparseVProofDiagnostics.self,
                from: try JSONEncoder().encode(diagnostics)
            )

            #expect(decoded.selectionMode == .cumulativeMass(0.995))
            #expect(decoded.selectionMode.canonicalKind == "cumulativeMass")
            #expect(decoded.skipRatio == 6.0 / 512.0)
            #expect(decoded.perLayerHeadDiagnostics.first?.layerIndex == 3)
            #expect(decoded.perLayerHeadDiagnostics.first?.headIndex == 7)
            #expect(decoded.perLayerHeadDiagnostics.first?.recentTokenCount == 256)
            #expect(decoded.perLayerHeadDiagnostics.first?.olderTokenCount == 128)
            #expect(decoded.perLayerHeadDiagnostics.first?.pageCandidateCount == 4)
        }

        @Test func sparseSelectionModesEmitCanonicalNamesButDecodeAliases() throws {
            let encoded = try JSONEncoder().encode(
                TurboQuantSparseVProofDiagnostics(
                    selectionMode: .candidateSparse(
                        recentTokens: 256,
                        candidatePages: 4,
                        olderTokenBudget: 128
                    ),
                    skippedValueTokens: 1,
                    consideredValueTokens: 4,
                    recentTokenCount: 2,
                    olderTokenCount: 1,
                    pageCandidateCount: 4,
                    retainedAttentionMass: 0.995,
                    selectionMS: 0.2,
                    avMS: 0.5,
                    denseK8V4ReferenceMS: 0.8
                )
            )
            let object = try #require(
                JSONSerialization.jsonObject(with: encoded) as? [String: Any]
            )
            let selection = try #require(object["selectionMode"] as? [String: Any])
            #expect(selection["kind"] as? String == "candidateSparse")
            #expect(selection["topK"] as? Int == 128)
            #expect(selection["recentTokens"] as? Int == 256)
            #expect(selection["candidatePages"] as? Int == 4)

            let legacy = """
            {
              "selectionMode": {
                "kind": "candidate-sparse",
                "recentTokens": 128,
                "candidatePages": 2,
                "topK": 64
              },
              "skippedValueTokens": 1,
              "consideredValueTokens": 4,
              "retainedAttentionMass": 0.995,
              "perLayerHeadDiagnostics": []
            }
            """
            let decoded = try JSONDecoder().decode(
                TurboQuantSparseVProofDiagnostics.self,
                from: Data(legacy.utf8)
            )
            #expect(
                decoded.selectionMode == .candidateSparse(
                    recentTokens: 128,
                    candidatePages: 2,
                    olderTokenBudget: 64
                )
            )
        }

        @Test func lowerVBenchmarkMatrixAnchorsRowsToDenseK8V4() {
            let rows = TurboQuantSparseLowerVBenchmarkMatrix.plannedRows(contexts: [32_768])

            #expect(rows.contains { $0.candidate.label == "K8/V4 dense baseline" })
            #expect(rows.contains { $0.candidate.valueBits == 3 && $0.candidate.sparseVSelection == .off })
            #expect(rows.contains { $0.candidate.valueBits == 2 && $0.candidate.sparseVSelection == .off })
            #expect(rows.contains { $0.candidate.valueBitPolicy == .residualVx })
            #expect(rows.contains { $0.candidate.sparseVSelection == .threshold(1e-5) })
            #expect(rows.contains { $0.candidate.sparseVSelection == .topK(256) })
            #expect(rows.contains { $0.candidate.sparseVSelection == .cumulativeMass(0.995) })
            #expect(rows.contains {
                $0.candidate.sparseVSelection == .hybrid(cumulativeMass: 0.995, maxTopK: 256)
            })
            #expect(rows.contains {
                $0.candidate.sparseVSelection == .candidateSparse(
                    recentTokens: 256,
                    candidatePages: 4,
                    olderTokenBudget: 128
                )
            })
            #expect(rows.contains { $0.candidate.boundaryCachePrecision == .raw })
            #expect(rows.allSatisfy { $0.baselineLabel == "dense K8/V4" })
            #expect(rows.allSatisfy { $0.contextLength == 32_768 })
        }

        @Test func lowerVBenchmarkCasesKeepSparseSelectionOutOfThresholdPolicy() {
            let thresholdCandidate = TurboQuantSparseLowerVBenchmarkCandidate(
                label: "threshold",
                valueBits: 3,
                sparseVSelection: .threshold(1e-5)
            )
            let topKCandidate = TurboQuantSparseLowerVBenchmarkCandidate(
                label: "top-k",
                valueBits: 3,
                sparseVSelection: .topK(256)
            )

            #expect(thresholdCandidate.sparseValuePolicy == .force(threshold: 1e-5))
            #expect(topKCandidate.sparseValuePolicy == .off)
            #expect(topKCandidate.fallbackReason == nil)
        }

        @Test func lowerVBenchmarkCasesUseConcreteCodecForValueBits() throws {
            let cases = TurboQuantSparseLowerVBenchmarkMatrix.benchmarkCases(
                contexts: [32_768],
                candidates: [
                    TurboQuantSparseLowerVBenchmarkCandidate(
                        label: "K8/V4 dense baseline",
                        valueBits: 4,
                        productionDefault: true,
                        requiresRealModelGate: false
                    ),
                    TurboQuantSparseLowerVBenchmarkCandidate(label: "K8/V3", valueBits: 3),
                    TurboQuantSparseLowerVBenchmarkCandidate(
                        label: "K8/V4 + top-k",
                        valueBits: 4,
                        sparseVSelection: .topK(256)
                    ),
                ]
            )

            let baseline = try #require(cases.first { $0.variantLabel == "dense-k8-v4" })
            let k8v3 = try #require(cases.first { $0.variantLabel == "k8-v3" })
            let topK = try #require(cases.first { $0.variantLabel == "k8-v4-top-k" })

            #expect(baseline.codec == .affineK8V4)
            #expect(k8v3.codec == .affineK8Vx)
            #expect(topK.codec == .affineK8V4)
            #expect(topK.sparseSelectionConfig == .topK(256))
            #expect(topK.sparseValuePolicy == .off)
            #expect(cases.allSatisfy { $0.runtimeMode == .capacityTurboQuant })
        }

        @Test func comparisonRowsUseDenseK8V4AsDecodeBaseline() throws {
            let baseline = TurboQuantBenchResult(
                label: "qwen3.5-2b-dense-k8-v4",
                anchorLabel: "dense-k8-v4",
                variantLabel: "dense-k8-v4",
                scheme: "turbo8",
                codec: "affine_k8_v4",
                contextLength: 32_768,
                status: .ok,
                detail: nil,
                valuePrecision: "turbo4v2",
                compressedTokensPerSecond: 100,
                plainTokensPerSecond: 80,
                speedRatioToPlain: 1.25,
                cosineSimilarity: 1,
                maxAbsErrorP95: 0,
                finite: true,
                compressedKVBytes: 1024,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )
            let sparse = TurboQuantBenchResult(
                label: "qwen3.5-2b-sparse-v-topk",
                anchorLabel: "dense-k8-v4",
                variantLabel: "sparse-v-topk-256",
                scheme: "turbo8",
                codec: "affine_k8_vx",
                contextLength: 32_768,
                status: .ok,
                detail: nil,
                valuePrecision: "turbo3_5",
                sparseVEnabled: true,
                sparseVSelectionConfig: .topK(256),
                sparseVSkipRatio: 0.5,
                sparseVSkippedValueTokens: 256,
                sparseVTotalValueTokens: 512,
                sparseVRecentTokenCount: 128,
                sparseVOlderTokenCount: 64,
                sparseVPageCandidateCount: 2,
                sparseVRetainedMass: 0.997,
                maxOutputError: 0.002,
                compressedTokensPerSecond: 140,
                plainTokensPerSecond: 80,
                speedRatioToPlain: 1.75,
                cosineSimilarity: 0.999,
                maxAbsErrorP95: 0.002,
                finite: true,
                compressedKVBytes: 768,
                plainKVBytes: 4096,
                memoryReductionRatio: 5.33
            )

            let rows = TurboQuantSparseLowerVBenchmarkMatrix.comparisonRows(
                from: [baseline, sparse]
            )
            let sparseRow = try #require(
                rows.first { $0.candidate.label == "sparse-v-topk-256" }
            )

            #expect(sparseRow.baselineDecodeTokensPerSecond == 100)
            #expect(sparseRow.speedRatioToDenseK8V4 == 1.4)
            #expect(sparseRow.plainDecodeTokensPerSecond == 80)
            #expect(sparseRow.speedRatioToPlainFP16 == 1.75)
            #expect(sparseRow.baselineKVMemoryBytes == 1024)
            #expect(sparseRow.plainKVBytes == 4096)
            #expect(sparseRow.memoryReductionRatioToPlain == 5.33)
            #expect(abs((sparseRow.memoryReductionRatioToDenseK8V4 ?? 0) - (1024.0 / 768.0)) < 1e-9)
            #expect(sparseRow.actualMixedBitsPerValue == 3)
            #expect(sparseRow.promotionEligible == true)
            #expect(sparseRow.promotionBlockReason == nil)
            #expect(sparseRow.calibrationSummary?.referenceConfig == "dense K8/V4")
            #expect(sparseRow.sparseVDiagnostics?.selectionMode == .topK(256))
            #expect(sparseRow.sparseVDiagnostics?.skippedValueTokens == 256)
            #expect(sparseRow.sparseVDiagnostics?.recentTokenCount == 128)
            #expect(sparseRow.sparseVDiagnostics?.olderTokenCount == 64)
            #expect(sparseRow.sparseVDiagnostics?.pageCandidateCount == 2)
            #expect(sparseRow.sparseVDiagnostics?.retainedAttentionMass == 0.997)
        }

        @Test func comparisonRowsSeparateMeasuredFromPromotionEligibility() throws {
            let baseline = TurboQuantBenchResult(
                label: "qwen3.5-2b-dense-k8-v4",
                anchorLabel: "dense-k8-v4",
                variantLabel: "dense-k8-v4",
                scheme: "turbo8",
                codec: "affine_k8_v4",
                contextLength: 32_768,
                status: .ok,
                detail: nil,
                valuePrecision: "turbo4v2",
                compressedTokensPerSecond: 100,
                plainTokensPerSecond: 80,
                speedRatioToPlain: 1.25,
                cosineSimilarity: 1,
                maxAbsErrorP95: 0,
                finite: true,
                compressedKVBytes: 1024,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )
            let fallback = TurboQuantBenchResult(
                label: "qwen3.5-2b-sparse-v-fallback",
                anchorLabel: "dense-k8-v4",
                variantLabel: "sparse-v-fallback",
                scheme: "turbo8",
                codec: "affine_k8_vx",
                contextLength: 32_768,
                status: .ok,
                detail: nil,
                valuePrecision: "turbo4v2",
                fallbackReason: "native sparse path fell back to dense",
                sparseVEnabled: true,
                sparseVSelectionConfig: .threshold(1e-5),
                compressedTokensPerSecond: 90,
                plainTokensPerSecond: 80,
                speedRatioToPlain: 1.125,
                cosineSimilarity: 1,
                maxAbsErrorP95: 0,
                finite: true,
                compressedKVBytes: 1024,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )
            let inactiveSparse = TurboQuantBenchResult(
                label: "qwen3.5-2b-sparse-v-inactive",
                anchorLabel: "dense-k8-v4",
                variantLabel: "sparse-v-inactive",
                scheme: "turbo8",
                codec: "affine_k8_vx",
                contextLength: 32_768,
                status: .ok,
                detail: nil,
                valuePrecision: "turbo4v2",
                sparseVEnabled: false,
                sparseVSelectionConfig: .topK(256),
                compressedTokensPerSecond: 95,
                plainTokensPerSecond: 80,
                speedRatioToPlain: 1.1875,
                cosineSimilarity: 1,
                maxAbsErrorP95: 0,
                finite: true,
                compressedKVBytes: 1024,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )

            let rows = TurboQuantSparseLowerVBenchmarkMatrix.comparisonRows(
                from: [baseline, fallback, inactiveSparse]
            )
            let fallbackRow = try #require(
                rows.first { $0.candidate.label == "sparse-v-fallback" }
            )
            let inactiveRow = try #require(
                rows.first { $0.candidate.label == "sparse-v-inactive" }
            )

            #expect(fallbackRow.status == TurboQuantSparseLowerVBenchmarkRow.Status.measured)
            #expect(fallbackRow.promotionEligible == false)
            #expect(fallbackRow.promotionBlockReason?.contains("fallback") == true)
            #expect(inactiveRow.status == TurboQuantSparseLowerVBenchmarkRow.Status.measured)
            #expect(inactiveRow.promotionEligible == false)
            #expect(inactiveRow.promotionBlockReason?.contains("Sparse-V requested but inactive") == true)

            let missingAnchorRows = TurboQuantSparseLowerVBenchmarkMatrix.comparisonRows(
                from: [fallback]
            )
            let missingAnchorRow = try #require(missingAnchorRows.first)
            #expect(missingAnchorRow.promotionEligible == false)
            #expect(missingAnchorRow.promotionBlockReason?.contains("missing dense K8/V4 baseline") == true)
        }
    }
}
