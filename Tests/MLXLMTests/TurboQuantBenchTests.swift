import Foundation
import Testing
import TurboQuantBench

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    /// On-device A-series synthetic attention benchmark.
    ///
    /// By default this runs a fast single-point smoke sweep so every test run still
    /// exercises (and regression-guards) the harness + Metal kernels. Set the
    /// environment variable `TQ_BENCH=1` to run the full context-length × scheme
    /// matrix at the production Qwen3.5-2B geometry — the run that, executed on a
    /// physical A-series device, yields smoke/regression throughput and memory numbers.
    /// Release parity gates now require real model inference evidence, not this
    /// synthetic attention-shape sweep.
    ///
    /// Run the full matrix on a connected iPhone:
    ///
    ///     xcodebuild test \
    ///       -scheme mlx-swift-lm-Package \
    ///       -destination 'platform=iOS,name=<your device>' \
    ///       -only-testing:MLXLMTests/TurboQuantBenchSuite \
    ///       TQ_BENCH=1
    ///
    /// (Or set `TQ_BENCH=1` in the scheme's Test action → Environment Variables.)
    /// The Simulator and Mac both run the desktop GPU, so only a physical device
    /// produces A-series numbers; the printed table + JSON are the deliverable.
    @Suite struct TurboQuantBenchSuite {

        @Test func turboQuantAttentionContextSweep() throws {
            FileHandle.standardError.write(Data(
                "SYNTHETIC KERNEL MICROBENCH — NOT real-model, NOT promotable (sinusoid K/V/Q, no checkpoint loaded)\n".utf8))
            let runFullMatrix = ProcessInfo.processInfo.environment["TQ_BENCH"] == "1"

            let profile = try #require(
                TurboQuantProfileRegistry.bundled.profile(
                    for: "mlx-community/Qwen3.5-2B-OptiQ-4bit",
                    modelType: "qwen3_5",
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ),
                "Qwen3.5-2B bundled profile must resolve for the benchmark geometry"
            )

            let contexts = runFullMatrix ? [8192, 16384, 32768, 65536, 131072] : [8192]
            let schemes: [TurboQuantScheme] =
                runFullMatrix ? [.turbo8, .turbo4v2, .turbo3_5] : [.turbo4v2]
            let iterations = runFullMatrix ? 12 : 3
            let warmup = runFullMatrix ? 3 : 1

            // Sweep one context at a time, snapshotting the device environment (G1 memory +
            // G2 thermalState) before and after each context's sub-sweep. Over the full matrix
            // (minutes of sustained GPU load) this yields a memory + thermal *trajectory* — the
            // device-cycle evidence the per-shape kernel rows cannot carry on their own.
            var results: [TurboQuantBenchResult] = []
            var environments: [TurboQuantBenchEnvironment] = [
                TurboQuantBench.captureEnvironment(label: "sweep/before")
            ]
            for context in contexts {
                environments.append(
                    TurboQuantBench.captureEnvironment(label: "ctx=\(context)/before"))
                let cases = schemes.map {
                    TurboQuantBenchCase.qwen35_2B(contextLength: context, scheme: $0)
                }
                results += TurboQuantBench.sweep(
                    profile: profile, cases: cases,
                    iterations: iterations, warmupIterations: warmup)
                environments.append(
                    TurboQuantBench.captureEnvironment(label: "ctx=\(context)/after"))
            }
            environments.append(TurboQuantBench.captureEnvironment(label: "sweep/after"))

            print(
                "\n=== TurboQuant A-series attention sweep "
                    + "(qwen3.5-2b geometry, \(runFullMatrix ? "full matrix" : "smoke")) ===")
            print(TurboQuantBench.renderTable(results))
            for env in environments {
                print(
                    "  env[\(env.label)] thermal=\(env.thermalState) "
                        + "active=\(env.activeMemoryBytes) peak=\(env.peakMemoryBytes) "
                        + "headroom=\(env.availableMemoryBytes)")
            }
            // Combined deliverable: the G1/G2 environment trajectory + the per-shape rows.
            struct Deliverable: Codable {
                var environments: [TurboQuantBenchEnvironment]
                var results: [TurboQuantBenchResult]
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let json = try? encoder.encode(Deliverable(environments: environments, results: results)),
                let text = String(data: json, encoding: .utf8)
            {
                print("\n--- TurboQuantBench JSON ---\n\(text)\n")
            }
            // G2 guard: any .critical thermalState during the matrix is a non-promotable run.
            #expect(!environments.contains { $0.thermalState == "critical" })

            #expect(results.count == contexts.count * schemes.count)
            // before + after + (before,after per context).
            #expect(environments.count == 2 + 2 * contexts.count)
            // Kernel-level failures are real bugs; environment-dependent skips are not.
            #expect(!results.contains { $0.status == .failed })
            // When a case does measure, the compressed output must be finite and
            // roughly track the FP16 reference (a collapsed cosine ⟹ codec regression).
            for result in results where result.status == .ok {
                #expect(result.finite)
                #expect(result.cosineSimilarity > 0.5)
                #expect(result.compressedTokensPerSecond > 0)
            }

            if ProcessInfo.processInfo.environment["TQ_BENCH_HYBRID"] == "1" {
                let hybridContexts = runFullMatrix ? [32768, 65536, 131072] : [32768]
                let hybridResults = hybridContexts.map {
                    TurboQuantBench.measureHybridSelector(
                        TurboQuantHybridBenchCase(
                            label: "qwen3.5-2b-hybrid-selector",
                            contextLength: $0,
                            selectorHints: [
                                TurboQuantColdSelectorHint(
                                    startToken: 0,
                                    endToken: 1024,
                                    semanticScore: 1,
                                    anchorFlags: [.system],
                                    sourceID: "system"
                                )
                            ]
                        ),
                        iterations: runFullMatrix ? 12 : 3,
                        warmupIterations: runFullMatrix ? 3 : 1
                    )
                }
                print("\n=== TurboQuant hybrid selector sweep ===")
                print(TurboQuantBench.renderTable(hybridResults))
                #expect(!hybridResults.contains { $0.status == .failed })
                for result in hybridResults where result.status == .ok {
                    #expect((result.selectedBudgetedColdTokens ?? 0) <= (result.coldBudgetTokens ?? 0))
                    #expect(result.selectorEscalation != "exhaustive")
                    #expect(result.fullScanFallbackCount == 0)
                }
            }
        }

        @Test func captureEnvironmentReportsThermalAndMemory() throws {
            let env = TurboQuantBench.captureEnvironment(label: "unit/probe")
            #expect(env.label == "unit/probe")
            #expect(["nominal", "fair", "serious", "critical", "unknown"].contains(env.thermalState))
            #expect(env.activeMemoryBytes >= 0)
            #expect(env.peakMemoryBytes >= 0)
            #expect(env.availableMemoryBytes >= 0)
            // Codable round-trip (the device deliverable is JSON).
            let decoded = try JSONDecoder().decode(
                TurboQuantBenchEnvironment.self, from: try JSONEncoder().encode(env))
            #expect(decoded.thermalState == env.thermalState)
            #expect(decoded.peakMemoryBytes == env.peakMemoryBytes)
        }

        @Test func sparseVTopKBenchCaseUsesNativeSelectionDiagnostics() throws {
            let profile = try #require(
                TurboQuantProfileRegistry.bundled.profile(
                    for: "mlx-community/Qwen3.5-2B-OptiQ-4bit",
                    modelType: "qwen3_5",
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ),
                "Qwen3.5-2B bundled profile must resolve for the benchmark geometry"
            )
            let benchCase = TurboQuantBenchCase.qwen35_2B(
                contextLength: 4096,
                scheme: .turbo8,
                codec: .affineK8V4,
                sparseSelectionConfig: .topK(128)
            )

            let result = TurboQuantBench.measure(
                profile: profile,
                benchCase,
                iterations: 1,
                warmupIterations: 0
            )

            #expect(result.status != .failed)
            guard result.status == .ok else { return }
            #expect(result.sparseVEnabled)
            #expect(result.sparseVSelectionConfig == .topK(128))
            #expect((result.sparseVSkippedValueTokens ?? 0) > 0)
            #expect((result.sparseVRetainedMass ?? 0) > 0)
            #expect(result.fallbackReason == nil)
            #expect(result.nativeDiagnostics?[1] == 12)
        }

        @Test func legacyResultJSONDefaultsWave0Fields() throws {
            let json = """
                {
                  "label": "qwen3.5-2b-turbo4v2-8192",
                  "scheme": "turbo4v2",
                  "contextLength": 8192,
                  "status": "ok",
                  "compressedTokensPerSecond": 22.22,
                  "plainTokensPerSecond": 634.54,
                  "speedRatioToPlain": 0.035,
                  "cosineSimilarity": 0.999824,
                  "maxAbsErrorP95": 0.01,
                  "finite": true,
                  "compressedKVBytes": 11010056,
                  "plainKVBytes": 33554432,
                  "memoryReductionRatio": 3.05
                }
                """

            let result = try JSONDecoder().decode(TurboQuantBenchResult.self, from: Data(json.utf8))

            #expect(result.measuredIterations == 0)
            #expect(result.warmupIterations == 0)
            #expect(result.cooldownMilliseconds == 0)
            #expect(result.route == "compressedFused")
            #expect(result.runtimeMode == "capacityTurboQuant")
            #expect(result.requestedRuntimeMode == "capacityTurboQuant")
            #expect(result.resolvedRuntimeMode == "capacityTurboQuant")
            #expect(result.keyPrecision == "turbo8")
            #expect(result.valuePrecision == "turbo4v2")
            #expect(result.backend == "swiftMetalKernel")
            #expect(result.swiftMetalCompressedTokensPerSecond == nil)
            #expect(result.nativeSpeedRatioToSwiftMetal == nil)
            #expect(result.nativePerfGateMinimumContextLength == nil)
            #expect(result.nativePerfGateRequiredSpeedup == nil)
            #expect(result.nativePerfGatePassed == nil)
            #expect(result.kernelFlags == nil)
            #expect(result.sparseVSelectionConfig == nil)
            #expect(result.sparseVSkippedValueTokens == nil)
            #expect(result.sparseVRetainedMass == nil)
            #expect(result.maxOutputError == nil)
            #expect(result.layerHeadDiagnostics == nil)
            #expect(result.selectedColdTokens == nil)
            #expect(result.selectorEscalation == nil)
        }

        @Test func wave1ResultJSONIncludesRuntimePolicyAndResidencyFields() throws {
            let policy = TurboQuantKVPrecisionPolicy.qwenQ4Default
            let result = TurboQuantBenchResult(
                label: "qwen3.5-2b-throughput-8192",
                scheme: "turbo8",
                contextLength: 8192,
                status: .ok,
                detail: nil,
                measuredIterations: 12,
                warmupIterations: 3,
                cooldownMilliseconds: 10,
                route: TurboQuantRuntimeRoute.throughputTurboQuantNativeSDPA.rawValue,
                runtimeMode: TurboQuantRuntimeMode.throughputTurboQuant.rawValue,
                requestedRuntimeMode: TurboQuantRuntimeMode.auto.rawValue,
                resolvedRuntimeMode: TurboQuantRuntimeMode.throughputTurboQuant.rawValue,
                keyPrecision: policy.key.rawValue,
                valuePrecision: policy.value.rawValue,
                precisionPolicy: policy,
                compressedTokensPerSecond: 90,
                plainTokensPerSecond: 100,
                compressedP95TokensPerSecond: 85,
                plainP95TokensPerSecond: 95,
                speedRatioToPlain: 0.9,
                cosineSimilarity: 1,
                maxAbsErrorP95: 0,
                finite: true,
                compressedKVBytes: 1024,
                compressedKeyBytes: 512,
                compressedValueBytes: 512,
                decodedActiveKVBytes: 4096,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )

            let decoded = try JSONDecoder().decode(
                TurboQuantBenchResult.self,
                from: try JSONEncoder().encode(result)
            )

            #expect(decoded.route == "throughputTurboQuantNativeSDPA")
            #expect(decoded.measuredIterations == 12)
            #expect(decoded.warmupIterations == 3)
            #expect(decoded.cooldownMilliseconds == 10)
            #expect(decoded.requestedRuntimeMode == "auto")
            #expect(decoded.resolvedRuntimeMode == "throughputTurboQuant")
            #expect(decoded.precisionPolicy == policy)
            #expect(decoded.compressedKeyBytes == 512)
            #expect(decoded.compressedValueBytes == 512)
            #expect(decoded.decodedActiveKVBytes == 4096)
            #expect(decoded.compressedP95TokensPerSecond == 85)
            #expect(decoded.plainP95TokensPerSecond == 95)
        }

        @Test func wave3ResultJSONIncludesNativePerfGateFields() throws {
            let result = TurboQuantBenchResult(
                label: "qwen3.5-2b-native-32768",
                scheme: "turbo4v2",
                contextLength: 32768,
                status: .ok,
                detail: nil,
                route: "capacityTurboQuantCompressed",
                selectedPath: "capacityTurboQuantCompressed",
                runtimeMode: TurboQuantRuntimeMode.capacityTurboQuant.rawValue,
                requestedRuntimeMode: TurboQuantRuntimeMode.capacityTurboQuant.rawValue,
                resolvedRuntimeMode: TurboQuantRuntimeMode.capacityTurboQuant.rawValue,
                backend: "nativeMLX",
                nativeDiagnostics: [3, 3, 64, 512, 0, 0, 0, 7],
                swiftMetalCompressedTokensPerSecond: 50,
                swiftMetalCompressedP95TokensPerSecond: 45,
                nativeSpeedRatioToSwiftMetal: 1.8,
                nativePerfGateMinimumContextLength: 32768,
                nativePerfGateRequiredSpeedup: 2,
                nativePerfGatePassed: false,
                compressedTokensPerSecond: 90,
                plainTokensPerSecond: 120,
                compressedP95TokensPerSecond: 80,
                plainP95TokensPerSecond: 110,
                speedRatioToPlain: 0.75,
                cosineSimilarity: 0.99,
                maxAbsErrorP95: 0.01,
                finite: true,
                compressedKVBytes: 1024,
                compressedKeyBytes: 512,
                compressedValueBytes: 512,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )

            let decoded = try JSONDecoder().decode(
                TurboQuantBenchResult.self,
                from: try JSONEncoder().encode(result)
            )

            #expect(decoded.backend == "nativeMLX")
            #expect(decoded.nativeDiagnostics == [3, 3, 64, 512, 0, 0, 0, 7])
            #expect(decoded.swiftMetalCompressedTokensPerSecond == 50)
            #expect(decoded.swiftMetalCompressedP95TokensPerSecond == 45)
            #expect(decoded.nativeSpeedRatioToSwiftMetal == 1.8)
            #expect(decoded.nativePerfGateMinimumContextLength == 32768)
            #expect(decoded.nativePerfGateRequiredSpeedup == 2)
            #expect(decoded.nativePerfGatePassed == false)
        }

        @Test func resultJSONIncludesTimingBreakdownFields() throws {
            let result = TurboQuantBenchResult(
                label: "qwen3.5-2b-sparse-timing",
                scheme: "turbo8",
                codec: TurboQuantKVCodec.affineK8V4.rawValue,
                contextLength: 32768,
                status: .ok,
                detail: nil,
                measuredIterations: 8,
                warmupIterations: 2,
                cooldownMilliseconds: 25,
                sparseVEnabled: true,
                sparseVSelectionConfig: .topK(256),
                qkMS: 0.11,
                softmaxMS: 0.22,
                selectionMS: 0.33,
                maskOrCompactionMS: 0.44,
                avMS: 0.55,
                denseK8V4ReferenceMS: 0.66,
                compressedTokensPerSecond: 90,
                plainTokensPerSecond: 100,
                speedRatioToPlain: 0.9,
                cosineSimilarity: 0.999,
                maxAbsErrorP95: 0.01,
                finite: true,
                compressedKVBytes: 1024,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )

            let decoded = try JSONDecoder().decode(
                TurboQuantBenchResult.self,
                from: try JSONEncoder().encode(result)
            )

            #expect(decoded.measuredIterations == 8)
            #expect(decoded.warmupIterations == 2)
            #expect(decoded.cooldownMilliseconds == 25)
            #expect(decoded.qkMS == 0.11)
            #expect(decoded.softmaxMS == 0.22)
            #expect(decoded.selectionMS == 0.33)
            #expect(decoded.maskOrCompactionMS == 0.44)
            #expect(decoded.avMS == 0.55)
            #expect(decoded.denseK8V4ReferenceMS == 0.66)
        }

        @Test func hybridSelectorResultJSONIncludesWave5Diagnostics() throws {
            let result = TurboQuantBench.measureHybridSelector(
                TurboQuantHybridBenchCase(
                    contextLength: 32768,
                    hotWindowTokens: 8192,
                    coldBlockTokens: 1024,
                    coldBudgetTokens: 2048,
                    maxColdBudgetTokens: 4096,
                    selectorPolicy: TurboQuantColdSelectorPolicy(
                        nearestBlockCount: 2,
                        minimumConfidence: 0,
                        allowMaxBudgetEscalation: false
                    ),
                    selectorHints: [
                        TurboQuantColdSelectorHint(
                            startToken: 0,
                            endToken: 1024,
                            semanticScore: 1,
                            anchorFlags: [.system],
                            sourceID: "system"
                        )
                    ]
                ),
                iterations: 1,
                warmupIterations: 0
            )

            let decoded = try JSONDecoder().decode(
                TurboQuantBenchResult.self,
                from: try JSONEncoder().encode(result)
            )

            #expect(decoded.route == "hybridTurboQuant")
            #expect(decoded.hotTokens == 8192)
            #expect(decoded.coldBlockCount == 24)
            #expect(decoded.selectedColdTokens != nil)
            #expect((decoded.selectedBudgetedColdTokens ?? 0) <= (decoded.coldBudgetTokens ?? 0))
            #expect(decoded.anchorColdTokens == 1024)
            #expect(decoded.selectorEscalation == TurboQuantColdSelectorEscalation.none.rawValue)
            #expect(decoded.fullScanFallbackCount == 0)
        }

        @Test func sparseVProofConfigsRepresentHardeningGrid() throws {
            let configs = TurboQuantSparseSelectionConfig.proofMatrix

            #expect(configs.filter { $0.mode == .threshold }.compactMap(\.threshold) == [
                1e-4, 5e-5, 1e-5,
            ])
            #expect(configs.filter { $0.mode == .topK }.compactMap(\.topK) == [128, 256, 512])
            #expect(
                configs.filter { $0.mode == .cumulativeMass }
                    .compactMap(\.cumulativeMassPercent) == [99.0, 99.5, 99.9])
            #expect(
                configs.filter { $0.mode == .hybridCumulativeFloorMaxTopK }.compactMap(\.maxTopK)
                    == [128, 256, 512])
            let defaultCase = TurboQuantBenchCase.qwen35_2B(
                contextLength: 8192,
                scheme: .turbo8
            )
            #expect(defaultCase.sparseSelectionConfig == nil)
            #expect(defaultCase.sparseValuePolicy == .off)
        }

        @Test func optimizationPathMatrixCoversRuntimeAndCodecPaths() throws {
            let rows = TurboQuantBench.optimizationPathMatrix(contextLength: 32768)

            #expect(rows.allSatisfy { $0.anchorLabel == "fp16-plain" })
            #expect(rows.contains {
                $0.runtimeMode == .rawPreferred
                    && $0.variantLabel == "fp16-plain-raw-sdpa"
            })
            #expect(rows.contains {
                $0.runtimeMode == .throughputTurboQuant
                    && $0.variantLabel == "throughput-k8-v4-decoded-sdpa"
            })
            #expect(rows.contains {
                $0.runtimeMode == .capacityTurboQuant
                    && $0.codec == .affineK8V4
                    && $0.variantLabel == "capacity-k8-v4-compressed"
            })
            #expect(rows.contains {
                $0.runtimeMode == .capacityTurboQuant
                    && $0.codec == .affineK8Vx
                    && $0.precisionPolicy?.value == .turbo3_5
            })
            #expect(rows.contains {
                $0.runtimeMode == .capacityTurboQuant
                    && $0.codec == .affineInt4
            })
            #expect(rows.contains {
                $0.runtimeMode == .capacityTurboQuant
                    && $0.codec == .polarQJL
                    && $0.scheme == .turbo4v2
            })
            #expect(rows.contains {
                $0.runtimeMode == .capacityTurboQuant
                    && $0.codec == .polarQJL
                    && $0.scheme == .turbo3_5
            })
            #expect(rows.contains {
                $0.runtimeMode == .capacityTurboQuant
                    && $0.codec == .polarWHT
                    && $0.variantLabel == "capacity-polar-wht-v3"
            })
            #expect(rows.allSatisfy {
                $0.runtimeMode == .capacityTurboQuant || $0.sparseSelectionConfig == nil
            })
        }

        @Test func sparseVHardeningMatrixAnchorsLowerVAndSparseRows() throws {
            let rows = TurboQuantBench.sparseVHardeningMatrix(contextLength: 32768)

            #expect(rows.count == 3 + TurboQuantSparseSelectionConfig.proofMatrix.count)
            #expect(rows.allSatisfy { $0.anchorLabel == "dense-k8-v4" })
            #expect(rows[0].codec == .affineK8V4)
            #expect(rows[0].variantLabel == "dense-k8-v4")
            #expect(rows.contains {
                $0.codec == .affineK8Vx && $0.precisionPolicy?.value == .turbo3_5
                    && $0.variantLabel == "k8-v3-protected-boundary"
            })
            #expect(rows.contains {
                $0.codec == .affineK8Vx && $0.precisionPolicy?.value == .turbo2_5
                    && $0.variantLabel == "k8-v2-protected-boundary"
            })
            #expect(rows.contains {
                $0.sparseSelectionConfig == .threshold(1e-4)
                    && $0.sparseValuePolicy == .force(threshold: 1e-4)
            })
            #expect(rows.contains {
                $0.sparseSelectionConfig == .topK(256)
                    && $0.sparseValuePolicy == .off
            })
            #expect(TurboQuantSparseSelectionConfig.candidateSparse(
                recentTokens: 256,
                candidatePages: 4,
                olderTokenBudget: 128
            ).nativeSelectionMode == .candidateSparse)
        }

        @Test func sparseVDiagnosticsRoundTripInBenchResult() throws {
            let result = TurboQuantBenchResult(
                label: "qwen3.5-2b-sparse",
                anchorLabel: "dense-k8-v4",
                variantLabel: "sparse-v-threshold",
                scheme: "turbo8",
                codec: TurboQuantKVCodec.affineK8V4.rawValue,
                contextLength: 32768,
                status: .ok,
                detail: nil,
                route: "capacityTurboQuantCompressed",
                sparseVEnabled: true,
                sparseVSelectionConfig: .hybrid(cumulativeFloorPercent: 99.5, maxTopK: 256),
                sparseVThreshold: 1e-5,
                sparseVSkipRatio: 0.25,
                sparseVSkippedValueTokens: 1024,
                sparseVTotalValueTokens: 4096,
                sparseVRetainedMass: 0.995,
                maxOutputError: 0.03125,
                layerHeadDiagnostics: [
                    TurboQuantBenchLayerHeadDiagnostics(
                        layerIndex: 3,
                        headIndex: 1,
                        skippedValueTokens: 128,
                        totalValueTokens: 512,
                        retainedMass: 0.997,
                        maxOutputError: 0.01,
                        cosineSimilarity: 0.999,
                        fallbackReason: nil
                    )
                ],
                compressedTokensPerSecond: 90,
                plainTokensPerSecond: 100,
                speedRatioToPlain: 0.9,
                cosineSimilarity: 0.998,
                maxAbsErrorP95: 0.03125,
                finite: true,
                compressedKVBytes: 1024,
                compressedKeyBytes: 512,
                compressedValueBytes: 512,
                plainKVBytes: 4096,
                memoryReductionRatio: 4
            )

            let decoded = try JSONDecoder().decode(
                TurboQuantBenchResult.self,
                from: try JSONEncoder().encode(result)
            )

            #expect(decoded.anchorLabel == "dense-k8-v4")
            #expect(decoded.sparseVSelectionConfig == .hybrid(cumulativeFloorPercent: 99.5, maxTopK: 256))
            #expect(decoded.sparseVSkippedValueTokens == 1024)
            #expect(decoded.sparseVTotalValueTokens == 4096)
            #expect(decoded.sparseVRetainedMass == 0.995)
            #expect(decoded.maxOutputError == 0.03125)
            #expect(decoded.layerHeadDiagnostics?.first?.layerIndex == 3)
            #expect(decoded.layerHeadDiagnostics?.first?.cosineSimilarity == 0.999)
        }
    }
}
