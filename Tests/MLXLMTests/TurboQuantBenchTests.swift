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

            let cases = contexts.flatMap { context in
                schemes.map {
                    TurboQuantBenchCase.qwen35_2B(contextLength: context, scheme: $0)
                }
            }
            let results = TurboQuantBench.sweep(
                profile: profile, cases: cases,
                iterations: iterations, warmupIterations: warmup)

            print(
                "\n=== TurboQuant A-series attention sweep "
                    + "(qwen3.5-2b geometry, \(runFullMatrix ? "full matrix" : "smoke")) ===")
            print(TurboQuantBench.renderTable(results))
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let json = try? encoder.encode(results),
                let text = String(data: json, encoding: .utf8)
            {
                print("\n--- TurboQuantBench JSON ---\n\(text)\n")
            }

            #expect(results.count == cases.count)
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
    }
}
