import Foundation
import Testing
import TurboQuantBench

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    /// On-device A-series attention benchmark.
    ///
    /// By default this runs a fast single-point smoke sweep so every test run still
    /// exercises (and regression-guards) the harness + Metal kernels. Set the
    /// environment variable `TQ_BENCH=1` to run the full context-length × scheme
    /// matrix at the production Qwen3.5-2B geometry — the run that, executed on a
    /// physical A-series device, yields the production-faithful throughput / quality
    /// / memory numbers the overhaul plan gates everything else on.
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
        }
    }
}
