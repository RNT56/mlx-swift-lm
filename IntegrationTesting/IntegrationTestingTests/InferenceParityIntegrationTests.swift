// Copyright © 2026 RNT56.
//
// End-to-end inference-parity: real full-model decode tok/s, TurboQuant codec vs FP16, on the
// default Qwen3.5-2B. This is the apples-to-apples metric the attention-only benchmarks could
// not give (kernel-only was 0.07–0.18x; end-to-end dilutes that). Downloads the model on first
// run (cached in ~/.cache/huggingface). Mac-runnable — no A-series device required.
//
//   xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
//     -scheme IntegrationTesting \
//     -only-testing:IntegrationTestingTests/InferenceParityIntegrationTests
//
// Contexts default to 4K/8K/16K/32K; override with TQ_PARITY_CONTEXTS="8192,131072".

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Testing
import Tokenizers

private let models = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

@Suite(.serialized)
struct InferenceParityIntegrationTests {

    @Test func qwen35_end_to_end_inference_parity() async throws {
        let config = LLMModelFactory.shared.configuration(id: IntegrationTestModelIDs.qwen35)
        let container = try await models.llmContainer(for: config)
        let results = try await InferenceParityBenchmark.run(
            container: container,
            contexts: Self.contextsFromEnv(),
            generateTokens: Self.generateTokensFromEnv())
        #expect(!results.isEmpty)
    }

    private static func contextsFromEnv() -> [Int] {
        if let raw = ProcessInfo.processInfo.environment["TQ_PARITY_CONTEXTS"] {
            let parsed = raw.split(separator: ",")
                .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            if !parsed.isEmpty { return parsed }
        }
        return [4096, 8192, 16384, 32768]
    }

    private static func generateTokensFromEnv() -> Int {
        if let raw = ProcessInfo.processInfo.environment["TQ_PARITY_GENERATE_TOKENS"],
            let value = Int(raw.trimmingCharacters(in: .whitespaces)),
            value > 0
        {
            return value
        }
        return 64
    }
}
