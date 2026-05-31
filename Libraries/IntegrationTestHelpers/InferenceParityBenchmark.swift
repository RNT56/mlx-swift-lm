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
import MLX
import MLXLLM
import MLXLMCommon

public enum InferenceParityBenchmark {

    /// One KV-cache configuration raced against FP16.
    public struct CacheConfig: Sendable {
        public let label: String
        public let strategy: KVCacheStrategy
        public let preset: TurboQuantPreset?

        public init(label: String, strategy: KVCacheStrategy, preset: TurboQuantPreset?) {
            self.label = label
            self.strategy = strategy
            self.preset = preset
        }
    }

    /// One measured (context × config) cell.
    public struct Measurement: Sendable {
        public let context: Int
        public let label: String
        public let decodeTokensPerSecond: Double
        public let prefillTokensPerSecond: Double
        public let generationTokenCount: Int
    }

    /// FP16 baseline + the two production-relevant TurboQuant presets (turbo4v2 = qwen default).
    public static let defaultConfigs: [CacheConfig] = [
        CacheConfig(label: "fp16", strategy: .none, preset: nil),
        CacheConfig(label: "turbo4v2", strategy: .turboQuant, preset: .turbo4v2),
        CacheConfig(label: "turbo8", strategy: .turboQuant, preset: .turbo8),
    ]

    /// Run the full sweep. For each context, races every config against FP16 and prints a
    /// table plus the decode-throughput ratio — the headline inference-parity number.
    @discardableResult
    public static func run(
        container: LLModelContainer,
        contexts: [Int],
        generateTokens: Int = 64,
        configs: [CacheConfig]? = nil
    ) async throws -> [Measurement] {
        let configs = configs ?? defaultConfigs
        var results: [Measurement] = []

        // Untimed warmup: realize weights + compile Metal kernels so the first timed cell is
        // not charged for one-time setup.
        _ = try? await measureOne(
            container: container, contextLength: 512, generateTokens: 4,
            config: CacheConfig(label: "warmup", strategy: .none, preset: nil))

        print("=== TurboQuant end-to-end inference parity (full decode loop, codec vs FP16) ===")
        print("ctx       config     decode tok/s   prefill tok/s   gen   ratio(vs fp16)")
        print("-------   --------   ------------   -------------   ---   --------------")

        for ctx in contexts {
            var fp16Decode: Double?
            for cfg in configs {
                do {
                    let m = try await measureOne(
                        container: container, contextLength: ctx,
                        generateTokens: generateTokens, config: cfg)
                    results.append(m)
                    if cfg.label == "fp16" { fp16Decode = m.decodeTokensPerSecond }

                    let ratio: String
                    if let base = fp16Decode, base > 0, cfg.label != "fp16" {
                        ratio = String(format: "%.3f", m.decodeTokensPerSecond / base)
                    } else {
                        ratio = "--"
                    }
                    print(
                        pad("\(ctx)", 7) + "   " + pad(cfg.label, 8) + "   "
                            + pad(round2(m.decodeTokensPerSecond), 12) + "   "
                            + pad(round1(m.prefillTokensPerSecond), 13) + "   "
                            + pad("\(m.generationTokenCount)", 3) + "   " + ratio)
                } catch {
                    print("\(ctx)   \(cfg.label)   FAILED: \(error)")
                }
            }
        }
        return results
    }

    private static func measureOne(
        container: LLModelContainer,
        contextLength: Int,
        generateTokens: Int,
        config: CacheConfig
    ) async throws -> Measurement {
        // Synthetic in-vocab prompt of exact length. Content is irrelevant for a speed
        // measurement; only KV-cache depth (= contextLength) matters.
        let promptIds = (0 ..< contextLength).map { Int32($0 % 1024 + 16) }

        return try await container.perform { context in
            // LMInput expects a 1-D [seq] token array (it adds the batch dim internally);
            // see the reference generate(promptTokens:) path. Passing [1, seq] corrupts shapes.
            let tokens = MLXArray(promptIds)  // [contextLength]
            let input = LMInput(tokens: tokens)

            var params = GenerateParameters(
                maxTokens: generateTokens,
                maxKVSize: contextLength + generateTokens + 16,
                kvCacheStrategy: config.strategy)
            // Chunk prefill so no single command buffer trips the macOS GPU watchdog
            // (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) at long context.
            params.prefillStepSize = 512
            if let preset = config.preset {
                params.turboQuantPreset = preset
            }

            var info: GenerateCompletionInfo?
            for await generation in try generate(
                input: input, parameters: params, context: context)
            {
                if case .info(let completionInfo) = generation { info = completionInfo }
            }
            guard let info else {
                throw IntegrationTestFailure(
                    "no completion info (ctx=\(contextLength), \(config.label))")
            }
            return Measurement(
                context: contextLength,
                label: config.label,
                decodeTokensPerSecond: info.tokensPerSecond,
                prefillTokensPerSecond: info.promptTokensPerSecond,
                generationTokenCount: info.generationTokenCount)
        }
    }

    // MARK: - Tiny formatting helpers (avoid String(format:) %@ width quirks)

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
    private static func round1(_ v: Double) -> String { String(format: "%.1f", v) }
    private static func round2(_ v: Double) -> String { String(format: "%.2f", v) }
}
