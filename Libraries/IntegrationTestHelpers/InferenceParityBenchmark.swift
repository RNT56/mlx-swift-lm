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
        public let runtimeMode: TurboQuantRuntimeMode?
        public let quantizedKVStart: Int?

        public init(
            label: String,
            strategy: KVCacheStrategy,
            preset: TurboQuantPreset?,
            kvBits: Int? = nil,
            kvGroupSize: Int? = nil,
            kvCodec: TurboQuantKVCodec? = nil,
            runtimeMode: TurboQuantRuntimeMode? = nil,
            quantizedKVStart: Int? = nil
        ) {
            self.label = label
            self.strategy = strategy
            self.preset = preset
            self.kvBits = kvBits
            self.kvGroupSize = kvGroupSize
            self.kvCodec = kvCodec
            self.runtimeMode = runtimeMode
            self.quantizedKVStart = quantizedKVStart
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

    /// FP16 baseline plus the production-relevant compressed routes.
    ///
    /// `affineK8V4`, `mlxAffine-q8`, and `affineInt4` exercise the native MLX affine family used
    /// by the fastest community ports as the practical throughput route. `turbo3_5`/`turbo4v2`/
    /// `turbo8` keep the current TurboQuant capacity/quality routes visible in the same table.
    public static let affineThroughputQuantizedKVStart = 16_384

    public static let defaultConfigs: [CacheConfig] = [
        CacheConfig(label: "fp16", strategy: .none, preset: nil),
        CacheConfig(
            label: "affineK8V4",
            strategy: .affineK8V4,
            preset: nil,
            kvBits: TurboQuantKVCodec.affineK8V4KeyBits,
            kvGroupSize: TurboQuantKVCodec.affineK8V4KeyGroupSize,
            kvCodec: .affineK8V4,
            quantizedKVStart: affineThroughputQuantizedKVStart
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
        CacheConfig(label: "turbo3_5", strategy: .turboQuant, preset: .turbo3_5),
        CacheConfig(label: "turbo4v2", strategy: .turboQuant, preset: .turbo4v2),
        CacheConfig(label: "turbo8", strategy: .turboQuant, preset: .turbo8),
    ]

    public static func configsFromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [CacheConfig]? {
        guard let raw = environment["TQ_PARITY_CONFIGS"] else { return nil }
        let requested = raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !requested.isEmpty else { return nil }
        if requested.contains(where: { $0.lowercased() == "all" }) {
            return defaultConfigs
        }

        let known = Dictionary(uniqueKeysWithValues: defaultConfigs.map { ($0.label, $0) })
        var selected: [CacheConfig] = []
        for label in requested {
            if let config = known[label] {
                selected.append(config)
            } else {
                switch label.lowercased().replacingOccurrences(of: "-", with: "_") {
                case "mlxaffine", "mlx_affine", "q8", "affine_q8":
                    selected.append(known["mlxAffine-q8"]!)
                case "affinek8v4", "affine_k8_v4", "k8v4", "k8_v4":
                    selected.append(known["affineK8V4"]!)
                case "int4", "affine_int4", "affineint4":
                    selected.append(known["affineInt4"]!)
                case "turbo35", "turbo_3_5":
                    selected.append(known["turbo3_5"]!)
                case "turbo4", "turbo_4_v2":
                    selected.append(known["turbo4v2"]!)
                default:
                    print("warning: ignoring unknown TQ_PARITY_CONFIGS entry '\(label)'")
                }
            }
        }
        return selected.isEmpty ? nil : selected
    }

    /// Run the full sweep. For each context, races every config against FP16 and prints a
    /// table plus the decode-throughput ratio — the headline inference-parity number.
    @discardableResult
    public static func run(
        container: LLModelContainer,
        contexts: [Int],
        generateTokens: Int = 64,
        configs: [CacheConfig]? = nil
    ) async throws -> [Measurement] {
        let configs = configs ?? configsFromEnvironment() ?? defaultConfigs
        var results: [Measurement] = []

        // Untimed warmup: realize weights + compile Metal kernels so the first timed cell is
        // not charged for one-time setup.
        _ = try? await measureOne(
            container: container, contextLength: 512, generateTokens: 4,
            config: CacheConfig(label: "warmup", strategy: .none, preset: nil))

        print("=== TurboQuant end-to-end inference parity (full decode loop, codec vs FP16) ===")
        print("ctx       config          decode tok/s   prefill tok/s   gen   ratio(vs fp16)")
        print("-------   -------------   ------------   -------------   ---   --------------")
        fflush(stdout)

        for ctx in contexts {
            var fp16Decode: Double?
            for cfg in configs {
                do {
                    print("running ctx=\(ctx) config=\(cfg.label) gen=\(generateTokens)")
                    fflush(stdout)
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
                        pad("\(ctx)", 7) + "   " + pad(cfg.label, 13) + "   "
                            + pad(round2(m.decodeTokensPerSecond), 12) + "   "
                            + pad(round1(m.prefillTokensPerSecond), 13) + "   "
                            + pad("\(m.generationTokenCount)", 3) + "   " + ratio)
                    fflush(stdout)
                } catch {
                    print("\(ctx)   \(cfg.label)   FAILED: \(error)")
                    fflush(stdout)
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
            if let kvBits = config.kvBits {
                params.kvBits = kvBits
            }
            if let kvGroupSize = config.kvGroupSize {
                params.kvGroupSize = kvGroupSize
            }
            if let kvCodec = config.kvCodec {
                params.kvCodec = kvCodec
            }
            if let runtimeMode = config.runtimeMode {
                params.turboQuantRuntimeMode = runtimeMode
            }
            if let quantizedKVStart = config.quantizedKVStart {
                params.quantizedKVStart = quantizedKVStart
            }
            if let preset = config.preset {
                params.turboQuantPreset = preset
            }

            var iterator = try TokenIterator(
                input: input,
                model: context.model,
                parameters: params
            )
            let generationStart = Date.timeIntervalSinceReferenceDate
            var generated = 0
            while iterator.next() != nil {
                generated += 1
            }
            Stream().synchronize()

            guard generated > 0 else {
                throw IntegrationTestFailure(
                    "no generated tokens (ctx=\(contextLength), \(config.label))")
            }
            let generationTime = Date.timeIntervalSinceReferenceDate - generationStart
            let promptTime = max(iterator.promptPrefillTime, Double.leastNonzeroMagnitude)
            return Measurement(
                context: contextLength,
                label: config.label,
                decodeTokensPerSecond: Double(generated)
                    / max(generationTime, Double.leastNonzeroMagnitude),
                prefillTokensPerSecond: Double(contextLength) / promptTime,
                generationTokenCount: generated)
        }
    }

    // MARK: - Tiny formatting helpers (avoid String(format:) %@ width quirks)

    private static func pad(_ s: String, _ width: Int) -> String {
        s.count >= width ? s : s + String(repeating: " ", count: width - s.count)
    }
    private static func round1(_ v: Double) -> String { String(format: "%.1f", v) }
    private static func round2(_ v: Double) -> String { String(format: "%.2f", v) }
}
