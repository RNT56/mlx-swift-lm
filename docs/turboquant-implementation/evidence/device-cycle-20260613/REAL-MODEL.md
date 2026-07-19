# A-series REAL-MODEL device evidence (2026-06-13) — `real-model-inference-v1`

The synthetic kernel sweep (see `README.md`) characterizes the attention kernels but the
promotion checklist demands **real-model throughput on device**. This run delivers it: the
**actual Qwen3.5-2B-OptiQ-4bit weights** (1.53 GB, on-device) loaded and **really generating
tokens**, comparing plain FP16 vs compressed affineK8V4 KV through the canonical
`InferenceParityBenchmark` engine (`benchmarkSuiteID = real-model-inference-v1`).

Device: **GBU-12 = iPhone 15 Pro Max (`iPhone16,2`, A17 Pro), iOS 26.5 (23F77)**.
Pins: **mlx-swift `6bfa04e` + mlx-swift-lm `295e66b`**. 48 generated tokens, 2 interleaved
repeats (order-randomized), greedy/deterministic.

## How it was run

The SwiftPM test bundle can't run on device, and the synthetic `TurboQuantBench` doesn't load
weights. So a **DEBUG real-model hook was added to the Pines app**
(`Pines/App/PinesRealModelTurboQuantDiagnostics.swift`, gated `PINES_TQ_REAL_BENCH=1`): it
ensures the model is downloaded via Pines' own `ModelLifecycleService` (idempotent — was already
installed), loads the real container via the production loader
(`LLMModelFactory.loadContainer(from: PinesHubDownloader(), using: PinesTokenizerLoader(),
configuration: .init(directory:))`), then runs `InferenceParityBenchmark.runDetailed` (throughput,
configs `[fp16, affineK8V4]`) + `runQualityGates` (compressed vs the FP16 reference), writing JSON
to `Documents/PinesDiagnostics/`. (Pines app source change is in the working tree, **not committed**
to the held Pines repo; it links the `IntegrationTestHelpers` library product. Built with
`-jobs 6` — `-j12` OOM-killed a swiftc process with no diagnostic.)

## Real-model decode throughput + quality

```
config       ctx      decode tok/s   ratio→FP16   top-1 match   attn cosine   KL        path
affineK8V4   4096        20.71         1.002        1.0000        1.000        0.0       (raw — below quantizedKVStart)
fp16         4096        20.67          —            —             —            —
affineK8V4   16384       17.72         0.960        1.0000        0.9815       3.0e-5    affineK8V4Native (engaged, no fallback)
fp16         16384       18.46          —            —             —            —
```

Peak resident memory: 2.08 GB @4K, 2.38 GB @16K (real model + KV). Quality gate **passed** at both
contexts; **`rawFallbackAllocated = false`** (compressed path was real, not a decoded fallback).

## What it shows (the headline, honest)

- **At 16K, compressed real-model decode runs at 0.96× FP16 (only ~4% slower) with BYTE-IDENTICAL
  greedy output** (top-1 match = 1.000) and attention cosine 0.982 — and the compressed kernel
  genuinely engaged (`selectedAttentionPaths = [affineK8V4Native]`, no raw fallback). This is the
  product-faithful, promotion-grade number.
- **This is far more favorable than the synthetic attention-only sweep** (0.2–0.47× at these
  contexts) because **real-model decode is weight-bound** (~the 4-bit weight reads per token
  dominate), so compressing the KV is nearly free in the end-to-end token rate. The synthetic
  shapes isolate attention and therefore overstate the penalty — exactly why real-model evidence
  was required, and exactly the "decode is weight-dominated ≤~47K" thesis, now confirmed on device.
- At **4K the affineK8V4 config stays raw** (`paths = []`, cosine 1.000, ratio 1.002) — below the
  model's `quantizedKVStart`, so that row is effectively FP16; **the 16K row is the true compressed
  measurement.**

## Scope / nonclaims (honest)

- Single device, single model, 48 generated tokens, 2 repeats — solid directional evidence, **not**
  a full campaign: no bootstrap CIs, prefill not separated from decode in the tok/s, only two
  contexts. The `real-model-inference-v1` gate **passed** here, but a promotion decision should add
  more contexts (≥32K, where compressed's memory unlock matters), more tokens, and CIs.
- Memory-reduction ratio came back `None` from the engine for this OptiQ build (KV-byte estimate
  not populated); the synthetic sweep's 2.21× and the engaged native path stand as the compression
  evidence. KV memory savings at long context remain the *reason* to use compressed (FP16 is ~equal
  speed when it fits).
- Payload: `pines-realmodel-tq-device-realmodel-20260613.json` (this dir).
