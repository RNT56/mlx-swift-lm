# Current TurboQuant Paths And Benchmarks

This document is the runnable inventory for the implemented `mlx-swift-lm`
TurboQuant surfaces. It covers production wiring, guarded experiments, benchmark
entry points, and the evidence needed before Pines can promote a mode.

## Runtime Paths

| Path | Strategy / config | Current role | Promotion state |
| --- | --- | --- | --- |
| FP16 raw SDPA | `KVCacheStrategy.none` | Short-context baseline and reference quality path. | Production baseline when memory admission allows. |
| Legacy TurboQuant | `.turboQuant` with `turbo8`, `turbo4v2`, or `turbo3_5` | Capacity route using the Polar/QJL compressed cache family. | Supported as capacity/diagnostic route, not speed parity. |
| Adaptive TurboQuant | `.adaptiveTurboQuant` | Raw-first compatibility routing. Keeps raw SDPA for short admitted contexts and routes to compressed cache when raw KV is not practical. | Compatibility route. |
| Hybrid TurboQuant selector | `.hybridTurboQuant` plus hot/cold diagnostics | Raw hot window plus selected cold-block metadata and selector diagnostics. | Selector/cache surface is testable; native segmented compressed math still needs evidence before product promotion. |
| MLX affine Q8 | `.mlxAffine` / `mlxAffine-q8` | Community-comparable MLX-native affine route. | Benchmark/reference route. |
| Affine K8/V4 | `.affineK8V4` / `affineK8V4` | Main speed/quality candidate and the closest match to upstream's published K8+V4 route: K stays affine 8-bit, V uses 4-bit affine lanes. | Wired and benchmarked; promotion still requires per-model quality, no fallback, and equal-machine long-context evidence. |
| Affine K8/V3 | `.affineK8Vx`, `turboQuantValueBits = 3` / `affineK8V3` | Lower-V experiment with report labeling and quality gates. | Guarded experiment. |
| Affine K8/V3 optimized | `.affineK8Vx`, V3 middle layers with K8/V4 protected edge layers / `affineK8V3-optimized` | Guarded Qwen3.5/Qwen3.6 lower-V candidate; current default protects 5 first and 5 last layers. | Guarded experiment; requires 32K+ task and logit gates before promotion. |
| Affine K8/V2 | `.affineK8Vx`, `turboQuantValueBits = 2` / `affineK8V2` | Lower-V experiment with report labeling and quality gates. | Guarded experiment. |
| Affine K8/V2 calibrated | `KVLayerPolicy` JSON plus `affineK8V2-calibrated` | Per-layer mixed policy candidate that restores measured harmful layers to K8/V4. | Guarded experiment; dense V2 remains non-promotable. |
| Affine K8/V2 residual-r1 | `KVLayerCodec.affineK8VxResidual(valueBits: 2, residualsPerGroup: 1)` / `affineK8V2-residual-r1` | V2 values plus one FP residual lane per value group and compact `uint8` lane metadata. | Guarded experiment; default group32 residual-r1 is not memory-promotable without measured evidence or further compaction. |
| Affine int4 | `.affineInt4` / `affineInt4` | Fast comparison route for native affine 4-bit KV. | Benchmark route; quality gates decide usability per model. |
| PolarWHT K/V | `polarWHTV3`, `polarWHTReferenceV3` | Full low-bit PolarWHT K and V diagnostic path. | Diagnostic only; low-bit K remains quality-sensitive. |
| Hybrid K8 + PolarWHT-V | `hybridK8PolarWHTV3`, `hybridK8PolarWHTV4` | Experimental value-only PolarWHT storage with affine K8 scoring and WHT-pulled V accumulation. | Quality-passing locally, but speed-regressed; blocked from default promotion unless `TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION=1` is set for a deliberate promotion run. |
| Sparse-V | threshold, top-k, cumulative mass, hybrid cumulative-plus-top-k, blockThreshold, pageTopK, candidateSparse | Explicit value-token, page-sparse, and candidate-sparse modes over the mixed affine decode path. | Rejected for promotion on current Qwen3.5 evidence; native modes remain explicit proof/debug diagnostics only. |

## Implemented Optimizations

- Exact prefill: prefill logits are produced by the exact path before cache
  conversion.
- Quantized start threshold: affine throughput routes start compression at
  `16_384` tokens by default so short caches stay on raw/cheap paths. Dynamic
  conversion triggers once the cache is at or beyond that threshold, so a 16K
  quality/decode gate exercises compressed attention.
- Materialized dynamic conversion: after conversion, the converted cache state is
  evaluated, the GPU stream is synchronized, and cache pressure is cleared.
  Conversion work is reported under `dynamicCacheQuantizationSeconds` so hidden
  first-decode conversion cost is visible in prompt-prefill timing.
- K8/lower-V policy: K can stay high precision while V uses 4, 3, or 2 bits.
- Residual V2 policy: `affineK8VxResidual` stores normal K8/V2 affine KV plus
  one `uint8` residual lane and one FP residual value per V group. The current
  route applies a guarded residual correction pass and records the residual
  attention path in diagnostics. Updated assumption: group32 residual-r1 is a
  quality probe, not a promotion candidate, because compact lane metadata still
  leaves memory near the K8/V4 gate. Promotion needs a fused native residual
  kernel and either group64 residuals, bitpacked lanes, smaller residual values,
  or measured runtime memory below `0.90x` dense K8/V4.
- Per-layer and boundary policy: protected boundary layers and per-layer mixed KV
  policy are represented in profiles and reports.
- Long-context scheduling: long compressed prefill/decode periodically
  synchronizes and clears cache pressure; decode-shaped long K8/Vx attention uses
  smaller split blocks to avoid GPU watchdog failures.
- Sparse-V modes: threshold, top-k, cumulative-mass, and hybrid
  cumulative-plus-top-k selection execute in the native mixed affine decode path
  and select from normalized softmax weights, not raw logits. Dense compressed AV
  remains the reference/fallback.
- Sparse-V TopK has a native single-block fused decode route for `qLen == 1`
  and `activeBlocks == 1` so 512-token TopK probes avoid the split
  score/select/partials/reduce launch pipeline. Multi-block TopK uses the split
  score/candidate compaction path and remains an explicit measured mode.
- Sparse-V `pageTopK` has three explicit native scorer routes. The sampled
  scorer keeps native `kernelKind` 13. The cached-summary two-dispatch scorer
  reports native `kernelKind` 14. The fused cached-summary decode route reports
  native `kernelKind` 15 for valid summaries and page `topK <= 8`. If summaries
  are absent or unsafe because of rotating/ring layout state, runtime routing
  falls back to sampled page scoring, not dense fallback.
  `TURBOQUANT_SPARSE_V_PAGE_RECENT_TOKENS=<n>` is an off-by-default research
  knob that keeps an exact recent-token floor in addition to selected older
  pages; recency variants report kernel kinds 16 (sampled), 17 (cached
  two-dispatch), or 18 (fused cached-summary).
- Quality gates: `TurboQuantInferenceParity --quality-gates` compares real-model
  single-token compressed decode logits against FP16 after the same cache
  conversion path used by generation. Diagnostics JSON records the candidate
  quality path (`selectedAttentionPaths`, `codecCounts`, fallback fields), and
  promotion blocks compressed candidates when the quality pass did not select the
  matching native compressed path.

Continuation handoff for the current work is recorded in
[TurboQuant Continuation Handoff - 2026-06-07](continuation-handoff-2026-06-07.md).
Use that file as the first pickup point for exact commands and artifacts.

## Upstream-Parity Finding

The `arozanov/turboquant-mlx` repository at commit `6e928d7` has two materially
different K8+V routes:

- The README's Qwen2.5-7B K8+V4 result is the mixed affine path: Apple
  `mx.quantized_matmul` for K and V through a forked
  `mixed_quantized_scaled_dot_product_attention`.
- The PolarWHT value route is represented by `hybrid_attention.py`, which uses
  `mx.quantized_matmul` for K scores plus `sparse_v_matvec` for packed
  PolarWHT V. Upstream labels this as experimental scaffold, not the published
  K8+V4 speed row.

Latest local 4K Qwen3-0.6B release smoke with corrected resident-KV accounting
on June 7, 2026. `affineK8V4` starts compression at 16K by default, so this 4K
row is a short-context raw-runtime smoke and not compressed affine evidence:

| Config | Decode tok/s | vs FP16 | vs K8/V4 | KV memx | Path |
| --- | ---: | ---: | ---: | ---: | --- |
| `fp16` | 55.63 | 1.000x | 1.041x | 1.00 | raw SDPA |
| `affineK8V4` | 53.44 | 0.961x | 1.000x | 1.00 | raw SDPA; blocked with `affine compressed native path was not selected` |
| `hybridK8PolarWHTV4` | 3.60 | 0.061x | 0.065x | 1.33 | resident affine K8 + WHT-pulled V |
| `hybridK8PolarWHTV4` tail | 1.93 | 0.041x | n/a | 1.33 | segmented affine K8 + WHT-pulled V tail |
| `hybridK8PolarWHTV4` generic | 1.72 | 0.040x | 0.039x | 1.33 | separate `quantizedMM` scores + PolarWHT AV |

The PolarWHT-V hybrid path passes the 4K logit gate but is not the route that
matches upstream's published K8+V4 performance. Promotion reports therefore
block `hybridK8PolarWHTV3/V4` by default with an explicit experimental reason.

Bounded 16K Qwen3-0.6B release evidence on June 7, 2026:

| Config | Generated tokens | Decode tok/s | vs FP16 | KV memx | Path / quality |
| --- | ---: | ---: | ---: | ---: | --- |
| `fp16` | 32 | 12.23 | 1.000x | 1.00 | raw SDPA |
| `affineK8V4` | 32 | 44.37 | 3.628x | 2.13 | `affineK8V4Native=28`; prompt `dynamicCacheQuantizationSeconds=2.455`; quality was not run in the same report |
| `hybridK8PolarWHTV4` | 8 | 0.78 | 0.070x | 1.33 | `metalHybridK8PolarWHTValue=28`; 16K quality passed, cosine 0.9901 |

Separate 16K affine K8/V4 quality evidence passed with top-1 `1.000`, KL
`1.1265e-7`, p95 max-logit error `1.0859`, and cosine `0.9942`. The latest
single-sample materialized-conversion throughput artifact did not include the
quality gate, so it is evidence that hidden first-decode conversion was moved
out of generation, not a promotion result.

The previous combined 16K quality plus 32-token throughput attempt wrote an
empty JSONL:

```text
/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/artifacts/turboquant-hybrid-smoke-20260607/diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion.jsonl
```

Rerun the combined gate from
[TurboQuant Continuation Handoff - 2026-06-07](continuation-handoff-2026-06-07.md)
before making any promotion or speed claim.

The attempted 32K three-config smoke on the same model was terminated after
eight minutes without producing a completed affine/hybrid row; use bounded 16K
and then longer amortized 32K runs for promotion evidence.

## Real-Model Inference Parity

Use this for the evidence that matters most: full model inference rather than an
attention-only kernel microbenchmark.

```bash
swift build --product TurboQuantInferenceParity -c release

.build/release/TurboQuantInferenceParity \
  --model-dir /path/to/mlx-model \
  --contexts 20000,32768,65536,131072 \
  --generate-tokens 16 \
  --throughput-repeats 3 \
  --throughput-cooldown 0.25 \
  --randomize-throughput-order \
  --configs fp16,affineK8V4,affineK8V3,affineK8V2,mlxAffine-q8,affineInt4,turbo4v2,turbo3_5,turbo8 \
  --quality-gates \
  --quality-cooldown 0.5 \
  --quality-contexts 20000,32768,65536,131072
```

When FP16 does not fit at a requested context, lower-V and Sparse-V experiments
can use dense K8/V4 as the real-model reference:

```bash
.build/release/TurboQuantInferenceParity \
  --model-dir /path/to/mlx-model \
  --contexts 20000,32768,65536,131072 \
  --generate-tokens 8 \
  --throughput-repeats 3 \
  --throughput-cooldown 0.25 \
  --randomize-throughput-order \
  --configs affineK8V4,affineK8V3-protectedK8V4-edge5,affineK8V2,affineK8V2-protectedK8V4-edge8,affineK8V2-residual-r1 \
  --quality-gates \
  --quality-cooldown 0.5 \
  --quality-contexts 20000,32768,65536,131072 \
  --quality-reference-config affineK8V4 \
  --emit-cache-policy-summary \
  --diagnostics-output artifacts/turboquant-lowerv2/run.json
```

Available config labels:

- `affineK8V2-calibrated`
- `affineK8V2-residual-r1`
- `affineK8V2-calibrated-residual-r1`
- `affineK8V2-protectedK8V4-edge8`
- `affineK8V2-protectedRaw-edge4`

Generate a deterministic calibrated policy seed before a real-model run:

```bash
swift build --product TurboQuantLowerV2Calibrate -c release

.build/release/TurboQuantLowerV2Calibrate \
  --layer-count 36 \
  --edge-size 8 \
  --residual-r1 \
  --output-policy artifacts/turboquant-lowerv2/lower-v2-policy.json \
  --output-summary artifacts/turboquant-lowerv2/lower-v2-policy.md

.build/release/TurboQuantInferenceParity \
  --model-dir /path/to/mlx-model \
  --contexts 20000,32768,65536,131072 \
  --throughput-repeats 3 \
  --throughput-cooldown 0.25 \
  --randomize-throughput-order \
  --configs affineK8V4,affineK8V2-calibrated-residual-r1 \
  --kv-layer-policy-json artifacts/turboquant-lowerv2/lower-v2-policy.json \
  --quality-gates \
  --quality-cooldown 0.5 \
  --quality-reference-config affineK8V4
```

```bash
.build/release/TurboQuantInferenceParity --list-configs
```

Supported aliases include `k8v4`, `k8v3`, `k8v3_optimized`, `k8v2`, `q8`,
`affine_int4`, `turbo35`, and `turbo4`.

Real-model Sparse-V overrides are available for TurboQuant configs:

```bash
.build/release/TurboQuantInferenceParity \
  --model-dir /path/to/mlx-model \
  --contexts 512,2048,8192,32768 \
  --generate-tokens 8 \
  --configs affineK8V4,turbo4v2 \
  --sparse-v topK \
  --sparse-v-top-k 128 \
  --quality-gates \
  --quality-cooldown 0.5 \
  --quality-reference-config affineK8V4 \
  --diagnostics-output artifacts/sparsev-realmodel.json
```

Sparse-V and CandidateSparse are no longer part of default acceptance or runtime
promotion. `TurboQuantSparseValuePolicy.auto` and profile sparse defaults resolve
to off; use the CLI overrides above only for explicit diagnostics.

Supported override modes are `off`, `threshold`, `topK`, `cumulativeMass`,
`hybridCumulativeMassTopK`, `blockThreshold`, `pageTopK`, and `candidateSparse`, with
`--sparse-v-threshold`, `--sparse-v-top-k`, `--sparse-v-cumulative-mass`, and
`--sparse-v-max-top-k` as mode parameters. `candidateSparse` also accepts
`--sparse-v-recent-tokens` and `--sparse-v-candidate-pages`, using
`--sparse-v-top-k` as the older-token budget.

## Current Sparse-V PageTopK Evidence

The fused cached page-summary implementation is represented in core operator
artifacts at
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/sparse-v-page-fused-20260602`.
The operator matrix uses qHeads `4`, kvHeads `1`, head dimension `128`, and
`turbo4v2`.

| Context | Retained pages | Sampled 13 ms | Cached 14 ms | Fused 15 ms | Fused/sample | Fused/cached | Skip ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 1 | 0.872 | 0.894 | 0.782 | 1.11x | 1.14x | 98.44% |
| 32768 | 2 | 1.379 | 1.303 | 1.255 | 1.10x | 1.04x | 96.88% |
| 32768 | 4 | 2.294 | 2.312 | 2.252 | 1.02x | 1.03x | 93.75% |
| 32768 | 8 | 4.293 | 4.438 | 4.330 | 0.99x | 1.02x | 87.50% |
| 65536 | 1 | 1.006 | 0.862 | 0.847 | 1.19x | 1.02x | 99.22% |
| 65536 | 2 | 1.401 | 1.420 | 1.323 | 1.06x | 1.07x | 98.44% |
| 65536 | 4 | 2.452 | 2.357 | 2.329 | 1.05x | 1.01x | 96.88% |
| 65536 | 8 | 4.673 | 4.488 | 4.539 | 1.03x | 0.99x | 93.75% |
| 131072 | 1 | 1.057 | 1.073 | 0.883 | 1.20x | 1.21x | 99.61% |
| 131072 | 2 | 1.653 | 1.526 | 1.444 | 1.14x | 1.06x | 99.22% |
| 131072 | 4 | 2.688 | 2.676 | 2.645 | 1.02x | 1.01x | 98.44% |
| 131072 | 8 | 5.215 | 5.291 | 5.087 | 1.03x | 1.04x | 96.88% |

This is useful native routing evidence, but not a promotion result. Fused
`kernelKind` 15 removes the separate cached page-score dispatch and improves the
operator path in most measured rows, but `TQ_MODEL_DIR` was not set for this
workspace run, so no real-model Sparse-V acceptance rows were produced. Dense
K8/V4 remains the default production routing and Sparse-V/pageTopK stays
explicit only.

## Current CandidateSparse Evidence

`candidateSparse` is active as an explicit research mode and now supports
pinned-prefix Qwen caches by building key candidate sketches in logical page
order. The native path reports `kernelKind` 19 and the KV cache reports
`keyCandidateSketchAvailable=true`; unsafe ring-offset layouts still route to
dense compressed fallback with requested-but-inactive diagnostics.

Latest local Qwen evidence uses
`mlx-community/Qwen3.5-2B-4bit` from the local Hugging Face cache and artifacts
under
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/candidate-sparse-20260603`.

| Context | Config | Decode tok/s | Dense K8/V4 tok/s | Ratio vs K8/V4 | Active layers | Kernel kind | Skip ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | `candidateSparse r256 p1 older64` | 4.13 | 42.96 | 0.096x | 6/6 | 19 | 38.46% |
| 2048 | `candidateSparse r512 p2 older128` | 4.10 | 46.47 | 0.088x | 6/6 | 19 | 68.81% |
| 8192 | `candidateSparse r512 p2 older128` | 2.06 | 42.26 | 0.049x | 6/6 | 19 | 92.19% |

This rejects the current prototype architecture for promotion. Even with 92%
value-token skip, real-model decode remains far slower than dense K8/V4 because
the kernel still pays expensive per-query-head candidate scoring, token
selection, and compressed K/V decode overhead.

The follow-up cooperative GQA fused kernel, `kernelKind` 20, is also rejected as
a promotion path. On Qwen3.5-2B-4bit at context 2048, `affineK8V4` measured
16.32 tok/s, prototype `candidateSparse r512 p2 older128` measured 2.60 tok/s
with `kernelKind` 19, and cooperative GQA fused `candidateSparse r512 p2
older128` measured 2.34 tok/s with `kernelKind` 20. It adds approximation risk
by sharing candidate selection across GQA repeats and is slower than the
prototype. `kernelKind` 20 is therefore opt-in only through
`TURBOQUANT_CANDIDATE_SPARSE_FUSED=1`; default explicit `candidateSparse` routes
to `kernelKind` 19.

Dense K8/V4 remains the default production route. `candidateSparse` stays
explicit for proof/debug runs only, and promotion remains blocked unless a new
architecture passes quality plus `>= 1.10x` p50 decode throughput.

## Latest K8/Vx Real-Model Baseline

Latest local grouped run:

- artifact root:
  `/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-current-20260603T192606Z`
- model:
  `/Users/mt/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-4bit/snapshots/674aaa7240b91e8012fcad5d791b7dfe5ba90207`
- workload: `generateTokens=16`, `throughputRepeats=3`,
  `throughputCooldown=0.25`, `qualityCooldown=0.5`
- completed groups: `affine` and `dense` exited `0`; `sparse` was stopped
  after the first 16K row proved requested Sparse-V was inactive.

Affine 32K rows from `TurboQuantInferenceParity`:

| Config | Decode tok/s | Ratio vs FP16 | Est. KV reduction | Quality | KL | P95 max abs |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `fp16` | 38.26 | 1.000 | 1.00x | reference | n/a | n/a |
| `affineInt4` | 29.24 | 0.764 | 2.67x | pass | 0.00000461 | 1.188 |
| `affineK8V3-protectedK8V4-edge5` | 28.41 | 0.743 | 2.22x | pass | 0.00000459 | 1.073 |
| `affineK8V3-protectedRaw` | 28.14 | 0.735 | 1.88x | pass | 0.00000314 | 1.152 |
| `affineK8V4` | 27.68 | 0.723 | 2.13x | pass | 0.00000336 | 0.957 |
| `affineK8V3-optimized` | 26.92 | 0.703 | 2.22x | pass | 0.00000459 | 1.073 |
| `affineK8V3` | 25.51 | 0.667 | 2.29x | fail | 0.0000221 | 2.875 |
| `affineK8V2-calibrated` | 25.07 | 0.655 | 2.23x | pass | 0.00000559 | 1.726 |
| `affineK8V2` | 23.18 | 0.606 | 2.46x | fail | 0.000346 | 4.727 |
| `mlxAffine-q8` | 22.13 | 0.578 | 1.78x | pass | 0.000000140 | 0.297 |
| `affineK8V2-residual-r1` | 12.06 | 0.315 | 2.46x | fail | 0.000108 | 4.430 |

Dense compressed 32K rows:

| Config | Decode tok/s | Ratio vs FP16 | Est. KV reduction | Peak active memory |
| --- | ---: | ---: | ---: | ---: |
| `fp16` | 65.27 | 1.000 | 1.00x | 1.979 GB |
| `turbo3_5` | 31.25 | 0.479 | 2.67x | 2.686 GB |
| `turbo4v2` | 28.43 | 0.436 | 2.56x | 2.692 GB |
| `turbo8` | 17.44 | 0.267 | 1.56x | 2.793 GB |

Current interpretation:

- `affineK8V4` remains the quality anchor and default promotion reference.
- `affineInt4` and protected K8/V3 are the only throughput candidates worth
  advancing from this run. They must keep quality/task gates before promotion.
- Raw K8/V3, raw K8/V2, V2 protected with too-small edges, and residual V2 are
  rejected for promotion by the 32K quality gates.
- Dense compressed `turbo3_5`/`turbo4v2`/`turbo8` are capacity/debug routes, not
  speed parity routes, on this evidence.
- Sparse-V is not product-promotable: 4K/8K requested sparse rows were active
  but only about `0.16x-0.17x` FP16 decode, and the first 16K row had
  `sparseActiveLayerCount=0`, no skipped tokens, and dense fallback diagnostics.

## One-Command Current Matrix

The script below builds and runs the current core and real-model surfaces, then
writes a timestamped artifact directory.

```bash
TQ_MODEL_DIR=/path/to/mlx-model \
scripts/run-turboquant-current-benchmarks.sh
```

Useful overrides:

```bash
TQ_CORE_CONTEXTS=8192,16384,32768,65536,131072 \
TQ_CORE_PRESETS=turbo4v2,turbo3_5,turbo8 \
TQ_CORE_PATHS=auto,affine-k8v4-native,affine-k8vx-native,online-fused,tiled-online-fused,two-stage \
TQ_CORE_LAYOUT_VERSION=4 \
TQ_BENCH_STRICT=1 \
TQ_REAL_CONTEXTS=32768,65536,131072 \
TQ_REAL_THROUGHPUT_REPEATS=3 \
TQ_REAL_THROUGHPUT_COOLDOWN=0.25 \
TQ_REAL_QUALITY_COOLDOWN=0.5 \
TQ_REAL_CONFIGS=fp16,affineK8V4,affineK8V3,affineK8V2,mlxAffine-q8,affineInt4 \
TQ_QUALITY_CONTEXTS=32768,65536 \
scripts/run-turboquant-current-benchmarks.sh
```

The current runner passes `--strict-configs` to real-model parity, writes
one diagnostics JSON plus streamed JSONL samples per real-model config group,
records real-model throughput/quality cooldowns, randomizes repeated throughput
rows by default, and exits non-zero under `TQ_BENCH_STRICT=1` if any logged
command fails. The default `TQ_REAL_CONFIGS=all` groups `affine`, `dense`, and
`sparse` in separate processes; quality gates run on the affine group by
default. Sparse grouped runs default to `4096,8192` because 16K currently
requests Sparse-V but measures inactive fallback unless explicitly overridden.
Layout V5/V6 sweeps must remain explicit via `TQ_CORE_ENABLE_LAYOUT_V5=1`;
Layout V4 stays the production default for new benchmark rows.

If `TQ_MODEL_DIR` is absent the script still runs core operator JSON and records
the real-model section as skipped.

## Acceptance Matrix

Use this narrower runner for the TurboQuant acceptance matrix. It compares
real-model dense K8/V4 against K8/V3, optimized V3, K8/V2, and explicitly
requested Sparse-V modes with
`TurboQuantInferenceParity --quality-reference-config affineK8V4`, then keeps
the synthetic Sparse-V proof rows through `TurboQuantQwenProof` for
attention-kernel diagnostics.

```bash
TQ_MODEL_DIR=/path/to/mlx-model \
scripts/run-turboquant-acceptance-matrix.sh
```

The generated `acceptance-matrix.md` records requested-vs-active Sparse-V
diagnostics and native kernel kinds, including pageTopK rows for
`TQ_ACCEPTANCE_PAGE_TOP_KS` (default `1,2,4,8`). A 2048 guarded row should
continue to show requested Sparse-V with zero active sparse layers and dense
fallback; synthetic Sparse-V wins do not certify product promotion without
real-model speed and quality gates.

## App-Hosted Synthetic Attention

`TurboQuantBench` remains the app-hostable synthetic attention-shape benchmark.
It is useful for A-series kernel regressions and memory diagnostics, but it does
not certify product quality by itself. Rows retain measured/warmup iteration
counts, `cooldownMilliseconds`, FP16/plain speed ratios, dense/compressed byte
counts, and memory reduction ratios.

```bash
TQ_BENCH=1 swift test --filter TurboQuantBenchSuite
```

Optional hybrid selector diagnostics:

```bash
TQ_BENCH=1 TQ_BENCH_HYBRID=1 swift test --filter TurboQuantBenchSuite
```

Sparse-V and lower-V proof grid tests:

```bash
swift test --filter TurboQuantSparseVBenchmarkPlanSuite
```

Native K8/Vx decode microbenchmark:

```bash
swift build --product TurboQuantNativeVxBenchmark -c release

.build/release/TurboQuantNativeVxBenchmark \
  --contexts 32768,65536 \
  --value-bits 4,3,2 \
  --iterations 7 \
  --warmup 2 \
  --cooldown-ms 25 \
  --output artifacts/turboquant-native-vx.json
```

The native Vx report records one FP16 raw-SDPA baseline per context and includes
`speedRatioToFP16`, `speedRatioToFP16P95`, `memoryBytesSavedVsFP16`,
`memoryReductionRatioToFP16`, and `cooldownMilliseconds` on every K8/Vx row.

Qwen proof Sparse-V smoke/profiling examples:

```bash
swift build --product TurboQuantQwenProof -c release

.build/release/TurboQuantQwenProof \
  --profiles qwen3.5-2b \
  --schemes turbo4v2 \
  --contexts 32768 \
  --query-lengths 1 \
  --iterations 3 \
  --warmup 1 \
  --sparse-v-mode top-k \
  --sparse-v-top-k 256

.build/release/TurboQuantQwenProof \
  --profiles qwen3.5-2b \
  --schemes turbo4v2 \
  --contexts 32768 \
  --query-lengths 1 \
  --iterations 3 \
  --warmup 1 \
  --sparse-v-mode cumulative-mass \
  --sparse-v-cumulative-mass 99.5

.build/release/TurboQuantQwenProof \
  --profiles qwen3.5-2b \
  --schemes turbo4v2 \
  --contexts 32768 \
  --query-lengths 1 \
  --iterations 3 \
  --warmup 1 \
  --sparse-v-mode hybrid \
  --sparse-v-hybrid-mass 99.5 \
  --sparse-v-max-top-k 256
```

## Required Promotion Evidence

Do not mark a path `Verified` or `Certified` until the exact tuple has:

- real-model throughput and quality gate output from `real-model-inference-v1`;
- quality diagnostics that show the candidate quality pass selected the expected
  native compressed path;
- memory and fallback diagnostics for the same model/profile/context;
- no hidden full-cache decompression or unbudgeted fallback allocation;
- physical iOS app-host evidence for product claims on iPhone;
- compatibility-pair status that remains non-green until all required logs exist.

Current expected posture: K8/V4 is the quality-preserving anchor. `affineInt4`
and protected K8/V3 are the only current throughput candidates to keep
advancing. Raw K8/V3, raw K8/V2, residual V2, and Sparse-V are
implementation-complete enough to measure, but remain non-promoted until
real-model quality, task, memory, and throughput evidence proves they beat
dense K8/V4 safely for the target device.

## Fused pageTopK Real-Model Outcome

Fused cached-summary pageTopK now reaches the intended native path in real-model
decode. The proof artifact is
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-acceptance-sparse-v-qwen35-2b-page-fused-diagnostic2-20260602T202338Z/pageTopK1.json`:
all 6 Sparse-V layers report `nativeKernelKind=15`,
`keyPageSummaryAvailable=true`, and summary shape `1x2x2x4`.

The bounded Qwen3.5-2B-4bit throughput artifact is
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-acceptance-sparse-v-qwen35-2b-page-fused-final-20260602T202406Z/pageTopK1.out`.
It completed throughput through 8192 before manual termination during the 2048
quality row:

| Context | Dense K8/V4 tok/s | Fused pageTopK1 tok/s | Ratio vs K8/V4 | Skip ratio |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 72.76 | 6.53 | 0.090x | 0.78% |
| 2048 | 81.52 | 5.74 | 0.070x | 75.05% |
| 8192 | 76.27 | 6.04 | 0.079x | 93.75% |

Completed quality artifact:
`/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-acceptance-sparse-v-qwen35-2b-page-fused-quality-20260602T202641Z/pageTopK1-quality.json`.
The 512 quality gate fails: top-1 `1.0`, KL `0.069922`, p95 abs error
`3.34375`, cosine `0.96738`.

Conclusion: fused pageTopK is implemented and measurable, but rejected for
promotion. It is slower than dense K8/V4 in real-model decode and fails the p95
quality gate. Keep dense K8/V4 as the default route; keep pageTopK explicit for
research and diagnostics only. The optional recent-token floor is a quality
recovery experiment, not a default route, and needs fresh real-model quality and
throughput artifacts before it can change promotion status.
