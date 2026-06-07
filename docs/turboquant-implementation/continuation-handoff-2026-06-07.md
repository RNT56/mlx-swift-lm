# TurboQuant Continuation Handoff - 2026-06-07

This is the current pickup point for agents continuing the upstream-equivalent
TurboQuant work in `mlx-swift-lm` and the paired native `mlx-swift` kernels.

## Current Status

- The practical upstream-comparable route is still affine K8/V4: affine/Q8
  keys with 4-bit affine values. Upstream's visible README K8+V4 row at commit
  `6e928d7` is not the PolarWHT-V route; it is the mixed affine route using
  MLX quantized matmul and mixed quantized SDPA.
- Full `polarWHTV3` and value-only `hybridK8PolarWHTV3/V4` are implemented as
  experimental diagnostics, not default promotion candidates. The hybrid path
  can pass bounded logit quality but is far slower than affine K8/V4 locally.
- Promotion now fails closed unless the throughput row selects the expected
  native compressed path and the quality row, when required, also selects the
  expected native compressed path.
- Affine dynamic conversion now triggers at `offset >= quantizedKVStart`.
  Default affine K8/V4 starts at `16_384`, so 4K runs correctly report raw
  residency and no compressed native path.
- Dynamic conversion is materialized immediately after conversion: converted
  cache state is evaluated, the GPU stream is synchronized, and cache pressure
  is cleared. Conversion cost is reported under
  `dynamicCacheQuantizationSeconds`, usually in the prompt-prefill timing
  block rather than the first decode token.

## Key Local Files

- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Libraries/IntegrationTestHelpers/InferenceParityBenchmark.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Libraries/MLXLMCommon/TurboQuantTiming.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/tools/TurboQuantInferenceParity/main.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Tests/MLXLMTests/TurboQuantInferenceParitySparseVTests.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/Tests/MLXLMTests/KVCacheTests.swift`

## Verified Commands

These passed after the latest implementation changes:

```bash
swift test --filter TurboQuantInferenceParitySparseVTests
swift test --filter KVCacheTests/testAdaptiveTurboQuantStartsRawAndConvertsAtThreshold
swift test --filter KVCacheTests
python3 scripts/turboquant-polarwht-acceptance-gate.py --self-test
swift build -c release --product TurboQuantInferenceParity
```

Native K8/V4 microbenchmark command:

```bash
.build/release/TurboQuantNativeVxBenchmark \
  --contexts 16384,32768 \
  --value-bits 4 \
  --iterations 5 \
  --warmup 1 \
  --cooldown-ms 25 \
  --query-heads 16 \
  --kv-heads 8 \
  --head-dim 128 \
  --output artifacts/turboquant-hybrid-smoke-20260607/native-vx-qwen06b-k8v4.json
```

It reported K8/V4 native attention at about `0.94x` FP16 for 16K and `1.43x`
FP16 for 32K in the operator-only harness. Treat this as kernel regression
evidence, not real-model promotion evidence.

## Latest Artifacts

Artifact root:

```text
/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/artifacts/turboquant-hybrid-smoke-20260607
```

Important rows:

- `diagnostics-4k-g32-fp16-affinek8v4-fixed.json`: 4K affine K8/V4 correctly
  remains raw because the default quantized start is 16K. It reports `1.00x`
  resident KV compression and blocks promotion with
  `affine compressed native path was not selected`.
- `diagnostics-16k-quality-affinek8v4-threshold-fixed.json`: 16K affine K8/V4
  quality passes with top-1 `1.000`, KL `1.1265e-7`, p95 max-logit error
  `1.0859`, and cosine `0.9942`. This artifact predates the newest quality
  path fields, so do not use it to verify `qualitySelectedAttentionPaths`.
- `diagnostics-16k-g1-affinek8v4-quality-path-fields.json`: early schema smoke
  from the quality path field work. Rerun now to verify the current JSON fields.
- `diagnostics-16k-g32-fp16-affinek8v4-materialized-conversion.json`: 16K
  throughput after materialized dynamic conversion. FP16 measured `12.23`
  tok/s and affine K8/V4 measured `44.37` tok/s with
  `selectedAttentionPaths=["affineK8V4Native"]`,
  `residentKVCompressionRatio=2.13`, and prompt
  `dynamicCacheQuantizationSeconds=2.455`. This is a single ordered sample and
  did not run quality in the same report, so it is not a promotion result.
- `diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion.jsonl`:
  zero-byte failed attempt. Rerun this exact combined gate first.

## First Rerun

Rerun a combined 16K quality plus 32-token throughput report now that conversion
materialization and quality path diagnostics are in place:

```bash
swift build -c release --product TurboQuantInferenceParity

TQ_QUALITY_PRINT_CACHE_DIAGNOSTICS=1 \
.build/release/TurboQuantInferenceParity \
  --model-dir /Users/mt/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068 \
  --contexts 16384 \
  --generate-tokens 32 \
  --throughput-repeats 1 \
  --throughput-cooldown 0.25 \
  --configs fp16,affineK8V4 \
  --quality-gates \
  --quality-contexts 16384 \
  --quality-cooldown 0.5 \
  --turboquant-timing \
  --diagnostics-output artifacts/turboquant-hybrid-smoke-20260607/diagnostics-16k-g32-fp16-affinek8v4-quality-materialized-conversion-rerun.json
```

Expected pass conditions for this rerun:

- `throughput[].label=="affineK8V4"` has
  `selectedAttentionPaths=["affineK8V4Native"]`.
- `throughput[].promotionGate.qualitySelectedAttentionPaths` includes
  `affineK8V4Native`.
- `throughput[].residentKVCompressionRatio > 1.0`.
- `throughput[].promotionBlockReasons` has no native-path, raw-fallback, or
  quality-path blockers. A speed floor blocker is acceptable only if the run is
  noisy and should trigger repeated randomized runs.
- `quality[].label=="affineK8V4"` includes `selectedAttentionPaths` and
  `codecCounts`.
- `promptPrefillTiming.dynamicCacheQuantizationCalls == 1` for affine K8/V4 and
  generation dynamic quantization is zero.

## Next Development Work

1. Rerun the combined 16K gate above. The previous attempt wrote an empty JSONL.
2. Run randomized repeated 16K and 32K affine K8/V4 reports:

```bash
.build/release/TurboQuantInferenceParity \
  --model-dir /path/to/mlx-model \
  --contexts 16384,32768 \
  --generate-tokens 32 \
  --throughput-repeats 3 \
  --throughput-cooldown 0.25 \
  --randomize-throughput-order \
  --configs fp16,affineK8V4 \
  --quality-gates \
  --quality-contexts 16384,32768 \
  --quality-cooldown 0.5 \
  --turboquant-timing \
  --diagnostics-output artifacts/turboquant-current/affine-k8v4-16k-32k.json
```

3. Build the same-machine upstream reproduction harness and stop using
   `TQ_POLARWHT_SKIP_UPSTREAM=1` for promotion runs.
4. Compare local affine K8/V4 against reproduced upstream K8+V4, current
   `affineInt4`, and FP16 at 32K, then 64K/128K where FP16 fits.
5. Keep PolarWHT hybrid promotion disabled by default. Only set
   `TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION=1` for deliberate experiments
   that report the experimental blocker and compare against dense affine K8/V4.

## Nonclaims

- Do not claim `0.98x` FP16 for this branch. Upstream's visible pinned README
  K8+V4 row is about `0.72x` FP16.
- Do not claim the single 16K `44.37` tok/s affine row as a product result. It
  is useful evidence that materialized conversion removed first-decode hidden
  work, but it lacks a same-report quality gate and repeated randomized samples.
- Do not promote full PolarWHT K/V or hybrid K8+PolarWHT-V. They remain
  diagnostic until native path, quality, resident memory, fallback, and
  same-machine upstream comparison gates all pass.
