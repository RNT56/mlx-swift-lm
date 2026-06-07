# External Port Optimization Map

Last updated: 2026-06-07.

This is the continuation map for bringing the local `mlx-swift-lm` TurboQuant
work up to the level shown by the current MLX, vLLM Metal, and Metal-kernel
ports. It is intentionally grounded in the latest local real-model benchmark
artifacts, not synthetic-only operator wins.

Local evidence source:

```text
/Users/mt/Programming/Schtack/mlx-forks/artifacts/turboquant-current-20260603T192606Z
/Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm/artifacts/turboquant-hybrid-smoke-20260607
```

Pinned upstream analysis:

```text
/Users/mt/Programming/Schtack/mlx-forks/TURBOQUANT_MLX_OPTIMIZATION_ANALYSIS_2026-06-04.md
```

## Bottom Line

The current implementation has useful candidate pieces, but it is not yet at
the external-port spec for speed parity plus real memory savings across context
lengths.

- June 7 implementation work fixed several measurement blockers: affine
  resident-KV estimates now respect `quantizedKVStart`, dynamic conversion
  triggers at `offset >= quantizedKVStart`, conversion is materialized before
  decode, and quality/promotion diagnostics now track selected native paths.
- The next required action is to rerun the combined 16K affine K8/V4
  quality-plus-throughput gate recorded in
  `docs/turboquant-implementation/continuation-handoff-2026-06-07.md`. The
  previous combined attempt wrote an empty JSONL, so the fast single-sample
  materialized-conversion row is not promotion evidence.
- `affineK8V4` and `affineInt4` are the only credible current bases for a
  product path. They pass 32K quality gates and reduce steady active memory, but
  they are still below equal-context FP16 decode speed at long context.
- Dense `turbo3_5`, `turbo4v2`, and `turbo8` are capacity/debug paths today,
  not optimization paths. At 32K they are much slower than FP16 and increase
  measured active memory because the path pays conversion/dequantization and
  transient costs.
- Sparse-V has diagnostic value only. It can be active at 4K/8K, but the active
  real-model rows are roughly 0.16x to 0.17x FP16 speed, and the 16K row fell
  back inactive.
- Synthetic attention-only benchmarks are no longer acceptable promotion
  evidence. They remain useful for kernel regression and shape coverage.

The rewrite should not be framed as a direct copy of `arozanov/turboquant-mlx`.
That pinned analysis shows the checked-in `4.6x` claim is real as packed
3-bit-per-vector payload math, while the checked README table at commit
`6e928d7` reports K8/V4 at `25.84 tok/s` versus `35.75 tok/s` FP16, or roughly
0.72x FP16 speed. It also shows the most distinctive upstream V optimization,
WHT pull-out, is codec-specific: it works because upstream V is WHT/codebook
encoded. The current Swift value path is affine-quantized, so sparse skipping
can transfer, but the WHT-linearity trick does not transfer unless the value
codec changes.

The Swift path also is not a plain fallback implementation. It already has
native/tiled/online fused routing, GQA block partials, query rotation hoisting,
sparse selection modes, and page/candidate summaries. The rewrite target is
therefore narrower: remove materialized/intermediate work from real-model hot
paths, make packed affine decode/pre-fill kernels direct, audit GQA fallbacks
for repeated KV allocation, and prove memory with real resident-byte metrics.

If the goal is full upstream parity with `arozanov/turboquant-mlx`, the correct
shape is an additive `polarWHT` path, not a mutation of the current affine
routes. That path needs its own codec identifiers, packed Lloyd-Max centroid
storage, deterministic WHT signs, WHT pull-out kernels, lifecycle reporting,
and fail-closed acceptance. The current `affineK8V4` and `affineInt4` paths
remain the production fallbacks while `polarWHT` is experimental.

As of June 7, the PolarWHT and hybrid K8+PolarWHT-V identifiers, storage, and
native routing scaffolds exist locally, but the path is not promotable. Local
bounded runs show quality can pass while speed is far behind affine K8/V4, and
the upstream README K8+V4 row being compared against is the mixed affine path.
Keep `TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION` unset unless deliberately
testing that experimental path.

## Source Map

| Source | Relevance | What it says for us |
| --- | --- | --- |
| [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx), pinned [commit `6e928d7`](https://github.com/arozanov/turboquant-mlx/tree/6e928d715595dee9f6b6cc3968baa44e1f408d28), and pinned [README](https://raw.githubusercontent.com/arozanov/turboquant-mlx/6e928d715595dee9f6b6cc3968baa44e1f408d28/README.md) | MLX TurboQuant port with mixed K/V cache, fused Metal kernels, sparse V, and GQA-aware kernels. | Treat as the closest MLX comparison target, but do not cite an unverified 98% speed claim from this commit. The README table shows about 72% FP16 speed for K8/V4 at Qwen 2.5 7B 32K. The portable ideas are pre-rotated Q scoring, SIMD reductions, no-repeat GQA indexing, and direct packed kernels; WHT pull-out requires a WHT/codebook value codec. |
| [arozanov/mlx-lm feature branch](https://github.com/arozanov/mlx-lm/tree/feature/turboquant-kv-cache) | mlx-lm integration branch linked by the port. | Reproduce the same server/cache setup before claiming our Swift path is comparable. |
| [sharpner/turboquant-mlx](https://github.com/sharpner/turboquant-mlx) and its [README](https://raw.githubusercontent.com/sharpner/turboquant-mlx/main/README.md) | Independent Apple Silicon reproduction with V2 affine and V3 codebook paths. | Confirms the failure mode we are seeing: MLX-native or software dequant/codebook paths are slow without custom kernels. V2 uses `mx.quantized_matmul`; V3 is quality-oriented but slow. |
| [vllm-metal TurboQuant docs](https://docs.vllm.ai/projects/vllm-metal/en/latest/turboquant/) | Production-oriented serving docs for TurboQuant on Apple Silicon. | Requires paged attention and explicitly does not run on the ordinary MLX KV cache path. This is a major architectural gap in our cache model. |
| [TurboQuant paper](https://arxiv.org/abs/2504.19874) | Original algorithm and quality target. | Use for algorithmic acceptance and quality suites, not as proof that our current kernels are fast. |
| [Open-TQ-Metal paper](https://arxiv.org/abs/2604.16957) | Apple Silicon fused compressed-domain attention design. | The spec is direct attention on packed int4 KV in Metal without intermediate dequantization. That is the main rewrite target for dense TurboQuant. |
| [mlx-vlm TurboQuant README section](https://github.com/Blaizzy/mlx-vlm/blob/main/README.md#turboquant-kv-cache) | MLX-VLM implementation and real memory examples. | Its real numbers show the distinction we must report: KV memory can shrink a lot while peak process memory shrinks less, depending on model and context. |
| [ml-explore/mlx issue #3404](https://github.com/ml-explore/mlx/issues/3404) | Native quantized SDPA feature request and prior-art summary. | The ecosystem diagnosis matches ours: full-cache dequantization defeats long-context memory savings, while Python-level custom Metal kernels can be several times slower than native SDPA. |

## Local Evidence Snapshot

Latest Qwen3.5-2B-4bit real-model benchmark at 32K:

| Config | Decode tok/s | Speed vs FP16 | Estimated KV | Peak active | End active | Quality |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fp16` | 38.26 | 1.00x | 1610.6 MB | 1989.0 MB | 1482.3 MB | Reference |
| `affineK8V4` | 27.68 | 0.72x | 755.0 MB | 1986.1 MB | 1249.7 MB | Pass |
| `affineInt4` | 29.24 | 0.76x | 604.0 MB | 1989.0 MB | 1205.6 MB | Pass |
| `turbo3_5` | 31.25 | 0.48x | 604.0 MB | 2685.9 MB | 1610.3 MB | Not quality-gated in dense group |
| `turbo4v2` | 28.43 | 0.44x | 629.1 MB | 2692.2 MB | 1616.6 MB | Not quality-gated in dense group |
| `turbo8` | 17.44 | 0.27x | 1031.8 MB | 2792.8 MB | 1717.3 MB | Not quality-gated in dense group |

Quality at 32K:

| Config | Gate | KL mean | P95 max logit error |
| --- | --- | ---: | ---: |
| `affineK8V4` | Pass | 3.36e-6 | 0.957 |
| `affineInt4` | Pass | 4.61e-6 | 1.188 |
| `affineK8V3-protectedK8V4-edge5` | Pass | 4.59e-6 | 1.073 |
| `affineK8V3` | Fail | 2.21e-5 | 2.875 |
| `affineK8V2` | Fail | 3.46e-4 | 4.727 |
| `affineK8V2-calibrated` | Pass | 5.59e-6 | 1.726 |

Sparse-V:

| Context | Best observed sparse decode | FP16 decode | Result |
| ---: | ---: | ---: | --- |
| 4096 | 14.77 tok/s | 88.55 tok/s | Active 6/6 sparse layers, too slow |
| 8192 | 14.56 tok/s | 84.55 tok/s | Active 6/6 sparse layers, too slow |
| 16384 | 10.39 tok/s | Not completed in sparse group | Requested 6 layers, active 0, fallback inactive |

## Path 1: Evidence And Memory Measurement Overhaul

Status: new development.

Why this matters:

- External ports report real KV bytes, peak memory, and sometimes end-to-end
  throughput. Our current output has enough to reject weak paths, but not enough
  to prove true process-level memory savings.
- The current `quantizedKVStart` behavior and exact prefill invariant can leave
  raw cache/transient allocations visible in peak memory. That is legitimate,
  but it must be reported separately from compressed resident KV.

Local surfaces:

- `tools/TurboQuantInferenceParity/main.swift`
- `Libraries/IntegrationTestHelpers/InferenceParityBenchmark.swift`
- `Libraries/MLXLMCommon/WiredMemoryUtils.swift`
- `Libraries/MLXLMCommon/KVCache.swift`
- `Libraries/MLXLMCommon/AttentionUtils.swift`
- `scripts/run-turboquant-current-benchmarks.sh`

Required work:

1. Add a real memory acceptance report that separates:
   - prefill peak active memory;
   - post-prefill active memory;
   - post-conversion active memory;
   - steady decode active memory;
   - process RSS/wired memory if available;
   - cache resident bytes per layer;
   - raw shadow/transient bytes;
   - `quantizedKVStart` and effective compression start.
2. Report the active cache format per layer at every sample.
3. Treat estimated KV bytes as a model-derived estimate, not proof of runtime
   memory savings.
4. Make `sparseRequestedButInactive` and native fallback reasons hard promotion
   blockers.
5. Demote synthetic-only runs to "kernel regression" in reports and docs.

Acceptance:

- A claim of memory savings must show lower steady active memory and lower
  resident KV bytes than FP16 on the same model/context.
- A claim of peak memory savings must show lower peak process memory than FP16
  on the same model/context.
- A path can be capacity-viable without being speed parity.

## Path 2: Mixed Affine K8/V4 And Affine Int4

Status: overhaul the current path.

External spec:

- `arozanov/turboquant-mlx` identifies mixed precision as the practical MLX
  path: K remains high precision enough for stable softmax, V is compressed
  more aggressively for memory.
- `sharpner/turboquant-mlx` shows the fast path is the affine/hardware-assisted
  V2 family, while software codebook paths are slower.
- vLLM Metal defaults also preserve higher K precision relative to V in its
  recommended config.

Current local state:

- `affineK8V4` is quality-stable in existing 32K Qwen3.5 evidence and in the
  latest 16K Qwen3-0.6B quality gate. The newest 16K single-sample throughput
  row after materialized conversion measured above FP16, but it did not include
  quality in the same report and must be rerun with repeated randomized samples.
- `affineInt4` is faster than K8/V4 on this model and passes the current 32K
  quality gate, but it is not yet broadly proven by architecture/model family.
- End active memory improves versus FP16; peak memory does not reliably improve.
- The default affine throughput route compresses after `16_384` tokens in many
  resolved profiles, so short and mid-context runs may not exercise compressed
  cache from token 0.
- Below-threshold runs now report raw resident KV, and promotion blocks affine
  candidates when no native compressed path was selected.

Local surfaces:

- `Libraries/MLXLMCommon/TurboQuantKVCache.swift`
- `Libraries/MLXLMCommon/AttentionUtils.swift`
- `Libraries/MLXLMCommon/Evaluate.swift`
- `Libraries/MLXLMCommon/KVLayerPolicy.swift`
- `Tests/MLXLMTests/TurboQuantProfileTests.swift`
- `tools/TurboQuantInferenceParity/main.swift`

Required work:

1. Rerun the combined 16K quality plus 32-token throughput gate from
   `continuation-handoff-2026-06-07.md`.
2. Benchmark compressed-from-token-0 affine runs separately from exact-prefill
   conversion runs.
3. Remove or account for raw shadow state after conversion.
4. Add direct packed affine QK and AV kernels for decode and long prefill
   instead of materializing dequantized intermediates.
5. Specialize grouped-query attention so GQA does not allocate repeated KV.
6. Apply pre-rotated query scoring and `simd_sum`-style reductions consistently
   to any remaining barrier-heavy QK path.
7. Do not target WHT pull-out for the current affine V codec. Reconsider it
   only if a WHT/codebook V codec becomes a deliberate rewrite path.
8. Add a same-machine reproduction harness for `arozanov/turboquant-mlx` and
   compare identical model, prompt, context, generation length, cooldown, and
   memory metrics.
9. Keep `affineK8V4` as the quality anchor and test `affineInt4` as the speed
   candidate.

Acceptance:

- At each target context, report speed ratio to FP16, steady active memory
  ratio to FP16, peak memory ratio to FP16, quality gate status, active kernel
  kind, and fallback state.
- Promotion target is at least equal-context quality pass and a clear capacity
  win. Speed parity remains unclaimed until measured.

## Path 3: Dense TurboQuant / Polar-QJL

Status: rewrite or demote.

External spec:

- Open-TQ-Metal computes attention directly on compressed int4 KV with custom
  Metal shaders and avoids intermediate dequantization matrices.
- The MLX issue #3404 diagnosis says dequantizing the full cache defeats
  long-context memory savings; Python-level Metal kernels can also lose badly to
  native SDPA dispatch and memory planning.
- `sharpner/turboquant-mlx` reports that codebook/software dequant paths are
  slow without custom kernels.

Current local state:

- `turbo3_5`, `turbo4v2`, and `turbo8` are slower than FP16 at 32K and have
  higher measured active memory than FP16 in the dense group.
- They are therefore not optimization candidates in their current form.

Local surfaces:

- `Libraries/MLXLMCommon/TurboQuantKVCache.swift`
- `Libraries/MLXLMCommon/AttentionUtils.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift/Source/MLX/TurboQuant.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-c`

Required work:

1. Decide whether dense TurboQuant is a product target. If yes, rewrite it
   around native compressed-domain Metal attention. If no, demote it from
   default benchmark acceptance.
2. Implement decode-first compressed-domain attention that never materializes
   full FP16 K/V matrices.
3. Extend to long prefill only after decode is proven.
4. Add attention-scale/model-family quality analysis; Open-TQ-Metal reports
   that attention scaling changes whether angular quantization survives.
5. If adopting WHT/codebook values, treat that as a codec rewrite, not a small
   kernel port from the current affine V path.
6. Make fallback explicit and fail-closed for acceptance.

Acceptance:

- Dense TurboQuant must beat the current affine path on either speed or memory
  while passing quality. Otherwise it stays a debug/capacity experiment.

## Path 4: Paged KV Cache / vLLM-Metal-Compatible Serving

Status: new development.

External spec:

- vLLM Metal requires paged attention for TurboQuant and says TurboQuant cannot
  run on the ordinary MLX KV cache path.
- vLLM-style serving gets much of its practical value from page/block cache
  management, not just quantization.

Current local state:

- The Swift path has raw, rotating, quantized, and compressed caches, but not a
  first-class paged KV allocation model comparable to vLLM Metal.
- Existing conversion behavior is cache-object centric rather than page-table
  centric.

Local surfaces:

- `Libraries/MLXLMCommon/KVCache.swift`
- `Libraries/MLXLMCommon/TurboQuantKVCache.swift`
- `Libraries/MLXLMCommon/TurboQuantCacheRuntimeSnapshot.swift`
- `Libraries/MLXLMCommon/TurboQuantAdmission.swift`
- `Libraries/MLXLMCommon/LayerPartitioning.swift`

Required work:

1. Design fixed-size KV pages with logical token order, page tables, refcounts,
   and compact residency accounting.
2. Define compressed page lifecycle: allocate raw page, fill, quantize, commit,
   release raw page, expose resident compressed bytes.
3. Add page-aware attention routing and page-aware fallback reasons.
4. Make snapshot/export/import page aware before product activation.
5. Re-run vLLM Metal-compatible comparisons only when page attention is real.

Acceptance:

- Paged TurboQuant must prove lower resident memory at long context without
  falling back to ordinary MLX KV cache.

## Path 5: Sparse-V And PageTopK

Status: rewrite or keep as diagnostics only.

External spec:

- `arozanov/turboquant-mlx` uses sparse V with fused Metal kernels and WHT
  pull-out style optimization. The important part is not sparse selection alone;
  it is fusing selection and value accumulation so skipped work is actually
  removed from the hot path.
- The WHT pull-out part is not portable to the current affine value codec.
  Current Swift sparse work should target fewer launches, fewer allocations, and
  no score/value materialization, not centroid-space WHT accumulation.

Current local state:

- Sparse-V is active at 4K/8K, but much slower than FP16.
- The 16K sparse sample requested sparse layers and got zero active sparse
  layers. That is an acceptance blocker.
- The current sparse paths are useful for diagnostics and kernel-kind coverage,
  not promotion.

Local surfaces:

- `Libraries/TurboQuantBench/TurboQuantSparseVBenchmarkPlan.swift`
- `Tests/MLXLMTests/TurboQuantSparseVBenchmarkPlanTests.swift`
- `Tests/MLXLMTests/TurboQuantInferenceParitySparseVTests.swift`
- `Libraries/MLXLMCommon/TurboQuantRuntimePolicy.swift`
- `Libraries/MLXLMCommon/AttentionUtils.swift`

Required work:

1. Keep `TurboQuantSparseValuePolicy.profileDefault == .off`.
2. Fail acceptance when `sparseRequestedButInactive == true`.
3. Do not run long sparse sweeps until the active native route is proven at
   that context length.
4. Rewrite sparse selection and AV into a single native route, avoiding
   per-head allocation, score materialization, and multi-dispatch reductions.
5. Share selection across GQA groups where correctness allows.
6. Reintroduce pageTopK only when page summaries are resident and verified in
   real model output.
7. Do not claim upstream WHT pull-out parity unless Swift V storage is changed
   to a WHT/codebook format and quality is revalidated.

Acceptance:

- Sparse-V needs a real-model speedup over dense affine K8/V4 at the same
  quality gate. A skip ratio without speedup is not a win.

## Path 6: Online, Tiled, And Two-Stage Core Paths

Status: demote unless exposed as real product routes.

Current local state:

- The latest core matrix recorded `onlineFused` and `tiledOnlineFused` as not
  callable for many rows because public routing reaches the native compressed
  path instead.
- `twoStageCompressed` can be measured as a core path, but it is not a real
  product speed candidate if it materializes intermediate score/value work.

Local surfaces:

- `Libraries/MLXLMCommon/AttentionUtils.swift`
- `Libraries/MLXLMCommon/TurboQuantKVCache.swift`
- `Libraries/TurboQuantBench/TurboQuantBench.swift`
- `/Users/mt/Programming/Schtack/mlx-forks/mlx-swift/Source/MLX/TurboQuant.swift`

Required work:

1. Remove not-callable paths from default acceptance reports, or label them as
   non-product kernel probes.
2. If online/tiled fused paths are still intended, expose a real public
   primitive and route a real-model config through it.
3. Keep two-stage only as a fallback/proof path until it beats native or affine
   routes on real-model metrics.

Acceptance:

- A path must be callable from a real-model config before it appears in a
  promotion table.

## Path 7: Cross-Repo Reproduction Harness

Status: new development.

Why this matters:

- We cannot answer "why are we worse than other ports" from synthetic Swift
  numbers alone.
- Same-machine reproduction removes hardware, model, prompt, and cooldown
  ambiguity.

Required work:

1. Add `scripts/reproduce-external-turboquant.sh` that can clone or reuse:
   - `arozanov/turboquant-mlx`;
   - `arozanov/mlx-lm` feature branch;
   - `sharpner/turboquant-mlx`;
   - optionally `vllm-project/vllm-metal`.
2. Standardize:
   - model path;
   - prompt token count;
   - generated token count;
   - warmup count;
   - throughput repeats;
   - cooldown;
   - memory sampling.
3. Emit one comparable CSV/JSON artifact with:
   - repo/commit;
   - backend path;
   - context;
   - decode tok/s;
   - prefill tok/s;
   - peak active/process memory;
   - steady active/process memory;
   - KV resident bytes;
   - quality proxy.

Acceptance:

- Any claim that another port is faster must include the commit, model, context,
  prompt, and metric definition.

## Path 8: Additive PolarWHT Upstream-Parity Rewrite

Status: implemented experimental scaffold; native optimization and promotion
evidence remain unfinished.

External spec:

- Match `arozanov/turboquant-mlx` at pinned commit `6e928d7` as a new path,
  while retaining current affine paths unchanged.
- Treat the upstream README's visible K8/V4 table as about 0.72x FP16 at Qwen
  2.5 7B 32K unless a same-machine reproduction proves a stronger claim.
- Preserve the distinction between packed payload compression and end-to-end
  runtime memory savings.

Current local state:

- Public labels and route scaffolding exist for full PolarWHT K/V and hybrid
  K8+PolarWHT-V.
- The current production Swift value codec is affine (`zero + q * scale`), so
  exact WHT pull-out is not available on existing affine K8/V4 values.
- Current dense `polar_qjl`, Sparse-V, pageTopK, and candidate-sparse routes
  remain diagnostic until real-model gates pass.
- `affineK8V4` and `affineInt4` remain valid fallback and comparison paths.
- Local bounded hybrid K8+PolarWHT-V runs can pass quality but are speed
  regressed, so default promotion blocks unless
  `TURBOQUANT_ENABLE_POLARWHT_HYBRID_PROMOTION=1` is set for deliberate
  experiments.

Implemented public/runtime identifiers to preserve:

- `TurboQuantKVCodec.polarWHT`.
- `TurboQuantBackend.polarWHTReference`.
- `TurboQuantBackend.metalPolarWHT`.
- `hybridK8PolarWHTV3` and `hybridK8PolarWHTV4` CLI/config labels.
- Route `KVLayerCodec.turboQuant(...)` through these identifiers without
  changing existing `polar_qjl` semantics.
- Keep CLI/report labels that distinguish `polarWHT`, `metalPolarWHT`, and
  hybrid K8 plus PolarWHT V from affine and existing dense TurboQuant labels.

Storage work:

- Continue keeping key and value PolarWHT sidecars separate. Hybrid K8+PolarWHT-V
  must never allocate or serialize a PolarWHT key sidecar.
- Preserve packed Lloyd-Max centroid indices plus per-vector norms and
  deterministic WHT signs for the PolarWHT value sidecar.
- Keep bit packing, boundaries, signs, and norm layout testable independently.
- Release raw shadow state after conversion and report resident packed bytes,
  norm bytes, affine K bytes, and any temporary buffers separately.

Kernel work:

1. Keep fused PolarWHT encode/decode covered by unit and native tests.
2. Finish and optimize pre-rotated QK scoring with SIMD-group reductions and
   GQA head mapping where full PolarWHT K/V is still being measured.
3. Optimize dense AV so it accumulates weighted centroids first and runs one
   WHT at the end.
4. Sparse AV with `threshold == 0` exact-equivalence tests and positive
   thresholds gated by quality.
5. Hybrid K8 plus PolarWHT V, using high-fidelity K scoring and WHT/codebook V
   accumulation.

Wiring to verify and finish:

- Keep capability probes and diagnostics flowing through `MLX.TurboQuant`, MLX
  C APIs, native kernel wrappers, and `AttentionUtils` path selection.
- Keep LM cache lifecycle wired in `TurboQuantKVCache`, `KVCache`, profiles,
  snapshots, and benchmark CLI labels.
- Fail closed when `metalPolarWHT` is unavailable, sparse was requested but
  inactive, raw fallback was used, or quality fails.

Real-model acceptance:

- Build the same-machine upstream reproduction harness before accepting speed
  or memory claims.
- Gate Qwen-family decode at 32K, 64K, and 128K where FP16 fits. When FP16 does
  not fit, report against the best FP16-fitting context and dense K8/V4
  reference.
- Report decode tok/s versus FP16, resident KV compression, peak active memory,
  steady active memory, selected backend/path, sparse active state, fallback
  reason, and quality gate.
- Compare `polarWHT` against reproduced upstream `turboquant-mlx`, current
  `affineK8V4`, `affineInt4`, and FP16.

Test requirements:

- Unit: WHT signs, Lloyd-Max boundaries, bit packing, encode/decode parity, and
  WHT pull-out equivalence with `threshold == 0`.
- Kernel: pre-rotated QK versus dense reference, GQA without repeat allocation,
  sparse V diagnostics, and SIMD reduction correctness for head dims 64, 128,
  and 256.
- Integration: cache conversion, snapshot import/export, profile parsing,
  fallback diagnostics, and real-model `TurboQuantInferenceParity`.

Acceptance:

- No promotion claim unless the native `metalPolarWHT` path is selected,
  fallback is inactive, resident memory is lower than the comparison path,
  quality passes, and the same-machine upstream comparison is reported.

## Immediate Work Queue

1. Rerun the combined 16K affine K8/V4 quality-plus-throughput gate from
   `continuation-handoff-2026-06-07.md`; the last attempt wrote an empty JSONL.
2. Build the real memory acceptance report. Without it, memory claims stay
   ambiguous.
3. Build the same-machine upstream reproduction harness and make it emit commit,
   model, context, prompt, tokens, cooldown, memory, fallback, and quality
   fields.
4. Reproduce `arozanov/turboquant-mlx` and `sharpner/turboquant-mlx` locally on
   the same Qwen model and prompts.
5. Expand PolarWHT storage and native tests: WHT signs, Lloyd-Max boundaries,
   bit packing, encode/decode parity, value-only sidecars, no hybrid key
   sidecar, and WHT pull-out identity.
6. Optimize the decode-only native `metalPolarWHT` and hybrid K8+PolarWHT-V
   kernel paths only after affine K8/V4 promotion gates are clean.
7. Re-run affine compressed-from-token-0 versus exact-prefill conversion using
   Qwen3.5-2B-4bit at 4K, 8K, 16K, 32K, and the largest FP16-fitting context so
   fallbacks remain measured.
8. Keep dense TurboQuant, PolarWHT, and sparse paths demoted from default
   promotion reports
   until they are direct compressed-domain kernels with real-model wins.
9. Start a paged KV design only after the affine and `polarWHT` paths have clean
   real memory
   instrumentation; otherwise the page work will not have trustworthy
   acceptance criteria.

## Decision Matrix

| Path | Keep | Overhaul | Rewrite | New development | Demote now |
| --- | --- | --- | --- | --- | --- |
| `affineK8V4` | Yes | Yes | Kernel hot path only | No | No |
| `affineInt4` | Yes | Yes | Kernel hot path only | No | No |
| Protected K8/V3 | Yes, as experiment | Yes | Maybe | No | No, but do not promote yet |
| K8/V2 dense | No product claim | No | Maybe after quality work | No | Yes |
| K8/V2 calibrated | Experiment only | Yes | Maybe | No | No promotion |
| Dense `turbo3_5`/`turbo4v2`/`turbo8` | Capacity/debug only | No | Yes, if kept | No | Yes from optimization claims |
| Sparse-V/pageTopK | Diagnostics only | No | Yes | Maybe page summaries | Yes from promotion |
| Online/tiled/two-stage | Kernel probes only | No | Only if product-routed | No | Yes from acceptance |
| `polarWHT` | Experiment only | No | Yes | Yes | No promotion until gates pass |
| Hybrid K8 plus PolarWHT V | Experiment only | No | Yes | Yes | No promotion until gates pass |
| Paged cache | No current product path | No | No | Yes | Not applicable |
| Benchmark/memory reporting | Yes | Yes | No | Yes | No |
