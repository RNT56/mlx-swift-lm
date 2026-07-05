# TurboQuant custom Metal kernel — anatomy and verified optimization roadmap

**Date:** 2026-07-01
**Scope:** the custom JIT (PolarQJL/TQCOOP) kernel family in `mlx-swift`, its host dispatch layers, and its relationship to the affine production path. Sources: six deep-read reports of the actual code plus an adversarial verification pass (two verdicts per proposal: premise-vs-code, performance-mechanism). Every proposal below survived verification; where a verifier weakened a claim, the corrected version is presented and marked.


**Provenance:** produced by a 40-agent analysis workflow (6 code deep-reads, 4 design lenses, 28 raw proposals consolidated to 14, then 2 adversarial verifiers per proposal). First draft 2026-07-01 shipped with the verification pass blocked by a usage limit; this version (2026-07-02 re-run) carries the full verification: 14/14 proposals survived, none refuted, many corrected — including reversals of first-draft claims (iPhone GQA routing, metadata-diet arithmetic, ghost-mode decision rule, affine-path applicability of host fixes).

---

## 1. Kernel anatomy

### 1.1 Kernel inventory and where it lives

The kernel family is ~27 kernel *bodies* as `string_view` sources in `turbo_quant_attention_jit.h` (5,662 lines; no literal `kernel void` — signatures are synthesized per-dispatch by `mlx::fast::metal_kernel`, registrations at `fast.cpp:2060–2818`). The header is a mechanical mirror of the live Swift-embedded copies in `mlx-swift/Source/MLX/TurboQuant.swift` (~10700+); **edits must land in both copies, and the Swift copy is the one mlx-swift-lm actually executes.** Dense decode uses three kernels:

- **Fused single-pass** (`jit.h:1232`, registered `fast.cpp:2061`): one 256-thread TG per (batch, q_head, q_token) row scans the whole context in 256-token tiles; ~19 barriers/tile.
- **Block partials** (non-GQA `jit.h:5050`; **GQA `jit.h:5185` — the live decode kernel**) + **block reduce** (`jit.h:5603`): split-K, 512-thread TGs, one (row, 512-token block) each.
- **Sparse/page/candidate pipeline** (~20 kernels, `jit.h:1367–5048`): 4–6 launches per layer per step with a full score round-trip through device memory. Measured 0.05–0.09× the dense path; diagnostic only. One kernel (`sparse_gqa_block_stats_topk_candidates`, jit.h:3589) is dispatched only under a hard-coded `false` (fast.cpp:3351) — fully dead.

### 1.2 Decode-step dispatch chain

Swift routed entry `turboQuantMetalScaledDotProductAttention` (TurboQuant.swift:6388) → decision ladder (nativeMLXCompressed → onlineFused → tiled → twoStage → packedFallback → baseline → **throw**, TurboQuant.swift:2008). Block sizing on both routes: `tq_recommended_block_width` (fast.cpp:1864–1892) returns **0 whenever `query_length != 1`** or N < 4096, and 0 again above 256K (active_blocks > 512) — outside that window every row falls to the monolithic fused kernel. Per token per layer: 1 launch (N<4K), 2 launches (4K–256K), 4–6 (sparse).

Per-call host cost is large and verified: the C++ `metal_kernel` closure rebuilds the full kernel source string (jit.h header alone is ~56KB; the file totals ~283KB) and constructs+runs a `std::regex` **on every call**, then `CustomKernelCache` (custom_kernel.cpp:56–72, unguarded static) does a full source string-equality compare per launch. The Swift layer per attention call re-runs `metalRuntimeAvailable()` (device probe + metallib filesystem scan, TurboQuant.swift:8877–8924), rebuilds `TurboQuantKernelAvailability.current` (second/third device probes, uname-via-`Mirror`, env re-bridge), and `validateStorageArray` calls `array.eval()` per plane (TurboQuant.swift:9637) — 10 planes/call, ~36–72 real synchronous CPU↔GPU round trips per token at 36 layers, which break asyncEval pipelining.

**Compile-key churn:** `BLOCK_COUNT` is a template constant on both routes → new Metal pipeline every ~512 tokens of context growth; the segmented raw-tail kernel embeds `RAW_LENGTH = rawKeys.dim(2)` (TurboQuant.swift:6791) → **full pipeline recompile every generated token** on the hybrid/segmented path. `logical_length`/`ring_offset` are correctly runtime buffers (the fix pattern already exists in-tree, TurboQuant.swift:10240–10246).

### 1.3 Codec and layout (v6)

Five token-major planes per tensor: packed, signs, high_mask, residual_signs, fp32 scales (3/group for K, 2 for V). **Executed key bits = preset − 1** (one bit diverted to the QJL sign plane): turbo8 = 7-bit, turbo4v2 = 3-bit, turbo3_5 = 2+1 split. Values are **affine** (all 15 `tq_decode_attention_value` call sites pass role=1; the PolarQJL value branch is dead). At layout v6: branch 2 (high_mask mixed-bit) is dead; `scales[2]` is written 0.0 and **never read by any kernel**; `residual_signs`/`high_mask` are bound but never dereferenced — however both are **already allocated as compact `[1]` placeholders** at v6, so they are signature debt, not resident waste. Per-token K bytes @D=128/gs=64: turbo8 152B, turbo4v2 88B, turbo3_5 80B; V4 80B, so turbo4v2 KV = 168B/token = 3.05× vs FP16 512B (confirmed by the measured mem-ratio smoke).

### 1.4 The QK/AV asymmetry — the load-bearing fact

`tq_packed_offset` (jit.h:398–411) is token-major: adjacent tokens are one full record apart. The strided QK loop maps **lane=token** (jit.h:5434), so a 32-lane load of one logical word gathers 10–28 cache lines (turbo4v2 packed 12, signs 4, scales 6; turbo8 packed 28). The AV loop maps **lane=dim** (jit.h:5568–5600): adjacent lanes share packed words, 1–2 lines per load. This is the in-kernel proof pair behind the measured +141% (strided QK) vs +9% (coalesced AV). **TQCOOP** (jit.h:5287–5432) retiles only the QK read phase — quad-per-key, contiguous D/4 chunks, `simd_shuffle_xor` reduce, byte-identical back-half — and is the only lever that ever measured positive (+20/+60/+30–55% turbo8/turbo4v2/turbo3_5 @131K).

### 1.5 Threadgroup/occupancy profile

The GQA partials kernel allocates 22–24.5KB tgmem (partial[4·512] + tile_scores[4·512] + two uint[512] + query_cache; jit.h:5199–5203) — **exactly 1 TG per 32KB core**, with ~20 full-TG barriers per block in two 9-level tree reduces (jit.h:5515–5555), and an AV phase where 384/512 lanes idle behind `lane < HEAD_DIM`. Contrast the stock quantized SDPA: 32×gqa-thread groups, quad-per-key coalescing by construction, register streaming softmax with zero hot-loop barriers, fp16 metadata folded algebraically.

### 1.6 q_seq and GQA limits

- **q_seq > 1 (speculative verify):** validated up to 8 by host plumbing, but block width = 0 → the fused kernel; each (q_head, q_token) row rescans the entire cache with zero cross-query amortization. Structurally pessimal exactly where N7 needs help.
- **GQA:** grouped kernel engages for repeats 2–4 (one TG per kv-head, K decoded once for ≤4 query heads, plus the Part-B one-time query-rotation hoist — assets stock lacks). Repeats > 4 (Qwen2.5-7B) falls to the generic per-query-head kernel.
- **Coop gates diverge by route:** Swift = `TQ_COOP=1` + repeats==4 + ≥32K + macAppleSilicon profile (hd128 eligible); C++ = repeats==4 + **head_dim==256 exactly** + ≥32K, **no env switch, unconditionally on** when eligible. A hd-128 model can never engage C++ coop; measured coop evidence came via the Swift route.
- **A-series (corrected during verification):** the earlier claim "iPhone never gets the GQA kernel" is only true for the Swift-kernel dispatch tier. The default-enabled first-preference **native route dispatches the GQA kernel with the rotation hoist on any device** (fast.cpp:3694–3711) for q_len==1, ≥4K, repeats 2–4 — *if* the native self-test passes on-device, which is unmeasured. The genuine iPhone exclusions: bf16-output models, sinks, materialized masks, q_len 2–8, native-probe failures, and coop (never, on any route the phone reaches).

---

## 2. Bottleneck model

The single most important framing: **there is no one bottleneck — the binding constraint changes with context, and the two kernel families bind on different things.**

**8K (custom JIT, measured 0.10× FP16):** host overhead, not bytes. Per token: ~72 × (283KB source rebuild + 283KB compare + regex ctor), repeated capability/filesystem probes, and 36–72 synchronous `eval()` round trips that destroy the asyncEval pipelining decode wins depend on. The GPU kernel is nearly irrelevant here. Caveat from verification: the string-churn term alone is only ~2–7ms/token; the dominant hypothesized term (eval-split → lost pipelining) is **unmeasured** — the 2026-07-01 anatomy audit's verify pass never ran. Also, the shipped dynamic strategy uses plain FP16 at 8K anyway, so compressed-at-8K is a diagnostic frontier, not the product one.

**32K:** wall clock splits ~59% weights / 41% attention. The **affine production path** (0.72×) is overhead/occupancy-bound, not bandwidth-bound: it moves 1.6× fewer bytes than FP16 yet realizes ~0.45× effective bandwidth — tiny 32×gqa threadgroups, per-call temp buffers, 2 launches/layer. The custom JIT here is 0.27–0.48× and pays both host overhead and the uncoalesced QK gather. Crucially, the affine path does **not** traverse the expensive Swift host tier (it dispatches via the `.quantized` cache state to stock SDPA), so JIT host fixes will not move the 0.72× number.

**131K:** the custom JIT is memory-access-bound in the QK phase — transaction count/latency from the token-major gather, proven by coop's wins and by the AV control. Whether it is *transaction-limited* (coalescing buys more) or *line-bandwidth-saturated* (it doesn't) is the key open question; the ghost-mode/roofline decomposition that would answer it exists in-tree but **has never been recorded**. The affine path's attention-isolated ratio actually rises with context (1.04× @32K, 1.47× @65K), i.e. the stock geometry is the right long-context shape.

**M2 Pro vs A-series:** zero A-series numbers exist for anything (standing P0-d, workspace nonclaim). Known structural differences: iPhone block width is 256 not 512; coop never engages; MSL 3.2 (needed for device-scope fences) requires iOS 18+; the metallib fs-scan cost likely vanishes on a packaged bundle (weakening the "host fix is 2× better on A-series" prediction); ~34–51GB/s DRAM vs ~200, but a 24MB SLC that may absorb per-kv-head rescans. Every A-series prediction below is explicitly conditional.

---

## 3. Verified roadmap

### Tier 1 — both verdicts strong/plausible, do next

#### T1.1 Host dispatch de-serialization *(premise: STRONG, performance: PLAUSIBLE)*

- **Mechanism:** memoize `(base_name, template values, input dtype/'s'/'c' markers, output dtypes) → {kernel_name, source}` in `metal_kernel` (kills the per-call regex, 283KB rebuild, and 283KB compare — fix the racy static while there); NSLock-cache `metalRuntimeAvailable()` / `TurboQuantKernelAvailability.current` (the `TurboQuantRuntimeProbe.shared` pattern already exists); replace hot-path `validateStorageArray` `eval()` with structural checks (or one asyncEval-compatible flush per token), keeping the deep check on the diagnostics tier.
- **Corrected scope (per both verifiers):** this is a **JIT/capacity-tier fix only**. Strike "benefits both kernel families" — the affine path only touches sub-microsecond getenv churn. Restate the split count as ~36–72 real sync round trips/token (first eval per layer flushes; the other 9 are cheap). Source figure is 283KB, not 80KB (strengthens the math). A-series multiplier softened: the fs-scan term likely vanishes on packaged iOS bundles.
- **Prediction (assumption-laden):** 8K compressed 0.10× → 0.25–0.4× *if* the eval-split share dominates — this is the one unmeasured load-bearing assumption; 16K/32K +3–8%. Bit-exact.
- **Anchors:** metal_kernel.cpp:293–347; custom_kernel.cpp:56–72; fast.cpp:2016–2056, 2850–2853; TurboQuant.swift:8877–8924, 727–734, 670–678, 9637, 9922.
- **Validation (falsifier first, pre-committed drop criterion):** Instruments CPU sampling + dtrace syscall counts over 100 decode steps at 8K/32K; `TURBOQUANT_GHOST_SDPA_MODE=3` launch-only bracket; if host terms are <1% of wall, drop. Then patch, verify marker via `swift package edit` + `LC_ALL=C grep -a -c`, A/B with the combined quality+throughput gate. Either outcome settles the 8K host-overhead theory — a P0 finding.
- **Effort:** S. **Devices:** M2 Pro primary; A-series gain unproven, needs TQ_BENCH.

#### T1.2 Runtime uniforms for RAW_LENGTH and BLOCK_COUNT *(premise: STRONG; performance verdict on the full proposal: WEAKENED — this sub-change is the part both verifiers confirmed)*

- **Mechanism:** move `RAW_LENGTH` (TurboQuant.swift:6791) and `BLOCK_COUNT` (TurboQuant.swift:7309/6888; fast.cpp:3706) from template constants to 0-dim-array runtime uniforms — the codebase's own proven idiom (`LOGICAL_LENGTH` already works this way). Kills the per-token pipeline recompile on the segmented/hybrid path and the every-~512-token recompile on the dense path. New base names to avoid variant-cache aliasing. Bit-exact (loop bounds only).
- **Corrected expectations:** compiles are per unique variant, **not per layer**; the segmented recompile recycles after one `coldBlockTokens` cycle; dense-path amortized saving is only 0.12–1.6ms/token + p99 boundary hiccups — **this does not rescue the 8K 0.10×** (that's T1.1's job), and the dense 8K bench cannot falsify the segmented claim. Also cap/decouple the reduce kernel's `THREADS_PER_BLOCK` or it still churns at pow2 boundaries above ~65K.
- **Validation:** count `newComputePipelineState` calls per token on the **hybrid/segmented** decode (the path that exhibits the per-token recompile) before/after; dense p99 inter-token latency A/B.
- **Effort:** S–M. **Devices:** both; A-series churn is worse today (blockWidth 256), so relatively larger benefit there. The mixed FP16-tail-in-split-K follow-on (mechanism c) is **demoted to Tier 2/3**: its "natively coalesced" framing was contradicted (FP16 blocks in a lane=token structure are a ~3× fewer-lines win, not coalescing), it is not bit-exact vs the current compressed path, and it needs a fresh same-report quality+throughput gate plus diagnostics coverage.

#### T1.3 Layout v7: 32-token tile-transposed K planes *(premise: STRONG, performance: PLAUSIBLE)*

- **Mechanism:** attack the proven root cause at rest. Within aligned 32-token tiles, transpose K packed/signs/scales to word-major (`offset = tile_base + word·128 + (phys_token & 31)·4`). The existing lane=token QK loop then reads word *w* of 32 consecutive tokens as ~1 contiguous line — native coalescing, single pass, no quad bookkeeping, no coop gates. V stays token-major (AV already coalesced). Centralized in the offset helpers behind `LAYOUT_VERSION==7`; unported kernels fail closed via the existing version gate. Explicitly **not** the disproven streaming-coalesce (that added runtime tgmem staging traffic; this changes addresses at rest with zero extra runtime work).
- **Corrected prediction (per performance verifier):** steady-state is **~2 lines** per 32-lane load, not 1 — the ring offset is not 32-aligned (`ring_capacity = capacity − pinned` need not be a 32-multiple) — so the claim is **6–14× fewer transactions, single pass, ≥ coop**. The 1.5–2× multiplier is **probe-gated**: it holds only if the kernel is still transaction-limited (not line-bandwidth-saturated) post-coalescing. Drop the 4K–32K benefit claim (host-overhead-dominated per §2) and the 2-repeat-GQA product claim (routing unchanged). Premise correction: hd128 *is* coop-eligible on the Swift route — v7's unique coverage is 2-repeat GQA, the A-series strided path, and gate-free single-pass operation.
- **Exactness:** bit-exact — addresses change, operand values and FMA order do not; enforce byte-identical v6-vs-v7 output including a **nonzero-ring-offset + wrap case** in the parity gate; grep-audit the raw `k_packed[base+pw]` indexing in the coop branches (excluded from v7) before trusting parity.
- **Validation:** implement v7 addressing only in the encoder + strided GQA partials kernel under a new name; (1) bit-parity vs v6; (2) `swift package edit` + marker grep, then TurboQuantBench/QwenProof at 8K/32K/131K × turbo8/turbo4v2/turbo3_5 comparing v6-strided vs v7-strided vs v6-coop. **Kill criterion: v7-strided must beat v6-coop @131K within ~1 day of bench time.** Add the deliberate gate-widening step (supportedVersions, TurboQuant.swift:2114 range, fast.cpp layout validation accept 7 for ported kernels only).
- **Effort:** L (largest edit surface: offset helpers feed ~24 kernels + encoder, dual-source sync). **Devices:** both; the A-series strided path is exactly where coop can never engage.

#### T1.4 K-metadata diet, stage 1 only: scalesPerGroup 3 → 2 (fp32) *(overall proposal weakened/weakened; stage 1 is the uncontested core)*

- **Mechanism:** delete the confirmed-dead third K scale (written 0.0 at TurboQuant.swift:12930, read by zero of the 13+9 consumers across both source copies; stride hardcoded `*3u` at jit.h:504–506). Bit-exact, byte-compare parity.
- **Corrected numbers (both verifiers found a 2× error):** half of the metadata is per-*group*, G=2 at D=128: K scales 24 → 16B/token now (→ 8B later with fp16), K token 88 → 80B; the 131K/8-head/36-layer K-scale plane drops 906 → 604MB (fp32 stage) and to ~302MB with fp16 (**−604MB**, K+V aggregate ~−906MB) — real against the 16GB M2 Pro wall and iPhone budgets. KV compression 3.05× → ~3.3× (stage 1) → ~3.56× (with fp16 V+K scales).
- **Speed expectation demoted to hypothesis:** the scale stream is already the *best*-coalesced stream (~6 lines/32 lanes vs 32 for packed); prior favors ~0% standalone. Keep the <5% abort criterion; claim residency, not speed. Drop the residual_signs/high_mask "dead allocation" claim entirely — already `[1]` placeholders; that item is v7-only signature cleanup (and high_mask is **live at v5/v4**, so v5/v6 signatures must survive).
- **fp16 scales = stage 2, Tier 2:** machinery exists (`TurboQuantScaleStorage.float16` behind layout≥5 + experimental flag), codec-validated quality-free, but bounded-error → requires the 16K combined quality+throughput gate before any default flip; fp32 stays the fail-closed default.
- **Effort:** M (signature changes across both copies, new base names). **Devices:** both; residency win is unconditional.

### Tier 2 — promising, run the named probe first

#### T2.1 Stock-shape geometry rebase (quad-per-key 32-lane groups + register streaming softmax) *(premise: STRONG, performance: PLAUSIBLE — gated)*

Rebase the TQ codecs onto the stock quantized-SDPA geometry (BD=4 quad-per-key, zero hot-loop barriers, simd-shuffle reduces, interleaved striding, grid.z q_seq, ≤1024-block pass-2 merge lifting the 256K cap), keeping the TQ-only assets: decode-once GQA fan-out, rotation hoist, ring/pinned mapping, codebook + sign-residual estimator. Not a rebrand of barrier-strip/SIMD-reduce — those swapped primitives inside the unchanged geometry while the coalescing wall dominated; coop is literally this design's pilot. **Named gates before committing to the L-effort rewrite:** (1) v7 must land and the coalescing wall must fall; (2) the **tgmem-diet residency probe** (T2.2) must show co-residency moves the needle — one day vs a rewrite; (3) the **recorded ghost sweep** must bound how much of stock's own 1.04× ceiling is host overhead the rebase inherits. Corrected framing: gains are ~4–8× effective at the barrier/AV-collapsed phases, not "order-of-magnitude"; register pressure of the fat codec in a streamed per-key loop is the admitted unknown — the falsifier must include a **fused QK+softmax+AV** variant and a 32K row, not front-half-only at 131K. Prediction: attention-isolated 0.8–1.0× FP16 ≥32K, up to +25–40% end-to-end @32K *if* 2–3× on the 41% attention slice materializes. Streaming softmax reassociates FP → cosine-gated, diagnostic tier until it beats affine K8/V4. Effort L.

#### T2.2 Threadgroup-memory diet (half staging → 2 TGs/core) *(premise: PLAUSIBLE, performance: WEAKENED)*

Corrected mechanism: `tile_scores` is true staging, but **`partial` is an in-place RMW tree-reduce workspace** — halving it means fp32-add-then-half-store per stride, injecting bounded rounding into the softmax max/denominator (**not bit-exact**; cosine-gated, ~1e-3 rel over 9 levels), and halving `partial` is load-bearing (tile_scores alone leaves 16KB → still 1 TG). Dead-array deletion in the sparse kernels is **probably a compiler-DCE no-op** — read `staticThreadgroupMemoryLength` on the compiled pipelines before crediting it; reclassify as hygiene if so. Redo math at D=256 (the coop-measured config): 24 → ~14.1KB, still 2 TGs. A-series claim dropped: the phone's plain-partials kernel is ~6.5KB and already multi-TG. **Named probes first:** `staticThreadgroupMemoryLength` + GPU-counters occupancy on the `_h16` pipeline (not a "5-minute read"), pre-registered kill criterion (resident simdgroups must ~double), and run the microbench with the P0 probe so a null distinguishes bandwidth-saturated from register-capped. Prediction +10–25% @131K *if* latency-bound with headroom — the exact question the probe answers. Effort S. `tile_physical_tokens` cannot shrink to ushort at 131K.

#### T2.3 q_seq-batched (2–8) GQA split-K verify kernel *(premise: STRONG, performance: WEAKENED)*

The single kernel-level attack on the q_seq forward-scaling wall that caps N7. The GQA partials kernel is already structurally q_seq-aware (tgmem query staging, rotation hoist, per-repeat accumulators, causal_limit math); generalize the repeat loop to repeats×q_seq, lift the `query_length != 1` gate. **Corrections that must reshape the design:** (a) the stated bit-exact contract is self-inconsistent — dropping BLOCK_TOKENS 512→256 changes split-K sum association vs the q_seq=1 path; either hold block width constant or downgrade to argmax-equivalence with an acceptance-divergence counter in N7 diagnostics; (b) the repeats×q_seq ≤ 8 cap **excludes Qwen3-4B (4 repeats × k=4 = 16)** — the model N7 was measured on; chunk q_seq inside the kernel instead; (c) rescope impact to compressed-cache ≥32K Mac — delete the 16K 1.76→2.0× and sub-16K crossover claims (FP16-path territory; forward cost there is MLP/lm_head-bound per N1); the internally consistent number is verify-forward k=4 ~2.7× → ~1.6–1.9× @32K; (d) not novel — this is roadmap T2.3 whose verifier pass previously never ran; (e) A-series depends on gate-widening (T2.4) and device evidence. **Named probe first (free): the q_seq=1-vs-4 JIT microbench** — if fused q_seq=4 is not ~4–5× q_seq=1 attention cost, the headroom is smaller than modeled. Win is transaction-count + occupancy, not DRAM bytes (per-kv-head scans partly SLC-absorbed). Effort M.

#### T2.4 Gate widening/unification: coop to repeats 2–4 & hd128 on C++, A-series path verification *(premise: WEAKENED, performance: PLAUSIBLE)*

The coop-gate incoherence is fully verified (C++: repeats==4 + hd256 + always-on, no env; Swift: repeats==4 + hd128-eligible + opt-in + mac-only) and corrupts cross-route evidence — unify it, including switch polarity, not just geometry. **Major premise correction:** "iPhone never gets GQA" is false for the default native route (§1.6) — the A-series work item is (a) confirm the native self-test passes on-device and read `kernel_kind` diagnostics to see which path actually serves 32K decode there, then (b) scope the unlock to the cases native rejects (bf16 models, q_len 2–8, sinks/masks, probe failures). Coop at repeats=2 is **not a pure gate flip**: the coop branch hard-codes `r<4` loops over `query_cache` rows initialized only to repeat_count — clamp/zero-fill under a **new kernel name**. Drop "bit-exact pure routing" (different kernel, different reduction structure — say "cosine-parity-gated"); drop the Hadamard-hoist from the speed prediction (compute reductions are hidden on this kernel); restate the A-series "~2×" as conditional on the phase being bandwidth-bound and missing SLC — which is exactly what the A-series probe run must establish first. Validation: M2 Pro afternoon (widened Swift gate, 2-repeat hd128 synthetic microbench, cosine parity + ≥+10% or abort); A-series via `xcodebuild test … TQ_BENCH=1` forced-vs-current. Superseded for the strided path if v7 lands — cheap insurance meanwhile. Effort M.

#### T2.5 fp16 scale storage (metadata diet stage 2) — see T1.4; gated on the 16K combined quality+throughput report.

### Tier 3 — novel/experimental, diagnostic tier only

#### T3.1 Single-launch split-K via last-TG-done device atomic *(premise: STRONG, performance: WEAKENED)*

Honest corrected scope: a JIT-tier launch-overhead regression fix worth ~0.2–0.5ms/token (≤1% affine-side) — **not** a crossover mover; drop the "widens the frontier"/"where the 0.72× lives" framing (the FP16 baseline shares the same 2-pass structure; affine's second pass is in the same encoder). Requires: hard **MSL 3.2 / iOS 18+ gate** (`atomic_thread_fence(thread_scope_device)` doesn't compile on the 3.1 fallback — fail closed to 2 launches); batched counter zero-fill (MLX `init_value` is an extra dispatch, which otherwise eats the saving); the last TG must **replicate the old reduce-tree shape** for row_sum or the bit-exact parity gate fails spuriously. Sequence the affine transfer separately and last (pinned-submodule stock-kernel edits for the smallest saving). Standalone Metal toy + 10k-trial parity on both devices first. Effort M.

#### T3.2 Scale-anchored certified block skip *(premise: WEAKENED, performance: WEAKENED)*

Structural skeleton (in-launch skip reusing the causal (−INF,0) sentinel pattern, 3-launch shape, eps=0 degenerate) is sound and code-accurate, but two corrections gut the original: (1) the norm bound is **not a theorem over the kernel's computed score** as stated (codebook centroids overshoot ‖u‖≤1; fix by storing the exact per-group decode norm in the currently-dead third scale slot — which conflicts with T1.4, pick one); (2) statistical geometry predicts the norm-only bound skips ~nothing at certified eps (Cauchy–Schwarz is ~√d loose and angle-blind — this is why Quest-style systems use per-channel min/max, ~2B/token block metadata, which is also cheaper than the corrected 16–24B/token norm screen). Also not novel in-tree: `sparse_page_scores` already computes this bound heuristically. **Run the $0 Python replay falsifier first** (dump real 16K/131K scale planes + queries; report s(eps) for both the norm bound and the min/max bound per layer) — pre-committed kill at s(1e-4)<0.4, which the verifier predicts the norm bound hits. M2 Pro diagnostic-tier claim only. Effort L (if the falsifier licenses it).

#### T3.3 Wide vectorized loads in the coop QK path *(premise: WEAKENED, performance: WEAKENED)*

Corrected: executed bits are preset−1, so real chunks are 28B/12B (turbo8/turbo4v2 @D=128) with live cross-word straddle branches — the case is *stronger* than originally modeled (~13→3 loads turbo8), but **drop the 16B-alignment/record-padding/v7 coupling entirely**: lane chunks are only 4B-aligned; use `packed_uint4`-style 4B-aligned wide loads, which work today with zero layout change and no turbo3_5 byte tax. "Stock fast_load_t precedent" overstated (stock uses 2–4B typed loads); what stock licenses is branchless unrolled extraction via funnel shifts. Adverse precedent: TQPROF_OPT* (also load-issue reduction) measured ~noise; the bet is that the post-coop regime differs. Delete the A-series upside claim (phone never runs coop). One-afternoon falsifier: new-named coop variant, `TQ_COOP=1` @131K, byte-compare parity; pair with the P0 probe so a null assigns cause. Bit-exact. Effort S.

#### T3.4 Token-parallel AV retile (4×HEAD_DIM mapping) *(premise: STRONG, performance: PLAUSIBLE — sequenced last by design)*

Kill the 75%-idle AV lanes and cut serial depth 512→128; the un-built AV counterpart of coop, explicitly not the disproven "cooperative AV as coalescing" (AV is already coalesced; the target is idle threads + latency). Standalone ceiling ≤~8% (AV's current share) — build only **after** v7/rebase double AV's share. Corrections: the falsifier was miscalibrated — under the MLP model the win should be **≥ at V4, not V8** (coop scaled inversely with bits); use phase-ablation timing as the kill test and record V4-vs-V8 as characterization. No cross-sub L1 sharing (subs read different tokens; the invariant is total bytes unchanged + per-token coalescing preserved). Needs two barriers when repurposing `partial[]` (lane-0 stats read is unfenced), and the identical retile in the non-GQA kernel (jit.h:5162–5182). Deterministic but not byte-identical (fixed-order 4-way combine) — cosine-gated, own name, default-off. Effort S. Sequence the roofline probe first (tests the "AV latency not already hidden" assumption).

#### T3.5 Texture-buffer scale plane probe *(premise: WEAKENED, performance: WEAKENED)*

Kept only as a corrected diagnostic probe: the stride-3 K scale record makes zero-copy `float2` texels impossible (odd records straddle texels) — use `R32Float` two-fetch (pure TEX-vs-buffer-L1 hypothesis test) or accept a stride-4 repack that belongs in v7; the named M2 Pro falsifier (TurboQuantNativeVxBenchmark) **never dispatches this kernel** — use QwenProof/TurboQuantBench at ≥32K with and without `TQ_COOP=1`; the mechanism cannot reduce L2/DRAM line traffic at all (same lines, different L1), so cap expectations at single-digit % on an unverified disjoint-TEX-L1 premise. Run only if the roofline probe shows the QK phase well under roofline; otherwise predictably null. RG32 only (bit-exact); defer RG16. Effort S–M.

---

## 4. Do-not-do list

No proposals were refuted outright in this round, but the following are closed:

- **Barrier-strip** — measured +6% noise-level blip; sync cost is hidden behind the coalescing wall.
- **SIMD tree-reduce swap** — measured −3–5%; same reason.
- **Streaming-coalesce (runtime tgmem staging)** — measured −20–30%; adds traffic instead of fixing addresses.
- **Compute micro-opts (TQPROF_OPT* class, codebook-switch tuning)** — decode is memory-access-bound on M2 Pro; every FMA/issue reduction so far was hidden.
- **QJL residual-drop for speed** — quality-free but speed-neutral; byte cuts alone don't convert to time (also the P2 group-size precedent).
- **"Cooperative AV as coalescing"** — AV is already coalesced (+9% control); only the T3.4 MLP framing is legitimate.
- **Byte-count reduction as a speed claim on the affine path** — affine moves 1.6× fewer bytes than FP16 and still runs 0.72×; the diagnosis is overhead/occupancy.
- **Dead-plane deletion as a perf win** — residual_signs/high_mask are already `[1]` placeholders; unreferenced tgmem is compiler-DCE'd. Hygiene only.
- **Halving `partial` via a register-side reduction restructure** — re-treads the disproven SIMD-reduce lever; the only legal variant is the cosine-gated half-store (T2.2).
- **Norm-only Cauchy–Schwarz block skip as "certified"** — angle-blind, predicted s≈0 at useful eps; only the min/max-metadata redesign (T3.2) may proceed, falsifier-first.
- **Claiming host-dispatch fixes help the affine 0.72×** — that path doesn't traverse the expensive Swift tier.
- **Standing nonclaims remain in force:** no 0.98× FP16, no PolarWHT parity/promotion, no sparse-V promotion, no memory savings from estimates, no iOS readiness without device evidence.

---

## 5. Measurement gaps — measure these first

**G1. Roofline/decomposition probe** *(the corrected P0; both verdicts on the probe proposal were WEAKENED — build the corrected version)*. What exists: a **wall-clock** GB/s probe (TurboQuantNativeVxBenchmark schemaVersion 3, artifacts in `mlx-swift-lm/artifacts/turboquant-roofline-20260607/`) — it cannot separate GPU time from host time or attribute per kernel. What's missing: GPU-time attribution + a **recorded** ghost 0/2/3 sweep. Corrections that must be respected: **the ghost decision rule is inverted in the original proposal** — mode-2 holds pointers L1-resident, so mode-2 ≈ mode-0 means ALU/launch/occupancy-bound (layout work NOT justified) and mode-2 ≪ mode-0 means DRAM-streaming-bound; per-dispatch counter sampling is impossible on Apple GPUs (stage-boundary only; MLX packs dispatches into one encoder — use debug-gated encoder splits or command-buffer GPU start/end, take launch share from ghost mode-3); no Apple-silicon bandwidth counter sets (computed-bytes ÷ GPU time is the primary mechanism everywhere); validation is "GPU-sum ≤ wall, report the gap as host share", and sanity-check that ghost mode-0 matches un-probed tok/s; run ghost with q_seq=1 unmasked decode shapes only; verify mode-2's held loads survive compiler CSE (disassembly or address-perturbation litmus). Byte constants: turbo4v2 V4 = 80B/token (not 40). **Gates:** T1.3's multiplier, T2.1's go/no-go, T2.2's null-interpretation, T3.3/T3.4/T3.5's cause assignment.

**G2. Host-overhead decomposition at 8K** — Instruments + dtrace + ghost mode-3 + pipeline-creation counting (T1.1/T1.2's own falsifiers). The 2026-07-01 anatomy audit's verify pass never ran; the eval-split-dominance hypothesis is the single largest unmeasured assumption in Tier 1. **Gates:** T1.1's headline prediction, T1.2's expectations.

**G3. q_seq=1-vs-4 JIT microbench** — free; mirrors the standing P0-b to the JIT path. **Gates:** T2.3 entirely.

**G4. A-series numbers (standing P0-d — still zero data)** — `xcodebuild test -destination 'platform=iOS,name=<device>' -only-testing:MLXLMTests/TurboQuantBenchSuite TQ_BENCH=1`, plus `kernel_kind` diagnostics to establish which path actually serves iPhone 32K decode (native GQA vs fused). **Gates:** T2.4's headline, every "A-series benefits at least as much" claim in the set, T3.1's fence behavior, all product/device claims per the promotion checklist.

**G5. Occupancy ground truth** — `staticThreadgroupMemoryLength` + GPU-counter occupancy on the live compiled pipelines. **Gates:** T2.2, and T2.1's residency leg.

---

## 6. Sequencing recommendation

Every kernel-side step inherits two standing rules: (i) **build gotcha** — mlx-swift-lm pins mlx-swift by git rev, so `swift package edit` + `touch` + verify the marker in the binary with `LC_ALL=C grep -a -c` (never `strings`) before trusting any number; (ii) **fail-closed** — every variant under its own kernel base name (variant-cache aliasing), default-off, cosine/parity-gated, diagnostic tier until it beats affine K8/V4 in a same-report quality+throughput gate; never land a JIT change ahead of its affine-path equivalent where one exists.

1. **Instrument (G1+G2, ~2–3 days).** Build the corrected probe: encoder-split GPU timing + computed-bytes GB/s + recorded ghost 0/2/3 sweep (with the corrected decision rule) on both the affine 32K case and the JIT partials kernel; Instruments/dtrace + pipeline-creation counts at 8K. Artifact under `artifacts/`. This single step arbitrates T1.3's multiplier, T2.1, T2.2, T3.3–T3.5, and either confirms or kills the 8K host-overhead theory before any code changes.
2. **T1.1 host de-serialization (S).** Land iff step 1's falsifier passes (host terms ≥ a few % of 8K wall). Bit-exact; re-run 8K/16K/32K with the combined gate. Also do the still-owed real-model chore in the same window: the combined 16K affine K8/V4 quality+throughput artifact (the prior one was zero-byte) — no promotion claim can rest on missing evidence.
3. **T1.2 runtime uniforms + T1.4 stage-1 scale diet (S/M, bit-exact).** Independent of each other and of step 2; both are parity-gated cleanups with real wins (recompile churn; −302MB fp32-stage / −604MB fp16-stage K-scale residency at 131K). New base names.
4. **T1.3 layout v7 (L).** One-day falsifier first (v7-strided vs v6-coop @131K, ring-wrap parity case included). If it loses to coop, stop: coop stays the coalescing answer and T2.1 is re-evaluated. If it wins, retire the coop gates for the strided path and re-run the 131K matrix.
5. **Post-v7 fork, guided by the probe:**
   - If transaction-limited headroom remains → **T2.2 tgmem-diet probe** (1 day, occupancy read + `_h16` microbench) → if residency moves the needle, commit to **T2.1 geometry rebase** (with the fused-variant falsifier and the recorded ghost bound on stock's ceiling); if not, T2.1 is falsified for one day's work.
   - In parallel: **G3 q_seq microbench** → **T2.3 batched verify kernel** with the corrected exactness contract and q_seq chunking for 4-repeat models — this is the N7 compounding play at ≥32K.
6. **A-series track (independent):** G4 harness run + `kernel_kind` instrumentation → **T2.4** gate unification (coop switch polarity + repeats-2 clamp fix) and scoped iPhone unlocks, strictly behind the fail-closed profile policy and device evidence. No product/device claim before this lands.
7. **Opportunistic Tier 3:** T3.3 (afternoon, corrected packed-load version, no v7 coupling), T3.4 (after v7/rebase, corrected falsifier), T3.2's $0 Python falsifier (any idle hour; it likely kills the norm bound and licenses or buries the min/max redesign), T3.1 and T3.5 only if the probe says launch share / L1-port pressure respectively is material.

Dependency chain in one line: **probe → (host fix ∥ uniforms ∥ scale diet) → v7 → {tgmem probe → rebase} ∥ {q_seq probe → batched verify} → A-series evidence → gate widening → Tier 3**, with the combined 16K affine artifact re-run at step 2 because the promotion checklist — real-model throughput + same-report quality + native-path selection + no fallback + reported memory — outranks every optimization in this document.
