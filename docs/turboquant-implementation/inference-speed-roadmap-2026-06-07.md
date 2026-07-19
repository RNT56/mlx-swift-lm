# TurboQuant Inference-Speed Roadmap — 2026-06-07

Author: cross-repo analysis pass after the affine-K8/V4 + segmented-attention push
(mlx `14649588`, mlx-c `5bed9b6`, mlx-swift `6bfa04e`, mlx-swift-lm `286c494`;
mlx submodule `aec1240e`, mlx-c submodule `2d5088c`).

This document supersedes the speed-direction guidance scattered across
`external-port-optimization-map.md` and the memory notes. It is grounded in a
build-effective-code audit of the actual decode kernels, not synthetic microbenches.
Every claim cites a file anchor or a measured artifact. It corrects three beliefs
that turned out to be wrong on inspection (see §6).

---

## 0. TL;DR — the reframe

1. **There are two decode-kernel families and only one ships.**
   - **Affine family** (`mlx` core `scaled_dot_product_attention.cpp` /
     `sdpa_vector.h`): the production routes `affineK8V4` (0.72× FP16) and
     `affineInt4` (0.76×). This is MLX's *stock* quantized SDPA and is already
     well-optimized (split-KV ladder to 1024 blocks ≥32K, L2-shared GQA,
     coalesced loads).
   - **PolarQJL JIT family** (`turbo_quant_attention_jit.h`): the dense
     `turbo4v2 / turbo8 / turbo3_5` presets (0.27–0.48×). Capacity/diagnostic
     only; gated *out* of production by the dynamic FP16-vs-compressed decision.
     **All of the hand-rolled TQCOOP work lives here**, on a path that never
     reaches a shipped decode.
2. **The affine production decode is overhead/occupancy-bound, NOT
   bandwidth-bound.** At 32K it moves ~1.6× fewer *total* bytes than FP16
   (937 MB weights + 654 MB KV vs 937 + 1611) yet runs at 0.72×. That means the
   quantized attention kernel achieves **< 0.5× the effective bandwidth** of the
   dense FP16 SDPA kernel. Closing that efficiency gap is the single biggest
   prize and is worth, if fully closed, up to ~1.6× vs FP16 at 32K and ~2.5× at
   128K.
3. **At the contexts we actually measure (≤32K), decode is *weight*-dominated,
   not KV-dominated.** Fixed weight traffic ≈ 937 MB/token (MLP 507 + attn-proj
   141 + lm_head 286). KV (affineK8V4) = 164 MB@8K, 654 MB@32K, 2617 MB@128K.
   KV exceeds weights only past **~47K** for affine (~19K for FP16). So any
   KV-byte lever yields *sub-linear* wall-clock below ~47K, and the only lever
   that touches the dominant weight stream is **speculative decode**.
4. **The speculative-decode path is already 90% wired for affine.** The fused
   `quant_sdpa_vector_2pass` kernel accepts `q_seq` 2..8 via function-constant 31
   (`sdpa_vector.h:185`, grid `blocks*q_seq_len`), the C++ admission accepts ≤32
   (`fast.cpp:1467`), and the LM gate accepts ≤32 (`KVCache.swift:4122`). The
   matmul fallback fires only at `q_seq > 32` (`scaled_dot_product_attention.cpp:1012`).
5. **We are flying blind.** There is **no achieved-bandwidth/roofline instrument**
   (only resident-byte accounting in `WiredMemoryUtils.swift`), **no A-series
   harness result** (the product target; pending, device-only), and the required
   **combined affine 16K quality+throughput artifact is still zero-byte**. These
   three gaps make most kernel proposals unprovable today.

**Consequence for priorities:** instrument first, then attack the affine kernel's
bandwidth-efficiency gap and the weight stream (speculation), then codec byte
cuts (capacity now, speed once the kernel is efficient), and treat all
PolarQJL/TQCOOP kernel work as diagnostic-only until/unless PolarQJL becomes a
product path.

---

## 0.5 Measured execution results (2026-06-07, M2 Pro)

P0 was executed end to end on this machine. Tooling: `TurboQuantNativeVxBenchmark`
extended with a `--query-lengths` sweep and achieved-GB/s + ms/qtok roofline
fields (schemaVersion 3). Artifacts under
`mlx-swift-lm/artifacts/turboquant-roofline-20260607/`.

**P0-a + P0-b — synthetic affine K8/V4, headDim 128, 16Q/8KV (Qwen3.5-2B-ish):**

| ctx | qL=1 fp16x (attn-only) | BWx (affine GB/s ÷ FP16 GB/s) | amortization ms/qtok qL1→qL4 |
| ---: | ---: | ---: | ---: |
| 16K | 0.94× | 0.39 | 1.9× |
| 32K | 1.04× | 0.44 | 2.6× |
| 65K | 1.47× | 0.62 | 1.8× |
| 131K | 1.09× | 0.46 | 2.1× |

Findings, all confirming §0:
1. **Affine realizes 0.39–0.62× of FP16-SDPA's effective bandwidth** — the
   predicted <0.5× efficiency gap is real and measured (not inferred). BWx rises
   with context (occupancy/latency-bound at batch=1), so the kernel is most
   efficient exactly at the long contexts where affine is actually used.
2. **The multi-query (speculative-verify) KV scan amortizes ~1.8–2.6×** — ms/qtok
   falls as qL rises. Sweet spot qL≈4 at headDim 128; qL=8 regresses at 32K
   (L2-thrash / occupancy: the amortization is via L2 reuse across the per-q-row
   threadgroups — each `q_seq_idx` is a separate threadgroup, `sdpa_vector.h:800,804`).
3. **Attention-isolated, affine ties-or-beats FP16 at ≥32K** (1.04–1.47×). The
   arithmetic closes numerically: BWx × byte-ratio = fp16x (0.44 × 2.37 ≈ 1.04 ✓).

**P0-c — real-model combined affine 16K gate (Qwen3.5-2B-4bit), the literal first
continuation task (previously a zero-byte JSONL):** now a valid promotion-grade
artifact (`affine-16k-combined.json`):
- decode: fp16 72.71 tok/s, affineK8V4 **54.70 tok/s = 0.752× FP16**; KV 2.13×.
- native path selected (`affineK8V4Native=6`), **no raw/decoded fallback**.
- quality **passed**: top-1 1.000, KL 2.0e-6, p95 abs 1.10, cosine 0.9931.
- This reconciles with P0-a: real-model 0.75× while attention-isolated affine ≈
  FP16 at 16K (0.94×) → the real-model gap is the weight-dominated regime (at 16K
  weights are ~85% of decode bytes), not a fixable attention-kernel deficiency.

**P1-b — speculative verify: proven lever, already wired, end-to-end blocked.**
The affine path already admits qLen ≤ 32 to the fused kernel
(`KVCache.swift:4122`, `mixedAffineK8V4ScaledDotProductAttention` routes qLen>1
non-sparse → `MLXFast.mixedQuantizedScaledDotProductAttention`, `KVCache.swift:4244`),
so a speculative verify pass amortizes (the measured 2.3×) with **no fallback to
full-FP16 decode**. End-to-end measurement is blocked: the only on-disk small
model (Qwen3-0.6B-8bit, vocab 151936) is **not tokenizer-compatible** with
Qwen3.5-2B-4bit (vocab 248320) and cannot serve as its draft. Unblock = a
same-tokenizer draft (e.g. a Qwen3.5-0.5B), or self-/Medusa-style speculation.

**P2 — value group 32→64: NEGATIVE result, do not ship.** Speed change is within
M2-Pro thermal noise (V4 1.305↔2.004 ms across adjacent runs; V3 moved the
opposite way) while quality measurably worsened (V3 cosine 0.98124 → 0.97660).
This empirically confirms the central thesis: **byte-cuts do not convert to speed
while the kernel sits at ~0.47× bandwidth** — so codec byte-cuts are
capacity-only until P1-a raises kernel efficiency.

**P1-a — affine kernel bandwidth lift: assessed, intentionally NOT shipped blind.**
The headroom is real (0.39–0.62× BW), but: (a) the kernel is MLX/Apple-tuned;
(b) there is no GPU performance-counter instrument yet, so cause-localization
(dequant vs metadata vs occupancy) is unavailable; (c) the explicit query-fan-out
restructure spills registers at headDim 128 (qL×elem accumulators ≈ 256 floats/
thread — matches the observed qL=8 regression); (d) it cannot be validated on the
A-series product target from here; (e) affine already beats FP16 in attention
isolation where it is used. Shipping a structural change to a core kernel that
cannot be cleanly measured would repeat the disproven-3× pattern. It is recorded
as the top grounded kernel-design step (needs Metal counters + A-series first).

**Net executed:** the gating measurement infrastructure (roofline + amortization)
is landed and is itself the critic's #2 infra gap; the #1 continuation task
(combined affine gate) is unblocked with a clean artifact; two candidate levers
were resolved with data (speculation = real but draft-blocked; byte-cuts =
non-levers for speed). The evidence redirects effort away from blind kernel hacks
and codec byte-cuts toward (1) a same-tokenizer draft for speculation and (2) GPU
counters + A-series to make P1-a measurable.

---

## 1. The byte budget (the math that drives everything)

Qwen3.5-2B-4bit, GQA kvHeads=8, headDim=128, 24 layers. Per **decode token**:

### 1.1 Per-token-per-kv-head KV bytes (head_dim=128)
| Path | K vector | V vector | K+V | vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 256 B | 256 B | 512 B | 1.00× |
| affineK8V4 | 136 B (128 mag + 8 meta) | 80 B (64 mag + 16 meta) | 216 B | **2.37×** |
| affineInt4 | 80 B | 80 B | 160 B | 3.20× |
| turbo4v2 (JIT) | 120 B (3 fp32 + 3 bitset planes) | 80 B | 200 B | 2.56× |
| turbo8 (JIT) | 168 B | 128 B | 296 B | 1.73× |

Metadata (per-group fp16/fp32 scale+bias) is **11–33 %** of compressed bytes and
is the part that erodes the nominal-bits ratio (affineK8V4 nominal 2.67× → effective 2.37×).
At small group sizes the scattered fp32 metadata is also the worst-coalescing traffic.

### 1.2 Whole-token bandwidth and the weight crossover
Fixed weight traffic per token ≈ **937 MB** (MLP ~507 + attn-proj ~141 + tied
lm_head ~286). KV traffic per token:

| Context | FP16 KV | affineK8V4 KV | Weights | FP16 total | affine total | affine vs FP16 bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 403 MB | 164 MB | 937 | 1340 | 1101 | 1.22× fewer |
| 32K | 1611 MB | 654 MB | 937 | 2548 | 1591 | **1.60× fewer** |
| 128K | 6442 MB | 2617 MB | 937 | 7379 | 3554 | 2.08× fewer |

**The contradiction that defines the work:** at 32K affine moves 1.60× fewer
total bytes yet decodes at 0.72× FP16. A purely bandwidth-bound kernel moving
1.60× fewer bytes would run at **1.60×** (≈ 61 tok/s). We measure 27.68 tok/s.
So the affine attention kernel realizes **~0.45× of FP16-SDPA's effective
bandwidth**. The lost ~2.2× is dequant compute + scattered per-group metadata
loads + lower decode occupancy of the quant vector kernel. **That efficiency gap,
not the byte count, is the affine speed ceiling.**

Wall-clock split (bandwidth model, affineK8V4 KV): 8K ≈ 85 % weights / 15 %
attention; 32K ≈ 59 % / 41 %; 128K ≈ 26 % / 74 %. KV/attention only becomes the
majority cost past ~47K.

---

## 2. What is already done — do not re-implement

The audit found these "ideas" already shipped; several earlier proposals were
rejected for proposing them:

- **GQA load-once-decode-n_rep** — affine shares KV across query heads via L2 by
  design (`sdpa_vector.h:478-480` comment); PolarQJL already decodes a key once
  and fans out across ≤4 repeats (`turbo_quant_attention_jit.h:1090-1226`).
- **Affine split-KV ladder** — `select_sdpa_blocks` already returns 512/1024
  blocks at N≥32K for the decode shape (`scaled_dot_product_attention.cpp:51-56`).
- **AV phase is already coalesced** — token-outer/dim-per-lane: 128 lanes read 16
  contiguous 32-bit words = 64 contiguous bytes (`turbo_quant_attention_jit.h:5568-5592`).
  ("Cooperative AV" as a *coalescing* win is illusory; only its 25 %-occupancy
  issue is real, and only on the diagnostic JIT path.)
- **fp16 scales + dead-plane prune** for uniform turbo presets
  (`TurboQuant.swift:9561`, compact `[1]` bitset shapes).
- **Quest-style page top-k** (PolarQJL) and **per-layer mixed precision**
  (`KVLayerPolicy.swift:278-317`) — implemented and benchmarked; both lost on
  real-model decode (page top-k 0.07–0.09× at ≤8K; protected-edge V3 already in
  the 32K table).
- **In-place cache donation** (refcount nil-ing, `TurboQuantKVCache.swift:6164`)
  and **dynamic FP16↔compressed admission** — both landed and propagated.
- **q_seq 2..8 admission** for affine across the whole stack (see §0.4).

---

## 3. Prioritized levers

Priority key: **P0** = unblock/instrument (do first; cheap; gates everything).
**P1** = highest expected tok/s per effort on the shipping path. **P2** =
real but conditional or capacity-first. **P3** = diagnostic-only or A-series-gated.

### P0 — Instrument and unblock (1–3 days, M2 Pro, no new kernels)

- **P0-a — Achieved-bandwidth / roofline probe.** Add per-decode GB/s (Metal
  performance counters, or calibrated bytes-moved ÷ kernel time) to
  `TurboQuantTiming` + `WiredMemoryUtils`. Without this, the entire memory-bound
  thesis and every sub-10 % proposal is unfalsifiable. This is the highest-leverage
  *infrastructure* task — it tells us whether the affine 0.45×-bandwidth gap is
  compute, metadata, or occupancy, which decides between §3-P1-a and the codec
  levers.
- **P0-b — The q_seq=1 vs q_seq=4 affine microbench.** Call
  `quant_sdpa_vector_2pass` (already reachable, no fallback at q_seq≤32) on
  affineK8V4 at 32K and 47K; measure per-call KV-byte traffic / wall-clock.
  **This one experiment resolves five proposals at once**: if one KV scan serves
  K query rows, the speculative cluster collapses to "wire verify routing + add a
  draft model" (P1); if KV is re-scanned per row, speculation on affine is a
  correctness-only fix and we pivot to P1-a.
- **P0-c — Produce the non-zero-byte combined affine 16K quality+throughput
  artifact** (the literal first continuation task; the prior run wrote 0 bytes).
  Nothing promotes until this exists.
- **P0-d — Finish/run the A-series on-device harness** (`TurboQuantBench`,
  already built). ~half of P1–P3 is unprovable on M2 Pro's wide bus; A-series is
  the product target and where compression actually wins. Build a same-machine
  upstream (`arozanov/turboquant-mlx` @ `6e928d7`) baseline alongside.

### P1 — The shipping-path tok/s levers

- **P1-a — Raise the affine quantized-SDPA kernel's effective bandwidth.**
  This directly attacks the 0.45×-of-FP16-bandwidth finding (§1.2). After P0-a
  localizes the loss, candidate sub-levers (in MLX core `sdpa_vector.h` quant
  vector kernel): (i) **co-locate per-group scale/bias with the packed payload**
  so metadata is a contiguous prefetch, not a scattered fp32 stream (11–33 % of
  bytes today, worst-coalescing); (ii) **vectorize the packed + metadata loads**
  (uint4/half4) at the quant fetch sites; (iii) **raise decode occupancy** —
  more SIMD-groups per threadgroup (the `TURBOQUANT_PERF_AUDIT` item 1.1 lever),
  distinct from FD1's split-*count* axis. Expected: each sub-lever is small alone
  but together could move affine from 0.72× toward ~1.0–1.3× at 32K and beat FP16
  decisively >47K. **Gate strictly on P0-a; do not ship byte-cut codec levers
  (P2) before this, or they stay capacity-only.**
  *Risk:* MLX-core change → cross-repo (mlx → mlx-c → mlx-swift → pin bump);
  exact-prefill untouched (decode-only). Distinct kernel-variant names required
  (the LANES=4/1 aliasing gotcha).

- **P1-b — Speculative-decode amortization on the affine path.** The only lever
  that amortizes the *dominant weight stream* (937 MB/token) AND KV across
  accepted tokens — exactly the weight-dominated ≤32K regime. Mostly wiring (the
  kernel/admission/gate already accept q_seq 2..8): route
  `SpeculativeTokenIterator` verify (`Evaluate.swift:1798`) for affine caches to
  the fused multi-query kernel instead of the rawExact/prefill ladder
  (`AttentionUtils.swift:2259`), ship a quality-matched draft (Qwen3.5 0.6B
  sibling), measure acceptance. Per-emitted-token bytes drop by the acceptance
  factor A (honest A ≈ 1.3–1.8 for a 0.6B→2B pair). Expected **~1.3–1.8× at
  ≥47K**, conditional on P0-b confirming one-scan amortization and on the draft
  fitting the A-series budget (≈350–400 MB extra resident — it fights the budget
  compression frees).
  *Risk:* quality **none** (speculative decode is exact by construction);
  feasibility = draft-model dependency + A-series memory.
  *Note:* the same idea on the **PolarQJL JIT** path is a separate, harder
  rewrite (one threadgroup per (head,q_token) re-scans the cache → zero
  amortization without a per-key-loaded-once / Q-accumulator restructure) and is
  diagnostic-only → P3.

### P2 — Codec byte cuts (capacity now; speed only after P1-a)

These reduce bytes moved. Per §1.2 they convert to **speed** only once the kernel
is bandwidth-bound (P1-a); until then they are **capacity/peak-memory** wins
(valuable on A-series). Sequence them as one ordered workstream:

- **P2-a — Larger affine value group (32→64).** Halves per-group value
  scale/bias metadata (value metadata is ~33 % of value bytes at g=32). Cheap,
  reversible; the quality-curve probe for P2-b. Expected: ~8 MB less resident at
  32K; <2 % tok/s alone.
- **P2-b — MXFP4 / microscaling values.** Replace per-group fp16 scale+bias with
  a shared fp8-E8M0 block exponent, coalescing metadata into the payload.
  ~20–30 MB less resident at 32K; speed ~noise until P1-a. Only if P2-a shows
  headroom and the per-plane-mode kernel exists.
- **P2-c — QuaRot-style static Hadamard pre-rotation → affine K6.** Rotate keys
  once (offline, into the projection) to flatten outliers, enabling 6-bit keys at
  K8 quality. ~14–15 % total KV cut (128 B→96 B keys). Quality must hold
  (KL/p95/cosine gates). Speed contingent on a clean K6 unpacker staying
  coalesced — capacity win first.

### P3 — Diagnostic-only / A-series-gated

- **P3-a — Raise PolarQJL split-KV occupancy** (merge of FD1 + SS3-B): lift the
  `partial[512]` reduce cap (`TurboQuant.swift:15489`) with a *single
  strided-loop* reduce (the `sdpa_vector_2pass_2` shape), so `activeBlockCount`
  can exceed 512 at ≥64K. Raises in-flight DRAM requests. **A-series only** (M2
  Pro's wide bus likely saturates at 256 blocks; abort-on-regression). PolarQJL
  is diagnostic, so this never reaches production unless PolarQJL is promoted.
  Distinct reduce-kernel name required.
- **P3-b — async-eval pipelining at ≥64K** (sync interval 2–4; the env knob
  `TURBOQUANT_DECODE_SYNC_INTERVAL` already exists — A/B it for free first).
  ~3–10 % at 64K, tapering at 128K.
- **P3-c — per-device autotuner** for (LANES, BLOCK_TOKENS, split-count,
  sync-interval) keyed by (device, context, scheme). ~0 on M2 Pro
  (already tuned); a regression-prevention + A-series-fit mechanism. Build only
  after P0-d.
- **P3-d — streamed per-layer conversion** — reframe as a **transient-peak-memory**
  fix (per-layer eval+release caps the convert peak at one FP16 + one compressed
  layer instead of all-24 + all-24), not a TTFT fix (MLX already overlaps the 24
  lazy convert ops). Decides OOM-vs-fit at long context on A-series.

### Deprioritized / disproven (do not pursue as framed)

- Cooperative AV as a *coalescing* win (AV already coalesced); cooperative AV
  only has a 25 %-occupancy angle and only on the diagnostic JIT path.
- uint4 loads / fp16-scales / mxfp4 **on the PolarQJL JIT path** — that path
  doesn't ship; do these on the affine kernel (P1-a) instead.
- TQCOOP default-on for production before A-series validation (and note it never
  engages on 2-repeat models — needs 4-repeat GQA).
- Sparse-V / pageTopK / candidate-sparse promotion (real-model 0.07–0.17×).
- Any "we match/beat upstream" claim without a same-machine upstream baseline.

---

## 4. Under-covered avenues worth a dedicated look (gaps the proposal sweep missed)

1. **The weight stream itself.** It dominates ≤47K and no lever except speculation
   touches it. Worth a roofline check (P0-a) of whether the 4-bit weight matmuls
   are already coalesced/occupancy-optimal for batch=1 GQA decode, and whether
   anything (e.g. lm_head over the full 248K vocab, ~286 MB/token, ~30 % of
   weights) can be trimmed (top-k vocab / tied-head tricks) at decode.
2. **TTFT / prefill** is ~90 % uncovered. Long-prompt latency is user-visible;
   the chunked prefill compute and the ~2.4 s@16K conversion stall both deserve
   first-class treatment (P3-d is the memory half).
3. **Short-context host overhead.** turbo4v2@8K = 0.10× FP16 is almost certainly
   *fixed per-token dispatch/host overhead* (graph rebuild, Swift sampling loop
   `Evaluate.swift:1814`), not bytes. The dynamic decision already routes short
   context to FP16, so this is low product priority — but if compressed must run
   short, the structural fix is host-side, not kernel-side.

---

## 5. Sequencing, conflicts, and validation discipline

- **Order:** P0-a/b/c (instrument + microbench + gate) → P1-a (affine kernel
  efficiency) → P1-b (speculation) and P2 (codec, now speed-relevant) in parallel
  → P3 on A-series. **Never** land a codec byte-cut before P1-a, and **never**
  land a PolarQJL kernel change ahead of an affine one (only affine ships).
- **Kernel-variant aliasing is a shared hazard.** Every new kernel variant must
  use a distinct name or MLX's compiled-variant cache cross-contaminates
  (the LANES=4/1 gotcha). Two variant-adding changes in the same window must
  coordinate names.
- **The pin trap.** mlx-swift-lm pins mlx-swift to a git rev and mlx-swift
  bundles mlx as a submodule; local kernel edits are silently ignored unless you
  `swift package edit` + `touch` + verify the marker is in the built binary with
  `LC_ALL=C grep -a -c` (NOT `strings`). Verify every kernel change actually
  ships before trusting a benchmark.
- **Promotion gate (unchanged):** real-model throughput + same-report quality +
  selected native compressed path + no raw/decoded fallback + resident
  compression >1.0× + peak/steady memory + same-machine upstream + A-series
  device evidence. The combined affine 16K artifact (P0-c) must exist first.

---

## 6. Corrections to prior belief (recorded so we don't relapse)

1. **"Cooperative AV / uint4 loads are the next kernel wins."** Wrong on the
   shipping path: AV is already coalesced, and both ideas live on the
   diagnostic PolarQJL kernel. The affine production path is MLX-stock.
2. **"Decode is uniformly memory-bandwidth-bound."** True for the PolarQJL JIT
   kernel (where TQCOOP's A/B proved it). The **affine production kernel is
   overhead/occupancy-bound** — it moves 1.6× fewer total bytes than FP16 yet
   realizes <0.5× its bandwidth. The fix is kernel *efficiency*, not fewer bytes.
3. **"KV is the decode bottleneck."** Only past ~47K. At the measured ≤32K
   regime, weights dominate (≈59 % at 32K), so KV-byte levers are sub-linear and
   speculation (weight amortization) is the bigger lever there.

---

## 7. Concrete next commands

```bash
# P0-c: the gating combined affine 16K artifact (rerun; prior was zero-byte)
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift build -c release --product TurboQuantInferenceParity
TQ_QUALITY_PRINT_CACHE_DIAGNOSTICS=1 .build/release/TurboQuantInferenceParity \
  --model-dir /path/to/Qwen3.5-2B-4bit --contexts 16384 --generate-tokens 32 \
  --throughput-repeats 1 --configs fp16,affineK8V4 --quality-gates \
  --quality-contexts 16384 --turboquant-timing \
  --diagnostics-output artifacts/turboquant-current/affine-16k-combined.json

# P0-b: q_seq=1 vs q_seq=4 affine KV-traffic microbench (settles the spec cluster)
#   -> extend TurboQuantNativeVxBenchmark (or a new probe) to call the affine
#      quant_sdpa_vector_2pass at queryLengths 1 and 4 at ctx 32768,49152 and
#      record per-call bytes-moved + tok/s. Confirm fused (not matmul) dispatch
#      via LC_ALL=C grep of the kernel marker in the built binary.

# P0-d: on-device A-series harness + same-machine upstream baseline
xcodebuild test -scheme mlx-swift-lm-Package \
  -destination 'platform=iOS,name=<device>' \
  -only-testing:MLXLMTests/TurboQuantBenchSuite TQ_BENCH=1
```
