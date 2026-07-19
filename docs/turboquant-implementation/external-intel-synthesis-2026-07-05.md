# External intel synthesis and evolved plan — 2026-07-05

> **SUPERSEDED IN PART (annotated 2026-07-19).** This doc's central framing —
> "the ≤32K deficit is HOST-side; the SDPA kernel is exonerated; stop attacking
> it with kernel work" — was based on the favorable hd128/kv8 synthetic shape.
> Later evidence: P1-1 fused-append falsified the host-ladder hypothesis
> (2026-07-05/06), M1 measured the kernel at 0.304× fp16 at the real
> 8q/2kv/hd256 shape (2026-07-10, `artifacts/affine-occupancy-20260710/`), and
> the 2026-07-19 stability probe (`artifacts/deficit-stability-20260719/`)
> showed the isolated microbench itself has ±50% round variance at that shape —
> disqualifying it as an instrument. Current state: the deficit is real
> (real-model 0.718× @16K is the reliable ground truth), its locus is OPEN, and
> adjudication must use real-model A/Bs or driver counters. See the 2026-07-19
> section of `continuation-handoff-2026-07-03.md`.

Six external sources were deeply investigated by parallel research agents and
cross-checked against local repo state (three additional grounding agents over
docs, kernel trees, and the affine path). Full evidence reports are archived at:

```text
artifacts/external-intel-20260705/
  medium-flashattn.md      VeloxQuant-MLX / Medium FlashAttention post-mortem
  mlx-qsdpa.md             Thump604/mlx-qsdpa fused quantized SDPA package
  mlx-issue-2395.md        Masked-row semantics, PR #2406 / #2608 contract
  mlx-sdpa-dispatch.md     Full SDPA dispatch tree at our mlx pin
  mlx-lm-pr-1067.md        arozanov TurboQuant mlx-lm PR + mlx #3328/#3026
  reddit-swiftlm.md        SharpAI SwiftLM + HN thread + ecosystem survey
  local-plan-state.md      Plan/debt/graveyard grounding
  local-kernel-state.md    Kernel-tree + uncommitted Phase 0 grounding
  local-affine-path.md     Affine route trace + 0.72x suspect ranking
```

Ideation/verification was completed inline against the measured-physics
constraints and the 25-item graveyard (the multi-agent ideation fleet was
blocked by a spend limit; every idea below therefore carries its evidence
chain and a falsifier so it can be independently re-verified). Authority
note: where this doc and older docs disagree, `continuation-handoff-2026-07-03.md`
plus the falsifier-verdict rerun are the current ground truth this doc builds
on: coop re-validated (G1 1.63x, G2 1.34x at 131K, engagement-proofed), v7
tile-transpose falsified, Phase 0 + T1.1/T1.2/T1.4 landed but uncommitted.

## 1. Headline: the second reframe of the 0.72x

The 2026-06-07 reframe said the affine route is overhead/occupancy-bound, not
bandwidth-bound. The new evidence sharpens that into something more specific
and more actionable:

**At <=32K the affine deficit does not live in the SDPA kernel at all.**
The on-disk roofline artifact (`artifacts/turboquant-roofline-20260607/`)
shows attention-ISOLATED affine K8/V4 is 1.12x FASTER than FP16 SDPA at 32K
(685 vs 611 calls/s), while the real-model 16K artifact shows 0.752x. Since
attention is ~15% of token bytes at 16K, the kernel cannot account for the
~4.5 ms/token real-model deficit. The top-ranked suspects (local-affine-path
report, with file:line):

- **Per-step cache-append ladder**: `AffineK8V4KVCache.updateQuantized` emits
  2 quantize kernels + 6 slice_updates + 6 slice views per layer per token
  (+150-290 dispatches/token model-wide), against a command buffer that
  commits every 40 ops on M2 Pro ('g') and every 20 ops on A-series ('p').
- **SliceUpdate donation-failure hazard**: `SliceUpdate::eval_gpu` copies the
  full-capacity cache plane whenever any slice view is alive
  (`use_count!=1`) — the same refcount family as the fixed 3A-a JIT bug,
  potentially copying hundreds of MB per token silently.
- Kernel-internal costs (block-ladder over-split, register pressure, serial
  softmax, 4 separate scale/bias streams) matter only for the >47K
  bandwidth-gap lever, not for the <=32K product window.

Four independent external implementations replicate the "overhead, not
bytes" physics on different hardware and codebases:

| Implementation | Result | What it isolates |
|---|---|---|
| Ours (fork, fused native) | 0.752x @16K real model; kernel-isolated 1.12x @32K | deficit is outside the kernel at <=32K |
| mlx-qsdpa (M2 Ultra) | non-GQA 1.17-1.28x FP16 @64-128K, GQA-16 only 0.61-0.78x | per-query-head duplicated dequant+FMA, not bytes |
| arozanov mlx-lm PR | K8+V2 == K8+V4 speed (25.52 vs 25.84 tok/s) | not bandwidth-bound |
| VeloxQuant-MLX (M4) | 16x fewer bytes, 3-4x SLOWER than fused SDPA | occupancy-starved (1,024 threads, ~1.2 GB/s effective) |
| sharpner LEAN (M4 Max) | stock affine ~105% of fp16 @8K | affine CAN exceed fp16; 0.72x is not physics |

## 2. Source digests (what each contributed)

**Medium / VeloxQuant-MLX.** The article's "beat MLX SDPA" was a mislabeled
baseline: full-cache dequant + inverse-WHT matmul + GQA `mx.repeat` + unfused
fp32 softmax. Against real fused SDPA his kernel was 3-4x slower; end-to-end
the fused path changed nothing (2.1 = 2.1 tok/s) and peak memory went UP
under 16x "compression" (persistent fp16 mirror + additive indices). Shipped
with a deliberately no-op dispatcher. Transfers: bytes/bandwidth-bound sanity
checks on every baseline; the q-codebook LUT (ADC) trick is pure overhead at
q_seq=1 but amortizes across a speculative verify batch.

**mlx-qsdpa.** Real, working fused affine quantized SDPA via
`mx.fast.metal_kernel` (simdgroup-per-key, lanes split head_dim, split-K
2-pass above 4K). Its GQA-vs-non-GQA data is the cleanest isolation of the
quantized-decode gap: per-query-head duplicated dequant+FMA. Its own v0.1.0
"0.98x FP16" claim was retracted by its own better benchmark. Also surfaced:
upstream mlx PR #3026 (generic native quantized SDPA, open — the future stock
baseline) and PR #3307 (fused steel prefill killed by the AGX GPU watchdog at
128K keys; mitigated by 32K-chunk logsumexp reduction).

**mlx issue #2395.** Final upstream contract (PR #2608, in our pin):
fully-masked rows produce unspecified FINITE values, never NaN — and the
values are legally path-dependent (zeros on the vector path, non-zero on
steel bool). Upstream's fix deliberately removed per-key branches from the
online-softmax hot loop (corroborates our branch-free discipline). Our JIT
kernels still use pre-#2608 `-INFINITY` row-max inits — safe today only
because causal decode never yields a fully-masked row. Float 0/1 masks are
silently ADDITIVE (the #2626 footgun).

**MLX SDPA dispatch tree (local pin, fully contains upstream to 2026-06-26).**
FP16: q_seq<=8 AND q_seq*gqa<=32 AND head_dim in {64,96,128,256} -> fused
vector kernel; q_seq>8 -> steel; else silent unfused graph. Fork-added
quantized SDPA: q_seq<=32, always 2-pass quad-per-key — already
TQCOOP-shaped, with 2 dispatches + 3 temporary allocations per layer per
token. Compressed speculative verify is natively supported to q_seq=32 —
MORE permissive than FP16, whose q_seq*gqa<=32 gate silently un-fuses drafts
(Qwen2.5-7B gqa=7 caps k<=4). The ghost modes and block-count env overrides
we planned to build already exist in-tree. Sparse-V's kernel (3 full key
sweeps, one simdgroup per q head, no split-K) is structurally unable to win.

**mlx-lm PR #1067 (arozanov).** Unmerged, conflict-dirty, zero maintainer
review; companion native-kernel PR mlx #3328 was closed ON POLICY (generic
quantized SDPA first, no scheme-specific builtins) — so our fork's native
TurboQuant kernels are a permanent fork asset, not an upstreamable diff. The
PR's cache dequantizes into a persistent full-context FP16 mirror (resident =
packed + FP16, worse than plain FP16) while `nbytes` reports packed only.
The 0.72x upstream row was measured through an UNFUSED Python
quantized_matmul chain — it is a composition floor, not a kernel ceiling.
Borrowables: first/last-N-layer FP16 exemption, storage-tier-only
quantization of parked sessions, disk-cache free-memory guard + class
allowlist, MLA/SSM incompatibility guards. Bench-fraud-by-accident case: a
shape bug made a kernel attend over ~8 keys and produce a plausible
constant-time table.

**Reddit SwiftLM + ecosystem.** SwiftLM is an unrelated, largely
agent-generated port whose headline claim is falsified by its own shipping
code (CPU scalar codec, full-history CPU dequant per call, dead Metal
helpers; 0.19x dense at 40K). Its value is methodological: phys_footprint
saturates at the RAM ceiling and hid a 17.6 GB win until they switched to
ioreg AGXAccelerator "Alloc system memory"; an unconditional per-layer sync
cost 3.4x; a degenerate-output speculative benchmark inflated acceptance.
Ecosystem: TheTom/llama-cpp-turboquant holds the real fused Lloyd-Max+QJL
Metal decode this cluster derives from; arozanov's butterfly-pulled-out WHT
trick (accumulate weighted centroids, single butterfly at end) claimed 4.5x
on V-attention; sharpner: never trade an MSE bit for a QJL bit (softmax
amplifies centroid loss); "turboquant-mlx" is TWO different repos (arozanov
vs sharpner) — always disambiguate by full name + commit.

## 3. Corrections to current plans (blunt)

1. **Stop attacking the <=32K affine 0.72x with SDPA-kernel work.** The
   kernel ties/beats FP16 in isolation there; the deficit is host-side
   (cache-append dispatch ladder + donation hazard + buffer-commit cadence).
   P1-a metadata-colocation / vectorized-load framing is demoted (~12% DRAM
   headroom at 16-32K, ~0% beyond, per ghost decomposition).
2. **"Effective bandwidth 0.39-0.62x" is a red-herring metric** for the
   affine route (launch/merge tracks 45-63% of kernel time). Bandwidth talk
   is only meaningful for the >47K lever after a ghost-mode DRAM verdict.
3. **"Upstream parity" against arozanov's 0.72x row is a weak bar** — that
   number is an unfused-Python floor. The real target is FP16 parity at
   <=32K via host fixes, and >1x only at >47K where KV bytes dominate.
4. **Scheme-specific kernels will never merge upstream** (mlx #3328 closed on
   policy). Plan for permanent fork maintenance; re-baseline against stock
   the day mlx #3026 merges.
5. **v7 tile-transpose stays dead; coop stays alive** (re-validated 1.34-1.63x
   at 131K, engagement-proofed). Any future layout work must first explain why
   the transpose recovered only ~77-79% of coop's win.
6. **N7 prefetch claims must use reconcile-cycle numbers** (~1.3-1.65x
   ceiling at >=16K greedy), not the unreproduced 1.76x.
7. **Sparse-V is structurally dead under the current kernel** — stop
   measuring it; the nonclaim is permanent until the kernel inherits 2-pass
   split geometry.
8. **The 32K/16GB "4.755x" spec-decode result is metric-confounded**:
   phys_footprint saturates at the RAM ceiling. Re-measure with the AGX
   driver-allocation counter before citing it.
9. **A-series will not inherit M2 Pro tunings**: arch 'p' commits every 20
   ops (vs 40), has different block ladders, and gen>=18 flips prefill to NAX
   steel. Record `mx.metal.device_info().architecture` in every artifact.

## 4. Evolved plan

### P0 — run next on M2 Pro (no new kernels; evidence + zero-code falsifiers)

- **P0-1 Commit the uncommitted stack.** ~2,000 lines across mlx-swift +
  Cmlx/mlx submodule + mlx-swift-lm (probe, T1.2, T1.4, T1.1, v7+race-fix),
  submodule gitlink unbumped, package-edit symlink session-scoped. The only
  copy of validated work is the working tree. Highest process risk in the
  workspace.
- **P0-2 Produce the randomized-repeats combined affine 16K/32K artifact.**
  Two single-sample artifacts already exist (0.752x + quality PASS;
  promotionEligible=true on Qwen3-0.6B) but the docs still carry the debt as
  open. This gates fp16-scales stage 2 and every promotion claim. Also
  reconcile the stale zero-byte-debt wording in the 07-01/07-03 docs.
- **P0-3 Affine host-side falsifier battery** (all env-gated, all >=25-iter
  interleaved A/B per `run_ab.sh`; artifact:
  `artifacts/affine-hostside-20260705/`):
  - (a) `MLX_MAX_OPS_PER_BUFFER` 40 vs 400 at 16K real-model — tests
    dispatch/flush amplification (H1).
  - (b) Donation-hit-rate counter (5-line env-gated patch in mlx copy path)
    — tests SliceUpdate full-plane copies (H2).
  - (c) `TURBOQUANT_SDPA_DECODE_BLOCKS` sweep {64,128,256,512,1024} at
    32K/65K/131K — validates the single-sample ~256-cap +10-15% finding.
  - (d) `TURBOQUANT_GHOST_SDPA_MODE` 0/2/3 interleaved — splits launch/merge
    vs DRAM vs ALU shares; K4/K5-class kernel work stays gated on a DRAM
    verdict.
  - (e) Dispatch-count trace pointed at the affine route (the T1.1 lesson:
    "slow kernels" were host overhead; verify which kernel actually serves
    tokens — the router auto-upgrades layout-6 to nativeMLXCompressed).
- **P0-4 Hardening + methodology batch** (each small, fail-closed):
  - Mask-dtype assertion at the Swift boundary: bool or exact -inf only
    (float 0/1 masks are silently additive — mlx #2626).
  - JIT kernels: adopt the #2608 recipe (finite_min row-max init, keep
    row_sum>0 guards) + fix the latent v6 threadgroup RAW race, under the
    new-kernel-name protocol.
  - TurboQuantBench: context-scaling sanity assertion (decode kernel time
    must grow with context; flat curve = automatic red flag — mlx #3328
    accident) + bytes/bandwidth-bound baseline sanity check (VeloxQuant).
  - Diagnostics schema: add ioreg AGXAccelerator "Alloc system memory"
    (GPU-driver allocations) alongside phys_footprint; re-measure the
    32K/16GB-confounded rows.
  - Parity harness: never assert on fully-masked-row values (legally
    path-dependent); exclude pad rows from cosine/logit comparisons.
  - Spec-decode benchmarks: repetition/entropy check on generated text
    (SwiftLM DFlash lesson — greedy-exact verify protects correctness, not
    benchmark honesty).

### P1 — this cycle, contingent on P0-3 verdicts

- **P1-1 Fused quantize-append cache-update kernel** (the big <=32K lever).
  One kernel per layer per token writing K codes+scales+biases and V
  codes+scales+biases in place: collapses ~8 graph nodes / 14 slice ops per
  layer to 1-2 dispatches and structurally sidesteps all six donation-hazard
  slice_updates. Likely worth more real-model tok/s at <=32K than any SDPA
  kernel change; the win amplifies on A-series ('p' = 20 ops/buffer).
  Cross-repo bottom-up (mlx kernel -> mlx-c -> mlx-swift -> cache), new
  kernel name, opt-in until the interleaved A/B passes. Gate: run after
  P0-3(a)/(b) confirm host-side ownership.
- **P1-2 Donation fix for the append path** (only if H2 fires): scope/nil
  retained slice views before append or append into a non-view-exposed ring
  — the known 3A-a recipe; cheaper than P1-1 and may capture most of its win.
- **P1-3 Speculative-verify pipeline hygiene**: quantize draft lengths to a
  fixed set {1,2,4} (q_seq is a function constant — every new length is a
  fresh PSO compile) + pre-warm those PSOs; cap affine verify k<=4 at D=128
  (measured qL=8 register-spill regression at 32K), lower for D=256; on the
  FP16 plain-KV admission path cap k <= min(8, 32/gqa) or verify silently
  un-fuses (Qwen2.5-7B gqa=7 -> k<=4). Always pass causal from the cache
  layer (free at q_seq==1, keeps the branchless fast loop).
- **P1-4 Block-ladder cap productization**: if P0-3(c) validates ~256 on
  'g', make it a per-arch table (function of device class + context), not an
  env var; A-series value must be re-measured on device.
- **P1-5 fp16-scales stage 2** (another ~halving of scale bytes; machinery
  complete): unblocked the moment P0-2 lands.
- **P1-6 Storage-tier-only compression** (product, zero live-decode cost):
  quantize KV of parked/LRU-evicted sessions, dequantize on resume —
  complements FP16-first admission and sidesteps the decode-speed problem
  entirely for the multi-session Pines case. Include the PR #1067 disk-cache
  guards: free-memory check before serialization (it transiently doubles
  memory) and a deserialization class allowlist. Quality gate on the resume
  path (roundtrip error equals live-compressed error; must be reported).

### P2 — needs P0/P1 verdicts or device data first

- **P2-1 GQA-pair dequant amortization** in the mixed_quant 2-pass kernel:
  at our 2-repeat GQA, hold both q-head vectors per thread, dequant K/V
  once, two dots/accumulates (~2x32 halfs extra registers at D=128). The
  mlx-qsdpa non-GQA-vs-GQA isolation says duplicated dequant is the
  quantized-decode gap; at 2-repeat the duplication is only 2x, and the
  kernel is launch/merge-bound at <=32K — so this is a >47K lever. Gate:
  ghost-mode DRAM/ALU verdict at 65K/131K first.
- **P2-2 First/last-N-layer FP16 exemption** as a capacity-tier profile
  option (upstream field data: default 1, MoE models needed 4-6) — quality
  lever for V3/low-bit tiers, no kernel work.
- **P2-3 Long-context prefill watchdog guard**: fused steel prefill dies by
  GPU watchdog at 128K keys on macOS (mlx #3307, closed unmerged). Audit our
  exact-prefill path at >=64K; add a chunked-logsumexp fallback or a
  fail-closed context cap before any 128K prefill claim.
- **P2-4 Register-pressure menu from mlx #3026** (drop per-thread k/v
  storage, uint16-vs-uint32 read width, manual unroll) — only after a ghost
  verdict implicates ALU/registers; D=256 models (Qwen3.5-2B class) are
  worst-hit (128 fp32 registers/thread).
- **P2-5 MLA/SSM cache-class admission guards** in TurboQuantAdmissionPlanner
  (upstream shipped silent garbage on MLA before guarding; we should refuse
  loudly).

### X — experimental/diagnostic tier, opt-in only

- **X-1 Butterfly-pulled-out WHT decode** for the PolarWHT-V diagnostic tier
  (arozanov: accumulate weighted centroids per-thread, single butterfly at
  end; claimed 4.5x on V-attention). Compute reduction on an access-bound
  kernel is usually hidden — ghost-gate before building.
- **X-2 q-LUT (ADC) amortization** for codebook tiers at q_seq>1 only
  (measured pure overhead at S_q=1); fold into the q_seq microbench if a
  PQ/codebook path is ever revisited.
- **X-3 Transient-dequant mid-tier**: store quantized, decode via chunked
  transient dequant + FP16 SDPA in the band where the quantized kernel still
  loses after host fixes (mlx-qsdpa ships this as its <16K crossover tier).
  Under our rules this is a declared decoded-fallback tier: diagnostics-
  visible, transient scratch budgeted by the 4.1 OOM guard, and PERMANENTLY
  promotion-ineligible unless the promotion checklist is explicitly revised.
- **X-4 Focused read of TheTom/llama-cpp-turboquant** (the real fused
  Lloyd-Max+QJL Metal decode, actively developed) for JIT-tier technique
  mining.
- **X-5 Low-bit K rules if ever revisited**: test 3-bit-codebook K before
  4-bit (upstream: 3-bit fits the post-rotation distribution better; K4
  destroys greedy decode even native); never trade an MSE bit for a QJL bit
  (softmax amplifies centroid-resolution loss); expect head_dim=128 to be
  the hardest quality case.

## 5. Sequencing (next 2-3 cycles)

**Cycle 1 (M2 Pro, no new kernels):** P0-1 commit -> P0-2 combined artifact
-> P0-3 falsifier battery -> P0-4 hardening batch. Exit criteria: H1/H2
verdicts in hand; block-cap validated or killed; combined artifact filed;
docs debt reconciled.

**Cycle 2 (contingent):** P1-2 if H2 fired (cheap recipe first), then P1-1
fused quantize-append (cross-repo, opt-in); P1-3 spec hygiene; P1-4 ladder
productization; P1-5 fp16-scales stage 2; start P1-6 storage-tier design.
Exit criteria: real-model 16K/32K affine ratio re-measured after host fixes;
FP16-vs-compressed admission crossover re-derived from the new numbers.

**Cycle 3 (device):** the single highest information-per-hour item remains
one A-series cycle via TurboQuantBench carrying the free instruments:
block-ladder retune, ops-per-buffer cadence, dispatch counts, spec-decode
crossover, coop fate on 'p', plus the Pines compatibility pair. Record
architecture strings in every artifact. Only after this do device/product
claims unlock.

## 6. New nonclaims (additions to the standing list)

- "V3 quality at V2 speed" (falsified by SwiftLM's own code and tables).
- Any iPhone TurboQuant performance number (none exist anywhere in the
  ecosystem, including ours).
- VeloxQuant "VecInfer-1bit exceeds fp16 throughput" (its fused dispatcher
  is a no-op at v0.25.0).
- Beating arozanov's 0.72x row as evidence of kernel superiority (it is an
  unfused-Python floor).
- Speculative-decode speedups measured on degenerate/repetitive output.

## 7. Watch list

- **mlx #3026** (generic native quantized SDPA): merge = the stock baseline
  moves; re-baseline before any upstream-comparison report.
- **mlx-lm #1067**: stalled (conflicts, zero review) — recheck before any
  upstream-comparison claim; also #1073/#1074.
- **llama.cpp**: ggerganov's attn-rot merged mainline; TQ3_0 3.5bpw KV PR
  closed — note both in any cross-runtime comparison.
- **VeloxQuant-MLX / rachittshah/mlx-turboquant / sharpner/turboquant-mlx**:
  growing same-machine comparison landscape; always cite full repo + commit
  ("turboquant-mlx" alone is ambiguous).
- Post-pin upstream SDPA delta is currently a single irrelevant change
  (asymmetric MLA head dims, c9ccaba997) — tracking cost is zero today.
