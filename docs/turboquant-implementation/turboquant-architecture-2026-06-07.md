# TurboQuant — Full Design & Architecture Report (for external audit)

2026-06-07. This is a self-contained technical description of the TurboQuant
long-context KV-cache-compression system: the architecture, every codec's math,
how the decode kernels are actually built (Metal-level), the decode-time
algorithms, the measured evidence, and an explicit list of what an external
reviewer should scrutinize. Every load-bearing claim carries a `file:line` anchor
so it can be checked against source. Repos referenced: `mlx` (C++/Metal core,
bundled in mlx-swift under `Source/Cmlx/mlx`), `mlx-c`, `mlx-swift`, `mlx-swift-lm`.

---

## 0. Executive summary — the thesis and the honest state

TurboQuant compresses the transformer KV cache so a phone can run long contexts.
The project's central, measured findings (which an auditor should test first):

1. **Two decode-kernel families exist; only one ships.**
   - **Production = affine K8/V4** (8-bit keys, 4-bit values, plain min/max affine
     quant) running on MLX's *stock* fused quantized SDPA. Measured **0.752× FP16**
     decode at 16K on Qwen3.5-2B-4bit, KV resident **2.13×** smaller, quality
     lossless (top-1 1.000, KL 1.7e-6, cosine 0.9931).
   - **Diagnostic = PolarQJL / PolarWHT** (the Google TurboQuant paper, arXiv
     2504.19874): rotation → data-free Gaussian codebook → unbiased 1-bit residual.
     Theoretically superior per-bit (4–7×) but runs on a custom JIT Metal kernel
     that is memory-access-bound on Apple GPUs and does **not** beat affine in
     practice, so it is gated out of production.

2. **Decode is weight-dominated, not KV-dominated, at product contexts (≤47K).**
   Per token ~937 MB of fixed weight traffic (MLP ~507 + attn-proj ~141 + tied
   lm_head ~286) dwarfs affine KV (164/654/2617 MB at 8K/32K/128K). The affine
   SDPA kernel is **occupancy/latency-bound at batch=1**, realizing only
   **0.39–0.62× of FP16-SDPA effective bandwidth** — so shrinking KV bytes does
   **not** convert to speed (a value group 32→64 change was measured speed-neutral,
   quality-worse). KV exceeds weights only past ~47K (affine) / ~19K (FP16).

3. **Therefore the only lever that raises real-model tok/s while keeping quality
   and compression is amortizing the weight stream across emitted tokens —
   speculative decode.** We ship a **no-draft, bit-exact** prompt-lookup
   speculator (lever ①). It is **context-gated** (crossover ≈12–16K on Qwen3-4B)
   because the multi-query verify forward cost scales with `q_seq`.

4. **Honesty guardrails (non-claims).** No 0.98× FP16 claim. No PolarQJL speed
   parity. The N4 Gaussian quantizer is validated at the CPU-codec level only (no
   Metal kernel yet). All speedups are M2 Pro single-sample synthetic — not
   promotion-grade (no randomized repeats, no same-machine upstream baseline, no
   on-device A-series). See §9.

---

## 1. System architecture

### 1.1 Repo stack and dependency shape
Downstream depends bottom-up: **Pines (iOS app) → mlx-swift-lm → mlx-swift → mlx
(core) / mlx-c (C ABI)**. Ownership: `mlx-swift-lm` owns model/cache policy,
admission, quality gates, benchmarks; `mlx-swift` owns Swift MLX bindings, native
capability probes, Metal packaging (and bundles `mlx` as a git submodule under
`Source/Cmlx/mlx`); `mlx`/`mlx-c` own the C++/Metal kernels and C ABI. Cross-repo
features go bottom-up; downstream pins move only after the fork stack validates.
**Build trap:** `mlx-swift-lm/Package.swift` pins `mlx-swift` by exact git rev, so
local kernel edits in mlx-swift are ignored unless the pin moves or
`swift package edit` is used (then `touch` + verify the marker is in the binary
with `LC_ALL=C grep -a -c`, *not* `strings`).

### 1.2 KV-cache type hierarchy (`KVCache.swift`, 5015 lines)
All caches subclass `BaseKVCache` (`KVCache.swift:270`).
- **Plain/upstream:** `KVCacheSimple` (:463, contiguous FP16, 256-token block
  growth), `RotatingKVCache` (:686, sliding window: pinned `keep` prefix + ring).
- **Quantized:** `QuantizedKVCache` (:1048, (wq:uint32, scales, biases) tuples),
  `AffineInt4KVCache` (:1337).
- **Production compressed:** `AffineK8V4KVCache` (:1353) — stores K and V each as a
  `(wq:uint32, scales, biases)` quantized tuple (+ optional V residual), grows in
  256-rounded blocks, writes **in place** into pre-sliced ranges
  (`updateQuantized` :1712), and **overrides plain `update` to `fatalError`**
  (:1839) to force callers through the quantized path. Carries a `KVCacheSimple`
  `rawFallbackCache` for unsupported head dims.
- **Diagnostic:** `TurboQuantKVCache` / `RotatingTurboQuantKVCache`
  (`TurboQuantKVCache.swift:4691`, PolarQJL codes + fallback ladder);
  `TurboQuantHybridKVCache` (per-layer affine-K + PolarWHT-V).
- **Hybrid recurrent:** `MambaCache` (`KVCache.swift:2694`) holds the Qwen3.5
  GatedDeltaNet SSM state in exactly 2 slots — **no offset-indexed K/V**.
  `CacheList` (:2723) composes per-layer caches, so the Qwen3.5 hybrid mixes
  `MambaCache` (18/24 recurrent layers) with an attention cache (6/24 layers).

### 1.3 Attention routing and the native gate (`AttentionUtils.swift`, `KVCache.swift`)
A forward produces an `AttentionKVState` enum (`.raw/.quantized/.turboQuant/
.hybridTurboQuant`, `AttentionUtils.swift:65`) and calls
`attentionWithKVStateThrowing` (:1817). In `.quantized`, the cache is type-checked
in priority order: native-affine-K8V4 → native-affine-int4 → TurboQuant-compressed
→ raw/decoded fallback. The **hard gate** for the production path is
`supportsNativeAffineK8V4ScaledDotProductAttention` (`KVCache.swift:4095`): GPU
device; env kill-switch unset; 4-D uint32 weights, fp16/bf16/fp32 queries;
**q_seq ≤ 32**; head_dim ∈ {64,128,256,512} divisible by both group sizes; GQA
factor integral ≤ 32; **key bits == 8**, value bits ∈ {2,3,4}, group sizes ∈
{32,64}. If it passes, `mixedAffineK8V4ScaledDotProductAttention` (:4192) calls
`MLXFast.mixedQuantizedScaledDotProductAttention` (the fused kernel, §2). If not, a
**decoded fallback** manually expands GQA and does `quantizedMM` QK + softmax +
`quantizedMM` AV (:4292–4351) — slower, allocates fp16 intermediates, flagged by
policy as a decoded fallback to avoid in hot paths.

### 1.4 Dynamic admission & lifecycle
The dynamic FP16-vs-compressed decision uses **plain FP16 when it fits the live
memory budget** (faster) and compresses only to reach contexts FP16 can't.
`quantizedKVStart` (default 16384) keeps short caches raw; `maybeQuantizeKVCache`
converts FP16→compressed once `offset ≥ quantizedKVStart`, materializes the
converted state, synchronizes, and clears cache pressure. **Exact prefill** is
preserved: prefill logits are produced on the raw path before conversion.

---

## 2. The production kernel — fused affine quantized SDPA (how it is built)

Files: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h`,
`.../backend/metal/scaled_dot_product_attention.cpp` (cited as `sdpa.cpp`).

This is a **2-pass split-KV (flash-decoding) vector kernel** with on-the-fly affine
dequant. It never materializes a dequantized FP16 cache.

### 2.1 Data flow
Swift admits affine decode → MLX builds a `QuantizedScaledDotProductAttention`
primitive over `[q, k_codes, k_scales, k_biases, v_codes, v_scales, v_biases,
(mask), (sinks)]` (`sdpa.cpp:1207`). `eval_gpu` checks `use_fallback` (:1012); if
accepted, normalizes layouts and calls `quant_sdpa_vector_2pass` (:1346).

### 2.2 Split-KV structure + block ladder
`quant_sdpa_vector_2pass` (`sdpa.cpp:681`) picks `blocks = select_sdpa_blocks(...)`
and dispatches **PASS 1** over grid `(num_kv_heads, batch, blocks*q_seq_len)` with
threadgroup `(32, gqa_factor, 1)` (`sdpa.cpp:737`). Each PASS-1 threadgroup is
exactly **one 32-lane SIMD group × gqa_factor query heads** stacked in `tg.y` to
share the KV L2 working set. It strides the sequence
`for(i=block_idx*BN+local_quad_gid; i<N; i+=blocks*BN)` (`sdpa_vector.h:874`) and
writes its block's `(global_sum, global_max, partial_out)` (:978). **PASS 2**
(`sdpa.cpp:840`, grid `(batch*heads, q_seq, 1)`, group `(1024,1,1)`) merges the
per-block stats with stable online-softmax rescaling and divides by the global
denominator (`sdpa_vector.h:1528`). **The split is mathematically exact** — only
parallel granularity changes.

The block ladder (`select_sdpa_blocks`, `sdpa.cpp:22`): quantized decode (q_seq≤1)
returns **512 at N≥32K (≤65536) else 1024**; Apple-silicon quantized N≥16384 with
n_simds≥4 → 256 (≤65536) else 1024. Env overrides exist
(`TURBOQUANT_SDPA_DECODE_BLOCKS`). **Design reason:** one threadgroup scanning tens
of thousands of tokens both serializes and risks macOS aborting the command buffer
as an interactivity hazard (comment `sdpa.cpp:47`); splitting raises occupancy and
bounds per-threadgroup runtime.

### 2.3 Quad-per-key lane mapping
`BD = (D>256)?8:4`, `BN = 32/BD`, `elem_per_thread = D/BD` (`sdpa_vector.h:783`).
For head_dim D=128: **BD=4, BN=8, elem_per_thread=32**. The 32-lane SIMD group is
partitioned into **BN=8 quads of BD=4 lanes**: `local_quad_gid = simd_lid/BD`
(which key), `local_quad_lid = simd_lid%BD` (which contiguous head-dim slice). So 8
quads process 8 different keys at once, and the 4 lanes of a quad each own a
contiguous 32-element slice of that key's 128-dim vector — **head_dim split across
lanes for coalesced loads** (`sdpa_vector.h:474`). Score reduction: `quad_sum` over
the 4 lanes, then `simd_shuffle_xor` ladder for cross-quad (`:878`, `:984`).

### 2.4 Affine dequant folded into the hot loop (no full-cache dequant)
`QuantOps<mode,bits,group>::dot` (`sdpa_vector.h:224`) computes the QK contribution
for one key over a lane's slice **without materializing dequantized K**. The affine
reconstruction `x = scale·code + bias` is folded as a single FMA at group end:
`score += fma(scale, Σ q_k·code_k, bias·Σ q_k)` (`sdpa_vector.h:323`) — algebraically
`Σ q_k·(scale·code_k + bias) = scale·(Σ q_k·code_k) + bias·(Σ q_k)`.
`::accumulate` (`:332`) does AV: `o = o·factor + exp_score·(scale·code + bias)`.
8-bit uses a word-fast path reading 4 codes at a time as uint32 (`:251`).
**This is the key efficiency: never write a dequantized fp16 cache; move ~2.4×
fewer KV bytes than fp16.**

### 2.5 Function constants & mixed K8/V4 specialization
Function constants (`sdpa_vector.h:181`): `has_affine_bias=27`, `quant_mode=28`,
`quant_bits=29` (key), `quant_group_size=30` (key), `quant_q_seq_len=31`,
`value_quant_bits=32`, `value_quant_group_size=33`. Value bits/group are appended
only when `mixed = (key_group≠value_group || key_bits≠value_bits)` (`sdpa.cpp:703`).
A `MIXED_QUANT_SDPA_DISPATCH` macro enumerates exactly Affine key=8 ×
key_group∈{32,64} × value_bits∈{4,3,2} × value_group∈{32,64} (`sdpa_vector.h:1271`)
— K8/V4 resolves here. Each combination is a **distinct compiled pipeline**
(hash-named, `sdpa.cpp:780`).

### 2.6 Multi-query (the speculative verifier) + GQA
`q_seq>1` (verifier batches, ≤32) is mapped into grid.z: `block_idx = tid.z/q_seq`,
`q_seq_idx = tid.z%q_seq` (`sdpa_vector.h:800`). Per-q-row causal mask:
`use_key = i <= N - q_seq_len + q_seq_idx` (`:916`). Pure single-token decode takes
a tight mask-free loop (`all_keys_visible`, `:871`). GQA:
`q_head = gqa_factor·kv_head + gqa_offset` (`:798`) — gqa_factor query heads sharing
one KV head are stacked in the threadgroup y-dim so they reuse the same dequantized
K/V from L2 **without materializing repeated KV tensors**.

### 2.7 Fallback gates
`use_fallback` (`sdpa.cpp:1012`) drops to `quantized_matmul` attention when:
training/CPU; dtype ∉ {f32,f16,bf16}; **q_seq > 32**; q_seq > N; head_dim ∉
{64,128,256,512}; or gqa_factor > 32. (q_seq≤32 is why a speculative verifier of up
to 32 draft tokens stays on the fused kernel.)

---

## 3. The diagnostic kernel — PolarQJL JIT (and why it doesn't ship)

File: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/turbo_quant_attention_jit.h`
(cited `jit.h`), assembled by `mlx-swift/Source/MLX/TurboQuant.swift`.

### 3.1 Two-kernel block-parallel decode
Same flash-decoding shape as §2 but hand-rolled. **Pass 1**
(`turbo_quant_gqa_block_partials_source`, `jit.h:5185`): one threadgroup per
(gqa_row, 512-token block); the query is staged to threadgroup memory and **rotated
once per (repeat, group)** and reused across all keys (the QJL rotation seed is
token-independent — turns O(N) query rotations into O(repeats·groups), `:5253`);
each lane decodes keys on the fly, computes QK (codebook level + QJL sign-residual),
runs a block-local online softmax, decodes values on the fly, accumulates per-block
AV, and lane0 emits `[max,sum]` stats. **Pass 2** (`turbo_quant_block_reduce_source`,
`:5603`) merges blocks with the standard flash rescale.

### 3.2 In-kernel codec decode
`tq_decode_attention_value` (`jit.h:651`): **role=1 affine** branch is `x = min +
q·scale` (`:691`). **role=0 QJL/WHT** branch decodes a magnitude code → unit-Gaussian
level `tq_codebook_unit(bits,code)·rsqrt(count)` (`:746`) → inverse product rotation
`tq_apply_product_rotation` (Walsh-Hadamard butterfly for power-of-two sizes, Givens
passes otherwise) → ×per-group norm. The QK score uses the **unbiased estimator**
`score = norm·Σ(q·level) + residual_norm·sqrt(π/(2·count))·Σ(sign·q)`
(`tq_product_attention_inner_product_group_quad`, `:1058`).

### 3.3 TQCOOP — cooperative coalesced decode (opt-in, default off)
The default `lane==token` mapping over a token-major layout is **uncoalesced (~32
cache-lines per SIMD load)**. The TQCOOP path (`jit.h:5287`, `LANES_PER_TOKEN==4`):
4 lanes cooperate on **one** key, each decoding a contiguous `HEAD_DIM/4` chunk so
the quad's chunks tile one cache line (~4× less L1 pressure); a quad walks 4 tokens
in 4 passes; cross-quad reduce via two `simd_shuffle_xor`; the back-half is
unchanged. Gated by `turboQuantCooperativeQuadDecodeActive` (`TurboQuant.swift:136`):
`TQ_COOP=1`, exactly 4 GQA repeats, **logicalLength ≥ 32768**, uniform/split codec,
head_dim%4==0. Dispatched under a **distinct kernel name**
`fusedAttentionGQABlockPartialsCoop` so MLX's compiled-variant cache never aliases
LANES==4 vs LANES==1. Measured M2 Pro @131K: turbo8 +~20%, turbo4v2 +60%, turbo3_5
+30–55%. **Ships default-off** (a shipped-iOS env var is no field kill switch;
A-series-unvalidated; regresses at short context) — and it only benefits the
diagnostic path, so its product value is currently nil.

---

## 4. Quantization math (the codecs an auditor must check)

### 4.1 Affine K8/V4 (production)
Classic asymmetric min/max affine per group (`encodeTurboQuantAffineValueReference`,
`TurboQuant.swift:7805`): `scale = (max−min)/(2^bits−1)`, `zero = min`,
`q = clamp(round((x−min)/scale), 0, 2^bits−1)`; decode `x = zero + q·scale`.
**Asymmetric precision by design:** keys 8-bit group64, values 4-bit group32 —
because key error enters the **softmax exponent** (sensitive → high precision) while
value error is **averaged** by the softmax-weighted sum (tolerant → 4-bit). (Field
reuse gotcha: `highScales` stores the *zero-point* for `.affineValue`.)

**Byte budget (head_dim 128):** FP16 = 256 B/K + 256 B/V = 512 B. affineK8V4 K =
128 B payload + 8 B meta = 136 B; V = 64 B payload + 16 B meta = 80 B; **K+V = 216 B
= 2.37× nominal** (2.67× payload-only; metadata erodes it). **Measured resident =
2.13×** (resident also carries raw warm windows + the quantizedKVStart prefix +
decode buffers — do not conflate nominal vs resident). affineInt4 = 160 B = 3.20×.

### 4.2 PolarQuant/QJL (the paper, diagnostic) — `encodeTurboQuantProductReference` (`:7914`)
Three stages implementing arXiv 2504.19874's **unbiased inner-product quantizer**:
- **Preconditioning:** normalize the group, then apply a **random orthogonal
  rotation** (Randomized Hadamard: random sign flip → Walsh-Hadamard butterfly →
  ×1/√n). After rotation every coordinate is ~**N(0, 1/n)** — the paper's key fact
  that makes a *data-free* codebook optimal.
- **MSE quantizer:** map each rotated coordinate to the nearest **Lloyd-Max
  centroid** for the unit Gaussian (fixed tables, e.g. 1-bit ±√(2/π)=±0.7979).
- **1-bit QJL residual:** store only the **sign** of the residual + one residual
  norm/group. The estimator
  `score = Σ norm·⟨quantizedRotated, queryRotated⟩ + residualNorm·sqrt(π/(2·count))·⟨signs, queryRotated⟩`
  (`:8092`). The `sqrt(π/(2n))` factor makes the sign-sketch an **unbiased** estimate
  of the residual's contribution to ⟨K,Q⟩ (since `E|r_i| = residualNorm·√(2/π)/√n`),
  i.e. the softmax score stays unbiased despite discarding residual magnitude.
- **Pre-rotated-Q identity:** `⟨Q, dequant(K_i)⟩ = (norm_i/D)·⟨rawWHT(signs·Q),
  centroid[K_i]⟩`, so Q is rotated once and reused for all keys.

### 4.3 PolarWHT-V pull-out (diagnostic; not portable to affine)
Because WHT is linear, `Σ_i w_i·WHT(c_i) = WHT(Σ_i w_i·c_i)` — the V matvec
accumulates weighted centroids in centroid space and runs **one** final butterfly
instead of one per token (`turboQuantPolarWHTReferenceAccumulate`, `:2608`). 3-bit
dim=128 = 13 uint32 words + 1 fp32 norm = 56 B vs 256 B = **4.57×** (the paper's
4.6×). **This trick is codec-specific** — affine values decode as `zero + q·scale`
with no WHT to pull out, so the production V path cannot use it.

### 4.4 N4 — data-free Gaussian Lloyd-Max payload quantizer (landed, mlx-swift `dc6b9bd`/`4a83f63`)
`TurboQuantReferenceFormat.gaussianLloydMax` (`TurboQuant.swift:244`).
`gaussianLloydMaxCentroids(bits:)` (`:7599`) builds the optimal scalar quantizer for
a unit Gaussian with **zero data**: lay the N(0,1) density `w=exp(−x²/2)` on an
8192-point grid over [−6,6], init 2^bits centroids uniformly, run **80 Lloyd
iterations** (assign-to-nearest, density-weighted centroid update
`c_k = Σ x_i·w_i / Σ w_i` over its cell). Encode (`:7647`): per group of 64, store
**one** RMS norm `σ = √(mean(x²))`, normalize `x/σ`, pick nearest centroid, bit-pack
the index — **no sign/mask/residual bitsets** (the "diet"). Decode: `x = c_idx·σ`.

**The diet measurement (the important correction, `TurboQuantCodecDietTests.swift`):**
the PolarQJL reference codec at 8.0 bits/value (2.0× vs fp16, cosine 0.998762) spends
**~81% of bytes on payload, ~19% on metadata**. So **the paper's 4–7× lives in the
PAYLOAD quantizer, not the scale metadata** (the earlier roadmap had this backwards).
Measured equal-quality: the data-free Gaussian quantizer matches the codec's cosine
at **~5 bits (3.2×)** vs the codec's 8 bits (2.0×) — the codec over-codes ~3
bits/value; 3-bit reaches **~5.33× at cosine 0.982** (paper range). fp16 scales are
quality-free (cosine Δ −6e-8). **Status: CPU-codec validated; no Metal `.gaussianLloydMax`
decode kernel yet, and it is a diagnostic-path lever (the affine production path
doesn't use it).**

### 4.5 Quality gates
Codec-level (`TurboQuant.swift:8460`): relative MSE, cosine, and an
**inner-product-relative-error** using a deterministic probe (because the inner
product is what actually drives softmax scores). Default thresholds: relMSE ≤ 0.02,
cosine ≥ 0.99, IP-rel-error ≤ 0.08. Model-level (mlx-swift-lm artifacts): top-1
agreement, KL of next-token distribution, p95 max-logit abs error.

---

## 5. Decode-time algorithms

### 5.1 Lever ① — no-draft prompt-lookup speculation (bit-exact)
**Proposer** (`PromptLookupSpeculator.swift`, pure CPU): maintains the running token
stream + an inverted index `hash(window) → start positions`. `propose()` takes the
trailing `ngram`-window, finds the **most-recent earlier occurrence** (reverse scan
of the hash bucket, `windowEquals` guards collisions), and returns up to
`maxProposalTokens` tokens that **followed** it — capturing verbatim repetition
(code/JSON/quotes/edits). Hash is FNV-1a (`:99`). `truncate(to:)` (`:63`) exactly
reverses appends for N7 rollback (the index is append-ordered).

**Verify round** (`NgramSpeculativeTokenIterator.speculateRound`): build
`[seed]+draft`, run **one** multi-query forward (`q_seq = draft+1`) over the shared
cache, **argmax all positions in one sync**, accept the matching prefix + one bonus
token, and **`trimPromptCache(numTokens: draft−accepted)`** to roll back rejects.
**Bit-exactness:** the verify recomputes the true greedy argmax at every position, so
a draft token survives only if it equals that argmax and one bonus argmax is always
emitted — the emitted stream is **byte-identical to plain greedy regardless of draft
quality** (validated PASS on all Qwen3-4B runs). One forward reads the weight stream
once and emits `accepted+1` tokens.

**Self-disable EMA:** `acceptanceEMA = 0.2·ratio + 0.8·EMA`; the proposal gate opens
only while `EMA ≥ 0.5` (after an 8-round warmup) → free-form text degrades gracefully
to single-token decode (no regression). **Fail-closed:** speculation engages **only**
for exact greedy (`temperature==0 && processor==nil`); temp>0 / active-processor fall
back to exact decode (the old temp>0 residual-sampling branch was found
distribution-wrong and removed).

### 5.2 N7 — optimistic-prefetch async pipeline (the top speed lever, default-off)
Plain `TokenIterator` is `asyncEval`-pipelined (~27 ms/tok @8K, *below* one
synchronous forward of ~35 ms); a synchronous spec round drains the GPU each round.
N7 (`prefetchRound`) overlaps: it **optimistically pre-issues round R+1 assuming R
fully accepts**, using R's argmax-at-last kept **lazy** (no CPU readback) as R+1's
seed and `propose().dropFirst()` as R+1's draft, `asyncEval`s it, **then** reads R's
argmax (GPU busy on R+1). On full accept: keep both. On misprediction: **roll back**
R+1's KV appends (`trimPromptCache(next.draft+1)`) + truncate the speculator + commit
R synchronously + trim R's rejects. Recurrent caches force the synchronous path.
**Measured (Qwen3-4B, byte-identical PASS):** 16K long-doc **1.43×→1.76×**, long-code
1.11→1.36×; 8K regression nearly erased (0.80→0.93–1.02×); short still ~0.98×. A
**long-context-only** lever — the forward q_seq scaling (k=4 ≈ 2.7× k=1, MLP +
152K-vocab lm_head bound) hard-caps ① at ~1.8×; KV-byte cuts move none of it.

### 5.3 Routing / admission (N2)
`makeGenerationIterator` (`Evaluate.swift:2814`) routes to the speculative iterator
**only** when `selfSpeculationMode==.promptLookup` AND prompt size ≥
`selfSpeculationMinPromptTokens` (default 8192; Pines wires 12288) AND
`canTrimPromptCache` (the hybrid `MambaCache` is non-trimmable → exact fallback,
which also sidesteps the untested recurrent-rollback path). The iterator calls the
same `resolvedGenerationParameters` as the plain iterator (determinism prerequisite).
Ships **default-off**.

### 5.4 The roofline (why ① is the lever)
At 32K, affine moves **1.60× fewer total bytes** than FP16 (1591 vs 2548 MB) yet
decodes at **0.72×** → realizes ~**0.45× FP16-SDPA bandwidth**; the lost ~2.2× is
dequant compute + scattered per-group metadata loads + low batch=1 occupancy.
Measured effective-BW ratio rises with context (0.39 @16K → 0.62 @65K). Multi-query
amortization (ms/qtok, qL1→qL4): 16K 1.24→0.66, 32K 1.52→0.59. So KV-byte cuts are
**capacity-only** for speed; weight-stream amortization (speculation) is the only
real-model tok/s lever ≤47K.

### 5.5 Other levers
- **N3 banked lm_head:** the tied head is ~30% of weight *bytes* but only ~6% of the
  ~22 ms batch=1 *forward latency* (decode is occupancy-bound). A K=8192 candidate
  subset is 2.6–4.1× faster in isolation but ~3–5% on plain decode → **build only
  coupled with ①** (the head is paid per verify position) + an argmax-containment
  exactness guard.
- **N5 recency-tiering:** protected-edge affine configs keep recent/edge tokens at
  higher precision, bulk compressed. *Wide* edge protection (edge6) recovers V3-bulk
  to cosine 0.9988 (near flat V4 0.9994, above flat V3 0.9973). Capacity/quality,
  modest, **not speed**.
- **N6 recurrent-sync fold:** collapses the duplicated per-token
  `eval+synchronize+clearCache` on the Qwen3.5 hybrid into one
  (`materializeRecurrentKVCacheState(synchronize:false)`); byte-identical, no % claim.

---

## 6. Measured evidence (artifacts under `mlx-swift-lm/artifacts/`)

| metric | value | source |
| --- | --- | --- |
| affine K8/V4 16K real-model | 0.752× FP16 (54.70 vs 72.71 tok/s), KV 2.13×, quality PASS (top-1 1.000, KL 1.7e-6, cos 0.9931), 6/6 KV layers native, no fallback | `turboquant-roofline-20260607/affine-16k-combined.json` |
| affine effective bandwidth | 0.39–0.62× FP16-SDPA (rises with ctx) | `turboquant-roofline-20260607/` |
| multi-query amortization | ~1.8–2.6× (ms/qtok qL1→qL4) | same |
| ① acceptance (Qwen3-0.6B, upper bound) | tok/forward: quote 8.83, doc-edit 4.0–4.4, code 2.2–2.9, json 2.0–2.4, prose 1.02–1.09 | `turboquant-acceptance-20260607/` |
| ① real speedup (Qwen3-4B, sync) | 16K long-doc 1.43×, long-code 1.11×; <12–16K regresses 0.80–0.96×; determinism PASS | `turboquant-acceptance-4b-20260607/N1-summary.md` |
| ① + N7 prefetch (Qwen3-4B) | 16K long-doc **1.76×**, long-code 1.36×; byte-identical PASS | `.../N7-summary.md` |
| N4 Gaussian codec (CPU) | 5-bit 3.2× @ cosine 0.9987 (≈codec quality), 3-bit 5.33× @ 0.982 | mlx-swift `TurboQuantCodecDietTests` |

---

## 7. Design rationale (the key decisions)

1. **Asymmetric K/V precision** (K8/V4): error placement, not symmetry — keys feed
   the exponent, values are averaged.
2. **Affine ships, PolarQJL is diagnostic:** affine reuses Apple's tuned SDPA and is
   already coalesced/split-maxed; PolarQJL is per-bit superior but its rotation +
   codebook decode is memory-access-bound on Apple GPUs.
3. **Speculation over KV-byte-cuts for speed:** decode is weight-dominated and the
   affine kernel is occupancy-bound, so only token-amortization moves tok/s.
4. **No-draft, bit-exact speculation:** verification by greedy argmax makes proposal
   quality a pure throughput knob; no second model, no training, no extra resident
   memory, no correctness risk.
5. **Fail-closed everywhere:** unsupported shapes → decoded fallback with
   diagnostics; temp>0/processor → exact decode; non-trimmable hybrid → exact decode.

---

## 8. The Qwen3.5-2B hybrid caveat
The *product* model (Qwen3.5-2B-4bit) is a **hybrid GatedDeltaNet** — only **6 of 24
layers** are full-attention (the rest are recurrent `MambaCache` with constant-size
state and no growing KV). So (a) affine KV compression touches only ¼ of layers
(decode is even more weight-dominated there), (b) speculation falls back to exact
decode on it (non-trimmable cache — the recurrent rollback is **untested**), and (c)
optimization work is done on **standard-attention** models (Qwen3-0.6B/4B) and must
be validated to transfer.

---

## 9. What an external reviewer should scrutinize (open questions / risks)

1. **The bit-exactness of ①.** Verify the argument in §5.1: does the greedy-argmax
   verify truly reproduce plain greedy for *every* accept/reject pattern, including
   the bonus token and the `trimPromptCache` boundary? (We validated byte-identical
   output empirically on Qwen3-4B; an independent proof/fuzz is welcome.)
2. **N7 misprediction rollback.** `prefetchRound` mutates KV + speculator + iterator
   state speculatively. Audit that **every** discard path fully restores state (KV
   trim count `next.draft+1` + R's rejects; speculator `truncate(to: specMark)`;
   `pendingTokens`/`acceptanceEMA`/`y`). A **forced-misprediction determinism test**
   is specced but the recurrent-cache rollback remains untested.
3. **The affine bandwidth gap.** Is 0.39–0.62× of FP16-SDPA bandwidth inherent to
   quantized vector decode at batch=1, or is there a real kernel win (metadata
   co-location, occupancy)? We argue it's largely inherent + Amdahl-bounded; a
   roofline counter (we lack one) would settle it.
4. **N4 → production.** The data-free Gaussian quantizer is the proven path to the
   paper's 4–7×, but it (a) needs a Metal `.gaussianLloydMax` decode kernel and (b)
   only helps the diagnostic PolarQJL path unless the *production V codec* is moved
   from affine to data-free Gaussian — which changes the value distribution
   assumption (post-rotation N(0,1) vs raw affine) and needs a real-model KL gate.
5. **Promotion-grade evidence is missing.** All numbers are M2 Pro single-sample
   synthetic. Per the project's own checklist, promotion needs randomized repeats,
   a **same-machine upstream baseline** (`arozanov/turboquant-mlx` @ `6e928d7`), and
   **on-device A-series** evidence. None exist yet.
6. **The hard ceiling on ①.** Forward `q_seq` scaling caps ① at ~1.8× and it's
   long-context-gated (≥12–16K). Is there a structurally different way to amortize
   the weight stream (e.g. genuinely parallel multi-token decode, batching) that
   isn't bounded by the per-q_seq forward cost?

---

## 10. File index (for verification)
- Routing/cache: `mlx-swift-lm/Libraries/MLXLMCommon/{KVCache,TurboQuantKVCache,AttentionUtils,TurboQuantAdmission,Evaluate}.swift`
- Speculation: `mlx-swift-lm/Libraries/MLXLMCommon/{PromptLookupSpeculator,NgramSpeculativeTokenIterator}.swift`
- Production kernel: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h`, `.../scaled_dot_product_attention.cpp`
- Diagnostic kernel + codecs: `mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/turbo_quant_attention_jit.h`, `mlx-swift/Source/MLX/TurboQuant.swift`, `mlx-swift/Tests/MLXTests/TurboQuantCodecDietTests.swift`
- Companion docs: `inference-speed-roadmap-2026-06-07.md`, `overhaul-plan-2026-06-07.md`, `continuation-handoff-2026-06-07-overhaul.md`
