# TurboQuant Speed-Overhaul Continuation Handoff — 2026-06-07 (overhaul track)

**This is the current pickup point for the inference-speed overhaul.** It supersedes
the speed-direction guidance in the earlier same-day affine-quality handoff
(`continuation-handoff-2026-06-07.md`, which remains valid for the affine quality gate).

## 0. Read order
1. This file.
2. `inference-speed-roadmap-2026-06-07.md` — the measured evidence base (roofline,
   amortization, weight-vs-KV crossover) and why the priorities are what they are.
3. `overhaul-plan-2026-06-07.md` — the lever ladder with per-lever design, gates,
   and the PolarQJL metadata-diet plan (the diagnostic-path focus).
4. Memory: `turboquant-affine-vs-jit-reframe`, `turboquant-kernel-build-wiring`.

## 1. The reframe (one paragraph)
Two decode-kernel families; **only the affine one ships** (`affineK8V4`/`affineInt4`
via MLX-stock `mixed_quant_sdpa_vector_2pass`). The PolarQJL JIT kernel + all TQCOOP
work is **diagnostic-only**. Decode is **weight-dominated** at the product contexts
(≤47K): fixed weights ~937 MB/token vs KV(affine) 164–654 MB. So **the only lever that
moves real-model tok/s while keeping quality + compression is amortizing the weight
stream across emitted tokens — speculative decode** (lever ①). Pure KV-byte cuts are
capacity-only for speed at the affine kernel's measured 0.39–0.62× effective bandwidth
(value g32→g64 measured speed-neutral, quality-worse). Qwen3.5-2B is a **hybrid
GatedDeltaNet** model (only 6/24 layers have KV), so it's the *product* model but a poor
*optimization* target — optimize on a **standard-attention** model (Qwen3-0.6B on disk;
Qwen3-4B/Llama-3.2-3B for representative validation, disk now freed).

## 2. DONE + validated this cycle (all in `mlx-swift-lm`, no kernel/cross-repo edits)

**Measurement infrastructure (P0):**
- `tools/TurboQuantNativeVxBenchmark` — added `--query-lengths`, `--key/value-group-size`,
  achieved-GB/s + ms/qtok roofline (schemaVersion 3). Measured: affine = **0.39–0.62×
  FP16-SDPA effective bandwidth**; multi-query KV scan **amortizes ~1.8–2.6×**;
  attention-isolated affine **beats FP16 at ≥32K**. Artifacts:
  `artifacts/turboquant-roofline-20260607/`.
- Combined affine 16K real-model gate (the prior zero-byte first-task) now valid:
  affineK8V4 **0.752× FP16**, KV 2.13×, native path, quality PASS (cosine 0.9931).
  `artifacts/turboquant-roofline-20260607/affine-16k-combined.json`.

**Lever ① — no-draft n-gram self-speculation (DONE + validated):**
- `Libraries/MLXLMCommon/PromptLookupSpeculator.swift` — n-gram proposer + acceptance
  simulator; `Tests/MLXLMTests/PromptLookupSpeculatorTests.swift` (10 tests, pass).
- `Libraries/MLXLMCommon/NgramSpeculativeTokenIterator.swift` — plain-`LanguageModel`
  speculative decoder: propose → one multi-query verify forward → accept prefix →
  `trimPromptCache` rollback → EMA self-disable. (`resolvedGenerationParameters` made
  internal in `Evaluate.swift` so it resolves identically to the plain iterator.)
- `tools/TurboQuantAcceptanceHarness` — acceptance replay (`--prompt-ids-file`) and
  end-to-end `--validate-speculative` (plain vs speculative, byte-identical assert + tok/s).
- **Result (Qwen3-0.6B, real model):** determinism gate **PASS (byte-identical)**;
  speedup quote **2.20×**, doc-edit **1.68×**, json **1.20×** (acceptance 1.0/0.95/0.76);
  code 0.79×, prose 0.78× (low-acceptance → the inherent speculative overhead on a tiny
  5 ms/token model; vanishes on the weight-dominated 4B/long-ctx target). Acceptance gate
  artifacts: `artifacts/turboquant-acceptance-20260607/`.

## 3. Lever ladder + status
| lever | what | status |
| --- | --- | --- |
| Instrumentation (roofline/amortization) | GB/s + ms/qtok | **DONE** |
| Combined affine 16K gate | promotion artifact | **DONE** |
| ① n-gram self-speculation | weight-stream amortization, bit-exact | **DONE + validated** (needs 4B run + product wiring) |
| ③ banked lm_head | shrink the 30%-of-weights head | **next — gating microbench first** |
| PolarQJL metadata diet | realize paper 4–7× compression | **pending (the focus)** |
| recency-tiering | exact-recent + harder cold tail | pending (capacity/quality) |
| ② recurrent-sync removal | Qwen3.5 product-only | pending |

## 4. Next steps (prioritized — goal · gate · commands · files · acceptance)

### N1. ① on Qwen3-4B / Llama-3.2-3B (disk freed) — show the clean weight-dominated speedup
The 0.6B understates ① (tiny-model fallback overhead). Validate on a standard-attention 3–4B.
- Download: `huggingface-cli download mlx-community/Qwen3-4B-4bit` (or `mlx-community/Llama-3.2-3B-Instruct-4bit`).
- Tokenize prompts (python `transformers`) → `--prompt-ids-file`; run:
  `.build/release/TurboQuantAcceptanceHarness --model-dir <snap> --prompt-ids-file <json> --validate-speculative --max-tokens 256 --ngram 3 --max-proposal 4 --eos <id>`
- **Acceptance:** determinism PASS (byte-identical) + speedup ≥ ~1.0 on prose (no
  regression at scale) and clear wins on structured/long-context. Also sweep at long
  context (16K/32K prompts) where the weight stream dominates most.

### N2. ① product wiring (admission-gate + Pines)
- Add a `GenerateParameters` opt-in (e.g. `selfSpeculationMode = .promptLookup` + ngram/width)
  and route `generate()` to `NgramSpeculativeTokenIterator` when enabled.
- **Admission rule:** enable only in the weight-dominated regime (large model OR long
  context); the EMA floor (default 0.5) is an on-device tuning constant — lower it on
  bigger/longer-context models.
- Wire through Pines `MLXRuntimeBridge` behind a flag; validate output identity in the app.
- **Gate:** forced-rejection determinism test on the hybrid Qwen3.5 (MambaCache rollback
  correctness is UNTESTED — see caveats); byte-identical greedy stream vs non-speculative.

### N3. ③ banked / candidate-subset lm_head — GATING microbench FIRST
- The tied head is ~30% of the weight stream. `gatherQuantizedMatmul` exists (mlx-swift
  `Source/MLX/Ops.swift:1427`).
- **Do first (cheap, kills it if it fails):** microbench `gatherQuantizedMatmul(rhsIndices,
  ~8192 rows, transpose:true, groupSize:64, bits:4, mode:.affine, batch=1)` vs the full
  vocab×hidden qmv head at batch=1 on M2. If not clearly faster at batch=1 → **stop**
  (the occupancy wall that made byte-cuts neutral). Verify the kernel actually ships with
  `LC_ALL=C grep -a -c` (the pin trap).
- If it wins: candidate-subset head + exactness guard (frequent full-vocab pass or a
  bounded mismatch window clearing KL≤2e-6 / top-1==1.000). Near-dependency for ① (verify
  pays the head per draft position).

### N4. PolarQJL metadata diet (THE diagnostic-path focus — realize paper 4–7×)
Root cause of our ~2.1–2.6× (vs paper 4–7×) is metadata, not payload. In priority order
(see `overhaul-plan-2026-06-07.md` §"PolarQJL diagnostic overhaul"):
1. fp32 → fp16 scales/norms (`TurboQuantScaleStorage.float16` exists, `mlx-swift
   Source/MLX/TurboQuant.swift:223`).
2. **One norm per vector, not per group** (the PolarQuant idea we under-use).
3. Collapse dead bitset planes per preset at layout v6 (decode-branch map in
   `turboquant-kernel-build-wiring` memory).
4. Data-free Gaussian Lloyd-Max quantizer → metadata collapses to rotation seed + 1 norm.
- **Quality lever:** proper rotation + unbiased QJL residual lets KEYS go 2.5–3.5 bit at
  affine-K8 quality → compression++.
- **Gates:** encode/decode parity unit tests; real-model KL/p95/cosine at matched bits;
  resident-byte report; same-machine upstream (`arozanov/turboquant-mlx` @ `6e928d7`)
  comparison before any parity claim. Ships as a **compression/quality** win; PolarQJL
  *speed* still needs hardware-aligned rotation (cf. IsoQuant SO(4), arXiv:2603.28430).
- **Pin trap:** `mlx-swift` is `swift package edit`-linked (`Packages/mlx-swift` → local),
  so kernel edits build — but `touch` the edited file and verify the marker is in the
  binary with `LC_ALL=C grep -a -c` (NOT `strings`). Use a DISTINCT kernel name for any
  new variant (compiled-variant cache aliasing).

### N5. Recency-tiering (compression + quality, speed-neutral)
Exact-FP16 last W≈256 tokens + harder-compressed cold tail (V4→V3 / K8→K6). Net
~2.1×→~2.8× effective KV and better quality. Use the existing hybrid cache dequant+concat
merge (`TurboQuantHybridKVCache.swift:1932`). Ship as **capacity/quality** only (extra
dispatch is net overhead in the weight-dominated regime). Gate: KL/p95/cos vs all-FP16.

### N6. ② Redundant recurrent sync (Qwen3.5 product-only)
Collapse the duplicated `eval+synchronize+clearCache` (`KVCache.swift:2864` inner +
`Evaluate.swift:1522` outer) + every-K `clearCache` cadence. Pure cadence change, can't
alter output. Gate: KVCacheTests determinism + timing probe before any % claim. Only
matters on the hybrid (no `MambaCache` on standard models).

### N7. ① refinement — async-pipelined single-token fallback
So the EMA-disabled path matches the plain iterator's throughput on small/fast models
(removes the tiny-model regression). Lower priority (vanishes on the product target).

## 5. Build / test / run
```bash
cd /Users/mt/Programming/Schtack/mlx-forks/mlx-swift-lm
swift test --filter PromptLookupSpeculatorTests           # ① core (10 tests)
swift build -c release --product TurboQuantAcceptanceHarness
swift build -c release --product TurboQuantNativeVxBenchmark
swift build -c release --product TurboQuantInferenceParity
# ① end-to-end validate (determinism + tok/s):
.build/release/TurboQuantAcceptanceHarness --model-dir <snap> \
  --prompt-ids-file <prompts.json> --validate-speculative \
  --max-tokens 256 --ngram 3 --max-proposal 4 --eos <eos_id>
# roofline / amortization:
.build/release/TurboQuantNativeVxBenchmark --contexts 32768,65536 \
  --value-bits 4 --query-lengths 1,4 --query-heads 16 --kv-heads 8 --head-dim 128
```
Tokenize prompts to IDs (in-package tools have no swift-transformers; python has it):
```python
from transformers import AutoTokenizer; import json
tok = AutoTokenizer.from_pretrained("<snapshot dir>")
json.dump([{"label":k,"ids":tok.encode(v)} for k,v in prompts.items()], open("prompts.json","w"))
```

## 6. Models / evidence
- On disk: `Qwen3-0.6B-8bit` (standard-attn dev), `Qwen3.5-2B-4bit` (hybrid product),
  `Qwen3-0.6B`, `bitnet-2B`. **Disk now ~41 GiB free** → pull `Qwen3-4B-4bit` for N1.
- Artifacts: `artifacts/turboquant-roofline-20260607/`, `artifacts/turboquant-acceptance-20260607/`.

## 7. Caveats / non-claims
- ① wins are workload-gated (high on structured/repetitive long-context; ~1.0× on
  free-form via the EMA gate). The 0.6B <1× on low-acceptance is a tiny-model artifact.
- Speculative rollback on the recurrent `MambaCache` (Qwen3.5 hybrid) is **UNTESTED** —
  forced-rejection determinism gate required before ① runs on the hybrid (N2).
- PolarQJL diet is a **compression/quality** win; do not claim PolarQJL *speed* parity
  without the hardware-aligned rotation kernel + same-machine upstream comparison.
- Do not claim 0.98× FP16; do not promote any path without the full promotion checklist
  (same-report quality + native path + no fallback + memory + A-series device evidence).

## 8. Gotchas
- **Build-wiring:** `mlx-swift` is `swift package edit`-linked; kernel edits build but
  need `touch` + `LC_ALL=C grep -a -c` binary verification (NOT `strings`).
- **Kernel-variant aliasing:** any new Metal variant needs its OWN kernel name or MLX's
  compiled-variant cache cross-contaminates.
- **Tokenizer:** in-package tools use `IdentityTokenizer` (token IDs only); tokenize text
  externally with python `transformers`.
- **MLP roofline gap:** the ~507 MB (54%) MLP weight stream at batch=1 vs batch=k is
  unmeasured and bounds ①'s ceiling — worth a roofline.
