#!/usr/bin/env bash
# n7-promotion-protocol.sh -- ready-to-run promotion evidence for N7
# (optimistic-prefetch speculative decode). RUN LATER, ON AC POWER.
#
# ================================ WHAT N7 IS ================================
# N7 is the async optimistic-prefetch pipeline in
#   Libraries/MLXLMCommon/NgramSpeculativeTokenIterator.swift
# (commit b030159). It is a HOST-SIDE DECODE SCHEDULER, not a kernel and not a
# KV-compression path: while the CPU reads + accepts speculative round R, the
# GPU is already computing round R+1 (issued on a full-acceptance assumption;
# rolled back on a misprediction). It is bit-exact vs plain greedy decode, opt-in
# (`enablePrefetch` / `GenerateParameters.selfSpeculationPrefetch`), greedy /
# no-processor only, and requires a TRIMMABLE KV cache.
#
# The win is CONTEXT-gated (N1/N7 findings): ~0.8-1.0x at 8K (neutral/slight
# regression), 1.1-1.8x at 16K, crossover ~12-16K. Hard ceiling ~1.8x (bounded by
# q_seq forward-cost scaling; KV-byte cuts move none of it).
#
# ========================= CRITICAL COMPATIBILITY NOTE =====================
# N7 requires `canTrimPromptCache(cache) == true`. The SHIPPING model,
# Qwen3.5-2B-4bit (model_type qwen3_5), is a HYBRID: 18 of its 24 layers are
# `linear_attention` and build a recurrent `MambaCache()` (Qwen35.swift:693),
# which is NOT trimmable (KVCache.swift:3011). Therefore:
#   * `NgramSpeculativeTokenIterator.init` THROWS on Qwen3.5-2B (guard at :195),
#     and the product factory `makeGenerationIterator` (Evaluate.swift:2858)
#     routes it to a plain TokenIterator -> N7 NEVER ENGAGES on the shipping model.
#   * MambaCache has an UNCONSUMED index-rollback scaffold
#     (`supportsSpeculativeRollback`/`beginVerify`/`recordVerifyStep`, KVCache.swift
#     :3026) that no iterator calls yet. Wiring that into N7 is the (unbuilt)
#     prerequisite for N7 on the hybrid.
# CONSEQUENCE: promotable N7 evidence can only be produced on a DENSE-attention
# model. Primary evidence model = Qwen3-4B-4bit (model_type qwen3, 36 dense
# layers, trimmable) -- the model N7 was originally validated on. The shipping
# Qwen3.5-2B is exercised here ONLY as a fail-closed compatibility probe
# (product path must degrade to plain decode, byte-identical, no crash).
#
# ============================== BENCH SURFACE ==============================
# Harness: `.build/release/TurboQuantAcceptanceHarness --validate-speculative`
#   - interleaves PLAIN greedy vs SPECULATIVE per prompt with alternating arm
#     order (`--ab-repeats`), bootstrap 95% CIs on tok/s and the speedup ratio,
#     a byte-identical determinism gate (== quality, since both arms are greedy),
#     acceptance rate + tokens/forward (== ENGAGEMENT proof), thermal capture,
#     peak/steady active memory, and a machine-readable JSON artifact (--output).
#   - `--prefetch` selects N7 (async). Omit it for the synchronous-spec baseline.
#   - `--validate-routing` drives the PRODUCT factory (.off vs .promptLookup) and
#     asserts byte-identity -- the admission/routing determinism gate, and the
#     graceful fail-closed probe for non-trimmable (hybrid) caches.
#   TurboQuantInferenceParity does NOT support speculation; N7 has its own runner.
#
# ============================= PROMOTION CRITERIA =========================
# For the PRIMARY model (Qwen3-4B-4bit), N7 is promotable only when ALL hold in
# the SAME report:
#   [ENGAGE]  acceptanceRate > 0 and tokensPerForward > 1 at the target contexts
#             (proves speculation actually ran + accepted; a fallback would show
#             acceptance ~0 / tok-per-forward ~1).
#   [EXACT]   G3 determinism gate PASS -- every N7 row byteIdentical to plain
#             greedy (this IS the quality gate for a greedy lever).
#   [SPEEDUP] at >= 16K: spec speedup median >= 1.2x AND speedupCi95Lo > 1.0
#             (CI cleanly excludes parity) for at least the long-doc workload.
#   [8K]      the 8K speedup is REPORTED (not hidden). It may be < 1.0x; that is
#             acceptable ONLY because the admission floor
#             (selfSpeculationMinPromptTokens, default 8192) keeps N7 out of the
#             short-context regime. Record it; do not bury it.
#   [THERMAL] no `.serious`/`.critical` thermal state during trials (harness
#             flags such runs NON-PROMOTABLE).
#   [MEMORY]  peak + steady active bytes reported in the artifact.
#   [DEVICE]  for a PRODUCT/device claim, this must also be run on an A-series
#             device (this script runs on the dev Mac; add the on-device cycle).
# For the SHIPPING model (Qwen3.5-2B-4bit): the only claim is FAIL-CLOSED --
# `--validate-routing` PASS (byte-identical, plain decode; N7 correctly inert).
#
# ================================ USAGE ===================================
#   scripts/n7-promotion-protocol.sh
# Runs on AC only (bench-caffeinated.sh refuses on battery). Configure via env:
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORKS_ROOT="$(cd "$LM_ROOT/.." && pwd)"
CAFFEINATE="$SCRIPT_DIR/bench-caffeinated.sh"
BUILDER="$SCRIPT_DIR/turboquant-build-acceptance-prompts.py"

DATE_TAG="${N7_DATE_TAG:-$(date -u +%Y%m%d)}"
ART="${N7_ARTIFACT_ROOT:-$FORKS_ROOT/artifacts/n7-promotion-$DATE_TAG}"

# --- Models ---------------------------------------------------------------
# Primary N7 evidence model (dense, trimmable). This is where the speedup claim
# is made. Second dense point optional (set N7_SECOND_* to add it).
PRIMARY_MODEL="${N7_PRIMARY_MODEL:-/Users/mt/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25}"
PRIMARY_EOS="${N7_PRIMARY_EOS:-151645}"
PRIMARY_TAG="${N7_PRIMARY_TAG:-qwen3-4b-4bit}"
# Shipping model (hybrid, NON-trimmable). Fail-closed compatibility probe only.
SHIPPING_MODEL="${N7_SHIPPING_MODEL:-/Users/mt/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-4bit/snapshots/674aaa7240b91e8012fcad5d791b7dfe5ba90207}"
SHIPPING_EOS="${N7_SHIPPING_EOS:-248044}"
SHIPPING_TAG="${N7_SHIPPING_TAG:-qwen3.5-2b-4bit}"

# --- Bench parameters -----------------------------------------------------
# Contexts span the crossover. 16K is the load-bearing promotion context. 32768
# on a 16 GB Mac is expected to be memory-wall-confounded for Qwen3-4B FP16 KV
# (N1); include it but read its number with the memory ceiling in mind. Drop it
# via N7_LONG_TARGETS on a memory-constrained host.
LONG_TARGETS="${N7_LONG_TARGETS:-8192,16384,32768}"
MAX_TOKENS="${N7_MAX_TOKENS:-256}"     # >= 64 so rounds accumulate (spec needs runway)
NGRAM="${N7_NGRAM:-3}"
MAX_PROPOSAL="${N7_MAX_PROPOSAL:-4}"
AB_REPEATS="${N7_AB_REPEATS:-6}"        # interleaved plain/spec trials, order alternated (even)
BOOTSTRAP="${N7_BOOTSTRAP:-4000}"       # bootstrap resamples for the 95% CIs
CYCLES="${N7_CYCLES:-3}"                # >= 3 campaign-level N7-on/off interleave cycles

# --- Reproducibility pins (echoed into every artifact) --------------------
gitrev() { git -C "$1" rev-parse HEAD 2>/dev/null || echo "unknown"; }
export TQ_COMMIT_MLX="$(gitrev "$FORKS_ROOT/mlx")"
export TQ_COMMIT_MLX_C="$(gitrev "$FORKS_ROOT/mlx-c")"
export TQ_COMMIT_MLX_SWIFT="$(gitrev "$FORKS_ROOT/mlx-swift")"
export TQ_COMMIT_MLX_SWIFT_LM="$(gitrev "$LM_ROOT")"
export TQ_COMMIT_PINES="$(gitrev /Users/mt/Programming/Schtack/pines)"
export TQ_DEVICE="${TQ_DEVICE:-$(scutil --get ComputerName 2>/dev/null || hostname)}"
export TQ_PLATFORM="${TQ_PLATFORM:-macOS}"

BIN="$LM_ROOT/.build/release/TurboQuantAcceptanceHarness"
CAMPAIGN_LOG="$ART/campaign.log"

mkdir -p "$ART"
echo "N7 promotion protocol :: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$CAMPAIGN_LOG"
echo "artifact root: $ART" | tee -a "$CAMPAIGN_LOG"
echo "pins: mlx=$TQ_COMMIT_MLX mlx-swift=$TQ_COMMIT_MLX_SWIFT lm=$TQ_COMMIT_MLX_SWIFT_LM" | tee -a "$CAMPAIGN_LOG"

# ------------------------------------------------------------------ guards
if [ ! -x "$CAFFEINATE" ]; then
  echo "FATAL: missing $CAFFEINATE (the AC/sleep guard wrapper)" | tee -a "$CAMPAIGN_LOG" >&2; exit 2
fi
if ! pmset -g batt | grep -q "AC Power"; then
  echo "REFUSED: not on AC power. N7 is a wall-clock tok/s claim; battery runs are invalid." \
    | tee -a "$CAMPAIGN_LOG" >&2; exit 3
fi

# ------------------------------------------------------------------ build
echo "== build (release) ==" | tee -a "$CAMPAIGN_LOG"
( cd "$LM_ROOT" && swift build -c release --product TurboQuantAcceptanceHarness ) \
  2>&1 | tee -a "$CAMPAIGN_LOG"
[ -x "$BIN" ] || { echo "FATAL: $BIN not built" | tee -a "$CAMPAIGN_LOG" >&2; exit 4; }

# ---------------------------------------------------------- prompt building
# Prompts are PRE-TOKENIZED per model (harness uses IdentityTokenizerLoader), so
# context length == prompt token count. Build with the model's OWN tokenizer.
# Requires `transformers` in the active python. Skips if the files already exist.
build_prompts() {  # $1 model-dir  $2 tag
  local mdir="$1" tag="$2"
  local short="$ART/prompts-short-$tag.json" long="$ART/prompts-long-$tag.json"
  if [ -s "$short" ] && [ -s "$long" ]; then
    echo "prompts present for $tag (reuse)" | tee -a "$CAMPAIGN_LOG"; return 0
  fi
  if ! python3 -c "import transformers" 2>/dev/null; then
    echo "WARN: python3 'transformers' unavailable; cannot build prompts for $tag." \
      | tee -a "$CAMPAIGN_LOG" >&2
    echo "      Build them once with:" | tee -a "$CAMPAIGN_LOG" >&2
    echo "      python3 $BUILDER --model-dir $mdir --short-out $short --long-out $long --long-targets $LONG_TARGETS" \
      | tee -a "$CAMPAIGN_LOG" >&2
    return 1
  fi
  python3 "$BUILDER" --model-dir "$mdir" --short-out "$short" --long-out "$long" \
    --long-targets "$LONG_TARGETS" 2>&1 | tee -a "$CAMPAIGN_LOG"
}

# --------------------------------------------------------- one bench command
# Wrap every harness invocation in the AC/sleep guard so a mid-run sleep VOIDs it.
run_guarded() {  # $1 logfile  (rest) harness args
  local log="$1"; shift
  echo "-- RUN: $BIN $* --> $log" | tee -a "$CAMPAIGN_LOG"
  "$CAFFEINATE" "$log" "$BIN" "$@" 2>&1 | tee -a "$log" | tail -40
}

# ===================================================================
# PRIMARY MODEL: full N7 evidence (dense / trimmable)
# ===================================================================
if build_prompts "$PRIMARY_MODEL" "$PRIMARY_TAG"; then
  P_SHORT="$ART/prompts-short-$PRIMARY_TAG.json"
  P_LONG="$ART/prompts-long-$PRIMARY_TAG.json"

  # (0) Engagement preflight -- cheap, catches a non-engaging model BEFORE the
  #     long matrix. On a trimmable model this shows acceptance > 0.
  run_guarded "$ART/preflight-$PRIMARY_TAG.log" \
    --model-dir "$PRIMARY_MODEL" --prompt-ids-file "$P_LONG" \
    --validate-speculative --prefetch --ab-repeats 1 --max-tokens 48 \
    --ngram "$NGRAM" --max-proposal "$MAX_PROPOSAL" --eos "$PRIMARY_EOS" \
    --output "$ART/preflight-$PRIMARY_TAG.json"

  # (1) N7-on vs N7-off, campaign-interleaved (>= CYCLES cycles) to cancel drift
  #     ACROSS arms. Each internal --ab-repeats already interleaves plain/spec.
  #       arm A (N7-on):  --validate-speculative --prefetch   -> N7 speedup vs plain
  #       arm B (N7-off): --validate-speculative              -> sync-spec vs plain (ablation)
  for c in $(seq 1 "$CYCLES"); do
    for src in "short:$P_SHORT" "long:$P_LONG"; do
      set="${src%%:*}"; pf="${src##*:}"
      run_guarded "$ART/n7on-$PRIMARY_TAG-$set-c$c.log" \
        --model-dir "$PRIMARY_MODEL" --prompt-ids-file "$pf" \
        --validate-speculative --prefetch \
        --ab-repeats "$AB_REPEATS" --bootstrap "$BOOTSTRAP" --max-tokens "$MAX_TOKENS" \
        --ngram "$NGRAM" --max-proposal "$MAX_PROPOSAL" --eos "$PRIMARY_EOS" \
        --output "$ART/n7on-$PRIMARY_TAG-$set-c$c.json"
      run_guarded "$ART/n7off-$PRIMARY_TAG-$set-c$c.log" \
        --model-dir "$PRIMARY_MODEL" --prompt-ids-file "$pf" \
        --validate-speculative \
        --ab-repeats "$AB_REPEATS" --bootstrap "$BOOTSTRAP" --max-tokens "$MAX_TOKENS" \
        --ngram "$NGRAM" --max-proposal "$MAX_PROPOSAL" --eos "$PRIMARY_EOS" \
        --output "$ART/n7off-$PRIMARY_TAG-$set-c$c.json"
    done
  done

  # (2) Product-routing determinism gate (the admission path that promotion turns
  #     on): makeGenerationIterator(.promptLookup, prefetch) must be byte-identical
  #     to plain .off decode. min-prompt floor is forced to 0 inside the gate.
  run_guarded "$ART/routing-$PRIMARY_TAG.log" \
    --model-dir "$PRIMARY_MODEL" --prompt-ids-file "$P_LONG" \
    --validate-routing --prefetch --max-tokens "$MAX_TOKENS" \
    --ngram "$NGRAM" --max-proposal "$MAX_PROPOSAL" --eos "$PRIMARY_EOS"
else
  echo "SKIP primary N7 matrix: prompts unavailable (see WARN above)." | tee -a "$CAMPAIGN_LOG" >&2
fi

# Optional second dense point (e.g. another dense model): set N7_SECOND_MODEL etc.
if [ -n "${N7_SECOND_MODEL:-}" ] && build_prompts "$N7_SECOND_MODEL" "${N7_SECOND_TAG:-second}"; then
  S_LONG="$ART/prompts-long-${N7_SECOND_TAG:-second}.json"
  run_guarded "$ART/n7on-${N7_SECOND_TAG:-second}-long.log" \
    --model-dir "$N7_SECOND_MODEL" --prompt-ids-file "$S_LONG" \
    --validate-speculative --prefetch --ab-repeats "$AB_REPEATS" --bootstrap "$BOOTSTRAP" \
    --max-tokens "$MAX_TOKENS" --ngram "$NGRAM" --max-proposal "$MAX_PROPOSAL" \
    --eos "${N7_SECOND_EOS:-0}" --output "$ART/n7on-${N7_SECOND_TAG:-second}-long.json"
fi

# ===================================================================
# SHIPPING MODEL: fail-closed compatibility probe (hybrid / NON-trimmable)
# ===================================================================
# N7 cannot run here (MambaCache is not trimmable). --validate-speculative would
# THROW by design, so we use the PRODUCT routing gate: makeGenerationIterator must
# degrade .promptLookup to plain decode, byte-identical, no crash. PASS == N7 is
# correctly inert on the shipping model (the only defensible claim for it).
if [ -d "$SHIPPING_MODEL" ] && build_prompts "$SHIPPING_MODEL" "$SHIPPING_TAG"; then
  run_guarded "$ART/failclosed-routing-$SHIPPING_TAG.log" \
    --model-dir "$SHIPPING_MODEL" --prompt-ids-file "$ART/prompts-long-$SHIPPING_TAG.json" \
    --validate-routing --prefetch --max-tokens 64 \
    --ngram "$NGRAM" --max-proposal "$MAX_PROPOSAL" --eos "$SHIPPING_EOS"
  echo "NOTE: on $SHIPPING_TAG, --validate-speculative is expected to THROW" \
    | tee -a "$CAMPAIGN_LOG"
  echo "      (KVCacheError: n-gram speculative decoding requires trimmable KV caches)." \
    | tee -a "$CAMPAIGN_LOG"
fi

# ===================================================================
# GRADE (mechanical; a human reads the CIs and thermal states)
# ===================================================================
echo "== GRADE ==" | tee -a "$CAMPAIGN_LOG"
echo "Check EACH primary N7-on artifact against the promotion criteria:" | tee -a "$CAMPAIGN_LOG"
echo "  jq '.allByteIdentical, (.rows[]|{prompt,promptTokens,byteIdentical,acceptanceRate,tokensPerForward,speedupMedian,speedupCi95Lo,speedupCi95Hi}), .thermalSeriousOrCritical, .peakActiveBytes, .steadyActiveBytes' \\" | tee -a "$CAMPAIGN_LOG"
echo "     $ART/n7on-$PRIMARY_TAG-long-c1.json" | tee -a "$CAMPAIGN_LOG"
echo "PASS requires, for the >=16K rows: byteIdentical==true, acceptanceRate>0," | tee -a "$CAMPAIGN_LOG"
echo "  tokensPerForward>1, speedupMedian>=1.2, speedupCi95Lo>1.0, thermalSeriousOrCritical==false." | tee -a "$CAMPAIGN_LOG"
echo "Record the 8K row's speedup verbatim (admission floor handles any <1.0)." | tee -a "$CAMPAIGN_LOG"
echo "Fail-closed: $ART/failclosed-routing-$SHIPPING_TAG.log must show 'N2 routing determinism gate: PASS'." | tee -a "$CAMPAIGN_LOG"
echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$CAMPAIGN_LOG"
