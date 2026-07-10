#!/bin/bash
# native-coop-ab-protocol.sh -- validate the WIDEN verdict: the native compressed
# decode kernel's built-in coop branch (kernel kind 4) vs strided (kind 3) on a
# real model, through the engagement-verified harness.
#
# Background (audit 2026-07-10): the native tq_block_partials kernel already
# contains the coop coalesced branch (LANES_PER_TOKEN=4). At the real
# Qwen3-0.6B geometry everything passes the gate except the 32768 context
# floor; TQ_COOP_MIN_CONTEXT (mlx fast.cpp, default 32768) is the measurement
# override. Native coop is default-ON above 32K (TQ_COOP=0 = kill switch).
# Expected ~+14.5% @16K (transfer estimate from the Swift-route A/B).
#
# Promotion rule (CLAUDE.md 2026-07-10): native path, real model, engagement
# verified (nativeKernelKinds [4] vs [3]), AC + caffeinated, gen >= 32.
# NOTE: keep sparse OFF (the widened gate also feeds the sparse block-stats
# call site); the turbo4v2Capacity config is dense.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d)
OUT=../artifacts/native-coop-ab-$STAMP
mkdir -p "$OUT"
LOG=$OUT/protocol.log
BIN=.build/release/TurboQuantInferenceParity
WRAP=scripts/bench-caffeinated.sh
MODEL=/Users/mt/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068

swift build -c release --product TurboQuantInferenceParity 2>&1 | tail -1 | tee -a "$LOG"
shasum -a 256 "$BIN" | tee -a "$LOG"
for r in ../mlx-swift/Source/Cmlx/mlx ../mlx-swift .; do
  echo "$r $(git -C $r rev-parse --short HEAD) dirty=$(git -C $r status --porcelain | wc -l)" | tee -a "$LOG"
done

run() { # name ctx extra-env...
  local name=$1 ctx=$2; shift 2
  env "$@" "$WRAP" "$LOG" "$BIN" \
    --model-dir "$MODEL" \
    --configs fp16,turbo4v2Capacity \
    --contexts "$ctx" \
    --generate-tokens 64 \
    --throughput-repeats 3 \
    --randomize-throughput-order \
    --throughput-cooldown 0.5 \
    --quality-gates --quality-contexts "$ctx" \
    --diagnostics-output "$OUT/$name.json" \
    > "$OUT/$name.stdout" 2>&1
  echo "$name exit=$?" | tee -a "$LOG"
}

# Interleaved rounds: strided (kind 3, default floor) vs coop (kind 4, floor
# lowered to 4096). MLX_TURBOQUANT_NATIVE_ATTENTION default (native ON).
for r in 1 2 3; do
  run "ab16k-r$r-STRIDED" 16384
  sleep 30
  run "ab16k-r$r-COOP" 16384 TQ_COOP_MIN_CONTEXT=4096
  sleep 30
done
# One 8K pair (coop expected smaller/neutral there).
run "ab8k-r1-STRIDED" 8192
sleep 30
run "ab8k-r1-COOP" 8192 TQ_COOP_MIN_CONTEXT=4096
# Zero-code-change design-regime pair at 32K: coop default-ON vs kill switch.
# (Long prefill; watchdogged by the wrapper's sleep-void, cap via timeout.)
run "ab32k-r1-COOPDEFAULT" 32768
sleep 30
run "ab32k-r1-KILLSWITCH" 32768 TQ_COOP=0

echo "GRADE: per arm check nativeKernelKinds ([3] strided vs [4] coop), decode" | tee -a "$LOG"
echo "tok/s, promotionGate, quality passed, no raw/decoded fallback, sparse off." | tee -a "$LOG"
