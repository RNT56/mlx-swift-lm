#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORKS_ROOT="$(cd "$LM_ROOT/.." && pwd)"
RUN_ID="${TQ_V3_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${TQ_V3_ARTIFACT_ROOT:-$FORKS_ROOT/artifacts/turboquant-v3-optimization-$RUN_ID}"

MODEL_DIR="${TQ_MODEL_DIR:-}"
QUALITY_CONTEXTS="${TQ_V3_QUALITY_CONTEXTS:-20000}"
THROUGHPUT_CONTEXTS="${TQ_V3_THROUGHPUT_CONTEXTS:-20000}"
QUALITY_CONFIGS="${TQ_V3_QUALITY_CONFIGS:-affineK8V4,affineK8V3-protectedK8V4-edge4,affineK8V3-optimized,affineK8V3-protectedK8V4-edge6}"
THROUGHPUT_CONFIGS="${TQ_V3_THROUGHPUT_CONFIGS:-affineK8V4,affineK8V3-optimized}"
GENERATE_TOKENS="${TQ_V3_GENERATE_TOKENS:-8}"
REFERENCE_CONFIG="${TQ_V3_REFERENCE_CONFIG:-affineK8V4}"
PREFILL_STEP_SIZE="${TQ_V3_PREFILL_STEP_SIZE:-adaptive}"
QUANTIZED_KV_START="${TQ_V3_QUANTIZED_KV_START:-config-default}"
SPLIT_QUALITY="${TQ_V3_SPLIT_QUALITY:-1}"
RUN_THROUGHPUT="${TQ_V3_RUN_THROUGHPUT:-1}"

SUMMARY="$ARTIFACT_ROOT/v3-optimization.md"
CSV="$ARTIFACT_ROOT/commands.csv"
QUALITY_JSON="$ARTIFACT_ROOT/quality.json"
THROUGHPUT_JSON="$ARTIFACT_ROOT/throughput.json"
QUALITY_LOGITS_ROOT="$ARTIFACT_ROOT/quality-logits"
FAILED=0

usage() {
  cat <<'USAGE'
Run focused TurboQuant V3 protected-edge optimization probes.

Environment:
  TQ_MODEL_DIR                 Required MLX model directory.
  TQ_V3_ARTIFACT_ROOT          Output directory.
  TQ_V3_QUALITY_CONTEXTS       Quality contexts. Default: 20000.
  TQ_V3_THROUGHPUT_CONTEXTS    Throughput contexts. Default: 20000.
  TQ_V3_QUALITY_CONFIGS        Quality configs. Default: K8/V4 plus V3 edge4/optimized/edge6.
  TQ_V3_THROUGHPUT_CONFIGS     Throughput configs. Default: K8/V4 plus optimized V3.
  TQ_V3_GENERATE_TOKENS        Decode tokens for throughput. Default: 8.
  TQ_V3_REFERENCE_CONFIG       Quality reference. Default: affineK8V4.
  TQ_V3_PREFILL_STEP_SIZE      Prefill chunk override. Default: adaptive.
  TQ_V3_QUANTIZED_KV_START     Affine raw-start override. Default: config default.
  TQ_V3_SPLIT_QUALITY          1 to dump/compare quality logits in separate processes. Default: 1.
  TQ_V3_RUN_THROUGHPUT         1 to run throughput rows. Set 0 for quality-only. Default: 1.

Example:
  TQ_MODEL_DIR=/path/to/Qwen3.5-2B-4bit \
  TQ_V3_QUALITY_CONTEXTS=20000,32768 \
  TQ_V3_THROUGHPUT_CONTEXTS=20000,32768 \
  scripts/run-turboquant-v3-optimization.sh
USAGE
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT" "$QUALITY_LOGITS_ROOT"
printf 'label,status,stdout,stderr\n' > "$CSV"

run_logged() {
  local label="$1"
  local cwd="$2"
  local log="$3"
  shift 3
  printf 'running %s\n' "$label"
  (
    cd "$cwd"
    "$@"
  ) >"$log.out" 2>"$log.err"
  local status=$?
  printf '%s,%s,%s,%s\n' "$label" "$status" "$log.out" "$log.err" >> "$CSV"
  if [[ $status -ne 0 ]]; then
    FAILED=1
    printf 'failed %s (status %s); see %s.err\n' "$label" "$status" "$log"
  fi
  return "$status"
}

split_csv() {
  local raw="$1"
  IFS=',' read -ra values <<< "$raw"
  for value in "${values[@]}"; do
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -n "$value" ]] && printf '%s\n' "$value"
  done
}

combine_quality_json() {
  shopt -s nullglob
  local files=("$ARTIFACT_ROOT"/quality-*.json)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    return 0
  fi
  jq -s '
    {
      modelPath: (.[0].modelPath // ""),
      generatedAt: (.[0].generatedAt // ""),
      throughput: [],
      quality: (map(.quality // []) | add)
    }
  ' "${files[@]}" > "$QUALITY_JSON"
}

combine_throughput_json() {
  shopt -s nullglob
  local files=("$ARTIFACT_ROOT"/throughput-*.json)
  shopt -u nullglob
  if [[ ${#files[@]} -eq 0 ]]; then
    return 0
  fi
  jq -s '
    {
      modelPath: (.[0].modelPath // ""),
      generatedAt: (.[0].generatedAt // ""),
      throughput: (map(.throughput // []) | add),
      quality: []
    }
  ' "${files[@]}" > "$THROUGHPUT_JSON"
}

write_summary() {
  {
    printf '# TurboQuant V3 Optimization\n\n'
    printf '%s\n' "- Run ID: \`$RUN_ID\`"
    printf '%s\n' "- Artifact root: \`$ARTIFACT_ROOT\`"
    printf '%s\n' "- Model: \`$MODEL_DIR\`"
    printf '%s\n' "- Quality contexts: \`$QUALITY_CONTEXTS\`"
    printf '%s\n' "- Quality configs: \`$QUALITY_CONFIGS\`"
    printf '%s\n' "- Throughput contexts: \`$THROUGHPUT_CONTEXTS\`"
    printf '%s\n\n' "- Throughput configs: \`$THROUGHPUT_CONFIGS\`"
    printf '%s\n\n' "- Prefill step size: \`$PREFILL_STEP_SIZE\`"
    printf '%s\n\n' "- Quantized KV start: \`$QUANTIZED_KV_START\`"
    printf '%s\n\n' "- Split quality: \`$SPLIT_QUALITY\`"
    printf '%s\n\n' "- Run throughput: \`$RUN_THROUGHPUT\`"

    printf '## Quality Vs %s\n\n' "$REFERENCE_CONFIG"
    printf '| Context | Candidate | Top-1 | KL mean | P95 abs | Cosine | Passed |\n'
    printf '| ---: | --- | ---: | ---: | ---: | ---: | --- |\n'
    if [[ -s "$QUALITY_JSON" ]]; then
      jq -r '
        .quality[] |
        [
          .context,
          .label,
          .deterministicTop1MatchRate,
          .logitKLDivergenceMean,
          .logitMaxAbsErrorP95,
          (.cosine // "n/a"),
          .passed
        ] | @tsv
      ' "$QUALITY_JSON" | while IFS=$'\t' read -r ctx label top1 kl p95 cosine passed; do
        printf '| %s | `%s` | %s | %s | %s | %s | %s |\n' \
          "$ctx" "$label" "$top1" "$kl" "$p95" "$cosine" "$passed"
      done
    else
      printf '| n/a | n/a | n/a | n/a | n/a | n/a | no quality JSON |\n'
    fi

    printf '\n## Throughput\n\n'
    printf '| Context | Config | Decode tok/s | Prefill tok/s | Ratio vs K8/V4 |\n'
    printf '| ---: | --- | ---: | ---: | ---: |\n'
    if [[ -s "$THROUGHPUT_JSON" ]]; then
      jq -r '
        (.throughput | group_by(.context)[]) as $group |
        (($group | map(select(.label == "affineK8V4")) | first | .decodeTokensPerSecond) // null) as $base |
        $group[] |
        [
          .context,
          .label,
          .decodeTokensPerSecond,
          .prefillTokensPerSecond,
          (if ($base // 0) > 0 then (.decodeTokensPerSecond / $base) else null end)
        ] | @tsv
      ' "$THROUGHPUT_JSON" | while IFS=$'\t' read -r ctx label decode prefill ratio; do
        printf '| %s | `%s` | %.4f | %.4f | %s |\n' \
          "$ctx" "$label" "$decode" "$prefill" "${ratio:-n/a}"
      done
    else
      printf '| n/a | n/a | n/a | n/a | no throughput JSON |\n'
    fi

    printf '\n## Artifacts\n\n'
    printf '%s\n' "- Commands: \`$CSV\`"
    printf '%s\n' "- Quality JSON: \`$QUALITY_JSON\`"
    printf '%s\n' "- Quality logits: \`$QUALITY_LOGITS_ROOT\`"
    printf '%s\n' "- Throughput JSON: \`$THROUGHPUT_JSON\`"

    printf '\n## Command Status\n\n'
    printf '| Label | Status | Stdout | Stderr |\n'
    printf '| --- | ---: | --- | --- |\n'
    if [[ -s "$CSV" ]]; then
      tail -n +2 "$CSV" | while IFS=, read -r label status stdout stderr; do
        printf '| `%s` | %s | `%s` | `%s` |\n' "$label" "$status" "$stdout" "$stderr"
      done
    else
      printf '| n/a | n/a | n/a | n/a |\n'
    fi
  } > "$SUMMARY"
}

if [[ -z "$MODEL_DIR" ]]; then
  write_summary
  printf 'TQ_MODEL_DIR is required; wrote summary to %s\n' "$SUMMARY"
  exit 1
fi

if ! run_logged "build-turboquantinferenceparity" "$LM_ROOT" "$ARTIFACT_ROOT/build" \
  swift build --product TurboQuantInferenceParity -c release
then
  write_summary
  printf 'TurboQuant V3 optimization artifacts written to %s\n' "$ARTIFACT_ROOT"
  exit 1
fi

if [[ "$SPLIT_QUALITY" == "1" || "$SPLIT_QUALITY" == "true" || "$SPLIT_QUALITY" == "yes" ]]; then
  ref_safe="$(printf '%s' "$REFERENCE_CONFIG" | tr -c 'A-Za-z0-9_.-' '_')"
  for quality_context in $(split_csv "$QUALITY_CONTEXTS"); do
    ref_logits="$QUALITY_LOGITS_ROOT/${quality_context}-${ref_safe}.json"
    run_logged "quality-logits-${quality_context}-${ref_safe}" "$LM_ROOT" "$QUALITY_LOGITS_ROOT/${quality_context}-${ref_safe}" \
      .build/release/TurboQuantInferenceParity \
      --model-dir "$MODEL_DIR" \
      --quality-logits-output "$ref_logits" \
      --quality-logits-config "$REFERENCE_CONFIG" \
      --quality-logits-context "$quality_context" || true

    for candidate in $(split_csv "$QUALITY_CONFIGS"); do
      if [[ "$candidate" == "$REFERENCE_CONFIG" ]]; then
        continue
      fi
      safe_label="$(printf '%s' "$candidate" | tr -c 'A-Za-z0-9_.-' '_')"
      candidate_logits="$QUALITY_LOGITS_ROOT/${quality_context}-${safe_label}.json"
      run_logged "quality-logits-${quality_context}-${safe_label}" "$LM_ROOT" "$QUALITY_LOGITS_ROOT/${quality_context}-${safe_label}" \
        .build/release/TurboQuantInferenceParity \
        --model-dir "$MODEL_DIR" \
        --quality-logits-output "$candidate_logits" \
        --quality-logits-config "$candidate" \
        --quality-logits-context "$quality_context" || true

      if [[ -s "$ref_logits" && -s "$candidate_logits" ]]; then
        run_logged "quality-compare-${quality_context}-${safe_label}" "$LM_ROOT" "$ARTIFACT_ROOT/quality-compare-${quality_context}-${safe_label}" \
          .build/release/TurboQuantInferenceParity \
          --quality-reference-logits "$ref_logits" \
          --quality-candidate-logits "$candidate_logits" \
          --diagnostics-output "$ARTIFACT_ROOT/quality-${quality_context}-${safe_label}.json" || true
      fi
    done
  done
else
  for candidate in $(split_csv "$QUALITY_CONFIGS"); do
    if [[ "$candidate" == "$REFERENCE_CONFIG" ]]; then
      continue
    fi
    safe_label="$(printf '%s' "$candidate" | tr -c 'A-Za-z0-9_.-' '_')"
    run_logged "quality-$safe_label" "$LM_ROOT" "$ARTIFACT_ROOT/quality-$safe_label" \
      .build/release/TurboQuantInferenceParity \
      --model-dir "$MODEL_DIR" \
      --contexts "$QUALITY_CONTEXTS" \
      --configs "$REFERENCE_CONFIG,$candidate" \
      --quality-gates \
      --quality-contexts "$QUALITY_CONTEXTS" \
      --quality-reference-config "$REFERENCE_CONFIG" \
      --skip-throughput \
      --diagnostics-output "$ARTIFACT_ROOT/quality-$safe_label.json" || true
  done
fi
combine_quality_json

if [[ "$RUN_THROUGHPUT" == "1" || "$RUN_THROUGHPUT" == "true" || "$RUN_THROUGHPUT" == "yes" ]]; then
  for candidate in $(split_csv "$THROUGHPUT_CONFIGS"); do
    safe_label="$(printf '%s' "$candidate" | tr -c 'A-Za-z0-9_.-' '_')"
    run_logged "throughput-$safe_label" "$LM_ROOT" "$ARTIFACT_ROOT/throughput-$safe_label" \
      .build/release/TurboQuantInferenceParity \
      --model-dir "$MODEL_DIR" \
      --contexts "$THROUGHPUT_CONTEXTS" \
      --generate-tokens "$GENERATE_TOKENS" \
      --configs "$candidate" \
      --diagnostics-output "$ARTIFACT_ROOT/throughput-$safe_label.json" || true
  done
  combine_throughput_json
fi

write_summary
printf 'TurboQuant V3 optimization artifacts written to %s\n' "$ARTIFACT_ROOT"
exit "$FAILED"
