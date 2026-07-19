#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORKS_ROOT="$(cd "$LM_ROOT/.." && pwd)"
MLX_SWIFT_ROOT="${MLX_SWIFT_ROOT:-$FORKS_ROOT/mlx-swift}"
RUN_ID="${TQ_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${TQ_BENCH_ARTIFACT_ROOT:-$FORKS_ROOT/artifacts/turboquant-current-$RUN_ID}"

CORE_CONTEXTS="${TQ_CORE_CONTEXTS:-8192,16384,32768,65536,131072}"
CORE_PRESETS="${TQ_CORE_PRESETS:-turbo4v2,turbo3_5,turbo8}"
CORE_PATHS="${TQ_CORE_PATHS:-auto,affine-k8v4-native,affine-k8vx-native,online-fused,tiled-online-fused,two-stage}"
CORE_ITERATIONS="${TQ_CORE_ITERATIONS:-12}"
CORE_WARMUP="${TQ_CORE_WARMUP:-3}"
CORE_LAYOUT_VERSION="${TQ_CORE_LAYOUT_VERSION:-}"
CORE_ENABLE_LAYOUT_V5="${TQ_CORE_ENABLE_LAYOUT_V5:-0}"
CORE_SCALE_STORAGE="${TQ_CORE_SCALE_STORAGE:-}"
CORE_BLOCK_TOKENS="${TQ_CORE_BLOCK_TOKENS:-}"

REAL_CONTEXTS="${TQ_REAL_CONTEXTS:-4096,8192,16384,32768}"
REAL_CONFIGS="${TQ_REAL_CONFIGS:-all}"
REAL_GENERATE_TOKENS="${TQ_REAL_GENERATE_TOKENS:-16}"
REAL_THROUGHPUT_REPEATS="${TQ_REAL_THROUGHPUT_REPEATS:-3}"
REAL_RANDOMIZE_ORDER="${TQ_REAL_RANDOMIZE_ORDER:-1}"
REAL_THROUGHPUT_COOLDOWN="${TQ_REAL_THROUGHPUT_COOLDOWN:-0.25}"
REAL_QUALITY_COOLDOWN="${TQ_REAL_QUALITY_COOLDOWN:-0.5}"
REAL_CONFIG_GROUPS="${TQ_REAL_CONFIG_GROUPS:-auto}"
REAL_SPARSE_CONTEXTS="${TQ_REAL_SPARSE_CONTEXTS:-4096,8192}"
REAL_INCLUDE_SPARSE_LONG="${TQ_REAL_INCLUDE_SPARSE_LONG:-0}"
REAL_QUALITY_GROUP="${TQ_REAL_QUALITY_GROUP:-affine}"
QUALITY_CONTEXTS="${TQ_QUALITY_CONTEXTS:-32768}"

REAL_AFFINE_CONFIGS="fp16,affineK8V4,affineK8V3,affineK8V2,affineK8V3-protectedK8V4,affineK8V3-protectedK8V4-edge4,affineK8V3-protectedK8V4-edge5,affineK8V3-optimized,affineK8V3-last2,affineK8V3-first1-last2,affineK8V3-first1-penultimate,affineK8V3-optimized-vgs64,affineK8V3-protectedK8V4-edge6,affineK8V2-protectedK8V4,affineK8V2-protectedK8V4-edge4,affineK8V2-protectedK8V4-edge6,affineK8V2-protectedK8V4-edge8,affineK8V2-calibrated,affineK8V2-residual-r1,affineK8V2-calibrated-residual-r1,affineK8V3-protectedRaw,affineK8V2-protectedRaw,affineK8V2-protectedRaw-edge4,mlxAffine-q8,affineInt4"
REAL_DENSE_CONFIGS="fp16,turbo3_5,turbo4v2,turbo8"
REAL_SPARSE_CONFIGS="fp16,turbo4v2SparseThreshold1e-4,turbo4v2SparseThreshold5e-5,turbo4v2SparseThreshold1e-5,turbo4v2SparseTopK128,turbo4v2SparseTopK256,turbo4v2SparseTopK512,turbo4v2SparseMass990,turbo4v2SparseMass995,turbo4v2SparseMass999,turbo4v2SparseHybrid995TopK128,turbo4v2SparseHybrid995TopK256,turbo4v2SparseHybrid995TopK512"

SKIP_CORE="${TQ_SKIP_CORE:-0}"
SKIP_REAL_MODEL="${TQ_SKIP_REAL_MODEL:-0}"
RUN_QUALITY_GATES="${TQ_RUN_QUALITY_GATES:-1}"
STRICT="${TQ_BENCH_STRICT:-0}"

usage() {
  cat <<'USAGE'
Run the currently supported TurboQuant benchmark surfaces and store logs/JSON.

Environment:
  TQ_MODEL_DIR              MLX model directory for real-model inference parity.
  TQ_BENCH_ARTIFACT_ROOT    Output directory. Default: ../artifacts/turboquant-current-<utc>.
  TQ_CORE_CONTEXTS          Core operator contexts. Default: 8192,16384,32768,65536,131072.
  TQ_CORE_PRESETS           Core presets. Default: turbo4v2,turbo3_5,turbo8.
  TQ_CORE_PATHS             Core paths. Default: auto,affine-k8v4-native,affine-k8vx-native,online-fused,tiled-online-fused,two-stage.
  TQ_CORE_LAYOUT_VERSION    Optional core --layout-version override.
  TQ_CORE_ENABLE_LAYOUT_V5  1 to pass --enable-layout-v5 for explicit V5/V6 sweeps. Default: 0.
  TQ_CORE_SCALE_STORAGE     Optional core --scale-storage override, for example float16.
  TQ_CORE_BLOCK_TOKENS      Optional core --block-tokens override.
  TQ_REAL_CONTEXTS          Real-model contexts. Default: 4096,8192,16384,32768.
  TQ_REAL_CONFIGS           Real-model configs. Default: all.
  TQ_REAL_THROUGHPUT_REPEATS
                            Median-select repeated real-model throughput samples. Default: 3.
  TQ_REAL_RANDOMIZE_ORDER   1 to shuffle real-model context/config samples. Default: 1.
  TQ_REAL_THROUGHPUT_COOLDOWN
                            Seconds to cool down after each throughput cell. Default: 0.25.
  TQ_REAL_QUALITY_COOLDOWN  Seconds to cool down after each quality pair. Default: 0.5.
  TQ_REAL_CONFIG_GROUPS     Config groups for TQ_REAL_CONFIGS=all. Default: auto
                            (affine,dense,sparse). Use off to run one process.
  TQ_REAL_SPARSE_CONTEXTS   Sparse group contexts when long sparse is gated. Default: 4096,8192.
  TQ_REAL_INCLUDE_SPARSE_LONG
                            1 to allow sparse group to use all TQ_REAL_CONTEXTS. Default: 0.
                            Use only when Sparse-V is known active at those lengths.
  TQ_REAL_QUALITY_GROUP     Group that receives quality gates in grouped runs. Default: affine.
  TQ_QUALITY_CONTEXTS       Real-model quality contexts. Default: 32768.
  TQ_RUN_QUALITY_GATES      1 to run real-model quality gates. Default: 1.
  TQ_SKIP_CORE              1 to skip mlx-swift core operator JSON.
  TQ_SKIP_REAL_MODEL        1 to skip mlx-swift-lm real-model parity.
  TQ_BENCH_STRICT           1 to exit non-zero if any logged command fails. Default: 0.

Examples:
  TQ_MODEL_DIR=/path/to/Qwen3.5-2B-4bit scripts/run-turboquant-current-benchmarks.sh
  TQ_REAL_CONTEXTS=32768,65536 TQ_REAL_CONFIGS=fp16,affineK8V4,k8v3,k8v2 scripts/run-turboquant-current-benchmarks.sh
USAGE
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT/core" "$ARTIFACT_ROOT/real-model"
SUMMARY="$ARTIFACT_ROOT/summary.md"
CSV="$ARTIFACT_ROOT/results.csv"

split_csv() {
  local raw="$1"
  IFS=',' read -ra values <<< "$raw"
  for value in "${values[@]}"; do
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -n "$value" ]] && printf '%s\n' "$value"
  done
}

csv_contains() {
  local needle="$1"
  local raw="$2"
  local value
  for value in $(split_csv "$raw"); do
    [[ "$value" == "$needle" ]] && return 0
  done
  return 1
}

csv_intersection() {
  local raw="$1"
  local allowed="$2"
  local out=()
  local value
  for value in $(split_csv "$raw"); do
    if csv_contains "$value" "$allowed"; then
      out+=("$value")
    fi
  done
  local IFS=,
  printf '%s' "${out[*]}"
}

real_group_configs() {
  case "$1" in
    affine) printf '%s' "$REAL_AFFINE_CONFIGS" ;;
    dense) printf '%s' "$REAL_DENSE_CONFIGS" ;;
    sparse) printf '%s' "$REAL_SPARSE_CONFIGS" ;;
    *) printf '%s' "$1" ;;
  esac
}

real_group_contexts() {
  if [[ "$1" == "sparse" && "$REAL_INCLUDE_SPARSE_LONG" != "1" ]]; then
    csv_intersection "$REAL_CONTEXTS" "$REAL_SPARSE_CONTEXTS"
  else
    printf '%s' "$REAL_CONTEXTS"
  fi
}

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
    FAILED_COUNT=$((FAILED_COUNT + 1))
    printf 'failed %s (status %s); see %s.err\n' "$label" "$status" "$log"
  fi
  return 0
}

{
  printf '# TurboQuant Current Benchmark Run\n\n'
  printf '%s\n' "- Run ID: \`$RUN_ID\`"
  printf '%s\n' "- Artifact root: \`$ARTIFACT_ROOT\`"
  printf '%s\n' "- mlx-swift root: \`$MLX_SWIFT_ROOT\`"
  printf '%s\n' "- mlx-swift-lm root: \`$LM_ROOT\`"
  printf '%s\n' "- Core contexts: \`$CORE_CONTEXTS\`"
  printf '%s\n' "- Core presets: \`$CORE_PRESETS\`"
  printf '%s\n' "- Core paths: \`$CORE_PATHS\`"
  printf '%s\n' "- Core layout version: \`${CORE_LAYOUT_VERSION:-default}\`"
  printf '%s\n' "- Core enable Layout V5+: \`$CORE_ENABLE_LAYOUT_V5\`"
  printf '%s\n' "- Core scale storage: \`${CORE_SCALE_STORAGE:-default}\`"
  printf '%s\n' "- Core block tokens: \`${CORE_BLOCK_TOKENS:-auto}\`"
  printf '%s\n' "- Real-model contexts: \`$REAL_CONTEXTS\`"
  printf '%s\n' "- Real-model configs: \`$REAL_CONFIGS\`"
  printf '%s\n' "- Real-model throughput repeats: \`$REAL_THROUGHPUT_REPEATS\`"
  printf '%s\n' "- Real-model randomized order: \`$REAL_RANDOMIZE_ORDER\`"
  printf '%s\n' "- Real-model throughput cooldown: \`$REAL_THROUGHPUT_COOLDOWN\` seconds"
  printf '%s\n' "- Real-model quality cooldown: \`$REAL_QUALITY_COOLDOWN\` seconds"
  printf '%s\n' "- Real-model config groups: \`$REAL_CONFIG_GROUPS\`"
  printf '%s\n' "- Real-model sparse contexts: \`$REAL_SPARSE_CONTEXTS\`"
  printf '%s\n' "- Real-model include sparse long contexts: \`$REAL_INCLUDE_SPARSE_LONG\`"
  printf '%s\n' "- Real-model quality group: \`$REAL_QUALITY_GROUP\`"
  printf '%s\n\n' "- Quality contexts: \`$QUALITY_CONTEXTS\`"
} > "$SUMMARY"
printf 'label,status,stdout,stderr\n' > "$CSV"
FAILED_COUNT=0

core_extra_args=()
if [[ -n "$CORE_LAYOUT_VERSION" ]]; then
  core_extra_args+=(--layout-version "$CORE_LAYOUT_VERSION")
fi
if [[ "$CORE_ENABLE_LAYOUT_V5" == "1" ]]; then
  core_extra_args+=(--enable-layout-v5)
fi
if [[ -n "$CORE_SCALE_STORAGE" ]]; then
  core_extra_args+=(--scale-storage "$CORE_SCALE_STORAGE")
fi
if [[ -n "$CORE_BLOCK_TOKENS" ]]; then
  core_extra_args+=(--block-tokens "$CORE_BLOCK_TOKENS")
fi

if [[ "$SKIP_CORE" != "1" ]]; then
  run_logged "build-core-turboquantbenchmark" "$MLX_SWIFT_ROOT" \
    "$ARTIFACT_ROOT/core/build-turboquantbenchmark" \
    swift build --product TurboQuantBenchmark -c release

  for context in $(split_csv "$CORE_CONTEXTS"); do
    for preset in $(split_csv "$CORE_PRESETS"); do
      for path in $(split_csv "$CORE_PATHS"); do
        output="$ARTIFACT_ROOT/core/core-${preset}-${path}-${context}"
        core_cmd=(
          .build/release/TurboQuantBenchmark --json --include-timestamp
          --iterations "$CORE_ITERATIONS" --warmup "$CORE_WARMUP"
          --context "$context" --preset "$preset"
        )
        if [[ "$path" == "auto" ]]; then
          core_cmd+=(--head-dim 256 --query-heads 16 --kv-heads 4 --query-length 1)
        else
          core_cmd+=(--path "$path" --head-dim 256 --query-heads 16 --kv-heads 4 --query-length 1)
        fi
        if [[ "${#core_extra_args[@]}" -gt 0 ]]; then
          core_cmd+=("${core_extra_args[@]}")
        fi
        run_logged "core-${preset}-${path}-${context}" "$MLX_SWIFT_ROOT" "$output" \
          "${core_cmd[@]}"
      done
    done
  done
fi

if [[ "$SKIP_REAL_MODEL" != "1" ]]; then
  if [[ -z "${TQ_MODEL_DIR:-}" ]]; then
    printf 'skipped-real-model,0,,TQ_MODEL_DIR not set\n' >> "$CSV"
    printf '\nReal-model inference parity skipped because `TQ_MODEL_DIR` was not set.\n' >> "$SUMMARY"
  else
    run_logged "build-real-model-parity" "$LM_ROOT" \
      "$ARTIFACT_ROOT/real-model/build-turboquantinferenceparity" \
      swift build --product TurboQuantInferenceParity -c release

    randomize_flag=()
    if [[ "$REAL_RANDOMIZE_ORDER" == "1" ]]; then
      randomize_flag=(--randomize-throughput-order)
    fi

    run_real_model_parity() {
      local label="$1"
      local contexts="$2"
      local configs="$3"
      local output="$4"
      local enable_quality="${5:-0}"
      local quality_flag=()
      local real_cmd

      if [[ "$RUN_QUALITY_GATES" == "1" && "$enable_quality" == "1" ]]; then
        quality_flag=(
          --quality-gates
          --quality-contexts "$QUALITY_CONTEXTS"
          --quality-cooldown "$REAL_QUALITY_COOLDOWN"
        )
      fi

      real_cmd=(
        .build/release/TurboQuantInferenceParity
        --model-dir "$TQ_MODEL_DIR"
        --contexts "$contexts"
        --generate-tokens "$REAL_GENERATE_TOKENS"
        --throughput-repeats "$REAL_THROUGHPUT_REPEATS"
        --throughput-cooldown "$REAL_THROUGHPUT_COOLDOWN"
        --configs "$configs"
        --strict-configs
        --diagnostics-output "$output.json"
        --diagnostics-samples-output "$output.samples.jsonl"
      )
      if [[ "${#randomize_flag[@]}" -gt 0 ]]; then
        real_cmd+=("${randomize_flag[@]}")
      fi
      if [[ "${#quality_flag[@]}" -gt 0 ]]; then
        real_cmd+=("${quality_flag[@]}")
      fi
      run_logged "$label" "$LM_ROOT" "$output" "${real_cmd[@]}"
    }

    real_model_grouped=0
    real_groups=()
    if [[ "$REAL_CONFIGS" == "all" ]]; then
      case "$REAL_CONFIG_GROUPS" in
        auto|default|"")
          real_model_grouped=1
          real_groups=(affine dense sparse)
          ;;
        off|none|0|false)
          ;;
        *)
          real_model_grouped=1
          for group in $(split_csv "$REAL_CONFIG_GROUPS"); do
            real_groups+=("$group")
          done
          ;;
      esac
    fi

    if [[ "$real_model_grouped" == "1" && "${#real_groups[@]}" -gt 0 ]]; then
      {
        printf '\n### Real-model grouped runs\n\n'
        printf 'Grouped runs keep each config family in a separate process and stream samples to JSONL.\n\n'
      } >> "$SUMMARY"
      for group in "${real_groups[@]}"; do
        group_configs="$(real_group_configs "$group")"
        group_contexts="$(real_group_contexts "$group")"
        output="$ARTIFACT_ROOT/real-model/inference-parity-$group"
        if [[ -z "$group_contexts" ]]; then
          printf 'real-model-parity-%s-skipped,0,,no contexts after sparse long-context filter\n' "$group" >> "$CSV"
          printf '%s\n' "- \`$group\`: skipped because no contexts remained after filtering." >> "$SUMMARY"
          continue
        fi
        printf '%s\n' "- \`$group\`: contexts \`$group_contexts\`, diagnostics \`$output.json\`, samples \`$output.samples.jsonl\`." >> "$SUMMARY"
        quality_enabled=0
        if [[ "$group" == "$REAL_QUALITY_GROUP" ]]; then
          quality_enabled=1
        fi
        run_real_model_parity "real-model-parity-$group" "$group_contexts" "$group_configs" "$output" "$quality_enabled"
      done
    else
      run_real_model_parity "real-model-parity" "$REAL_CONTEXTS" "$REAL_CONFIGS" \
        "$ARTIFACT_ROOT/real-model/inference-parity" 1
    fi
  fi
fi

{
  printf '\n## Result Index\n\n'
  printf 'Command results are recorded in `%s`.\n\n' "$CSV"
  printf 'Core operator logs live in `%s/core`.\n\n' "$ARTIFACT_ROOT"
  printf 'Real-model logs live in `%s/real-model`.\n' "$ARTIFACT_ROOT"
} >> "$SUMMARY"

printf 'TurboQuant benchmark artifacts written to %s\n' "$ARTIFACT_ROOT"

if [[ "$FAILED_COUNT" -ne 0 ]]; then
  printf 'TurboQuant benchmark run recorded %s failed command(s).\n' "$FAILED_COUNT" >&2
  if [[ "$STRICT" == "1" ]]; then
    exit 1
  fi
fi
