#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORKS_ROOT="$(cd "$LM_ROOT/.." && pwd)"
RUN_ID="${TQ_ACCEPTANCE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${TQ_ACCEPTANCE_ARTIFACT_ROOT:-$FORKS_ROOT/artifacts/turboquant-acceptance-$RUN_ID}"

REAL_CONTEXTS="${TQ_ACCEPTANCE_REAL_CONTEXTS:-20000,32768,65536,131072}"
QUALITY_CONTEXTS="${TQ_ACCEPTANCE_QUALITY_CONTEXTS:-$REAL_CONTEXTS}"
REAL_GENERATE_TOKENS="${TQ_ACCEPTANCE_GENERATE_TOKENS:-8}"
REAL_THROUGHPUT_REPEATS="${TQ_ACCEPTANCE_THROUGHPUT_REPEATS:-3}"
REAL_RANDOMIZE_THROUGHPUT="${TQ_ACCEPTANCE_RANDOMIZE_THROUGHPUT:-1}"
REAL_THROUGHPUT_SEED="${TQ_ACCEPTANCE_THROUGHPUT_SEED:-20260602}"
REAL_CONFIGS="${TQ_ACCEPTANCE_REAL_CONFIGS:-affineK8V4,affineK8V3-protectedK8V4-edge5,affineK8V2,affineK8V2-protectedK8V4-edge8,affineK8V2-calibrated,affineK8V2-residual-r1,affineK8V2-calibrated-residual-r1}"
KV_LAYER_POLICY_JSON="${TQ_ACCEPTANCE_KV_LAYER_POLICY_JSON:-}"

PROOF_PROFILES="${TQ_ACCEPTANCE_PROFILES:-qwen3.5-2b}"
PROOF_CONTEXTS="${TQ_ACCEPTANCE_PROOF_CONTEXTS:-32768}"
PROOF_QUERY_LENGTHS="${TQ_ACCEPTANCE_QUERY_LENGTHS:-1}"
PROOF_ITERATIONS="${TQ_ACCEPTANCE_ITERATIONS:-3}"
PROOF_WARMUP="${TQ_ACCEPTANCE_WARMUP:-1}"
SPARSE_THRESHOLD="${TQ_ACCEPTANCE_SPARSE_THRESHOLD:-1e-5}"
SPARSE_TOP_K="${TQ_ACCEPTANCE_SPARSE_TOP_K:-256}"
SPARSE_MASS="${TQ_ACCEPTANCE_SPARSE_MASS:-99.5}"
SPARSE_MASS_FRACTION="$(
  awk -v mass="$SPARSE_MASS" 'BEGIN { if (mass > 1) printf "%.6f", mass / 100; else printf "%.6f", mass }'
)"
SPARSE_MAX_TOP_K="${TQ_ACCEPTANCE_SPARSE_MAX_TOP_K:-256}"
PAGE_TOP_KS="${TQ_ACCEPTANCE_PAGE_TOP_KS:-1,2,4,8}"
PAGE_RECENT_TOKENS="${TQ_ACCEPTANCE_PAGE_RECENT_TOKENS:-0}"
CANDIDATE_SPARSE_ROWS="${TQ_ACCEPTANCE_CANDIDATE_SPARSE_ROWS:-512:2:128}"

RUN_SPARSE="${TQ_ACCEPTANCE_RUN_SPARSE:-0}"
SKIP_REAL_MODEL="${TQ_ACCEPTANCE_SKIP_REAL_MODEL:-0}"
SKIP_PROOF="${TQ_ACCEPTANCE_SKIP_PROOF:-0}"

SUMMARY="$ARTIFACT_ROOT/acceptance-matrix.md"
GATE_JSON="$ARTIFACT_ROOT/polarwht-acceptance-gate.json"
CSV="$ARTIFACT_ROOT/commands.csv"
REAL_DIR="$ARTIFACT_ROOT/real-model"
PROOF_DIR="$ARTIFACT_ROOT/proof"

usage() {
  cat <<'USAGE'
Run the TurboQuant acceptance matrix and write a Markdown report.

The runner compares dense affine K8/V4 against lower-V candidates with
TurboQuantInferenceParity when TQ_MODEL_DIR is set. Sparse-V rows are
explicit diagnostic/proof rows only and run only when TQ_ACCEPTANCE_RUN_SPARSE=1.

Environment:
  TQ_MODEL_DIR                       MLX model directory for real-model parity.
  TQ_ACCEPTANCE_ARTIFACT_ROOT        Output directory.
  TQ_ACCEPTANCE_REAL_CONTEXTS        Real-model contexts. Default: 20000,32768,65536,131072.
  TQ_ACCEPTANCE_QUALITY_CONTEXTS     Quality contexts. Default: real contexts.
  TQ_ACCEPTANCE_GENERATE_TOKENS      Decode tokens per real-model cell. Default: 8.
  TQ_ACCEPTANCE_THROUGHPUT_REPEATS   Median-selected real-model throughput samples. Default: 3.
  TQ_ACCEPTANCE_RANDOMIZE_THROUGHPUT 1 to randomize throughput sample order. Default: 1.
  TQ_ACCEPTANCE_THROUGHPUT_SEED      Deterministic throughput randomization seed. Default: 20260602.
  TQ_ACCEPTANCE_REAL_CONFIGS         Real-model configs. Default: K8/V4, V3 edge5, V2 dense/protected/calibrated/residual.
  TQ_ACCEPTANCE_KV_LAYER_POLICY_JSON Optional calibrated KVLayerPolicy JSON for calibrated rows.
  TQ_ACCEPTANCE_PROFILES             Qwen proof profiles. Default: qwen3.5-2b.
  TQ_ACCEPTANCE_PROOF_CONTEXTS       Qwen proof contexts. Default: 32768.
  TQ_ACCEPTANCE_QUERY_LENGTHS        Qwen proof query lengths. Default: 1.
  TQ_ACCEPTANCE_ITERATIONS           Qwen proof iterations. Default: 3.
  TQ_ACCEPTANCE_WARMUP               Qwen proof warmup iterations. Default: 1.
  TQ_ACCEPTANCE_SPARSE_THRESHOLD     Sparse-V threshold. Default: 1e-5.
  TQ_ACCEPTANCE_SPARSE_TOP_K         Sparse-V top-k. Default: 256.
  TQ_ACCEPTANCE_SPARSE_MASS          Sparse-V cumulative mass percent. Default: 99.5.
  TQ_ACCEPTANCE_SPARSE_MAX_TOP_K     Sparse-V hybrid max top-k. Default: 256.
  TQ_ACCEPTANCE_PAGE_TOP_KS          Sparse-V pageTopK retained pages. Default: 1,2,4,8.
  TQ_ACCEPTANCE_PAGE_RECENT_TOKENS   Optional exact recent-token floor for pageTopK rows. Default: 0.
  TQ_ACCEPTANCE_CANDIDATE_SPARSE_ROWS CandidateSparse triples recent:pages:olderTopK. Default: 512:2:128.
  TQ_ACCEPTANCE_RUN_SPARSE           1 to run rejected Sparse-V diagnostic rows. Default: 0.
  TQ_ACCEPTANCE_SKIP_REAL_MODEL      1 to skip TurboQuantInferenceParity.
  TQ_ACCEPTANCE_SKIP_PROOF           1 to skip TurboQuantQwenProof.

Example:
  TQ_MODEL_DIR=/path/to/mlx-model scripts/run-turboquant-acceptance-matrix.sh
USAGE
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$REAL_DIR" "$PROOF_DIR"
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
    printf 'failed %s (status %s); see %s.err\n' "$label" "$status" "$log"
  fi
  return 0
}

split_csv() {
  tr ',' '\n' <<< "$1" | sed '/^[[:space:]]*$/d'
}

append_real_model_tables() {
  local log="$REAL_DIR/k8vx-vs-dense-k8v4.out"
  if [[ ! -s "$log" ]]; then
    printf 'No real-model output was captured.\n\n' >> "$SUMMARY"
    return
  fi

  printf '### Throughput Vs Dense K8/V4\n\n' >> "$SUMMARY"
  printf '| Context | Config | Decode tok/s | Prefill tok/s | Ratio vs K8/V4 |\n' >> "$SUMMARY"
  printf '| ---: | --- | ---: | ---: | ---: |\n' >> "$SUMMARY"
  awk '
    /^[0-9]+[[:space:]]+affineK8V[234]/ {
      ctx=$1; cfg=$2; decode=$3; prefill=$4
      rows[++n]=ctx "|" cfg "|" decode "|" prefill
      if (cfg == "affineK8V4") base[ctx]=decode
    }
    END {
      for (i=1; i<=n; i++) {
        split(rows[i], f, "|")
        ratio="n/a"
        if (f[1] in base && base[f[1]] > 0) {
          ratio=sprintf("%.3f", f[3] / base[f[1]])
        }
        printf "| %s | `%s` | %s | %s | %s |\n", f[1], f[2], f[3], f[4], ratio
      }
    }
  ' "$log" >> "$SUMMARY"
  printf '\n' >> "$SUMMARY"

  printf '### Quality Vs Dense K8/V4\n\n' >> "$SUMMARY"
  printf '| Context | Candidate | Reference | Top-1 | KL mean | P95 abs | Cosine | Passed |\n' >> "$SUMMARY"
  printf '| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |\n' >> "$SUMMARY"
  awk '
    /^[0-9]+[[:space:]]+affineK8V[23]/ {
      printf "| %s | `%s` | `affineK8V4` | %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5, $6, $7
    }
  ' "$log" >> "$SUMMARY"
  printf '\n' >> "$SUMMARY"
}

append_real_model_sparse_tables() {
  printf '### Real-Model Sparse-V Diagnostics\n\n' >> "$SUMMARY"
  printf '| Run | Context | Config | Decode tok/s | Kernel kinds | Requested layers | Active layers | Skip ratio | Fallback |\n' >> "$SUMMARY"
  printf '| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- |\n' >> "$SUMMARY"

  local found=0
  for json in "$REAL_DIR"/sparse-*.json; do
    [[ -s "$json" ]] || continue
    if ! jq -e '.throughput' "$json" >/dev/null 2>&1; then
      continue
    fi
    found=1
    local run_label
    run_label="$(basename "$json" .json)"
    jq -r --arg run "$run_label" '
      .throughput[] |
      [
        $run,
	        .context,
	        .label,
	        .decodeTokensPerSecond,
	        ((.nativeKernelKinds // []) | join("/")),
	        .sparseRequestedLayerCount,
	        .sparseActiveLayerCount,
	        (.sparseSkipRatio // "n/a"),
	        (.sparseFallbackReason // "")
	      ] |
	      @tsv
	    ' "$json" | while IFS=$'\t' read -r run ctx cfg decode kernel_kinds requested active skip fallback; do
	      printf '| `%s` | %s | `%s` | %s | `%s` | %s | %s | %s | %s |\n' \
	        "$run" "$ctx" "$cfg" "$decode" "${kernel_kinds:-}" "$requested" "$active" "$skip" "${fallback:-}" >> "$SUMMARY"
	    done
	  done

  if [[ $found -eq 0 ]]; then
    printf '| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | No Sparse-V real-model JSON captured. |\n' >> "$SUMMARY"
  fi
  printf '\n' >> "$SUMMARY"
}

append_proof_table() {
  printf '### Sparse-V Proof Rows\n\n' >> "$SUMMARY"
  printf '| Run | Context | Status | Sparse-V mode | Skip ratio | Retained mass | Cosine vs dense | P95 abs vs dense | Fallback |\n' >> "$SUMMARY"
  printf '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |\n' >> "$SUMMARY"

  local found=0
  for json in "$PROOF_DIR"/*.out; do
    [[ -s "$json" ]] || continue
    if ! jq -e '.results' "$json" >/dev/null 2>&1; then
      continue
    fi
    found=1
    local run_label
    run_label="$(basename "$json" .out)"
    jq -r --arg run "$run_label" '
      .results[] |
      [
        $run,
        .benchmarkCase.contextLength,
        .status,
        .sparseVSelectionMode,
        (.sparseVSkipRatio // 0),
        (.sparseVRetainedMass // "n/a"),
        (.sparseVDenseCosineSimilarity // "n/a"),
        (.sparseVDenseMaxAbsErrorP95 // "n/a"),
        (.fallbackReason // "")
      ] |
      @tsv
    ' "$json" | while IFS=$'\t' read -r run ctx status mode skip retained cosine p95 fallback; do
      printf '| `%s` | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$run" "$ctx" "$status" "$mode" "$skip" "$retained" "$cosine" "$p95" "${fallback:-}" >> "$SUMMARY"
    done
  done

  if [[ $found -eq 0 ]]; then
    printf '| n/a | n/a | skipped | n/a | n/a | n/a | n/a | n/a | No proof JSON captured. |\n' >> "$SUMMARY"
  fi
  printf '\n' >> "$SUMMARY"
}

write_header() {
  {
    printf '# TurboQuant Acceptance Matrix\n\n'
    printf '%s\n' "- Run ID: \`$RUN_ID\`"
    printf '%s\n' "- Artifact root: \`$ARTIFACT_ROOT\`"
    printf '%s\n' "- mlx-swift-lm root: \`$LM_ROOT\`"
    printf '%s\n' "- Real-model contexts: \`$REAL_CONTEXTS\`"
    printf '%s\n' "- Real-model throughput repeats: \`$REAL_THROUGHPUT_REPEATS\`"
    printf '%s\n' "- Real-model throughput randomized: \`$REAL_RANDOMIZE_THROUGHPUT\`"
    printf '%s\n' "- Real-model configs: \`$REAL_CONFIGS\`"
    printf '%s\n' "- Sparse diagnostic rows enabled: \`$RUN_SPARSE\`"
    printf '%s\n' "- Sparse pageTopK retained pages: \`$PAGE_TOP_KS\`"
    printf '%s\n' "- Sparse pageTopK recent-token floor: \`$PAGE_RECENT_TOKENS\`"
    printf '%s\n' "- CandidateSparse rows: \`$CANDIDATE_SPARSE_ROWS\`"
    printf '%s\n' "- Quality contexts: \`$QUALITY_CONTEXTS\`"
    printf '%s\n' "- Qwen proof profiles: \`$PROOF_PROFILES\`"
    printf '%s\n' "- Qwen proof contexts: \`$PROOF_CONTEXTS\`"
    printf '%s\n\n' "- Command index: \`$CSV\`"
  } > "$SUMMARY"
}

write_missing_flags() {
  {
    printf '## Blocked CLI Coverage\n\n'
    printf 'The following acceptance rows cannot be run as real-model `TurboQuantInferenceParity` rows with the current CLI surface:\n\n'
    printf '| Variant family | Missing flags |\n'
    printf '| --- | --- |\n'
    printf '| Qwen proof affine K8/Vx protected-boundary override | `--codec affine-k8-v4`, `--codec affine-k8-vx`, `--value-bits`, `--protected-boundary-first`, `--protected-boundary-last`, `--boundary-cache-precision` |\n\n'
    printf 'Current workaround: Qwen proof protected-boundary override rows remain synthetic-only; real-model lower-V rows use `TurboQuantInferenceParity`. Rejected Sparse-V diagnostics run only when `TQ_ACCEPTANCE_RUN_SPARSE=1`.\n\n'
  } >> "$SUMMARY"
}

append_structured_gate() {
  if ! python3 "$SCRIPT_DIR/turboquant-polarwht-acceptance-gate.py" \
    --artifact-root "$ARTIFACT_ROOT" \
    --output "$GATE_JSON"; then
    printf 'failed PolarWHT acceptance gate; see %s\n' "$GATE_JSON" >&2
  fi

  python3 - "$GATE_JSON" "$SUMMARY" <<'PY'
import json
import pathlib
import sys

gate_path = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
with summary.open("a", encoding="utf-8") as handle:
    handle.write("## PolarWHT Structured Gate\n\n")
    if not gate_path.exists():
        handle.write("Structured gate JSON was not produced.\n\n")
        return_code = 0
    else:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        upstream = (gate.get("sameMachineReproduction") or {}).get("upstream") or {}
        local = (gate.get("sameMachineReproduction") or {}).get("local") or {}
        fp16_claim = gate.get("fp16SpeedClaim") or {}
        handle.write(f"- Promotion status: `{gate.get('status')}`\n")
        handle.write(f"- Upstream reproduction: `{upstream.get('status')}`\n")
        handle.write(f"- Local PolarWHT gate: `{local.get('status')}`\n")
        handle.write(f"- 0.98x FP16 claim: `{fp16_claim.get('status')}`\n")
        handle.write(f"- Gate JSON: `{gate_path}`\n")
        if fp16_claim.get("caveat"):
            handle.write(f"- Caveat: {fp16_claim.get('caveat')}\n")
        blockers = gate.get("promotionBlockReasons") or []
        if blockers:
            handle.write("\nGate blockers:\n")
            for blocker in blockers:
                handle.write(f"- {blocker}\n")
        handle.write("\n")
PY
}

write_header

if [[ "$SKIP_REAL_MODEL" != "1" ]]; then
  if [[ -z "${TQ_MODEL_DIR:-}" ]]; then
    printf 'skipped-real-model,0,,TQ_MODEL_DIR not set\n' >> "$CSV"
    printf '## Real-Model K8/Vx Acceptance\n\n' >> "$SUMMARY"
    printf 'Skipped because `TQ_MODEL_DIR` was not set.\n\n' >> "$SUMMARY"
  else
	    run_logged "build-real-model-parity" "$LM_ROOT" "$REAL_DIR/build-turboquantinferenceparity" \
	      swift build --product TurboQuantInferenceParity -c release
	    policy_args=()
	    if [[ -n "$KV_LAYER_POLICY_JSON" ]]; then
	      policy_args=(--kv-layer-policy-json "$KV_LAYER_POLICY_JSON")
	    fi
	    throughput_args=(--throughput-repeats "$REAL_THROUGHPUT_REPEATS" --throughput-seed "$REAL_THROUGHPUT_SEED")
	    if [[ "$REAL_RANDOMIZE_THROUGHPUT" == "1" ]]; then
	      throughput_args+=(--randomize-throughput-order)
	    fi
	    real_model_cmd=(
	      .build/release/TurboQuantInferenceParity
	      --model-dir "$TQ_MODEL_DIR"
	      --contexts "$REAL_CONTEXTS"
	      --generate-tokens "$REAL_GENERATE_TOKENS"
	      "${throughput_args[@]}"
	      --configs "$REAL_CONFIGS"
	      --strict-configs
	      --quality-gates
	      --quality-contexts "$QUALITY_CONTEXTS"
	      --quality-reference-config affineK8V4
	      --quality-candidate-first
	    )
	    if [[ ${#policy_args[@]} -gt 0 ]]; then
	      real_model_cmd+=("${policy_args[@]}")
	    fi
	    real_model_cmd+=(
	      --emit-cache-policy-summary
	      --diagnostics-output "$REAL_DIR/k8vx-vs-dense-k8v4.json"
	    )
	    run_logged "real-model-k8vx-vs-dense-k8v4" "$LM_ROOT" "$REAL_DIR/k8vx-vs-dense-k8v4" \
	      "${real_model_cmd[@]}"

    if [[ "$RUN_SPARSE" == "1" ]]; then
      sparse_base=(
        .build/release/TurboQuantInferenceParity
        --model-dir "$TQ_MODEL_DIR"
        --contexts "$REAL_CONTEXTS"
        --generate-tokens "$REAL_GENERATE_TOKENS"
        "${throughput_args[@]}"
        --configs affineK8V4,turbo4v2
        --strict-configs
        --quality-gates
        --quality-contexts "$QUALITY_CONTEXTS"
        --quality-reference-config affineK8V4
        --quality-candidate-first
      )
      run_logged "real-model-sparse-threshold" "$LM_ROOT" "$REAL_DIR/sparse-threshold" \
        "${sparse_base[@]}" \
        --sparse-v threshold \
        --sparse-v-threshold "$SPARSE_THRESHOLD" \
        --diagnostics-output "$REAL_DIR/sparse-threshold.json"
      run_logged "real-model-sparse-top-k" "$LM_ROOT" "$REAL_DIR/sparse-top-k" \
        "${sparse_base[@]}" \
        --sparse-v topK \
        --sparse-v-top-k "$SPARSE_TOP_K" \
        --diagnostics-output "$REAL_DIR/sparse-top-k.json"
      run_logged "real-model-sparse-cumulative" "$LM_ROOT" "$REAL_DIR/sparse-cumulative" \
        "${sparse_base[@]}" \
        --sparse-v cumulativeMass \
        --sparse-v-cumulative-mass "$SPARSE_MASS_FRACTION" \
        --diagnostics-output "$REAL_DIR/sparse-cumulative.json"
      run_logged "real-model-sparse-hybrid" "$LM_ROOT" "$REAL_DIR/sparse-hybrid" \
        "${sparse_base[@]}" \
        --sparse-v hybridCumulativeMassTopK \
        --sparse-v-cumulative-mass "$SPARSE_MASS_FRACTION" \
        --sparse-v-max-top-k "$SPARSE_MAX_TOP_K" \
        --diagnostics-output "$REAL_DIR/sparse-hybrid.json"
      for page_top_k in $(split_csv "$PAGE_TOP_KS"); do
        run_logged "real-model-sparse-page-top-k-$page_top_k" "$LM_ROOT" "$REAL_DIR/sparse-page-top-k-$page_top_k" \
          "${sparse_base[@]}" \
          --sparse-v pageTopK \
          --sparse-v-top-k "$page_top_k" \
          --diagnostics-output "$REAL_DIR/sparse-page-top-k-$page_top_k.json"
        if [[ "$PAGE_RECENT_TOKENS" != "0" ]]; then
          run_logged "real-model-sparse-page-top-k-$page_top_k-recent-$PAGE_RECENT_TOKENS" "$LM_ROOT" "$REAL_DIR/sparse-page-top-k-$page_top_k-recent-$PAGE_RECENT_TOKENS" \
            env TURBOQUANT_SPARSE_V_PAGE_RECENT_TOKENS="$PAGE_RECENT_TOKENS" \
            "${sparse_base[@]}" \
            --sparse-v pageTopK \
            --sparse-v-top-k "$page_top_k" \
            --diagnostics-output "$REAL_DIR/sparse-page-top-k-$page_top_k-recent-$PAGE_RECENT_TOKENS.json"
        fi
      done
      for candidate_row in $(split_csv "$CANDIDATE_SPARSE_ROWS"); do
        IFS=':' read -r candidate_recent candidate_pages candidate_older_top_k <<< "$candidate_row"
        if [[ -z "${candidate_recent:-}" || -z "${candidate_pages:-}" || -z "${candidate_older_top_k:-}" ]]; then
          printf 'skipping malformed CandidateSparse row %s\n' "$candidate_row"
          continue
        fi
        candidate_label="r${candidate_recent}-p${candidate_pages}-older${candidate_older_top_k}"
        run_logged "real-model-sparse-candidate-$candidate_label" "$LM_ROOT" "$REAL_DIR/sparse-candidate-$candidate_label" \
          "${sparse_base[@]}" \
          --sparse-v candidateSparse \
          --sparse-v-recent-tokens "$candidate_recent" \
          --sparse-v-candidate-pages "$candidate_pages" \
          --sparse-v-top-k "$candidate_older_top_k" \
          --diagnostics-output "$REAL_DIR/sparse-candidate-$candidate_label.json"
      done
    fi

    printf '## Real-Model K8/Vx Acceptance\n\n' >> "$SUMMARY"
    append_real_model_tables
    if [[ "$RUN_SPARSE" == "1" ]]; then
      printf '## Real-Model Sparse-V Diagnostics\n\n' >> "$SUMMARY"
      append_real_model_sparse_tables
    else
      printf '## Real-Model Sparse-V Diagnostics\n\n' >> "$SUMMARY"
      printf 'Skipped because `TQ_ACCEPTANCE_RUN_SPARSE=1` was not set. Sparse-V and CandidateSparse are rejected/explicit diagnostic modes, not default acceptance candidates.\n\n' >> "$SUMMARY"
    fi
  fi
fi

if [[ "$SKIP_PROOF" != "1" ]]; then
  run_logged "build-qwen-proof" "$LM_ROOT" "$PROOF_DIR/build-turboquantqwenproof" \
    swift build --product TurboQuantQwenProof -c release

  proof_base=(
    .build/release/TurboQuantQwenProof
    --profiles "$PROOF_PROFILES"
    --schemes turbo8
    --contexts "$PROOF_CONTEXTS"
    --query-lengths "$PROOF_QUERY_LENGTHS"
    --iterations "$PROOF_ITERATIONS"
    --warmup "$PROOF_WARMUP"
    --runtime-mode capacity
  )

  run_logged "proof-dense-k8v4-baseline" "$LM_ROOT" "$PROOF_DIR/dense-k8v4-baseline" \
    "${proof_base[@]}"
  if [[ "$RUN_SPARSE" == "1" ]]; then
    run_logged "proof-sparse-threshold" "$LM_ROOT" "$PROOF_DIR/sparse-threshold" \
      "${proof_base[@]}" --sparse-v force --sparse-v-mode threshold --sparse-v-threshold "$SPARSE_THRESHOLD"
    run_logged "proof-sparse-top-k" "$LM_ROOT" "$PROOF_DIR/sparse-top-k" \
      "${proof_base[@]}" --sparse-v-mode top-k --sparse-v-top-k "$SPARSE_TOP_K"
    run_logged "proof-sparse-cumulative" "$LM_ROOT" "$PROOF_DIR/sparse-cumulative" \
      "${proof_base[@]}" --sparse-v-mode cumulative-mass --sparse-v-cumulative-mass "$SPARSE_MASS"
    run_logged "proof-sparse-hybrid" "$LM_ROOT" "$PROOF_DIR/sparse-hybrid" \
      "${proof_base[@]}" --sparse-v-mode hybrid \
      --sparse-v-hybrid-mass "$SPARSE_MASS" --sparse-v-max-top-k "$SPARSE_MAX_TOP_K"

    printf '## Sparse-V Synthetic Proof Diagnostics\n\n' >> "$SUMMARY"
    append_proof_table
  else
    printf '## Sparse-V Synthetic Proof Diagnostics\n\n' >> "$SUMMARY"
    printf 'Skipped because `TQ_ACCEPTANCE_RUN_SPARSE=1` was not set.\n\n' >> "$SUMMARY"
  fi
fi

write_missing_flags
append_structured_gate

{
  printf '## Artifacts\n\n'
  printf '%s\n' "- Command CSV: \`$CSV\`"
  printf '%s\n' "- PolarWHT gate JSON: \`$GATE_JSON\`"
  printf '%s\n' "- Real-model logs: \`$REAL_DIR\`"
  printf '%s\n' "- Proof JSON/logs: \`$PROOF_DIR\`"
} >> "$SUMMARY"

printf 'TurboQuant acceptance matrix written to %s\n' "$SUMMARY"
