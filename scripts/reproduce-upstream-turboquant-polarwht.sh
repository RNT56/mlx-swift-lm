#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORKS_ROOT="$(cd "$LM_ROOT/.." && pwd)"
RUN_ID="${TQ_POLARWHT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${TQ_POLARWHT_ARTIFACT_ROOT:-$FORKS_ROOT/artifacts/turboquant-polarwht-upstream-$RUN_ID}"

UPSTREAM_REPO="${TQ_UPSTREAM_REPO:-https://github.com/arozanov/turboquant-mlx.git}"
UPSTREAM_COMMIT="${TQ_UPSTREAM_COMMIT:-6e928d715595dee9f6b6cc3968baa44e1f408d28}"
UPSTREAM_ROOT="${TQ_UPSTREAM_ROOT:-$ARTIFACT_ROOT/upstream/turboquant-mlx}"
UPSTREAM_MODEL="${TQ_UPSTREAM_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
UPSTREAM_PYTHON="${TQ_UPSTREAM_PYTHON:-python3}"
UPSTREAM_INSTALL="${TQ_UPSTREAM_INSTALL:-0}"
UPSTREAM_COMPAT="${TQ_POLARWHT_UPSTREAM_COMPAT:-auto}"
UPSTREAM_BENCHMARK_SCRIPT="scripts/bench_real_model.py"
UPSTREAM_BENCHMARK_ADAPTER="none"

LOCAL_MODEL_DIR="${TQ_MODEL_DIR:-}"
LOCAL_CONTEXTS="${TQ_POLARWHT_CONTEXTS:-32768,65536,131072}"
LOCAL_CONFIGS="${TQ_POLARWHT_LOCAL_CONFIGS:-fp16,affineK8V4,affineInt4,turbo4v2,polarWHTV3,polarWHTReferenceV3,hybridK8PolarWHTV3,hybridK8PolarWHTV4,hybridK8PolarWHTV3Reference}"
QUALITY_CONTEXTS="${TQ_POLARWHT_QUALITY_CONTEXTS:-$LOCAL_CONTEXTS}"
GENERATE_TOKENS="${TQ_POLARWHT_GENERATE_TOKENS:-16}"
PROMPT_TOKENS="${TQ_POLARWHT_PROMPT_TOKENS:-$LOCAL_CONTEXTS}"
BITS="${TQ_POLARWHT_BITS:-3}"
THROUGHPUT_REPEATS="${TQ_POLARWHT_THROUGHPUT_REPEATS:-3}"
COOLDOWN_SECONDS="${TQ_POLARWHT_COOLDOWN_SECONDS:-1.0}"
RUN_QUALITY_GATES="${TQ_POLARWHT_QUALITY_GATES:-1}"
STRICT="${TQ_POLARWHT_STRICT:-0}"
SKIP_UPSTREAM="${TQ_POLARWHT_SKIP_UPSTREAM:-0}"
SKIP_LOCAL="${TQ_POLARWHT_SKIP_LOCAL:-0}"

UPSTREAM_DIR="$ARTIFACT_ROOT/upstream-runs"
LOCAL_DIR="$ARTIFACT_ROOT/local-runs"
COMMANDS_CSV="$ARTIFACT_ROOT/commands.csv"
SUMMARY_MD="$ARTIFACT_ROOT/summary.md"
METADATA_JSON="$ARTIFACT_ROOT/run-metadata.json"
GATE_JSON="$ARTIFACT_ROOT/polarwht-acceptance-gate.json"

usage() {
  cat <<'USAGE'
Reproduce upstream turboquant-mlx and local Swift TurboQuant runs on the same machine.

This harness intentionally reports upstream reproduction and local PolarWHT
availability separately. It does not promote a 0.98x-FP16 claim unless local
diagnostics show metalPolarWHT was active, sparse requests were active when
requested, no raw fallback was used, and quality gates passed.

Environment:
  TQ_MODEL_DIR                    Local MLX model directory for TurboQuantInferenceParity.
  TQ_UPSTREAM_MODEL               Upstream mlx-lm model id/path. Default: mlx-community/Qwen2.5-7B-Instruct-4bit.
  TQ_POLARWHT_CONTEXTS            Contexts for local runs. Default: 32768,65536,131072.
  TQ_POLARWHT_PROMPT_TOKENS       Prompt-token contexts for upstream runs. Default: same as contexts.
  TQ_POLARWHT_GENERATE_TOKENS     Decode tokens per cell. Default: 16.
  TQ_POLARWHT_LOCAL_CONFIGS       Local configs. Default: fp16,affineK8V4,affineInt4,turbo4v2,polarWHTV3,polarWHTReferenceV3,hybridK8PolarWHTV3,hybridK8PolarWHTV4,hybridK8PolarWHTV3Reference.
  TQ_POLARWHT_QUALITY_CONTEXTS    Local quality contexts. Default: same as contexts.
  TQ_POLARWHT_THROUGHPUT_REPEATS  Local repeated samples. Default: 3.
  TQ_POLARWHT_COOLDOWN_SECONDS    Cooldown after local cells. Default: 1.0.
  TQ_UPSTREAM_ROOT                Reused upstream checkout path. Default: artifact-local checkout.
  TQ_UPSTREAM_COMMIT              Upstream commit. Default: 6e928d715595dee9f6b6cc3968baa44e1f408d28.
  TQ_UPSTREAM_INSTALL             1 to run python -m pip install -e <upstream>. Default: 0.
  TQ_POLARWHT_UPSTREAM_COMPAT     auto/1 to use a recorded benchmark-script adapter for current mlx-lm layer access. Default: auto.
  TQ_POLARWHT_SKIP_UPSTREAM       1 to skip upstream Python run.
  TQ_POLARWHT_SKIP_LOCAL          1 to skip local Swift run.
  TQ_POLARWHT_STRICT              1 to exit non-zero if any command fails. Default: 0.

Example:
  TQ_MODEL_DIR=/path/to/mlx-model TQ_UPSTREAM_INSTALL=1 scripts/reproduce-upstream-turboquant-polarwht.sh
USAGE
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$UPSTREAM_DIR" "$LOCAL_DIR"
printf 'label,status,seconds,stdout,stderr\n' > "$COMMANDS_CSV"
FAILED_COUNT=0

split_csv() {
  local raw="$1"
  IFS=',' read -ra values <<< "$raw"
  for value in "${values[@]}"; do
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    [[ -n "$value" ]] && printf '%s\n' "$value"
  done
}

json_string_array() {
  python3 - "$1" <<'PY'
import json
import sys
items = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(json.dumps(items))
PY
}

run_logged() {
  local label="$1"
  local cwd="$2"
  local log="$3"
  shift 3
  printf 'running %s\n' "$label"
  local start end status seconds
  start="$(date +%s)"
  (
    cd "$cwd"
    "$@"
  ) >"$log.out" 2>"$log.err"
  status=$?
  end="$(date +%s)"
  seconds=$((end - start))
  printf '%s,%s,%s,%s,%s\n' "$label" "$status" "$seconds" "$log.out" "$log.err" >> "$COMMANDS_CSV"
  if [[ $status -ne 0 ]]; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
    printf 'failed %s (status %s); see %s.err\n' "$label" "$status" "$log"
  fi
  return 0
}

git_commit() {
  local root="$1"
  git -C "$root" rev-parse HEAD 2>/dev/null || printf 'unavailable'
}

write_metadata() {
  local contexts_json prompt_json configs_json quality_json
  contexts_json="$(json_string_array "$LOCAL_CONTEXTS")"
  prompt_json="$(json_string_array "$PROMPT_TOKENS")"
  configs_json="$(json_string_array "$LOCAL_CONFIGS")"
  quality_json="$(json_string_array "$QUALITY_CONTEXTS")"
  python3 - "$METADATA_JSON" <<PY
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

def command_output(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

payload = {
    "schemaVersion": 1,
    "generatedAt": datetime.now(timezone.utc).isoformat(),
    "artifactRoot": "$ARTIFACT_ROOT",
    "machine": {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "macOS": command_output(["sw_vers"]),
        "chip": command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memoryBytes": command_output(["sysctl", "-n", "hw.memsize"]),
    },
    "tools": {
        "swift": command_output(["swift", "--version"]),
        "python": command_output(["$UPSTREAM_PYTHON", "--version"]),
    },
    "repos": {
        "mlx-swift-lm": "$(git_commit "$LM_ROOT")",
        "mlx-swift": "$(git_commit "$FORKS_ROOT/mlx-swift")",
        "upstreamRepo": "$UPSTREAM_REPO",
        "upstreamCommit": "$UPSTREAM_COMMIT",
    },
    "run": {
        "upstreamModel": "$UPSTREAM_MODEL",
        "upstreamBenchmarkScript": "$UPSTREAM_BENCHMARK_SCRIPT",
        "upstreamBenchmarkAdapter": "$UPSTREAM_BENCHMARK_ADAPTER",
        "localModelDir": "$LOCAL_MODEL_DIR",
        "contexts": $contexts_json,
        "promptTokens": $prompt_json,
        "qualityContexts": $quality_json,
        "generateTokens": int("$GENERATE_TOKENS"),
        "bits": int("$BITS"),
        "localConfigs": $configs_json,
        "throughputRepeats": int("$THROUGHPUT_REPEATS"),
        "cooldownSeconds": float("$COOLDOWN_SECONDS"),
        "qualityGates": "$RUN_QUALITY_GATES" == "1",
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
}

checkout_upstream() {
  if [[ -d "$UPSTREAM_ROOT/.git" ]]; then
    run_logged "upstream-fetch" "$UPSTREAM_ROOT" "$UPSTREAM_DIR/fetch" \
      git fetch --depth 1 origin "$UPSTREAM_COMMIT"
  else
    mkdir -p "$(dirname "$UPSTREAM_ROOT")"
    run_logged "upstream-clone" "$(dirname "$UPSTREAM_ROOT")" "$UPSTREAM_DIR/clone" \
      git clone --depth 1 "$UPSTREAM_REPO" "$UPSTREAM_ROOT"
  fi
  run_logged "upstream-checkout" "$UPSTREAM_ROOT" "$UPSTREAM_DIR/checkout" \
    git checkout --detach "$UPSTREAM_COMMIT"
  if [[ "$UPSTREAM_INSTALL" == "1" ]]; then
    run_logged "upstream-install" "$UPSTREAM_ROOT" "$UPSTREAM_DIR/install" \
      "$UPSTREAM_PYTHON" -m pip install -e "$UPSTREAM_ROOT"
  fi
}

prepare_upstream_benchmark() {
  UPSTREAM_BENCHMARK_SCRIPT="scripts/bench_real_model.py"
  UPSTREAM_BENCHMARK_ADAPTER="none"
  case "$UPSTREAM_COMPAT" in
    0|false|False|FALSE|off|OFF|none)
      return 0
      ;;
  esac

  local compat_script="$UPSTREAM_DIR/bench_real_model_codex_compat.py"
  python3 - "$UPSTREAM_ROOT/scripts/bench_real_model.py" "$compat_script" <<'PY'
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
text = source.read_text(encoding="utf-8")
needle = "n_layers = len(model.model.layers)"
replacement = (
    "model_core = getattr(model, \"model\", model)\n"
    "    n_layers = len(model_core.layers)"
)
if needle not in text:
    raise SystemExit(f"adapter needle not found in {source}")
target.write_text(text.replace(needle, replacement), encoding="utf-8")
PY
  UPSTREAM_BENCHMARK_SCRIPT="$compat_script"
  UPSTREAM_BENCHMARK_ADAPTER="mlx-lm-layer-access"
}

parse_upstream_results() {
  python3 - "$UPSTREAM_DIR" "$ARTIFACT_ROOT/upstream-results.csv" <<'PY'
import csv
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
line_re = re.compile(
    r"^\s*(?P<label>.+?)\s+prefill=\s*(?P<prefill>[0-9.]+)\s+tok/s\s+"
    r"decode=\s*(?P<decode>[0-9.]+)\s+tok/s\s+peak=\s*(?P<peak>[0-9.]+)\s+GB\s+n=(?P<n>[0-9]+)"
)
rows = []
for log in sorted(root.glob("upstream-ctx-*.out")):
    context = log.stem.replace("upstream-ctx-", "")
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = line_re.match(raw)
        if match:
            rows.append({
                "context": context,
                "label": match.group("label").strip(),
                "prefillTokS": match.group("prefill"),
                "decodeTokS": match.group("decode"),
                "peakMemoryGB": match.group("peak"),
                "generatedTokens": match.group("n"),
            })
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "context",
            "label",
            "prefillTokS",
            "decodeTokS",
            "peakMemoryGB",
            "generatedTokens",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
PY
}

write_summary() {
  python3 - "$ARTIFACT_ROOT" "$SUMMARY_MD" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
metadata = json.loads((root / "run-metadata.json").read_text(encoding="utf-8"))
commands = list(csv.DictReader((root / "commands.csv").open(encoding="utf-8")))
upstream_rows = []
upstream_csv = root / "upstream-results.csv"
if upstream_csv.exists():
    upstream_rows = list(csv.DictReader(upstream_csv.open(encoding="utf-8")))
local_report_path = root / "local-runs" / "local-diagnostics.json"
local_report = None
if local_report_path.exists():
    try:
        local_report = json.loads(local_report_path.read_text(encoding="utf-8"))
    except Exception:
        local_report = None

local_rows = []
quality_rows = []
promotion_blockers = []
if local_report:
    local_rows = local_report.get("throughput", [])
    quality_rows = local_report.get("quality", [])
    native_polar_rows = [row for row in local_rows if row.get("label") == "hybridK8PolarWHTV3"]
    reference_polar_rows = [
        row for row in local_rows if row.get("label") == "polarWHTReferenceV3"
    ]
    if not native_polar_rows:
        promotion_blockers.append("no local hybridK8PolarWHTV3 throughput rows")
    if not reference_polar_rows:
        promotion_blockers.append("no local polarWHTReferenceV3 throughput rows")
    for row in native_polar_rows + reference_polar_rows:
        gate = row.get("promotionGate") or {}
        blockers = gate.get("promotionBlockReasons") or row.get("promotionBlockReasons") or []
        if blockers:
            promotion_blockers.append(
                f"{row.get('label')} ctx {row.get('context')}: {', '.join(blockers)}"
            )
        if not gate and not row.get("promotionEligible", False):
            promotion_blockers.append(
                f"{row.get('label')} ctx {row.get('context')}: promotion gate unavailable"
            )
else:
    promotion_blockers.append("local diagnostics JSON was not produced")

gate_path = root / "polarwht-acceptance-gate.json"
gate_report = None
if gate_path.exists():
    try:
        gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
        promotion_blockers = gate_report.get("promotionBlockReasons") or []
    except Exception:
        gate_report = None

failed = [row for row in commands if row.get("status") not in ("0", "")]
lines = [
    "# PolarWHT Upstream Reproduction",
    "",
    f"- Artifact root: `{root}`",
    f"- Upstream commit: `{metadata['repos']['upstreamCommit']}`",
    f"- Local mlx-swift-lm commit: `{metadata['repos']['mlx-swift-lm']}`",
    f"- Local mlx-swift commit: `{metadata['repos']['mlx-swift']}`",
    f"- Upstream model: `{metadata['run']['upstreamModel']}`",
    f"- Upstream benchmark script: `{metadata['run'].get('upstreamBenchmarkScript') or 'scripts/bench_real_model.py'}`",
    f"- Upstream benchmark adapter: `{metadata['run'].get('upstreamBenchmarkAdapter') or 'none'}`",
    f"- Local model dir: `{metadata['run']['localModelDir'] or 'unset'}`",
    f"- Contexts: `{','.join(metadata['run']['contexts'])}`",
    f"- Generated tokens: `{metadata['run']['generateTokens']}`",
    f"- Local configs: `{','.join(metadata['run']['localConfigs'])}`",
    "",
    "## Command Status",
    "",
    "| Label | Status | Seconds | Stdout | Stderr |",
    "| --- | ---: | ---: | --- | --- |",
]
for row in commands:
    lines.append(
        f"| `{row['label']}` | {row['status']} | {row['seconds']} | `{row['stdout']}` | `{row['stderr']}` |"
    )
lines.extend(["", "## Upstream Decode", ""])
if upstream_rows:
    lines.extend([
        "| Prompt tokens | Path | Prefill tok/s | Decode tok/s | Peak GB | Tokens |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in upstream_rows:
        lines.append(
            f"| {row['context']} | `{row['label']}` | {row['prefillTokS']} | "
            f"{row['decodeTokS']} | {row['peakMemoryGB']} | {row['generatedTokens']} |"
        )
else:
    lines.append("No upstream result rows were parsed.")
lines.extend(["", "## Local Decode", ""])
if local_rows:
    lines.extend([
        "| Context | Config | Backend | Paths | Decode tok/s | Vs FP16 | Vs K8/V4 | Resident KV comp | Peak active bytes | Steady active bytes | Sparse active/requested | Promotion | Blockers |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ])
    for row in local_rows:
        sparse = f"{row.get('sparseActiveLayerCount')}/{row.get('sparseRequestedLayerCount')}"
        paths = "/".join(row.get("selectedAttentionPaths") or row.get("codecCounts", {}).keys())
        blockers = "; ".join(row.get("promotionBlockReasons") or [])
        promotion = "eligible" if row.get("promotionEligible") else "blocked"
        lines.append(
            f"| {row.get('context')} | `{row.get('label')}` | `{row.get('requestedBackend') or ''}` | "
            f"`{paths}` | {row.get('decodeTokensPerSecond')} | "
            f"{row.get('speedRatioToFP16')} | {row.get('speedRatioToAffineK8V4')} | "
            f"{row.get('residentKVCompressionRatio') or row.get('estimatedMemoryReductionRatio')} | "
            f"{row.get('peakActiveMemoryBytes')} | {row.get('steadyActiveMemoryBytes')} | "
            f"{sparse} | {promotion} | {blockers} |"
        )
else:
    lines.append("No local throughput rows were produced.")
lines.extend(["", "## Local Quality", ""])
if quality_rows:
    lines.extend([
        "| Context | Config | Reference | Top-1 | KL mean | P95 abs | Cosine | Passed | Reason |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in quality_rows:
        lines.append(
            f"| {row.get('context')} | `{row.get('label')}` | `{row.get('referenceLabel')}` | "
            f"{row.get('deterministicTop1MatchRate')} | {row.get('logitKLDivergenceMean')} | "
            f"{row.get('logitMaxAbsErrorP95')} | {row.get('cosine')} | {row.get('passed')} | {row.get('reason') or ''} |"
        )
else:
    lines.append("No local quality rows were produced.")
lines.extend(["", "## Structured Gate", ""])
if gate_report:
    upstream = (gate_report.get("sameMachineReproduction") or {}).get("upstream") or {}
    local = (gate_report.get("sameMachineReproduction") or {}).get("local") or {}
    fp16_claim = gate_report.get("fp16SpeedClaim") or {}
    lines.extend([
        f"- Promotion status: `{gate_report.get('status')}`",
        f"- Upstream reproduction: `{upstream.get('status')}`",
        f"- Local PolarWHT gate: `{local.get('status')}`",
        f"- 0.98x FP16 claim: `{fp16_claim.get('status')}`",
        f"- Gate JSON: `{gate_path}`",
    ])
    if fp16_claim.get("caveat"):
        lines.append(f"- Caveat: {fp16_claim.get('caveat')}")
    blockers = gate_report.get("promotionBlockReasons") or []
    if blockers:
        lines.extend(["", "Gate blockers:"])
        lines.extend(f"- {blocker}" for blocker in blockers)
else:
    lines.append("Structured gate JSON was not produced.")
lines.extend(["", "## Promotion Gate", ""])
if promotion_blockers:
    lines.append("No promotion claim: " + "; ".join(dict.fromkeys(promotion_blockers)))
else:
    lines.append("Promotion blockers were not detected by this harness; review raw diagnostics before publishing.")
if failed:
    lines.extend(["", "## Failed Commands", ""])
    for row in failed:
        lines.append(f"- `{row['label']}` exited {row['status']}; stderr `{row['stderr']}`")
summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

write_acceptance_gate() {
  local command=(
    python3 "$SCRIPT_DIR/turboquant-polarwht-acceptance-gate.py"
    --artifact-root "$ARTIFACT_ROOT"
    --output "$GATE_JSON"
  )
  if [[ "$STRICT" == "1" ]]; then
    command+=(--strict)
  fi
  if ! "${command[@]}"; then
    FAILED_COUNT=$((FAILED_COUNT + 1))
    printf 'failed PolarWHT acceptance gate; see %s\n' "$GATE_JSON" >&2
  fi
}

write_metadata

if [[ "$SKIP_UPSTREAM" != "1" ]]; then
  checkout_upstream
  prepare_upstream_benchmark
  write_metadata
  for prompt_tokens in $(split_csv "$PROMPT_TOKENS"); do
    run_logged "upstream-real-model-ctx-$prompt_tokens" "$UPSTREAM_ROOT" \
      "$UPSTREAM_DIR/upstream-ctx-$prompt_tokens" \
      "$UPSTREAM_PYTHON" "$UPSTREAM_BENCHMARK_SCRIPT" \
        --model "$UPSTREAM_MODEL" \
        --prompt-tokens "$prompt_tokens" \
        --max-tokens "$GENERATE_TOKENS" \
        --bits "$BITS"
  done
  parse_upstream_results
fi

if [[ "$SKIP_LOCAL" != "1" ]]; then
  if [[ -z "$LOCAL_MODEL_DIR" ]]; then
    printf 'missing TQ_MODEL_DIR; local Swift run skipped\n' >&2
    FAILED_COUNT=$((FAILED_COUNT + 1))
  else
    run_logged "local-build-turboquant-inference-parity" "$LM_ROOT" \
      "$LOCAL_DIR/build-turboquant-inference-parity" \
      swift build --product TurboQuantInferenceParity -c release
    quality_args=()
    if [[ "$RUN_QUALITY_GATES" == "1" ]]; then
      quality_args+=(--quality-gates)
    fi
    run_logged "local-turboquant-inference-parity" "$LM_ROOT" \
      "$LOCAL_DIR/local-turboquant-inference-parity" \
      .build/release/TurboQuantInferenceParity \
        --model-dir "$LOCAL_MODEL_DIR" \
        --contexts "$LOCAL_CONTEXTS" \
        --quality-contexts "$QUALITY_CONTEXTS" \
        --generate-tokens "$GENERATE_TOKENS" \
        --configs "$LOCAL_CONFIGS" \
        --strict-configs \
        --throughput-repeats "$THROUGHPUT_REPEATS" \
        --throughput-cooldown "$COOLDOWN_SECONDS" \
        --quality-cooldown "$COOLDOWN_SECONDS" \
        --diagnostics-output "$LOCAL_DIR/local-diagnostics.json" \
        --diagnostics-samples-output "$LOCAL_DIR/local-samples.jsonl" \
        --emit-cache-policy-summary \
        "${quality_args[@]}"
  fi
fi

write_acceptance_gate
write_summary
printf 'wrote %s\n' "$SUMMARY_MD"

if [[ "$STRICT" == "1" && "$FAILED_COUNT" -ne 0 ]]; then
  exit 1
fi
exit 0
