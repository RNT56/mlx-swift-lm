#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any


NATIVE_POLAR_LABEL = "hybridK8PolarWHTV3"
REFERENCE_POLAR_LABEL = "polarWHTReferenceV3"
NATIVE_POLAR_PATH = "metalHybridK8PolarWHTValue"
FP16_CLAIM_THRESHOLD = 0.98


def load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def unique(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(value for value in values if value))


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def normalize_contexts(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    contexts: list[int] = []
    for value in values:
        try:
            contexts.append(int(value))
        except (TypeError, ValueError):
            continue
    return contexts


def local_diagnostics_path(root: pathlib.Path) -> pathlib.Path | None:
    candidates = [
        root / "local-diagnostics.json",
        root / "local-runs" / "local-diagnostics.json",
        root / "real-model" / "k8vx-vs-dense-k8v4.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def required_contexts(metadata: dict[str, Any] | None, local: dict[str, Any] | None) -> list[int]:
    if metadata:
        contexts = normalize_contexts((metadata.get("run") or {}).get("contexts"))
        if contexts:
            return contexts
    if local:
        contexts = normalize_contexts((local.get("run") or {}).get("contexts"))
        if contexts:
            return contexts
    return []


def command_failures(commands: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    for row in commands:
        status = row.get("status", "")
        if status not in ("", "0"):
            label = row.get("label", "unknown")
            failures.append(f"{label} exited {status}")
    return failures


def row_missing_fields(row: dict[str, Any]) -> list[str]:
    required: list[tuple[str, list[str], bool]] = [
        ("decodeTokensPerSecond", ["decodeTokensPerSecond"], True),
        ("speedRatioToFP16", ["speedRatioToFP16"], True),
        (
            "residentKVCompressionRatio",
            ["residentKVCompressionRatio", "estimatedMemoryReductionRatio"],
            True,
        ),
        ("peakActiveMemoryBytes", ["peakActiveMemoryBytes"], True),
        ("steadyActiveMemoryBytes", ["steadyActiveMemoryBytes"], True),
        ("requestedBackend", ["requestedBackend"], False),
        ("selectedAttentionPaths", ["selectedAttentionPaths"], False),
        ("sparseActiveLayerCount", ["sparseActiveLayerCount"], False),
        ("sparseRequestedLayerCount", ["sparseRequestedLayerCount"], False),
        ("sparseRequestedButInactive", ["sparseRequestedButInactive"], False),
        ("sparseFallbackReason", ["sparseFallbackReason"], False),
        ("qualityPassed", ["qualityPassed"], False),
        ("promotionGate", ["promotionGate"], False),
    ]
    missing: list[str] = []
    for public_name, keys, numeric in required:
        value = first_present(row, keys)
        if numeric:
            if as_number(value) is None:
                missing.append(public_name)
        elif public_name == "sparseFallbackReason":
            if not any(key in row for key in keys):
                missing.append(public_name)
        elif value is None:
            missing.append(public_name)
    return missing


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    gate = row.get("promotionGate") or {}
    return {
        "context": row.get("context"),
        "label": row.get("label"),
        "decodeTokensPerSecond": row.get("decodeTokensPerSecond"),
        "speedRatioToFP16": row.get("speedRatioToFP16"),
        "speedRatioToAffineK8V4": row.get("speedRatioToAffineK8V4"),
        "residentKVCompressionRatio": row.get("residentKVCompressionRatio")
        or row.get("estimatedMemoryReductionRatio"),
        "peakActiveMemoryBytes": row.get("peakActiveMemoryBytes"),
        "steadyActiveMemoryBytes": row.get("steadyActiveMemoryBytes"),
        "requestedBackend": row.get("requestedBackend") or gate.get("requestedBackend"),
        "selectedAttentionPaths": row.get("selectedAttentionPaths")
        or gate.get("selectedAttentionPaths")
        or [],
        "sparseActiveLayerCount": row.get("sparseActiveLayerCount"),
        "sparseRequestedLayerCount": row.get("sparseRequestedLayerCount"),
        "sparseRequestedButInactive": row.get("sparseRequestedButInactive")
        or gate.get("sparseRequestedButInactive"),
        "sparseFallbackReason": row.get("sparseFallbackReason"),
        "fallbackReasons": row.get("fallbackReasons") or gate.get("fallbackReasons") or [],
        "nativeFallbackReasons": row.get("nativeFallbackReasons") or [],
        "qualityPassed": row.get("qualityPassed") if "qualityPassed" in row else gate.get("qualityPassed"),
        "qualityReason": row.get("qualityReason") or gate.get("qualityReason"),
        "promotionEligible": row.get("promotionEligible", gate.get("promotionEligible")),
        "promotionBlockReasons": row.get("promotionBlockReasons")
        or gate.get("promotionBlockReasons")
        or [],
        "promotionGate": gate,
    }


def audit_native_row(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    context = row.get("context")
    gate = row.get("promotionGate") or {}
    gate_reasons = row.get("promotionBlockReasons") or gate.get("promotionBlockReasons") or []
    for missing in row_missing_fields(row):
        blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: missing {missing}")
    if gate.get("requiresMetalPolarWHT") is not True:
        blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: gate did not require metalPolarWHT")
    if gate.get("metalPolarWHTAvailable") is not True:
        blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: metalPolarWHT unavailable")
    if gate.get("rawFallbackAllocated") is True:
        blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: raw fallback allocated")
    if gate.get("decodedFallbackPathActive") is True:
        blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: decoded fallback path active")
    selected_paths = row.get("selectedAttentionPaths") or gate.get("selectedAttentionPaths") or []
    if NATIVE_POLAR_PATH not in selected_paths:
        blockers.append(
            f"{NATIVE_POLAR_LABEL} ctx {context}: {NATIVE_POLAR_PATH} path was not selected"
        )
    requested = as_int(row.get("sparseRequestedLayerCount"))
    active = as_int(row.get("sparseActiveLayerCount"))
    if gate.get("sparseRequestedButInactive") is True or (
        requested is not None and requested > 0 and active == 0
    ):
        blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: Sparse-V requested but inactive")
    quality_passed = row.get("qualityPassed") if "qualityPassed" in row else gate.get("qualityPassed")
    if quality_passed is not True:
        quality_reasons = [reason for reason in gate_reasons if reason.startswith("quality gate")]
        if quality_reasons:
            blockers.extend(f"{NATIVE_POLAR_LABEL} ctx {context}: {reason}" for reason in quality_reasons)
        else:
            blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: quality gate missing or failed")
    if row.get("promotionEligible", gate.get("promotionEligible")) is not True:
        skipped_reason_fragments = [
            "metalpolarwht unavailable",
            "raw fallback allocated",
            "decoded or unavailable fallback path active",
            "sparse-v requested but inactive",
            "quality gate",
        ]
        uncategorized_reasons = [
            reason
            for reason in gate_reasons
            if not any(fragment in reason.lower() for fragment in skipped_reason_fragments)
        ]
        if uncategorized_reasons:
            blockers.extend(
                f"{NATIVE_POLAR_LABEL} ctx {context}: {reason}"
                for reason in uncategorized_reasons
            )
        elif not gate_reasons:
            blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: promotion gate blocked")
    return blockers


def audit_reference_row(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    context = row.get("context")
    for missing in row_missing_fields(row):
        if missing == "promotionGate":
            continue
        blockers.append(f"{REFERENCE_POLAR_LABEL} ctx {context}: missing {missing}")
    quality_passed = row.get("qualityPassed")
    if quality_passed is not True:
        blockers.append(f"{REFERENCE_POLAR_LABEL} ctx {context}: quality gate missing or failed")
    return blockers


def build_upstream_report(
    metadata: dict[str, Any] | None,
    commands: list[dict[str, str]],
    upstream_rows: list[dict[str, str]],
) -> dict[str, Any]:
    upstream_commands = [row for row in commands if row.get("label", "").startswith("upstream-")]
    blockers: list[str] = []
    failures = command_failures(upstream_commands)
    blockers.extend(f"upstream command failed: {failure}" for failure in failures)
    if not upstream_rows:
        blockers.append("no upstream decode rows parsed")
    upstream_commit = ((metadata or {}).get("repos") or {}).get("upstreamCommit")
    if not upstream_commit:
        blockers.append("upstream commit missing from metadata")
    return {
        "status": "reproduced" if not blockers else "blocked",
        "upstreamCommit": upstream_commit,
        "commandCount": len(upstream_commands),
        "rowCount": len(upstream_rows),
        "contexts": unique([str(row.get("context")) for row in upstream_rows]),
        "rows": upstream_rows,
        "blockers": unique(blockers),
    }


def build_local_report(
    metadata: dict[str, Any] | None,
    local: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if local is None:
        return {
            "status": "blocked",
            "requiredContexts": required_contexts(metadata, None),
            "rowCount": 0,
            "nativePolarWHTRows": [],
            "referencePolarWHTRows": [],
            "blockers": ["local diagnostics JSON was not produced"],
        }

    rows = local.get("throughput") or []
    required = required_contexts(metadata, local)
    native_rows = [row for row in rows if row.get("label") == NATIVE_POLAR_LABEL]
    reference_rows = [row for row in rows if row.get("label") == REFERENCE_POLAR_LABEL]

    native_by_context = {as_int(row.get("context")): row for row in native_rows}
    reference_by_context = {as_int(row.get("context")): row for row in reference_rows}
    if not native_rows:
        blockers.append(f"no local {NATIVE_POLAR_LABEL} throughput rows")
    if not reference_rows:
        blockers.append(f"no local {REFERENCE_POLAR_LABEL} throughput rows")
    for context in required:
        if context not in native_by_context:
            blockers.append(f"missing {NATIVE_POLAR_LABEL} throughput row for context {context}")
        if context not in reference_by_context:
            blockers.append(f"missing {REFERENCE_POLAR_LABEL} throughput row for context {context}")
    for row in native_rows:
        blockers.extend(audit_native_row(row))
    for row in reference_rows:
        blockers.extend(audit_reference_row(row))

    local_run = local.get("run") or {}
    return {
        "status": "eligible" if not blockers else "blocked",
        "modelPath": local.get("modelPath"),
        "repoCommits": local.get("repoCommits") or {},
        "run": {
            "contexts": local_run.get("contexts"),
            "qualityContexts": local_run.get("qualityContexts"),
            "generateTokens": local_run.get("generateTokens"),
            "throughputRepeats": local_run.get("throughputRepeats"),
            "throughputCooldownSeconds": local_run.get("throughputCooldownSeconds"),
            "qualityCooldownSeconds": local_run.get("qualityCooldownSeconds"),
            "qualityReferenceLabel": local_run.get("qualityReferenceLabel"),
            "qualityCandidateFirst": local_run.get("qualityCandidateFirst"),
            "runQualityGates": local_run.get("runQualityGates"),
            "sparseOverride": local_run.get("sparseOverride"),
        },
        "memory": local.get("memory"),
        "requiredContexts": required,
        "rowCount": len(rows),
        "nativePolarWHTRows": [summarize_row(row) for row in native_rows],
        "referencePolarWHTRows": [summarize_row(row) for row in reference_rows],
        "blockers": unique(blockers),
    }


def build_fp16_claim_report(local_report: dict[str, Any], threshold: float) -> dict[str, Any]:
    blockers: list[str] = []
    rows = local_report.get("nativePolarWHTRows") or []
    ratios: list[float] = []
    for row in rows:
        context = row.get("context")
        ratio = as_number(row.get("speedRatioToFP16"))
        if ratio is None:
            blockers.append(f"{NATIVE_POLAR_LABEL} ctx {context}: missing speedRatioToFP16")
        else:
            ratios.append(ratio)
            if ratio < threshold:
                blockers.append(
                    f"{NATIVE_POLAR_LABEL} ctx {context}: {ratio:.3f}x FP16 below {threshold:.3f}x"
                )
    if not rows:
        blockers.append(f"no {NATIVE_POLAR_LABEL} rows available for FP16 speed claim")
    if local_report.get("status") != "eligible":
        blockers.append("local promotion gate blocked")
    return {
        "status": "supported" if not blockers else "blocked",
        "thresholdVsFP16": threshold,
        "bestSpeedRatioToFP16": max(ratios) if ratios else None,
        "worstSpeedRatioToFP16": min(ratios) if ratios else None,
        "rowsMeetingThreshold": [
            row for row in rows if (as_number(row.get("speedRatioToFP16")) or -math.inf) >= threshold
        ],
        "blockers": unique(blockers),
        "caveat": (
            "Upstream's visible README table reports K8+V4 decode around 0.72x FP16; "
            "a 0.98x FP16 claim is unsupported unless this gate has reproduced local rows "
            "at or above the threshold on the same machine."
        ),
    }


def build_report(root: pathlib.Path, threshold: float) -> dict[str, Any]:
    metadata = load_json(root / "run-metadata.json")
    commands = read_csv(root / "commands.csv")
    upstream_rows = read_csv(root / "upstream-results.csv")
    local_path = local_diagnostics_path(root)
    local = load_json(local_path) if local_path else None

    upstream = build_upstream_report(metadata, commands, upstream_rows)
    local_report = build_local_report(metadata, local)
    fp16_claim = build_fp16_claim_report(local_report, threshold)
    command_blockers = [f"command failed: {failure}" for failure in command_failures(commands)]
    blockers = unique(
        command_blockers
        + upstream.get("blockers", [])
        + local_report.get("blockers", [])
    )

    metadata_run = (metadata or {}).get("run") or {}
    return {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "artifactRoot": str(root),
        "status": "eligible" if not blockers else "blocked",
        "promotionEligible": not blockers,
        "promotionBlockReasons": blockers,
        "sameMachineReproduction": {
            "upstream": upstream,
            "local": local_report,
        },
        "fp16SpeedClaim": fp16_claim,
        "run": {
            "upstreamModel": metadata_run.get("upstreamModel"),
            "upstreamBenchmarkScript": metadata_run.get("upstreamBenchmarkScript"),
            "upstreamBenchmarkAdapter": metadata_run.get("upstreamBenchmarkAdapter"),
            "localModelDir": metadata_run.get("localModelDir") or local_report.get("modelPath"),
            "contexts": metadata_run.get("contexts") or local_report.get("requiredContexts"),
            "promptTokens": metadata_run.get("promptTokens"),
            "qualityContexts": metadata_run.get("qualityContexts")
            or (local_report.get("run") or {}).get("qualityContexts"),
            "generateTokens": metadata_run.get("generateTokens")
            or (local_report.get("run") or {}).get("generateTokens"),
            "throughputRepeats": metadata_run.get("throughputRepeats")
            or (local_report.get("run") or {}).get("throughputRepeats"),
            "cooldownSeconds": metadata_run.get("cooldownSeconds")
            or (local_report.get("run") or {}).get("throughputCooldownSeconds"),
            "qualityGates": metadata_run.get("qualityGates")
            or (local_report.get("run") or {}).get("runQualityGates"),
        },
        "machine": (metadata or {}).get("machine"),
        "memory": local_report.get("memory"),
        "commands": {
            "count": len(commands),
            "failures": command_failures(commands),
        },
        "sourceFiles": {
            "metadata": str(root / "run-metadata.json") if (root / "run-metadata.json").exists() else None,
            "commands": str(root / "commands.csv") if (root / "commands.csv").exists() else None,
            "upstreamResults": str(root / "upstream-results.csv")
            if (root / "upstream-results.csv").exists()
            else None,
            "localDiagnostics": str(local_path) if local_path else None,
        },
    }


def write_report(report: dict[str, Any], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "local-runs").mkdir()
        (root / "run-metadata.json").write_text(
            json.dumps(
                {
                    "repos": {"upstreamCommit": "6e928d7"},
                    "run": {
                        "upstreamModel": "mlx-community/Qwen",
                        "localModelDir": "/models/qwen",
                        "contexts": ["32768"],
                        "promptTokens": ["32768"],
                        "qualityContexts": ["32768"],
                        "generateTokens": 16,
                        "throughputRepeats": 3,
                        "cooldownSeconds": 1.0,
                        "qualityGates": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (root / "commands.csv").write_text(
            "label,status,seconds,stdout,stderr\n"
            "upstream-real-model-ctx-32768,0,10,out,err\n"
            "local-turboquant-inference-parity,0,10,out,err\n",
            encoding="utf-8",
        )
        (root / "upstream-results.csv").write_text(
            "context,label,prefillTokS,decodeTokS,peakMemoryGB,generatedTokens\n"
            "32768,K8+V4,100,72,12,16\n",
            encoding="utf-8",
        )
        row_base = {
            "status": "ok",
            "context": 32768,
            "decodeTokensPerSecond": 99.0,
            "prefillTokensPerSecond": 100.0,
            "speedRatioToFP16": 0.99,
            "speedRatioToAffineK8V4": 1.02,
            "residentKVCompressionRatio": 4.6,
            "peakActiveMemoryBytes": 123,
            "steadyActiveMemoryBytes": 100,
            "requestedBackend": "metalPolarWHT",
            "selectedAttentionPaths": [NATIVE_POLAR_PATH],
            "sparseActiveLayerCount": 0,
            "sparseRequestedLayerCount": 0,
            "sparseRequestedButInactive": False,
            "sparseFallbackReason": None,
            "qualityPassed": True,
            "qualityReason": None,
            "promotionEligible": True,
            "promotionBlockReasons": [],
            "promotionGate": {
                "promotionEligible": True,
                "promotionBlockReasons": [],
                "requestedBackend": "metalPolarWHT",
                "selectedAttentionPaths": [NATIVE_POLAR_PATH],
                "requiresMetalPolarWHT": True,
                "metalPolarWHTAvailable": True,
                "rawFallbackAllocated": False,
                "decodedFallbackPathActive": False,
                "sparseRequestedButInactive": False,
                "qualityPassed": True,
                "fallbackReasons": [],
            },
        }
        reference = dict(row_base)
        reference.update(
            {
                "label": REFERENCE_POLAR_LABEL,
                "requestedBackend": "polarWHTReference",
                "selectedAttentionPaths": ["polarWHTReferenceHybrid"],
                "promotionEligible": False,
                "promotionBlockReasons": [
                    "polarWHTReference is a measured reference path, not a native metalPolarWHT promotion"
                ],
                "promotionGate": {
                    **row_base["promotionGate"],
                    "promotionEligible": False,
                    "promotionBlockReasons": [
                        "polarWHTReference is a measured reference path, not a native metalPolarWHT promotion"
                    ],
                    "requestedBackend": "polarWHTReference",
                    "selectedAttentionPaths": ["polarWHTReferenceHybrid"],
                    "requiresMetalPolarWHT": False,
                    "metalPolarWHTAvailable": None,
                },
            }
        )
        native = dict(row_base)
        native["label"] = NATIVE_POLAR_LABEL
        (root / "local-runs" / "local-diagnostics.json").write_text(
            json.dumps(
                {
                    "modelPath": "/models/qwen",
                    "run": {
                        "contexts": [32768],
                        "qualityContexts": [32768],
                        "generateTokens": 16,
                        "throughputRepeats": 3,
                        "throughputCooldownSeconds": 1.0,
                        "runQualityGates": True,
                    },
                    "throughput": [native, reference],
                    "quality": [],
                    "memory": {"estimateSource": "self-test"},
                }
            ),
            encoding="utf-8",
        )
        report = build_report(root, FP16_CLAIM_THRESHOLD)
        assert report["promotionEligible"] is True, report["promotionBlockReasons"]
        assert report["fp16SpeedClaim"]["status"] == "supported"

        native["promotionGate"] = dict(native["promotionGate"])
        native["promotionGate"]["metalPolarWHTAvailable"] = False
        native["promotionEligible"] = False
        native["promotionBlockReasons"] = ["metalPolarWHT unavailable"]
        (root / "local-runs" / "local-diagnostics.json").write_text(
            json.dumps(
                {
                    "modelPath": "/models/qwen",
                    "run": {"contexts": [32768], "runQualityGates": True},
                    "throughput": [native, reference],
                    "memory": {"estimateSource": "self-test"},
                }
            ),
            encoding="utf-8",
        )
        blocked = build_report(root, FP16_CLAIM_THRESHOLD)
        assert blocked["promotionEligible"] is False
        assert any("metalPolarWHT unavailable" in reason for reason in blocked["promotionBlockReasons"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed PolarWHT upstream/local acceptance gate JSON."
    )
    parser.add_argument("--artifact-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--fp16-threshold", type=float, default=FP16_CLAIM_THRESHOLD)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if args.artifact_root is None or args.output is None:
        parser.error("--artifact-root and --output are required unless --self-test is used")

    report = build_report(args.artifact_root, args.fp16_threshold)
    write_report(report, args.output)
    print(
        f"wrote {args.output} "
        f"(status={report['status']}, fp16Claim={report['fp16SpeedClaim']['status']})"
    )
    if args.strict and not report["promotionEligible"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
