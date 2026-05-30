# TurboQuant Profiles

TurboQuant profiles describe a safe runtime KV-cache policy for ordinary MLX model
weights. They are not calibrated model files and they do not replace Hugging Face
or MLX weight artifacts.

Use a profile when an application wants a model-aware default for
``GenerateParameters``:

```swift
let parameters = GenerateParameters(
    turboQuantModelID: "mlx-community/Qwen3-4B-4bit",
    modelType: "qwen3",
    keyHeadDimension: 128,
    valueHeadDimension: 128
)
```

If a bundled profile matches, the initializer enables
``KVCacheStrategy/turboQuant`` and fills in the preset, value bits, group size,
backend, and optimization policy. If no profile matches, the supplied base
parameters are returned unchanged.

Profile matching is intentionally conservative. Model ID patterns select
candidates, but config metadata remains authoritative: bundled profiles require
`model_type` and key/value head dimensions from the model's `config.json`.
Missing metadata intentionally returns no profile so callers keep their
conservative base parameters. Matching model sizes should use
explicit `B` or `M` size tokens only; quantization labels such as `4bit` and
`8bit` are not evidence that a model is 4B or 8B. Resolution APIs that expose diagnostics should
include rejection reasons for metadata, size, shape, mask, and context failures.

The root `TurboQuantProfiles/` directory contains JSON copies of the bundled
schema-version-2 profiles for downstream apps, release tooling, and manual
inspection. The Swift registry lives in ``TurboQuantProfileRegistry`` so library
consumers do not need package resources at runtime.

Schema version 2 separates advisory matching from measured product manifests.
The manifest validator reports exact missing or mismatched fields for the model
fingerprint, TurboQuant layout/preset/value-bit policy, and measured outcomes.
App preflight code should use strict fingerprint-aware selection when enabling
TurboQuant automatically; a fingerprint mismatch returns no TurboQuant profile so
the caller keeps its baseline or packed-cache path instead of guessing.

Bundled profiles keep the quality-first TurboQuant default of 4-bit V2 keys,
4-bit values, and group size 64 except where device evidence makes that unsafe.
Per-profile optimization policy is tuned from config-backed model shape, context
metadata, and device evidence: Qwen3.5/Qwen3.6 use Turbo8 with exact initial
prefill plus block-parallel fused compressed decode, while Gemma 3 1B
and Llama 3.2 3B use Turbo8 with exact initial prefill and raw-free compressed
decode; non-Qwen lower-bit guarded profiles use conservative routing. Safe context lengths mirror the
public model configuration or model card where available; callers should still
apply their own memory-fit policy before admitting very long prompts.

Bundled Qwen3.5/Qwen3.6 profiles cover current MLX Community dense and MoE
families with config-backed 256-dimensional KV head checks: `qwen3.5-0.8b`,
`qwen3.5-2b`, `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-27b`, `qwen3.6-27b`,
`qwen3.5-40b`, `qwen3.6-40b`, `qwen3.5-35b-a3b`, `qwen3.6-35b-a3b`,
`qwen3.5-97b-a10b`, `qwen3.5-122b-a10b`, and `qwen3.5-397b-a17b`.
The production precision for these profiles is Turbo8 with the `.preferThroughput`
runtime policy. That policy keeps exact initial prefill, skips resident raw
shadow decode, and routes decode through block-parallel fused compressed
attention for Qwen-style 256-dimensional grouped-query heads. Turbo4V2 and
Turbo3.5 are valid guarded proof candidates
only: Turbo4V2 uses 4-bit key magnitudes and 4-bit values, while Turbo3.5 uses
mixed 3/4-bit key magnitudes and 4-bit values. The profile API exposes both
explicitly, but release tooling must promote them per model, device, OS, and
context length only after the Qwen proof pipeline passes quality, memory, and
throughput gates. Qwen3.7 is intentionally not profiled until open MLX weights
exist.

Bundled Gemma profiles cover config-backed MLX Community legacy Gemma, Gemma 2,
Gemma 3, Gemma 3n, and Gemma 4 variants: `gemma-2b`, `gemma-7b`,
`gemma-2-2b`, `gemma-2-9b`, `gemma-2-27b`, `gemma-3-270m`, `gemma-3-1b`,
`gemma-3-4b`, `gemma-3-12b`, `gemma-3-27b`, `gemma-3n-e2b`,
`gemma-3n-e4b`, `gemma-4-e2b`, `gemma-4-e4b`, `gemma-4-26b-a4b`, and
`gemma-4-31b`. SEA-LION v3/v4 and named Gemma derivatives route through the
matching Gemma profile when their config metadata matches. Gemma 4 profiles
accept both the sliding 256-dimensional heads and global 512-dimensional heads.
Gemma 3 profiles accept the runtime-decoded head dimensions; callers should use
explicit `head_dim` when present and the Gemma 3 runtime default of 256 when MLX
configs omit it. These profiles require both `model_type` and KV head
dimensions; models whose configs omit non-inferable head dimensions
intentionally fall back to the caller's conservative defaults.

Bundled Llama profiles cover config-backed text-only Llama families and
Llama-compatible dense derivatives: `llama-2-7b`, `llama-2-13b`,
`llama-2-70b`, `llama-3-3b`, `llama-3-8b`, `llama-compatible-8b`,
`llama-3-16b`, `llama-3-70b`, `llama-compatible-70b`,
`llama-3.1-4b`, `llama-3.1-8b`, `llama-3.1-16b`, `llama-3.1-70b`,
`llama-3-120b`, `llama-3.1-120b`, `llama-3.1-405b`, `llama-3.2-1b`, `llama-3.2-3b`,
`llama-3.3-3b`, `llama-3.3-70b`, `llama-compatible-135m`,
`llama-compatible-160m`, `llama-compatible-1b`, `llama-compatible-2b`,
`llama-compatible-3b`, `llama-compatible-30b`, and `llama-compatible-4b`. These profiles require
`model_type=llama` and explicit KV head dimensions. Llama 3.2 Vision, MLLama,
Llama 4, LLava, Bunny, Idefics, Mixtral, and Mamba-Codestral are intentionally
excluded until the linked runtime exposes the corresponding supported model
surface.

Bundled Mistral profiles cover config-backed dense text, nested Mistral3 text,
Mistral4 MoE, and Pixtral VLM variants: `mistral-7b`, `mistral-nemo-12b`,
`ministral-8b-2410`, `mistral-compatible-8b`, `codestral-22b`,
`mistral-compatible-22b`, `mistral-small-22b`, `mistral-small-24b`,
`mistral-compatible-24b`, `mistral-large-2407`, `devstral-small-24b`,
`magistral-small-24b`,
`ministral3-3b`, `ministral3-8b`, `ministral3-14b`,
`mistral-small-3.1-24b`, `mistral-small-3.2-24b`,
`devstral-small-2-24b`, `devstral-2-123b`, `mistral-medium-3.5-128b`,
`mistral-small-4-119b-a6b`, and `pixtral-12b`. Mistral3 and Pixtral profiles
require nested `text_config.model_type` metadata where the public config exposes
a nested text model. Mistral Small 4 additionally requires routed expert
metadata matching its MoE config. Mixtral and Mamba-Codestral remain unsupported
until runtime model support exists.

## Safety Contract

Profiles are advisory. The runtime still enforces:

- Metal availability and bundled metallib loading.
- TurboQuant runtime self-tests.
- Query/key/value shape compatibility.
- Mask support for the selected attention path.
- Fallback to MLX packed quantization when the Metal path is unavailable.

Measured product outcomes must include the device class, OS, max context by
product mode, actual bytes per token, decode p50/p95, prefill p50, memory, and
quality gates such as logit KL, top-1 match, and long-context retrieval. Bundled
profiles that have not been reproduced on target hardware remain usable only as
conservative routing profiles; ``TurboQuantProfile/productManifestValidation(actualFingerprint:requireMeasuredOutcomes:)``
reports the exact fields that still need release evidence.

Use the benchmark entrypoints to generate profile evidence:

```sh
swift run TurboQuantBenchmark --iterations 25
swift run TurboQuantModelBenchmark --iterations 25
swift run TurboQuantQwenProof --release-matrix --strict --iterations 25 --dtype float16 --min-extended-tokens-per-second 20
```

The core benchmark records kernel-level decode, matmul, QK/AV, and fused-attention
behavior. The model benchmark records compressed prefill, decode attention, and
rotating-cache growth. The Qwen proof benchmark records Qwen3.5/Qwen3.6 profile
coverage, valid precision candidates, TurboQuant compressed attention vs plain
attention quality, short-context plain-route speed parity, extended-context
compressed decode p50/p95 throughput, and extended-context memory compression. Its release
matrix is a decode-throughput gate at query length 1 and gates production
throughput on the p95-latency token rate; use explicit
`--query-lengths` values for wider prompt/prefill stress sweeps. Promote bundled profile fields from pending to measured only when
the JSON includes the model revision, device, OS, latency, memory, and quality
data needed to reproduce the decision.

Production decode hardening keeps normal model attention layouts recoverable:
Q/K/V tensors are canonicalized before compressed cache encode and compressed
attention, compressed-update failures fall back to packed quantized attention
when the selected policy allows it, compressed-decode profiles do not keep packed
fallback caches resident on the hot path, and rotating single-token window masks
match the raw rotating cache mask order after wraparound.

The release matrix is intentionally capped at the current production throughput
contexts. Run `swift run TurboQuantQwenProof --contexts 32768 --query-lengths 1
--schemes turbo8,turbo4v2,turbo3_5 --dtype float16 --strict` as the 32K
production gate. Use `--experimental-contexts 65536,131072,262144` to append
64K/128K/256K proof rows for stress evidence without turning them into
production certification gates. The proof report labels those rows with
`gateScope = "largeContextExperiment"`, `strictGateRequired = false`, and
`certificationStatus = "experiment-only-not-production-certified"`; pass
`--require-experimental-gates` only when intentionally promoting those rows into
the strict gate set. Use `--warmup` to keep first-use Metal compilation outside
reported p50/p95 timing.

The pinned core `cff5d0ad87f79585ac778224c21a5278d25a4e79` adds a Mac-gated
GQA block-partials fused kernel for Qwen-style grouped-query decode. On the
local Mac proof machine it keeps 32K production rows green for Turbo8,
Turbo4V2, and Turbo3.5; 64K remains an experiment row because p95 is still not
consistently above the 20 tok/s floor under sustained load.

## Converted Weights

TurboQuant profile selection is independent of converted model weights. For
weight compression, use `swift run TurboQuantConverter` from the `mlx-swift`
fork. Converted safetensors carry `quant_method=turboquant` and
`linear_class=TurboQuantLinear` metadata plus schema-versioned TurboQuant
checkpoint metadata. `loadWeights` validates the metadata before replacing
matching linear layers with `TurboQuantLinear`. Missing metadata, newer schema
versions, newer layout versions, or profile-name mismatches fail with typed
``TurboQuantCheckpointMetadataValidationError`` values instead of loading a
guessed TurboQuant format.

## Scheme Aliases

``TurboQuantScheme/turbo4v2`` maps to the `.turbo4v2` runtime preset with
4-bit key magnitudes and 4-bit values. It is the balanced default profile
scheme. ``TurboQuantScheme/turbo3_5`` maps to mixed 3/4-bit key magnitudes and
4-bit values. ``TurboQuantScheme/turbo2_5`` is the most memory-oriented scheme
(2.5-bit key magnitudes, 2-bit values) and should be reserved for
memory-pressure profiles.

> Note: A former `turbo3` alias advertised 3.0-bit keys but always executed the
> 2.5-bit `turbo2_5` codec. It has been removed; the legacy `"turbo3"` string
> still decodes to `turbo2_5` so persisted configurations keep parsing.
