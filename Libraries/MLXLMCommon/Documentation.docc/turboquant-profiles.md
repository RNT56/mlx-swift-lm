# TurboQuant Profiles

TurboQuant profiles describe a safe runtime KV-cache policy for ordinary MLX model
weights. They are not calibrated model files and they do not replace Hugging Face
or MLX weight artifacts.

Use a profile when an application wants a model-aware default for
``GenerateParameters``:

```swift
let parameters = GenerateParameters(
    turboQuantModelID: "mlx-community/Qwen3-4B-4bit",
    keyHeadDimension: 128,
    valueHeadDimension: 128
)
```

If a bundled profile matches, the initializer enables
``KVCacheStrategy/turboQuant`` and fills in the preset, value bits, group size,
backend, and optimization policy. If no profile matches, the supplied base
parameters are returned unchanged.

Profile matching is intentionally conservative. Model ID patterns select
candidates, but config metadata remains authoritative: a candidate with an
`architecture` or `model_type` requirement should be rejected when the model's
`config.json` declares a different family. Matching model sizes should use
explicit `B` or `M` size tokens only; quantization labels such as `4bit` and
`8bit` are not evidence that a model is 4B or 8B. Resolution APIs that expose diagnostics should
include rejection reasons for metadata, size, shape, mask, and context failures.

The root `TurboQuantProfiles/` directory contains JSON copies of the bundled
profiles for downstream apps, release tooling, and manual inspection. The Swift
registry lives in ``TurboQuantProfileRegistry`` so library consumers do not need
package resources at runtime.

Bundled Qwen3.5/Qwen3.6 profiles cover current MLX Community dense and MoE
families with config-backed 256-dimensional KV head checks: `qwen3.5-0.8b`,
`qwen3.5-2b`, `qwen3.5-4b`, `qwen3.5-9b`, `qwen3.5-27b`, `qwen3.6-27b`,
`qwen3.5-40b`, `qwen3.6-40b`, `qwen3.5-35b-a3b`, `qwen3.6-35b-a3b`,
`qwen3.5-97b-a10b`, `qwen3.5-122b-a10b`, and `qwen3.5-397b-a17b`.
Qwen3.7 is intentionally not profiled until open MLX weights exist.

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

Bundled Llama profiles cover config-backed text-only Llama families and small
Llama-compatible dense derivatives: `llama-2-7b`, `llama-2-13b`,
`llama-2-70b`, `llama-3-3b`, `llama-3-8b`, `llama-3-70b`,
`llama-3.1-4b`, `llama-3.1-8b`, `llama-3.1-16b`, `llama-3.1-70b`,
`llama-3.1-120b`, `llama-3.1-405b`, `llama-3.2-1b`, `llama-3.2-3b`,
`llama-3.3-3b`, `llama-3.3-70b`, `llama-compatible-135m`,
`llama-compatible-160m`, `llama-compatible-1b`, `llama-compatible-2b`,
`llama-compatible-3b`, and `llama-compatible-4b`. These profiles require
`model_type=llama` and explicit KV head dimensions. Llama 3.2 Vision, MLLama,
Llama 4, LLava, Bunny, Idefics, Mixtral, and Mamba-Codestral are intentionally
excluded until the linked runtime exposes the corresponding supported model
surface.

Bundled Mistral profiles cover config-backed dense text, nested Mistral3 text,
Mistral4 MoE, and Pixtral VLM variants: `mistral-7b`, `mistral-nemo-12b`,
`ministral-8b-2410`, `codestral-22b`, `mistral-small-22b`,
`mistral-small-24b`, `devstral-small-24b`, `magistral-small-24b`,
`ministral3-3b`, `ministral3-8b`, `ministral3-14b`,
`mistral-small-3.1-24b`, `mistral-small-3.2-24b`,
`devstral-small-2-24b`, `mistral-medium-3.5-128b`,
`mistral-small-4-119b-a6b`, and `pixtral-12b`. Mistral3 and Pixtral profiles
require nested `text_config.model_type` metadata. Mistral Small 4 additionally
requires routed expert metadata matching its MoE config. Mixtral and
Mamba-Codestral remain unsupported until runtime model support exists.

## Safety Contract

Profiles are advisory. The runtime still enforces:

- Metal availability and bundled metallib loading.
- TurboQuant runtime self-tests.
- Query/key/value shape compatibility.
- Mask support for the selected attention path.
- Fallback to MLX packed quantization when the Metal path is unavailable.

Measured fields such as perplexity delta, token throughput, and memory footprint
remain optional and should stay empty until reproduced on the target hardware and
model revision.

## Converted Weights

TurboQuant profile selection is independent of converted model weights. For
weight compression, use `swift run TurboQuantConverter` from the `mlx-swift`
fork. Converted safetensors carry `quant_method=turboquant` and
`linear_class=TurboQuantLinear` metadata; `loadWeights` uses that metadata to
replace matching linear layers with `TurboQuantLinear` before applying the packed
weight, scale, and bias tensors.

## Scheme Aliases

``TurboQuantScheme/turbo4v2`` maps to the `.turbo4v2` runtime preset with
4-bit keys and values. It is the balanced default profile scheme. ``TurboQuantScheme/turbo3``
maps to the more memory-oriented `.turbo2_5` runtime preset and should be reserved
for memory-pressure profiles.
