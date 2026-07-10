# Architecture Ruling 2026-07-10: kernel home, dispatch tiers, and the promotion rule

Status: ACCEPTED (user-approved 2026-07-10). Binding for all future TurboQuant
work in this workspace. Summarized in the root `CLAUDE.md` ("Architecture
Ruling" section); this file carries the rationale and evidence.

## Context: how we got a two-path architecture

TurboQuant grew two parallel decode implementations:

- **The Swift path** (`tiledOnlineFused` / `gqa_block_partials*` kernels):
  the original R&D implementation. Metal kernel source lives as Swift strings
  in `mlx-swift/Source/MLX/TurboQuant.swift`, JIT-compiled via
  `MLXFast.metalKernel`, with hand-maintained byte-identical twins in mlx
  core's `turbo_quant_attention_jit.h`. All kernel R&D happened here (PolarQJL
  decode, coop coalescing, H16 diet, the `_rf1` race fixes) because iteration
  is fast: edit a string, rebuild in seconds.
- **The native C++ path** (`nativeMLXCompressed`, kernel kind 3): the
  productized implementation inside mlx core (`fast.cpp` / metal backend),
  exposed via mlx-c. Default-on (rollout gate `MLX_TURBOQUANT_NATIVE_ATTENTION`,
  default 1). Used by iPhone too.

Measured on a real model at the same context (Qwen3-0.6B-8bit @16K, gen=64,
engagement-verified; `artifacts/coop-realmodel-ab-20260706/`):

| route | decode tok/s |
|---|---|
| Swift strided | 4.67 |
| Swift + coop | 5.19 (+11.1% over strided) |
| **Native default** | **5.91** (+13.9% over coop'd Swift) |

## What the two-tier setup got wrong (the evidence)

1. **The tiers diverged into different kernels**, so lab wins do not transfer:
   coop's +11% (real, growing with context: +4.4% @8K -> +11.1% @16K) is a
   *hypothesis about a different implementation* until re-measured natively.
2. **Byte-identical twin maintenance** (Swift string + `jit.h` +
   `mlx-generated` mirror) was hand-synced and verified with `LC_ALL=C grep -a`
   — a recurring bug source (compiled-variant cache aliasing 2026-05-29,
   mirror-desync hazard in P1-1).
3. **The lab was benchmarked as production**: every pre-2026-07-06 coop number
   was a synthetic microbench or a Swift-path run misread as the shipping path
   (see `artifacts/coop-applicability-reframe-20260705/reframe.md` and the
   2026-07-06 benchmark-validation fix).
4. **Host dispatch economics dominate decode** at real sizes (T1.1 host
   de-serialization +240%; P1-1 post-mortem). The C++ primitive route
   integrates with MLX's lazy graph/donation/scheduler; the Swift custom-kernel
   route pays extra host overhead and sits outside that machinery.

## The ruling

1. **Kernel home = mlx core (C++/MSL).** Every kernel of record lives ONCE in
   `mlx/backend/metal/kernels/*`, consumed by both the AOT metallib and the JIT
   path. Metal optimizations land there, dispatched by C++ `fast::` primitives.
2. **mlx-c / mlx-swift = thin bindings**: ABI, capability probes, engagement
   telemetry. No kernels of record in Swift.
3. **mlx-swift-lm = policy only**: admission, cache lifecycle, speculative
   decode, profiles, quality gates, benchmark CLIs.
4. **Prototyping is allowed anywhere** (`MLXFast.metalKernel` remains a lab
   capability), but a prototype win is a research note, never a result.
5. **The promotion rule:** a kernel optimization is "done" only when it is
   measured on the NATIVE path, on a REAL model, through the
   engagement-verified harness (`TurboQuantInferenceParity`,
   `swiftDispatchedKernels` / native kernel kinds, promotion gate), on an awake
   AC-powered machine (`scripts/bench-caffeinated.sh`), with
   `--generate-tokens >= 32` (post-prefill transients dominate shorter runs;
   the 0.19-tok/s artifact of 2026-07-06 was a gen=4 measurement).

## Port / keep / develop / retire

- **PORT (gated):** coop's coalescing idea into the native compressed decode
  kernel — IF the native-kernel anatomy audit finds uncoalesced per-key loads.
  The audit decides port-or-park; do not port on faith.
- **KEEP:** native C++ compressed path as the only production decode route;
  affine K8/V4 native path (the Qwen3.5 product lever); the engagement-verified
  harness; `MLXFast.metalKernel` for prototyping; the Swift tiled path
  TEMPORARILY for its one production job — qLen>8 prefill chunks the native
  fused kernel does not cover.
- **DEVELOP:** all kernel-shaped work in mlx core; native prefill coverage for
  qLen>8 (removes the Swift path's last production role AND attacks the
  compressed-prefill slowness — the 33-minute 16K prefill ran through the
  fallback path); N7 speculative-decode promotion + admission policy in
  mlx-swift-lm.
- **RETIRE:** the kernel twins — once native prefill coverage lands, delete the
  `TurboQuant.swift` kernel copies and freeze `jit.h` as the single JIT source;
  regenerate `mlx-generated` in the build flow (never hand-mirror); delete the
  dead diagnostic kernel families (PolarWHT hybrid, Sparse-V) — off-path and
  carrying a known latent race-hazard class (Tier 3 sparse kernels share the
  v6 block-partials hazard fixed by `_rf1`).

## Active workstreams under this ruling (2026-07-10)

1. Native-kernel coalescing audit (decides the coop port). Read-only.
2. Affine K8/V4 SDPA occupancy — close the 0.67x gap on the shipping Qwen3.5
   path (deficit locus: SDPA decode occupancy + host dispatch around SDPA; the
   append ladder was falsified as the cause by P1-1).
3. N7 spec-decode promotion evidence through the fixed harness.
4. Owed clean re-runs (AC + caffeinated): native@16K/8K, 16K coop r2, 4B
   `_coop`, bit-exactness, 32K pair.
5. Native qLen>8 prefill coverage, then twin retirement.

## Consequences

- Lab numbers can never again masquerade as product evidence (harness guards +
  this ruling).
- One kernel source of truth ends the twin-sync failure class.
- Swift-side kernel work stops accruing except where it feeds a native port.
- The A-series device run remains the gate for all device/product claims.
