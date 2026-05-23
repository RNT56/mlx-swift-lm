# Agent Notes for the RNT56 MLX Swift LM Fork

This repository is the `RNT56/mlx-swift-lm` fork used by downstream projects. Treat it as an integrated fork layered on top of `RNT56/mlx-swift`.

## Branch Model

- `main` is the current integrated fork state for downstream consumers.
- Historical integration branches may point at the same commit as `main`. Treat those as compatibility refs, not as the canonical branch names for new work.
- Do not assume older topic branches are active. Most remote branches are historical experiments or upstream maintenance work.

As of 2026-05-18, active branches should contain this expected code baseline:

```text
f72c30d6744f1280c641c53f9e15c82a074b5a3a
Enable TurboQuant for shared and latent attention
```

Branch heads may be later docs-only or maintenance commits, but they should not drop this baseline unless the fork stack is intentionally rebuilt.

## Dependency Stack

The intended stack is:

```text
RNT56/mlx-swift-lm
  -> RNT56/mlx-swift
      -> RNT56/mlx
      -> RNT56/mlx-c
```

This repo owns the `RNT56/mlx-swift-lm -> RNT56/mlx-swift` link in `Package.swift`.

The expected `Package.swift` dependency is:

```swift
.package(
    url: "https://github.com/RNT56/mlx-swift",
    revision: "6820f3c6b85bdd73a288f5796ba78c4cd40efd91"
)
```

That `mlx-swift` revision points its MLX submodule at:

```text
RNT56/mlx:
8f13e02fa85252f2a569a43c6759f07490b816a5
Find SwiftPM metallib bundle near test binary

RNT56/mlx-c:
fff19671eed2e556bdf4552328a1791a8f37b651
Expose quantized SDPA C API
```

Do not downgrade this dependency to an older runtime pin when preserving downstream runtime behavior.

## Important Downstream Changes

The active branch must include the LM-side TurboQuant and robustness work, including:

- `9ecfb3f` Thread deterministic TurboQuant cache seeds
- `3beb127` Add raw-free rotating TurboQuant cache
- `e489fb1` Pin hardened MLX and preserve rotating fallback metadata
- `3f175ff` Route TurboQuant by verified device profile
- `f3479d9` Pin MLX runtime profile refinement
- `d245b7b` Complete TurboQuant KV cache integration
- `c2b5697` Implement rotating quantized KV cache
- `f72c30d` Enable TurboQuant for shared and latent attention
- `3f0a284` Add TurboQuant linear layer support
- `5db40d3` Add TurboQuant checkpoint conversion tooling
- `73e8cd9` Harden model config and runtime error paths
- `c402565` Complete VLM processor TODOs
- `b68f708` Add incomplete-marker audit and compile verification
- `36b527d` Document model compatibility requirements
- `4bb7cbc` Pin mlx-swift to Metal fallback revision

Before updating downstream pins, confirm the active branch contains these changes.

## Validation

Use focused checks before pushing:

```sh
git status --short --branch
swift package resolve
swift package describe --type text
swift build --target MLXLMCommon
```

For changes touching model execution, KV cache behavior, VLM processors, or config parsing, also run the most relevant tests available locally.

For changes touching TurboQuant profile policy, validate both the Swift registry
and the root JSON profile files:

```sh
swift test --filter TurboQuantProfileTests
```

`Package.resolved` is ignored in this repository. Updating it locally during `swift package resolve` is useful for validation, but it is not a tracked source-of-truth file unless the repository policy changes.

## Downstream Pin Workflow

When this repo gets a new commit that downstream apps should consume:

1. Confirm `Package.swift` still pins the intended `RNT56/mlx-swift` revision.
2. Push any compatibility refs only when they are intentionally meant to stay equivalent.
3. Update downstream app pins, especially `RNT56/pines`, to the matching `mlx-swift` and `mlx-swift-lm` commits.

For `pines`, `project.yml` is the source of truth for XcodeGen package pins. Regenerate/check `Pines.xcodeproj/project.pbxproj` after changing it.
