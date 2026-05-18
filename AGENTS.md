# Agent Notes for the Schtack MLX Swift LM Fork

This repository is the `RNT56/mlx-swift-lm` fork used by Schtack projects. Treat it as an integrated fork layered on top of `RNT56/mlx-swift`.

## Branch Model

- `main` is the current integrated fork state for downstream consumers.
- `schtack/turboquant-kv` intentionally points at the same commit as `main`. Keep this branch as a named integration branch for Schtack work.
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
```

This repo owns the `RNT56/mlx-swift-lm -> RNT56/mlx-swift` link in `Package.swift`.

The expected `Package.swift` dependency is:

```swift
.package(
    url: "https://github.com/RNT56/mlx-swift",
    revision: "5db40d34a96a9c6889b6583d6cc09f8b8f05ea5e"
)
```

That `mlx-swift` revision points its MLX submodule at:

```text
RNT56/mlx:
f2ed827ef3c51ba7e5a0f7936fcb7c5cfcedb4e6
Add embedded Metal default library fallback
```

Do not downgrade this dependency to the older `cf6d72f54e8619e52a746b88a0fb00f172e4ba10` pin when preserving Schtack runtime behavior.

## Important Schtack Changes

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
2. Push both `main` and `schtack/turboquant-kv` if they are meant to stay equivalent.
3. Update downstream app pins, especially `RNT56/pines`, to the matching `mlx-swift` and `mlx-swift-lm` commits.

For `pines`, `project.yml` is the source of truth for XcodeGen package pins. Regenerate/check `Pines.xcodeproj/project.pbxproj` after changing it.
