# Agent Notes for the Schtack MLX Swift LM Fork

This repository is the `RNT56/mlx-swift-lm` fork used by Schtack projects. Treat it as an integrated fork layered on top of `RNT56/mlx-swift`.

## Branch Model

- `main` is the current integrated fork state for downstream consumers.
- `schtack/turboquant-kv` intentionally points at the same commit as `main`. Keep this branch as a named integration branch for Schtack work.
- Do not assume older topic branches are active. Most remote branches are historical experiments or upstream maintenance work.

As of 2026-05-16, the expected active commit is:

```text
mlx-swift-lm main / schtack/turboquant-kv:
4bb7cbc6aafdf6abec4c34bf36f9e649444539f7
Pin mlx-swift to Metal fallback revision
```

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
    revision: "dd13c2b55a743473d458058e9d9fb028233065ec"
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

`Package.resolved` is ignored in this repository. Updating it locally during `swift package resolve` is useful for validation, but it is not a tracked source-of-truth file unless the repository policy changes.

## Downstream Pin Workflow

When this repo gets a new commit that downstream apps should consume:

1. Confirm `Package.swift` still pins the intended `RNT56/mlx-swift` revision.
2. Push both `main` and `schtack/turboquant-kv` if they are meant to stay equivalent.
3. Update downstream app pins, especially `RNT56/pines`, to the matching `mlx-swift` and `mlx-swift-lm` commits.

For `pines`, `project.yml` is the source of truth for XcodeGen package pins. Regenerate/check `Pines.xcodeproj/project.pbxproj` after changing it.
