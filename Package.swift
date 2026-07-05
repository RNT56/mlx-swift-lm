// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import CompilerPluginSupport
import PackageDescription

let package = Package(
    name: "mlx-swift-lm",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "MLXLLM",
            targets: ["MLXLLM"]),
        .library(
            name: "MLXVLM",
            targets: ["MLXVLM"]),
        .library(
            name: "MLXLMCommon",
            targets: ["MLXLMCommon"]),
        .library(
            name: "MLXEmbedders",
            targets: ["MLXEmbedders"]),
        .library(
            name: "MLXHuggingFace",
            targets: ["MLXHuggingFace"]),
        .library(
            name: "BenchmarkHelpers",
            targets: ["BenchmarkHelpers"]),
        .library(
            name: "IntegrationTestHelpers",
            targets: ["IntegrationTestHelpers"]),
        .library(
            name: "TurboQuantBench",
            targets: ["TurboQuantBench"]),
        .executable(
            name: "TurboQuantModelBenchmark",
            targets: ["TurboQuantModelBenchmark"]),
        .executable(
            name: "TurboQuantQwenProof",
            targets: ["TurboQuantQwenProof"]),
        .executable(
            name: "TurboQuantInferenceParity",
            targets: ["TurboQuantInferenceParity"]),
        .executable(
            name: "TurboQuantNativeVxBenchmark",
            targets: ["TurboQuantNativeVxBenchmark"]),
        .executable(
            name: "TurboQuantAcceptanceHarness",
            targets: ["TurboQuantAcceptanceHarness"]),
        .executable(
            name: "TurboQuantHeadBenchmark",
            targets: ["TurboQuantHeadBenchmark"]),
        .executable(
            name: "TurboQuantCacheUpdateBenchmark",
            targets: ["TurboQuantCacheUpdateBenchmark"]),
        .executable(
            name: "TurboQuantLowerV2Calibrate",
            targets: ["TurboQuantLowerV2Calibrate"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/RNT56/mlx-swift",
            revision: "16b68ff62c3dbb6d229b0bddf88c1def72a42d6b"),
        // 602.0.0 floor: swift.org publishes signed prebuilt swift-syntax artifacts only for
        // >= 602 tags on current toolchains; a 600.x/601.x resolution falls back to the full
        // source compile of swift-syntax.
        .package(url: "https://github.com/swiftlang/swift-syntax.git", "602.0.0" ..< "604.0.0"),
    ],
    targets: [
        .target(
            name: "MLXLLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXVLM",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXVLM",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXLMCommon",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            path: "Libraries/MLXLMCommon",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "MLXEmbedders",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .target(name: "MLXLMCommon"),
            ],
            path: "Libraries/MLXEmbedders",
            exclude: [
                "README.md"
            ]
        ),
        .target(
            name: "BenchmarkHelpers",
            dependencies: [
                "MLXLMCommon",
                "IntegrationTestHelpers",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/BenchmarkHelpers"
        ),
        .target(
            name: "IntegrationTestHelpers",
            dependencies: [
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/IntegrationTestHelpers",
            exclude: ["README.md"]
        ),
        .target(
            name: "TurboQuantBench",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Libraries/TurboQuantBench"
        ),
        .executableTarget(
            name: "TurboQuantModelBenchmark",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantModelBenchmark"
        ),
        .executableTarget(
            name: "TurboQuantQwenProof",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantQwenProof"
        ),
        .executableTarget(
            name: "TurboQuantInferenceParity",
            dependencies: [
                "IntegrationTestHelpers",
                "MLXLLM",
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantInferenceParity"
        ),
        .executableTarget(
            name: "TurboQuantNativeVxBenchmark",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantNativeVxBenchmark"
        ),
        .executableTarget(
            name: "TurboQuantAcceptanceHarness",
            dependencies: [
                "MLXLLM",
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantAcceptanceHarness"
        ),
        .executableTarget(
            name: "TurboQuantHeadBenchmark",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantHeadBenchmark"
        ),
        .executableTarget(
            name: "TurboQuantCacheUpdateBenchmark",
            dependencies: [
                "MLXLMCommon",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "tools/TurboQuantCacheUpdateBenchmark"
        ),
        .executableTarget(
            name: "TurboQuantLowerV2Calibrate",
            dependencies: [
                "MLXLMCommon"
            ],
            path: "tools/TurboQuantLowerV2Calibrate"
        ),
        .testTarget(
            name: "MLXLMTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                "MLXLMCommon",
                "MLXLLM",
                "MLXVLM",
                "MLXEmbedders",
                "IntegrationTestHelpers",
                "TurboQuantBench",
            ],
            path: "Tests/MLXLMTests",
            exclude: [
                "README.md"
            ],
            resources: [.process("Resources/1080p_30.mov"), .process("Resources/audio_only.mov")]
        ),
        .macro(
            name: "MLXHuggingFaceMacros",
            dependencies: [
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax"),
            ],
            path: "Libraries/MLXHuggingFaceMacros"
        ),
        .target(
            name: "MLXHuggingFace",
            dependencies: [
                "MLXHuggingFaceMacros",
                "MLXLMCommon",
            ],
            path: "Libraries/MLXHuggingFace"
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
