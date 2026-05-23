import Foundation
import Testing

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    @Suite
    struct TurboQuantProfileTests {

        @Test func testSchemeAliasesDecodeExternalNames() {
            #expect(TurboQuantScheme(normalizing: "turbo4v2") == .turbo4v2)
            #expect(TurboQuantScheme(normalizing: "turbo-4-v2") == .turbo4v2)
            #expect(TurboQuantScheme(normalizing: "turbo3.5") == .turbo3_5)
            #expect(TurboQuantScheme(normalizing: "turbo3") == .turbo3)
            #expect(TurboQuantScheme(normalizing: "off") == .disabled)
            #expect(TurboQuantScheme(normalizing: "unknown") == nil)
        }

        @Test func testBundledRegistryMatchesKnownModelIDs() throws {
            let registry = TurboQuantProfileRegistry.bundled

            let qwen = try #require(
                registry.profile(
                    for: "mlx-community/Qwen3-4B-4bit",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(qwen.id == "qwen3-4b")
            #expect(qwen.recommendedScheme == .turbo4v2)

            let llama = try #require(
                registry.profile(
                    for: "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(llama.id == "llama-3.1-8b")

            let unsupported = registry.profile(
                for: "mlx-community/Qwen3-4B-4bit",
                keyHeadDimension: 512,
                valueHeadDimension: 512
            )
            #expect(unsupported == nil)
        }

        @Test func testQuantizationBitSuffixDoesNotImplyModelSize() throws {
            #expect(TurboQuantModelDescriptor.inferParameterCountB(from: "Qwen3-4bit") == nil)
            #expect(
                TurboQuantModelDescriptor.inferParameterCountB(
                    from: "PaliGemma-mix-448-8bit"
                ) == nil
            )
            #expect(
                TurboQuantModelDescriptor.inferParameterCountB(
                    from: "mlx-community/Qwen3-4B-4bit"
                ) == 4
            )
            let smolLMSize = try #require(
                TurboQuantModelDescriptor.inferParameterCountB(
                    from: "mlx-community/SmolLM2-135M-Instruct"
                )
            )
            #expect(abs(smolLMSize - 0.135) < 0.0001)

            let registry = TurboQuantProfileRegistry(
                profiles: [
                    TurboQuantProfile(
                        id: "qwen3-4b",
                        modelPatterns: ["*qwen3*4b*", "*qwen-3*4b*"],
                        architecture: "qwen3",
                        minParametersB: 3.5,
                        maxParametersB: 4.5,
                        supportedKeyHeadDimensions: [128]
                    ),
                    TurboQuantProfile(
                        id: "paligemma-3b",
                        modelPatterns: ["*paligemma*3b*"],
                        architecture: "paligemma",
                        minParametersB: 2.5,
                        maxParametersB: 3.5,
                        supportedKeyHeadDimensions: [128]
                    ),
                ]
            )

            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3-4bit",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/PaliGemma-mix-448-8bit",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
        }

        @Test func testModelTypeGatesProfileApplicationAndReportsDiagnostics() throws {
            let registry = TurboQuantProfileRegistry(
                profiles: [
                    TurboQuantProfile(
                        id: "qwen3-4b",
                        modelPatterns: ["*qwen3*4b*"],
                        architecture: "qwen3",
                        modelTypes: ["qwen3"],
                        minParametersB: 3.5,
                        maxParametersB: 4.5,
                        supportedKeyHeadDimensions: [128]
                    )
                ]
            )

            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3-4B-4bit",
                    modelType: "gemma3",
                    parameterCountB: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )

            let selection = registry.selection(
                for: TurboQuantModelDescriptor(
                    modelID: "mlx-community/Qwen3-4B-4bit",
                    modelType: "gemma3",
                    parameterCountB: 4
                ),
                keyHeadDimension: 512,
                valueHeadDimension: 128
            )

            #expect(selection.profile == nil)
            #expect(
                selection.rejectionReasons.contains {
                    $0.contains("model type 'gemma3' is not supported")
                }
            )
            #expect(
                selection.rejectionReasons.contains {
                    $0.contains("key head dimension 512 is not supported")
                }
            )

            let accepted = try #require(
                registry.profile(
                    for: "mlx-community/Qwen3-4B-4bit",
                    modelType: "qwen3",
                    parameterCountB: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(accepted.id == "qwen3-4b")
        }

        @Test func testKnownVisionLanguageRepositoriesDoNotUseTextOnlyProfilesByName() {
            let registry = TurboQuantProfileRegistry.bundled

            #expect(
                registry.profile(
                    for: "mlx-community/llava-llama-3-8b-v1_1-4bit",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Bunny-Llama-3-8B-V-4bit",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/llava-phi-3-mini-4bit",
                    keyHeadDimension: 96,
                    valueHeadDimension: 96
                ) == nil
            )
        }

        @Test func testGemma3nProfileIsNotShadowedByGemma34B() throws {
            let profile = try #require(
                TurboQuantProfileRegistry.bundled.profile(
                    for: "mlx-community/gemma-3n-E4B-it-lm-4bit",
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                )
            )

            #expect(profile.id == "gemma-3n")
            #expect(profile.architecture == "gemma3n")
        }

        @Test func testQwen35TwoBUses256HeadDimensions() throws {
            let profile = try #require(
                TurboQuantProfileRegistry.bundled.profile(
                    for: "mlx-community/Qwen3.5-2B-4bit",
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                )
            )

            #expect(profile.id == "qwen3.5-2b")
            #expect(profile.supports(keyHeadDimension: 256, valueHeadDimension: 256))
            #expect(!profile.supports(keyHeadDimension: 128, valueHeadDimension: 128))
        }

        @Test func testExpandedSmallModelProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled

            let qwen06 = try #require(
                registry.profile(
                    for: "mlx-community/Qwen3-0.6B-4bit",
                    modelType: "qwen3",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(qwen06.id == "qwen3-0.6b")
            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3-0.6B-4bit",
                    modelType: "qwen3",
                    keyHeadDimension: 64,
                    valueHeadDimension: 64
                ) == nil
            )

            let phi4Mini = try #require(
                registry.profile(
                    for: "mlx-community/Phi-4-mini-instruct-4bit",
                    modelType: "phi3",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(phi4Mini.id == "phi-4-mini")

            let smolLM = try #require(
                registry.profile(
                    for: "mlx-community/SmolLM2-135M-Instruct",
                    modelType: "llama",
                    keyHeadDimension: 64,
                    valueHeadDimension: 64
                )
            )
            #expect(smolLM.id == "smollm-small")

            let qwen25Coder = try #require(
                registry.profile(
                    for: "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
                    modelType: "qwen2",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(qwen25Coder.id == "qwen2.5-small")
        }

        @Test func testProfileAppliesGenerateParameters() {
            let parameters = GenerateParameters(
                turboQuantModelID: "mlx-community/Qwen3-4B-4bit",
                keyHeadDimension: 128,
                valueHeadDimension: 128
            )

            #expect(parameters.kvCacheStrategy == .turboQuant)
            #expect(parameters.kvGroupSize == 64)
            #expect(parameters.turboQuantPreset == .turbo4v2)
            #expect(parameters.turboQuantBackend == .metalPolarQJL)
            #expect(parameters.turboQuantOptimizationPolicy == .auto)
            #expect(parameters.turboQuantValueBits == 4)
        }

        @Test func testMemoryProfileMapsToAggressiveRuntimePreset() {
            let profile = TurboQuantProfile(
                id: "memory-test",
                modelPatterns: ["*memory-test*"],
                supportedKeyHeadDimensions: [128],
                recommendedScheme: .turbo3,
                fallbackScheme: .turbo4v2,
                keyBits: 2.5,
                valueBits: 2,
                qualityProfile: .memory,
                optimizationPolicy: .auto
            )

            let parameters = GenerateParameters(turboQuantProfile: profile)
            #expect(parameters.kvCacheStrategy == .turboQuant)
            #expect(parameters.turboQuantPreset == .turbo2_5)
            #expect(parameters.turboQuantOptimizationPolicy == .preferMemory)
            #expect(parameters.turboQuantValueBits == 2)
        }

        @Test func testRootJSONProfilesDecodeAndMatchBundledIDs() throws {
            let profileDirectory = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("TurboQuantProfiles")

            let profiles = try TurboQuantProfileRegistry.loadJSONProfiles(from: profileDirectory)
            #expect(profiles.count == Set(profiles.map(\.id)).count)
            #expect(
                TurboQuantProfileRegistry.bundled.profiles.count
                    == Set(TurboQuantProfileRegistry.bundled.profiles.map(\.id)).count
            )
            #expect(
                Set(profiles.map(\.id)) == Set(TurboQuantProfileRegistry.bundled.profiles.map(\.id))
            )

            for jsonProfile in profiles {
                let bundledProfile = try #require(
                    TurboQuantProfileRegistry.bundled.profiles.first { $0.id == jsonProfile.id }
                )
                #expect(bundledProfile.schemaVersion == jsonProfile.schemaVersion)
                #expect(bundledProfile.exactModelIDs == jsonProfile.exactModelIDs)
                #expect(bundledProfile.modelPatterns == jsonProfile.modelPatterns)
                #expect(bundledProfile.includePatterns == jsonProfile.includePatterns)
                #expect(bundledProfile.excludePatterns == jsonProfile.excludePatterns)
                #expect(bundledProfile.architecture == jsonProfile.architecture)
                #expect(bundledProfile.modelTypes == jsonProfile.modelTypes)
                #expect(bundledProfile.modalities == jsonProfile.modalities)
                #expect(bundledProfile.minParametersB == jsonProfile.minParametersB)
                #expect(bundledProfile.maxParametersB == jsonProfile.maxParametersB)
                #expect(
                    bundledProfile.supportedKeyHeadDimensions
                        == jsonProfile.supportedKeyHeadDimensions
                )
                #expect(
                    bundledProfile.supportedValueHeadDimensions
                        == jsonProfile.supportedValueHeadDimensions
                )
                #expect(bundledProfile.supportedContextLengths == jsonProfile.supportedContextLengths)
                #expect(bundledProfile.safeContextLength == jsonProfile.safeContextLength)
                #expect(bundledProfile.status == jsonProfile.status)
                #expect(bundledProfile.source == jsonProfile.source)
                #expect(bundledProfile.confidence == jsonProfile.confidence)
            }

            let registry = TurboQuantProfileRegistry(profiles: profiles)
            let glm = try #require(
                registry.profile(
                    for: "RNT56/GLM4-MoE-Lite-4bit",
                    keyHeadDimension: 96,
                    valueHeadDimension: 64
                )
            )
            #expect(glm.id == "glm4-moe-lite")
            #expect(glm.supports(keyHeadDimension: 96, valueHeadDimension: 64))
            #expect(!glm.supports(keyHeadDimension: 96, valueHeadDimension: 96))
        }
    }
}
