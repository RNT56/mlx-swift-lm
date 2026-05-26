import Foundation
import Testing

@testable import MLXLMCommon

extension MLXRuntimeSwiftTests {

    @Suite
    struct TurboQuantProfileTests {

        @Test func testSchemeAliasesDecodeExternalNames() {
            #expect(TurboQuantScheme(normalizing: "turbo8") == .turbo8)
            #expect(TurboQuantScheme(normalizing: "turbo-8") == .turbo8)
            #expect(TurboQuantScheme(normalizing: "turbo4v2") == .turbo4v2)
            #expect(TurboQuantScheme(normalizing: "turbo-4-v2") == .turbo4v2)
            #expect(TurboQuantScheme(normalizing: "turbo3.5") == .turbo3_5)
            #expect(TurboQuantScheme(normalizing: "turbo3") == .turbo3)
            #expect(TurboQuantScheme(normalizing: "off") == .disabled)
            #expect(TurboQuantScheme(normalizing: "unknown") == nil)
        }

        @Test func testProfileStatusSeparatesGuardedAndCertifiedStates() {
            #expect(TurboQuantProfileStatus.guarded.isActive)
            #expect(TurboQuantProfileStatus.guarded.requiresGuardedFallbackDisclosure)
            #expect(!TurboQuantProfileStatus.guarded.isDeviceEvidenceBacked)
            #expect(TurboQuantProfileStatus.verified.isDeviceEvidenceBacked)
            #expect(TurboQuantProfileStatus.certified.isDeviceEvidenceBacked)
            #expect(!TurboQuantProfileStatus.deprecated.isActive)
        }

        @Test func testBundledRegistryMatchesKnownModelIDs() throws {
            let registry = TurboQuantProfileRegistry.bundled

            let qwen = try #require(
                registry.profile(
                    for: "mlx-community/Qwen3-4B-4bit",
                    modelType: "qwen3",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(qwen.id == "qwen3-4b")
            #expect(qwen.recommendedScheme == .turbo4v2)

            let llama = try #require(
                registry.profile(
                    for: "mlx-community/Llama-3.1-8B-Instruct-4bit",
                    modelType: "llama",
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(llama.id == "llama-3.1-8b")

            let unsupported = registry.profile(
                for: "mlx-community/Qwen3-4B-4bit",
                modelType: "qwen3",
                keyHeadDimension: 512,
                valueHeadDimension: 512
            )
            #expect(unsupported == nil)
        }

        @Test func testBundledRegistryAppliesPerFamilyOptimizationMetadata() throws {
            let registry = TurboQuantProfileRegistry.bundled

            let qwen35 = try #require(
                registry.profile(
                    for: "mlx-community/Qwen3.5-4B-4bit",
                    modelType: "qwen3_5",
                    textConfigModelType: "qwen3_5_text",
                    modality: .visionText,
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256,
                    contextLength: 262144
                )
            )
            #expect(qwen35.id == "qwen3.5-4b")
            #expect(qwen35.safeContextLength == 262144)
            #expect(qwen35.recommendedScheme == .turbo8)
            #expect(qwen35.optimizationPolicy == TurboQuantOptimizationPolicy.auto)
            #expect(qwen35.valueBits == 8)
            #expect(qwen35.status == .guarded)

            let qwen35OptiQ2B = try #require(
                registry.profile(
                    for: "mlx-community/Qwen3.5-2B-OptiQ-4bit",
                    modelType: "qwen3_5",
                    parameterCountB: 2,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256,
                    contextLength: 24576
                )
            )
            #expect(qwen35OptiQ2B.id == "qwen3.5-2b")
            #expect(qwen35OptiQ2B.recommendedScheme == .turbo8)
            #expect(
                qwen35OptiQ2B.optimizationPolicy == TurboQuantOptimizationPolicy.auto
            )
            #expect(qwen35OptiQ2B.valueBits == 8)
            #expect(qwen35OptiQ2B.status == .guarded)

            let gemma3Small = try #require(
                registry.profile(
                    for: "mlx-community/gemma-3-270m-it-4bit",
                    modelType: "gemma3_text",
                    parameterCountB: 0.27,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256,
                    contextLength: 32768
                )
            )
            #expect(gemma3Small.id == "gemma-3-270m")
            #expect(gemma3Small.safeContextLength == 32768)
            #expect(
                gemma3Small.optimizationPolicy == TurboQuantOptimizationPolicy.conservative
            )
            let gemma31B = try #require(
                registry.profile(
                    for: "mlx-community/gemma-3-1b-it-4bit",
                    modelType: "gemma3_text",
                    parameterCountB: 1,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256,
                    contextLength: 24576
                )
            )
            #expect(gemma31B.id == "gemma-3-1b")
            #expect(gemma31B.recommendedScheme == .turbo8)
            #expect(gemma31B.optimizationPolicy == TurboQuantOptimizationPolicy.auto)
            #expect(gemma31B.valueBits == 8)
            #expect(gemma31B.status == .guarded)
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-270m-it-4bit",
                    modelType: "gemma3_text",
                    parameterCountB: 0.27,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256,
                    contextLength: 65536
                ) == nil
            )

            let llama33 = try #require(
                registry.profile(
                    for: "mlx-community/Llama-3.3-70B-Instruct-4bit",
                    modelType: "llama",
                    parameterCountB: 70,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128,
                    contextLength: 131072
                )
            )
            #expect(llama33.id == "llama-3.3-70b")
            #expect(llama33.optimizationPolicy == TurboQuantOptimizationPolicy.conservative)

            let llama32_3B = try #require(
                registry.profile(
                    for: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    modelType: "llama",
                    parameterCountB: 3,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128,
                    contextLength: 131072
                )
            )
            #expect(llama32_3B.id == "llama-3.2-3b")
            #expect(llama32_3B.recommendedScheme == .turbo8)
            #expect(llama32_3B.optimizationPolicy == TurboQuantOptimizationPolicy.auto)
            #expect(llama32_3B.valueBits == 8)
            #expect(llama32_3B.status == .guarded)

            let mistral4 = try #require(
                registry.profile(
                    for: "mlx-community/Mistral-Small-4-119B-Instruct-2603-4bit",
                    modelType: "mistral3",
                    textConfigModelType: "mistral4",
                    parameterCountB: 119,
                    routedExperts: 128,
                    expertsPerToken: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128,
                    contextLength: 1_048_576
                )
            )
            #expect(mistral4.id == "mistral-small-4-119b-a6b")
            #expect(mistral4.safeContextLength == 1_048_576)
            #expect(mistral4.optimizationPolicy == TurboQuantOptimizationPolicy.conservative)
        }

        @Test func testBundledProfilesRequireStrictModelAndHeadMetadata() throws {
            let bundledProfiles = TurboQuantProfileRegistry.bundled.profiles
            #expect(!bundledProfiles.isEmpty)
            #expect(bundledProfiles.allSatisfy { $0.requiresModelType })
            #expect(bundledProfiles.allSatisfy { $0.requiresHeadDimensions })

            let profileDirectory = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("TurboQuantProfiles")
            let jsonProfiles = try TurboQuantProfileRegistry.loadJSONProfiles(
                from: profileDirectory)
            #expect(jsonProfiles.allSatisfy { $0.requiresModelType })
            #expect(jsonProfiles.allSatisfy { $0.requiresHeadDimensions })
        }

        @Test func testLegacySmallProfilesFailClosedForMissingMetadata() throws {
            let registry = TurboQuantProfileRegistry.bundled

            let strictCases:
                [(
                    modelID: String,
                    modelType: String,
                    keyHeadDimension: Int,
                    valueHeadDimension: Int,
                    expectedProfileID: String
                )] = [
                    ("mlx-community/Qwen3-4B-4bit", "qwen3", 128, 128, "qwen3-4b"),
                    ("mlx-community/Phi-4-mini-instruct-4bit", "phi3", 128, 128, "phi-4-mini"),
                    ("mlx-community/SmolLM3-3B-4bit", "llama", 128, 128, "smollm3-3b"),
                    (
                        "mlx-community/Granite-3.3-8B-Instruct-4bit", "granite", 128, 128,
                        "granite-small"
                    ),
                    ("mlx-community/LFM2-1.2B-4bit", "lfm2", 128, 128, "lfm2-small"),
                    ("mlx-community/GLM-4.7-Flash-4bit", "glm4_moe_lite", 96, 64, "glm4-moe-lite"),
                ]

            for profileCase in strictCases {
                #expect(
                    registry.profile(
                        for: profileCase.modelID,
                        keyHeadDimension: profileCase.keyHeadDimension,
                        valueHeadDimension: profileCase.valueHeadDimension
                    ) == nil
                )
                #expect(
                    registry.profile(
                        for: profileCase.modelID,
                        modelType: profileCase.modelType
                    ) == nil
                )

                let matched = try #require(
                    registry.profile(
                        for: profileCase.modelID,
                        modelType: profileCase.modelType,
                        keyHeadDimension: profileCase.keyHeadDimension,
                        valueHeadDimension: profileCase.valueHeadDimension
                    )
                )
                #expect(matched.id == profileCase.expectedProfileID)
                #expect(
                    matched.supports(
                        keyHeadDimension: profileCase.keyHeadDimension,
                        valueHeadDimension: profileCase.valueHeadDimension
                    ))
                #expect(!matched.supports())
            }
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
                    modelType: "gemma3n",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                )
            )

            #expect(profile.id == "gemma-3n-e4b")
            #expect(profile.architecture == "gemma3n")
        }

        @Test func testGemma2ProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, Double, Int, TurboQuantOptimizationPolicy)] = [
                ("mlx-community/gemma-2-2b-it-4bit", "gemma-2-2b", 2, 256, .conservative),
                ("mlx-community/gemma-2-baku-2b-it-4bit", "gemma-2-2b", 2, 256, .conservative),
                ("mlx-community/gemma-2-9b-it-4bit", "gemma-2-9b", 9, 256, .conservative),
                (
                    "mlx-community/Gemma-SEA-LION-v3-9B-IT-mlx-4bit", "gemma-2-9b", 9, 256,
                    .conservative
                ),
                ("mlx-community/gemma-2-27b-it-4bit", "gemma-2-27b", 27, 128, .conservative),
                (
                    "mlx-community/TheDrummer_Big-Tiger-Gemma-27B-v1_4bit", "gemma-2-27b", 27,
                    128, .conservative
                ),
            ]

            for (modelID, expectedProfileID, parameterCountB, headDimension, optimizationPolicy)
                in examples
            {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: "gemma2",
                        parameterCountB: parameterCountB,
                        keyHeadDimension: headDimension,
                        valueHeadDimension: headDimension
                    )
                )
                #expect(profile.id == expectedProfileID)
                #expect(profile.optimizationPolicy == optimizationPolicy)
            }
        }

        @Test func testGemma3ProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, String, Double, Int)] = [
                ("mlx-community/gemma-3-270m-it-4bit", "gemma-3-270m", "gemma3_text", 0.27, 256),
                ("mlx-community/gemma-3-1b-it-4bit", "gemma-3-1b", "gemma3_text", 1, 256),
                ("mlx-community/swahili-gemma-1b-mlx-fp16", "gemma-3-1b", "gemma3_text", 1, 256),
                ("mlx-community/gemma-3-4b-it-4bit", "gemma-3-4b", "gemma3", 4, 256),
                ("mlx-community/gemma-3-4b-it-qat-4bit", "gemma-3-4b", "gemma3", 4, 256),
                ("mlx-community/gemma-3-text-4b-it-4bit", "gemma-3-4b", "gemma3", 4, 256),
                ("mlx-community/Gemma-SEA-LION-v4-4B-VL-mlx-3bit", "gemma-3-4b", "gemma3", 4, 256),
                (
                    "mlx-community/gemma-3-text-4b-320-head-test", "gemma-3-4b", "gemma3_text", 4,
                    320
                ),
                ("mlx-community/gemma-3-12b-it-4bit", "gemma-3-12b", "gemma3", 12, 256),
                ("mlx-community/gemma-3-12b-it-qat-4bit", "gemma-3-12b", "gemma3", 12, 256),
                (
                    "mlx-community/gemma-3-12b-explicit-240-test", "gemma-3-12b", "gemma3_text", 12,
                    240
                ),
                ("mlx-community/gemma-3-27b-it-4bit", "gemma-3-27b", "gemma3", 27, 128),
                (
                    "mlx-community/Gemma-SEA-LION-v4-27B-IT-mlx-4bit", "gemma-3-27b", "gemma3", 27,
                    128
                ),
            ]

            for (modelID, expectedProfileID, modelType, parameterCountB, headDimension) in examples
            {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: modelType,
                        parameterCountB: parameterCountB,
                        keyHeadDimension: headDimension,
                        valueHeadDimension: headDimension
                    )
                )
                #expect(profile.id == expectedProfileID)
            }
        }

        @Test func testLegacyGemmaProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, Double)] = [
                ("mlx-community/gemma-1.1-2b-it-4bit", "gemma-2b", 2),
                ("mlx-community/gemma-1.1-7b-it-4bit", "gemma-7b", 7),
                ("mlx-community/zephyr-7b-gemma-v0.1-4bit", "gemma-7b", 7),
            ]

            for (modelID, expectedProfileID, parameterCountB) in examples {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: "gemma",
                        parameterCountB: parameterCountB,
                        keyHeadDimension: 256,
                        valueHeadDimension: 256
                    )
                )
                #expect(profile.id == expectedProfileID)
            }
        }

        @Test func testGemma3nAndGemma4ProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, String, Double)] = [
                ("mlx-community/gemma-3n-E2B-it-lm-4bit", "gemma-3n-e2b", "gemma3n", 2),
                ("mlx-community/gemma-3n-E2B-it-4bit", "gemma-3n-e2b", "gemma3n", 4),
                ("mlx-community/gemma-3n-E4B-it-lm-4bit", "gemma-3n-e4b", "gemma3n", 4),
                (
                    "mlx-community/Huihui-gemma-3n-E4B-it-abliterated-lm-4bit",
                    "gemma-3n-e4b",
                    "gemma3n",
                    4
                ),
                ("mlx-community/gemma-4-e2b-it-4bit", "gemma-4-e2b", "gemma4", 2),
                ("mlx-community/Gemma4-E2B-IT-Text-int4", "gemma-4-e2b", "gemma4_text", 2),
                ("mlx-community/gemma-4-e4b-it-4bit", "gemma-4-e4b", "gemma4", 4),
                ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4-26b-a4b", "gemma4", 26),
                ("mlx-community/gemma-4-26b-it-OptiQ-4bit", "gemma-4-26b-a4b", "gemma4", 26),
                ("mlx-community/gemma-4-31b-it-4bit", "gemma-4-31b", "gemma4", 31),
                (
                    "mlx-community/gemma-4-31B-it-assistant-bf16", "gemma-4-31b",
                    "gemma4_assistant", 31
                ),
            ]

            for (modelID, expectedProfileID, modelType, parameterCountB) in examples {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: modelType,
                        parameterCountB: parameterCountB,
                        keyHeadDimension: 256,
                        valueHeadDimension: 256
                    )
                )
                #expect(profile.id == expectedProfileID)
            }
        }

        @Test func testGemmaProfilesFailClosedForIncompatibleMetadata() {
            let registry = TurboQuantProfileRegistry.bundled

            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-4b-it-qat-4bit",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-4b-it-qat-4bit",
                    modelType: "gemma3",
                    parameterCountB: 4
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-4b-it-qat-4bit",
                    modelType: "gemma2",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-4b-it-qat-4bit",
                    modelType: "gemma3",
                    parameterCountB: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-12b-it-4bit",
                    modelType: "gemma3",
                    parameterCountB: 12,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3-4b-it-4bit",
                    modelType: "gemma3",
                    parameterCountB: 4
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-3n-E4B-it-lm-4bit",
                    modelType: "gemma3",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-4-31b-it-4bit",
                    modelType: "gemma4",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/embeddinggemma-300m-4bit",
                    modelType: "gemma3_text",
                    parameterCountB: 0.3,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/paligemma-3b-mix-448-8bit",
                    modelType: "gemma",
                    parameterCountB: 3,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-1.1-7b-it-4bit",
                    parameterCountB: 7,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/gemma-1.1-7b-it-4bit",
                    modelType: "gemma",
                    parameterCountB: 7
                ) == nil
            )
        }

        @Test func testLlamaProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, Double, Int)] = [
                ("mlx-community/Llama-2-7B-Chat-4bit", "llama-2-7b", 7, 128),
                ("mlx-community/Llama-2-13B-Chat-4bit", "llama-2-13b", 13, 128),
                ("mlx-community/Llama-2-70B-Chat-4bit", "llama-2-70b", 70, 128),
                ("mlx-community/Meta-Llama-3-3B-Instruct-4bit", "llama-3-3b", 3, 128),
                ("mlx-community/Meta-Llama-3-8B-Instruct-4bit", "llama-3-8b", 8, 128),
                (
                    "mlx-community/ArrowCanaria-Llama-8B-RL-v0.1-MLX-4bit", "llama-compatible-8b",
                    8, 128
                ),
                ("mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/LLaMA-Pro-8B-mlx", "llama-compatible-8b", 8, 128),
                ("mlx-community/LLama-3.1-Tulu-3-8B-4b", "llama-compatible-8b", 8, 128),
                ("mlx-community/Llama-3-Groq-8B-Tool-Use-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/Llama-3-Karamaru-v1-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/Meta-Llama-Guard-2-8B-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/Llama3-ChatQA-1.5-8B-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/defog-llama-3-sqlcoder-8b", "llama-compatible-8b", 8, 128),
                ("mlx-community/llama-3-youko-8b-instruct-4bit-mlx", "llama-compatible-8b", 8, 128),
                (
                    "mlx-community/Llama-3-Swallow-8B-Instruct-v0.1-4bit", "llama-compatible-8b", 8,
                    128
                ),
                (
                    "mlx-community/Llama-3.1-Swallow-8B-Instruct-v0.5-4bit", "llama-compatible-8b",
                    8, 128
                ),
                (
                    "mlx-community/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct-4bit",
                    "llama-compatible-8b", 8, 128
                ),
                ("mlx-community/Llama-3.1-SuperNova-Lite-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/Llama-SEA-LION-v3-8B-IT-mlx-4bit", "llama-compatible-8b", 8, 128),
                ("mlx-community/Llama-3-16B-Instruct-v0.1-4bit", "llama-3-16b", 16, 128),
                ("mlx-community/Meta-Llama-3-70B-Instruct-4bit", "llama-3-70b", 70, 128),
                (
                    "mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit", "llama-compatible-70b", 70,
                    128
                ),
                ("mlx-community/Llama-3-Groq-70B-Tool-Use-4bit", "llama-compatible-70b", 70, 128),
                (
                    "mlx-community/Llama-3-Swallow-70B-Instruct-v0.1-4bit", "llama-compatible-70b",
                    70, 128
                ),
                ("mlx-community/Llama-3.1-Tulu-3-70B-4bit", "llama-compatible-70b", 70, 128),
                ("mlx-community/r1-1776-distill-llama-70b-4bit", "llama-compatible-70b", 70, 128),
                (
                    "mlx-community/Wayfarer-Large-70B-Llama-3.3-4bit", "llama-compatible-70b", 70,
                    128
                ),
                (
                    "mlx-community/deepcogito-cogito-v1-preview-llama-70B-4bit",
                    "llama-compatible-70b", 70, 128
                ),
                ("mlx-community/Meta-Llama-3.1-4B-Instruct-4bit", "llama-3.1-4b", 4, 128),
                (
                    "mlx-community/Llama-3.1-Nemotron-Nano-4B-v1.1-4bit", "llama-compatible-4b", 4,
                    128
                ),
                ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "llama-3.1-8b", 8, 128),
                ("mlx-community/Meta-Llama-3.1-16B-Instruct-4bit", "llama-3.1-16b", 16, 128),
                ("mlx-community/Meta-Llama-3.1-70B-Instruct-4bit", "llama-3.1-70b", 70, 128),
                (
                    "mlx-community/Llama-3.1-Swallow-70B-Instruct-v0.5-4bit",
                    "llama-compatible-70b", 70, 128
                ),
                ("mlx-community/Llama-3.1-Nemotron-70B-Instruct-HF-4bit", "llama-3.1-70b", 70, 128),
                ("mlx-community/Llama-3.1-120B-4bit", "llama-3.1-120b", 120, 128),
                ("mlx-community/Meta-Llama-3-120B-Instruct-4bit", "llama-3-120b", 120, 128),
                ("mlx-community/Meta-Llama-3.1-405B-Instruct-4bit", "llama-3.1-405b", 405, 128),
                ("mlx-community/Llama-3.2-1B-Instruct-4bit", "llama-3.2-1b", 1, 64),
                ("mlx-community/Llama-3.2-3B-Instruct-4bit", "llama-3.2-3b", 3, 128),
                ("mlx-community/Llama-3.3-3B-Instruct-4bit", "llama-3.3-3b", 3, 128),
                ("mlx-community/Llama-3.3-70B-Instruct-4bit", "llama-3.3-70b", 70, 128),
                ("mlx-community/AMD-Llama-135M-4bit", "llama-compatible-135m", 0.135, 64),
                ("mlx-community/Llama-160M-4bit", "llama-compatible-160m", 0.16, 64),
                ("mlx-community/tiny-llama-1b-4bit", "llama-compatible-1b", 1, 128),
                (
                    "mlx-community/MiniCPM-2B-sft-4bit-llama-format-mlx", "llama-compatible-2b", 2,
                    64
                ),
                ("mlx-community/Custom-Llama-2B-4bit", "llama-compatible-2b", 2, 128),
                ("mlx-community/Impish_LLAMA_3B-6bit", "llama-compatible-3b", 3, 128),
                ("mlx-community/Custom-Llama-3B-4bit", "llama-compatible-3b", 3, 128),
                ("mlx-community/yayi2-30b-llama-hf-4bit-mlx", "llama-compatible-30b", 30, 112),
                ("mlx-community/llama-30b-supercot", "llama-compatible-30b", 30, 128),
                ("mlx-community/Custom-Llama-4B-4bit", "llama-compatible-4b", 4, 128),
            ]

            for (modelID, expectedProfileID, parameterCountB, headDimension) in examples {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: "llama",
                        parameterCountB: parameterCountB,
                        keyHeadDimension: headDimension,
                        valueHeadDimension: headDimension
                    )
                )
                #expect(profile.id == expectedProfileID)
            }
        }

        @Test func testMistralProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let textExamples: [(String, String, String, String?, Double)] = [
                ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistral-7b", "mistral", nil, 7),
                (
                    "mlx-community/mistral-ft-optimized-1227-4bit-mlx", "mistral-7b", "mistral",
                    nil, 7
                ),
                (
                    "mlx-community/Mistral-Nemo-Instruct-2407-4bit", "mistral-nemo-12b", "mistral",
                    nil, 12
                ),
                (
                    "mlx-community/Mistral-NeMo-Minitron-8B-Instruct-4bit", "mistral-compatible-8b",
                    "mistral", nil, 8
                ),
                (
                    "mlx-community/Mistralai-8B-Diagnosis-QA", "mistral-compatible-8b", "mistral",
                    nil, 8
                ),
                (
                    "mlx-community/Ministral-8B-Instruct-2410-4bit", "ministral-8b-2410", "mistral",
                    nil, 8
                ),
                ("mlx-community/Codestral-22B-v0.1-4bit", "codestral-22b", "mistral", nil, 22),
                (
                    "mlx-community/Mistral-22B-v0.1-4bit-MLX", "mistral-compatible-22b", "mistral",
                    nil, 22
                ),
                (
                    "mlx-community/Mistral-Small-Instruct-2409-4bit", "mistral-small-22b",
                    "mistral", nil, 22
                ),
                (
                    "mlx-community/Mistral-Small-Instruct-2501-4bit", "mistral-small-24b",
                    "mistral", nil, 24
                ),
                (
                    "mlx-community/Devstral-Small-2505-4bit", "devstral-small-24b", "mistral", nil,
                    24
                ),
                (
                    "mlx-community/Devstral-Samll-2507-bf16", "devstral-small-24b", "mistral", nil,
                    24
                ),
                (
                    "mlx-community/Magistral-Small-2506-4bit", "magistral-small-24b", "mistral",
                    nil, 24
                ),
                (
                    "mlx-community/DeepHermes-3-Mistral-24B-Preview-4bit", "mistral-compatible-24b",
                    "mistral", nil, 24
                ),
                (
                    "mlx-community/Dolphin-Mistral-24B-Venice-Edition-4bit",
                    "mistral-compatible-24b", "mistral", nil, 24
                ),
                (
                    "mlx-community/Dolphin3.0-R1-Mistral-24B-4bit", "mistral-compatible-24b",
                    "mistral", nil, 24
                ),
                (
                    "mlx-community/Mistral-Large-Instruct-2407-4bit", "mistral-large-2407",
                    "mistral", nil, 123
                ),
                (
                    "mlx-community/Ministral-3-3B-Instruct-2512-4bit", "ministral3-3b", "mistral3",
                    "ministral3", 3
                ),
                (
                    "mlx-community/Ministral-3-8B-Instruct-2512-4bit", "ministral3-8b", "mistral3",
                    "ministral3", 8
                ),
                (
                    "mlx-community/Ministral-3-14B-Instruct-2512-4bit", "ministral3-14b",
                    "mistral3", "ministral3", 14
                ),
                (
                    "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
                    "mistral-small-3.1-24b", "mistral3", "mistral", 24
                ),
                (
                    "mlx-community/Mistral-Small-3.1-Text-24B-Instruct-2503-4bit",
                    "mistral-small-3.1-24b", "mistral", nil, 24
                ),
                (
                    "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                    "mistral-small-3.2-24b", "mistral3", "mistral", 24
                ),
                (
                    "mlx-community/Devstral-Small-2-24B-4bit", "devstral-small-2-24b", "mistral3",
                    "mistral", 24
                ),
                (
                    "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", "devstral-small-2-24b",
                    "mistral3", "ministral3", 24
                ),
                (
                    "mlx-community/Devstral-2-123B-Instruct-2512-4bit", "devstral-2-123b",
                    "ministral3", nil, 123
                ),
                (
                    "mlx-community/Mistral-Medium-3.5-128B-Instruct-4bit",
                    "mistral-medium-3.5-128b", "mistral3", "mistral", 128
                ),
                (
                    "mlx-community/Mistral-Medium-3.5-128B-4bit", "mistral-medium-3.5-128b",
                    "mistral3", "ministral3", 128
                ),
            ]

            for (modelID, expectedProfileID, modelType, textConfigModelType, parameterCountB)
                in textExamples
            {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: modelType,
                        textConfigModelType: textConfigModelType,
                        parameterCountB: parameterCountB,
                        keyHeadDimension: 128,
                        valueHeadDimension: 128
                    )
                )
                #expect(profile.id == expectedProfileID)
            }

            let small4 = try #require(
                registry.profile(
                    for: "mlx-community/Mistral-Small-4-119B-A6B-Instruct-4bit",
                    modelType: "mistral3",
                    textConfigModelType: "mistral4",
                    parameterCountB: 119,
                    routedExperts: 128,
                    expertsPerToken: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(small4.id == "mistral-small-4-119b-a6b")

            let pixtral = try #require(
                registry.profile(
                    for: "mlx-community/Pixtral-12B-2409-4bit",
                    modelType: "pixtral",
                    textConfigModelType: "mistral",
                    modality: .visionText,
                    parameterCountB: 12,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                )
            )
            #expect(pixtral.id == "pixtral-12b")
        }

        @Test func testLlamaAndMistralProfilesFailClosedForIncompatibleMetadata() {
            let registry = TurboQuantProfileRegistry.bundled

            #expect(
                registry.profile(
                    for: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                    parameterCountB: 8,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                    modelType: "llama",
                    parameterCountB: 8
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                    modelType: "mistral",
                    parameterCountB: 8,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
                    modelType: "llama",
                    parameterCountB: 17,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
                    modelType: "mllama",
                    parameterCountB: 11,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/llava-llama-3-8b-v1_1-4bit",
                    modelType: "llama",
                    parameterCountB: 8,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Bunny-Llama-3-8B-V-4bit",
                    modelType: "llama",
                    parameterCountB: 8,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Idefics-Llama-8B-4bit",
                    modelType: "llama",
                    parameterCountB: 8,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Llama-OuteTTS-1.0-1B-4bit",
                    modelType: "llama",
                    parameterCountB: 1,
                    keyHeadDimension: 64,
                    valueHeadDimension: 64
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
                    modelType: "mistral",
                    parameterCountB: 47,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mamba-Codestral-7B-v0.1-4bit",
                    modelType: "mistral",
                    parameterCountB: 7,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/e5-mistral-7b-instruct-mlx",
                    modelType: "mistral",
                    parameterCountB: 7,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Snorkel-Mistral-PairRM-DPO-4bit-mlx",
                    modelType: "mistral",
                    parameterCountB: 7,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                    modelType: "mistral3",
                    parameterCountB: 24,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit",
                    modelType: "mistral3",
                    textConfigModelType: "ministral3",
                    parameterCountB: 24,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mistral-Small-4-119B-A6B-Instruct-4bit",
                    modelType: "mistral3",
                    textConfigModelType: "mistral4",
                    parameterCountB: 119,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mistral-Small-4-119B-A6B-Instruct-4bit",
                    modelType: "mistral3",
                    textConfigModelType: "mistral4",
                    parameterCountB: 119,
                    routedExperts: 64,
                    expertsPerToken: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Mistral-Small-4-119B-A6B-Instruct-4bit",
                    modelType: "mistral3",
                    textConfigModelType: "mistral4",
                    parameterCountB: 119,
                    routedExperts: 128,
                    expertsPerToken: 2,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
        }

        @Test func testQwen35TwoBUses256HeadDimensions() throws {
            let profile = try #require(
                TurboQuantProfileRegistry.bundled.profile(
                    for: "mlx-community/Qwen3.5-2B-4bit",
                    modelType: "qwen3_5",
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                )
            )

            #expect(profile.id == "qwen3.5-2b")
            #expect(profile.supports(keyHeadDimension: 256, valueHeadDimension: 256))
            #expect(!profile.supports(keyHeadDimension: 128, valueHeadDimension: 128))
        }

        @Test func testQwen35AndQwen36DenseProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, Double)] = [
                ("mlx-community/Qwen3.5-0.8B-MLX-4bit", "qwen3.5-0.8b", 0.8),
                ("mlx-community/Qwen3.5-2B-MLX-4bit", "qwen3.5-2b", 2),
                ("mlx-community/Qwen3.5-4B-MLX-4bit", "qwen3.5-4b", 4),
                ("mlx-community/Qwen3.5-9B-MLX-4bit", "qwen3.5-9b", 9),
                ("mlx-community/Qwen3.5-27B-4bit", "qwen3.5-27b", 27),
                ("mlx-community/mlx_qwen35_27b_q6", "qwen3.5-27b", 27),
                ("mlx-community/Qwen3.6-27B-4bit", "qwen3.6-27b", 27),
                (
                    "mlx-community/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-4.5bit-msq",
                    "qwen3.5-40b",
                    40
                ),
                (
                    "mlx-community/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit",
                    "qwen3.6-40b",
                    40
                ),
            ]

            for (modelID, expectedProfileID, parameterCountB) in examples {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: "qwen3_5",
                        parameterCountB: parameterCountB,
                        keyHeadDimension: 256,
                        valueHeadDimension: 256
                    )
                )
                #expect(profile.id == expectedProfileID)
                #expect(profile.supports(keyHeadDimension: 256, valueHeadDimension: 256))
                #expect(!profile.supports(keyHeadDimension: 128, valueHeadDimension: 128))
            }
        }

        @Test func testQwen35AndQwen36MoEProfilesMatchCurrentMLXCommunityConfigs() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let examples: [(String, String, Double)] = [
                ("mlx-community/Qwen3.5-35B-A3B-4bit", "qwen3.5-35b-a3b", 35),
                ("mlx-community/Qwen3.6-35B-A3B-4bit", "qwen3.6-35b-a3b", 35),
                ("mlx-community/Qwen3.5-REAP-97B-A10B-4bit", "qwen3.5-97b-a10b", 97),
                ("mlx-community/Qwen3.5-122B-A10B-4bit", "qwen3.5-122b-a10b", 122),
                ("mlx-community/Qwen3.5-397B-A17B-4bit", "qwen3.5-397b-a17b", 397),
            ]

            for (modelID, expectedProfileID, parameterCountB) in examples {
                let profile = try #require(
                    registry.profile(
                        for: modelID,
                        modelType: "qwen3_5_moe",
                        parameterCountB: parameterCountB,
                        keyHeadDimension: 256,
                        valueHeadDimension: 256
                    )
                )
                #expect(profile.id == expectedProfileID)
            }
        }

        @Test func testQwen35ProfilesFailClosedForIncompatibleMetadata() {
            let registry = TurboQuantProfileRegistry.bundled

            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3.5-4B-MLX-4bit",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3.5-4B-MLX-4bit",
                    modelType: "qwen3_5",
                    parameterCountB: 4
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3.5-4B-MLX-4bit",
                    modelType: "qwen3",
                    parameterCountB: 4,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3.5-4B-MLX-4bit",
                    modelType: "qwen3_5",
                    parameterCountB: 4,
                    keyHeadDimension: 128,
                    valueHeadDimension: 128
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3.5-122B-A10B-4bit",
                    modelType: "qwen3_5",
                    parameterCountB: 122,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
            #expect(
                registry.profile(
                    for: "mlx-community/Qwen3.7-27B-4bit",
                    modelType: "qwen3_5",
                    parameterCountB: 27,
                    keyHeadDimension: 256,
                    valueHeadDimension: 256
                ) == nil
            )
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
                modelType: "qwen3",
                keyHeadDimension: 128,
                valueHeadDimension: 128
            )

            #expect(parameters.kvCacheStrategy == .turboQuant)
            #expect(parameters.kvGroupSize == 64)
            #expect(parameters.turboQuantPreset == .turbo4v2)
            #expect(parameters.turboQuantBackend == .metalPolarQJL)
            #expect(parameters.turboQuantOptimizationPolicy == .conservative)
            #expect(parameters.turboQuantFallbackPolicy == .compressedDecodeAllowed)
            #expect(parameters.turboQuantValueBits == 4)
        }

        @Test func testLowerBitBundledProfilesAreGuardedConservative() {
            let lowerBitProfiles = TurboQuantProfileRegistry.bundled.profiles.filter {
                $0.recommendedScheme.requiresGuardedQualityValidation
            }

            #expect(!lowerBitProfiles.isEmpty)
            #expect(lowerBitProfiles.allSatisfy { $0.status == .guarded })
            #expect(lowerBitProfiles.allSatisfy { $0.optimizationPolicy == .conservative })
        }

        @Test func testTurbo8QualitySensitiveProfilesUseExactPrefillRawFreeDecodePolicy() throws {
            let registry = TurboQuantProfileRegistry.bundled
            let profileIDs = [
                "qwen3.5-0.8b",
                "qwen3.5-2b",
                "gemma-3-1b",
                "llama-3.2-3b",
            ]

            for profileID in profileIDs {
                let profile = try #require(registry.profiles.first { $0.id == profileID })
                #expect(profile.recommendedScheme == .turbo8)
                #expect(profile.valueBits == 8)
                #expect(profile.optimizationPolicy == .auto)
            }
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

        @Test func testProductManifestValidationReportsExactBundledGaps() {
            let validations = TurboQuantProfileRegistry.bundled.profiles.map {
                $0.productManifestValidation()
            }

            #expect(!validations.isEmpty)
            #expect(
                validations.allSatisfy { validation in
                    validation.isValid
                        || validation.issues.allSatisfy {
                            !$0.profileID.isEmpty && !$0.field.isEmpty && !$0.reason.isEmpty
                        }
                }
            )
            #expect(
                validations.contains {
                    $0.issues.contains { $0.field == "model_fingerprint" }
                }
            )
        }

        @Test func testProfileSchemaVersionFailsClosedDuringSelection() {
            let profiles = [
                TurboQuantProfile(
                    schemaVersion: 1,
                    id: "legacy-schema",
                    modelPatterns: ["*schema-test*"],
                    architecture: "qwen3",
                    modelTypes: ["qwen3"],
                    requiresModelType: true,
                    requiresHeadDimensions: true,
                    supportedKeyHeadDimensions: [128]
                ),
                TurboQuantProfile(
                    schemaVersion: TurboQuantProfile.currentSchemaVersion + 1,
                    id: "future-schema",
                    modelPatterns: ["*schema-test*"],
                    architecture: "qwen3",
                    modelTypes: ["qwen3"],
                    requiresModelType: true,
                    requiresHeadDimensions: true,
                    supportedKeyHeadDimensions: [128]
                ),
            ]
            let selection = TurboQuantProfileRegistry(profiles: profiles).selection(
                for: TurboQuantModelDescriptor(
                    modelID: "mlx-community/schema-test-4B-4bit",
                    modelType: "qwen3",
                    parameterCountB: 4
                ),
                keyHeadDimension: 128,
                valueHeadDimension: 128
            )

            #expect(selection.profile == nil)
            #expect(
                selection.rejectionReasons.contains {
                    $0.contains("schema version 1 is unsupported; expected 2")
                }
            )
            #expect(
                selection.rejectionReasons.contains {
                    $0.contains("schema version 3 is unsupported; expected 2")
                }
            )
            #expect(
                selection.mismatches.contains {
                    $0.field == "schema_version"
                        && $0.expected == "2"
                        && $0.actual == "1"
                        && $0.disablesTurboQuant
                }
            )
        }

        @Test func testUnsupportedLayoutVersionFailsClosedDuringSelection() {
            let unsupportedLayoutVersion = 9999
            let profile = TurboQuantProfile(
                id: "unsupported-layout",
                modelPatterns: ["*layout-test*"],
                architecture: "qwen3",
                modelTypes: ["qwen3"],
                requiresModelType: true,
                requiresHeadDimensions: true,
                supportedKeyHeadDimensions: [128],
                turboQuant: TurboQuantProfileTurboQuantManifest(
                    layoutVersion: unsupportedLayoutVersion,
                    keyPreset: .turbo4v2,
                    valueBits: 4,
                    groupSize: 64
                )
            )
            let selection = TurboQuantProfileRegistry(profiles: [profile]).selection(
                for: TurboQuantModelDescriptor(
                    modelID: "mlx-community/layout-test-4B-4bit",
                    modelType: "qwen3",
                    parameterCountB: 4
                ),
                keyHeadDimension: 128,
                valueHeadDimension: 128
            )

            #expect(selection.profile == nil)
            #expect(
                selection.rejectionReasons.contains {
                    $0.contains("TurboQuant layout version")
                }
            )
            #expect(
                selection.mismatches.contains {
                    $0.field == "turbo_quant.layout_version"
                        && $0.actual == String(unsupportedLayoutVersion)
                }
            )
        }

        @Test func testManifestMismatchDTOExportsStableFields() {
            let expectedFingerprint = TurboQuantModelFingerprint(
                family: "qwen3",
                hiddenSize: 4096,
                layerCount: 36,
                attentionHeads: 32,
                kvHeads: 8,
                headDim: 128,
                rope: TurboQuantRoPEFingerprint(type: "llama", theta: 1_000_000),
                slidingWindow: TurboQuantSlidingWindowFingerprint(enabled: false),
                cacheType: "standard"
            )
            let actualFingerprint = TurboQuantModelFingerprint(
                family: "qwen3",
                hiddenSize: 5120,
                layerCount: 36,
                attentionHeads: 32,
                kvHeads: 8,
                headDim: 128,
                rope: TurboQuantRoPEFingerprint(type: "llama", theta: 1_000_000),
                slidingWindow: TurboQuantSlidingWindowFingerprint(enabled: false),
                cacheType: "standard"
            )
            let profile = TurboQuantProfile(
                id: "strict-qwen3",
                modelPatterns: ["*qwen3*"],
                architecture: "qwen3",
                modelTypes: ["qwen3"],
                requiresModelType: true,
                requiresHeadDimensions: true,
                supportedKeyHeadDimensions: [128],
                modelFingerprint: expectedFingerprint
            )

            let validation = profile.productManifestValidation(
                actualFingerprint: actualFingerprint,
                requireMeasuredOutcomes: false
            )
            #expect(!validation.isValid)
            #expect(
                validation.mismatches.contains {
                    $0.field == "model_fingerprint.hidden_size"
                        && $0.expected == "4096"
                        && $0.actual == "5120"
                        && $0.disablesTurboQuant
                }
            )
        }

        @Test func testFingerprintMismatchFailsClosedWithExactField() throws {
            let expectedFingerprint = TurboQuantModelFingerprint(
                family: "qwen3",
                hiddenSize: 4096,
                layerCount: 36,
                attentionHeads: 32,
                kvHeads: 8,
                headDim: 128,
                rope: TurboQuantRoPEFingerprint(type: "llama", theta: 1_000_000),
                slidingWindow: TurboQuantSlidingWindowFingerprint(enabled: false),
                cacheType: "standard"
            )
            let measured = TurboQuantMeasuredOutcome(
                deviceClass: "A18",
                osVersion: "iOS 18.0",
                maxContextByMode: [
                    TurboQuantUserMode.fastest.rawValue: 8192,
                    TurboQuantUserMode.balanced.rawValue: 32768,
                    TurboQuantUserMode.maxContext.rawValue: 65536,
                    TurboQuantUserMode.batterySaver.rawValue: 4096,
                ],
                actualBytesPerToken: 1024,
                decodeP50Seconds: 0.010,
                decodeP95Seconds: 0.016,
                prefillP50Seconds: 0.120,
                logitKL: 0.001,
                top1MatchRate: 0.99,
                longContextRetrievalScore: 1.0
            )
            let profile = TurboQuantProfile(
                id: "strict-qwen3",
                modelPatterns: ["*qwen3*"],
                architecture: "qwen3",
                modelTypes: ["qwen3"],
                requiresModelType: true,
                requiresHeadDimensions: true,
                supportedKeyHeadDimensions: [128],
                modelFingerprint: expectedFingerprint,
                measuredOutcomes: [measured]
            )
            #expect(
                profile.productManifestValidation(actualFingerprint: expectedFingerprint).isValid)

            let mismatchedFingerprint = TurboQuantModelFingerprint(
                family: "qwen3",
                hiddenSize: 5120,
                layerCount: 36,
                attentionHeads: 32,
                kvHeads: 8,
                headDim: 128,
                rope: TurboQuantRoPEFingerprint(type: "llama", theta: 1_000_000),
                slidingWindow: TurboQuantSlidingWindowFingerprint(enabled: false),
                cacheType: "standard"
            )
            let registry = TurboQuantProfileRegistry(profiles: [profile])
            let selection = registry.selection(
                for: TurboQuantModelDescriptor(
                    modelID: "mlx-community/Qwen3-4B-4bit",
                    modelType: "qwen3",
                    fingerprint: mismatchedFingerprint
                ),
                keyHeadDimension: 128,
                valueHeadDimension: 128,
                requireFingerprint: true
            )

            #expect(selection.profile == nil)
            #expect(
                selection.rejectionReasons.contains {
                    $0.contains("model_fingerprint.hidden_size expected 4096, got 5120")
                }
            )
        }

        @Test func testQualityGateGoldenJSONProvidesPinesFields() throws {
            struct GoldenBenchmarkResult: Codable {
                var id: String
                var quality: TurboQuantQualityGateReport
            }
            struct GoldenBenchmarkEnvelope: Codable {
                var schemaVersion: Int
                var qualityGate: TurboQuantQualityGateReport
                var results: [GoldenBenchmarkResult]
            }

            let golden = """
                {
                  "schemaVersion": 2,
                  "qualityGate": {
                    "schemaVersion": 1,
                    "gateVersion": 1,
                    "benchmarkSuiteID": "fallback-equivalence-v1",
                    "deterministicTop1MatchRate": 1.0,
                    "logitKLDivergenceMean": 0.0,
                    "logitMaxAbsErrorP95": 0.0,
                    "attentionOutputCosineMean": 1.0,
                    "noNaNOrInf": true,
                    "fallbackEquivalent": true,
                    "prefillExact": false,
                    "gateReason": "prefill exactness failed",
                    "passed": false
                  },
                  "results": [
                    {
                      "id": "hd64_ctx1_q1_fp16_causal_contiguous",
                      "quality": {
                        "schemaVersion": 1,
                        "gateVersion": 1,
                        "benchmarkSuiteID": "fallback-equivalence-v1",
                        "deterministicTop1MatchRate": 1.0,
                        "logitKLDivergenceMean": 0.0,
                        "logitMaxAbsErrorP95": 0.0,
                        "attentionOutputCosineMean": 1.0,
                        "noNaNOrInf": true,
                        "fallbackEquivalent": true,
                        "prefillExact": false,
                        "snapshotRoundtripEquivalent": null,
                        "gateReason": "prefill exactness failed",
                        "passed": false
                      }
                    }
                  ]
                }
                """
            let envelope = try JSONDecoder().decode(
                GoldenBenchmarkEnvelope.self,
                from: Data(golden.utf8)
            )

            #expect(envelope.qualityGate.schemaVersion == 1)
            #expect(envelope.qualityGate.gateVersion == 1)
            #expect(envelope.qualityGate.benchmarkSuiteID == "fallback-equivalence-v1")
            #expect(envelope.qualityGate.noNaNOrInf)
            #expect(envelope.qualityGate.fallbackEquivalent)
            #expect(!envelope.qualityGate.prefillExact)
            #expect(envelope.qualityGate.snapshotRoundtripEquivalent == nil)
            #expect(!envelope.qualityGate.passed)
            #expect(envelope.qualityGate.gateReason?.contains("prefill exactness failed") == true)
            #expect(envelope.results.first?.quality.logitMaxAbsErrorP95 == 0)
        }

        @Test func testQualityGateFailsClosedForNaNOrInfFlag() {
            let gate = TurboQuantQualityGateReport.evaluated(
                deterministicTop1MatchRate: 1,
                logitKLDivergenceMean: 0,
                logitMaxAbsErrorP95: 0,
                noNaNOrInf: false,
                fallbackEquivalent: true,
                prefillExact: true
            )

            #expect(!gate.passed)
            #expect(gate.gateReason?.contains("NaN or Inf detected") == true)
        }

        @Test func testQualityGateFailsClosedForUnmeasuredPrefillExactness() {
            let gate = TurboQuantQualityGateReport.evaluated(
                benchmarkSuiteID: .fallbackEquivalenceV1,
                deterministicTop1MatchRate: 1,
                logitKLDivergenceMean: 0,
                logitMaxAbsErrorP95: 0,
                noNaNOrInf: true,
                fallbackEquivalent: true,
                prefillExact: false
            )

            #expect(!gate.passed)
            #expect(gate.gateReason?.contains("prefill exactness failed") == true)
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
                #expect(bundledProfile.textConfigModelTypes == jsonProfile.textConfigModelTypes)
                #expect(bundledProfile.modalities == jsonProfile.modalities)
                #expect(bundledProfile.minParametersB == jsonProfile.minParametersB)
                #expect(bundledProfile.maxParametersB == jsonProfile.maxParametersB)
                #expect(bundledProfile.requiresModelType == jsonProfile.requiresModelType)
                #expect(
                    bundledProfile.requiresTextConfigModelType
                        == jsonProfile.requiresTextConfigModelType
                )
                #expect(
                    bundledProfile.requiresHeadDimensions == jsonProfile.requiresHeadDimensions
                )
                #expect(bundledProfile.minRoutedExperts == jsonProfile.minRoutedExperts)
                #expect(bundledProfile.maxRoutedExperts == jsonProfile.maxRoutedExperts)
                #expect(
                    bundledProfile.supportedExpertsPerToken == jsonProfile.supportedExpertsPerToken)
                #expect(
                    bundledProfile.supportedKeyHeadDimensions
                        == jsonProfile.supportedKeyHeadDimensions
                )
                #expect(
                    bundledProfile.supportedValueHeadDimensions
                        == jsonProfile.supportedValueHeadDimensions
                )
                #expect(bundledProfile.recommendedScheme == jsonProfile.recommendedScheme)
                #expect(bundledProfile.fallbackScheme == jsonProfile.fallbackScheme)
                #expect(bundledProfile.keyBits == jsonProfile.keyBits)
                #expect(bundledProfile.valueBits == jsonProfile.valueBits)
                #expect(bundledProfile.groupSize == jsonProfile.groupSize)
                #expect(bundledProfile.safeMaskModes == jsonProfile.safeMaskModes)
                #expect(
                    bundledProfile.supportedContextLengths == jsonProfile.supportedContextLengths)
                #expect(bundledProfile.safeContextLength == jsonProfile.safeContextLength)
                #expect(bundledProfile.qualityProfile == jsonProfile.qualityProfile)
                #expect(bundledProfile.backend == jsonProfile.backend)
                #expect(bundledProfile.optimizationPolicy == jsonProfile.optimizationPolicy)
                #expect(
                    bundledProfile.requiresMetalSelfTest
                        == jsonProfile.requiresMetalSelfTest
                )
                #expect(
                    bundledProfile.requiresFusedAttentionSelfTest
                        == jsonProfile.requiresFusedAttentionSelfTest
                )
                #expect(bundledProfile.status == jsonProfile.status)
                #expect(bundledProfile.source == jsonProfile.source)
                #expect(bundledProfile.confidence == jsonProfile.confidence)
            }

            let registry = TurboQuantProfileRegistry(profiles: profiles)
            let glm = try #require(
                registry.profile(
                    for: "RNT56/GLM4-MoE-Lite-4bit",
                    modelType: "glm4_moe_lite",
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
