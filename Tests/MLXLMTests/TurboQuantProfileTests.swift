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
            #expect(Set(profiles.map(\.id)) == Set(TurboQuantProfileRegistry.bundled.profiles.map(\.id)))

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
