// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import MLXLLM
import MLXVLM
import XCTest

private struct FixedTokenIterator: TokenIteratorProtocol {
    let tokens: [Int]
    let maxTokens: Int?
    let promptPrefillTime: TimeInterval = 0

    private var index = 0
    private(set) var tokenCount = 0

    init(tokens: [Int], maxTokens: Int? = nil) {
        self.tokens = tokens
        self.maxTokens = maxTokens
    }

    mutating func next() -> Int? {
        guard index < tokens.count else {
            return nil
        }
        let token = tokens[index]
        index += 1
        tokenCount += 1
        return token
    }
}

private struct FixedTextTokenizer: Tokenizer {
    let vocabulary: [Int: String]
    let tokenIds: [String: Int]

    let bosToken: String? = nil
    let eosToken: String? = "<eos>"
    let unknownToken: String? = "<unk>"

    init(vocabulary: [Int: String]) {
        self.vocabulary = vocabulary
        self.tokenIds = Dictionary(uniqueKeysWithValues: vocabulary.map { ($0.value, $0.key) })
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        []
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { vocabulary[$0] ?? "" }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenIds[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocabulary[id]
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

final class StopStringTests: XCTestCase {

    func testGenerationConfigDecodesStopStringsArray() throws {
        let data = Data(
            """
            {
              "eos_token_id": [1, 2],
              "stop_strings": ["<|im_end|>", "<end_of_turn>"]
            }
            """.utf8)

        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: data)

        XCTAssertEqual(config.eosTokenIds?.values, [1, 2])
        XCTAssertEqual(config.stopStrings, ["<|im_end|>", "<end_of_turn>"])
    }

    func testGenerationConfigDecodesSingleStopStringAndStopAlias() throws {
        let data = Data(
            """
            {
              "stop_strings": "<turn|>",
              "stop": ["<fallback>"]
            }
            """.utf8)

        let config = try JSONDecoder().decode(GenerationConfigFile.self, from: data)

        XCTAssertEqual(config.stopStrings, ["<turn|>", "<fallback>"])
    }

    func testModelConfigurationResolutionPreservesStopStrings() {
        let config = ModelConfiguration(
            id: "org/model",
            extraEOSTokens: ["<extra>"],
            stopStrings: ["<stop>"],
            eosTokenIds: [7]
        )

        let resolved = config.resolved(
            modelDirectory: URL(filePath: "/tmp/model"),
            tokenizerDirectory: URL(filePath: "/tmp/tokenizer")
        )

        XCTAssertEqual(resolved.extraEOSTokens, ["<extra>"])
        XCTAssertEqual(resolved.stopStrings, ["<stop>"])
        XCTAssertEqual(resolved.eosTokenIds, [7])
    }

    func testStopStringSplitAcrossChunksStopsAndIsNotEmitted() async {
        let tokenizer = FixedTextTokenizer(vocabulary: [
            1: "hel",
            2: "lo<",
            3: "stop",
            4: ">hidden",
            5: "tail",
            98: "<unk>",
            99: "<eos>",
        ])
        let configuration = ModelConfiguration(id: "test", stopStrings: ["<stop>"])
        let iterator = FixedTokenIterator(tokens: [1, 2, 3, 4, 5])

        let (output, info) = await collectGeneration(
            configuration: configuration,
            tokenizer: tokenizer,
            iterator: iterator
        )

        XCTAssertEqual(output, "hello")
        XCTAssertEqual(info?.stopReason, .stop)
    }

    func testEOSTokenIdsAndExtraEOSTokensStillStopGeneration() async {
        let tokenizer = FixedTextTokenizer(vocabulary: [
            1: "a",
            2: "b",
            7: "<config-eos>",
            42: "<extra>",
            98: "<unk>",
            99: "<eos>",
        ])

        let eosConfiguration = ModelConfiguration(id: "test", eosTokenIds: [7])
        let (eosOutput, eosInfo) = await collectGeneration(
            configuration: eosConfiguration,
            tokenizer: tokenizer,
            iterator: FixedTokenIterator(tokens: [1, 7, 2])
        )
        XCTAssertEqual(eosOutput, "a")
        XCTAssertEqual(eosInfo?.stopReason, .stop)

        let extraConfiguration = ModelConfiguration(id: "test", extraEOSTokens: ["<extra>"])
        let (extraOutput, extraInfo) = await collectGeneration(
            configuration: extraConfiguration,
            tokenizer: tokenizer,
            iterator: FixedTokenIterator(tokens: [1, 42, 2])
        )
        XCTAssertEqual(extraOutput, "a")
        XCTAssertEqual(extraInfo?.stopReason, .stop)
    }

    func testKnownRegistryEntriesExposeFamilyStopDefaults() {
        assertStops(LLMRegistry.gemma3_1B_qat_4bit, "<end_of_turn>")
        assertStops(LLMRegistry.gemma3n_E4B_it_lm_4bit, "<end_of_turn>")
        assertStops(LLMRegistry.gemma4_e2b_it_4bit, "<turn|>")
        assertStops(LLMRegistry.qwen3_0_6b_4bit, "<|im_end|>")
        assertStops(LLMRegistry.qwen3_5_2b_4bit, "<|im_end|>")
        assertStops(LLMRegistry.phi3_5_4bit, "<|end|>")
        assertStops(LLMRegistry.phi3_5MoE, "<|end|>")
        assertStops(LLMRegistry.llama3_8B_4bit, "<|eot_id|>")
        assertStops(LLMRegistry.llama3_2_3B_4bit, "<|eot_id|>")

        assertStops(VLMRegistry.gemma3_4B_qat_4bit, "<end_of_turn>")
        assertStops(VLMRegistry.gemma4_E2B_it_4bit, "<end_of_turn>")
        assertStops(VLMRegistry.qwen2VL2BInstruct4Bit, "<|im_end|>")
        assertStops(VLMRegistry.qwen3VL4BInstruct8Bit, "<|im_end|>")
        assertStops(VLMRegistry.qwen3_5_27B_4bit, "<|im_end|>")
    }

    private func collectGeneration<TOKEN: TokenIteratorProtocol>(
        configuration: ModelConfiguration,
        tokenizer: Tokenizer,
        iterator: consuming TOKEN
    ) async -> (String, GenerateCompletionInfo?) {
        let (stream, task) = generateTask(
            promptTokenCount: 0,
            modelConfiguration: configuration,
            tokenizer: tokenizer,
            iterator: iterator
        )

        var output = ""
        var info: GenerateCompletionInfo?
        for await event in stream {
            switch event {
            case .chunk(let text):
                output += text
            case .info(let completionInfo):
                info = completionInfo
            case .toolCall:
                XCTFail("Unexpected tool call")
            }
        }
        await task.value

        return (output, info)
    }

    private func assertStops(
        _ configuration: ModelConfiguration,
        _ token: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertTrue(configuration.extraEOSTokens.contains(token), file: file, line: line)
        XCTAssertTrue(configuration.stopStrings.contains(token), file: file, line: line)
    }
}
