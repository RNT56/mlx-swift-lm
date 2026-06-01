import Foundation
import IntegrationTestHelpers
import MLX
import MLXLLM
import MLXLMCommon

private struct IdentityTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.utf8.map { Int($0) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map { UInt8(truncatingIfNeeded: $0) }, as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.utf8.first.map(Int.init)
    }

    func convertIdToToken(_ id: Int) -> String? {
        String(UnicodeScalar(UInt8(truncatingIfNeeded: id)))
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        messages.flatMap { message in
            "\(message["role"] ?? ""): \(message["content"] ?? "")\n".utf8.map { Int($0) }
        }
    }
}

private struct IdentityTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        IdentityTokenizer()
    }
}

private func argumentString(_ name: String, default defaultValue: String? = nil) -> String? {
    let arguments = CommandLine.arguments
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else {
        return defaultValue
    }
    return arguments[index + 1]
}

private func argumentInt(_ name: String, default defaultValue: Int) -> Int {
    argumentString(name).flatMap(Int.init) ?? defaultValue
}

private func argumentInts(_ name: String, default defaultValue: [Int]) -> [Int] {
    guard let raw = argumentString(name) else { return defaultValue }
    let values = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return values.isEmpty ? defaultValue : values
}

@main
struct TurboQuantInferenceParityCLI {
    static func main() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = argumentString("--model-dir", default: environment["TQ_MODEL_DIR"]) else {
            throw IntegrationTestFailure(
                "missing model directory; pass --model-dir or set TQ_MODEL_DIR")
        }
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let contexts = argumentInts("--contexts", default: [4096, 8192, 16384])
        let generateTokens = argumentInt("--generate-tokens", default: 32)

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelURL,
            using: IdentityTokenizerLoader()
        )
        _ = try await InferenceParityBenchmark.run(
            container: container,
            contexts: contexts,
            generateTokens: generateTokens
        )
        Stream().synchronize()
    }
}
