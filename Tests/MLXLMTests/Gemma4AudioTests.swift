import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class Gemma4AudioTests: XCTestCase {
    func testGemma4ConfigurationDecodesAudioConfig() throws {
        let json = """
            {
              "model_type": "gemma4",
              "text_config": {},
              "vision_config": {},
              "audio_token_id": 258881,
              "audio_config": {
                "model_type": "gemma4_audio",
                "hidden_size": 1024,
                "num_hidden_layers": 12,
                "num_attention_heads": 8,
                "output_proj_dims": 1536
              }
            }
            """

        let config = try JSONDecoder().decode(Gemma4Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.audioTokenId, 258_881)
        XCTAssertEqual(config.audioConfig?.hiddenSize, 1024)
        XCTAssertEqual(config.audioConfig?.outputProjDims, 1536)
    }

    func testAudioFeatureExtractorProducesFeatureMaskPair() {
        let extractor = Gemma4AudioFeatureExtractor()
        let waveform = (0 ..< 16_000).map { index in
            sin(Float(index) * 2.0 * Float.pi * 440.0 / 16_000.0)
        }

        let (features, mask) = extractor.extract(waveform: waveform)

        XCTAssertEqual(features.shape.count, 3)
        XCTAssertEqual(features.dim(0), 1)
        XCTAssertEqual(features.dim(2), 128)
        XCTAssertEqual(mask.shape, [1, features.dim(1)])
        XCTAssertGreaterThan(features.dim(1), 0)
        XCTAssertGreaterThan(mask.asType(.int32).sum().item(Int.self), 0)
    }

    func testMessageGeneratorIncludesAudioContentMarker() {
        let message = Chat.Message.user(
            "transcribe",
            audio: [.data(Data([0, 1, 2]), format: "wav")]
        )

        let generated = Gemma4MessageGenerator().generate(message: message)
        let content = generated["content"] as? [[String: String]]

        XCTAssertEqual(content?.last?["type"], "audio")
    }
}
