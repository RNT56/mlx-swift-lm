// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation

public enum DFlashDraftRegistry {
    static let registry: [String: String] = [
        "Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
        "Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
        "Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
        "Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
        "Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
        "Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
        "Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
    ]

    public static func resolveDraftRef(modelRef: String, draftRef: String? = nil) -> String? {
        if let draftRef { return draftRef }

        let stripped = stripModelOrg(modelRef).lowercased()
        for (key, value) in registry where key.lowercased() == stripped {
            return value
        }

        var bestMatch: (key: String, value: String)?
        for (key, value) in registry {
            let lowered = key.lowercased()
            if stripped == lowered
                || stripped.hasPrefix(lowered + "-")
                || stripped.hasPrefix(lowered + "_")
            {
                if bestMatch == nil || key.count > bestMatch!.key.count {
                    bestMatch = (key, value)
                }
            }
        }

        return bestMatch?.value
    }

    public static func supportedBaseModels() -> [String] {
        Array(registry.keys).sorted()
    }

    private static func stripModelOrg(_ modelRef: String) -> String {
        modelRef.split(separator: "/").last.map(String.init) ?? modelRef
    }
}
