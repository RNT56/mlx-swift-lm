import Foundation
import MLXLMCommon

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

private func argumentBool(_ name: String) -> Bool {
    CommandLine.arguments.contains(name)
}

private func argumentCSVInts(_ name: String) -> [Int] {
    guard let raw = argumentString(name) else { return [] }
    return raw.split(separator: ",")
        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
}

private func write(_ text: String, to path: String) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try Data(text.utf8).write(to: url, options: .atomic)
}

private func usage() {
    print(
        """
        TurboQuantLowerV2Calibrate

        Emits a deterministic K8/V2 KVLayerPolicy JSON for real-model acceptance runs.

        Options:
          --layer-count <n>          Required model attention layer count.
          --edge-size <n>            K8/V4 protected first/last layers. Default: 8.
          --restored-layers <csv>    Additional measured harmful layers to restore to K8/V4.
          --residual-r1              Use affineK8VxResidual(valueBits:2,residualsPerGroup:1) middle.
          --output-policy <path>     Policy JSON output. Default: lower-v2-policy.json.
          --output-summary <path>    Markdown summary output. Default: lower-v2-policy.md.
        """
    )
}

@main
struct TurboQuantLowerV2Calibrate {
    static func main() throws {
        if CommandLine.arguments.contains("--help") {
            usage()
            return
        }

        let layerCount = argumentInt("--layer-count", default: 0)
        guard layerCount > 0 else {
            throw NSError(
                domain: "TurboQuantLowerV2Calibrate",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "--layer-count must be positive"]
            )
        }

        let edgeSize = max(0, argumentInt("--edge-size", default: 8))
        let restoredLayers = Set(argumentCSVInts("--restored-layers").filter {
            $0 >= 0 && $0 < layerCount
        })
        let useResidual = argumentBool("--residual-r1")
        let middleCodec: KVLayerCodec = useResidual
            ? .affineK8VxResidual(valueBits: 2, residualsPerGroup: 1)
            : .affineK8Vx(valueBits: 2)

        var protected = Set<Int>()
        for index in 0 ..< min(edgeSize, layerCount) {
            protected.insert(index)
            protected.insert(layerCount - 1 - index)
        }
        protected.formUnion(restoredLayers)

        let rules = protected.sorted().map {
            KVLayerRule(layerIndex: $0, codec: .affineK8V4)
        }
        let policy = KVLayerPolicy(defaultCodec: middleCodec, rules: rules)
        try policy.validate(layerCount: layerCount)

        let remainingV2 = max(0, layerCount - protected.count)
        let v2Ratio = Double(remainingV2) / Double(layerCount)
        let promotionAllowed = v2Ratio >= 0.60
        let averageValueBits =
            (Double(remainingV2) * 2.0 + Double(protected.count) * 4.0) / Double(layerCount)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let policyJSON = String(decoding: try encoder.encode(policy), as: UTF8.self)
        let policyPath = argumentString("--output-policy", default: "lower-v2-policy.json")!
        try write(policyJSON + "\n", to: policyPath)

        let summaryPath = argumentString("--output-summary", default: "lower-v2-policy.md")!
        let summary = """
        # Lower V2 Calibration Policy

        - Layer count: \(layerCount)
        - Middle codec: \(middleCodec.summary)
        - Protected/restored layers: \(protected.sorted().map(String.init).joined(separator: ","))
        - Remaining V2 layers: \(remainingV2) / \(layerCount)
        - Effective average value bits: \(String(format: "%.3f", averageValueBits))
        - Promotion allowed by 60% V2 default: \(promotionAllowed ? "yes" : "no")
        - Policy hash: \(policy.stableHash)

        This policy is a deterministic seed for real-model acceptance. Use measured
        per-layer failure attribution to populate `--restored-layers`.
        """
        try write(summary + "\n", to: summaryPath)

        print("wrote policy: \(policyPath)")
        print("wrote summary: \(summaryPath)")
    }
}
