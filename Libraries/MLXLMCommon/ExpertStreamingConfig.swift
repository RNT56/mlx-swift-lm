// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Expert weight streaming mode.
public enum ExpertStreamingMode: Sendable, Equatable {
    /// Expert streaming disabled. Expert weights are loaded normally.
    case disabled

    /// Expert weights are backed by the OS page cache from the model directory.
    case mmapPageCache(modelDirectory: URL)

    /// Expert weights are read directly from the model directory when supported.
    case directNVMe(modelDirectory: URL)
}

/// Shared configuration for MoE expert streaming.
///
/// Set ``mode`` before loading model weights. The loader uses this to create an
/// ``ExpertStreamerManager`` and relax missing-parameter verification for expert
/// tensors that are expected to be streamed later.
public final class ExpertStreamingConfig: @unchecked Sendable {
    public static let shared = ExpertStreamingConfig()

    public var mode: ExpertStreamingMode = .disabled

    private init() {}

    public var isEnabled: Bool {
        mode != .disabled
    }

    public var modelDirectory: URL? {
        switch mode {
        case .disabled:
            nil
        case .mmapPageCache(let modelDirectory), .directNVMe(let modelDirectory):
            modelDirectory
        }
    }

    public var useDirectNVMe: Bool {
        if case .directNVMe = mode {
            return true
        }
        return false
    }

    public func activate(modelDirectory: URL, useDirectIO: Bool = false) {
        #if os(macOS)
            mode =
                useDirectIO
                ? .directNVMe(modelDirectory: modelDirectory)
                : .mmapPageCache(modelDirectory: modelDirectory)
        #else
            mode = .mmapPageCache(modelDirectory: modelDirectory)
        #endif
    }

    public func deactivate() {
        mode = .disabled
        ExpertStreamerManager.shared = nil
    }

    /// Compatibility accessor for call sites that still expect the old
    /// environment-variable payload.
    public var legacyEnvPath: String? {
        modelDirectory?.path
    }
}

/// Metadata hook for switch-linear layers that can receive safetensors lookup
/// information from ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:lazyLoad:)``.
///
/// The conformance below stores metadata out-of-line so this loader wave can
/// remain additive without editing `SwitchLayers.swift`.
public protocol ExpertStreamingSwitchLinear: AnyObject {
    var tensorName: String? { get set }
    var unstackedSSDMap: [Int: (path: String, tensorName: String)]? { get set }
    var weightScaleInv: MLXArray? { get set }
    var expertCount: Int { get }
}

private final class ExpertStreamingSwitchLinearMetadata: @unchecked Sendable {
    var tensorName: String?
    var unstackedSSDMap: [Int: (path: String, tensorName: String)]?
    var weightScaleInv: MLXArray?
}

nonisolated(unsafe) private var expertStreamingSwitchLinearMetadata = [
    ObjectIdentifier: ExpertStreamingSwitchLinearMetadata
]()

private func metadata(for switchLinear: SwitchLinear) -> ExpertStreamingSwitchLinearMetadata {
    let key = ObjectIdentifier(switchLinear)
    if let metadata = expertStreamingSwitchLinearMetadata[key] {
        return metadata
    }
    let metadata = ExpertStreamingSwitchLinearMetadata()
    expertStreamingSwitchLinearMetadata[key] = metadata
    return metadata
}

extension SwitchLinear: ExpertStreamingSwitchLinear {
    public var tensorName: String? {
        get { metadata(for: self).tensorName }
        set { metadata(for: self).tensorName = newValue }
    }

    public var unstackedSSDMap: [Int: (path: String, tensorName: String)]? {
        get { metadata(for: self).unstackedSSDMap }
        set { metadata(for: self).unstackedSSDMap = newValue }
    }

    public var weightScaleInv: MLXArray? {
        get { metadata(for: self).weightScaleInv }
        set { metadata(for: self).weightScaleInv = newValue }
    }

    public var expertCount: Int {
        numExperts
    }
}

/// Safetensors tensor-name to shard-file lookup support for expert streaming.
public final class ExpertStreamerManager: @unchecked Sendable {
    nonisolated(unsafe) public static var shared: ExpertStreamerManager?

    public let modelDirectory: URL
    public let weightMap: [String: String]

    public init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
        self.weightMap = Self.loadWeightMap(modelDirectory: modelDirectory)
    }

    public func getFile(for tensorName: String) -> String? {
        weightMap[tensorName]
    }

    public func contains(_ tensorName: String) -> Bool {
        weightMap[tensorName] != nil
    }

    private static func loadWeightMap(modelDirectory: URL) -> [String: String] {
        let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: String]
        {
            return weightMap
        }

        guard
            let enumerator = FileManager.default.enumerator(
                at: modelDirectory, includingPropertiesForKeys: nil)
        else {
            return [:]
        }

        var map = [String: String]()
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            let relativePath =
                url.path.hasPrefix(modelDirectory.path + "/")
                ? String(url.path.dropFirst(modelDirectory.path.count + 1))
                : url.lastPathComponent
            for tensorName in (try? tensorNames(in: url)) ?? [] {
                map[tensorName] = relativePath
            }
        }
        return map
    }

    private static func tensorNames(in url: URL) throws -> [String] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let headerLengthData = handle.readData(ofLength: 8)
        guard headerLengthData.count == 8 else { return [] }

        let headerLength = headerLengthData.withUnsafeBytes { bytes in
            bytes.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard headerLength <= UInt64(Int.max) else { return [] }

        let headerData = handle.readData(ofLength: Int(headerLength))
        guard headerData.count == Int(headerLength),
            let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else {
            return []
        }

        return json.keys.filter { $0 != "__metadata__" }
    }
}
