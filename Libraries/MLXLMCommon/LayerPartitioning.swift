// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Protocol for models that support per-layer GPU/CPU device placement.
public protocol LayerPartitionable: AnyObject {
    /// Number of layers to run on GPU. `nil` means all layers use the default device.
    var gpuLayerCount: Int? { get set }

    /// Total number of partitionable transformer layers.
    var totalLayerCount: Int { get }
}

extension LayerPartitionable {
    public func setGPULayers(_ count: Int?) {
        if let count {
            gpuLayerCount = min(max(0, count), totalLayerCount)
        } else {
            gpuLayerCount = nil
        }
    }

    public func isGPULayer(_ index: Int) -> Bool {
        guard let gpuLayerCount else {
            return true
        }
        return index < gpuLayerCount
    }

    public var cpuLayerCount: Int {
        guard let gpuLayerCount else {
            return 0
        }
        return max(0, totalLayerCount - gpuLayerCount)
    }

    public var partitionSummary: String {
        guard let gpuLayerCount else {
            return "\(totalLayerCount)/\(totalLayerCount) GPU (full)"
        }
        return "\(gpuLayerCount) GPU / \(totalLayerCount - gpuLayerCount) CPU"
    }
}

/// Protocol for MoE models that can evaluate/cache-clear layer by layer while
/// expert weights are streamed.
public protocol StreamableMoE: AnyObject {
    var streamExperts: Bool { get set }
}

/// Execute a transformer layer on the selected device and optionally force
/// layer-local evaluation for streaming-style memory pressure control.
public func partitionedLayerCall<T>(
    index: Int,
    gpuLayerCount: Int?,
    stream: Bool = false,
    cacheToEval: KVCache? = nil,
    body: () -> T
) -> T {
    let isExpertStreaming = ExpertStreamingConfig.shared.isEnabled

    let result: T
    if let gpuLayerCount, index >= gpuLayerCount, !isExpertStreaming {
        result = Device.withDefaultDevice(.cpu, body)
    } else {
        result = body()
    }

    if isExpertStreaming {
        return result
    }

    if stream, let array = result as? MLXArray {
        if let cacheToEval {
            eval(cacheToEval.state + [array])
        } else {
            eval(array)
        }
        Stream.gpu.synchronize()
        Memory.clearCache()
    }

    return result
}
