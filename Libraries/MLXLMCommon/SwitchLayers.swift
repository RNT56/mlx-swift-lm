import Foundation
@preconcurrency import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

public struct ExpertRange: Equatable, Sendable {
    public let id: Int
    public let start: Int
    public let end: Int

    public init(id: Int, start: Int, end: Int) {
        self.id = id
        self.start = start
        self.end = end
    }
}

private func expertRanges(from indices: MLXArray) -> [ExpertRange] {
    let cpuIndices = indices.asArray(UInt32.self)
    var ranges = [ExpertRange]()
    var start = 0

    while start < cpuIndices.count {
        let expert = Int(cpuIndices[start])
        var end = start + 1
        while end < cpuIndices.count && Int(cpuIndices[end]) == expert {
            end += 1
        }
        ranges.append(ExpertRange(id: expert, start: start, end: end))
        start = end
    }

    return ranges
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let isExpertStreaming = ExpertStreamingConfig.shared.isEnabled
        let doSort = indices.size >= 64 || isExpertStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        if isExpertStreaming {
            MLX.eval(idx)
            let ranges = expertRanges(from: idx)
            let xGate = gateProj.computeStreamedExperts(x, ranges: ranges)
            let xUp = upProj.computeStreamedExperts(x, ranges: ranges)
            let activated =
                if let activationProduct {
                    activationProduct(xGate, xUp)
                } else {
                    activation(xGate) * xUp
                }
            x = downProj.computeStreamedExperts(activated, ranges: ranges)
        } else {
            let xUp = upProj(x, idx, sortedIndices: doSort)
            let xGate = gateProj(x, idx, sortedIndices: doSort)
            let activated =
                if let activationProduct {
                    activationProduct(xGate, xUp)
                } else {
                    activation(xGate) * xUp
                }
            x = downProj(
                activated,
                idx,
                sortedIndices: doSort)
        }

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight
/// of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate
/// `gate_proj` / `up_proj`. Used by Gemma 4 26B MoE.
public class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let isExpertStreaming = ExpertStreamingConfig.shared.isEnabled
        let doSort = indices.size >= 64 || isExpertStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gateUp: MLXArray
        let ranges: [ExpertRange]?
        if isExpertStreaming {
            MLX.eval(idx)
            let parsedRanges = expertRanges(from: idx)
            ranges = parsedRanges
            gateUp = gateUpProj.computeStreamedExperts(x, ranges: parsedRanges)
        } else {
            ranges = nil
            gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        }

        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated =
            if let activationProduct {
                activationProduct(parts[0], parts[1])
            } else {
                activation(parts[0]) * parts[1]
            }
        if let ranges {
            x = downProj.computeStreamedExperts(activated, ranges: ranges)
        } else {
            x = downProj(
                activated,
                idx,
                sortedIndices: doSort)
        }

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchLinear

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public func resolveSSDInfo() -> (path: String, tensorName: String)? {
        #if os(macOS)
            guard ExpertStreamingConfig.shared.useDirectNVMe else {
                return nil
            }

            if let firstUnstacked = unstackedSSDMap?[0] {
                return (firstUnstacked.path, firstUnstacked.tensorName)
            }

            guard let tensorName,
                let filename = ExpertStreamerManager.shared?.getFile(for: tensorName),
                let modelDirectory = ExpertStreamingConfig.shared.modelDirectory
            else {
                return nil
            }

            return (modelDirectory.appendingPathComponent(filename).path, tensorName)
        #else
            return nil
        #endif
    }

    public func resolveSSDInfo(expertIndex: Int) -> (
        path: String, tensorName: String, readIndex: UInt32
    )? {
        #if os(macOS)
            guard ExpertStreamingConfig.shared.useDirectNVMe else {
                return nil
            }

            if let unstacked = unstackedSSDMap?[expertIndex] {
                return (unstacked.path, unstacked.tensorName, 0)
            }

            guard let base = resolveSSDInfo() else {
                return nil
            }

            return (base.path, base.tensorName, UInt32(expertIndex))
        #else
            return nil
        #endif
    }

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ExpertStreamingConfig.shared.isEnabled {
            MLX.eval(indices)
            return computeStreamedExperts(x, ranges: expertRanges(from: indices))
        }

        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func allocateExpertBuffers(_ count: Int) -> [MLXArray] {
        (0 ..< count).map { _ in
            MLXArray.zeros([1, outputDims, inputDims]).asType(weight.dtype)
        }
    }

    @discardableResult
    public func loadExpertWeights(_ buffers: [MLXArray], ranges: [ExpertRange]) -> Bool {
        guard buffers.count == ranges.count else {
            return false
        }

        for (buffer, range) in zip(buffers, ranges) {
            guard let ssd = resolveSSDInfo(expertIndex: range.id) else {
                return false
            }

            let status = MLXFast.preadInto(
                buffer,
                safetensorsPath: ssd.path,
                tensorName: ssd.tensorName,
                expertIndex: ssd.readIndex)
            if status != 0 {
                return false
            }
        }

        return true
    }

    public func computeStreamedExperts(_ x: MLXArray, ranges: [ExpertRange]) -> MLXArray {
        if ranges.isEmpty {
            return zeroOutput(like: x)
        }

        if ExpertStreamingConfig.shared.useDirectNVMe,
            ranges.allSatisfy({ resolveSSDInfo(expertIndex: $0.id) != nil })
        {
            let buffers = allocateExpertBuffers(ranges.count)
            MLX.eval(buffers)
            if loadExpertWeights(buffers, ranges: ranges) {
                return computeExperts(x, buffers: buffers, ranges: ranges)
            }
        }

        return computeExperts(x, buffers: nil, ranges: ranges)
    }

    public func computeExperts(
        _ x: MLXArray, buffers: [MLXArray]?, ranges: [ExpertRange]
    ) -> MLXArray {
        var expertResults = [MLXArray]()

        for (i, range) in ranges.enumerated() {
            let rangeX = x[range.start ..< range.end]
            let expertIndices = MLXArray.zeros([range.end - range.start], type: UInt32.self)

            let expertWeight: MLXArray
            if let buffers {
                expertWeight = buffers[i]
            } else {
                expertWeight = weight[range.id ..< range.id + 1]
                MLX.eval(expertWeight)
                MLXFast.prefault(expertWeight)
            }

            var expertOutput = MLX.gatherMM(
                rangeX,
                expertWeight.swappedAxes(-1, -2),
                rhsIndices: expertIndices,
                sortedIndices: true)

            if let bias {
                let expertBias = bias[range.id ..< range.id + 1]
                expertOutput =
                    expertOutput
                    + MLX.expandedDimensions(
                        expertBias[expertIndices], axis: -2)
            }

            expertOutput = canonicalizedExpertOutput(expertOutput, like: rangeX)
            expertResults.append(expertOutput)
        }

        return expertResults.isEmpty
            ? zeroOutput(like: x)
            : MLX.concatenated(expertResults, axis: 0)
    }

    func zeroOutput(like x: MLXArray) -> MLXArray {
        var shape = x.shape
        shape[shape.count - 1] = outputDims
        return MLXArray.zeros(shape).asType(x.dtype)
    }

    func canonicalizedExpertOutput(_ expertOutput: MLXArray, like x: MLXArray) -> MLXArray {
        let canonicalShape = Array(x.shape.dropLast()) + [outputDims]
        if expertOutput.shape == canonicalShape {
            return expertOutput
        }
        return expertOutput.reshaped(canonicalShape)
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ExpertStreamingConfig.shared.isEnabled {
            MLX.eval(indices)
            return computeStreamedExperts(x, ranges: expertRanges(from: indices))
        }

        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    override public func computeExperts(
        _ x: MLXArray, buffers: [MLXArray]?, ranges: [ExpertRange]
    ) -> MLXArray {
        var expertResults = [MLXArray]()

        for (i, range) in ranges.enumerated() {
            let rangeX = x[range.start ..< range.end]
            let expertIndices = MLXArray.zeros([range.end - range.start], type: UInt32.self)

            let expertWeight: MLXArray
            if let buffers {
                expertWeight = buffers[i]
            } else {
                expertWeight = weight[range.id ..< range.id + 1]
                MLX.eval(expertWeight)
                MLXFast.prefault(expertWeight)
            }

            let expertScales = scales[range.id ..< range.id + 1]
            var expertBiases: MLXArray? = nil
            if let biases {
                expertBiases = biases[range.id ..< range.id + 1]
            }

            var expertOutput = MLX.gatherQuantizedMM(
                rangeX,
                expertWeight,
                scales: expertScales,
                biases: expertBiases,
                rhsIndices: expertIndices,
                transpose: true,
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                sortedIndices: true)

            if let bias {
                let expertBias = bias[range.id ..< range.id + 1]
                expertOutput =
                    expertOutput
                    + MLX.expandedDimensions(
                        expertBias[expertIndices], axis: -2)
            }

            expertOutput = canonicalizedExpertOutput(expertOutput, like: rangeX)
            expertResults.append(expertOutput)
        }

        return expertResults.isEmpty
            ? zeroOutput(like: x)
            : MLX.concatenated(expertResults, axis: 0)
    }
}
