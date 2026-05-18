// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX

/// Provider for optional DFlash-specialized kernels.
///
/// This keeps the experimental DFlash model hooks out of concrete model modules. Model code can
/// probe the registry later without adding package cycles.
public protocol DFlashKernelProvider: Sendable {
    func gatedDeltaKernelWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray)
}

/// Process-wide opt-in registry for DFlash kernels.
public enum DFlashKernelRegistry {
    public nonisolated(unsafe) static var provider: DFlashKernelProvider?
}

/// Experimental DFlash kernels with portable MLX-ops fallbacks.
///
/// The Metal kernels from the reference implementation are intentionally not wired into default
/// generation yet. These ops paths make the core cache/runtime APIs compile and remain testable.
public enum DFlashKernels {
    public static func tapeReplayKernel(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let hv = tape.dim(2)
        let hk = k.dim(2)
        let repeatFactor = hv / hk
        let repeatedK = repeatFactor > 1 ? MLX.repeated(k, count: repeatFactor, axis: 2) : k
        let steps = tape.dim(1)

        var state = state
        for t in 0 ..< steps {
            let decay: MLXArray =
                if g.ndim == 4 {
                    g[0..., t, 0..., .newAxis, 0...]
                } else {
                    expandedDimensions(g[0..., t, 0...], axes: [2, 3])
                }
            let delta = tape[0..., t, 0..., .newAxis]
            let key = expandedDimensions(repeatedK[0..., t, 0...], axis: -2)
            let next = state * decay + delta * key

            if let mask {
                let gate = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(state.dtype)
                state = next * gate + state * (1 - gate)
            } else {
                state = next
            }
        }
        return state
    }

    public static func gatedDeltaKernelWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let hv = v.dim(2)
        let hk = q.dim(2)
        let repeatFactor = hv / hk
        let repeatedQ = repeatFactor > 1 ? MLX.repeated(q, count: repeatFactor, axis: 2) : q
        let repeatedK = repeatFactor > 1 ? MLX.repeated(k, count: repeatFactor, axis: 2) : k
        let steps = q.dim(1)

        var state = state
        var outputs = [MLXArray]()
        var tapeEntries = [MLXArray]()
        outputs.reserveCapacity(steps)
        tapeEntries.reserveCapacity(steps)

        for t in 0 ..< steps {
            let decay: MLXArray =
                if g.ndim == 4 {
                    g[0..., t, 0..., .newAxis, 0...]
                } else {
                    expandedDimensions(g[0..., t, 0...], axes: [2, 3])
                }
            let decayedState = state * decay
            let key = repeatedK[0..., t, 0...]
            let query = repeatedQ[0..., t, 0...]
            let value = v[0..., t, 0...]
            let kvMemory = (decayedState * expandedDimensions(key, axis: -2)).sum(axis: -1)
            let delta = (value - kvMemory) * expandedDimensions(beta[0..., t, 0...], axis: -1)
            let next = decayedState
                + expandedDimensions(key, axis: -2) * expandedDimensions(delta, axis: -1)
            let output = (next * expandedDimensions(query, axis: -2)).sum(axis: -1)

            if let mask {
                let stateGate = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(state.dtype)
                let outputGate = expandedDimensions(mask[0..., t], axes: [1, 2]).asType(output.dtype)
                state = next * stateGate + state * (1 - stateGate)
                outputs.append(output * outputGate)
                tapeEntries.append((delta * outputGate).asType(.float32))
            } else {
                state = next
                outputs.append(output)
                tapeEntries.append(delta.asType(.float32))
            }
        }

        return (MLX.stacked(outputs, axis: 1), state, MLX.stacked(tapeEntries, axis: 1))
    }

    public static func sdpaFallback(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray? = nil
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
}

public final class DFlashDefaultKernelProvider: DFlashKernelProvider, @unchecked Sendable {
    public init() {}

    public func gatedDeltaKernelWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        DFlashKernels.gatedDeltaKernelWithTape(
            q: q,
            k: k,
            v: v,
            g: g,
            beta: beta,
            state: state,
            mask: mask
        )
    }
}
