// Copyright 2026 SwiftLM Contributors
// MIT License - see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX

public protocol DFlashRollbackCache: AnyObject {
    var isArmed: Bool { get }
    func armRollback(prefixLen: Int)
    func rollback(nAccepted: Int)
    func clearTransients()
    func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray)
}

/// Rollback-capable cache for recurrent target layers.
///
/// This subclasses `MambaCache` so existing hybrid layer code can continue to recognize it as an
/// SSM cache once model-specific DFlash conformances are added.
public final class RecurrentRollbackCache: MambaCache, DFlashRollbackCache, @unchecked Sendable {
    public static let defaultConvKernelSize = 4

    private var armed = false
    private var tape: MLXArray?
    private var tapeK: MLXArray?
    private var tapeG: MLXArray?
    private var tapeQKV: MLXArray?
    private var snapshotState: [MLXArray?]?

    public override init(leftPadding: [Int]? = nil) {
        super.init(leftPadding: leftPadding)
    }

    public var isArmed: Bool { armed }

    public func armRollback(prefixLen: Int = 0) {
        armed = true
        tape = nil
        tapeK = nil
        tapeG = nil
        tapeQKV = nil
        snapshotState = [self[0], self[1]]
    }

    public func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray) {
        self.tape = tape
        tapeK = k
        tapeG = g
        tapeQKV = qkv
    }

    public func rollback(nAccepted: Int) {
        guard let snapshotState else {
            clearTransients()
            return
        }

        if snapshotState.indices.contains(0), let conv = snapshotState[0] {
            self[0] = conv
        }
        if snapshotState.indices.contains(1), let recurrent = snapshotState[1] {
            self[1] = recurrent
        }

        if let tape, let tapeK, let tapeG, let state = self[1] {
            let acceptedSteps = nAccepted + 1
            let replayed = DFlashKernels.tapeReplayKernel(
                tape: tape[0..., ..<acceptedSteps, 0..., 0...],
                k: tapeK[0..., ..<acceptedSteps, 0..., 0...],
                g: tapeG[0..., ..<acceptedSteps, 0...],
                state: state
            )
            self[1] = replayed
            self[0] = rebuildConvState(acceptedSteps: acceptedSteps)
        }

        clearTransients()
    }

    private func rebuildConvState(acceptedSteps: Int) -> MLXArray? {
        guard let tapeQKV else { return self[0] }
        let keep = Self.defaultConvKernelSize - 1
        guard keep > 0 else { return nil }

        let prefix: MLXArray
        if let snapshotState, snapshotState.indices.contains(0), let conv = snapshotState[0] {
            prefix = conv
        } else {
            prefix = MLXArray.zeros([tapeQKV.dim(0), keep, tapeQKV.dim(-1)], dtype: tapeQKV.dtype)
        }

        let convInput = concatenated([prefix, tapeQKV], axis: 1)
        let start = acceptedSteps
        let end = min(start + keep, convInput.dim(1))
        return convInput[0..., start ..< end, 0...]
    }

    public func clearTransients() {
        armed = false
        tape = nil
        tapeK = nil
        tapeG = nil
        tapeQKV = nil
        snapshotState = nil
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = RecurrentRollbackCache(leftPadding: leftPaddingValues)
        let state = self.state
        if !state.isEmpty {
            new.state = state.map { $0[.ellipsis] }
        }
        new.offset = offset
        return new
    }
}

/// Lightweight snapshot rollback for recurrent caches when tape replay is not desired.
public final class MambaSnapshotCache: MambaCache, DFlashRollbackCache, @unchecked Sendable {
    private var snapshotConv: MLXArray?
    private var snapshotRecurrent: MLXArray?
    private var armed = false

    public override init(leftPadding: [Int]? = nil) {
        super.init(leftPadding: leftPadding)
    }

    public var isArmed: Bool { armed }

    public func armRollback(prefixLen: Int = 0) {
        armed = true
        snapshotConv = self[0]
        snapshotRecurrent = self[1]
    }

    public func rollback(nAccepted: Int) {
        self[0] = snapshotConv
        self[1] = snapshotRecurrent
        clearTransients()
    }

    public func clearTransients() {
        armed = false
        snapshotConv = nil
        snapshotRecurrent = nil
    }

    public func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray) {}

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = MambaSnapshotCache(leftPadding: leftPaddingValues)
        let state = self.state
        if !state.isEmpty {
            new.state = state.map { $0[.ellipsis] }
        }
        new.offset = offset
        return new
    }
}
