import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// Parity coverage for the P1-1 fused quantize-append path in
// `AffineK8V4KVCache`. The op-level cases pin `MLXFast.quantizeAppendKV`
// against the stock quantize + slice-update reference; the cache-level cases
// pin a fused-on cache against a byte-identical fused-off cache and prove the
// fused kernel actually engaged. All cases build random tensors and require
// Metal, so they are skipped on non-GPU hosts.

extension MLXRuntimeSwiftTests {

    @Suite(.serialized)
    struct AffineFusedQuantizeAppendTests {

        // MARK: - Fixtures

        /// Default affine K8/V4 geometry with head_dim 128.
        private static let headDim = 128
        private static let keyGroupSize = TurboQuantKVCodec.affineK8V4KeyGroupSize   // 64
        private static let keyBits = TurboQuantKVCodec.affineK8V4KeyBits             // 8
        private static let valueGroupSize = TurboQuantKVCodec.affineK8V4ValueGroupSize // 32
        private static let valueBits = TurboQuantKVCodec.affineK8V4ValueBits         // 4
        private static let kvHeads = 2

        private static var keyCodeWords: Int { headDim * keyBits / 32 }   // 32
        private static var keyGroups: Int { headDim / keyGroupSize }      // 2
        private static var valueCodeWords: Int { headDim * valueBits / 32 } // 16
        private static var valueGroups: Int { headDim / valueGroupSize }  // 4

        private static func gpuAvailable() -> Bool {
            Device.defaultDevice().deviceType == .gpu
        }

        /// Deterministic fp16 row batch shaped [1, kvHeads, steps, headDim].
        private static func randomKV(steps: Int, seed: UInt64) -> (MLXArray, MLXArray) {
            MLXRandom.seed(seed)
            let k = MLXRandom.normal([1, kvHeads, steps, headDim]).asType(.float16)
            let v = MLXRandom.normal([1, kvHeads, steps, headDim]).asType(.float16)
            return (k, v)
        }

        private static func zeroPlane(_ lastDim: Int, capacity: Int, dtype: DType) -> MLXArray {
            MLXArray.zeros([1, kvHeads, capacity, lastDim], dtype: dtype)
        }

        private static func bytesEqual(_ lhs: MLXArray, _ rhs: MLXArray, _ label: String) {
            #expect(lhs.shape == rhs.shape, "shape mismatch: \(label) \(lhs.shape) vs \(rhs.shape)")
            #expect(lhs.dtype == rhs.dtype, "dtype mismatch: \(label)")
            let equal = MLX.all(lhs .== rhs).item(Bool.self)
            #expect(equal, "byte mismatch: \(label)")
        }

        // MARK: - (a) Op-level bit-exactness vs stock quantize + slice-update

        @Test func fusedOpMatchesStockQuantizeSliceUpdate() throws {
            guard Self.gpuAvailable() else { return }
            guard turboQuantNativeQuantizeAppendKVAvailable() else {
                Issue.record("native quantize-append probe reported unavailable on a GPU host")
                return
            }
            let capacity = 8
            let (kNew, vNew) = Self.randomKV(steps: 1, seed: 101)

            var opKCodes = Self.zeroPlane(Self.keyCodeWords, capacity: capacity, dtype: .uint32)
            var opKScales = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            var opKBiases = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            var opVCodes = Self.zeroPlane(Self.valueCodeWords, capacity: capacity, dtype: .uint32)
            var opVScales = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)
            var opVBiases = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)

            let refKCodes = Self.zeroPlane(Self.keyCodeWords, capacity: capacity, dtype: .uint32)
            let refKScales = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let refKBiases = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let refVCodes = Self.zeroPlane(Self.valueCodeWords, capacity: capacity, dtype: .uint32)
            let refVScales = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)
            let refVBiases = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)

            let (kCodes, kScales, kBiases) = quantized(
                kNew, groupSize: Self.keyGroupSize, bits: Self.keyBits, mode: .affine)
            let (vCodes, vScales, vBiases) = quantized(
                vNew, groupSize: Self.valueGroupSize, bits: Self.valueBits, mode: .affine)
            let kBiasesRef = try #require(kBiases)
            let vBiasesRef = try #require(vBiases)

            for offset in [0, 3, 7] {
                let range = offset ..< (offset + 1)
                refKCodes[.ellipsis, range, 0...] = kCodes
                refKScales[.ellipsis, range, 0...] = kScales
                refKBiases[.ellipsis, range, 0...] = kBiasesRef
                refVCodes[.ellipsis, range, 0...] = vCodes
                refVScales[.ellipsis, range, 0...] = vScales
                refVBiases[.ellipsis, range, 0...] = vBiasesRef

                let out = try MLXFast.quantizeAppendKV(
                    keysNew: kNew,
                    valuesNew: vNew,
                    kCodes: opKCodes,
                    kScales: opKScales,
                    kBiases: opKBiases,
                    vCodes: opVCodes,
                    vScales: opVScales,
                    vBiases: opVBiases,
                    seqOffset: offset,
                    steps: 1,
                    keyGroupSize: Self.keyGroupSize,
                    keyBits: Self.keyBits,
                    valueGroupSize: Self.valueGroupSize,
                    valueBits: Self.valueBits)
                opKCodes = out.kCodes
                opKScales = out.kScales
                opKBiases = out.kBiases
                opVCodes = out.vCodes
                opVScales = out.vScales
                opVBiases = out.vBiases
            }

            Self.bytesEqual(opKCodes, refKCodes, "kCodes")
            Self.bytesEqual(opKScales, refKScales, "kScales")
            Self.bytesEqual(opKBiases, refKBiases, "kBiases")
            Self.bytesEqual(opVCodes, refVCodes, "vCodes")
            Self.bytesEqual(opVScales, refVScales, "vScales")
            Self.bytesEqual(opVBiases, refVBiases, "vBiases")
        }

        @Test func fusedOpMatchesStockForMultiStepWrite() throws {
            guard Self.gpuAvailable() else { return }
            guard turboQuantNativeQuantizeAppendKVAvailable() else { return }
            let capacity = 8
            let steps = 4
            let offset = 2
            let (kNew, vNew) = Self.randomKV(steps: steps, seed: 202)

            let opKCodes = Self.zeroPlane(Self.keyCodeWords, capacity: capacity, dtype: .uint32)
            let opKScales = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let opKBiases = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let opVCodes = Self.zeroPlane(Self.valueCodeWords, capacity: capacity, dtype: .uint32)
            let opVScales = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)
            let opVBiases = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)

            let refKCodes = Self.zeroPlane(Self.keyCodeWords, capacity: capacity, dtype: .uint32)
            let refKScales = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let refKBiases = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let refVCodes = Self.zeroPlane(Self.valueCodeWords, capacity: capacity, dtype: .uint32)
            let refVScales = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)
            let refVBiases = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)

            let (kCodes, kScales, kBiases) = quantized(
                kNew, groupSize: Self.keyGroupSize, bits: Self.keyBits, mode: .affine)
            let (vCodes, vScales, vBiases) = quantized(
                vNew, groupSize: Self.valueGroupSize, bits: Self.valueBits, mode: .affine)
            let kBiasesRef = try #require(kBiases)
            let vBiasesRef = try #require(vBiases)

            let range = offset ..< (offset + steps)
            refKCodes[.ellipsis, range, 0...] = kCodes
            refKScales[.ellipsis, range, 0...] = kScales
            refKBiases[.ellipsis, range, 0...] = kBiasesRef
            refVCodes[.ellipsis, range, 0...] = vCodes
            refVScales[.ellipsis, range, 0...] = vScales
            refVBiases[.ellipsis, range, 0...] = vBiasesRef

            let out = try MLXFast.quantizeAppendKV(
                keysNew: kNew,
                valuesNew: vNew,
                kCodes: opKCodes,
                kScales: opKScales,
                kBiases: opKBiases,
                vCodes: opVCodes,
                vScales: opVScales,
                vBiases: opVBiases,
                seqOffset: offset,
                steps: steps,
                keyGroupSize: Self.keyGroupSize,
                keyBits: Self.keyBits,
                valueGroupSize: Self.valueGroupSize,
                valueBits: Self.valueBits)

            Self.bytesEqual(out.kCodes, refKCodes, "kCodes")
            Self.bytesEqual(out.kScales, refKScales, "kScales")
            Self.bytesEqual(out.kBiases, refKBiases, "kBiases")
            Self.bytesEqual(out.vCodes, refVCodes, "vCodes")
            Self.bytesEqual(out.vScales, refVScales, "vScales")
            Self.bytesEqual(out.vBiases, refVBiases, "vBiases")
        }

        // MARK: - Cache equivalence helpers

        private static func makeCache(fused: Bool, maxSize: Int? = nil) -> AffineK8V4KVCache {
            AffineK8V4KVCache(maxSize: maxSize, fusedQuantizeAppend: fused)
        }

        /// Asserts both caches carry byte-identical quantized state and bookkeeping.
        private static func assertCacheStatesEqual(
            _ fused: AffineK8V4KVCache,
            _ plain: AffineK8V4KVCache,
            label: String
        ) throws {
            #expect(fused.offset == plain.offset, "offset mismatch \(label)")
            let a = try #require(fused.getQuantizedState(), "fused missing state \(label)")
            let b = try #require(plain.getQuantizedState(), "plain missing state \(label)")
            bytesEqual(a.0.0, b.0.0, "\(label) kCodes")
            bytesEqual(a.0.1, b.0.1, "\(label) kScales")
            bytesEqual(try #require(a.0.2), try #require(b.0.2), "\(label) kBiases")
            bytesEqual(a.1.0, b.1.0, "\(label) vCodes")
            bytesEqual(a.1.1, b.1.1, "\(label) vScales")
            bytesEqual(try #require(a.1.2), try #require(b.1.2), "\(label) vBiases")
        }

        // MARK: - (b) Cache-level N-step equivalence

        @Test func fusedCacheMatchesLadderOverSingleTokenSequence() throws {
            guard Self.gpuAvailable() else { return }
            guard turboQuantNativeQuantizeAppendKVAvailable() else { return }

            let fused = Self.makeCache(fused: true)
            let plain = Self.makeCache(fused: false)
            let checkpoints: Set<Int> = [1, 5, 70]

            for step in 1 ... 70 {
                let (k, v) = Self.randomKV(steps: 1, seed: UInt64(1_000 + step))
                _ = fused.updateQuantized(keys: k, values: v)
                _ = plain.updateQuantized(keys: k, values: v)
                if checkpoints.contains(step) {
                    try Self.assertCacheStatesEqual(fused, plain, label: "single-token step \(step)")
                }
            }

            #expect(fused.attentionDiagnostics.fusedAppendCount == 70)
            #expect(fused.attentionDiagnostics.fusedAppendFallbackReason == nil)
            #expect(fused.attentionDiagnostics.fusedAppendFallbackCount == 0)
            // Flag-off cache reports nil counters (feature not engaged).
            #expect(plain.attentionDiagnostics.fusedAppendCount == nil)
        }

        @Test func fusedCacheMatchesLadderAcrossStorageGrowthBoundary() throws {
            guard Self.gpuAvailable() else { return }
            guard turboQuantNativeQuantizeAppendKVAvailable() else { return }

            // Prefill ~250 tokens via the normal multi-token update, then take
            // 10 single-token fused steps that cross the 256 storage-growth
            // boundary (250 -> 260).
            let fused = Self.makeCache(fused: true)
            let plain = Self.makeCache(fused: false)

            let (prefillK, prefillV) = Self.randomKV(steps: 250, seed: 5_000)
            _ = fused.updateQuantized(keys: prefillK, values: prefillV)
            _ = plain.updateQuantized(keys: prefillK, values: prefillV)
            try Self.assertCacheStatesEqual(fused, plain, label: "after 250-token prefill")

            for step in 1 ... 10 {
                let (k, v) = Self.randomKV(steps: 1, seed: UInt64(6_000 + step))
                _ = fused.updateQuantized(keys: k, values: v)
                _ = plain.updateQuantized(keys: k, values: v)
            }
            try Self.assertCacheStatesEqual(fused, plain, label: "after crossing 256 boundary (260)")

            #expect(fused.offset == 260)
            // Only the 10 single-token appends went through the fused kernel; the
            // 250-token prefill is a single multi-step update (steps > 8 -> ladder).
            #expect(fused.attentionDiagnostics.fusedAppendCount == 10)
            #expect(fused.attentionDiagnostics.fusedAppendFallbackReason == "stepsOutOfRange")
        }

        // MARK: - (c) Guard fallback observability

        @Test func residualConfigFallsBackToLadderObservably() throws {
            guard Self.gpuAvailable() else { return }
            guard turboQuantNativeQuantizeAppendKVAvailable() else { return }

            // Residual affine only supports V2/R1; construct that geometry with
            // the fused flag ON. Every append must hit the residualsEnabled guard.
            let fused = AffineK8V4KVCache(
                valueGroupSize: 64, valueBits: 2, residualsPerGroup: 1,
                fusedQuantizeAppend: true)
            let plain = AffineK8V4KVCache(
                valueGroupSize: 64, valueBits: 2, residualsPerGroup: 1,
                fusedQuantizeAppend: false)

            for step in 1 ... 8 {
                let (k, v) = Self.randomKV(steps: 1, seed: UInt64(7_000 + step))
                _ = fused.updateQuantized(keys: k, values: v)
                _ = plain.updateQuantized(keys: k, values: v)
            }
            try Self.assertCacheStatesEqual(fused, plain, label: "residual fallback")

            #expect(fused.attentionDiagnostics.fusedAppendCount == 0)
            #expect(fused.attentionDiagnostics.fusedAppendFallbackCount == 8)
            #expect(fused.attentionDiagnostics.fusedAppendFallbackReason == "residualsEnabled")
        }

        // MARK: - (d) Attention-output smoke through the normal read path

        @Test func fusedCacheReadPathMatchesLadderAttentionOutput() throws {
            guard Self.gpuAvailable() else { return }
            guard turboQuantNativeQuantizeAppendKVAvailable() else { return }

            let fused = Self.makeCache(fused: true)
            let plain = Self.makeCache(fused: false)

            // Seed both caches with an identical short sequence.
            for step in 1 ... 6 {
                let (k, v) = Self.randomKV(steps: 1, seed: UInt64(8_000 + step))
                _ = fused.updateQuantized(keys: k, values: v)
                _ = plain.updateQuantized(keys: k, values: v)
            }
            try Self.assertCacheStatesEqual(fused, plain, label: "pre-attention seed")

            // One decode-shaped query per cache through the normal cache read.
            MLXRandom.seed(9_001)
            let queries = MLXRandom.normal([1, 4, 1, Self.headDim]).asType(.float16)
            let (nextK, nextV) = Self.randomKV(steps: 1, seed: 9_100)
            let scale = 1 / sqrt(Float(Self.headDim))

            let fusedResult = try attentionWithCacheUpdateReturningStateThrowing(
                queries: queries, keys: nextK, values: nextV,
                cache: fused, scale: scale, mask: .causal)
            let plainResult = try attentionWithCacheUpdateReturningStateThrowing(
                queries: queries, keys: nextK, values: nextV,
                cache: plain, scale: scale, mask: .causal)

            #expect(fusedResult.output.shape == plainResult.output.shape)
            Self.bytesEqual(fusedResult.output, plainResult.output, "attention output")
            // The extra decode-step append also engaged the fused kernel.
            #expect((fused.attentionDiagnostics.fusedAppendCount ?? 0) >= 7)
            #expect(fused.attentionDiagnostics.fusedAppendFallbackReason == nil)
        }

        // MARK: - (e) Fail-closed: a native op error surfaces as a Swift throw

        /// An unsupported bit width makes the native op throw at construction.
        /// The binding must surface that as a Swift `throw` (not unpack an empty
        /// result vector and trap out of bounds), so the cache can fall back to
        /// the ladder instead of crashing.
        @Test func nativeFailureThrowsInsteadOfTrapping() throws {
            guard Self.gpuAvailable() else { return }
            let capacity = 8
            let (kNew, vNew) = Self.randomKV(steps: 1, seed: 202)
            let kCodes = Self.zeroPlane(Self.keyCodeWords, capacity: capacity, dtype: .uint32)
            let kScales = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let kBiases = Self.zeroPlane(Self.keyGroups, capacity: capacity, dtype: .float16)
            let vCodes = Self.zeroPlane(Self.valueCodeWords, capacity: capacity, dtype: .uint32)
            let vScales = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)
            let vBiases = Self.zeroPlane(Self.valueGroups, capacity: capacity, dtype: .float16)

            #expect(throws: (any Error).self) {
                _ = try MLXFast.quantizeAppendKV(
                    keysNew: kNew,
                    valuesNew: vNew,
                    kCodes: kCodes,
                    kScales: kScales,
                    kBiases: kBiases,
                    vCodes: vCodes,
                    vScales: vScales,
                    vBiases: vBiases,
                    seqOffset: 0,
                    steps: 1,
                    keyGroupSize: Self.keyGroupSize,
                    keyBits: 4,  // unsupported for keys (only key_bits=8) -> native throw
                    valueGroupSize: Self.valueGroupSize,
                    valueBits: Self.valueBits)
            }
        }
    }
}
