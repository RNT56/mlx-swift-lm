// Copyright © 2026 Apple Inc.
//
// N3 gating microbench: does the quantized lm_head latency at batch=1 SCALE with the
// output (vocab) dimension, or is it occupancy/launch-floored?
//
// The tied lm_head is ~30% of the decode weight stream (Qwen3-4B: 151936 vocab × 2560
// hidden) and the N1 forward-scaling probe showed it is part of why the full-model
// forward scales with q_seq. A candidate-subset / banked head (compute logits for only
// K candidate tokens) is only worth building (with an exactness guard) IF a smaller
// output is proportionally faster at batch=1. If batch=1 head latency is ~flat until
// large N, the head is occupancy-bound — the same wall that made KV byte-cuts neutral —
// and N3 dies. NO product claim: this is a kernel gating decision.

import Foundation
import MLX

private func arg(_ name: String, default def: String? = nil) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), a.indices.contains(i + 1) else { return def }
    return a[i + 1]
}
private func argInt(_ name: String, _ def: Int) -> Int { arg(name).flatMap(Int.init) ?? def }
private func argInts(_ name: String, _ def: [Int]) -> [Int] {
    guard let raw = arg(name) else { return def }
    let v = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return v.isEmpty ? def : v
}
private func median(_ xs: [Double]) -> Double {
    let s = xs.sorted()
    guard !s.isEmpty else { return 0 }
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}
private func nowMs() -> Double { Date.timeIntervalSinceReferenceDate * 1000.0 }

let hidden = argInt("--hidden", 2560)
let groupSize = argInt("--group-size", 64)
let bits = argInt("--bits", 4)
let batch = argInt("--batch", 1)
let vocab = argInt("--vocab", 151936)
let sizes = argInts("--sizes", [2048, 8192, 16384, 32768, 65536, 131072, 151936])
let candidate = argInt("--candidate", 8192)
let iters = argInt("--iterations", 20)
let warmup = argInt("--warmup", 5)

guard Device.defaultDevice().deviceType == .gpu else {
    FileHandle.standardError.write(Data("GPU device required\n".utf8)); exit(3)
}

func makeHead(_ rows: Int, seed: UInt64) -> (wq: MLXArray, scales: MLXArray, biases: MLXArray?) {
    MLXRandom.seed(seed)
    let w = (MLXRandom.normal([rows, hidden]) * 0.02).asType(.float16)
    let q = quantized(w, groupSize: groupSize, bits: bits, mode: .affine)
    eval(q.wq, q.scales, q.biases ?? q.wq)
    Stream().synchronize()
    return q
}

func timeMatmul(_ x: MLXArray, _ h: (wq: MLXArray, scales: MLXArray, biases: MLXArray?)) -> Double {
    func once() {
        let o = quantizedMatmul(
            x, h.wq, scales: h.scales, biases: h.biases,
            transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
        eval(o)
    }
    for _ in 0 ..< warmup { once(); Stream().synchronize() }
    var samples: [Double] = []
    for _ in 0 ..< iters {
        let t0 = nowMs()
        once(); Stream().synchronize()
        samples.append(nowMs() - t0)
    }
    return median(samples)
}

let x = (MLXRandom.normal([batch, hidden]) * 0.02).asType(.float16)
eval(x); Stream().synchronize()

// ---- Scaling sweep: batch=1 quantized head latency vs output size N ----
print("hidden=\(hidden) groupSize=\(groupSize) bits=\(bits) batch=\(batch)  (iters=\(iters), median ms)")
print("\n# Scaling: quantized head latency vs output (vocab) size")
print("   N        ms      us/1k-rows   ms/ms(N=min)")
print("--------   ------   ----------   ------------")
var sweep: [(n: Int, ms: Double)] = []
for N in sizes.sorted() {
    let h = makeHead(N, seed: 0xBEEF &+ UInt64(N))
    let ms = timeMatmul(x, h)
    sweep.append((N, ms))
}
let baseMs = sweep.first?.ms ?? 1
let baseN = sweep.first?.n ?? 1
for s in sweep {
    let usPerK = s.ms * 1000.0 / (Double(s.n) / 1000.0)
    print(
        [
            String(format: "%7d", s.n), String(format: "%6.3f", s.ms),
            String(format: "%10.2f", usPerK), String(format: "%11.2fx", s.ms / baseMs),
        ].joined(separator: "   "))
}
// Floor + marginal decomposition (the endpoint ratio alone is misleading: a fixed
// launch/occupancy FLOOR dominates at small N, the per-row MARGINAL dominates at large
// N). Fit floor + ns/row from the two largest points (the output-bound regime).
if sweep.count >= 2 {
    let a = sweep[sweep.count - 2]
    let b = sweep[sweep.count - 1]
    let marginalNsPerRow = (b.ms - a.ms) * 1e6 / Double(b.n - a.n)
    let floorMs = b.ms - marginalNsPerRow * Double(b.n) / 1e6
    let marginalFracAtMax = (b.ms - floorMs) / b.ms
    print(
        String(
            format:
                "\nfixed floor ≈ %.3f ms + marginal ≈ %.2f ns/row. At N=%d the marginal is "
                + "%.0f%% of head latency => %@.",
            floorMs, marginalNsPerRow, b.n, marginalFracAtMax * 100,
            marginalFracAtMax > 0.5
                ? "OUTPUT-BOUND at full vocab (a smaller output pays)"
                : "FLOOR-BOUND (a smaller output barely helps)"))
}

// ---- Realism: candidate-subset head from the full vocab (gather included) ----
print("\n# Candidate-subset realism (full vocab=\(vocab), K=\(candidate))")
let full = makeHead(vocab, seed: 0xF00D)
let fullMs = timeMatmul(x, full)
// Strided candidate set across the vocab.
let stride = Swift.max(1, vocab / candidate)
let idx = MLXArray((0 ..< candidate).map { Int32(($0 * stride) % vocab) })
eval(idx); Stream().synchronize()
// Pre-taken subset (gather EXCLUDED from the timer) — isolates the matmul cost.
let subWq = full.wq.take(idx, axis: 0)
let subScales = full.scales.take(idx, axis: 0)
let subBiases = full.biases?.take(idx, axis: 0)
eval(subWq, subScales, subBiases ?? subWq); Stream().synchronize()
let subMatmulMs = timeMatmul(x, (subWq, subScales, subBiases))
// Gather INCLUDED (realistic per-step cost if the candidate set changes each step).
let gatherOnce: () -> Void = {
    let gwq = full.wq.take(idx, axis: 0)
    let gsc = full.scales.take(idx, axis: 0)
    let gbi = full.biases?.take(idx, axis: 0)
    let o = quantizedMatmul(
        x, gwq, scales: gsc, biases: gbi,
        transpose: true, groupSize: groupSize, bits: bits, mode: .affine)
    eval(o)
}
for _ in 0 ..< warmup { gatherOnce(); Stream().synchronize() }
var gatherSamples: [Double] = []
for _ in 0 ..< iters {
    let t0 = nowMs()
    gatherOnce(); Stream().synchronize()
    gatherSamples.append(nowMs() - t0)
}
let gatherMatmulMs = median(gatherSamples)

print("   full head (\(vocab))             : \(String(format: "%.3f", fullMs)) ms")
print("   subset matmul only (\(candidate))     : \(String(format: "%.3f", subMatmulMs)) ms  (\(String(format: "%.2f", fullMs / subMatmulMs))x vs full)")
print("   subset gather+matmul (\(candidate))   : \(String(format: "%.3f", gatherMatmulMs)) ms  (\(String(format: "%.2f", fullMs / gatherMatmulMs))x vs full)")

let viable = gatherMatmulMs < fullMs * 0.9
print(
    "\n=== N3 verdict: \(viable ? "VIABLE" : "DIES") — candidate-subset head is "
        + "\(String(format: "%.2f", fullMs / gatherMatmulMs))x the full head at batch=1 "
        + "(gather included). \(viable ? "Worth building with an exactness guard." : "Occupancy-floored: smaller output does not pay; stop (same wall as KV byte-cuts).") ===")

if let outputPath = arg("--output") {
    struct Report: Codable {
        var schemaVersion = 1
        var hidden: Int; var groupSize: Int; var bits: Int; var batch: Int; var vocab: Int
        var iterations: Int
        var sweep: [SweepRow]
        var fullMs: Double; var candidate: Int; var subsetMatmulMs: Double; var subsetGatherMatmulMs: Double
        var verdictViable: Bool
    }
    struct SweepRow: Codable { var n: Int; var ms: Double }
    let report = Report(
        hidden: hidden, groupSize: groupSize, bits: bits, batch: batch, vocab: vocab,
        iterations: iters, sweep: sweep.map { SweepRow(n: $0.n, ms: $0.ms) },
        fullMs: fullMs, candidate: candidate, subsetMatmulMs: subMatmulMs,
        subsetGatherMatmulMs: gatherMatmulMs, verdictViable: viable)
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    try enc.encode(report).write(to: URL(fileURLWithPath: outputPath), options: .atomic)
    print("\nwrote report: \(outputPath)")
}
