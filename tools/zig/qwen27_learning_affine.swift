import Foundation
import Metal

let root = URL(fileURLWithPath: CommandLine.arguments[1])
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.languageVersion = .version4_1
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
var source = try String(contentsOf: root.appendingPathComponent("zig/kernels/metal/core/learn_linear.metal"), encoding: .utf8)
if CommandLine.arguments.contains("--reorder-control") {
    source = source.replacingOccurrences(of: "for (uint out = 0; out < n; out++) {", with: "for (uint step = 0; step < n; step++) { const uint out = n - 1 - step;")
}
let reference = #"""
#include <metal_stdlib>
using namespace metal;
struct learn_shape { uint4 u; float4 f; };

// A raw affine4 matrix's input gradient in full fp32, in output-column order; no relaxed float MMA operand.
[[max_total_threads_per_threadgroup(256)]]
kernel void tf_learn_q4_scalar(const device float* grad [[buffer(0)]], const device uint* words [[buffer(1)]],
                             const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                             device float* dx [[buffer(4)]], constant learn_shape& p [[buffer(5)]],
                             uint i [[thread_position_in_grid]]) {
  const uint rows = p.u.x, n = p.u.y, k = p.u.z;
  if (i >= rows * k) return;
  const uint row = i / k, column = i % k;
  float sum = 0.0f;
  for (uint out = 0; out < n; out++) {
    const size_t at = size_t(out) * k + column;
    const uint q = (words[at / 8] >> (4 * (column & 7))) & 15;
    const size_t group = size_t(out) * (k / 64) + column / 64;
    const float weight = float(scales[group]) * float(q) + float(biases[group]);
    sum = fma(grad[size_t(row) * n + out], weight, sum);
  }
  dx[i] = p.u.w != 0 ? dx[i] + sum : sum;
}

"""#
let fast = try device.makeLibrary(source: source, options: options)
let scalar = try device.makeLibrary(source: reference, options: options)
let pipelines = try [fast.makeFunction(name: "tf_learn_q4_back")!, scalar.makeFunction(name: "tf_learn_q4_scalar")!].map {
    try device.makeComputePipelineState(function: $0)
}
func bytes<T>(_ a: [T]) -> Data { a.withUnsafeBytes { Data($0) } }
func buffer(_ d: Data) -> MTLBuffer {
    var guarded = d
    guarded.append(Data(repeating: 0xa5, count: 64))
    return guarded.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: guarded.count, options: .storageModeShared)! }
}
func snapshot(_ b: MTLBuffer) -> Data { Data(bytes: b.contents(), count: b.length) }
func bf(_ x: Float) -> UInt16 {
    let u = x.bitPattern
    return UInt16(truncatingIfNeeded: (u &+ 0x7fff &+ ((u >> 16) & 1)) >> 16)
}
var cases = 0
var shapes = [1, 3, 4, 7, 8, 17].flatMap { rows in [17, 257, 521].flatMap { n in [64, 128].map { k in (rows, n, k, false) } } }
shapes += [(1024, 17, 64, false), (7, 248320, 64, false), (7, 17, 17408, false), (7, 257, 128, true)]
for (rows, n, k, plant) in shapes { for add in [0, 1] {
    var gradient: [Float] = (0..<rows*n).map { i in
        if i < n { return i % 2 == 0 ? 0.0 : -0.0 }
        return Float((i * 17) % 127 - 63) / 8192 + (i % 5 == 0 ? 0.999267578125 : -0.00317478179931640625)
    }
    if plant {
        gradient[2*n] = .nan
        gradient[3*n] = .infinity
        gradient[4*n] = -.infinity
        gradient[5*n] = .infinity
        gradient[5*n+1] = -.infinity
    }
    let words = (0..<n*k/8).map { UInt32(truncatingIfNeeded: $0 &* 1550719 &+ 191) }
    let scales = (0..<n*k/64).map { bf(Float(($0 * 3) % 17 + 1) / 512) }
    let biases = (0..<n*k/64).map { bf(Float(($0 * 7) % 29 - 14) / 128) }
    let inputs = [buffer(bytes(gradient)), buffer(bytes(words)), buffer(bytes(scales)), buffer(bytes(biases))]
    let before = inputs.map(snapshot)
    let initial = bytes((0..<rows*k).map { Float(($0 * 11) % 31 - 15) / 128 })
    var first: Data? = nil
    for _ in 0..<5 {
        let outputs = [buffer(initial), buffer(initial)]
        for which in 0..<2 {
            let cb = queue.makeCommandBuffer()!, e = cb.makeComputeCommandEncoder()!
            e.setComputePipelineState(pipelines[which])
            for (i, b) in (inputs + [outputs[which]]).enumerated() { e.setBuffer(b, offset: 0, index: i) }
            let p = bytes([UInt32(rows), UInt32(n), UInt32(k), UInt32(add)]) + bytes([Float](repeating: 0, count: 4))
            p.withUnsafeBytes { e.setBytes($0.baseAddress!, length: p.count, index: 5) }
            let count = (which == 0 ? (rows+3)/4 : rows) * k
            e.dispatchThreadgroups(MTLSize(width: (count+255)/256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            if let error = cb.error { throw error }
        }
        let got = snapshot(outputs[0]), want = snapshot(outputs[1])
        precondition(got == want, "affine bit mismatch rows=\(rows) n=\(n) k=\(k) add=\(add)")
        precondition(got.suffix(64) == Data(repeating: 0xa5, count: 64), "guard overwritten")
        if let first { precondition(got == first, "repeat changed") } else { first = got }
        precondition(inputs.map(snapshot) == before, "input changed")
    }
    cases += 1
} }
print("affine scalar/tiled bit equality: \(cases) cases, five repeats, add/replace, zero/nonfinite rows, tails, guards and unchanged inputs")
