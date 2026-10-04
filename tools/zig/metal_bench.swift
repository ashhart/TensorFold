// GPU time per dispatch (command-buffer timestamps) of Metal kernels from a .metal source (compiled as MLX compiles
// its run-time kernels and the Zig engine will: safe math, fast fp32 functions, Metal 4.0) or a .metallib, plus each
// run's output bytes. Dev-time only.
// usage: metal_bench SPEC.json   (the spec format is written by tools/zig/check_mlx_ops.py)
import Foundation
import Metal

struct Spec: Decodable { let runs: [Run] }
struct Run: Decodable {
    struct Lib: Decodable { let source: String?; let metallib: String? }
    struct Const: Decodable { let index: Int; let type: String; let value: Double }
    struct Buf: Decodable {
        let slot: Int; let file: String?; let bytes: Int?; let i32: [Int32]?; let u64: [UInt64]?; let f32: [Float]?
        let copies: Int?; let out: String?
    }
    let label: String; let library: Lib; let function: String; let constants: [Const]?; let buffers: [Buf]
    let threadgroups: [Int]?; let threads: [Int]?; let threadgroup: [Int]; let reps: Int; let chain: Int
}

let spec = try JSONDecoder().decode(Spec.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
var libraries: [String: MTLLibrary] = [:]

func library(_ lib: Run.Lib) throws -> MTLLibrary {
    let key = lib.source ?? lib.metallib!
    if let found = libraries[key] { return found }
    let made: MTLLibrary
    if let path = lib.source {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        options.languageVersion = .version4_0
        made = try device.makeLibrary(source: String(contentsOfFile: path, encoding: .utf8), options: options)
    } else {
        made = try device.makeLibrary(URL: URL(fileURLWithPath: lib.metallib!))
    }
    libraries[key] = made
    return made
}

for run in spec.runs {
    let values = MTLFunctionConstantValues()
    for c in run.constants ?? [] {
        if c.type == "bool" {
            var v = c.value != 0
            values.setConstantValue(&v, type: .bool, index: c.index)
        } else {
            var v = Int32(c.value)
            values.setConstantValue(&v, type: .int, index: c.index)
        }
    }
    let fn = try library(run.library).makeFunction(name: run.function, constantValues: values)
    let pso = try device.makeComputePipelineState(function: fn)
    // (buffer, slot, bytes of one copy, copies)
    var bound: [(MTLBuffer, Int, Int, Int)] = []
    var outs: [(MTLBuffer, String, Int)] = []
    for b in run.buffers {
        var data: Data
        if let f = b.file { data = try Data(contentsOf: URL(fileURLWithPath: f), options: .alwaysMapped) }
        else if let v = b.i32 { data = v.withUnsafeBufferPointer { Data(buffer: $0) } }
        else if let v = b.u64 { data = v.withUnsafeBufferPointer { Data(buffer: $0) } }
        else if let v = b.f32 { data = v.withUnsafeBufferPointer { Data(buffer: $0) } }
        else { data = Data(count: b.bytes ?? 16) }
        let one = max(data.count, 16), copies = max(1, b.copies ?? 1)
        let buf = device.makeBuffer(length: one * copies, options: .storageModeShared)!
        data.withUnsafeBytes { src in
            for c in 0..<copies { memcpy(buf.contents() + c * one, src.baseAddress!, data.count) }
        }
        bound.append((buf, b.slot, one, copies))
        if let o = b.out { outs.append((buf, o, data.count)) }
    }
    let tg = MTLSize(width: run.threadgroup[0], height: run.threadgroup[1], depth: run.threadgroup[2])
    var times: [Double] = []
    var dispatch = 0
    for rep in 0..<run.reps {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for c in 0..<run.chain {
            for (buf, slot, one, copies) in bound { enc.setBuffer(buf, offset: (dispatch % copies) * one, index: slot) }
            dispatch += 1
            if let g = run.threadgroups {
                enc.dispatchThreadgroups(MTLSize(width: g[0], height: g[1], depth: g[2]), threadsPerThreadgroup: tg)
            } else {
                let g = run.threads!
                enc.dispatchThreads(MTLSize(width: g[0], height: g[1], depth: g[2]), threadsPerThreadgroup: tg)
            }
            if c + 1 < run.chain { enc.memoryBarrier(scope: .buffers) }
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if rep == 0 {
            for (buf, path, count) in outs {
                try Data(bytes: buf.contents(), count: count).write(to: URL(fileURLWithPath: path))
            }
        }
        times.append((cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(run.chain))
    }
    let warm = (times.count > 3 ? Array(times.dropFirst(3)) : times).sorted()
    print(String(format: "%@ %.3f %.3f %.3f", run.label, warm[warm.count / 2], warm.first!, warm.last!))
}
