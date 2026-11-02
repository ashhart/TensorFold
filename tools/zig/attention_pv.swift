import Foundation
import Metal
import CryptoKit

func replaceOnce(_ source: String, _ old: String, _ new: String) -> String {
    precondition(source.components(separatedBy: old).count == 2, "SourceAnchor: \(old)")
    return source.replacingOccurrences(of: old, with: new)
}
func expand(_ path: URL) throws -> String {
    let text = try String(contentsOf: path, encoding: .utf8)
    return try text.components(separatedBy: "\n").map { line in
        if line.hasPrefix("#include \"") {
            let name = line.components(separatedBy: "\"")[1]
            return try expand(path.deletingLastPathComponent().appendingPathComponent(name))
        }
        return line
    }.joined(separator: "\n")
}
func json(_ value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .prettyPrinted])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([10]))
}
func bf16(_ value: Double) -> UInt16 {
    let word = Float(value).bitPattern
    return UInt16(truncatingIfNeeded: (word &+ 0x7fff &+ ((word >> 16) & 1)) >> 16)
}
func checked(_ command: MTLCommandBuffer) {
    command.commit()
    command.waitUntilCompleted()
    precondition(command.status == .completed, "CommandFailed: \(String(describing: command.error))")
}
let args = CommandLine.arguments
precondition(args.count >= 3, "attention_pv SOURCE_ROOT check|bench [rows] [red]")
let root = URL(fileURLWithPath: args[1])
let mode = args[2]
let expectRed = args.last == "red"
let selection = ProcessInfo.processInfo.environment["TF_ATTENTION_PV_FP32"] ?? "0"
precondition(["0", "1"].contains(selection), "TF_ATTENTION_PV_FP32 must be 0 or 1")
let attention = (selection == "1" ? "#define TF_ATTN_PV_FP32 1\n" : "") +
    (try expand(root.appendingPathComponent("zig/kernels/metal/prefill/attention_nax.metal")))
guard let device = MTLCreateSystemDefaultDevice(), device.supportsFamily(.apple10) else {
    fatalError("RequiresM5TensorUnits")
}
let queue = device.makeCommandQueue()!
let options = MTLCompileOptions()
options.languageVersion = .version4_1
options.mathMode = .safe
options.mathFloatingPointFunctions = .fast
func library(_ source: String) throws -> MTLLibrary { try device.makeLibrary(source: source, options: options) }
func pipeline(_ library: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
func buffer<T>(_ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
}
func parameters(_ rows: Int, _ keys: Int) -> [Int32] {
    let prefix = [1, 1, rows, keys, 1, (rows + 63) / 64, (keys + 31) / 32,
                  rows / 64, keys / 32, rows % 64, keys % 32, keys - rows]
    return (prefix + [rows*128, rows*128, 128, keys*128, keys*128, 128,
                      keys*128, keys*128, 128, rows*128, rows*128, 128]).map(Int32.init)
}
func dispatch(_ pipe: MTLComputePipelineState, _ buffers: [MTLBuffer], _ groups: Int, _ command: MTLCommandBuffer) {
    let enc = command.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    enc.endEncoding()
}
let sentinel: UInt32 = 0x13579bdf
if mode == "check" {
    let body = attention.components(separatedBy: "// ---- entry points:")[0]
    var instrumented = replaceOnce(body, "device T* O, const device int* P", "device float* O, const device int* P")
    instrumented = replaceOnce(instrumented, "s[f][e] = fast::exp2(s[f][e] - top_new[e >> 2]);",
        "s[f][e] = s[f][e] == kMasked ? 0.0f : (kb*BK + 16*f + home.x + TF_COL(e) == int(F[3]) ? F[1] : (kb*BK + 16*f + home.x + TF_COL(e) == int(F[3])+1 ? F[2] : 0.0f));")
    let signature = """
    kernel void NAME(const device bfloat* Q [[buffer(0)]], const device bfloat* K [[buffer(1)]],
      const device bfloat* V [[buffer(2)]], const device int* P [[buffer(3)]], const device float* F [[buffer(4)]],
      device float* O [[buffer(5)]], uint3 tg [[threadgroup_position_in_grid]],
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
      tfp::FUNCTION<bfloat, AQ, AK, CAUSAL>(Q, K, V, O, P, F, tg, sg, lane);
    }
    """
    var shader = replaceOnce(instrumented, "device float* O, const device int* P", "device T* O, const device int* P")
    let raw = replaceOnce(instrumented, "inline void attention_nax(", "inline void attention_raw(")
        .replacingOccurrences(of: "acc[i][e] = acc[i][e] * inv[e >> 2];", with: "acc[i][e] = acc[i][e];")
    let rawBody = raw.components(separatedBy: "// One 64-query block")[1]
    shader += "\nnamespace tfp {\n// One 64-query block" + rawBody
    for aq in [false, true] { for ak in [false, true] { for causal in [false, true] { for raw in [false, true] {
        let name = "pv_\(aq ? 1 : 0)\(ak ? 1 : 0)\(causal ? 1 : 0)\(raw ? 1 : 0)"
        let sig = raw ? signature : signature.replacingOccurrences(of: "device float* O", with: "device bfloat* O")
        shader += sig.replacingOccurrences(of: "NAME", with: name)
            .replacingOccurrences(of: "FUNCTION", with: raw ? "attention_raw" : "attention_nax")
            .replacingOccurrences(of: "AQ", with: String(aq)).replacingOccurrences(of: "AK", with: String(ak))
            .replacingOccurrences(of: "CAUSAL", with: String(causal))
    } } } }
    let lib = try library(shader)
    let scalarText = try expand(root.appendingPathComponent("zig/kernels/metal/core/shared_attention.metal"))
    let scalar = replaceOnce(scalarText, "weight[t] = precise::exp(score[t] - next);",
        "weight[t] = first+t == 0 ? 1.0f : (first+t == 1 ? 0.999267578125f : 0.0f);")
    let scalarLib = try library("#define TF_D 128\n#define TF_QH 1\n#define TF_KVH 1\n#define TF_RD 0\n#define TF_OUT float\n" + scalar)
    let scalarPipe = try pipeline(scalarLib, "tf_shared_lane_attention")
    var records = [[String: Any]](), mismatches = 0, comparisons = 0, repeats = 0, guards = 0, unchanged = 0
    let cases: [(String, Float, Float, Float)] = [
        ("catch", 1, 0.999267578125, 1),
        ("third-part", 1, Float(bitPattern: 0x3f7f1234), 1),
        ("small-residual", Float(bitPattern: 0x03800001), Float(bitPattern: 0x03800000), Float(pow(2.0, 120))),
        ("exact-cancel", 1, 1, 1)]
    for (label, a, b, v) in cases {
        for (rows, keys, first) in [(1,2,0), (17,33,31), (64,64,0), (65,65,31)] {
            let aq = rows % 64 == 0, ak = keys % 32 == 0
            for causal in [false, true] { for raw in [false, true] {
                let name = "pv_\(aq ? 1 : 0)\(ak ? 1 : 0)\(causal ? 1 : 0)\(raw ? 1 : 0)"
                let pipe = try pipeline(lib, name)
                let q = buffer([UInt16](repeating: 0, count: rows*128))
                let k = buffer([UInt16](repeating: 0, count: keys*128))
                var values = [UInt16](repeating: 0, count: keys*128)
                for c in 0..<128 { values[first*128+c] = bf16(Double(v)); values[(first+1)*128+c] = bf16(-Double(v)) }
                let vb = buffer(values), pb = buffer(parameters(rows, keys)), fb = buffer([Float(0),a,b,Float(first)])
                let out = buffer([UInt32](repeating: sentinel, count: rows*128+16))
                let inputs = [q,k,vb,pb,fb]
                let saved = inputs.map { Data(bytes: $0.contents(), count: $0.length) }
                var firstWords: [UInt32]? = nil, bad = 0
                for _ in 0..<5 {
                    let command = queue.makeCommandBuffer()!
                    dispatch(pipe, inputs+[out], (rows+63)/64, command)
                    checked(command)
                    let bits = out.contents().bindMemory(to: UInt32.self, capacity: rows*128+16)
                    let bf = out.contents().bindMemory(to: UInt16.self, capacity: (rows*128+16)*2)
                    for row in 0..<rows {
                        let bound = causal ? row+keys-rows : keys-1
                        let pa = first <= bound ? Double(a) : 0, pbb = first+1 <= bound ? Double(b) : 0
                        let numerator = pa*Double(v)-pbb*Double(v)
                        let ref = raw ? numerator : (pa+pbb == 0 ? 0 : numerator/(pa+pbb))
                        for c in 0..<128 {
                            let equal = !raw && pa+pbb == 0 ? (bf[row*128+c] & 0x7f80 == 0x7f80 && bf[row*128+c] & 0x7f != 0) : (raw ? bits[row*128+c] == Float(ref).bitPattern : bf[row*128+c] == bf16(ref))
                            if !equal { bad += 1 }
                            comparisons += 1
                        }
                    }
                    let count = raw ? rows*128 : rows*64
                    let words = Array(UnsafeBufferPointer(start: bits, count: count))
                    if let prior = firstWords { precondition(words == prior, "RepeatChanged") } else { firstWords = words }
                    repeats += 1
                    for i in count..<count+16 { precondition(bits[i] == sentinel, "GuardChanged"); guards += 1 }
                    for (i,input) in inputs.enumerated() {
                        precondition(saved[i] == Data(bytes: input.contents(), count: input.length), "InputChanged")
                        unchanged += input.length
                    }
                }
                mismatches += bad
                records.append(["case":label,"rows":rows,"keys":keys,"first":first,"causal":causal,"raw":raw,"mismatches":bad])
            } }
        }
    }
    let q = buffer([UInt16](repeating: 0, count: 128)), k = buffer([UInt16](repeating: 0, count: 256))
    let v = buffer([UInt16](repeating: 0x3f80,count:128)+[UInt16](repeating:0xbf80,count:128))
    let mask = buffer([UInt8](repeating:1,count:2)), meta = buffer([UInt32(1),2,2,1,0])
    let out = buffer([UInt32](repeating:sentinel,count:144))
    let scalarCmd = queue.makeCommandBuffer()!
    let enc = scalarCmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(scalarPipe)
    for (i,b) in [q,k,v,mask,meta,out].enumerated() { enc.setBuffer(b,offset:0,index:i) }
    enc.dispatchThreadgroups(MTLSize(width:1,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:32,height:1,depth:1))
    enc.endEncoding(); checked(scalarCmd)
    let scalarValues = out.contents().bindMemory(to:Float.self,capacity:144)
    let scalarExpected = bf16((1.0-0.999267578125)/(1.0+0.999267578125))
    for i in 0..<128 { precondition(bf16(Double(scalarValues[i])) == scalarExpected,"ScalarMismatch") }
    for i in 128..<144 { precondition(scalarValues[i].bitPattern==sentinel,"ScalarGuardChanged") }
    try json(["mode":"check","precision":selection,"device":device.name,"os":ProcessInfo.processInfo.operatingSystemVersionString,
      "comparisons":comparisons,"mismatches":mismatches,"dispatches":repeats,"guards":guards,
      "unchanged_input_bytes_checked":unchanged,"scalar_bf16_hex":String(scalarExpected,radix:16),
      "scalar_comparisons":128,"expected_red":expectRed,"records":records])
    exit(expectRed ? (mismatches > 0 ? 0 : 1) : (mismatches == 0 ? 0 : 1))
} else {
    precondition(mode == "bench" && args.count >= 4)
    let rows = Int(args[3])!
    precondition([2048,8192,32768].contains(rows))
    let lib = try library(attention)
    let name = "custom_kernel_tf_attention_nax_bf16_t_t_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t"
    let pipe = try pipeline(lib,name)
    let q = buffer((0..<rows*128).map { bf16(Double(($0*17)%31-15)/32) })
    let k = buffer((0..<rows*128).map { bf16(Double(($0*11)%29-14)/32) })
    let v = buffer((0..<rows*128).map { bf16(Double(($0*7)%23-11)/16) })
    let p = buffer(parameters(rows,rows)), f = buffer([Float(1/sqrt(128.0))])
    let out = buffer([UInt16](repeating:0,count:rows*128))
    var times = [Double]()
    for _ in 0..<4 {
        let command = queue.makeCommandBuffer()!
        dispatch(pipe,[q,k,v,p,f,out],rows/64,command)
        checked(command)
        times.append(command.gpuEndTime-command.gpuStartTime)
    }
    try json(["mode":"bench","precision":selection,"device":device.name,"os":ProcessInfo.processInfo.operatingSystemVersionString,
      "rows":rows,"heads":1,"dimension":128,"causal":true,"gpu_seconds":times,
      "output_sha256":SHA256.hash(data:Data(bytes:out.contents(),count:out.length)).map{String(format:"%02x",$0)}.joined(),
      "max_threads":pipe.maxTotalThreadsPerThreadgroup,"static_tg_bytes":pipe.staticThreadgroupMemoryLength,
      "note":"first sample warmup, remaining three timed; no model loaded"])
}
