// Per-visit timing harness for the decode attention kernel (derived from the prefill bench used while tuning it).
// Build: swiftc -O attn_decode_bench.swift -o attn_decode_bench. Usage: [NV=visits] attn_decode_bench BASE_DIR CAND_DIR POS0 ROWS WINDOW REPS
// (the dirs hold common.metal and attention.metal; WINDOW 0 is a global layer, 1024 a sliding one).
import Foundation
import Metal

func bf16(_ v: Float) -> UInt16 { let w = v.bitPattern; return UInt16(truncatingIfNeeded: (w &+ 0x7fff &+ ((w >> 16) & 1)) >> 16) }
struct Args { var rows: UInt32; var pos0: UInt32; var heads: UInt32; var kv: UInt32; var extent: UInt32; var window: UInt32; var ring: UInt32; var chunks: UInt32; var scale: Float; var p0: UInt32 = 0; var p1: UInt32 = 0; var p2: UInt32 = 0 }
let env = ProcessInfo.processInfo.environment
let a = CommandLine.arguments
// usage: bench baseDir candDir pos0 rows window(0|1024) reps [rpt]
let baseDir = a[1], candDir = a[2]
let pos0 = UInt32(a[3])!, rows = UInt32(a[4])!, window = UInt32(a[5])!, reps = Int(a[6])!
let rpt = a.count > 7 ? UInt32(a[7])! : 1   // rows per threadgroup in the candidate's dispatch
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
func lib(_ dir: String, _ pre: [String: NSObject] = [:]) -> MTLLibrary {
  let c = try! String(contentsOfFile: dir + "/common.metal"), at = try! String(contentsOfFile: dir + "/attention.metal")
  let o = MTLCompileOptions(); o.languageVersion = MTLLanguageVersion(rawValue: (4 << 16) | 1)!; o.mathMode = .safe; o.preprocessorMacros = pre
  return try! dev.makeLibrary(source: c + "\n" + at, options: o)
}
var pre: [String: NSObject] = [:]
for kv in (env["PRE"] ?? "").split(separator: ",") { let p = kv.split(separator: "="); pre[String(p[0])] = NSNumber(value: p.count > 1 ? Int(p[1])! : 1) }
let lb = lib(baseDir), lc = lib(candDir, pre)
func pipe(_ l: MTLLibrary, _ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: l.makeFunction(name: n)!) }
let heads: UInt32 = 24, kv: UInt32 = 8, G = 3
let ring: UInt32 = window == 0 ? 0 : 1536
let last = pos0 + rows
let keys = window != 0 ? min(last, window) : last
let chunks = (keys + 2047) / 2048
let slots = window == 0 ? Int(last) + 8 : Int(ring)
var rng = SystemRandomNumberGenerator()
var seed: UInt64 = 12345
func rnd() -> Float { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Float(Int64(bitPattern: seed >> 11) & 0xfffff) / Float(0x80000) - 1 }
func buf(_ n: Int, _ sc: Float) -> MTLBuffer {
  let b = dev.makeBuffer(length: n * 2, options: .storageModeShared)!
  let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
  for i in 0..<n { p[i] = bf16(rnd() * sc) }
  return b
}
let H = Int(heads), R = Int(rows)
let bq = buf(R * H * 128, 2.0), brel = buf(R * H * 16, 2.0), bsink = buf(R * H, 1.0), bgate = buf(R * H * 128, 1.0)
let NV = Int(env["NV"] ?? "8")!
let kc0 = buf(slots * Int(kv) * 128, 1.0), vc0 = buf(slots * Int(kv) * 128, 1.0)
var kcs: [MTLBuffer] = [kc0], vcs: [MTLBuffer] = [vc0]
for _ in 1..<NV { kcs.append(dev.makeBuffer(bytes: kc0.contents(), length: kc0.length, options: .storageModeShared)!); vcs.append(dev.makeBuffer(bytes: vc0.contents(), length: vc0.length, options: .storageModeShared)!) }
let proj = buf(16 * 1024, 0.5)
let perRow = H * Int(chunks) * 132 * 4
let scratchBytes = 64 << 20
let scratch = dev.makeBuffer(length: scratchBytes, options: .storageModePrivate)!
func run(_ l: MTLLibrary, rpt: UInt32, out: MTLBuffer) -> Double {
  var rowsMode = (l === lc && env["ROWS"] != nil)
  let part = pipe(l, rowsMode ? "inkling2_attn_rows_partial" : "inkling2_attn_partial_g3"), fin = pipe(l, rowsMode ? "inkling2_attn_rows_final" : "inkling2_attn_final_g3"), mrg = pipe(l, "inkling2_attn_merge")
  let block = chunks == 1 ? R : min(R, scratchBytes / perRow)
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  for vi in 0..<NV {
  var r0 = 0
  while r0 < R {
    let n = min(block, R - r0)
    var args = Args(rows: UInt32(n), pos0: pos0 + UInt32(r0), heads: heads, kv: kv, extent: 1024, window: window, ring: ring, chunks: chunks, scale: 0.08838834764831845)
    e.setComputePipelineState(chunks == 1 ? fin : part)
    e.setBuffer(bq, offset: r0 * H * 256, index: 0); e.setBuffer(brel, offset: r0 * H * 32, index: 1)
    e.setBuffer(bsink, offset: r0 * H * 2, index: 2); e.setBuffer(bgate, offset: r0 * H * 256, index: 3)
    e.setBuffer(kcs[vi], offset: 0, index: 4); e.setBuffer(vcs[vi], offset: 0, index: 5); e.setBuffer(proj, offset: 0, index: 6)
    e.setBuffer(out, offset: r0 * H * 256, index: 7); e.setBuffer(scratch, offset: 0, index: 8)
    let nrt = (n + Int(rpt) - 1) / Int(rpt)
    var grid = (l === lc && env["ORDER"] != nil) ? MTLSize(width: nrt, height: Int(kv), depth: Int(chunks)) : MTLSize(width: Int(chunks), height: Int(kv), depth: nrt)
    if l === lc, let rbs = env["RB"], let rb = Int(rbs) { let nrb = (n + rb - 1) / rb; args.p0 = UInt32(rb); args.p1 = UInt32(nrb); grid = MTLSize(width: rb, height: Int(chunks), depth: Int(kv) * nrb) }
    if rowsMode { grid = MTLSize(width: Int(chunks), height: env["ROWS"] == "3" ? Int(kv) : H, depth: (n + 7) / 8) }
    e.setBytes(&args, length: MemoryLayout<Args>.size, index: 9)
    e.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    if chunks > 1 {
      e.setComputePipelineState(mrg)
      e.setBuffer(bgate, offset: r0 * H * 256, index: 0); e.setBuffer(out, offset: r0 * H * 256, index: 1); e.setBuffer(scratch, offset: 0, index: 2)
      e.setBytes(&args, length: MemoryLayout<Args>.size, index: 3)
      e.dispatchThreadgroups(MTLSize(width: H, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    r0 += n
  }
  }
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  return (cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(NV)
}
let outA = dev.makeBuffer(length: R * H * 256, options: .storageModeShared)!, outB = dev.makeBuffer(length: R * H * 256, options: .storageModeShared)!
memset(outA.contents(), 0, R * H * 256); memset(outB.contents(), 0, R * H * 256)
_ = run(lb, rpt: 1, out: outA); _ = run(lc, rpt: rpt, out: outB)
var ta: [Double] = [], tb: [Double] = []
for _ in 0..<reps { ta.append(run(lb, rpt: 1, out: outA)); tb.append(run(lc, rpt: rpt, out: outB)); tb.append(0); tb.removeLast() }
let same = memcmp(outA.contents(), outB.contents(), R * H * 256) == 0
if !same { let x = outA.contents().bindMemory(to: UInt16.self, capacity: R * H * 128), y = outB.contents().bindMemory(to: UInt16.self, capacity: R * H * 128)
  var bad = 0; var rowsBad = Set<Int>(); var first = -1
  for i in 0..<(R * H * 128) { if x[i] != y[i] { bad += 1; rowsBad.insert(i / (H * 128)); if first < 0 { first = i } } }
  print("mismatch elems=\(bad) rows=\(rowsBad.count) first=\(first) (row \(first / (H * 128)) head \((first / 128) % H) d \(first % 128)) minrow=\(rowsBad.min() ?? -1) maxrow=\(rowsBad.max() ?? -1)") }
func med(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
print("pos0=\(pos0) rows=\(rows) win=\(window) chunks=\(chunks) base_ms=\(String(format: "%.3f", med(ta))) cand_ms=\(String(format: "%.3f", med(tb))) equal=\(same)")
