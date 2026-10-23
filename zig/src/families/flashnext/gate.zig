//! GPU-gated lanes: the head's confidence in its drafts sets each GPU-side round's window width (1 to WMAX rows) on
//! the GPU, and the costly kernels leave the rows past it alone; the verdict reads only the rows the gate opened.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;

/// The widest gated window: the absorb's draft and one gated chain step (wider windows paid less than their head steps).
pub const WMAX = 3;

/// Gate words: the window's rows, whether the chain step runs (0/1), the kept drafts' probability product (f32 bits),
/// each draft's probability, then the chain step's row counts in the layouts its kernels read (dims, four streams, one row).
const ROWS = 0;
const RUN = 1;
const CUM = 2;
const PROB = 4;
const MD = 8;
const MD4 = 16;
const MD1 = 24;
const WORDS = 32;

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// A draft's probability over the head's draft vocabulary (the softmax's largest value), then the round's width:
    \\// c = (draft rows, step, threshold bits). Step 0 is the absorb's draft, step 1 the chain step's (skipped unless it ran).
    \\kernel void fz_gate(device const bfloat* L [[buffer(0)]], device int* G [[buffer(1)]], constant uint4& c [[buffer(2)]],
    \\    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  const uint step = c.y;
    \\  if (step > 0 && G[1] == 0) return;
    \\  threadgroup float red[32];
    \\  float m = -INFINITY;
    \\  for (uint i = t; i < c.x; i += 1024) m = max(m, float(L[i]));
    \\  m = simd_max(m);
    \\  if (lane == 0) red[sg] = m;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  m = simd_max(red[lane]);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  float s = 0.0f;
    \\  for (uint i = t; i < c.x; i += 1024) s += metal::exp(float(L[i]) - m);
    \\  s = simd_sum(s);
    \\  if (lane == 0) red[sg] = s;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t != 0) return;
    \\  float total = 0.0f;
    \\  for (int k = 0; k < 32; k++) total += red[k];
    \\  const float p = 1.0f / total, cum = (step == 0 ? 1.0f : as_type<float>(G[2])) * p;
    \\  const bool kept = cum >= as_type<float>(c.z);
    \\  if (step == 0) G[0] = kept ? 2 : 1;
    \\  else if (kept) G[0] = 3;
    \\  const int run = step == 0 && kept ? 1 : 0;
    \\  G[1] = run; G[8] = run; G[16] = 4 * run; G[17] = 16 * run; G[24] = run;
    \\  G[2] = as_type<int>(cum);
    \\  G[4 + step] = as_type<int>(p);
    \\}
;

pub const Gate = struct {
    buf: mtl.Buffer,
    pipe: mtl.Pipeline,
    tau: f32,
    n: usize, // the head's draft vocabulary

    fn at(g: *const Gate, word: usize) Buf {
        return .{ .b = g.buf, .off = word * 4 };
    }

    /// The window's verified rows (the window kernels' and the verdict's row count).
    pub fn rows(g: *const Gate) Buf {
        return g.at(ROWS);
    }

    /// The gated chain step's row count, dims (one row), dims over its four streams and its one-row MoE dims.
    pub fn run(g: *const Gate) Buf {
        return g.at(RUN);
    }
    pub fn md(g: *const Gate) Buf {
        return g.at(MD);
    }
    pub fn md4(g: *const Gate) Buf {
        return g.at(MD4);
    }
    pub fn md1(g: *const Gate) Buf {
        return g.at(MD1);
    }

    /// Before a reply's rounds: the first window (drafted on the host) runs whole.
    pub fn open(g: *const Gate) void {
        g.buf.slice(i32, WORDS)[ROWS] = WMAX;
    }

    /// The gate after draft `step` (0: the absorb's, 1: the chain step's) from the head's draft logits.
    pub fn encode(g: *const Gate, r: *Run, logits: Buf, step: u32) void {
        r.enc.setPipeline(g.pipe);
        r.enc.setBuffer(logits.b, logits.off, 0);
        r.enc.setBuffer(g.buf, 0, 1);
        const c = [4]u32{ @intCast(g.n), step, @bitCast(g.tau), 0 };
        r.enc.setBytes(std.mem.asBytes(&c), 2);
        r.enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        if (!r.serial) r.enc.barrier();
    }

    /// The probabilities the gate saw in the last round it read (dev log).
    pub fn probs(g: *const Gate) [2]f32 {
        const w = g.buf.slice(u32, WORDS);
        return .{ @bitCast(w[PROB]), @bitCast(w[PROB + 1]) };
    }
};

/// A gate keeping a chain while the drafts' probability product stays at `tau` or above, over `n` draft rows.
pub fn init(r: *Run, tau: f32, n: usize) !*Gate {
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, source, mtl.CompileOptions.mlx()));
    const g = try r.arena.create(Gate);
    g.* = .{ .buf = try r.buffer(WORDS * 4), .pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_gate", false)), .tau = tau, .n = n };
    const w = g.buf.slice(i32, WORDS);
    @memset(w, 0);
    w[MD + 1] = 16; // the dims layouts' second word, as mdims_w's
    w[MD1 + 1] = 16;
    g.open();
    return g;
}

test "gate words: the chain step's views sit where the kernel writes its row counts" {
    try std.testing.expect(MD + 8 <= MD4 and MD4 + 2 <= MD1 and MD1 + 8 <= WORDS);
    try std.testing.expectEqual(@as(usize, 3), WMAX);
}
