//! GPU-gated lanes: a chain step runs while the drafts' probability product holds; the window verifies those rows.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;

/// A gated reply's widest window, its rows (the host drafts the first window to WMAX_FIRST rows, its narrowest).
pub const WMAX = 4;
pub const WMAX_FIRST = 3;

/// Words: rows, probability product, each draft's; chain step j's run flag and dims at STEP + (j - 1) * STRIDE.
const ROWS = 0;
const CUM = 1;
const PROB = 2;
const STEP = 32;
const STRIDE = 32;
const MD = 8;
const MD4 = 16;
const MD1 = 24;
const STEPS = replay.MAXR - 2; // a window of MAXR rows: the absorb's draft, then this many chain steps
const WORDS = STEP + STEPS * STRIDE;

const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\// Chain step j's words: whether it runs and its row counts (dims, four streams, one row).
    \\inline void fz_gate_open(device int* G, int j, int run) {
    \\  device int* s = G + 32 + (j - 1) * 32;
    \\  s[0] = run; s[8] = run; s[16] = 4 * run; s[17] = 16 * run; s[24] = run;
    \\}
    \\// A draft's probability over the head's draft vocabulary (the softmax's largest value), then the round's width:
    \\// c = (draft rows, step, threshold bits, the window's rows). Step 0 is the absorb's draft, step j chain step j's
    \\// (skipped unless it ran); a kept draft opens the next chain step while the window has a row for its draft.
    \\kernel void fz_gate(device const bfloat* L [[buffer(0)]], device int* G [[buffer(1)]], constant uint4& c [[buffer(2)]],
    \\    uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    \\    uint sg [[simdgroup_index_in_threadgroup]]) {
    \\  const int step = int(c.y), W = int(c.w);
    \\  if (step > 0 && G[32 + (step - 1) * 32] == 0) return;
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
    \\  const float p = 1.0f / total, cum = (step == 0 ? 1.0f : as_type<float>(G[1])) * p;
    \\  const bool kept = cum >= as_type<float>(c.z);
    \\  if (step == 0) {
    \\    G[0] = kept ? 2 : 1;
    \\    for (int j = 1; j <= W - 2; j++) fz_gate_open(G, j, j == 1 && kept ? 1 : 0);
    \\  } else if (kept) {
    \\    G[0] = step + 2;
    \\    if (step + 1 <= W - 2) fz_gate_open(G, step + 1, 1);
    \\  }
    \\  G[1] = as_type<int>(cum);
    \\  G[2 + step] = as_type<int>(p);
    \\}
;

pub const Gate = struct {
    buf: mtl.Buffer,
    pipe: mtl.Pipeline,
    tau: f32,
    n: usize, // the head's draft vocabulary
    width: usize = WMAX, // the host's window for a gated round, rows

    fn at(g: *const Gate, word: usize) Buf {
        return .{ .b = g.buf, .off = word * 4 };
    }

    /// The window's verified rows (the window kernels' and the verdict's row count).
    pub fn rows(g: *const Gate) Buf {
        return g.at(ROWS);
    }

    /// Chain step j's row count, dims (one row), dims over its four streams and its one-row MoE dims.
    pub fn run(g: *const Gate, j: usize) Buf {
        return g.at(STEP + (j - 1) * STRIDE);
    }
    pub fn md(g: *const Gate, j: usize) Buf {
        return g.at(STEP + (j - 1) * STRIDE + MD);
    }
    pub fn md4(g: *const Gate, j: usize) Buf {
        return g.at(STEP + (j - 1) * STRIDE + MD4);
    }
    pub fn md1(g: *const Gate, j: usize) Buf {
        return g.at(STEP + (j - 1) * STRIDE + MD1);
    }

    /// Before a reply's rounds: the first window (drafted on the host) runs whole.
    pub fn open(g: *const Gate, w: usize) void {
        g.buf.slice(i32, WORDS)[ROWS] = @intCast(w);
    }

    /// The gate after draft `step` (0: the absorb's, j: chain step j's) of a `w`-row window, from the head's logits.
    pub fn encode(g: *const Gate, r: *Run, logits: Buf, step: u32, w: usize) void {
        r.enc.setPipeline(g.pipe);
        r.enc.setBuffer(logits.b, logits.off, 0);
        r.enc.setBuffer(g.buf, 0, 1);
        const c = [4]u32{ @intCast(g.n), step, @bitCast(g.tau), @intCast(w) };
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

/// A gate over `n` draft rows keeping a chain while the probability product stays at `tau`, windows to `width` rows.
pub fn init(r: *Run, tau: f32, n: usize, width: usize) !*Gate {
    if (width < WMAX_FIRST or width > replay.MAXR) return error.GateWidth;
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, source, mtl.CompileOptions.mlx()));
    const g = try r.arena.create(Gate);
    g.* = .{ .buf = try r.buffer(WORDS * 4), .pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, "fz_gate", false)), .tau = tau, .n = n, .width = width };
    const w = g.buf.slice(i32, WORDS);
    @memset(w, 0);
    for (0..STEPS) |j| { // the dims layouts' second word, as mdims_w's
        w[STEP + j * STRIDE + MD + 1] = 16;
        w[STEP + j * STRIDE + MD1 + 1] = 16;
    }
    g.open(WMAX_FIRST);
    return g;
}

test "gate words: each chain step's views sit where the kernel writes its row counts" {
    try std.testing.expect(PROB + replay.MAXR <= STEP and MD + 8 <= MD4 and MD4 + 2 <= MD1 and MD1 + 8 <= STRIDE);
    try std.testing.expect(WMAX_FIRST <= WMAX and WMAX <= replay.MAXR);
}
