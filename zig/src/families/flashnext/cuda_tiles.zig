//! Prompt rows on tiles keeping #952's bits: the dense matmul (qmm.cu's arithmetic), the read-out and the experts.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const parts = @import("cuda_batch_state.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const low = wt.low;
const eps = wt.eps;
const hc_sk = wt.hc_sk;
const pss_n = (dims / 256) * streams;
const Q4 = wt.Q4;
const HcW = wt.HcW;
const Kernels = parts.Kernels;
const Scratch = parts.Scratch;
/// qmm.cu's 16 x 64 lane tile's shared bytes.
const qmm_smem: u32 = 9472;

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// tile_of's sweep: row tiles a column pass keeps in 12 MiB of the input.
fn sweepOf(m: usize, bm: usize, k: usize) usize {
    return @max(@as(usize, 1), @min(cdiv(m, bm), (12 << 20) / (bm * k * 2)));
}

/// x [m, k] times w: the native lane, the 16-row lane tile, or from 64 rows the prompt tile (folded with K slices).
pub fn matmul(k: *const Kernels, stream: cuda.Stream, x: u64, xs: u64, w: Q4, out: u64, part: u64, native_part: u64, m: usize, sk: usize) !usize {
    if (w.native) |native| {
        if (sk == 1) {
            try k.native_lane.matmul(stream, x, w.k, m, native, out, native_part);
            return 1;
        }
        // Legacy callers consume SK slices; the native lane has already reduced its own shape-driven slices.
        try k.native_lane.matmulF32(stream, x, w.k, m, native, part, native_part);
        const bytes = m * w.n * 4;
        if (sk > 1) try stream.d.check(stream.d.api.cuMemsetD8Async(part + bytes, 0, bytes * (sk - 1), stream.handle), "cuMemsetD8Async");
        return sk;
    }
    // from 64 rows the prompt tile (flashnext_fact.cu), bit for bit the 16-row lane tile
    const big = m >= 64;
    const bm: usize = if (big) 64 else 16;
    const bn: usize = if (big) 128 else 64;
    var a: cuda.Args = .{};
    for ([_]u64{ x, xs, w.w.ptr, w.s.ptr, w.b.ptr, out, part }) |v| a.add(v);
    for ([_]usize{ m, w.n, w.k, sk, (w.n + 127) / 128 * 128, w.k, sweepOf(m, bm, w.k) }) |v| a.add(cint(v));
    const fold = big and sk > 1;
    const f = if (fold) k.fact_fold else if (big) k.fact else k.lane;
    const shared: u32 = if (big) parts.fact_smem else qmm_smem;
    const z: usize = if (fold) 1 else sk;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(cdiv(m, bm) * cdiv(w.n, bn)), .z = @intCast(z) }, .block = .{ .x = if (big) 256 else 128 }, .shared = shared }, stream, &a);
    return z;
}

/// Rows [r0, r0 + n)'s read-out on flashnext_hcf.cu: down with its K slices folded, the reduce, up in bf16, the mix.
pub fn readout(tri: anytype, k: *const Kernels, stream: cuda.Stream, s: *Scratch, hc_w: HcW, r0: usize, n: usize) !void {
    const h = s.h + r0 * wide * 2;
    const pss = s.pss + r0 * pss_n * 4;
    const rinv = s.rinv + r0 * streams * 4;
    const act = s.act + r0 * low * 2;
    const xs_act = s.xs_act + r0 * (low / 32) * 4;
    const inj_ptr = if (hc_w.has_inj == 1) s.inj + r0 * streams * 2 else act;
    const nc = dims / 256;
    var a: cuda.Args = .{};
    for ([_]u64{ h, 0, hc_w.down.w.ptr, hc_w.down.s.ptr, hc_w.down.b.ptr, s.hc_part }) |v| a.add(v);
    for ([_]usize{ n, hc_w.n_down, wide, hc_sk, wide, sweepOf(n, 32, wide) }) |v| a.add(cint(v));
    a.add(pss);
    a.add(hc_w.scale.ptr);
    a.add(eps);
    for ([_]usize{ dims, nc, streams }) |v| a.add(cint(v));
    try cuda.launch.launch(k.hcf_down, .{ .grid = .{ .x = @intCast(cdiv(n, 32) * cdiv(hc_w.n_down, 192)) }, .block = .{ .x = 256 }, .shared = parts.hcf_down_smem }, stream, &a);
    try tri.hcReduceAct(s.hc_part, act, xs_act, inj_ptr, 1, n, streams, low, hc_w.n_down, hc_w.has_inj);
    var b: cuda.Args = .{};
    for ([_]u64{ act, xs_act, hc_w.up.w.ptr, hc_w.up.s.ptr, hc_w.up.b.ptr, s.proj }) |v| b.add(v);
    for ([_]usize{ n, wide, low, 1, low, sweepOf(n, 64, low) }) |v| b.add(cint(v));
    b.add(@as(u64, 0));
    b.add(@as(u64, 0));
    b.add(@as(f32, 0));
    for ([_]usize{ dims, nc, streams }) |v| b.add(cint(v));
    try cuda.launch.launch(k.hcf_up, .{ .grid = .{ .x = @intCast(cdiv(n, 64) * cdiv(wide, 128)) }, .block = .{ .x = 256 }, .shared = parts.hcf_up_smem }, stream, &b);
    try tri.hcRinv(pss, rinv, eps, n, dims, nc, streams);
    try tri.hcMixH(s.proj, h, rinv, hc_w.scale.ptr, s.mixed + r0 * dims * 2, s.xsm + r0 * (dims / 32) * 4, n, dims, streams);
}

/// The grouped experts on flashnext_fexp.cu's 64-pair tiles (expert_rows' bits), wn column blocks a CTA.
pub fn experts(stream: cuda.Stream, f: cuda.Function, wn: usize, x: u64, x_stride: usize, slot_n: usize, w: u64, kg: usize, nb: usize, plan: cuda.grouped.Plan, out: u64, n: usize, items: usize) !void {
    var a = cuda.experts.Affine4Experts.argsOf(x, x_stride, slot_n, w, kg, nb, plan, out, n);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(items * cdiv(nb, wn)) }, .block = .{ .x = 256 }, .shared = parts.fexp_smem }, stream, &a);
}
