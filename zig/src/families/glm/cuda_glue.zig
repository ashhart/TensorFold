//! GLM-5.3-Flash's glue on CUDA: the Python family's cuda/glue.py on glm_glue.cu, whose kernels keep Triton's bytes.

const std = @import("std");
const cuda = @import("cuda");
const mm_mod = @import("cuda_mm.zig");

/// What the forward's kernels launch from: our glue's functions.
pub const Set = struct { glue: *const Kernels, mm: ?*const mm_mod.Kernels = null, packs: ?*const mm_mod.Packed = null }; // mm: linears tiled at load

/// Our glm_glue.cu's functions.
pub const Kernels = struct {
    rmsnorm_4096: cuda.Function,
    rmsnorm_1536: cuda.Function,
    rmsnorm_512: cuda.Function,
    residual_add: cuda.Function,
    stream_mean: cuda.Function,
    rank_sum: cuda.Function,
    hc_post: cuda.Function,
    combine: cuda.Function,
    slice_sum: cuda.Function,
    top1: cuda.Function,
    put_words: cuda.Function,
    embed_b16: cuda.Function,
    conv_shift: cuda.Function,
    swiglu_512: cuda.Function,
    swiglu_64: cuda.Function,
    group_sums: cuda.Function,
    topk: cuda.Function,
    router_part: cuda.Function,
    router_tile: cuda.Function,
    hc_partial: cuda.Function,
    hc_partial8: cuda.Function,
    hc_finish: cuda.Function,
    absorb_q: cuda.Function,
    expand_v: cuda.Function,
    lat_write: cuda.Function,
    index_write: cuda.Function,
    pool_keys: cuda.Function,
    select_rows: cuda.Function,
    bmm_decode: cuda.Function,
    bmm_rows: cuda.Function,
    bmm_tile: cuda.Function,
    bmm_tile1: cuda.Function,
    bmm_d32s1: cuda.Function,
    bmm_d32s2: cuda.Function,
    bmm_d32s3: cuda.Function,
    bmm_d64s2: cuda.Function,
    chunks_16: cuda.Function,
    chunks_32: cuda.Function,
    merge: cuda.Function,
    scores: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| @field(k, name) = try m.function("tf_glm_" ++ name);
        try k.bmm_tile.allowDynamicShared(bmm_tile_smem);
        try k.bmm_tile1.allowDynamicShared(bmm_tile1_smem[1]);
        try k.chunks_16.allowDynamicShared(chunksSmem(16));
        try k.chunks_32.allowDynamicShared(chunksSmem(32));
        return k;
    }
};

/// An element-wise kernel's launch: a block of 256 columns, `rows` x the blocks to cover `d`.
fn rowsLaunch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 256) }, .block = .{ .x = 256 } }, stream, a);
}

/// As above, 8 columns a thread (16-byte loads), 1024 a block of 128 (Triton's BLOCK); d % 8 == 0, rows 16B-aligned.
fn rows8Launch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize, ptrs: anytype) !void {
    if (d % 8 != 0) return error.WidthNotEights;
    inline for (ptrs) |p| if (p % 16 != 0) return error.RowsUnaligned;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 1024) }, .block = .{ .x = 128 } }, stream, a);
}

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn pow2(d: usize) i64 {
    return @intCast(std.math.ceilPowerOfTwoAssert(usize, d));
}

/// glue.rmsnorm: bf16(w * bf16(x / sqrt(mean(x^2) + eps))) into `out`, with `xs` each 64 outputs' fp32 sum.
pub fn rmsnorm(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const f, const warps: u32 = switch (d) {
        4096 => .{ set.glue.rmsnorm_4096, 8 },
        1536 => .{ set.glue.rmsnorm_1536, 4 },
        512 => .{ set.glue.rmsnorm_512, 4 },
        else => return error.NoRmsnormWidth,
    };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(x_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(o_stride)));
    a.add(xs orelse @as(u64, 0));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(xs != null)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// swiglu's column block: 512 where it divides the width, else 64 (TP=3's 704 and 640, the latter 128 in glue.py).
fn swigluBlock(w: usize) !usize {
    if (w % 512 == 0) return 512;
    if (w % 64 == 0) return 64;
    return error.NoSwigluBlock;
}

/// glue.swiglu: [gate | up] -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)), 64-sums in `xs`.
pub fn swiglu(set: Set, stream: cuda.Stream, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
    const block = try swigluBlock(w);
    const f, const threads: u32 = if (block == 512) .{ set.glue.swiglu_512, 128 } else .{ set.glue.swiglu_64, 64 };
    var a: cuda.Args = .{};
    inline for (.{ gu, out, xs, limit, i64of(w) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(w / block) }, .block = .{ .x = threads } }, stream, &a);
}

/// glm_glue_bmm.cuh's Tile::smem(f32) of tf_glm_bmm_tile: 3 stages of 128 x and 64 w rows, 64 bf16 a row.
pub const bmm_tile_smem: u32 = 3 * (128 + 64) * 64 * 2;
/// tf_glm_bmm_tile1's [bf16, fp32 out]: 3 stages of 128 x and 128 w rows, 32 bf16 a row, or the fp32 tile if larger.
pub const bmm_tile1_smem = [2]u32{ 3 * (128 + 128) * 32 * 2, 128 * (128 + 4) * 4 };
/// glm_glue_attn.cuh's wide_smem<hb>(): q (hb x 512), two 32-key tiles, p (hb x 32) in bf16, then max/sum scratch.
pub fn chunksSmem(comptime hb: u32) u32 {
    return (hb * 512 + 2 * 32 * 512 + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// K slices of the router's dot (glue.ROUTER_KS): fixed by the shape, as the sliced and the single pass differ in bits.
pub const router_ks = 8;

/// glue.router: out [m][ne] fp32 = x . w rows; d a multiple of router_ks x 64: K slices in `part`, added in order.
pub fn router(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    if (d % (router_ks * 64) != 0) return error.RouterWidth; // the Python family's single-pass _router: no GLM width
    var a: cuda.Args = .{};
    inline for (.{ x, w, part, i64of(m), i64of(x_stride), i64of(d), i64of(ne), @as(c_int, router_ks) }) |v| a.add(v);
    if (x % 16 == 0 and x_stride % 8 == 0) // rows on 16 bytes: 64-row tiles staged by cp.async (decode's rows past m load as zeros)
        try cuda.launch.launch(set.glue.router_tile, .{ .grid = .{ .x = cdiv(m, 64), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a)
    else
        try cuda.launch.launch(set.glue.router_part, .{ .grid = .{ .x = cdiv(m, 16), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a);
    try sliceSum(set, stream, part, out, m * ne, router_ks, true);
}

/// glue.select: top_k by sigmoid + bias (lower id on ties), normed if `norm`, times `scale`, then the shared expert.
pub fn select(set: Set, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    var a: cuda.Args = .{};
    inline for (.{ logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
}

/// glue.combine: out [rows][d] fp32 = sum over slots in order of wts[r][k] y[r][k] (y fp32 in decode, bf16 in prefill).
pub fn combine(set: Set, stream: cuda.Stream, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ y, @as(c_int, @intFromBool(y_f32)), wts, out, i64of(d), cint(slots) }) |v| a.add(v);
    try rows8Launch(set.glue.combine, stream, &a, rows, d, .{ y, out });
}

/// K blocks of the hyper-connection mixing dots (glue.HC_BLOCKS): fixed by the shape, like the router's slices.
pub const hc_blocks = 16;

/// The mixing kernels' scratch: rows x hc_blocks x 32 fp32 (24 dots and the sum of squares of each K block).
pub fn hcPartBytes(rows: usize) usize {
    return rows * hc_blocks * 32 * 4;
}

/// glue.hc_pre: streams [rows][s d] -> the collapsed RMS-normed row and its 64-sums `xs`; `post`, `comb` (Sinkhorn).
pub fn hcPre(set: Set, stream: cuda.Stream, x: u64, fn_w: u64, base: u64, scale: u64, norm_w: u64, out: u64, xs: u64, post: u64, comb: u64, part: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    var a: cuda.Args = .{};
    inline for (.{ x, fn_w, part, i64of(s * d), @as(c_int, hc_blocks) }) |v| a.add(v);
    // Decode windows (16 blocks a row) take each block's 8 steps at once; prompt rows 4 rows a 128-thread block.
    if (rows < 64) {
        try cuda.launch.launch(set.glue.hc_partial8, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks }, .block = .{ .x = 1024 } }, stream, &a);
    } else {
        a.add(i64of(rows));
        try cuda.launch.launch(set.glue.hc_partial, .{ .grid = .{ .x = cdiv(rows, 4), .y = hc_blocks }, .block = .{ .x = 128 } }, stream, &a);
    }
    var b: cuda.Args = .{};
    inline for (.{ x, part, base, scale, norm_w, out, xs, post, comb, eps, hc_eps, @as(c_int, hc_blocks), cint(iters) }) |v| b.add(v);
    try cuda.launch.launch(set.glue.hc_finish, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &b);
}

/// glue.hc_post: branch = bf16(`g`, rank 0 first); xout[s] = bf16(post[s] branch + sum_t comb[t][s] x[t]); x may alias.
pub fn hcPost(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, post: u64, comb: u64, rows: usize, d: usize, s: usize, world: usize) !void {
    if (s != 4) return error.StreamsNotFour;
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, post, comb, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.hc_post, stream, &a, rows, d, .{ x, xout, g });
}

/// glue.rank_sum: out bf16 = partials summed rank 0 first, `me`'s from `own`, others' from `peers` in rank order.
pub fn rankSum(set: Set, stream: cuda.Stream, own: u64, peers: u64, peer_stride: usize, out: u64, me: usize, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ own, peers, out, i64of(peer_stride), i64of(d), cint(world), cint(me) }) |v| a.add(v);
    try rows8Launch(set.glue.rank_sum, stream, &a, rows, d, .{ own, peers, out });
}

/// glue.residual_add (the MTP block's plain residual): xout = bf16(x + bf16(the partials `g` summed rank 0 first)).
pub fn residualAdd(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.residual_add, stream, &a, rows, d, .{ x, xout, g });
}

/// glue.stream_mean: out [rows][d] = bf16 of the s streams' mean (the final collapse).
pub fn streamMean(set: Set, stream: cuda.Stream, x: u64, out: u64, rows: usize, d: usize, s: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, out, i64of(d), cint(s) }) |v| a.add(v);
    try rows8Launch(set.glue.stream_mean, stream, &a, rows, d, .{ x, out });
}

/// glue.embed of a BF16 table: each row's token (ids i32) as `copies` streams of d values, out [rows][copies d].
pub fn embedB16(set: Set, stream: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ ids, table, out, i64of(d), cint(copies) }) |v| a.add(v);
    try rowsLaunch(set.glue.embed_b16, stream, &a, rows, d);
}

/// forward._shift_conv: each layer's 3-row conv window becomes rows keep .. keep + 2 of [window; projection rows].
pub fn convShift(set: Set, stream: cuda.Stream, conv: u64, proj: u64, keep: usize, proj_layer: usize, proj_row: usize, layers: usize, c: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ conv, proj, i64of(keep), i64of(3 * c), i64of(proj_layer), i64of(proj_row), i64of(c) }) |v| a.add(v);
    try rowsLaunch(set.glue.conv_shift, stream, &a, layers, c);
}

/// glue._router_sum and qmm._reduce: `total` values summed over `ks` slices in slice order, fp32 or bf16 out.
pub fn sliceSum(set: Set, stream: cuda.Stream, part: u64, out: u64, total: usize, ks: usize, wide: bool) !void {
    var a: cuda.Args = .{};
    inline for (.{ part, out, i64of(total), cint(ks), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.slice_sum, .{ .grid = .{ .x = cdiv(total, 1024) }, .block = .{ .x = 128 } }, stream, &a);
}

test "the norm's block is the next power of two of the width" {
    try std.testing.expectEqual(@as(i64, 4096), pow2(4096));
    try std.testing.expectEqual(@as(i64, 8192), pow2(6144));
}

test "the wrappers' shape rules as glue.py's" {
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(6144)); // swiglu's block at the dense TP=2 width
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(1024)); // a routed expert's rows at TP=2
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(704)); // TP=3's first two ranks
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(640)); // TP=3's last rank
    try std.testing.expectError(error.NoSwigluBlock, swigluBlock(96));
    try std.testing.expectEqual(@as(i64, 512), pow2(288 + 1)); // _topk's BLOCK for 288 experts
    try std.testing.expectEqual(@as(i64, 16), pow2(8 + 1)); // its SLOTP for top-8 and the shared slot
}
