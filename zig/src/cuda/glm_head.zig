//! GLM-5.3-Flash's head on one rank (forward.compute's end, decode.sample_rows greedy): logits and the token pick.
const std = @import("std");
const driver = @import("driver.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const glue_ = @import("glm_glue.zig");
const bmm_ = @import("glm_bmm.zig");
const Stream = stream_.Stream;

/// The rank's head: the final norm [D] bf16, its lm_head rows [vocab / world, D] bf16 from vocab id `offset`.
pub const HeadW = struct { norm: u64, head: bmm_.B16, offset: u32 };

/// One forward's buffers for up to `rows` rows, one allocation.
pub const Scratch = struct {
    buf: memory.DeviceBuffer,
    rows: usize,
    hidden: u64, // [rows, D] bf16: the streams' mean
    fnormed: u64, // [rows, D] bf16
    fxs: u64, // [rows, D / 64] f32: the norm's group sums (the head's matmul reads none)
    logits: u64, // [rows, n] bf16
    part: u64, // the matmul's split-K partials
    cand: u64, // [rows, 2] f32: each row's top value and the bits of its vocab id

    pub fn init(d: *const driver.Driver, s: Stream, w: HeadW, dims: usize, rows: usize) !Scratch {
        if (rows == 0 or dims == 0 or dims % 64 != 0) return error.GlmHeadShape;
        const n: usize = w.head.n;
        const sizes = [_]usize{ rows * dims * 2, rows * dims * 2, rows * dims / 64 * 4, rows * n * 2, bmm_.Bmm.partBytes(w.head, rows), rows * 2 * 4 };
        var total: usize = 0;
        for (sizes) |z| total += std.mem.alignForward(usize, z, 256);
        var buf = try memory.DeviceBuffer.alloc(d, total);
        errdefer buf.free();
        try buf.fill8(0, s.handle);
        var at: [sizes.len]u64 = undefined;
        var p = buf.ptr;
        for (sizes, &at) |z, *a| {
            a.* = p;
            p += std.mem.alignForward(usize, z, 256);
        }
        return .{ .buf = buf, .rows = rows, .hidden = at[0], .fnormed = at[1], .fxs = at[2], .logits = at[3], .part = if (sizes[4] > 0) at[4] else 0, .cand = at[5] };
    }

    pub fn deinit(sc: *Scratch) void {
        sc.buf.free();
    }
};

/// Into sc.cand: streams' mean, final norm, logits, each row's top-1; `last` (a prompt chunk) reads its last row.
pub fn forward(g: *const glue_.Glue, b: *const bmm_.Bmm, s: Stream, eps: f32, dims: usize, streams: usize, w: HeadW, sc: *const Scratch, x: u64, rows: usize, last: bool) !void {
    if (rows == 0 or rows > sc.rows) return error.GlmHeadRows;
    if (w.head.k != dims) return error.GlmHeadShape;
    const d: u64 = dims;
    try g.streamMean(s, x, sc.hidden, rows, dims, streams);
    try g.rmsnorm(s, sc.hidden, @intCast(dims), w.norm, sc.fnormed, @intCast(dims), sc.fxs, eps, rows, dims);
    const first: usize = if (last) rows - 1 else 0;
    const m = rows - first;
    try b.matmul(s, w.head, sc.fnormed + first * d * 2, @intCast(dims), m, sc.logits, false, sc.part);
    try g.greedyPack(s, sc.logits, sc.cand, m, w.head.n, w.offset);
}

/// sample_rows' greedy pick from every rank's gathered candidates [world, rows, 2]: highest value, ties to lowest id.
pub fn choose(gathered: []const f32, world: usize, rows: usize, out: []u32) !void {
    if (gathered.len != world * rows * 2 or out.len < rows) return error.GlmHeadShape;
    for (0..rows) |i| {
        var best: f32 = -std.math.inf(f32);
        var id: u32 = std.math.maxInt(u32);
        for (0..world) |r| {
            const v = gathered[(r * rows + i) * 2];
            const t: u32 = @bitCast(gathered[(r * rows + i) * 2 + 1]);
            if (v > best or (v == best and t < id)) {
                best = v;
                id = t;
            }
        }
        out[i] = id;
    }
}

test "the choice takes the highest value across ranks, ties to the lowest id" {
    const f = struct {
        fn id(t: u32) f32 {
            return @bitCast(t);
        }
    }.id;
    // rows: rank 1 wins; a tie the lower id (rank 1's) wins; rank 0 wins
    const g = [_]f32{ 1.0, f(5), 2.0, f(900), 3.0, f(7), 2.0, f(100), 2.0, f(800), 1.0, f(77) };
    var out: [3]u32 = undefined;
    try choose(&g, 2, 3, &out);
    try std.testing.expectEqualSlices(u32, &.{ 100, 800, 7 }, &out);
    try std.testing.expectError(error.GlmHeadShape, choose(g[0..10], 2, 3, &out));
}
