//! A pipeline stage's forward (RFC #2005): its own layers between the boundary rows h and r, and the head on rank 0.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");

/// A stage's layers [first, end) of the model; `heads`: it also runs the final norm and the head (rank 0 of a ring).
pub const Span = struct { first: usize, end: usize, heads: bool };

/// One engine's geometry, or with `span` its stage's: v1 is a ring with the head on rank 0, over MLX checkpoints.
pub fn geometry(whole: E.Geometry, span: ?Span) !E.Geometry {
    const sp = span orelse return whole;
    if (sp.heads != (sp.first == 0)) return error.StageHeadOffRankZero;
    if (whole.kinds.modelopt or whole.exl3 or whole.bf16 or whole.pocket > 0) return error.StageNeedsMlxCheckpoint;
    return whole.stage(sp.first, sp.end, sp.heads);
}

/// A forward's launches: all (one engine), a stage's layers (rank 0: from the embedding), or the norm and head.
pub const Part = enum(u2) { whole, layers, head };

/// The boundary's two planes (h, then r) of `rows` rows in host memory, sent or filled in place by the ring.
pub const Hop = struct {
    ctx: *anyopaque,
    give: *const fn (ctx: *anyopaque, rows: usize, bytes: []const u8) anyerror!void,
    take: *const fn (ctx: *anyopaque, rows: usize, bytes: []u8) anyerror!void,
};

/// Host bytes of a boundary of `rows` rows: two bf16 planes of `hidden` values a row.
pub fn bytes(hidden: usize, rows: usize) usize {
    return 2 * rows * hidden * 2;
}

/// A stage's launches: rows in (past rank 0), its layers, rows out, rank 0 then its head; `rows` unbucketed.
pub fn run(e: *E.Engine, shape: E.Engine.Shape, rows: usize) !void {
    const g = e.g;
    var s = shape;
    s.part = .layers;
    if (!g.embeds) try take(e, rows);
    try e.launchShape(s);
    try give(e, rows);
    if (!g.heads) return;
    try take(e, rows);
    s.part = .head;
    try e.launchShape(s);
}

/// The boundary rows from the stage before, onto h and r before the next launch (without a ring: as they are).
fn take(e: *E.Engine, rows: usize) !void {
    const hop = e.hop orelse return;
    const buf = e.boundary.?.bytes[0..bytes(e.g.hidden, rows)];
    try hop.take(hop.ctx, rows, buf);
    const plane = buf.len / 2;
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(e.h, buf.ptr, plane, e.stream.handle), "cuMemcpyHtoDAsync");
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(e.r, buf[plane..].ptr, plane, e.stream.handle), "cuMemcpyHtoDAsync");
}

/// This stage's h and r to the next one, once the layers that wrote them are done.
fn give(e: *E.Engine, rows: usize) !void {
    const hop = e.hop orelse return;
    const buf = e.boundary.?.bytes[0..bytes(e.g.hidden, rows)];
    const plane = buf.len / 2;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(buf.ptr, e.h, plane, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(buf[plane..].ptr, e.r, plane, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
    try hop.give(hop.ctx, rows, buf);
}

const small: E.Geometry = .{
    .hidden = 2048,
    .intermediate = 6144,
    .layers = 8,
    .vocab = 248320,
    .k_heads = 16,
    .v_heads = 16,
    .q_heads = 8,
    .kv_heads = 2,
    .eps = 1e-6,
    .tied = true,
    .context = 4096,
    .linear = @splat(false),
};

test "Qwen stage: the head only on rank 0, which holds layer 0" {
    const g = small;
    try std.testing.expectEqual(g, try geometry(g, null));
    try std.testing.expectError(error.StageHeadOffRankZero, geometry(g, .{ .first = 0, .end = 4, .heads = false }));
    try std.testing.expectError(error.StageHeadOffRankZero, geometry(g, .{ .first = 4, .end = 8, .heads = true }));
    const a = try geometry(g, .{ .first = 0, .end = 4, .heads = true });
    const b = try geometry(g, .{ .first = 4, .end = 8, .heads = false });
    try std.testing.expect(a.heads and a.embeds and !b.heads and !b.embeds and b.first == 4 and b.whole == 8);
}

test "Qwen stage: a ModelOpt, EXL3, BF16 or pocket checkpoint is refused as a stage, served whole" {
    const g = small;
    var formats: [4]E.Geometry = @splat(g);
    formats[0].kinds.modelopt = true;
    formats[1].exl3 = true;
    formats[2].bf16 = true;
    formats[3].pocket = 64;
    for (formats) |f| {
        try std.testing.expectEqual(f, try geometry(f, null));
        for ([_]Span{ .{ .first = 0, .end = 4, .heads = true }, .{ .first = 4, .end = 8, .heads = false } }) |sp|
            try std.testing.expectError(error.StageNeedsMlxCheckpoint, geometry(f, sp));
    }
}
