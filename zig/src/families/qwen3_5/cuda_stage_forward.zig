//! A pipeline stage's forward (RFC #2005): its own layers between the boundary rows h and r, and the head on rank 0.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("cuda_engine.zig");

/// A stage's layers [first, end) of the model; `heads`: it also runs the final norm and the head (rank 0 of a ring).
pub const Span = struct { first: usize, end: usize, heads: bool };

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
