//! A stream's drafter context on the host: its ring's span, its held chain, and the scalars a frame or absorb uploads.
const std = @import("std");
const hip = @import("hip");
const dr = @import("lanes").drafter;

pub const Segment = hip.draft_ops.Segment;

/// Drafts a stream holds at most: a block of 16, the ring's padded block (config.ringRows), but its pending row.
pub const most_held = 15;

/// A stream's context: its layers' key and value ring (device address) over [begin, end), and the chain it holds.
pub const Ctx = struct {
    ring: u64,
    begin: u64 = 0,
    end: u64 = 0,
    held: [most_held]u32 = undefined,
    held_n: usize = 0,

    /// The context emptied, its ring kept.
    pub fn clear(ctx: *Ctx) void {
        ctx.* = .{ .ring = ctx.ring };
    }
};

/// The rows an absorb runs: target rows [start, start + n).
pub const Span = struct { start: u64, n: usize };

/// Where an absorb of rows [start, start + n) lands: a rewind or a gap moves the span, past `window` rows are dropped.
pub fn open(ctx: *Ctx, start_: u64, n_: usize, window: usize) Span {
    var start = start_;
    var n = n_;
    if (start < ctx.end) ctx.end = start;
    if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
    if (n > window) {
        start += n - window;
        n = window;
        ctx.begin = start;
    }
    ctx.end = start;
    return .{ .start = start, .n = n };
}

/// The span absorbed: the context ends after it, at most `window` rows behind its end.
pub fn close(ctx: *Ctx, s: Span, window: usize) void {
    ctx.end = s.start + s.n;
    if (ctx.end > window and ctx.begin < ctx.end - window) ctx.begin = ctx.end - window;
}

/// Each absorbed row's rope position (start + i), as qkRope reads them.
pub fn absorbPositions(start: u64, pos: []i32) !void {
    if (start + pos.len > std.math.maxInt(i32)) return error.PositionTooFar;
    for (pos, 0..) |*p, i| p.* = @intCast(start + i);
}

/// What a frame needs of the config.
pub const Shape = struct { block: usize, vocab: usize, mask: u32 };

/// A frame's ids (pending, then masks), rope positions (the context's end + m % block) and segments, checked.
pub fn stage(s: Shape, items: []const dr.Hold, contexts: []const *Ctx, ids: []u32, pos: []i32, segs: []Segment) !void {
    const rows = items.len * s.block;
    if (items.len == 0 or contexts.len != items.len or ids.len < rows or pos.len < rows or segs.len < items.len)
        return error.BadDraftFrame;
    for (items, contexts, 0..) |it, ctx, index| {
        if (it.position == 0 or ctx.end != it.position - 1 or ctx.begin > ctx.end) return error.DraftContextBehind;
        if (it.position + s.block > std.math.maxInt(i32)) return error.PositionTooFar;
        // a tree's mask has to keep each row's own key first: a row that sees no key comes out of the merge as NaN
        if (it.lanes != null) return error.TreesUnsupported;
        if (it.pending >= s.vocab or it.depth > s.block - 1) return error.BadDraftFrame;
        for (contexts[0..index]) |prior| {
            if (ctx == prior or ctx.ring == prior.ring) return error.OverlappingDraftContexts;
        }
    }
    for (items, contexts, 0..) |it, ctx, i| {
        ids[i * s.block] = it.pending;
        @memset(ids[i * s.block + 1 ..][0 .. s.block - 1], s.mask);
        for (pos[i * s.block ..][0..s.block], 0..) |*p, m| p.* = @intCast(ctx.end + m);
        segs[i] = .{ .ring = ctx.ring, .begin = @intCast(ctx.begin), .position = @intCast(ctx.end) };
    }
}

test "an absorb past the window keeps its last rows; a rewind and a gap move the context" {
    var ctx: Ctx = .{ .ring = 1 };
    const s = open(&ctx, 0, 100, 64);
    try std.testing.expectEqual(Span{ .start = 36, .n = 64 }, s);
    close(&ctx, s, 64);
    try std.testing.expectEqual(@as(u64, 36), ctx.begin);
    try std.testing.expectEqual(@as(u64, 100), ctx.end);
    // one more row slides the window's start
    close(&ctx, open(&ctx, 100, 1, 64), 64);
    try std.testing.expectEqual(@as(u64, 37), ctx.begin);
    try std.testing.expectEqual(@as(u64, 101), ctx.end);
    // a rewind (rejected drafts' rows) ends the context at the absorb's start
    close(&ctx, open(&ctx, 98, 2, 64), 64);
    try std.testing.expectEqual(@as(u64, 100), ctx.end);
    try std.testing.expectEqual(@as(u64, 37), ctx.begin);
    // a gap starts the context over
    close(&ctx, open(&ctx, 500, 3, 64), 64);
    try std.testing.expectEqual(@as(u64, 500), ctx.begin);
    try std.testing.expectEqual(@as(u64, 503), ctx.end);
}

test "a frame stages each stream's pending token, its masks, positions from its end and its segment" {
    var a: Ctx = .{ .ring = 1000, .begin = 2, .end = 10 };
    var b: Ctx = .{ .ring = 2000, .begin = 0, .end = 4 };
    const shape: Shape = .{ .block = 4, .vocab = 100, .mask = 99 };
    var ids: [8]u32 = undefined;
    var pos: [8]i32 = undefined;
    var segs: [2]Segment = undefined;
    const items = [_]dr.Hold{
        .{ .stream = undefined, .pending = 7, .position = 11, .depth = 3 },
        .{ .stream = undefined, .pending = 8, .position = 5, .depth = 2 },
    };
    try stage(shape, &items, &.{ &a, &b }, &ids, &pos, &segs);
    try std.testing.expectEqualSlices(u32, &.{ 7, 99, 99, 99, 8, 99, 99, 99 }, &ids);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 4, 5, 6, 7 }, &pos);
    try std.testing.expectEqual(Segment{ .ring = 1000, .begin = 2, .position = 10 }, segs[0]);
    try std.testing.expectEqual(Segment{ .ring = 2000, .begin = 0, .position = 4 }, segs[1]);
}

test "a frame refuses a context behind its hold, a shared ring, a tree, and too deep a chain" {
    var a: Ctx = .{ .ring = 1000, .end = 10 };
    var b: Ctx = .{ .ring = 1000, .end = 4 };
    const shape: Shape = .{ .block = 4, .vocab = 100, .mask = 99 };
    var ids: [8]u32 = undefined;
    var pos: [8]i32 = undefined;
    var segs: [2]Segment = undefined;
    const behind = [_]dr.Hold{.{ .stream = undefined, .pending = 7, .position = 12, .depth = 3 }};
    try std.testing.expectError(error.DraftContextBehind, stage(shape, &behind, &.{&a}, &ids, &pos, &segs));
    const deep = [_]dr.Hold{.{ .stream = undefined, .pending = 7, .position = 11, .depth = 4 }};
    try std.testing.expectError(error.BadDraftFrame, stage(shape, &deep, &.{&a}, &ids, &pos, &segs));
    const two = [_]dr.Hold{
        .{ .stream = undefined, .pending = 7, .position = 11, .depth = 3 },
        .{ .stream = undefined, .pending = 7, .position = 5, .depth = 3 },
    };
    try std.testing.expectError(error.OverlappingDraftContexts, stage(shape, &two, &.{ &a, &b }, &ids, &pos, &segs));
    var parents = [_]i32{-1};
    var zero = [_]u8{0};
    const shape_tree: lanes_shape.Shape = .{ .parents = &parents, .depths = &zero, .ranks = &zero, .expected = 1 };
    const tree = [_]dr.Hold{.{ .stream = undefined, .pending = 7, .position = 11, .depth = 1, .lanes = &shape_tree }};
    try std.testing.expectError(error.TreesUnsupported, stage(shape, &tree, &.{&a}, &ids, &pos, &segs));
}

const lanes_shape = @import("lanes").shape;

test "absorbed rows take their own positions" {
    var pos: [3]i32 = undefined;
    try absorbPositions(40, &pos);
    try std.testing.expectEqualSlices(i32, &.{ 40, 41, 42 }, &pos);
    try std.testing.expectError(error.PositionTooFar, absorbPositions(std.math.maxInt(i32), &pos));
}
