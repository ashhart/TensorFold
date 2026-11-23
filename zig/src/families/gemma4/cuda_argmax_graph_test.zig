//! TF_GEMMA4_ARGMAX_GRAPH's choices on fakes: which rounds fold their argmax into the graph, which read tok.
const std = @import("std");
const cuda = @import("cuda");
const ag = @import("cuda_argmax_graph.zig");
const t = std.testing;
const Pick = cuda.tree_round.Pick;

const Lane = struct { sampled: bool };
const Window = struct { stream: usize, masks: []const u32 = &.{} };
const Lanes = struct {
    all: []Lane,
    pub fn getPtr(l: Lanes, s: usize) ?*Lane {
        return if (s < l.all.len) &l.all[s] else null;
    }
};
/// What `tokens` touches: the stream's wait, the two downloads (counted), the greedy flags and the switch.
const Buf = struct {
    reads: *usize,
    bytes: []const u8,
    pub fn download(b: Buf, offset: usize, out: []u8) !void {
        b.reads.* += 1;
        @memcpy(out, b.bytes[offset..][0..out.len]);
    }
};
const Stream = struct {
    pub fn synchronize(_: Stream) !void {}
};
fn X(comptime Lean: type) type {
    return struct { m: struct { lean: ?Lean }, lanes: Lanes = .{ .all = &.{} }, s: Stream = .{}, tok: Buf, picks: Buf, greedy: []const bool };
}

test "argmax graph: a verify round folds its argmax only with the switch and every window greedy and unmasked" {
    var lanes = [_]Lane{ .{ .sampled = false }, .{ .sampled = false }, .{ .sampled = true } };
    var reads: usize = 0;
    const none = Buf{ .reads = &reads, .bytes = &.{} };
    const greedy = [_]Window{ .{ .stream = 0 }, .{ .stream = 1 } };
    var on: X(u8) = .{ .m = .{ .lean = 1 }, .lanes = .{ .all = &lanes }, .tok = none, .picks = none, .greedy = &.{} };
    try t.expect(ag.fold(&on, &greedy));
    try t.expect(!ag.fold(&on, &[_]Window{ .{ .stream = 0 }, .{ .stream = 2 } })); // a sampled window: today's draws
    try t.expect(!ag.fold(&on, &[_]Window{.{ .stream = 0, .masks = &.{1} }})); // a grammar's greedy rows: host picks
    var off = on;
    off.m.lean = null;
    try t.expect(!ag.fold(&off, &greedy)); // without the switch: today's launches
}

test "argmax graph: greedy entries take their picks; only with the switch and every entry a pick is tok not read" {
    var picks: [4]Pick = undefined;
    for (&picks, 0..) |*p, i| p.* = .{ .token = @intCast(10 + i), .nonfinite = 0 };
    const tok = [_]u32{ 90, 91, 92, 93 };
    var tok_reads: usize = 0;
    var pick_reads: usize = 0;
    const tb = Buf{ .reads = &tok_reads, .bytes = std.mem.sliceAsBytes(&tok) };
    const pb = Buf{ .reads = &pick_reads, .bytes = std.mem.sliceAsBytes(&picks) };
    const all_greedy = [_]bool{ true, true, true, true };
    const mixed = [_]bool{ true, false, true, true };
    for ([_]?u8{ null, 1 }) |lean| for ([_][]const bool{ &all_greedy, &mixed }) |greedy| {
        tok_reads = 0;
        var x: X(u8) = .{ .m = .{ .lean = lean }, .tok = tb, .picks = pb, .greedy = greedy };
        var out: [4]u32 = @splat(0);
        try ag.tokens(&x, 0, &out);
        const skipped = lean != null and greedy.ptr == &all_greedy;
        try t.expectEqual(@as(usize, if (skipped) 0 else 1), tok_reads);
        for (out, greedy, 0..) |v, g, i| try t.expectEqual(if (g) @as(u32, @intCast(10 + i)) else tok[i], v);
    };
    // a non-finite pick is still an error
    picks[2].nonfinite = 1;
    var x: X(u8) = .{ .m = .{ .lean = 1 }, .tok = tb, .picks = pb, .greedy = &all_greedy };
    var out: [4]u32 = undefined;
    try t.expectError(error.NonFiniteLogits, ag.tokens(&x, 0, &out));
}
