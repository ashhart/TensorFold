//! TF_Q35_ATTN_SKIP_EMPTY=1: a forward leaves out the decode attention launch whose rows all return unwritten (opt-in).
const std = @import("std");
const shapes = @import("cuda_shapes.zig");

/// qwen35_attn_tc.cuh GROUP_LEN: a stream this long (Seg.len) is attn_rows', a shorter one attn_one's (attn_partial).
pub const group_len: usize = 8192;

/// The switch; default off. The startup check refused every malformed value already, so a plain default stands here.
pub fn on() bool {
    return @import("sys").environment.read(bool, "TF_Q35_ATTN_SKIP_EMPTY", false) catch false;
}

/// The launch no row of `parts` reaches: each part's staged Seg.len is its cache.len (Engine.stagedSeg).
pub fn of(enabled: bool, parts: anytype) shapes.Skip {
    if (!enabled) return .none;
    var short = false;
    var long = false;
    for (parts) |p| {
        if (p.cache.len >= group_len) long = true else short = true;
    }
    if (short == long) return .none; // both kinds (or no part): both launches work
    return if (long) .partial else .rows;
}

/// The forward's attention choices: MT2 exactly as before, then the skip.
pub fn attention(shape: *shapes.Shape, e: anytype, parts: anytype, record: bool, total: usize) void {
    shape.mt2 = shapes.useMt2(e.mt2.enabled, e.kv8, !record, total, parts.len, parts[0].cache.len);
    shape.skip = of(on(), parts);
}

const Cache = struct { len: usize };
const Part = struct { cache: *const Cache };

fn testParts(comptime n: usize, lens: [n]usize, caches: *[n]Cache) [n]Part {
    var out: [n]Part = undefined;
    for (lens, caches, &out) |l, *c, *p| {
        c.* = .{ .len = l };
        p.* = .{ .cache = c };
    }
    return out;
}

test "attention skip: off never skips; on, only the launch every stream's rows return from" {
    var c1: [1]Cache = undefined;
    var c2: [2]Cache = undefined;
    for ([_]usize{ 0, 1, 2047, 8191, 8192, 8193, 65536 }) |len| {
        const one = testParts(1, .{len}, &c1);
        try std.testing.expectEqual(shapes.Skip.none, of(false, &one));
        try std.testing.expectEqual(if (len >= 8192) shapes.Skip.partial else shapes.Skip.rows, of(true, &one));
    }
    try std.testing.expectEqual(shapes.Skip.rows, of(true, &testParts(2, .{ 100, 8191 }, &c2)));
    try std.testing.expectEqual(shapes.Skip.partial, of(true, &testParts(2, .{ 8192, 9000 }, &c2)));
    // both kinds: both launch
    try std.testing.expectEqual(shapes.Skip.none, of(true, &testParts(2, .{ 8191, 8192 }, &c2)));
    try std.testing.expectEqual(shapes.Skip.none, of(true, &[_]Part{}));
}

test "attention skip: short and long contexts key apart, and off keys as before" {
    for ([_]usize{ 1, 4, 8, 16 }) |rows| for ([_]bool{ false, true }) |mt2| {
        var off = shapes.Shape.init(rows, 1, true, false, 1);
        off.mt2 = mt2;
        var short = off;
        short.skip = .rows;
        var long = off;
        long.skip = .partial;
        const keys = [_]u64{ off.key(), short.key(), long.key() };
        for (keys, 0..) |a, i| for (keys[i + 1 ..]) |b| try std.testing.expect(a != b);
        try std.testing.expectEqual(@as(u64, 0), off.key() >> 33); // the bits above join stay clear when off
    };
}

test {
    _ = @import("cuda_attn_skip_test.zig");
}
