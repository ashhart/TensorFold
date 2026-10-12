//! Host test of TF_DSV41_STREAM_RB: the reader, and the long-prefill program with `_stream` as `_stream_rb<N>`.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("calls_check.zig");
const pk = @import("serve_knobs.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "TF_DSV41_STREAM_RB: 0 / 1 / 2 / 4" {
    defer Fake.pairs = &.{};
    Fake.pairs = &.{};
    try testing.expectEqual(@as(i64, 0), try pk.streamRb(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_STREAM_RB", "1" }};
    try testing.expectEqual(@as(i64, 1), try pk.streamRb(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_STREAM_RB", "2" }};
    try testing.expectEqual(@as(i64, 2), try pk.streamRb(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_STREAM_RB", "4" }};
    try testing.expectEqual(@as(i64, 4), try pk.streamRb(&Fake.get));
    Fake.pairs = &.{.{ "TF_DSV41_STREAM_RB", "3" }};
    try testing.expectError(error.BadKnob, pk.streamRb(&Fake.get));
}

fn arg(c: calls.Call, name: []const u8) ?calls.Arg {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return x.arg;
    return null;
}

test "TF_DSV41_STREAM_RB on the long-context prefill: _stream_rb<N> in place of _stream, the same tensors and merge" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(aa, &w, @embedFile("fixtures/dense-k2.txt"));
    try check.parseExperts(aa, &w, @embedFile("fixtures/expert-k2.txt"));
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const base: block.Options = .{ .gm_v2 = .one, .limit = 1 << 20, .rope_rows = (1 << 20) + 2048, .index_budget = 64 << 20 };
    for ([_]i64{ 2048, 2047, 300 }) |n| for ([_]i64{ 2 * 4096, 131072 - 2048 }) |start| {
        const off = try bp.emitReplay(aa, &cfg, &w, base, &enc, n, start, .encoder);
        var streams: usize = 0;
        for (off) |c| streams += @intFromBool(std.mem.eql(u8, c.name, "_stream"));
        try testing.expect(streams > 0); // layers 2 / 8 / 14 past 4,096 keys
        for ([_]i64{ 1, 2, 4 }) |rb| {
            var o = base;
            o.stream_rb = rb;
            const on = try bp.emitReplay(aa, &cfg, &w, o, &enc, n, start, .encoder);
            try testing.expectEqual(off.len, on.len);
            const want = switch (rb) {
                1 => "_stream_pf",
                2 => "_stream_rb2",
                else => "_stream_rb4",
            };
            var seen: usize = 0;
            for (off, on) |x, y| {
                if (!std.mem.eql(u8, x.name, "_stream")) {
                    try testing.expectEqualStrings(x.name, y.name);
                    try testing.expectEqual(x.grid, y.grid);
                    continue;
                }
                try testing.expectEqual(@as(i64, 0), arg(x, "MODE").?.i);
                try testing.expectEqualStrings(want, y.name);
                try testing.expectEqual(@divFloor(x.grid[0] + rb - 1, rb), y.grid[0]);
                try testing.expectEqual(x.grid[1], y.grid[1]);
                try testing.expectEqual(x.grid[0], arg(y, "R").?.i);
                for ([_][]const u8{ "QI", "W", "IK", "POS", "BUF", "PT" }) |nm| {
                    const u = arg(x, nm).?;
                    const v = arg(y, nm).?;
                    try testing.expectEqual(std.meta.activeTag(u), std.meta.activeTag(v));
                    if (u != .t) continue; // PT without a paged pool
                    try testing.expectEqual(u.t.offset, v.t.offset);
                    try testing.expectEqualSlices(i64, u.t.shape, v.t.shape);
                    try testing.expectEqual(u.t.dt, v.t.dt);
                }
                for ([_][]const u8{ "w_stride", "nsplit", "RATIO", "H", "D", "BP", "SPLIT", "K", "CAP", "PSH" }) |nm| try testing.expectEqual(arg(x, nm).?.i, arg(y, nm).?.i);
                try testing.expectEqual(arg(x, "WS").?.f, arg(y, "WS").?.f);
                try testing.expectEqual(arg(x, "SCALE").?.f, arg(y, "SCALE").?.f);
                seen += 1;
            }
            try testing.expectEqual(streams, seen);
        }
    };
}
