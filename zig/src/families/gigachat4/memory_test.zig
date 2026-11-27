//! Memory floors: the map's weight bytes equal the headers', and the 37B release on one GB10 at its native window.
const std = @import("std");
const st = @import("safetensors");
const config_test = @import("config_test.zig");
const memory = @import("memory.zig");
const names_test = @import("names_test.zig");
const plan = @import("plan.zig");

const t = std.testing;

fn headerBytes(h: st.Header) u64 {
    var n: u64 = 0;
    for (h.values()) |e| {
        var b: u64 = e.dtype.size();
        for (e.shape[0..e.rank]) |d| b *= d;
        n += b;
    }
    return n;
}

test "the map's stored bytes equal each release's headers and the 37B index's total_size" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, []const u8, u64, u64 }{
        .{ try config_test.json37b(a), names_test.header_37b, 38_502_021_376, 40_032_796_240 },
        .{ config_test.json_708b, names_test.header_708b, 714_789_875_712, 719_702_848_992 },
        .{ try config_test.json37bBf16(a), names_test.header_37b_bf16, 74_942_934_656, 76_515_806_080 },
    };
    for (cases) |c| {
        const cfg = try names_test.parsed(c[0]);
        try t.expectEqual(c[2], try memory.weightBytes(&cfg, .backbone));
        try t.expectEqual(c[3], try memory.weightBytes(&cfg, .with_mtp));
        try t.expectEqual(headerBytes(try names_test.unfold(a, c[1])), try memory.weightBytes(&cfg, .with_mtp));
    }
}

test "per-position caches and per-stream state of both releases" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const small = try names_test.parsed(try config_test.json37b(arena.allocator()));
    try t.expectEqual(@as(u64, 8 * (576 * 2 + 128 * 2)), memory.tokenBytes(&small));
    try t.expectEqual(@as(u64, 576 * 2), memory.mtpTokenBytes(&small));
    try t.expectEqual(@as(u64, 24 * (3 * 8192 * 2 + 32 * 128 * 128 * 4)), try memory.streamBytes(&small));
    const big = try names_test.parsed(config_test.json_708b);
    try t.expectEqual(@as(u64, 16 * (576 * 2 + 128 * 2)), memory.tokenBytes(&big));
    try t.expectEqual(@as(u64, 48 * (3 * 16384 * 2 + 64 * 128 * 128 * 4)), try memory.streamBytes(&big));
}

test "37B on one GB10: one stream at 1,048,576 positions fits; the 708B's weights alone do not" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try names_test.parsed(try config_test.json37b(arena.allocator()));
    const window: u64 = c.max_context;
    try t.expectEqual(@as(u64, 50_364_692_736), try memory.floor(&c, .backbone, window, 1));
    try t.expectEqual(@as(u64, 53_103_427_152), try memory.floor(&c, .with_mtp, window, 1));
    try t.expect(try memory.floor(&c, .with_mtp, window, 1) < memory.gb10_bytes);
    try t.expectEqual(@as(u64, 7), try memory.streamsIn(&c, .backbone, window, memory.gb10_bytes));
    try t.expectEqual(@as(u64, 6), try memory.streamsIn(&c, .with_mtp, window, memory.gb10_bytes));
    const big = try names_test.parsed(config_test.json_708b);
    try t.expect(try memory.weightBytes(&big, .backbone) > memory.gb10_bytes);
    try t.expectEqual(@as(u64, 0), try memory.streamsIn(&big, .backbone, 4096, memory.gb10_bytes));
    try t.expectError(error.Overflow, memory.floor(&c, .backbone, std.math.maxInt(u64), 1));
}

test "the state sums are checked: admitted widest dims overflow, and so do 48 layers that each fit" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const top = "16777216"; // the largest dimension config.json admits
    const widest = try names_test.parsed(try config_test.edited(a, config_test.json_708b, &.{
        .{ "linear_num_key_heads", top }, .{ "linear_num_value_heads", top }, .{ "linear_key_head_dim", top }, .{ "linear_value_head_dim", top },
    }));
    try t.expectError(error.Overflow, memory.streamBytes(&widest));
    try t.expectError(error.Overflow, memory.perStream(&widest, .backbone, 1));
    const deep = try names_test.parsed(try config_test.edited(a, config_test.json_708b, &.{
        .{ "linear_num_value_heads", top }, .{ "linear_value_head_dim", top }, .{ "linear_key_head_dim", "1024" },
    }));
    const d = deep.delta;
    const layer = (@as(u64, d.conv) - 1) * (2 * @as(u64, d.key_heads) * d.key_dim + @as(u64, d.value_heads) * d.value_dim) * 2 + @as(u64, d.value_heads) * d.value_dim * d.key_dim * 4;
    try t.expectEqual(@as(u32, 48), plan.build(&deep).count(.gated_deltanet));
    try t.expectError(error.Overflow, std.math.mul(u64, layer, 48));
    try t.expectError(error.Overflow, memory.streamBytes(&deep));
}

test "zero streams cost the weights, and a stream past u64 bytes fits no room, at any context" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try names_test.parsed(try config_test.json37b(arena.allocator()));
    const huge = std.math.maxInt(u64);
    for ([_]memory.Weights{ .backbone, .with_mtp }) |w| {
        const weights = try memory.weightBytes(&c, w);
        try t.expectEqual(weights, try memory.floor(&c, w, huge, 0));
        try t.expectError(error.Overflow, memory.floor(&c, w, huge, 1));
        try t.expectEqual(@as(u64, 0), try memory.streamsIn(&c, w, huge, weights));
        try t.expectEqual(@as(u64, 0), try memory.streamsIn(&c, w, huge, weights - 1));
        try t.expectEqual(@as(u64, 0), try memory.streamsIn(&c, w, huge, memory.gb10_bytes));
        try t.expectEqual(@as(u64, 0), try memory.streamsIn(&c, w, c.max_context, weights));
    }
}

test "streamsIn is the most streams whose floor fits the room" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try names_test.parsed(try config_test.json37b(arena.allocator()));
    for ([_]memory.Weights{ .backbone, .with_mtp }) |w| {
        const weights = try memory.weightBytes(&c, w);
        for ([_]u64{ 0, 1, 4096, c.max_context }) |context| {
            const one = try memory.perStream(&c, w, context);
            for ([_]u64{ weights, weights + one - 1, weights + one, weights + 3 * one + 1, memory.gb10_bytes }) |room| {
                const n = try memory.streamsIn(&c, w, context, room);
                try t.expect(try memory.floor(&c, w, context, n) <= room);
                try t.expect(try memory.floor(&c, w, context, n + 1) > room);
            }
        }
    }
}

test "a stream that costs nothing (every layer MLA, no positions) is not limited by memory" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try names_test.parsed(try config_test.edited(a, try config_test.json37b(a), &.{.{ "full_attention_layers", "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]" }}));
    try t.expectEqual(@as(u32, 0), plan.build(&c).count(.gated_deltanet));
    try t.expectEqual(@as(u64, 0), try memory.streamBytes(&c));
    for ([_]memory.Weights{ .backbone, .with_mtp }) |w| {
        const weights = try memory.weightBytes(&c, w);
        try t.expectEqual(@as(u64, 0), try memory.perStream(&c, w, 0));
        try t.expectEqual(weights, try memory.floor(&c, w, 0, std.math.maxInt(u64)));
        try t.expectEqual(@as(u64, std.math.maxInt(u64)), try memory.streamsIn(&c, w, 0, weights));
        try t.expectEqual(@as(u64, std.math.maxInt(u64)), try memory.streamsIn(&c, w, 0, memory.gb10_bytes));
        try t.expectEqual(@as(u64, 0), try memory.streamsIn(&c, w, 0, weights - 1));
        try t.expectEqual(@as(u64, 2), try memory.streamsIn(&c, w, 1, weights + 2 * try memory.perStream(&c, w, 1)));
    }
}
