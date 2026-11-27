//! Memory floors: the map's weight bytes equal the headers', and the 37B release on one GB10 at its native window.
const std = @import("std");
const st = @import("safetensors");
const config_test = @import("config_test.zig");
const memory = @import("memory.zig");
const names_test = @import("names_test.zig");

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
    try t.expectEqual(@as(u64, 24 * (3 * 8192 * 2 + 32 * 128 * 128 * 4)), memory.streamBytes(&small));
    const big = try names_test.parsed(config_test.json_708b);
    try t.expectEqual(@as(u64, 16 * (576 * 2 + 128 * 2)), memory.tokenBytes(&big));
    try t.expectEqual(@as(u64, 48 * (3 * 16384 * 2 + 64 * 128 * 128 * 4)), memory.streamBytes(&big));
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
