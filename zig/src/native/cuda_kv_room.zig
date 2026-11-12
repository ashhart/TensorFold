//! The room the streams are planned in: the device's, and a family's planes kept in a carveout for the first window.
const std = @import("std");
const kv_pool = @import("core").kv_pool;
const pool = @import("cuda_kv_pool.zig");

/// `room` plus the bytes a family keeps in a carveout for its first window (0 for a family without one).
pub fn streamsRoom(loaded: anytype, room: u64) u64 {
    return room +| if (comptime @hasField(@TypeOf(loaded), "carved_bytes")) loaded.carved_bytes else 0;
}

test "a first window's carved planes: the pool gains their whole pages while the prompt cache is at its budget" {
    try std.testing.expectEqual(@as(u64, 1500), streamsRoom(.{ .carved_bytes = @as(u64, 500) }, 1000));
    try std.testing.expectEqual(@as(u64, 1000), streamsRoom(.{ .free_streams = @as(u32, 0) }, 1000));
    // flashnext's bf16 caches with the MTP head (30,784 bytes a token), pages of 512 rows, five streams
    const l: kv_pool.Layout = .{ .fixed_bytes = 1 << 20, .token_bytes = 30784, .page_tokens = 512, .snap_bytes = 0 };
    const window: u64 = 262144 + 32;
    const first: u64 = 8 << 30;
    const carved: u64 = 2032 << 20;
    var pages: [2]usize = undefined;
    for ([_]u64{ 0, carved }, 0..) |c, i| {
        const s = pool.served(l, 0, first, first, streamsRoom(.{ .carved_bytes = c }, 30 << 30));
        const p = try kv_pool.plan(s.layout.?, .{ .room = s.room, .engine = 0, .lanes = 5, .fixed = true, .cache = 16 << 30, .window = window });
        try std.testing.expectEqual(@as(u64, 16 << 30), p.copied); // the cache whole in both
        pages[i] = p.pages;
    }
    try std.testing.expectEqual(@as(usize, 135), pages[1] - pages[0]); // floor((left + carved) / page) - floor(left / page)
    const tokens = (pages[1] - pages[0]) * l.page_tokens;
    try std.testing.expect(@abs(@as(i64, @intCast(tokens)) - @as(i64, @intCast(carved / l.token_bytes))) < l.page_tokens);
}
