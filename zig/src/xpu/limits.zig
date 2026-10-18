//! The XPU runner's size settings, checked before any buffer is sized from them (host-tested, no device).
const std = @import("std");

/// The model's own context limit: a stream holds 1 to this many positions.
pub const MAX_CONTEXT = 262144;

/// XPU_CHUNK: prompt rows per forward, a multiple of the GDN chunk (64) from 64 to 8192.
pub fn chunk(text: []const u8) error{BadChunk}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.BadChunk;
    if (n == 0 or n % 64 != 0 or n > 8192) return error.BadChunk;
    return n;
}

/// A stream's positions, refused outside 1 .. MAX_CONTEXT.
pub fn context(n: i64) error{BadContext}!u32 {
    if (n <= 0 or n > MAX_CONTEXT) return error.BadContext;
    return @intCast(n);
}

/// GDN chunks of 64 rows over `rows`, rounded up: the last one is padded.
pub fn chunks(rows: u32) u32 {
    return (rows + 63) / 64;
}

/// Attention splits of `split` keys over `keys`, rounded up: the last one may be partial.
pub fn splits(keys: u32, split: u32) u32 {
    return (keys + split - 1) / split;
}

test "XPU_CHUNK must be a positive multiple of 64 within 8192" {
    try std.testing.expectEqual(@as(u32, 1024), try chunk("1024"));
    try std.testing.expectEqual(@as(u32, 64), try chunk("64"));
    for ([_][]const u8{ "66", "0", "8256", "-64", "x", "" }) |t| try std.testing.expectError(error.BadChunk, chunk(t));
}

test "a context outside 1 .. MAX_CONTEXT is refused" {
    try std.testing.expectEqual(@as(u32, 32768), try context(32768));
    try std.testing.expectError(error.BadContext, context(0));
    try std.testing.expectError(error.BadContext, context(-1));
    try std.testing.expectError(error.BadContext, context(MAX_CONTEXT + 1));
    try std.testing.expectError(error.BadContext, context(1 << 33));
}

test "chunk and split counts round up" {
    try std.testing.expectEqual(@as(u32, 4), chunks(194)); // 193 rows + a rest pad to 256
    try std.testing.expectEqual(@as(u32, 18), chunks(1152));
    try std.testing.expectEqual(@as(u32, 129), splits(4097, 32));
    try std.testing.expectEqual(@as(u32, 128), splits(4096, 32));
}
