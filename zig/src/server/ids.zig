//! Random ids: ``uuid4().hex`` prefixes for replies, calls and items.
const std = @import("std");
const log = @import("log.zig");
const Allocator = std.mem.Allocator;

/// ``prefix`` then ``n`` random lowercase hex digits.
pub fn make(a: Allocator, prefix: []const u8, n: usize) Allocator.Error![]const u8 {
    var bytes: [32]u8 = undefined;
    log.io().random(&bytes);
    const out = try a.alloc(u8, prefix.len + n);
    @memcpy(out[0..prefix.len], prefix);
    const digits = "0123456789abcdef";
    for (0..n) |i| out[prefix.len + i] = digits[(bytes[i / 2] >> @intCast(4 * (1 - i % 2))) & 0xf];
    return out;
}
