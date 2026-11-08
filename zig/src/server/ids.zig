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

/// A random lowercase UUID v4, as ``str(uuid.uuid4())``.
pub fn uuid4(io: std.Io) [36]u8 {
    var b: [16]u8 = undefined;
    io.random(&b);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const hex = std.fmt.bytesToHex(b, .lower);
    var out: [36]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{s}-{s}-{s}-{s}-{s}", .{ hex[0..8], hex[8..12], hex[12..16], hex[16..20], hex[20..32] }) catch unreachable;
    return out;
}
