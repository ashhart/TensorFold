//! An image's cache identity for every family: SHA-256 over its bytes, then over that and its patch grid, 62 bits.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The bits an identity keeps: the server's cache key carries all of them, 31 in each marker around the image's rows.
pub const bits = 62;

/// A family's tag and an image's encoded bytes, hashed: what `withGrid` names a prepared grid of.
pub fn ofBytes(tag: u64, bytes: []const u8) u64 {
    var h = Sha256.init(.{});
    h.update(std.mem.asBytes(&tag));
    h.update(bytes);
    return first(&h);
}

/// Those bytes prepared on a `grid` (rows, columns) of patches: one picture at two budgets is two images.
pub fn withGrid(bytes_id: u64, grid: [2]u32) u64 {
    var h = Sha256.init(.{});
    h.update(std.mem.asBytes(&bytes_id));
    h.update(std.mem.asBytes(&grid));
    return first(&h) & ((1 << bits) - 1);
}

fn first(h: *Sha256) u64 {
    var d: [Sha256.digest_length]u8 = undefined;
    h.final(&d);
    return std.mem.readInt(u64, d[0..8], .little);
}

test "an identity names the bytes and the grid, in 62 bits" {
    const a = withGrid(ofBytes(1, "abc"), .{ 4, 6 });
    try std.testing.expectEqual(a, withGrid(ofBytes(1, "abc"), .{ 4, 6 }));
    try std.testing.expect(a != withGrid(ofBytes(1, "abd"), .{ 4, 6 }));
    try std.testing.expect(a != withGrid(ofBytes(2, "abc"), .{ 4, 6 })); // another family's tag
    try std.testing.expect(a != withGrid(ofBytes(1, "abc"), .{ 6, 4 }));
    try std.testing.expect(a < 1 << bits);
}
