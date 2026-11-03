//! Host repack of Q4 group-64 weights into 32-column tiles for qmm_q4_tile; whole vectors and scales move unchanged.
const std = @import("std");

/// Columns (outputs) in one tile; one Honeykrisp subgroup, so a lane's 16-byte load sits next to its neighbours'.
pub const columns: usize = 32;
const vector: usize = 16;
const group: usize = 64;

/// A projection can be tiled when k fills whole groups; the last tile gets zero columns the kernel never computes.
pub fn fits(n: u64, k: u64) bool {
    return n != 0 and k != 0 and k % group == 0;
}

/// Columns stored for n outputs: n rounded up to whole tiles.
pub fn padded(n: usize) usize {
    return (n + columns - 1) / columns * columns;
}

/// Words (rows of k/32 vectors) into [tiles][k/32 vectors][32 lanes]; src is n * k / 2 bytes, dst padded(n) * k / 2.
pub fn tileWords(dst: []u8, src: []const u8, n: usize, k: usize) void {
    std.debug.assert(fits(n, k) and src.len == n * k / 2 and dst.len == padded(n) * k / 2);
    @memset(dst, 0);
    const vectors = k / 32;
    for (0..n) |column| {
        for (0..vectors) |j| {
            const from = (column * vectors + j) * vector;
            const to = (((column / columns) * vectors + j) * columns + column % columns) * vector;
            @memcpy(dst[to..][0..vector], src[from..][0..vector]);
        }
    }
}

/// Scales or biases (rows of k/64) into [tiles][k/64 groups][32 lanes]; src is n * k / 32 bytes, dst padded(n) * k / 32.
pub fn tileParams(dst: []u8, src: []const u8, n: usize, k: usize) void {
    std.debug.assert(fits(n, k) and src.len == n * k / 32 and dst.len == padded(n) * k / 32);
    @memset(dst, 0);
    const groups = k / group;
    for (0..n) |column| {
        for (0..groups) |g| {
            const from = (column * groups + g) * 2;
            const to = (((column / columns) * groups + g) * columns + column % columns) * 2;
            @memcpy(dst[to..][0..2], src[from..][0..2]);
        }
    }
}

const testing = std.testing;

fn byteSum(bytes: []const u8) u64 {
    var sum: u64 = 0;
    for (bytes) |b| sum += b;
    return sum;
}

test "fits needs whole groups of k, and padded rounds n up to whole tiles" {
    try testing.expect(fits(32, 64));
    try testing.expect(fits(16, 2048));
    try testing.expect(fits(248320, 2048));
    try testing.expect(!fits(32, 96));
    try testing.expect(!fits(0, 64));
    try testing.expectEqual(@as(usize, 32), padded(1));
    try testing.expectEqual(@as(usize, 32), padded(32));
    try testing.expectEqual(@as(usize, 64), padded(33));
    try testing.expectEqual(@as(usize, 248320), padded(248320));
}

// The shader reads column c, vector j at ((c / 32) * (k / 32) + j) * 32 + c % 32 vectors; scales likewise per group.
test "a tiled vector is the row's vector at the address the shader computes, padding is zero, and nothing is lost" {
    const n = 70;
    const k = 192;
    var src: [n * k / 2]u8 = undefined;
    for (&src, 0..) |*byte, i| byte.* = @truncate((i *% 2654435761 >> 7) | 1);
    var dst: [padded(n) * k / 2]u8 = undefined;
    tileWords(&dst, &src, n, k);
    for (0..n) |c| {
        for (0..k / 32) |j| {
            const at = (((c / 32) * (k / 32) + j) * 32 + c % 32) * 16;
            try testing.expectEqualSlices(u8, src[(c * (k / 32) + j) * 16 ..][0..16], dst[at..][0..16]);
        }
    }
    try testing.expectEqual(byteSum(&src), byteSum(&dst));
    for (n..padded(n)) |c| {
        for (0..k / 32) |j| {
            const at = (((c / 32) * (k / 32) + j) * 32 + c % 32) * 16;
            try testing.expectEqual(@as(u64, 0), byteSum(dst[at..][0..16]));
        }
    }
}

test "tiled scales are the row's scale at the address the shader computes, padding is zero, and nothing is lost" {
    const n = 40;
    const k = 256;
    var src: [n * k / 32]u8 = undefined;
    for (&src, 0..) |*byte, i| byte.* = @truncate((i *% 40503 >> 3) | 1);
    var dst: [padded(n) * k / 32]u8 = undefined;
    tileParams(&dst, &src, n, k);
    for (0..n) |c| {
        for (0..k / 64) |g| {
            const at = (((c / 32) * (k / 64) + g) * 32 + c % 32) * 2;
            try testing.expectEqualSlices(u8, src[(c * (k / 64) + g) * 2 ..][0..2], dst[at..][0..2]);
        }
    }
    try testing.expectEqual(byteSum(&src), byteSum(&dst));
}
