//! Host repack of Q4 group-64 weights into 32-column tiles for qmm_q4_tile; whole vectors and scales move unchanged.
const std = @import("std");

/// Columns (outputs) in one tile; one Honeykrisp subgroup, so a lane's 16-byte load sits next to its neighbours'.
pub const columns: usize = 32;
const vector: usize = 16;
const group: usize = 64;

/// A projection can be tiled when n fills whole tiles and k whole groups.
pub fn fits(n: u64, k: u64) bool {
    return n != 0 and n % columns == 0 and k != 0 and k % group == 0;
}

/// Packed words, rows of k/32 vectors, into [n/32 tiles][k/32 vectors][32 lanes]; both slices hold n * k / 2 bytes.
pub fn tileWords(dst: []u8, src: []const u8, n: usize, k: usize) void {
    std.debug.assert(fits(n, k) and src.len == n * k / 2 and dst.len == src.len);
    const vectors = k / 32;
    for (0..n) |column| {
        for (0..vectors) |j| {
            const from = (column * vectors + j) * vector;
            const to = (((column / columns) * vectors + j) * columns + column % columns) * vector;
            @memcpy(dst[to..][0..vector], src[from..][0..vector]);
        }
    }
}

/// Scales or biases, rows of k/64 bf16, into [n/32 tiles][k/64 groups][32 lanes]. Both slices hold n * k / 32 bytes.
pub fn tileParams(dst: []u8, src: []const u8, n: usize, k: usize) void {
    std.debug.assert(fits(n, k) and src.len == n * k / 32 and dst.len == src.len);
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

test "fits needs whole tiles of columns and whole groups of k" {
    try testing.expect(fits(32, 64));
    try testing.expect(fits(248320, 2048));
    try testing.expect(!fits(16, 2048));
    try testing.expect(!fits(48, 2048));
    try testing.expect(!fits(32, 96));
    try testing.expect(!fits(0, 64));
}

// The shader reads column c, vector j at ((c / 32) * (k / 32) + j) * 32 + c % 32 vectors; scales likewise per group.
test "a tiled vector is the row's vector at the address the shader computes, and nothing is lost" {
    const n = 96;
    const k = 192;
    var src: [n * k / 2]u8 = undefined;
    for (&src, 0..) |*byte, i| byte.* = @truncate(i *% 2654435761 >> 7);
    var dst: [src.len]u8 = undefined;
    tileWords(&dst, &src, n, k);
    for (0..n) |c| {
        for (0..k / 32) |j| {
            const at = (((c / 32) * (k / 32) + j) * 32 + c % 32) * 16;
            try testing.expectEqualSlices(u8, src[(c * (k / 32) + j) * 16 ..][0..16], dst[at..][0..16]);
        }
    }
    var a = src;
    var b = dst;
    std.mem.sort(u8, &a, {}, std.sort.asc(u8));
    std.mem.sort(u8, &b, {}, std.sort.asc(u8));
    try testing.expectEqualSlices(u8, &a, &b);
}

test "tiled scales are the row's scale at the address the shader computes, and nothing is lost" {
    const n = 64;
    const k = 256;
    var src: [n * k / 32]u8 = undefined;
    for (&src, 0..) |*byte, i| byte.* = @truncate(i *% 40503 >> 3);
    var dst: [src.len]u8 = undefined;
    tileParams(&dst, &src, n, k);
    for (0..n) |c| {
        for (0..k / 64) |g| {
            const at = (((c / 32) * (k / 64) + g) * 32 + c % 32) * 2;
            try testing.expectEqualSlices(u8, src[(c * (k / 64) + g) * 2 ..][0..2], dst[at..][0..2]);
        }
    }
    var a = src;
    var b = dst;
    std.mem.sort(u8, &a, {}, std.sort.asc(u8));
    std.mem.sort(u8, &b, {}, std.sort.asc(u8));
    try testing.expectEqualSlices(u8, &a, &b);
}
