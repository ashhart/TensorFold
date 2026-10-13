//! Block-FP8 checkpoints (128 x 128 e4m3 blocks, fp32 scale_inv) repacked for qmmf.zig as linear.py does.
const std = @import("std");
const qmmf = @import("qmmf.zig");

pub const block = 128; // a checkpoint scale's rows and inputs
pub const group = qmmf.group; // inputs a kernel stage reads, one column scale each

/// Rows padded to whole 128-row blocks, as the kernel's 64-column tiles read them.
pub fn padded(n: usize) usize {
    return (n + block - 1) / block * block;
}

/// The k of byte `byte` of `lane` in a k32 step (fragment_index's kin).
fn kin(lane: usize, byte: usize) usize {
    const p = 16 * (byte / 4) + 4 * (lane % 4) + byte % 4;
    return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
}

/// e4m3 codes [n, k] in the kernel's order [npad/64][k/64][8][32][2][8], rows past n zero (linear._fragment_order).
pub fn packCodes(out: []u8, codes: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    std.debug.assert(out.len == npad * k and codes.len == n * k and k % group == 0);
    var i: usize = 0;
    for (0..npad / 64) |t| for (0..k / group) |g| for (0..8) |a| for (0..32) |lane| for (0..2) |h| for (0..8) |b| {
        const row = t * 64 + a * 8 + lane / 4;
        const col = g * group + h * 32 + kin(lane, b);
        out[i] = if (row < n) codes[row * k + col] else 0;
        i += 1;
    };
}

/// Block scales [ceil(n/128), k/128] as the kernel's tiles [npad/64][k/64][64] of fp32, rows past n one (from_rows).
pub fn packScales(out: []f32, scale_inv: []const f32, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / group;
    std.debug.assert(out.len == npad * kg and scale_inv.len == ((n + block - 1) / block) * (k / block));
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |c| {
        const row = t * 64 + c;
        out[(t * kg + g) * 64 + c] = if (row < n) scale_inv[(row / block) * (k / block) + g * group / block] else 1.0;
    };
}

/// The lane matmul's view of a packed block-FP8 projection.
pub fn weight(codes: u64, scales: u64, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .fp8g, .codes = codes, .scales = scales, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

test "the code repack is a permutation that puts each byte where fragment_index says" {
    const gpa = std.testing.allocator;
    const n = 130;
    const k = 128;
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 7 + 1);
    const out = try gpa.alloc(u8, padded(n) * k);
    defer gpa.free(out);
    packCodes(out, codes, n, k);
    const idx = ((((0 * 2 + 1) * 8 + 3) * 32 + 9) * 2 + 1) * 8 + 5; // (t 0, g 1, a 3, lane 9, h 1, byte 5): row 26
    try std.testing.expectEqual(codes[26 * k + 64 + 32 + kin(9, 5)], out[idx]);
    var zeros: usize = 0;
    for (out) |b| zeros += @intFromBool(b == 0);
    try std.testing.expect(zeros >= (padded(n) - n) * k);
    var seen: [32]bool = @splat(false); // kin covers a k32 step once over a lane quad's 4 x 8 bytes
    for (0..4) |lane| for (0..8) |b| {
        try std.testing.expect(!seen[kin(lane, b)]);
        seen[kin(lane, b)] = true;
    };
}

test "scale tiles: each 64-input group takes its 128 x 128 block's scale, padded rows one" {
    const n = 130;
    const k = 256;
    var inv: [2 * 2]f32 = .{ 1, 2, 3, 4 };
    var out: [256 * 4]f32 = undefined;
    packScales(&out, &inv, n, k);
    try std.testing.expectEqual(@as(f32, 2), out[(0 * 4 + 3) * 64 + 5]); // row 5, group 3 -> block (0, 1)
    try std.testing.expectEqual(@as(f32, 3), out[(2 * 4 + 0) * 64 + 1]); // row 129 -> block (1, 0)
    try std.testing.expectEqual(@as(f32, 1), out[(2 * 4 + 1) * 64 + 2]); // row 130: padding
}
