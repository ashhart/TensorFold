//! E4M3 codes with one UE8M0 byte per 32 inputs, repacked losslessly for the lane matmul.
const std = @import("std");
const fp8 = @import("fp8.zig");
const qmmf = @import("qmmf.zig");

pub const group = 32;
pub const padded = fp8.padded;
pub const packCodes = fp8.packCodes;

/// Row-major [n, k/32] bytes to [npad/64][k/64][64][2]; padded rows have scale 1.
pub fn packScales(out: []u8, scales: []const u8, n: usize, k: usize) void {
    const npad = padded(n);
    const kg = k / qmmf.group;
    std.debug.assert(k % qmmf.group == 0 and out.len == npad * (k / group) and scales.len == n * (k / group));
    for (0..npad / 64) |t| for (0..kg) |g| for (0..64) |r| for (0..2) |h| {
        const row = t * 64 + r;
        out[((t * kg + g) * 64 + r) * 2 + h] = if (row < n) scales[row * (k / group) + g * 2 + h] else 127;
    };
}

pub fn weight(codes: u64, scales: u64, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .mxfp8, .codes = codes, .scales = scales, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

test "every scale byte lands in its independent 32-input group, including padded rows" {
    const a = std.testing.allocator;
    const n = 130;
    const k = 256;
    const scales = try a.alloc(u8, n * (k / group));
    defer a.free(scales);
    for (scales, 0..) |*v, i| v.* = @truncate(i * 13 + 7);
    const out = try a.alloc(u8, padded(n) * (k / group));
    defer a.free(out);
    packScales(out, scales, n, k);
    for (0..padded(n)) |row| for (0..k / group) |g| {
        const at = (((row / 64) * (k / 64) + g / 2) * 64 + row % 64) * 2 + g % 2;
        try std.testing.expectEqual(if (row < n) scales[row * (k / group) + g] else @as(u8, 127), out[at]);
    };
    const w = weight(1, 2, n, k);
    try std.testing.expect(w.mode == .mxfp8 and w.npad == 256 and w.scale == 1.0);
}
