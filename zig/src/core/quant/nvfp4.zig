//! ModelOpt NVFP4 weights (e2m1 codes, an e4m3 scale every 16 inputs, a global scale) on the host: fp64 references.
const std = @import("std");
const types = @import("types.zig");
const fp8 = @import("fp8.zig");

const Tensor = types.Tensor;

/// Inputs one e4m3 scale covers.
pub const group = 16;

/// An e2m1 nibble's value.
pub fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

/// As stored: `weight` u8 [..., N, K / 2] (even columns low nibbles), `scale` e4m3 [..., N, K / 16], `global` a matrix.
pub const Host = struct { weight: Tensor, scale: Tensor, global: []const f32 };

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel() * 2;
}

/// fp64 dequantization of every row of `h` (a stack's leading dimensions folded), code * scale * global, into `out`.
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale.rank != rank or h.weight.dtype != .u8 or h.scale.dtype != .f8_e4m3) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1] * 2;
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale.shape[d]) return error.UnexpectedTensor;
    if (h.scale.shape[rank - 2] != n or h.scale.shape[rank - 1] * group != k or out.len != elements(h)) return error.UnexpectedTensor;
    const stack = elements(h) / (n * k);
    if (h.global.len != stack) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const row = e * n + r;
        const byte = h.weight.bytes[row * (k / 2) + c / 2];
        const nib: u8 = if (c & 1 == 0) byte & 0xF else byte >> 4;
        out[row * k + c] = e2m1(nib) * fp8.e4m3(h.scale.bytes[row * (k / group) + c / group]) * h.global[e];
    };
}

test "an NVFP4 stack against hand-decoded values" {
    // two experts of [1, 32]: two scale groups a row; byte 0 holds columns 0 (low) and 1 (high)
    var codes: [2 * 16]u8 = @splat(0);
    codes[0] = 0xF2; // column 0: e2m1 2 -> 1.0; column 1: e2m1 0xF -> -6
    codes[8] = 0x07; // column 16: 6
    codes[16 + 15] = 0x30; // expert 1, column 31: e2m1 3 -> 1.5
    const scales = [_]u8{ 0x38, 0x40, 0x38, 0x30 }; // 1, 2 | 1, 0.5
    const global = [_]f32{ 0.25, 2 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 16, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [64]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.25), out[0]);
    try std.testing.expectEqual(@as(f64, -1.5), out[1]);
    try std.testing.expectEqual(@as(f64, 3), out[16]);
    try std.testing.expectEqual(@as(f64, 1.5), out[32 + 31]);
    try std.testing.expectEqual(@as(f64, 0), out[2]);
}

test "a reference refuses mismatched expert stacks" {
    const codes: [16]u8 = @splat(0x22);
    const scales: [2]u8 = @splat(0x38);
    const global = [_]f32{ 1, 1 };
    const h: Host = .{
        .weight = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 1, 8, 1, 1 }, .bytes = &codes },
        .scale = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &scales },
        .global = &global,
    };
    var out: [32]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}
