//! Block-FP8 weights (e4m3 codes, a scale_inv a 128 x 128 block) on the host: code values and fp64 references.
const std = @import("std");
const types = @import("types.zig");
const convert = @import("convert.zig");

const Tensor = types.Tensor;

/// Rows and columns one scale_inv entry covers.
pub const block = 128;

/// An e4m3 code's value; 0x7f and 0xff are NaN, as the GPU's conversion reads them.
pub fn e4m3(b: u8) f64 {
    if (b & 0x7f == 0x7f) return std.math.nan(f64);
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8.0 * std.math.pow(f64, 2, -6) else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// x rounded to the nearest e4m3 value (ties to even), saturating at 448, as an FP8 activation quantizer does.
pub fn toE4m3(x: f64) f64 {
    const a = @min(@abs(x), 448.0);
    if (a == 0) return 0;
    const e = @max(std.math.floor(std.math.log2(a)), -6.0); // below 2^-6 the step is the subnormals' 2^-9
    const step = std.math.pow(f64, 2, e - 3);
    var q = @round(a / step);
    if (@abs(a / step - std.math.trunc(a / step) - 0.5) < 1e-12 and @mod(q, 2) == 1) q -= 1;
    return std.math.copysign(@min(q * step, 448.0), x);
}

/// Stored e4m3 [..., N, K] and fp32/bf16/fp16 scales [..., ceil(N / 128), ceil(K / 128)].
pub const Host = struct { weight: Tensor, scale_inv: Tensor };

/// Weights of a projection: its rows (a stack's leading dimensions folded) by K.
pub fn elements(h: Host) usize {
    return h.weight.numel();
}

/// fp64 dequantization of `h`'s rows (a stack's leading dimensions folded), code * scale_inv, into `out` [rows, K].
pub fn reference(h: Host, out: []f64) error{UnexpectedTensor}!void {
    const rank = h.weight.rank;
    if (rank < 2 or rank > h.weight.shape.len or h.scale_inv.rank != rank or h.weight.dtype != .f8_e4m3 or !convert.tableDtype(h.scale_inv.dtype)) return error.UnexpectedTensor;
    const n = h.weight.shape[rank - 2];
    const k = h.weight.shape[rank - 1];
    if (n == 0 or k == 0) return error.UnexpectedTensor;
    for (0..rank - 2) |d| if (h.weight.shape[d] == 0 or h.weight.shape[d] != h.scale_inv.shape[d]) return error.UnexpectedTensor;
    const nb = h.scale_inv.shape[rank - 2];
    const kb = h.scale_inv.shape[rank - 1];
    if (nb != (n + block - 1) / block or kb != (k + block - 1) / block or out.len != h.weight.numel()) return error.UnexpectedTensor;
    const stack = h.weight.numel() / (n * k);
    if (h.scale_inv.numel() != stack * nb * kb) return error.UnexpectedTensor;
    for (0..stack) |e| for (0..n) |r| for (0..k) |c| {
        const at = (e * n + r) * k + c;
        const s = convert.load(h.scale_inv.dtype, h.scale_inv.bytes, (e * nb + r / block) * kb + c / block);
        out[at] = e4m3(h.weight.bytes[at]) * s;
    };
}

test "e4m3 codes and their rounding" {
    try std.testing.expectEqual(@as(f64, 1), e4m3(0x38));
    try std.testing.expectEqual(@as(f64, -2), e4m3(0xc0));
    try std.testing.expectEqual(@as(f64, 448), e4m3(0x7e));
    try std.testing.expectEqual(std.math.pow(f64, 2, -9), e4m3(0x01));
    try std.testing.expect(std.math.isNan(e4m3(0x7f)) and std.math.isNan(e4m3(0xff)));
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(e4m3(b), toE4m3(e4m3(b)));
    }
    try std.testing.expectEqual(@as(f64, 1), toE4m3(1.0625)); // halfway between 1 and 1.125: to even
    try std.testing.expectEqual(@as(f64, 1.25), toE4m3(1.1875));
    try std.testing.expectEqual(@as(f64, -448), toE4m3(-1e6));
}

test "a block-FP8 stack against hand-decoded values" {
    // two experts of [2, 130]: the last two columns fall in each row's second scale block
    var codes: [2 * 2 * 130]u8 = @splat(0x38);
    codes[129] = 0xc0;
    codes[260 + 130 + 128] = 0x40;
    const inv = [_]f32{ 0.5, 4, 1, 8 };
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 3, .shape = .{ 2, 2, 130, 1, 1 }, .bytes = &codes },
        .scale_inv = .{ .dtype = .f32, .rank = 3, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&inv) },
    };
    var out: [codes.len]f64 = undefined;
    try reference(h, &out);
    try std.testing.expectEqual(@as(f64, 0.5), out[0]);
    try std.testing.expectEqual(@as(f64, -8), out[129]);
    try std.testing.expectEqual(@as(f64, 1), out[260]);
    try std.testing.expectEqual(@as(f64, 16), out[260 + 130 + 128]);
}

test "a reference refuses an unsupported f64 scale before loading it" {
    const codes = [_]u8{0x38};
    const inv = [_]f64{1};
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = &codes },
        .scale_inv = .{ .dtype = .f64, .rank = 2, .shape = .{ 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(&inv) },
    };
    var out: [1]f64 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &out));
}

test "a reference refuses an empty projection" {
    const h: Host = .{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 0, 1, 1, 1, 1 }, .bytes = &.{} },
        .scale_inv = .{ .dtype = .f32, .rank = 2, .shape = .{ 0, 1, 1, 1, 1 }, .bytes = &.{} },
    };
    try std.testing.expectError(error.UnexpectedTensor, reference(h, &.{}));
}
