//! An optional group-32 affine proposal head; the main checkpoint and its verifier remain unchanged.
const std = @import("std");

pub fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn value(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

fn code(x: f32) u8 {
    const bounded = std.math.clamp(x, 0, 15);
    const lo = @floor(bounded);
    const integer: u8 = @intFromFloat(lo);
    const fraction = bounded - lo;
    return integer + @intFromBool(fraction > 0.5 or (fraction == 0.5 and integer & 1 == 1));
}

/// Codes use the FP32 range before its BF16 table rounding, matching the Python proposal-head recipe.
pub fn row(input: []const u8, words: []u8, scales: []u8, biases: []u8) !void {
    const n = input.len / 2;
    if (n == 0 or n % 32 != 0 or input.len != n * 2 or words.len != n / 2 or scales.len != n / 16 or biases.len != scales.len)
        return error.UnexpectedTensor;
    for (0..n / 32) |group| {
        const at = group * 32;
        var lo = std.math.inf(f32);
        var hi = -std.math.inf(f32);
        for (0..32) |i| {
            const v = value(input, at + i);
            if (!std.math.isFinite(v)) return error.UnexpectedTensor;
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const scale = @max((hi - lo) / 15.0, 1e-8);
        if (!std.math.isFinite(scale)) return error.UnexpectedTensor;
        std.mem.writeInt(u16, scales[group * 2 ..][0..2], bf16(scale), .little);
        std.mem.writeInt(u16, biases[group * 2 ..][0..2], bf16(lo), .little);
        for (0..16) |j| {
            const low = code((value(input, at + j * 2) - lo) / scale);
            const high = code((value(input, at + j * 2 + 1) - lo) / scale);
            words[group * 16 + j] = low | (high << 4);
        }
    }
}

test "proposal codes have known nearest-even ties and a flat row stays on its exact bias" {
    var input: [128]u8 = undefined;
    for (0..64) |i| std.mem.writeInt(u16, input[i * 2 ..][0..2], bf16(if (i < 32) @as(f32, @floatFromInt(@min(i, 30))) / 2 else 7), .little);
    var words: [32]u8 = undefined;
    var scales: [4]u8 = undefined;
    var biases: [4]u8 = undefined;
    try row(&input, &words, &scales, &biases);
    try std.testing.expectEqual(@as(u8, 0x00), words[0]);
    try std.testing.expectEqual(@as(u8, 0x21), words[1]);
    try std.testing.expectEqual(@as(u8, 0x22), words[2]);
    try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, scales[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0x40e0), std.mem.readInt(u16, biases[2..4], .little));
    for (words[16..]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    std.mem.writeInt(u16, input[0..2], 0x7f80, .little);
    try std.testing.expectError(error.UnexpectedTensor, row(&input, &words, &scales, &biases));
}

test "proposal approximation is bounded per value against an independent FP64 dequantizer" {
    var input: [128]u8 = undefined;
    for (0..64) |i| std.mem.writeInt(u16, input[i * 2 ..][0..2], bf16(@sin(@as(f32, @floatFromInt(i)) * 0.37) * 0.05), .little);
    var words: [32]u8 = undefined;
    var scales: [4]u8 = undefined;
    var biases: [4]u8 = undefined;
    try row(&input, &words, &scales, &biases);
    for (0..2) |g| {
        var lo = std.math.inf(f64);
        var hi = -std.math.inf(f64);
        for (0..32) |i| {
            const v: f64 = value(&input, g * 32 + i);
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const scale = (hi - lo) / 15;
        const stored_s: f64 = value(&scales, g);
        const stored_b: f64 = value(&biases, g);
        const bound = scale * 0.50001 + 15 * @abs(stored_s - scale) + @abs(stored_b - lo);
        for (0..32) |i| {
            const at = g * 32 + i;
            const q = (words[at / 2] >> @as(u3, @intCast((at % 2) * 4))) & 15;
            const back = @as(f64, @floatFromInt(q)) * stored_s + stored_b;
            try std.testing.expect(@abs(back - value(&input, at)) <= bound);
        }
    }
}
