//! Independent FP64 bitfield arithmetic against every finite scale and code, with a wrong-order control.
const std = @import("std");
const mx = @import("mxfp4");
const st = @import("safetensors");
const t = std.testing;

fn reference(code: u8, scale: u8) f64 {
    const exponent = (code >> 1) & 3;
    const mantissa: f64 = @floatFromInt(code & 1);
    var magnitude: f64 = if (exponent == 0) mantissa * 0.5 else 1 + mantissa * 0.5;
    if (exponent > 1) for (1..exponent) |_| {
        magnitude *= 2;
    };
    const power = @as(i32, scale) - 127;
    var factor: f64 = 1;
    for (0..@abs(power)) |_| factor *= if (power < 0) 0.5 else 2;
    return (if (code & 8 != 0) -magnitude else magnitude) * factor;
}

test "gpt-oss MXFP4 preserves every code and all 255 finite UE8M0 scales exactly in FP64" {
    var block: [16]u8 = undefined;
    for (&block, 0..) |*byte, i| byte.* = @as(u8, @intCast(i)) | (@as(u8, @intCast(15 - i)) << 4);
    for (0..255) |s| {
        var out: [32]f64 = undefined;
        try mx.decodeBlock(&block, @intCast(s), .low_first, &out);
        for (out, 0..) |value, i| {
            const code = if (i % 2 == 0) block[i / 2] & 15 else block[i / 2] >> 4;
            try t.expectEqual(@as(u64, @bitCast(reference(code, @intCast(s)))), @as(u64, @bitCast(value)));
        }
    }
    var out: [32]f64 = undefined;
    try t.expectError(error.BadMxfp4Scale, mx.decodeBlock(&block, 255, .low_first, &out));
}

test "gpt-oss MXFP4 packed reader respects expert row block and nibble boundaries" {
    var blocks: [2 * 2 * 2 * 16]u8 = @splat(0);
    var scales: [8]u8 = @splat(127);
    blocks[0] = 0xF2;
    blocks[16 + 15] = 0x30;
    blocks[64 + 32] = 0x77;
    scales[6] = 128;
    const b: st.Tensor = .{ .dtype = .u8, .rank = 4, .shape = .{ 2, 2, 2, 16, 1 }, .bytes = &blocks };
    const s: st.Tensor = .{ .dtype = .u8, .rank = 3, .shape = .{ 2, 2, 2, 1, 1 }, .bytes = &scales };
    const h = try mx.Host.init(b, s, .low_first);
    try t.expectEqual(@as(f64, 1), try h.value(0, 0, 0));
    try t.expectEqual(@as(f64, -6), try h.value(0, 0, 1));
    try t.expectEqual(@as(f64, 1.5), try h.value(0, 0, 63));
    try t.expectEqual(@as(f64, 12), try h.value(1, 1, 0));
    for (0..2) |expert| for (0..2) |index| {
        var got: [64]f64 = undefined;
        try h.row(expert, index, &got);
        var want: [64]f64 = @splat(0);
        if (expert == 0 and index == 0) {
            want[0] = 1;
            want[1] = -6;
            want[63] = 1.5;
        }
        if (expert == 1 and index == 1) {
            want[0] = 12;
            want[1] = 12;
        }
        try t.expectEqualSlices(f64, &want, &got);
    };
    try t.expectError(error.BadMxfp4Index, h.value(2, 0, 0));
    try t.expectError(error.BadMxfp4Index, h.value(0, 0, 64));
    const swapped = try mx.Host.init(b, s, .high_first);
    try t.expectEqual(@as(f64, -6), try swapped.value(0, 0, 0));
    try t.expect((try h.value(0, 0, 0)) != (try swapped.value(0, 0, 0)));
    var bad = b;
    bad.shape[3] = 15;
    try t.expectError(error.BadMxfp4Shape, mx.Host.init(bad, s, .low_first));
    bad = b;
    bad.dtype = .bf16;
    try t.expectError(error.BadMxfp4Shape, mx.Host.init(bad, s, .low_first));
    bad = b;
    bad.bytes = blocks[0..127];
    try t.expectError(error.BadMxfp4Shape, mx.Host.init(bad, s, .low_first));
    bad = b;
    bad.shape[0] = std.math.maxInt(usize);
    try t.expectError(error.BadMxfp4Shape, mx.Host.init(bad, s, .low_first));
    scales[6] = 255;
    try t.expectError(error.BadMxfp4Scale, h.value(1, 1, 0));
}
