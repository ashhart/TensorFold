//! The production CUDA quantizer under a CPU warp shim, with independent flags, guards and finite references.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const ref = @import("w4a8_ref.zig");
const t = std.testing;
extern fn tf_w4a8_quantize_host([*]const u16, [*]u8, [*]f32, [*]f32, c_int, c_int, c_int, *u32) c_int;
extern fn tf_w4a8_quantize_parameters() c_int;
test "w4a8 host: actual quantizer flags every nonfinite lane in every group before integer conversion" {
    try t.expectEqual(@as(c_int, 7), tf_w4a8_quantize_parameters());
    for ([_]usize{ 64, 192, 320 }) |k| {
        const stride = k + 64;
        const x = try t.allocator.alloc(u16, 3 * stride);
        defer t.allocator.free(x);
        const codes = try t.allocator.alloc(u8, 3 * k);
        defer t.allocator.free(codes);
        const scales = try t.allocator.alloc(f32, 3 * k / 64);
        defer t.allocator.free(scales);
        const sums = try t.allocator.alloc(f32, 3 * k / 64);
        defer t.allocator.free(sums);
        for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |plant| {
            for (0..3) |row| for (0..k / 64) |group| for (0..64) |lane| {
                @memset(x, c.toBf16(1));
                @memset(codes, 0xa5);
                @memset(scales, -19);
                @memset(sums, -23);
                x[row * stride + group * 64 + lane] = c.toBf16(plant);
                var flag = [_]u32{ 0x12345678, 0, 0x87654321 };
                try t.expectEqual(@as(c_int, 0), tf_w4a8_quantize_host(x.ptr, codes.ptr, scales.ptr, sums.ptr, 3, @intCast(k), @intCast(stride), &flag[1]));
                try t.expectEqualSlices(u32, &.{ 0x12345678, 1, 0x87654321 }, &flag);
                try t.expect(std.mem.allEqual(u8, codes[(row * k + group * 64)..][0..64], 0xa5));
                try t.expectEqual(@as(f32, -19), scales[row * (k / 64) + group]);
                try t.expectEqual(@as(f32, -23), sums[row * (k / 64) + group]);
            };
        }
    }
}
test "w4a8 host: finite normalization and sums retain their reference, while overflow flags instead" {
    var x: [64]u16 = undefined;
    var codes: [64]u8 = undefined;
    var scales: [1]f32 = undefined;
    var sums: [1]f32 = undefined;
    for ([_]f32{ 0, 1, 0x1p-133, 127, 65536 }) |scale| {
        for (&x, 0..) |*v, i| v.* = c.toBf16(if (i % 2 == 0) scale else -scale);
        var flag: u32 = 0;
        try t.expectEqual(@as(c_int, 0), tf_w4a8_quantize_host(&x, &codes, &scales, &sums, 1, 64, 64, &flag));
        try t.expectEqual(@as(u32, 0), flag);
        const want = try ref.group(&x);
        for (codes, 0..) |got, i| try t.expectEqual(@as(u8, @bitCast(want.codes[i / 32 * 32 + ref.inputAt(i % 32)])), got);
        try t.expectApproxEqAbs(want.scale, scales[0], @abs(want.scale) * 0x1p-21 + 0x1p-149);
        try t.expectEqual(@as(f32, @floatCast(want.sum)), sums[0]);
    }
    @memset(&x, 0x7f7f);
    var flag: u32 = 0;
    _ = tf_w4a8_quantize_host(&x, &codes, &scales, &sums, 1, 64, 64, &flag);
    try t.expectEqual(@as(u32, 1), flag);
}
