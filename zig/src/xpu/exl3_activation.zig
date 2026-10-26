// SPDX-License-Identifier: Apache-2.0
const std = @import("std");

const inv_sqrt128: f32 = 0.08838834764831845;

fn hadamard128(block: *[128]f32) void {
    var d: usize = 1;
    while (d < 128) : (d *= 2) {
        var base: usize = 0;
        while (base < 128) : (base += 2 * d) {
            for (0..d) |j| {
                const a = block[base + j];
                const b = block[base + j + d];
                block[base + j] = a + b;
                block[base + j + d] = a - b;
            }
        }
    }
}

pub fn rotateInBlock(x: []const f32, suh: []const f32, rot: []f32) void {
    std.debug.assert(x.len == 128 and suh.len == 128 and rot.len == 128);
    for (0..128) |i| rot[i] = x[i] * suh[i];
    hadamard128(rot);
    for (rot) |*v| v.* *= inv_sqrt128;
}

fn twoHalfBack(f: f32) f32 {
    const hi: f16 = @floatCast(f);
    const lo: f16 = @floatCast(f - @as(f32, @floatCast(hi)));
    return @as(f32, @floatCast(hi)) + @as(f32, @floatCast(lo));
}

test "0x3f800801 stays in fp32 rot; the old hi+lo fp16 path changes the bits" {
    const want: u32 = 0x3f800801;
    const f: f32 = @bitCast(want);
    try std.testing.expectEqual(want, @as(u32, @bitCast(f)));
    try std.testing.expect(want != @as(u32, @bitCast(twoHalfBack(f))));
}
