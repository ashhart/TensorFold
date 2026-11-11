const std = @import("std");
const rows = @import("rows.zig");

pub const Triple = extern struct { a: f32, b: f32, c: f32 };

fn bits(value: u32) f32 {
    return @bitCast(value);
}

pub fn expected(value: Triple) u32 {
    const result = @mulAdd(f32, value.a, value.b, value.c);
    return if (std.math.isNan(result)) 0x7fc00000 else @bitCast(result);
}

const special = [_]Triple{
    .{ .a = bits(0x3f800001), .b = bits(0x3f7fffff), .c = bits(0x28000001) },
    .{ .a = bits(0x3f800001), .b = bits(0x3f7fffff), .c = bits(0x28000000) },
    .{ .a = bits(0x3f800001), .b = bits(0x3f7fffff), .c = bits(0x27ffffff) },
    .{ .a = bits(1), .b = 0.5, .c = 0 },
    .{ .a = bits(1), .b = bits(0x3f000001), .c = 0 },
    .{ .a = bits(0x80000001), .b = 0.5, .c = 0 },
    .{ .a = bits(1), .b = 0.5, .c = bits(1) },
    .{ .a = bits(1), .b = 0.5, .c = bits(0x80000001) },
    .{ .a = bits(0x00800000), .b = bits(0x3f7fffff), .c = 0 },
    .{ .a = bits(0x7f7fffff), .b = 1, .c = bits(0x73000000) },
    .{ .a = bits(0x7f7fffff), .b = 1, .c = -bits(0x73000000) },
    .{ .a = bits(0x7f7fffff), .b = 2, .c = -bits(0x7f7fffff) },
    .{ .a = -0.0, .b = 1, .c = -0.0 },
    .{ .a = -0.0, .b = 1, .c = 0 },
    .{ .a = 0, .b = std.math.inf(f32), .c = 1 },
};

pub fn fill(values: []Triple, mode: rows.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (values) |*value| value.* = switch (mode) {
        .random => .{ .a = bits(random.int(u32)), .b = bits(random.int(u32)), .c = bits(random.int(u32)) },
        .nonfinite => .{ .a = random.float(f32), .b = random.float(f32), .c = random.float(f32) },
        .stress => .{ .a = 1024, .b = 1024, .c = -1048576 + 64 },
    };
    if (mode == .random) @memcpy(values[0..@min(values.len, special.len)], special[0..@min(values.len, special.len)]);
    if (mode == .nonfinite) {
        const exceptional = [_]Triple{
            .{ .a = std.math.nan(f32), .b = 1, .c = 0 },
            .{ .a = std.math.inf(f32), .b = 1, .c = 0 },
            .{ .a = -std.math.inf(f32), .b = 1, .c = 0 },
            .{ .a = std.math.inf(f32), .b = 1, .c = -std.math.inf(f32) },
        };
        @memcpy(values[0..@min(values.len, exceptional.len)], exceptional[0..@min(values.len, exceptional.len)]);
    }
}

test "SAM FMA oracle keeps subnormals, halfway cases and the fused overflow cancellation" {
    const want = [_]u32{ 0x3f800001, 0x3f800000, 0x3f800000, 0, 1, 0x80000000, 2, 0x80000000, 0x00800000, 0x7f800000, 0x7f7ffffe, 0x7f7fffff, 0x80000000, 0, 0x7fc00000 };
    for (special, want) |value, reference| try std.testing.expectEqual(reference, expected(value));
}
