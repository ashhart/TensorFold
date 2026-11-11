const std = @import("std");

pub fn firstNonFinite(comptime T: type, values: []const T) ?usize {
    const Bits = switch (T) {
        f32 => u32,
        u16, f16 => u16,
        else => @compileError("expected FP32/FP16 values or BF16 bits"),
    };
    const exponent: Bits = if (T == f32) 0x7f800000 else if (T == f16) 0x7c00 else 0x7f80;
    const V = @Vector(16, Bits);
    const mask: V = @splat(exponent);
    var i: usize = 0;
    while (values.len - i >= 16) : (i += 16) {
        const bits: V = @bitCast(values[i..][0..16].*);
        if (@reduce(.Or, (bits & mask) == mask)) break;
    }
    for (values[i..], i..) |value, at| {
        const bits: Bits = @bitCast(value);
        if (bits & exponent == exponent) return at;
    }
    return null;
}

test "finite scan preserves first bad index across SIMD boundaries and tails" {
    inline for (.{ f32, u16, f16 }) |T| {
        const Bits = if (T == f32) u32 else u16;
        const valid = if (T == f32)
            [_]Bits{ 0, 0x80000000, 1, 0x007fffff, 0x7f7fffff, 0xff7fffff }
        else if (T == f16)
            [_]Bits{ 0, 0x8000, 1, 0x03ff, 0x7bff, 0xfbff }
        else
            [_]Bits{ 0, 0x8000, 1, 0x007f, 0x7f7f, 0xff7f };
        const invalid = if (T == f32)
            [_]Bits{ 0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001 }
        else if (T == f16)
            [_]Bits{ 0x7c00, 0xfc00, 0x7e00, 0x7c01 }
        else
            [_]Bits{ 0x7f80, 0xff80, 0x7fc0, 0x7f81 };
        var values: [53]T = undefined;
        for (&values, 0..) |*value, i| value.* = @bitCast(valid[i % valid.len]);
        for (0..values.len + 1) |len| try std.testing.expectEqual(null, firstNonFinite(T, values[0..len]));
        for (invalid) |bad| {
            for (0..values.len) |at| {
                const saved = values[at];
                values[at] = @bitCast(bad);
                try std.testing.expectEqual(@as(?usize, at), firstNonFinite(T, &values));
                values[at] = saved;
            }
        }
        values[16] = @bitCast(invalid[0]);
        values[32] = @bitCast(invalid[1]);
        try std.testing.expectEqual(@as(?usize, 16), firstNonFinite(T, &values));
    }
}
