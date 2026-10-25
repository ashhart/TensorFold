//! Focused scalar-source regressions name both rounding policies instead of silently conflating their bits.
const std = @import("std");
extern fn tf_hc_pointwise_cpu([*]const u16, [*]const u16, [*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int, c_int, c_int) void;
test "HC pointwise: native FMA and separate BF16-product seam are both explicit" {
    const x = [_]u16{ 0xbf82, 0xbf82 };
    const branch = [_]u16{0x3f81};
    const gate = [_]u16{ 0x3f81, 0x3f81 };
    var out: [2]u16 = undefined;
    tf_hc_pointwise_cpu(&x, &branch, &gate, &out, 1, 1, 2, 0, 0, 0, 0);
    try std.testing.expectEqualSlices(u16, &.{ 0x3880, 0x3880 }, &out);
    tf_hc_pointwise_cpu(&x, &branch, &gate, &out, 1, 1, 2, 0, 1, 0, 0);
    try std.testing.expectEqualSlices(u16, &.{ 0, 0 }, &out);
}
test "HC pointwise: BF16-sigmoid attention gate and native scalar seam stay distinguishable" {
    const x = [_]u16{0x3e01};
    const gate = [_]u16{0x3f00};
    var out: [1]u16 = undefined;
    tf_hc_pointwise_cpu(&x, &gate, &gate, &out, 1, 1, 1, 1, 0, 0, 0);
    try std.testing.expectEqual(@as(u16, 0x3da1), out[0]);
    tf_hc_pointwise_cpu(&x, &gate, &gate, &out, 1, 1, 1, 1, 1, 0, 0);
    try std.testing.expectEqual(@as(u16, 0x3da0), out[0]);
}

fn value(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}
test "HC output gate: all finite BF16 gates preserve representable large-input products" {
    const inputs = [_]u16{ 0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x0080, 0x8080, 0x3000, 0xb000, 0x3e01, 0xbe01, 0x3f80, 0xbf80, 0x4780, 0xc780, 0x7f7f, 0xff7f };
    var x: [inputs.len]u16 = inputs;
    var gates: [inputs.len]u16 = undefined;
    var out: [inputs.len]u16 = undefined;
    for (0..65536) |bits| {
        const g = value(@intCast(bits));
        if (!std.math.isFinite(g)) continue;
        @memset(&gates, @intCast(bits));
        tf_hc_pointwise_cpu(&x, &gates, &gates, &out, 1, inputs.len, 1, 1, 0, 0, 0);
        const e = @exp(-@abs(g));
        for (x, out) |input, result| {
            const want = value(input) * (if (g < 0) e else @as(f64, 1)) / (1 + e);
            const allowance = @abs(want) * (0x1p-8 + 32 * 0x1p-24) + 0x1p-133;
            const got = value(result);
            if (@abs(got - want) > allowance) std.debug.print("gate{x} x{x}: {d} vs {d} bound{d}\n", .{ bits, input, got, want, allowance });
            try std.testing.expect(@abs(got - want) <= allowance);
        }
    }
}
