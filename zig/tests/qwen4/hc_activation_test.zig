//! Independent binary64 values and actual-source controls for both hyper-connection activation roles.
const std = @import("std");
const core = @import("core");
const h = core.hc_activation;
const q = core.quant.convert;
extern fn tf_hc_activation_cpu([*]const u16, [*]u16, c_int, c_int, c_int, c_int, c_int) void;
fn f(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}
test "HC activation: binary64 bounds all finite BF16 inputs and every stream count" {
    var x: [256]u16 = undefined;
    var out: [256]u16 = undefined;
    for (2..9) |streams| for ([_]c_int{ 0, 1 }) |injection| {
        for (0..256) |page| {
            for (&x, 0..) |*v, i| v.* = @intCast(page * 256 + i);
            tf_hc_activation_cpu(&x, &out, 256, @intCast(streams), injection, 0, 0);
            for (x, out) |input, result| {
                const value = f(input);
                if (!std.math.isFinite(value)) continue;
                const divided: f64 = f(q.bf16(value / @as(f32, @floatFromInt(streams))));
                const exponential = @exp(-divided);
                const sigmoid = 1 / (1 + exponential);
                const want: f64 = if (injection == 1) 2 * @as(f64, f(q.bf16(@floatCast(sigmoid)))) else divided / (1 + exponential);
                const got: f64 = f(result);
                const allowance = @abs(want) * 0x1p-8 + @abs(want) * 0x1p-20 + 0x1p-133;
                if (@abs(got - want) > allowance) std.debug.print("HC input{x} mode{d} streams{d}: {d} vs {d}\n", .{ input, injection, streams, got, want });
                try std.testing.expect(@abs(got - want) <= allowance);
            }
        }
    };
}
test "HC activation: scalar controls and in-place alias policy" {
    var x = [_]u16{ 0x7fc0, 0x3f80, 0x4000, 0x4080 };
    var out: [4]u16 = @splat(0xffff);
    tf_hc_activation_cpu(&x, &out, 4, 4, 0, 1, 0);
    try std.testing.expect(!std.math.isNan(f(out[0])));
    x[0] = 0;
    tf_hc_activation_cpu(&x, &out, 4, 4, 0, 2, 0);
    const first = out;
    tf_hc_activation_cpu(&x, &out, 4, 4, 0, 2, 1);
    try std.testing.expect(!std.mem.eql(u16, &first, &out));
    @memset(&out, 0xffff);
    tf_hc_activation_cpu(&x, &out, 4, 4, 0, 3, 0);
    try std.testing.expectEqual(@as(u16, 0xffff), out[3]);
    const p: h.Params = .{ .rows = 1, .cols = 4, .streams = 4, .injection = 0 };
    try std.testing.expectEqual(@as(usize, 4), try h.validate(p, .{ .address = 0x1000, .len = 8 }, .{ .address = 0x1000, .len = 8 }));
    try std.testing.expectError(error.HcActivationAlias, h.validate(p, .{ .address = 0x1000, .len = 8 }, .{ .address = 0x1002, .len = 8 }));
}
