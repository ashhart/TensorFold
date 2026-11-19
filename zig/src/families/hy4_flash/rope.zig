//! Hy4-flash uses adjacent rotary pairs with interleaved time, height and width positions.
const std = @import("std");

pub const pairs = 32;
pub const Config = struct {
    rope_type: []const u8,
    rope_theta: f32,
    mrope_interleaved: bool,
    mrope_section: [3]u32,

    pub fn parameters(self: Config, width: u32) !Parameters {
        if (width != pairs * 2 or !self.mrope_interleaved or !std.mem.eql(u8, self.rope_type, "default") or
            !std.math.isFinite(self.rope_theta) or self.rope_theta <= 0)
            return error.UnsupportedHy4Rotary;
        var result: Parameters = undefined;
        var counts: [3]u32 = @splat(0);
        for (&result.frequencies, &result.axes, 0..) |*freq, *axis, i| {
            axis.* = switch (i % 3) {
                1 => if (i / 3 < self.mrope_section[1]) 1 else 0,
                2 => if (i / 3 < self.mrope_section[2]) 2 else 0,
                else => 0,
            };
            counts[axis.*] += 1;
            freq.* = 1 / std.math.pow(f32, self.rope_theta, @as(f32, @floatFromInt(i)) / pairs);
        }
        if (!std.mem.eql(u32, &counts, &self.mrope_section)) return error.UnsupportedHy4Rotary;
        return result;
    }
};

pub const Parameters = extern struct { frequencies: [pairs]f32, axes: [pairs]u32 };

/// Strides are in BF16 elements. Both base pointers address the first rotary element of head zero.
pub const View = extern struct {
    rows: u32,
    heads: u32,
    input_stride: u32,
    output_stride: u32,

    pub fn check(self: View) !void {
        if (self.rows == 0 or self.rows > std.math.maxInt(u16) or self.heads == 0 or self.heads > 64 or
            self.input_stride < 2 * pairs or self.output_stride < 2 * pairs) return error.BadHy4RotaryView;
    }
};

test "checkpoint rotary tables use distinct frequency bases and exactly twelve time pairs" {
    const config = @import("config.zig");
    const parsed = try config.parse(std.testing.allocator, @embedFile("config_fixture.json"));
    defer parsed.deinit();
    const sliding = try parsed.value.text_config.rotary(.sliding_attention);
    const sparse = try parsed.value.text_config.rotary(.deepseek_sparse_attention);
    try std.testing.expectEqualSlices(u32, &sliding.axes, &sparse.axes);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, sliding.axes[0..3]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 0, 0 }, sliding.axes[27..32]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), sliding.frequencies[16], 0.0000001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.00031622777), sparse.frequencies[16], 0.0000001);
}
