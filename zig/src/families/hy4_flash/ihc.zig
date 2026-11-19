//! FP32 weights and parameters for Hy4-flash's independent hyper-connections.
const std = @import("std");
const core = @import("core");
const loader = @import("loader.zig");

pub const streams = 4;
pub const Which = union(enum) { attention: u32, feed_forward: u32, head };

pub const Shape = extern struct {
    rows: u32,
    width: u32,
    gates: u32,
    eps: f32,
    gate_eps: f32,
    post_magnitude: f32,

    pub fn check(self: Shape) !void {
        if (self.rows == 0 or self.rows > std.math.maxInt(u16) or self.width == 0 or self.width > 8192 or
            (self.gates != streams and self.gates != 2 * streams) or !std.math.isFinite(self.eps) or self.eps <= 0 or
            !std.math.isFinite(self.gate_eps) or self.gate_eps < 0 or
            !std.math.isFinite(self.post_magnitude) or self.post_magnitude < 0) return error.BadHy4IhcShape;
    }
};

pub const Weights = struct {
    project: core.checkpoint_host.Tensor,
    bias: core.checkpoint_host.Tensor,
    scale: core.checkpoint_host.Tensor,
    shape: Shape,

    pub fn read(model: *loader.Loaded, which: Which, rows: u32) !Weights {
        const config = model.config.value.text_config;
        if (config.hc_mult != streams) return error.BadHy4IhcShape;
        var prefix: [120]u8 = undefined;
        const base = switch (which) {
            .attention => |layer| try blockPrefix(model, &prefix, layer, "hc_attn_layer"),
            .feed_forward => |layer| try blockPrefix(model, &prefix, layer, "hc_mlp_layer"),
            .head => "model.hc_head.hc_head_",
        };
        const gates: u32 = if (which == .head) streams else 2 * streams;
        const shape: Shape = .{
            .rows = rows,
            .width = config.hidden_size,
            .gates = gates,
            .eps = config.rms_norm_eps,
            .gate_eps = config.hc_eps,
            .post_magnitude = config.hc_magnitude,
        };
        try shape.check();
        var name: [128]u8 = undefined;
        return .{
            .project = try model.checkpoint.expect(try std.fmt.bufPrint(&name, "{s}fn", .{base}), .f32, &.{ gates, streams * shape.width }),
            .bias = try model.checkpoint.expect(try std.fmt.bufPrint(&name, "{s}base", .{base}), .f32, &.{gates}),
            .scale = try model.checkpoint.expect(try std.fmt.bufPrint(&name, "{s}scale", .{base}), .f32, &.{gates / streams}),
            .shape = shape,
        };
    }
};

fn blockPrefix(model: *const loader.Loaded, buffer: []u8, layer: u32, name: []const u8) ![]const u8 {
    if (layer >= model.config.value.text_config.num_hidden_layers) return error.BadHy4IhcLayer;
    return std.fmt.bufPrint(buffer, "model.layers.{d}.{s}.hc_pre.hc_", .{ layer, name });
}
