//! The learned change's shape on Nemotron: the shared constants (living/learn_shape.zig) and where its blocks sit.
const std = @import("std");
const Config = @import("config.zig").Config;
const shape = @import("living").learn_shape;

pub const block = shape.block;
pub const max_rank = shape.max_rank;
pub const max_blocks = shape.max_blocks;
pub const scale = shape.scale;
pub const avoid_dims = shape.avoid_dims;
pub const candidates = shape.candidates;
pub const hyper = shape.hyper;
pub const hyperAt = shape.hyperAt;
pub const settle = shape.settle;
pub const a_norm = shape.a_norm;
pub const unit = shape.unit;

/// Layer i's output projection as the checkpoint names its module: the one a lesson's change sits after.
pub fn moduleName(buf: []u8, c: Config, i: usize) ![]const u8 {
    const leaf = switch (c.kinds[i]) {
        .moe => "shared_experts.down_proj",
        .mamba => "out_proj",
        .attention => "o_proj",
    };
    return std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.{s}", .{ i, leaf });
}

/// Layer i's output projection's input width: what its change reads.
pub fn inWidth(c: Config, i: usize) usize {
    return switch (c.kinds[i]) {
        .moe => c.shared_width,
        .mamba => c.inner(),
        .attention => c.heads * c.head_dim,
    };
}

/// The widest site input over every layer.
pub fn maxIn(c: Config) usize {
    var w: usize = 0;
    for (0..c.layers) |i| w = @max(w, inWidth(c, i));
    return w;
}
