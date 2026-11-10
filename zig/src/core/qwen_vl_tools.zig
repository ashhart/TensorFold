//! The Qwen-VL tower on Metal with its settings, shards, tables and float64 reference, and core's pictures: one module.
pub const config = @import("qwen_vl/config.zig");
pub const Config = config.Config;
pub const tower = @import("qwen_vl/tower.zig");
pub const weights = @import("qwen_vl/weights.zig");
pub const shards = @import("qwen_vl/shards.zig");
pub const grid = @import("qwen_vl/grid.zig");
pub const reference = @import("qwen_vl/reference.zig");
pub const picture = @import("picture/picture.zig");

/// What a family's picture rules read from their `picture` import, so a tool can hand them this module instead.
pub const Rgb = picture.Rgb;
pub const resample = picture.resample;
pub const patches = picture.patches;

test {
    _ = config;
    _ = tower;
    _ = weights;
    _ = shards;
    _ = grid;
    _ = reference;
}
