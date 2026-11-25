//! Analyze the actual Inkling 2 Mini native opener without loading a device or checkpoint.
const core_base = @import("core_base");
pub const safetensors = core_base.safetensors;
pub const checkpoint_file_cache = core_base.checkpoint_file_cache;
pub const checkpoint_host = core_base.checkpoint_host;
pub const contract = core_base.contract;
pub const picture = core_base.picture;
pub const bf16_round = core_base.bf16_round;
pub const checkpoint = core_base.checkpoint;
pub const Checkpoint = core_base.Checkpoint;
pub const lane_projection = @import("src/core/lane_projection.zig");
pub const row_projection = @import("src/core/row_projection.zig");
pub const affine4 = @import("src/core/affine4.zig");
pub const affine4_lane = @import("src/core/affine4_lane.zig");
export fn compileInkling2Native() usize {
    return @intFromPtr(&@import("src/inkling2_native.zig").open);
}
