//! Inkling 2 Mini Metal host tests rooted with the production engine sources.
const core_base = @import("core_base");
pub const safetensors = core_base.safetensors;
pub const checkpoint_file_cache = core_base.checkpoint_file_cache;
pub const checkpoint_host = core_base.checkpoint_host;
pub const contract = core_base.contract;
pub const picture = core_base.picture;
pub const bf16_round = core_base.bf16_round;
pub const checkpoint = core_base.checkpoint;
pub const Checkpoint = core_base.Checkpoint;
pub const lane_projection = @import("core/lane_projection.zig");
pub const row_projection = @import("core/row_projection.zig");
pub const affine4 = @import("core/affine4.zig");
pub const affine4_lane = @import("core/affine4_lane.zig");
const serving = if (@import("builtin").os.tag.isDarwin()) @import("native/inkling2_serving_test.zig") else struct {};

test {
    _ = @import("families/inkling2/metal/host_test.zig");
    _ = @import("inkling2_native.zig");
    _ = @import("native/inkling2_admit_test.zig");
    _ = serving;
}
