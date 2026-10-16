//! Qwen3.5 / Qwen3.6 text checkpoints on the host for the GPU backends: configuration, tensor index, projections, cuts.

pub const config = @import("weights/config.zig");
pub const table = @import("weights/table.zig");
pub const host = @import("weights/host.zig");
pub const checkpoint = @import("weights/checkpoint.zig");
pub const slicing = @import("weights/slicing.zig");
pub const Config = config.Config;
pub const Spec = config.Spec;
pub const Checkpoint = checkpoint.Checkpoint;

test {
    _ = config;
    _ = table;
    _ = @import("weights/shard.zig");
    _ = @import("weights/projection.zig");
    _ = @import("weights/checkpoint_test.zig");
    _ = @import("weights/real_test.zig");
    @import("std").testing.refAllDecls(slicing);
    _ = @import("config.zig");
}
