//! Nemotron-H (Nemotron 3.5 Lightning) on Metal: FusedDecode's kernels for rounds, mlx_lm's prefill on ours, no MLX.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const layers = @import("layers.zig");
pub const tree = @import("tree.zig");
pub const head_tree = @import("head_tree.zig");
pub const encoder = @import("encoder.zig");
pub const timing = @import("timing.zig");
pub const model = @import("model.zig");
pub const backend = @import("backend.zig");
pub const mtp = @import("mtp.zig");
pub const head_block = @import("head_block.zig");
pub const gpu_round = @import("gpu_round.zig");
pub const copy_lanes = @import("copy_lanes.zig");
pub const prefill = @import("prefill.zig");
pub const prefill_launch = @import("prefill_launch.zig");
pub const simd_attention = @import("simd_attention.zig");

pub const Model = model.Model;
