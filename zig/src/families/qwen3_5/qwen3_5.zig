//! Native Metal Qwen3.5-2B, with its tied affine head and hybrid recurrent/attention state.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const Model = @import("model.zig").Model;
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const backend = @import("backend.zig");
/// The 35B-A3B (qwen3_5_moe) on the 27B's Metal engine.
pub const native_moe = @import("metal_native_moe.zig");
