//! Native Metal LFM2-MoE (LFM2.5-24B-A2B): short convs and attention, sigmoid-routed experts, a tied affine head.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const Model = @import("model.zig").Model;
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const backend = @import("backend.zig");
