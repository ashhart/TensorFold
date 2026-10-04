//! Kimi K3 on Metal: KDA and MLA layers, latent MoE with MXFP4 experts, attention residuals, mixed lane rounds.
const std = @import("std");

pub const config = @import("config.zig");
pub const kernels = @import("kernels.zig");
pub const store = @import("store.zig");
pub const shards = @import("shards.zig");
pub const weights = @import("weights.zig");
pub const prepare = @import("prepare.zig");
pub const state = @import("state.zig");
pub const round = @import("round.zig");
pub const experts = @import("experts.zig");
pub const layer = @import("layer.zig");
pub const parallel = @import("parallel.zig");
pub const forward = @import("forward.zig");

test {
    std.testing.refAllDecls(@This());
}
