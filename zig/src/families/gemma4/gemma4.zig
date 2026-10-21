//! Gemma 4 dense on Metal: config admission and the decode glue around the core lane projections.
pub const config = @import("config.zig");
pub const kernels = @import("kernels.zig");
pub const weights = @import("weights.zig");
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const prompt = @import("prompt.zig");
pub const Model = @import("model.zig").Model;
pub const backend = @import("backend.zig");
pub const dflash = @import("dflash.zig");
pub const dflash_weights = @import("dflash_weights.zig");
pub const assistant = @import("assistant.zig");
pub const assistant_weights = @import("assistant_weights.zig");
pub const image = @import("image.zig");
pub const vision = @import("vision.zig");
pub const vision_unified = @import("vision_unified.zig");
pub const residency = @import("residency.zig");
pub const snapshot = @import("snapshot.zig");

test {
    _ = assistant;
    _ = assistant_weights;
    _ = config;
    _ = dflash_weights;
    _ = image;
    _ = snapshot;
    _ = vision;
}

pub const affine_mm = @import("affine_mm.zig");

pub const contracts = @import("contracts.zig");
pub const fused_lane = @import("fused_lane.zig");
