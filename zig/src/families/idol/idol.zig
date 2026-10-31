//! Idol image to 3D: its denoiser on Metal (a denoiser beside the lane engine, not a token family).
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const model = @import("model.zig");
pub const reference = @import("reference.zig");
pub const Model = model.Model;
pub const Condition = model.Condition;
pub const Timing = model.Timing;

test {
    _ = config;
    _ = host;
    _ = weights;
    _ = kernels;
    _ = model;
    _ = reference;
}
