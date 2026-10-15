//! Native Metal Step-5 (stepfun-ai/Step-5-Preview): sliding and full GQA attention, 352 routed experts, fp32 residual.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const Model = @import("model.zig").Model;
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const backend = @import("backend.zig");

test {
    _ = config;
    _ = backend;
}
