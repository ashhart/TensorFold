//! Gemma 4 dense on Metal: config admission and the decode glue around the core lane projections.
pub const config = @import("config.zig");
pub const kernels = @import("kernels.zig");
pub const weights = @import("weights.zig");
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const Model = @import("model.zig").Model;
pub const backend = @import("backend.zig");

test {
    _ = config;
}
