//! Native Metal Spark2.5 (XHToken/Spark-X2.5/2.6): sliding and full GQA with a head gate, exact GELU, fp32 residual.
pub const config = @import("config.zig");
pub const weights = @import("metal_weights.zig");
pub const Model = @import("metal_model.zig").Model;
pub const state = @import("metal_state.zig");
pub const forward = @import("metal_forward.zig");
pub const backend = @import("metal_backend.zig");
pub const budget = @import("metal_budget.zig");
pub const residency = @import("metal_residency.zig");
pub const snapshot = @import("metal_snapshot.zig");

test {
    _ = config;
    _ = state;
    _ = budget;
    _ = snapshot;
}
