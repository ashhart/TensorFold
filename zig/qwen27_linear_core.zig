//! Actual loader dependencies at a recording Metal boundary.
pub const checkpoint_metal = @import("src/core/checkpoint_metal.zig");
pub const dense_pocket = @import("src/core/dense_pocket.zig");
pub const row_projection = @import("src/core/row_projection.zig");
pub const lane_projection = @import("src/core/lane_projection.zig");
