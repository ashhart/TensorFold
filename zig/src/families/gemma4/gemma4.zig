//! Gemma 4 dense on Metal: config admission and the decode glue around the core lane projections.
pub const config = @import("config.zig");
pub const kernels = @import("kernels.zig");

test {
    _ = config;
}
