//! Real Metal config and route tests with only the projection declarations their width rules read.
pub const row_projection = @import("src/core/row_projection.zig");
pub const lane_projection = @import("src/core/lane_projection.zig");
pub const quant = @import("src/core/quant/quant.zig");
pub const qwen35 = struct {
    pub const config = @import("src/families/qwen3_5/config.zig");
};

test {
    _ = @import("src/families/qwen27/config.zig");
    _ = @import("src/native/qwen27_engine.zig");
}
