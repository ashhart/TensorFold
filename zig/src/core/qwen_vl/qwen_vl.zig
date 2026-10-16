//! Image input every Qwen3.5-VL checkpoint shares, backend-neutral: settings, geometry, M-RoPE, the tower's tables.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const geometry = @import("geometry.zig");
pub const rope = @import("rope.zig");
pub const grid = @import("grid.zig");

test {
    _ = config;
    _ = geometry;
    _ = rope;
    _ = grid;
}
