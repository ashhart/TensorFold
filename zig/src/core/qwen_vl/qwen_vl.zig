//! Image and video input every Qwen3.5-VL checkpoint shares, backend-neutral: settings, geometry, M-RoPE, tower tables.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const geometry = @import("geometry.zig");
pub const rope = @import("rope.zig");
pub const grid = @import("grid.zig");
pub const video = @import("video.zig");

test {
    _ = config;
    _ = geometry;
    _ = rope;
    _ = grid;
    _ = video;
}
