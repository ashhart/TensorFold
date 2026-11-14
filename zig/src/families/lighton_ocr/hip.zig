//! LightOnOCR-3 1B on HIP; the existing Qwen3.5 family owns the 0.8B and 4B models.
pub const config = @import("config.zig");
pub const native = @import("backend/hip/native.zig");
pub const image = @import("image.zig");
pub const weights = @import("weights.zig");
pub const vision = @import("backend/hip/vision.zig");
pub const snapshot = @import("backend/hip/snapshot.zig");
pub const ops = @import("backend/hip/ops.zig");
pub const text = @import("backend/hip/text.zig");
pub const tower = @import("backend/hip/tower.zig");

test {
    _ = native;
    _ = config;
    _ = image;
    _ = weights;
    _ = vision;
    _ = snapshot;
    _ = ops;
    _ = text;
    _ = tower;
}
