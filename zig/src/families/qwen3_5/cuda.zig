//! The Qwen3.5 hybrids on CUDA (2B, Qwen3.8-27B): the engine and its native family entry; Metal's 2B is qwen3_5.zig.
pub const config = @import("config.zig");
pub const geometry = @import("cuda_geometry.zig");
pub const engine = @import("cuda_engine.zig");
pub const native = @import("cuda_native.zig");
pub const draft = @import("cuda_draft.zig");
pub const vision = @import("cuda_vision.zig");
pub const reuse = @import("cuda_reuse.zig");

test {
    _ = config;
    _ = geometry;
    _ = @import("draw.zig");
    _ = @import("cuda_draft_weights.zig");
    _ = draft;
    _ = vision;
    _ = @import("cuda_vision_prompt.zig");
    _ = reuse;
    _ = native;
}
