//! Qwen3.5-2B on CUDA: the engine and its native family entry (the Metal engine is qwen3_5.zig).
pub const config = @import("config.zig");
pub const engine = @import("cuda_engine.zig");
pub const native = @import("cuda_native.zig");

test {
    _ = config;
    _ = @import("draw.zig");
}
