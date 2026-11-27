//! Real Step-5 geometry code with Metal objects left untouched by every selected fixture.
pub const safetensors = @import("src/core/safetensors.zig");
pub const checkpoint_file_cache = @import("src/core/checkpoint_file_cache.zig");
test "step5 host: imports" {
    _ = @import("src/families/step5/config.zig");
    _ = @import("src/families/step5/weights.zig");
    _ = @import("src/families/step5/vision.zig");
}
