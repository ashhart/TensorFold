//! GLM-OCR's HIP components; native serving is registered only after graph validation.
pub const config = @import("config.zig");
pub const schema = @import("schema.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const weights = @import("hip_weights.zig");

test {
    _ = weights;
}
