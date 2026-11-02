//! GLM-OCR's HIP components; native serving is registered only after graph validation.
pub const config = @import("config.zig");
pub const schema = @import("schema.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const processor = @import("processor.zig");
pub const weights = @import("hip_weights.zig");
pub const ops = @import("hip_ops.zig");
pub const vision = @import("hip_vision.zig");
pub const decoder = @import("hip_text.zig");
pub const positions = @import("positions.zig");
pub const input = @import("hip_input.zig");
pub const native = @import("hip_native.zig");

test {
    _ = weights;
    _ = ops;
    _ = vision;
    _ = decoder;
    _ = input;
    _ = native;
}
