//! dots.mocr's HIP checkpoint owner; tower, decoder and serving integration are in development.
pub const config = @import("config.zig");
pub const image = @import("image.zig");
pub const weights = @import("hip_weights.zig");

test {
    _ = weights;
}
