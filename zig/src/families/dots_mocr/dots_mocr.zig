//! dots.mocr's checked checkpoint and processor contracts; native HIP execution is in development.
pub const config = @import("config.zig");
pub const image = @import("image.zig");
pub const weights = @import("weights.zig");

test {
    _ = @import("config_test.zig");
    _ = @import("image_test.zig");
    _ = @import("ocr_test.zig");
    _ = @import("weights_test.zig");
}
