//! Metal device checks for Inkling 2 Mini's image tower: kernels, the 23-layer tower and the original checkpoint.
test {
    _ = @import("families/inkling2/metal/vision_test.zig");
    _ = @import("families/inkling2/metal/vision_real_test.zig");
}
