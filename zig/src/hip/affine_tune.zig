//! `zig build hip-affine-tune`: times the MLX affine prompt tiles on this GPU and proposes table rows.

test {
    _ = @import("kernel_tests/affine/tune.zig");
}
