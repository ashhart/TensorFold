//! `zig build hip-dense-tune`: times the dot2 and gfx12 WMMA BF16 dense tiles at Qwen3-8B's projection shapes.

test {
    _ = @import("kernel_tests/dense_tune.zig");
}
