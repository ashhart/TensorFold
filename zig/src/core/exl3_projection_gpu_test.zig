//! GPU qualification tests, split by the path they exercise; `test-exl3-gpu` runs all three files.
test {
    _ = @import("exl3_projection_gpu_test_decode.zig");
    _ = @import("exl3_projection_gpu_test_prefill.zig");
}
