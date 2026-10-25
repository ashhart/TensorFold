//! Recording Metal and CUDA launch tables with the actual shared arithmetic body on the CPU.
test {
    _ = @import("tests/qwen4/metal_session_test.zig");
    _ = @import("tests/qwen4/hc_launch_test.zig");
    _ = @import("tests/qwen4/projection_launch_test.zig");
}
