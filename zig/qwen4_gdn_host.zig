//! Recording launch qualification uses real native adapters without a shader compiler or device.
test {
    _ = @import("tests/qwen4/cuda_draw_launch_test.zig");
    _ = @import("tests/qwen4/metal_attention_launch_test.zig");
    _ = @import("tests/qwen4/hc_ops_launch_test.zig");
    _ = @import("tests/qwen4/attention_launch_test.zig");
    _ = @import("tests/qwen4/gdn_launch_test.zig");
}
