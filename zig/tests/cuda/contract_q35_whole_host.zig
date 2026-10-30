//! Real CUDA tile-loader addressing without CUDA instructions or a device.
const std = @import("std");
extern fn tf_test_q35_whole() c_int;
test "Qwen whole tiles ignore page metadata and retain every head offset, including masked tails" {
    try std.testing.expectEqual(@as(c_int, 1), tf_test_q35_whole());
}
