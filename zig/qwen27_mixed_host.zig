//! CPU checks for the production projection stage and the Metal head-reader ABI.
const std = @import("std");
const abi = @import("src/families/qwen27/glue.zig");
pub const checkpoint_host = @import("src/core/checkpoint.zig");
pub const safetensors = @import("src/core/safetensors.zig");
pub const Checkpoint = checkpoint_host.Checkpoint;
pub const quant = struct {
    pub const keys = @import("src/core/quant/keys.zig");
};
test {
    _ = @import("src/families/qwen27/attention_input.zig");
    _ = abi;
    _ = @import("tests/contract/q27_mixed_inputs.zig");
}
test "qwen27 mixed: grouped heads validate row pitches without changing the default ABI prefix" {
    var h = abi.Head{ .width = 256, .rows = 64, .stride = 256, .activation = .bf16, .gain = .bf16, .eps = 1e-6 };
    try std.testing.expectEqual(@as(u32, 64), try abi.headThreads(h));
    try std.testing.expectEqual(@as(u32, 1), h.heads);
    try std.testing.expectEqual(@as(u32, 0), h.row_stride);
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(abi.Head));
    try std.testing.expectEqual(@as(usize, 20), @offsetOf(abi.Head, "eps"));
    h.heads = 4;
    h.row_stride = 2048;
    try std.testing.expectEqual(@as(u32, 64), try abi.headThreads(h));
    h.row_stride = 1023;
    try std.testing.expectError(error.BadHead, abi.headThreads(h));
    h.heads = 0;
    try std.testing.expectError(error.BadHead, abi.headThreads(h));
    h.heads = 3;
    h.row_stride = 2048;
    try std.testing.expectError(error.BadHead, abi.headThreads(h));
}
