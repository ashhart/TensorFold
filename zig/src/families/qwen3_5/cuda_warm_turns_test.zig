//! Qwen3.5 CUDA's warm turns: the dense family and its MoE wrapper declare the wait and a usable end.
const std = @import("std");
const native = @import("cuda_native.zig");
const moe = @import("cuda_native_moe.zig");

test "Qwen warm turns: the dense family and its MoE wrapper share the wait and keep only usable ends" {
    try std.testing.expectEqual(@as(u64, 150 * std.time.ns_per_ms), native.warm_quiet);
    try std.testing.expect(native.warm_usable);
    try std.testing.expectEqual(@as(u64, 150 * std.time.ns_per_ms), moe.warm_quiet);
    try std.testing.expect(moe.warm_usable);
}
