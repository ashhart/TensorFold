//! Weight-only projections reserve 128 threads on Metal; nvcc 13.0 CUDA resources are measured only on sm_121.
const std = @import("std");
const registry = @import("registry.zig");
const receipts = @import("measurements/qwen4_130_sm121.zig");
const limits = @import("limits.zig");
const suffixes = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail", "_bad_sums" };
pub const cuda = blk: {
    var records: [suffixes.len + 1]registry.CudaRecord = undefined;
    for (suffixes, 0..) |suffix, i| {
        const symbol = "tf_qwen4_projection" ++ suffix;
        records[i] = .{ .launch = .{ .name = "qwen4_projection " ++ symbol, .symbol = symbol, .image = "qwen4_projection", .threads = 128 }, .usage = receipts.usage("qwen4_projection " ++ symbol), .later_pending = true };
    }
    records[suffixes.len] = .{ .launch = .{ .name = "qwen4_gdn qwen4_gdn_chain", .symbol = "qwen4_gdn_chain", .image = "qwen4_gdn", .threads = 1024 }, .usage = receipts.usage("qwen4_gdn qwen4_gdn_chain"), .later_pending = true };
    break :blk records;
};
pub const metal = blk: {
    var records: [suffixes.len]limits.MetalLaunch = undefined;
    for (suffixes, 0..) |suffix, i| records[i] = .{ .name = "weight-only tf_weight_only_projection" ++ suffix, .threads = 128, .tg_memory = 0, .reserved = 128 };
    break :blk records;
};
test "weight-only limits: unmeasured CUDA compiles never count as a pass" {
    for (cuda) |record| {
        try std.testing.expectEqual(limits.sms.len - 1, try registry.checkCuda(std.testing.allocator, &.{record}, true));
        var missing = record;
        missing.usage = registry.none;
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{missing}, true));
    }
    try std.testing.expectEqual(@as(usize, 0), try registry.checkMetal(std.testing.allocator, &metal, false));
}
