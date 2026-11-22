//! Scalar HC activations reserve 256 threads; nvcc 13.0 CUDA resources are measured only on sm_121.
const registry = @import("registry.zig");
const receipts = @import("measurements/qwen4_130_sm121.zig");
const limits = @import("limits.zig");
const suffixes = [_][]const u8{ "", "_bad_nan", "_bad_repeat", "_bad_tail" };
pub const cuda = blk: {
    var records: [suffixes.len * 2]registry.CudaRecord = undefined;
    for (suffixes, 0..) |suffix, i| records[i] = .{ .launch = .{ .name = "qwen4_hc qwen4_hc_activation" ++ suffix, .symbol = "qwen4_hc_activation" ++ suffix, .image = "qwen4_hc", .threads = 256 }, .usage = receipts.usage("qwen4_hc qwen4_hc_activation" ++ suffix), .later_pending = true };
    for (suffixes, 0..) |suffix, i| records[i + suffixes.len] = .{ .launch = .{ .name = "qwen4_hc qwen4_hc_pointwise" ++ suffix, .symbol = "qwen4_hc_pointwise" ++ suffix, .image = "qwen4_hc", .threads = 256 }, .usage = receipts.usage("qwen4_hc qwen4_hc_pointwise" ++ suffix), .later_pending = true };
    break :blk records;
};
pub const metal = blk: {
    var records: [suffixes.len * 2]limits.MetalLaunch = undefined;
    for (suffixes, 0..) |suffix, i| records[i] = .{ .name = "hc-activation qwen4_hc_activation" ++ suffix, .threads = 256, .tg_memory = 0, .reserved = 256 };
    for (suffixes, 0..) |suffix, i| records[i + suffixes.len] = .{ .name = "hc-pointwise qwen4_hc_pointwise" ++ suffix, .threads = 256, .tg_memory = 0, .reserved = 256 };
    break :blk records;
};
