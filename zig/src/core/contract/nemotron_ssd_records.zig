//! Nemotron's SSD steps on gemm_nax's f32 kernels, as prefill_mixers.mamba launches them through prefill_launch.gemm.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
const gemm_fn = "custom_kernel_tf_gemm_nax_f32_{s}_64_128_256_2_4_float_float_int32_t_float";

/// y = sur x dtx and the state dd^T x bh^T by whole blocks; tensor-op kernels whose source reserves 256 threads.
pub const records = blk: {
    var out: [5]M = undefined;
    for (&out, [_][]const u8{ "n_n_n_n_n", "n_n_t_n_n", "n_n_t_n_t", "t_t_t_t_n", "t_t_t_t_t" }) |*r, tags| r.* = .{ .name = "nemotron ssd gemm " ++ std.fmt.comptimePrint(gemm_fn, .{tags}), .threads = 256, .tg_memory = 0, .reserved = 256 };
    break :blk out;
};

test "Nemotron's SSD gemm launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
