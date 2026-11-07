//! Ripple's DFlash2 draft attention (64 KiB dynamic shared) and top-32 (1,024 threads) on CUDA: their served launches.
const registry = @import("registry.zig");
const limits = @import("limits.zig");

const zero: limits.Usage = .{ .regs = 0, .static = 0, .bound = 0 };

/// Usage zero (no nvcc 13.0 record yet); nvcc 13.3's five-SM receipts are in measurements/pending_133.zig.
pub const cuda = [_]registry.CudaRecord{
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_attn_part", .symbol = "tf_rcd_attn_part", .image = "ricursive_dflash2_attn", .threads = 256, .dynamic = 65536, .opt_in = true, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, zero } },
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_attn_merge", .symbol = "tf_rcd_attn_merge", .image = "ricursive_dflash2_attn", .threads = 256, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, zero } },
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_top32_part", .symbol = "tf_rcd_top32_part", .image = "ricursive_dflash2_attn", .threads = 1024, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, zero } },
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_top32_merge", .symbol = "tf_rcd_top32_merge", .image = "ricursive_dflash2_attn", .threads = 1024, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, zero } },
};

test "ricursive DFlash2 attention CUDA: each launch is in the live registry once, at its kernel's threads" {
    const std = @import("std");
    const t = std.testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(u32, if (std.mem.indexOf(u8, r.launch.name, "top32") != null) 1024 else 256), r.launch.threads);
    }
}
