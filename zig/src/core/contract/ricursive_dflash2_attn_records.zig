//! Ripple's DFlash2 draft attention (64 KiB dynamic shared) and top-32 (1,024 threads) on CUDA: their served launches.
const registry = @import("registry.zig");
const limits = @import("limits.zig");

const zero: limits.Usage = .{ .regs = 0, .static = 0, .bound = 0 };

/// sm_121's usage is nvcc 13.3's (kernel-contract-limits, -Dsm=121); sm_80, 86, 89 and 120 are zero until measured.
pub const cuda = [_]registry.CudaRecord{
    // TODO(spark): sm_80, 86, 89 and 120 from a five-SM kernel-contract-limits run
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_attn_part", .symbol = "tf_rcd_attn_part", .image = "ricursive_dflash2_attn", .threads = 256, .dynamic = 65536, .opt_in = true, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, .{ .regs = 60, .static = 0, .bound = 256 } } },
    // TODO(spark): sm_80, 86, 89 and 120 from a five-SM kernel-contract-limits run
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_attn_merge", .symbol = "tf_rcd_attn_merge", .image = "ricursive_dflash2_attn", .threads = 256, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, .{ .regs = 40, .static = 0, .bound = 256 } } },
    // TODO(spark): sm_80, 86, 89 and 120 from a five-SM kernel-contract-limits run
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_top32_part", .symbol = "tf_rcd_top32_part", .image = "ricursive_dflash2_attn", .threads = 1024, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, .{ .regs = 23, .static = 16384, .bound = 1024 } } },
    // TODO(spark): sm_80, 86, 89 and 120 from a five-SM kernel-contract-limits run
    .{ .launch = .{ .name = "ricursive_dflash2_attn tf_rcd_top32_merge", .symbol = "tf_rcd_top32_merge", .image = "ricursive_dflash2_attn", .threads = 1024, .min_sm = 80 }, .usage = .{ zero, zero, zero, zero, .{ .regs = 23, .static = 16384, .bound = 1024 } } },
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
