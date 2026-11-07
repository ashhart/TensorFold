//! Ripple's DFlash2 drafter glue on CUDA (ricursive_dflash2.cu), 256 threads each; its embed is tf_rc_embed's record.
const registry = @import("registry.zig");

/// Usage zero (no nvcc 13.0 record yet); nvcc 13.3's five-SM receipts are in measurements/pending_133.zig.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ricursive_dflash2 tf_rcd_add_norm", .symbol = "tf_rcd_add_norm", .image = "ricursive_dflash2", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "ricursive_dflash2 tf_rcd_prep", .symbol = "tf_rcd_prep", .image = "ricursive_dflash2", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "ricursive_dflash2 tf_rcd_conv", .symbol = "tf_rcd_conv", .image = "ricursive_dflash2", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "ricursive_dflash2 tf_rcd_silu", .symbol = "tf_rcd_silu", .image = "ricursive_dflash2", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
};

test "ricursive DFlash2 CUDA: each launch is in the live registry once, at the kernels' 256 threads" {
    const std = @import("std");
    const t = std.testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(u32, 256), r.launch.threads);
    }
}
