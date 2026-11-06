//! Ripple's CUDA attention prep: one stream's rows, or each row at its own position and stream; a warp a head.
const registry = @import("registry.zig");

/// Usage pending: no nvcc has compiled the image yet, so every value is zero until the Spark's receipts land.
pub const cuda = [_]registry.CudaRecord{
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive_prep tf_rc_prep", .symbol = "tf_rc_prep", .image = "ricursive_prep", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "ricursive_prep tf_rc_prep_rows", .symbol = "tf_rc_prep_rows", .image = "ricursive_prep", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
};

test "ricursive prep CUDA: its launch is in the live registry once, at the kernel's 256 threads" {
    const std = @import("std");
    const t = std.testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(u32, 256), r.launch.threads);
    }
}
