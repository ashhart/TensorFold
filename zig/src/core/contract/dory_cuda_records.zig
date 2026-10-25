//! Dory nvcc 13.0.88 receipts cover sm121; other SMs and later compilers remain pending.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_l2_rows", .symbol = "tf_l2_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 64, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "norms tf_slerp_rows", .symbol = "tf_slerp_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 113, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows_l2 at dim 256", .symbol = "tf_qk_rows_l2", .image = "qk_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 30, .static = 64, .bound = 512 } }, .later_pending = true },
    .{ .launch = .{ .name = "act_rows tf_gate_l2_rows", .symbol = "tf_gate_l2_rows", .image = "act_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 109, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "act_rows tf_swiglu_l2_rows", .symbol = "tf_swiglu_l2_rows", .image = "act_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 121, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_1", .symbol = "tf_dense_rows_scaled_1", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 38, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_4", .symbol = "tf_dense_rows_scaled_4", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 45, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_16", .symbol = "tf_dense_rows_scaled_16", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 98, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_32", .symbol = "tf_dense_rows_scaled_32", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 180, .static = 0, .bound = 256 } }, .later_pending = true },
};

test "Dory sm121 receipts do not qualify other SMs or compilers" {
    const std = @import("std");
    const reg = @import("registry.zig");
    for (cuda) |r| {
        for (r.usage[0..4]) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        try std.testing.expect(r.usage[4].regs != 0);
        try std.testing.expectEqual(r.usage, reg.usageFor(r, .{ .major = 13, .minor = 0 }));
        try std.testing.expectEqual(reg.none, reg.usageFor(r, .{ .major = 13, .minor = 2 }));
    }
}
