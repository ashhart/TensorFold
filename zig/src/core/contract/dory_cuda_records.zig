//! Dory nvcc 13.0.88 receipts cover all five SMs; later compiler selection remains independent.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_l2_rows", .symbol = "tf_l2_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 48, .static = 32, .bound = 256 }, .{ .regs = 64, .static = 32, .bound = 256 }, .{ .regs = 64, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "norms tf_slerp_rows", .symbol = "tf_slerp_rows", .image = "norms", .threads = 256 }, .usage = .{ .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 }, .{ .regs = 113, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows_l2 at dim 256", .symbol = "tf_qk_rows_l2", .image = "qk_rows", .threads = 256 }, .usage = .{ .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 }, .{ .regs = 30, .static = 64, .bound = 512 } }, .later_pending = true },
    .{ .launch = .{ .name = "act_rows tf_gate_l2_rows", .symbol = "tf_gate_l2_rows", .image = "act_rows", .threads = 256 }, .usage = .{ .{ .regs = 107, .static = 32, .bound = 256 }, .{ .regs = 109, .static = 32, .bound = 256 }, .{ .regs = 109, .static = 32, .bound = 256 }, .{ .regs = 109, .static = 32, .bound = 256 }, .{ .regs = 109, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "act_rows tf_swiglu_l2_rows", .symbol = "tf_swiglu_l2_rows", .image = "act_rows", .threads = 256 }, .usage = .{ .{ .regs = 109, .static = 32, .bound = 256 }, .{ .regs = 118, .static = 32, .bound = 256 }, .{ .regs = 118, .static = 32, .bound = 256 }, .{ .regs = 121, .static = 32, .bound = 256 }, .{ .regs = 121, .static = 32, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_1", .symbol = "tf_dense_rows_scaled_1", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_4", .symbol = "tf_dense_rows_scaled_4", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 45, .static = 0, .bound = 256 }, .{ .regs = 45, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_16", .symbol = "tf_dense_rows_scaled_16", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 79, .static = 0, .bound = 256 }, .{ .regs = 79, .static = 0, .bound = 256 }, .{ .regs = 79, .static = 0, .bound = 256 }, .{ .regs = 98, .static = 0, .bound = 256 }, .{ .regs = 98, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_32", .symbol = "tf_dense_rows_scaled_32", .image = "dense_rows", .threads = 256 }, .usage = .{ .{ .regs = 127, .static = 0, .bound = 256 }, .{ .regs = 127, .static = 0, .bound = 256 }, .{ .regs = 127, .static = 0, .bound = 256 }, .{ .regs = 180, .static = 0, .bound = 256 }, .{ .regs = 180, .static = 0, .bound = 256 } }, .later_pending = true },
};

test "Dory five-SM receipts do not qualify later compilers" {
    const std = @import("std");
    const reg = @import("registry.zig");
    for (cuda) |r| {
        for (r.usage) |u| try std.testing.expect(u.regs != 0);
        try std.testing.expectEqual(@as(usize, 0), try reg.checkCuda(std.testing.allocator, &.{r}, true));
        try std.testing.expectEqual(r.usage, reg.usageFor(r, .{ .major = 13, .minor = 0 }));
        try std.testing.expectEqual(reg.none, reg.usageFor(r, .{ .major = 13, .minor = 2 }));
    }
}
