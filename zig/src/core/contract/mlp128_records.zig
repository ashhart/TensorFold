//! Launch limits for the dense 27B MLP column tile.
const reg = @import("registry.zig");
pub const records = [_]reg.CudaRecord{
    .{ .launch = .{ .name = "qmm_group MLP128", .symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi128ELi1ELi8ELi4ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii", .image = "qmm_group", .threads = 256, .dynamic = 26880, .min_sm = 120 }, .usage = .{ .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 }, .{ .regs = 0, .static = 0 }, .{ .regs = 54, .static = 0, .bound = 256 }, .{ .regs = 54, .static = 0, .bound = 256 } }, .later_pending = true },
};

test "MLP128 compiler receipts remain independently qualified" {
    const std = @import("std");
    const row = records[0];
    try std.testing.expectEqualDeep(row.usage, reg.usageFor(row, .{ .major = 13, .minor = 0 }));
    try std.testing.expectEqualDeep(row.usage, reg.usageFor(row, .{ .major = 13, .minor = 3 }));
    for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 4 }, .{ .major = 14, .minor = 0 } }) |compiler|
        try std.testing.expectEqualDeep(reg.none, reg.usageFor(row, compiler));
}
