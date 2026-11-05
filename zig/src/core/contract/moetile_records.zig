//! Shared-down launch geometry and nvcc 13.0 receipts; later compiler receipts remain pending.
const std = @import("std");
const registry = @import("registry.zig");
const limits = @import("limits.zig");

fn usage(regs: [5]u32) [5]limits.Usage {
    var out: [5]limits.Usage = undefined;
    for (&out, regs) |*u, count| u.* = .{ .regs = count, .static = 0, .bound = 128 };
    return out;
}
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "qmmf shared down half16 stride", .symbol = "tf_qmmf_shared16_half", .image = "qmmf", .threads = 128, .dynamic = 24576, .min_sm = 80 }, .usage = usage(.{ 64, 64, 64, 40, 40 }), .later_pending = true },
    .{ .launch = .{ .name = "qmmf shared down 16 stride", .symbol = "tf_qmmf_shared16", .image = "qmmf", .threads = 128, .dynamic = 40960, .min_sm = 80 }, .usage = usage(.{ 79, 79, 79, 38, 38 }), .later_pending = true },
    .{ .launch = .{ .name = "qmmf shared down 32 stride", .symbol = "tf_qmmf_shared32", .image = "qmmf", .threads = 128, .dynamic = 49152, .min_sm = 80 }, .usage = usage(.{ 93, 93, 93, 56, 56 }), .later_pending = true },
    .{ .launch = .{ .name = "qmmf shared down 64 stride", .symbol = "tf_qmmf_shared64", .image = "qmmf", .threads = 128, .dynamic = 65536, .opt_in = true, .min_sm = 80 }, .usage = usage(.{ 118, 118, 118, 72, 72 }), .later_pending = true },
};

test "shared down records hold nvcc 13.0 resources and keep later compilers pending" {
    try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &records, true));
    for (records) |record| {
        try std.testing.expectEqual(record.usage, registry.usageFor(record, .{ .major = 13, .minor = 0 }));
        for (record.usage) |u| {
            try std.testing.expect(u.regs > 0);
            try std.testing.expectEqual(@as(u32, 0), u.static);
            try std.testing.expectEqual(@as(u32, 128), u.bound);
        }
        for ([_]limits.Nvcc{ .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |release|
            try std.testing.expectEqual(registry.none, registry.usageFor(record, release));
    }
}
