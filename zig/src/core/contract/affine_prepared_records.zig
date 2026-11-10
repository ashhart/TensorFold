//! CUDA 13.5 SM120/121 compile receipts for the exact affine-code preparation and products.
const std = @import("std");
const r = @import("registry.zig");
const limits = @import("limits.zig");

fn entry(comptime symbol: []const u8, comptime regs: u32, comptime shared: u32) r.CudaRecord {
    const absent: limits.Usage = .{ .regs = 0, .static = 0 };
    const used: limits.Usage = .{ .regs = regs, .static = shared, .bound = 256 };
    return .{
        .launch = .{ .name = "affine prepared " ++ symbol, .symbol = symbol, .image = "affine_fp8_prepared", .threads = 256, .min_sm = 120 },
        .usage = .{ absent, absent, absent, used, used },
        .first_nvcc = .{ .major = 13, .minor = 5 },
    };
}
pub const records = [_]r.CudaRecord{
    entry("tf_affine_fp8_prepare", 21, 0),
    entry("tf_affine_fp8_prepared", 96, 26624),
    entry("tf_affine_fp8_prepared_f32", 96, 26624),
};

test "FP8 prompt affine prepared receipts qualify only measured compilers and Blackwell profiles" {
    for (records) |record| {
        try std.testing.expectEqual(r.none, r.usageFor(record, null));
        try std.testing.expectEqual(r.none, r.usageFor(record, .{ .major = 13, .minor = 3 }));
        try std.testing.expectEqual(record.usage, r.usageFor(record, .{ .major = 13, .minor = 5 }));
        try std.testing.expect(record.usage[3].regs > 0);
        try std.testing.expectEqual(record.usage[3], record.usage[4]);
    }
}
