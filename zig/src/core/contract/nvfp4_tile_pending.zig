//! Exact K32 launch geometry and CUDA 13.0.88 resources; later compiler receipts remain independent.
const std = @import("std");
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts_tile tf_nvfp4_tile_swiglu", .symbol = "tf_nvfp4_tile_swiglu", .image = "nvfp4_experts_tile", .threads = 128, .dynamic = 34816, .min_sm = 80 }, .usage = .{
        .{ .regs = 255, .static = 0, .bound = 128 },
        .{ .regs = 255, .static = 0, .bound = 128 },
        .{ .regs = 255, .static = 0, .bound = 128 },
        .{ .regs = 255, .static = 0, .bound = 128 },
        .{ .regs = 255, .static = 0, .bound = 128 },
    }, .later_pending = true },
    .{ .launch = .{ .name = "nvfp4_experts_tile tf_nvfp4_tile_down_f32", .symbol = "tf_nvfp4_tile_down_f32", .image = "nvfp4_experts_tile", .threads = 128, .dynamic = 19200, .min_sm = 80 }, .usage = .{
        .{ .regs = 123, .static = 0, .bound = 128 },
        .{ .regs = 123, .static = 0, .bound = 128 },
        .{ .regs = 128, .static = 0, .bound = 128 },
        .{ .regs = 128, .static = 0, .bound = 128 },
        .{ .regs = 128, .static = 0, .bound = 128 },
    }, .later_pending = true },
};
test "K32 expert tile records require independent compiler measurements on every served SM" {
    for (records) |record| {
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{record}, true));
        var missing = record;
        missing.usage = registry.none;
        try std.testing.expectEqual(@as(usize, 5), try registry.checkCuda(std.testing.allocator, &.{missing}, true));
        try std.testing.expectEqual(registry.none, registry.usageFor(record, .{ .major = 13, .minor = 2 }));
    }
}
