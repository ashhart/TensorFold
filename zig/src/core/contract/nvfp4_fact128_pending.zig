//! Exact FACT128 launch geometry is registered; unmeasured compiler resources fail qualification until supplied.
const std = @import("std");
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact128_swiglu", .symbol = "tf_nvfp4_fact128_swiglu", .image = "nvfp4_experts", .threads = 256, .dynamic = 51200 }, .usage = registry.none },
    .{ .launch = .{ .name = "nvfp4_experts tf_nvfp4_fact128_down_f32", .symbol = "tf_nvfp4_fact128_down_f32", .image = "nvfp4_experts", .threads = 256, .dynamic = 41984 }, .usage = registry.none },
};

test "FACT128 records require independent compiler measurements on every served SM" {
    for (records) |record| try std.testing.expectEqual(@as(usize, 5), try registry.checkCuda(std.testing.allocator, &.{record}, true));
}
