//! Exact K32 launch geometry is registered; unmeasured compiler resources fail qualification until supplied.
const std = @import("std");
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nvfp4_experts_tile tf_nvfp4_tile_swiglu", .symbol = "tf_nvfp4_tile_swiglu", .image = "nvfp4_experts_tile", .threads = 128, .dynamic = 34816, .min_sm = 80 }, .usage = @import("registry.zig").none },
    .{ .launch = .{ .name = "nvfp4_experts_tile tf_nvfp4_tile_down_f32", .symbol = "tf_nvfp4_tile_down_f32", .image = "nvfp4_experts_tile", .threads = 128, .dynamic = 19200, .min_sm = 80 }, .usage = @import("registry.zig").none },
};
test "K32 expert tile records require independent compiler measurements on every served SM" {
    for (records) |record| try std.testing.expectEqual(@as(usize, 5), try registry.checkCuda(std.testing.allocator, &.{record}, true));
}
