//! dots.mocr's CUDA glue: tf_dots_qkv_rope's usage from nvcc 13.0 images for every SM.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

pub const cuda = [_]R{
    .{ .launch = .{ .name = "dots_mocr tf_dots_qkv_rope", .image = "dots_mocr", .symbol = "tf_dots_qkv_rope", .threads = 64 }, .usage = .{ .{ .regs = 28, .static = 0, .bound = 64 }, .{ .regs = 24, .static = 0, .bound = 64 }, .{ .regs = 24, .static = 0, .bound = 64 }, .{ .regs = 25, .static = 0, .bound = 64 }, .{ .regs = 25, .static = 0, .bound = 64 } } },
};

test "dots.mocr records: every launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
