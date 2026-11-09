//! dots.mocr's CUDA glue and the dense engine's other launches: usage from nvcc 13.0 images for every SM.
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

pub const cuda = [_]R{
    .{ .launch = .{ .name = "dots_mocr tf_dots_qkv_rope", .image = "dots_mocr", .symbol = "tf_dots_qkv_rope", .threads = 64 }, .usage = .{ .{ .regs = 28, .static = 0, .bound = 64 }, .{ .regs = 24, .static = 0, .bound = 64 }, .{ .regs = 24, .static = 0, .bound = 64 }, .{ .regs = 25, .static = 0, .bound = 64 }, .{ .regs = 25, .static = 0, .bound = 64 } } },
    .{ .launch = .{ .name = "embed_rows tf_embed_rows_bf16", .image = "embed_rows", .symbol = "tf_embed_rows_bf16", .threads = 256 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "lfm2 tf_lfm2_silu_mul", .image = "lfm2", .symbol = "tf_lfm2_silu_mul", .threads = 256 }, .usage = .{ .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 } } },
};

test "dots.mocr records: every launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
