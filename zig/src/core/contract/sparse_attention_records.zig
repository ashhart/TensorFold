//! sparse_attention.cu's launch: rows over listed keys, a block a (row, key head), every SM (nvcc 13.0 and 13.2 alike).
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;

const use: @import("limits.zig").Usage = .{ .regs = 168, .static = 0, .bound = 128 };

pub const cuda = [_]R{
    .{ .launch = .{ .name = "sparse attention tf_sparse_attention (up to 16 query heads a key head)", .symbol = "tf_sparse_attention", .image = "sparse_attention", .threads = 128, .dynamic = 65536, .opt_in = true }, .usage = .{ use, use, use, use, use } },
};

test "sparse attention records: the launch fits every SM by its measured record" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
