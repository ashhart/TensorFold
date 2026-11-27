//! ERNIE 5.5's CUDA launches (ernie5_5/head.cu): MultiMax on the head's fp32 logits.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (nvcc -cubin per SM at the image's flags), the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "ernie5_5 head tf_ernie55_multimax", .symbol = "tf_ernie55_multimax", .image = "ernie5_5_head", .threads = 256 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
};

test "ERNIE 5.5 CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
