//! GigaChat 4's CUDA launches: GatedNorm (gigachat4/gated_norm.cu), one row a block of 256 threads.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (nvcc -cubin per SM at the image's flags), the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "gigachat4 norm tf_gc4_gated_norm", .symbol = "tf_gc4_gated_norm", .image = "gigachat4_norm", .threads = 256, .dynamic = 10240 }, .usage = .{ .{ .regs = 38, .static = 96, .bound = 256 }, .{ .regs = 38, .static = 96, .bound = 256 }, .{ .regs = 38, .static = 96, .bound = 256 }, .{ .regs = 38, .static = 96, .bound = 256 }, .{ .regs = 38, .static = 96, .bound = 256 } } },
};

test "GigaChat 4 CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
