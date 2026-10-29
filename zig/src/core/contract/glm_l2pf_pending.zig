//! GLM's L2 prefetch (TF_GLM_L2PF) from DeepSeek V4.1's l2pf and l2pace sources: nvcc 13.0 resources.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "l2pf _ZN6tfl2pf15segments_kernelILb1EEEvPKlij", .symbol = "_ZN6tfl2pf15segments_kernelILb1EEEvPKlij", .image = "l2pf", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 28, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "l2pace _ZN12dsv41_l2pace12paced_kernelEPKlijmm", .symbol = "_ZN12dsv41_l2pace12paced_kernelEPKlijmm", .image = "l2pace", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 32 }, .{ .regs = 23, .static = 0, .bound = 32 }, .{ .regs = 23, .static = 0, .bound = 32 }, .{ .regs = 20, .static = 0, .bound = 32 }, .{ .regs = 20, .static = 0, .bound = 32 } }, .later_pending = true },
};
test "GLM L2 prefetch: measured 13.0 launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (records) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
