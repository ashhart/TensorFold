//! Hy4-flash FP8 Engram CUDA launch limits across SM 80, 86, 89, 120, and 121 (nvcc 13.0).
const registry = @import("registry.zig");

/// nvcc 13.0's usage for hy4_engram_fp8.fatbin across the five recorded SMs.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "hy4_engram_fp8 hy4_engram_fp8_embed", .symbol = "hy4_engram_fp8_embed", .image = "hy4_engram_fp8", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "hy4_engram_fp8 hy4_engram_fp8_bias", .symbol = "hy4_engram_fp8_bias", .image = "hy4_engram_fp8", .threads = 256 }, .usage = .{ .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "hy4_engram_fp8 hy4_engram_fp8_gate", .symbol = "hy4_engram_fp8_gate", .image = "hy4_engram_fp8", .threads = 256 }, .usage = .{ .{ .regs = 62, .static = 340, .bound = 256 }, .{ .regs = 62, .static = 340, .bound = 256 }, .{ .regs = 62, .static = 340, .bound = 256 }, .{ .regs = 64, .static = 340, .bound = 256 }, .{ .regs = 64, .static = 340, .bound = 256 } } },
};

test "Hy4 FP8 Engram CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
