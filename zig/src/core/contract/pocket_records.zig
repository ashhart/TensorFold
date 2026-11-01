//! Nemotron's pocket launches (nemotron_ops.cu): relu^2 up into act rows, down into the shared expert's slot.
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nemotron_ops tf_pocket_in", .symbol = "tf_pocket_in", .image = "nemotron_ops", .threads = 128 }, .usage = .{ .{ .regs = 31, .static = 0, .bound = 128 }, .{ .regs = 38, .static = 0, .bound = 128 }, .{ .regs = 38, .static = 0, .bound = 128 }, .{ .regs = 38, .static = 0, .bound = 128 }, .{ .regs = 38, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "nemotron_ops tf_pocket_in_rows", .symbol = "tf_pocket_in_rows", .image = "nemotron_ops", .threads = 256 }, .usage = .{ .{ .regs = 126, .static = 0, .bound = 256 }, .{ .regs = 98, .static = 0, .bound = 256 }, .{ .regs = 98, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemotron_ops tf_pocket_out", .symbol = "tf_pocket_out", .image = "nemotron_ops", .threads = 128 }, .usage = .{ .{ .regs = 48, .static = 256, .bound = 128 }, .{ .regs = 63, .static = 256, .bound = 128 }, .{ .regs = 63, .static = 256, .bound = 128 }, .{ .regs = 64, .static = 256, .bound = 128 }, .{ .regs = 64, .static = 256, .bound = 128 } } },
    .{ .launch = .{ .name = "nemotron_ops tf_pocket_out_rows", .symbol = "tf_pocket_out_rows", .image = "nemotron_ops", .threads = 128 }, .usage = .{ .{ .regs = 40, .static = 4096, .bound = 128 }, .{ .regs = 40, .static = 4096, .bound = 128 }, .{ .regs = 40, .static = 4096, .bound = 128 }, .{ .regs = 48, .static = 4096, .bound = 128 }, .{ .regs = 48, .static = 4096, .bound = 128 } } },
};

/// nvcc 13.2's tf_pocket_in on sm_120 and sm_121 (39 registers where 13.0 takes 38); the rest compile as 13.0's.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "nemotron_ops tf_pocket_in", .usage = .{ .{ .regs = 31, .static = 0, .bound = 128 }, .{ .regs = 38, .static = 0, .bound = 128 }, .{ .regs = 38, .static = 0, .bound = 128 }, .{ .regs = 39, .static = 0, .bound = 128 }, .{ .regs = 39, .static = 0, .bound = 128 } } },
};

test "pocket CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
