//! The MXFP4 experts' CUDA launches (experts_mxfp4.cu): the SiTU up's and the two downs' decode and prompt rows.
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "experts_mxfp4 tf_mxfp4_experts_down_f32", .symbol = "tf_mxfp4_experts_down_f32", .image = "experts_mxfp4", .threads = 128 }, .usage = .{ .{ .regs = 88, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_mxfp4 tf_mxfp4_experts_down_bf16", .symbol = "tf_mxfp4_experts_down_bf16", .image = "experts_mxfp4", .threads = 128 }, .usage = .{ .{ .regs = 88, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 }, .{ .regs = 89, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 }, .{ .regs = 96, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_mxfp4 tf_mxfp4_experts_prompt_down_f32", .symbol = "tf_mxfp4_experts_prompt_down_f32", .image = "experts_mxfp4", .threads = 128 }, .usage = .{ .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 205, .static = 0, .bound = 128 }, .{ .regs = 205, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_mxfp4 tf_mxfp4_experts_prompt_down_bf16", .symbol = "tf_mxfp4_experts_prompt_down_bf16", .image = "experts_mxfp4", .threads = 128 }, .usage = .{ .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 180, .static = 0, .bound = 128 }, .{ .regs = 205, .static = 0, .bound = 128 }, .{ .regs = 205, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_mxfp4 tf_mxfp4_experts_up_situ", .symbol = "tf_mxfp4_experts_up_situ", .image = "experts_mxfp4", .threads = 128 }, .usage = .{ .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 128, .static = 0, .bound = 128 }, .{ .regs = 134, .static = 0, .bound = 128 }, .{ .regs = 134, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_mxfp4 tf_mxfp4_experts_prompt_up_situ", .symbol = "tf_mxfp4_experts_prompt_up_situ", .image = "experts_mxfp4", .threads = 128 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 }, .{ .regs = 198, .static = 0, .bound = 128 } } },
};

test "MXFP4 experts CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
