//! The bf16 experts' CUDA launches (experts_bf16.cu): decode and prompt rows by epilogue, and the repack.
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_up", .symbol = "tf_bf16_experts_up", .image = "experts_bf16", .threads = 128 }, .usage = .{ .{ .regs = 76, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_down_f32", .symbol = "tf_bf16_experts_down_f32", .image = "experts_bf16", .threads = 128 }, .usage = .{ .{ .regs = 76, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_down_bf16", .symbol = "tf_bf16_experts_down_bf16", .image = "experts_bf16", .threads = 128 }, .usage = .{ .{ .regs = 76, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 }, .{ .regs = 72, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_prompt_up", .symbol = "tf_bf16_experts_prompt_up", .image = "experts_bf16", .threads = 128 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 162, .static = 0, .bound = 128 }, .{ .regs = 162, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_prompt_down_f32", .symbol = "tf_bf16_experts_prompt_down_f32", .image = "experts_bf16", .threads = 128 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 162, .static = 0, .bound = 128 }, .{ .regs = 162, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_prompt_down_bf16", .symbol = "tf_bf16_experts_prompt_down_bf16", .image = "experts_bf16", .threads = 128 }, .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 162, .static = 0, .bound = 128 }, .{ .regs = 162, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "experts_bf16 tf_bf16_experts_pack", .symbol = "tf_bf16_experts_pack", .image = "experts_bf16", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 } } },
};

/// nvcc 13.2's prompt forms on sm_120 and sm_121 (164 registers where 13.0 takes 162); the rest compile as 13.0's.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "experts_bf16 tf_bf16_experts_prompt_up", .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 } } },
    .{ .name = "experts_bf16 tf_bf16_experts_prompt_down_f32", .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 } } },
    .{ .name = "experts_bf16 tf_bf16_experts_prompt_down_bf16", .usage = .{ .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 168, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 }, .{ .regs = 164, .static = 0, .bound = 128 } } },
};

/// nvcc 13.3's decode forms on sm_120 and sm_121 (70 registers where 13.0 and 13.2 take 72).
pub const cuda_133 = [_]registry.Override{
    .{ .name = "experts_bf16 tf_bf16_experts_up", .usage = .{ .{ .regs = 76, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 70, .static = 0, .bound = 128 }, .{ .regs = 70, .static = 0, .bound = 128 } } },
    .{ .name = "experts_bf16 tf_bf16_experts_down_f32", .usage = .{ .{ .regs = 76, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 70, .static = 0, .bound = 128 }, .{ .regs = 70, .static = 0, .bound = 128 } } },
    .{ .name = "experts_bf16 tf_bf16_experts_down_bf16", .usage = .{ .{ .regs = 76, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 84, .static = 0, .bound = 128 }, .{ .regs = 70, .static = 0, .bound = 128 }, .{ .regs = 70, .static = 0, .bound = 128 } } },
};

test "bf16 experts CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
