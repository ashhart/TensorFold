//! Qwen3-ASR's CUDA launches: the tower's conv taps and flatten (vit.cu), and full attention at head dim 64.
const std = @import("std");
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "vit tf_vit_conv_cols (Qwen3-ASR)", .symbol = "tf_vit_conv_cols", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit tf_vit_conv_flatten (Qwen3-ASR)", .symbol = "tf_vit_conv_flatten", .image = "vit", .threads = 256 }, .usage = .{ .{ .regs = 31, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 36, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit vision_kernel<64, 8> (Qwen3-ASR)", .symbol = "_ZN20tf_prefill_attention13vision_kernelILi64ELi8EEEvPK13__nv_bfloat16PS1_iif", .image = "prefill_attention", .threads = 256, .dynamic = 36864 }, .usage = .{ .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 } } },
};

/// nvcc 13.2's: the attention at 64 takes 8 more registers on sm_120 and sm_121; the copies compile the same.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "vit vision_kernel<64, 8> (Qwen3-ASR)", .usage = .{ .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 } } },
};

test "Qwen3-ASR CUDA: every launch is in the live registry and fits every SM" {
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
