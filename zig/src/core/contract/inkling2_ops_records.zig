//! Inkling 2's inkling2_ops, inkling2_rows images: every SM's compile records (nvcc 13.0, 13.2, 13.3).
const registry = @import("registry.zig");

/// nvcc 13.0's usage (nvcc -cubin per SM at the images' flags), the compiler of the release fatbins; 13.2 pending.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_embed", .symbol = "tf_ink2_embed", .image = "inkling2_ops", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 21, .static = 128, .bound = 256 }, .{ .regs = 19, .static = 128, .bound = 256 }, .{ .regs = 19, .static = 128, .bound = 256 }, .{ .regs = 22, .static = 128, .bound = 256 }, .{ .regs = 22, .static = 128, .bound = 256 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_mix", .symbol = "tf_ink2_mix", .image = "inkling2_ops", .threads = 512, .min_sm = 80 }, .usage = .{ .{ .regs = 64, .static = 45712, .bound = 512 }, .{ .regs = 62, .static = 45712, .bound = 512 }, .{ .regs = 62, .static = 45712, .bound = 512 }, .{ .regs = 72, .static = 45712, .bound = 512 }, .{ .regs = 72, .static = 45712, .bound = 512 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_mix_rows", .symbol = "tf_ink2_mix_rows", .image = "inkling2_ops", .threads = 512, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 656, .bound = 512 }, .{ .regs = 40, .static = 656, .bound = 512 }, .{ .regs = 40, .static = 656, .bound = 512 }, .{ .regs = 40, .static = 656, .bound = 512 }, .{ .regs = 40, .static = 656, .bound = 512 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_rowdot", .symbol = "tf_ink2_rowdot", .image = "inkling2_ops", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 }, .{ .regs = 38, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_topk", .symbol = "tf_ink2_topk", .image = "inkling2_ops", .threads = 512, .min_sm = 80 }, .usage = .{ .{ .regs = 24, .static = 384, .bound = 512 }, .{ .regs = 23, .static = 384, .bound = 512 }, .{ .regs = 23, .static = 384, .bound = 512 }, .{ .regs = 36, .static = 384, .bound = 512 }, .{ .regs = 36, .static = 384, .bound = 512 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_conv", .symbol = "tf_ink2_conv", .image = "inkling2_ops", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 40, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 }, .{ .regs = 48, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_conv_rewind", .symbol = "tf_ink2_conv_rewind", .image = "inkling2_ops", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 128 }, .{ .regs = 28, .static = 0, .bound = 128 }, .{ .regs = 28, .static = 0, .bound = 128 }, .{ .regs = 34, .static = 0, .bound = 128 }, .{ .regs = 34, .static = 0, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_softcap", .symbol = "tf_ink2_softcap", .image = "inkling2_ops", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 }, .{ .regs = 18, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_top1", .symbol = "tf_ink2_top1", .image = "inkling2_ops", .threads = 1024, .min_sm = 80 }, .usage = .{ .{ .regs = 13, .static = 256, .bound = 1024 }, .{ .regs = 16, .static = 256, .bound = 1024 }, .{ .regs = 16, .static = 256, .bound = 1024 }, .{ .regs = 18, .static = 256, .bound = 1024 }, .{ .regs = 18, .static = 256, .bound = 1024 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_race1", .symbol = "tf_ink2_race1", .image = "inkling2_ops", .threads = 1024, .min_sm = 80 }, .usage = .{ .{ .regs = 19, .static = 256, .bound = 1024 }, .{ .regs = 20, .static = 256, .bound = 1024 }, .{ .regs = 20, .static = 256, .bound = 1024 }, .{ .regs = 24, .static = 256, .bound = 1024 }, .{ .regs = 24, .static = 256, .bound = 1024 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_tobf", .symbol = "tf_ink2_tobf", .image = "inkling2_ops", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 }, .{ .regs = 12, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_put_words", .symbol = "tf_ink2_put_words", .image = "inkling2_ops", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 8, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 }, .{ .regs = 10, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_rowdot_tiles", .symbol = "tf_ink2_rowdot_tiles", .image = "inkling2_rows", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 102, .static = 24576, .bound = 256 }, .{ .regs = 102, .static = 24576, .bound = 256 }, .{ .regs = 102, .static = 24576, .bound = 256 }, .{ .regs = 106, .static = 24576, .bound = 256 }, .{ .regs = 106, .static = 24576, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "inkling2 ops tf_ink2_prompt_tail", .symbol = "tf_ink2_prompt_tail", .image = "inkling2_rows", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 30, .static = 0, .bound = 32 }, .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 24, .static = 0, .bound = 32 }, .{ .regs = 24, .static = 0, .bound = 32 } } },
};
/// nvcc 13.2's, where it differs: pending (no 13.2 compiler measured).
pub const cuda_132 = [_]registry.Override{};
/// nvcc 13.3's, where it differs.
pub const cuda_133 = [_]registry.Override{
    .{ .name = "inkling2 ops tf_ink2_mix", .usage = .{ .{ .regs = 76, .static = 45712, .bound = 512 }, .{ .regs = 76, .static = 45712, .bound = 512 }, .{ .regs = 76, .static = 45712, .bound = 512 }, .{ .regs = 74, .static = 45712, .bound = 512 }, .{ .regs = 74, .static = 45712, .bound = 512 } } },
    .{ .name = "inkling2 ops tf_ink2_rowdot", .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },
    .{ .name = "inkling2 ops tf_ink2_prompt_tail", .usage = .{ .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 30, .static = 0, .bound = 32 }, .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 23, .static = 0, .bound = 32 }, .{ .regs = 23, .static = 0, .bound = 32 } } },
};

test "Inkling 2 glue: every launch is registered once and fits every SM" {
    const std = @import("std");
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
