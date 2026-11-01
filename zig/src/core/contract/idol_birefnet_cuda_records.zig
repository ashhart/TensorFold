//! Idol's background remover's kernels on CUDA (idol_birefnet.cu): every SM's compile records from nvcc 13.0.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (cuobjdump -res-usage of the glue-flag image; sm_120's 1024 reserved shared bytes left out).
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "idol birefnet tf_br_conv", .symbol = "tf_br_conv", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 71, .static = 16896, .bound = 256 }, .{ .regs = 71, .static = 16896, .bound = 256 }, .{ .regs = 71, .static = 16896, .bound = 256 }, .{ .regs = 64, .static = 16896, .bound = 256 }, .{ .regs = 64, .static = 16896, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_norm", .symbol = "tf_br_norm", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_window", .symbol = "tf_br_window", .image = "idol_birefnet", .threads = 192, .min_sm = 80 }, .usage = .{ .{ .regs = 80, .static = 36000, .bound = 192 }, .{ .regs = 154, .static = 36000, .bound = 192 }, .{ .regs = 154, .static = 36000, .bound = 192 }, .{ .regs = 85, .static = 36000, .bound = 192 }, .{ .regs = 85, .static = 36000, .bound = 192 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_resize", .symbol = "tf_br_resize", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_tiles", .symbol = "tf_br_tiles", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 24, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_conv3", .symbol = "tf_br_conv3", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_deform", .symbol = "tf_br_deform", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 39, .static = 0, .bound = 256 }, .{ .regs = 39, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_pool", .symbol = "tf_br_pool", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 17, .static = 32, .bound = 256 }, .{ .regs = 17, .static = 32, .bound = 256 }, .{ .regs = 17, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_gate", .symbol = "tf_br_gate", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol birefnet tf_br_head", .symbol = "tf_br_head", .image = "idol_birefnet", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
};

test "Idol BiRefNet CUDA: measured launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (cuda) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
