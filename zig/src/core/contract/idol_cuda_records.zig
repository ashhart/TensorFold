//! Idol's denoiser kernels on CUDA (idol.cu): every SM's compile records from nvcc 13.0, the release compiler.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (cuobjdump -res-usage of the glue-flag image; sm_120's 1024 reserved shared bytes left out).
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "idol tf_idol_norm", .symbol = "tf_idol_norm", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 36, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_norm32", .symbol = "tf_idol_norm32", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_repo", .symbol = "tf_idol_repo", .image = "idol", .threads = 64, .min_sm = 80 }, .usage = .{ .{ .regs = 38, .static = 704, .bound = 64 }, .{ .regs = 66, .static = 704, .bound = 64 }, .{ .regs = 44, .static = 704, .bound = 64 }, .{ .regs = 37, .static = 704, .bound = 64 }, .{ .regs = 37, .static = 704, .bound = 64 } } },
    .{ .launch = .{ .name = "idol tf_idol_qk", .symbol = "tf_idol_qk", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 27, .static = 0, .bound = 256 }, .{ .regs = 27, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_gate_add", .symbol = "tf_idol_gate_add", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_bias", .symbol = "tf_idol_bias", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 15, .static = 0, .bound = 256 }, .{ .regs = 15, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_embed", .symbol = "tf_idol_embed", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_cond", .symbol = "tf_idol_cond", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_bias32", .symbol = "tf_idol_bias32", .image = "idol", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol tf_idol_attention", .symbol = "tf_idol_attention", .image = "idol", .threads = 256, .dynamic = 36864, .min_sm = 80 }, .usage = .{ .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 } } },
};

test "Idol CUDA: measured launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (cuda) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
