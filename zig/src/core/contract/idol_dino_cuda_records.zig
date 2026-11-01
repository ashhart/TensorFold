//! Idol's DINOv3 kernels on CUDA (idol_dino.cu): every SM's compile records from nvcc 13.0, the release compiler.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (cuobjdump -res-usage of the glue-flag image; sm_120's 1024 reserved shared bytes left out).
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "idol dino tf_idol_dino_bias", .symbol = "tf_idol_dino_bias", .image = "idol_dino", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol dino tf_idol_dino_norm", .symbol = "tf_idol_dino_norm", .image = "idol_dino", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 31, .static = 32, .bound = 256 }, .{ .regs = 31, .static = 32, .bound = 256 }, .{ .regs = 31, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "idol dino tf_idol_dino_final", .symbol = "tf_idol_dino_final", .image = "idol_dino", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 5152, .bound = 256 }, .{ .regs = 39, .static = 5152, .bound = 256 }, .{ .regs = 39, .static = 5152, .bound = 256 }, .{ .regs = 40, .static = 5152, .bound = 256 }, .{ .regs = 40, .static = 5152, .bound = 256 } } },
    .{ .launch = .{ .name = "idol dino tf_idol_dino_heads", .symbol = "tf_idol_dino_heads", .image = "idol_dino", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 19, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol dino tf_idol_dino_scale_add", .symbol = "tf_idol_dino_scale_add", .image = "idol_dino", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 14, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol dino tf_idol_dino_swiglu", .symbol = "tf_idol_dino_swiglu", .image = "idol_dino", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol dino tf_idol_dino_attention", .symbol = "tf_idol_dino_attention", .image = "idol_dino", .threads = 256, .dynamic = 36864, .min_sm = 80 }, .usage = .{ .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 132, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 } } },
};

test "Idol DINOv3 CUDA: measured launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (cuda) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
