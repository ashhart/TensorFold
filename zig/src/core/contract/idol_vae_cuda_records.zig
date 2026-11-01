//! Idol's image encoder kernels on CUDA (idol_vae.cu): every SM's compile records from nvcc 13.0, the release compiler.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (cuobjdump -res-usage of the glue-flag image; sm_120's 1024 reserved shared bytes left out).
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "idol vae tf_iv_conv_in", .symbol = "tf_iv_conv_in", .image = "idol_vae", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 37, .static = 0, .bound = 256 }, .{ .regs = 37, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_conv", .symbol = "tf_iv_conv", .image = "idol_vae", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 128, .static = 16640, .bound = 256 }, .{ .regs = 128, .static = 16640, .bound = 256 }, .{ .regs = 128, .static = 16640, .bound = 256 }, .{ .regs = 145, .static = 16640, .bound = 256 }, .{ .regs = 145, .static = 16640, .bound = 256 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_group_part", .symbol = "tf_iv_group_part", .image = "idol_vae", .threads = 512, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 2048, .bound = 512 }, .{ .regs = 40, .static = 2048, .bound = 512 }, .{ .regs = 40, .static = 2048, .bound = 512 }, .{ .regs = 34, .static = 2048, .bound = 512 }, .{ .regs = 40, .static = 2048, .bound = 512 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_group_finish", .symbol = "tf_iv_group_finish", .image = "idol_vae", .threads = 1024, .min_sm = 80 }, .usage = .{ .{ .regs = 31, .static = 0, .bound = 1024 }, .{ .regs = 40, .static = 0, .bound = 1024 }, .{ .regs = 40, .static = 0, .bound = 1024 }, .{ .regs = 44, .static = 0, .bound = 1024 }, .{ .regs = 44, .static = 0, .bound = 1024 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_norm", .symbol = "tf_iv_norm", .image = "idol_vae", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_bias", .symbol = "tf_iv_bias", .image = "idol_vae", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_softmax", .symbol = "tf_iv_softmax", .image = "idol_vae", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 91, .static = 32, .bound = 256 }, .{ .regs = 109, .static = 32, .bound = 256 }, .{ .regs = 109, .static = 32, .bound = 256 }, .{ .regs = 196, .static = 32, .bound = 256 }, .{ .regs = 196, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "idol vae tf_iv_tokens", .symbol = "tf_iv_tokens", .image = "idol_vae", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 19, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
};

test "Idol VAE CUDA: measured launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (cuda) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
