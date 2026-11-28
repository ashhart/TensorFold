//! Idol's decoder kernels on CUDA (idol_decoder.cu): every SM's compile records from nvcc 13.0, the release compiler.
const registry = @import("registry.zig");

/// nvcc 13.0's usage (cuobjdump -res-usage of the glue-flag image; sm_120's 1024 reserved shared bytes left out).
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "idol decoder tf_idol_dec_embed", .symbol = "tf_idol_dec_embed", .image = "idol_decoder", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 20, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 }, .{ .regs = 22, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol decoder tf_idol_dec_half", .symbol = "tf_idol_dec_half", .image = "idol_decoder", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol decoder tf_idol_dec_kv", .symbol = "tf_idol_dec_kv", .image = "idol_decoder", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol decoder tf_idol_dec_heads", .symbol = "tf_idol_dec_heads", .image = "idol_decoder", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 }, .{ .regs = 21, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol decoder tf_idol_dec_attention", .symbol = "tf_idol_dec_attention", .image = "idol_decoder", .threads = 256, .dynamic = 36864, .min_sm = 80 }, .usage = .{ .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 }, .{ .regs = 120, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "idol decoder tf_idol_dec_few", .symbol = "tf_idol_dec_few", .image = "idol_decoder", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 128, .static = 2352, .bound = 256 }, .{ .regs = 128, .static = 2352, .bound = 256 }, .{ .regs = 128, .static = 2352, .bound = 256 }, .{ .regs = 128, .static = 2352, .bound = 256 }, .{ .regs = 128, .static = 2352, .bound = 256 } } },
};

test "Idol decoder CUDA: measured launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (cuda) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
