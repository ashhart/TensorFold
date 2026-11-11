//! Nemotron 3.5 Transcribe's audio encoder launches (nemo_audio.cu): its glue and head-64 attention.
const registry = @import("registry.zig");

/// nvcc 13.2's usage from each SM's cubin (sm_120 and sm_121 without the 1 KiB the driver reserves).
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nemo_audio tf_nemo_mel", .symbol = "tf_nemo_mel", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 34, .static = 2628, .bound = 256 }, .{ .regs = 42, .static = 2628, .bound = 256 }, .{ .regs = 42, .static = 2628, .bound = 256 }, .{ .regs = 40, .static = 2628, .bound = 256 }, .{ .regs = 40, .static = 2628, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_norm", .symbol = "tf_nemo_norm", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 1024, .bound = 256 }, .{ .regs = 32, .static = 1024, .bound = 256 }, .{ .regs = 32, .static = 1024, .bound = 256 }, .{ .regs = 36, .static = 1024, .bound = 256 }, .{ .regs = 36, .static = 1024, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_qk", .symbol = "tf_nemo_qk", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 }, .{ .regs = 34, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_attn64", .symbol = "tf_nemo_attn64", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 52, .static = 43264, .bound = 256 }, .{ .regs = 53, .static = 43264, .bound = 256 }, .{ .regs = 53, .static = 43264, .bound = 256 }, .{ .regs = 69, .static = 43264, .bound = 256 }, .{ .regs = 69, .static = 43264, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_taps", .symbol = "tf_nemo_taps", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_relu", .symbol = "tf_nemo_relu", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_preds", .symbol = "tf_nemo_preds", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 29, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_nemo_fuse", .symbol = "tf_nemo_fuse", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 38, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 }, .{ .regs = 40, .static = 32, .bound = 256 } } },
};

test "Transcribe audio CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
