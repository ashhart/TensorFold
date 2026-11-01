//! Nemotron 3.5 Transcribe's audio encoder launches (nemo_audio.cu); attention holds up to 2,048 scores shared.
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins; nvcc 13.2 compiles each the same.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "nemo_audio tf_na_qk_rope", .symbol = "tf_na_qk_rope", .image = "nemo_audio", .threads = 32 }, .usage = .{ .{ .regs = 38, .static = 512, .bound = 32 }, .{ .regs = 39, .static = 512, .bound = 32 }, .{ .regs = 39, .static = 512, .bound = 32 }, .{ .regs = 39, .static = 512, .bound = 32 }, .{ .regs = 39, .static = 512, .bound = 32 } } },
    .{ .launch = .{ .name = "nemo_audio tf_na_attention", .symbol = "tf_na_attention", .image = "nemo_audio", .threads = 128, .dynamic = 8192 }, .usage = .{ .{ .regs = 32, .static = 528, .bound = 128 }, .{ .regs = 40, .static = 528, .bound = 128 }, .{ .regs = 40, .static = 528, .bound = 128 }, .{ .regs = 40, .static = 528, .bound = 128 }, .{ .regs = 40, .static = 528, .bound = 128 } } },
    .{ .launch = .{ .name = "nemo_audio tf_na_im2col3", .symbol = "tf_na_im2col3", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_na_relu", .symbol = "tf_na_relu", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 }, .{ .regs = 8, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "nemo_audio tf_na_zero_rows", .symbol = "tf_na_zero_rows", .image = "nemo_audio", .threads = 256 }, .usage = .{ .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 10, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
};

test "Nemotron 3.5 Transcribe audio CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
