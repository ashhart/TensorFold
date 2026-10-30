//! MOSS-Image's CUDA launches: its own steps (moss_image.cu) and full attention at head dim 128.
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins; nvcc 13.2 compiles each the same.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "moss_image tf_moss_norm", .symbol = "tf_moss_norm", .image = "moss_image", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 32, .bound = 256 }, .{ .regs = 34, .static = 32, .bound = 256 }, .{ .regs = 34, .static = 32, .bound = 256 }, .{ .regs = 33, .static = 32, .bound = 256 }, .{ .regs = 33, .static = 32, .bound = 256 } } },
    .{ .launch = .{ .name = "moss_image tf_moss_qk", .symbol = "tf_moss_qk", .image = "moss_image", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 }, .{ .regs = 40, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "moss_image tf_moss_situ", .symbol = "tf_moss_situ", .image = "moss_image", .threads = 256 }, .usage = .{ .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 18, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "moss_image tf_moss_gate_add", .symbol = "tf_moss_gate_add", .image = "moss_image", .threads = 256 }, .usage = .{ .{ .regs = 11, .static = 0, .bound = 256 }, .{ .regs = 11, .static = 0, .bound = 256 }, .{ .regs = 11, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 }, .{ .regs = 12, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "vit vision_kernel<128, 8> (MOSS-Image)", .symbol = "_ZN20tf_prefill_attention13vision_kernelILi128ELi8EEEvPK13__nv_bfloat16PS1_iif", .image = "prefill_attention", .threads = 256, .dynamic = 69632, .opt_in = true }, .usage = .{ .{ .regs = 178, .static = 0, .bound = 256 }, .{ .regs = 178, .static = 0, .bound = 256 }, .{ .regs = 178, .static = 0, .bound = 256 }, .{ .regs = 170, .static = 0, .bound = 256 }, .{ .regs = 170, .static = 0, .bound = 256 } } },
};

test "MOSS-Image CUDA: every launch is in the live registry and fits every SM" {
    const t = @import("std").testing;
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(@import("std").mem.eql(u8, active.launch.name, r.launch.name));
        try t.expectEqual(@as(usize, 1), found);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
