//! Ripple's prompt attention on CUDA: window_attention.cu's two-heads-a-block instances, for kernel-contract-limits.
const registry = @import("registry.zig");
const limits = @import("limits.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins: not yet measured (every value 0).
pub const cuda = [_]registry.CudaRecord{
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "window attention wattn_kernel<128, 8, 2, 8>", .symbol = "_ZN16tf_window_prompt12wattn_kernelILi128ELi8ELi2ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifii", .image = "window_attention", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
    // TODO(spark): fill from kernel-contract-limits
    .{ .launch = .{ .name = "window attention tf_window_prompt_h2_pg", .symbol = "tf_window_prompt_h2_pg", .image = "window_attention", .threads = 256, .dynamic = 65536, .opt_in = true }, .usage = .{ .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 }, .{ .regs = 0, .static = 0, .bound = 0 } } },
};

test "Ripple's prompt attention: unmeasured records cannot pass a compiled resource check" {
    const t = @import("std").testing;
    for (cuda) |r| try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
}
