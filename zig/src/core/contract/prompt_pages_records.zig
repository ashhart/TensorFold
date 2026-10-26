//! The Qwen3.5 prompt tiles (prefill_attention.cu) and their paged twins: measured nvcc 13.0 resources on every SM.
const registry = @import("registry.zig");
fn rec(name: []const u8, symbol: []const u8, regs: [5]u32) registry.CudaRecord {
    return .{ .launch = .{ .name = name, .symbol = symbol, .image = "prefill_attention", .threads = 256, .dynamic = 98304, .opt_in = true }, .usage = over(regs) };
}
pub const records = [_]registry.CudaRecord{
    rec("prompt hpattn_kernel<256, 8, 2, 2>", "_ZN20tf_prefill_attention13hpattn_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifi", .{ 254, 254, 254, 255, 255 }),
    rec("prompt hpattn8_kernel<256, 8, 2, 2>", "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi", .{ 255, 255, 255, 255, 255 }),
    rec("prompt hpattn_paged_kernel<256, 8, 2, 2>", "_ZN20tf_prefill_attention19hpattn_paged_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiifiPKix", .{ 251, 251, 251, 231, 231 }),
    rec("prompt hpattn8_paged_kernel<256, 8, 2, 2>", "_ZN20tf_prefill_attention20hpattn8_paged_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKix", .{ 255, 255, 255, 255, 255 }),
};
test "prompt tile launches: measured resources satisfy every SM launch limit" {
    const t = @import("std").testing;
    for (records) |r| try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
}

/// The prompt tiles nvcc 13.2 compiles to other register counts, from every SM's image.
pub const cuda_132 = [_]registry.Override{
    .{ .name = "prompt hpattn_kernel<256, 8, 2, 2>", .usage = over(.{ 253, 253, 253, 255, 255 }) },
    .{ .name = "prompt hpattn_paged_kernel<256, 8, 2, 2>", .usage = over(.{ 251, 251, 251, 232, 232 }) },
};
fn over(regs: [5]u32) [5]@import("limits.zig").Usage {
    var usage: [5]@import("limits.zig").Usage = undefined;
    for (&usage, regs) |*u, r| u.* = .{ .regs = r, .static = 0, .bound = 256 };
    return usage;
}
