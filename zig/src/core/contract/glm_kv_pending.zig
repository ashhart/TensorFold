//! GLM E4M3 latent cache resources from nvcc 13.0; numerical device gates remain separate.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_lat_write_fp8", .symbol = "tf_glm_lat_write_fp8", .image = "glm_glue", .threads = 512, .min_sm = 80 }, .usage = .{ .{ .regs = 26, .static = 64, .bound = 512 }, .{ .regs = 28, .static = 64, .bound = 512 }, .{ .regs = 28, .static = 64, .bound = 512 }, .{ .regs = 25, .static = 64, .bound = 512 }, .{ .regs = 25, .static = 64, .bound = 512 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_16_fp8", .symbol = "tf_glm_chunks_16_fp8", .image = "glm_glue", .threads = 256, .dynamic = 83328, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 }, .{ .regs = 64, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_32_fp8", .symbol = "tf_glm_chunks_32_fp8", .image = "glm_glue", .threads = 256, .dynamic = 101120, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 }, .{ .regs = 127, .static = 0, .bound = 256 }, .{ .regs = 130, .static = 0, .bound = 256 }, .{ .regs = 130, .static = 0, .bound = 256 } }, .later_pending = true },
};
test "GLM FP8 KV: measured 13.0 launches fit their physical resource limits" {
    const t = @import("std").testing;
    for (records) |r| {
        for (r.usage) |u| try t.expect(u.regs > 0);
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
