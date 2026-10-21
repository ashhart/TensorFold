//! GLM E4M3 latent cache launches need compiler resources and native numerical gates before qualification.
const registry = @import("registry.zig");
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_lat_write_fp8", .symbol = "tf_glm_lat_write_fp8", .image = "glm_glue", .threads = 512, .min_sm = 80 }, .usage = none },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_16_fp8", .symbol = "tf_glm_chunks_16_fp8", .image = "glm_glue", .threads = 256, .dynamic = 83328, .opt_in = true, .min_sm = 80 }, .usage = none },
    .{ .launch = .{ .name = "glm_glue tf_glm_chunks_32_fp8", .symbol = "tf_glm_chunks_32_fp8", .image = "glm_glue", .threads = 256, .dynamic = 101120, .opt_in = true, .min_sm = 80 }, .usage = none },
};
test "GLM FP8 KV: pending launches have no invented compiler-resource qualification" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(none, r.usage);
        try t.expectEqual(limits.sms.len, try registry.checkCuda(t.allocator, &.{r}, true));
    }
}
