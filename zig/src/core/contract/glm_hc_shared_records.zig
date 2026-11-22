//! Shared decode HC launch: resource usage is pending a real nvcc receipt on every SM.
const registry = @import("registry.zig");
const limits = @import("limits.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "glm_glue tf_glm_hc_partial8_shared", .symbol = "tf_glm_hc_partial8_shared", .image = "glm_glue", .threads = 1024, .min_sm = 80 }, .usage = @splat(.{ .regs = 0, .static = 0 }), .later_pending = true },
};
test "GLM shared HC records remain unqualified without measured compiler usage" {
    for (records[0].usage) |u| try @import("std").testing.expectEqual(@as(u32, 0), u.regs);
    try @import("std").testing.expectEqual(@as(usize, limits.sms.len), records[0].usage.len);
}
