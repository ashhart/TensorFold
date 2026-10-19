//! GLM sampling launches retain pending 13.0 resources; 13.2 measurements are selected separately.
const std = @import("std");
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");
const none: [limits.sms.len]limits.Usage = @splat(.{ .regs = 0, .static = 0 });

/// Shared sampling and GLM greedy launches used for prompt scoring and grammar-constrained draws.
pub const sampling = [_]R{
    .{ .launch = .{ .name = "sample tf_draw", .symbol = "tf_draw", .image = "sample", .threads = 1024 }, .usage = none },
    .{ .launch = .{ .name = "sample tf_logprob_rows", .symbol = "tf_logprob_rows", .image = "sample", .threads = 1024 }, .usage = none },
    .{ .launch = .{ .name = "sample tf_grammar_mask", .symbol = "tf_grammar_mask", .image = "sample", .threads = 256 }, .usage = none },
    .{ .launch = .{ .name = "glm_glue tf_glm_top1", .symbol = "tf_glm_top1", .image = "glm_glue", .threads = 1024 }, .usage = none },
};

test "GLM 13.0 pending records cannot pass the measured resource gate" {
    const registry = @import("registry.zig");
    for (sampling) |r| {
        try std.testing.expectEqual(none, r.usage);
        try std.testing.expectEqual(limits.sms.len, try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
