//! GLM sampling resources from nvcc 13.0 and independent later compiler sets.
const std = @import("std");
const R = @import("registry.zig").CudaRecord;

/// Shared sampling and GLM greedy launches: prompt scoring, grammar-constrained draws and MTP drafts' chances.
pub const sampling = [_]R{
    .{ .launch = .{ .name = "sample tf_draw", .symbol = "tf_draw", .image = "sample", .threads = 1024 }, .usage = .{ .{ .regs = 52, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 } }, .later_pending = true },
    .{ .launch = .{ .name = "sample tf_logprob_rows", .symbol = "tf_logprob_rows", .image = "sample", .threads = 1024 }, .usage = .{ .{ .regs = 27, .static = 520, .bound = 1024 }, .{ .regs = 28, .static = 520, .bound = 1024 }, .{ .regs = 28, .static = 520, .bound = 1024 }, .{ .regs = 40, .static = 520, .bound = 1024 }, .{ .regs = 40, .static = 520, .bound = 1024 } }, .later_pending = true },
    .{ .launch = .{ .name = "sample tf_grammar_mask", .symbol = "tf_grammar_mask", .image = "sample", .threads = 256 }, .usage = .{ .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 }, .{ .regs = 16, .static = 0, .bound = 256 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_top1", .symbol = "tf_glm_top1", .image = "glm_glue", .threads = 1024 }, .usage = .{ .{ .regs = 29, .static = 8192, .bound = 1024 }, .{ .regs = 30, .static = 8192, .bound = 1024 }, .{ .regs = 30, .static = 8192, .bound = 1024 }, .{ .regs = 36, .static = 8192, .bound = 1024 }, .{ .regs = 36, .static = 8192, .bound = 1024 } }, .later_pending = true },
    .{ .launch = .{ .name = "glm_glue tf_glm_top1_sums", .symbol = "tf_glm_top1_sums", .image = "glm_glue", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 8192, .bound = 1024 }, .{ .regs = 31, .static = 8192, .bound = 1024 }, .{ .regs = 31, .static = 8192, .bound = 1024 }, .{ .regs = 39, .static = 8192, .bound = 1024 }, .{ .regs = 39, .static = 8192, .bound = 1024 } }, .later_pending = true },
};

test "GLM sampling has measured 13.0 resources on every target" {
    const registry = @import("registry.zig");
    for (sampling) |r| {
        for (r.usage) |u| try std.testing.expect(u.regs > 0);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
