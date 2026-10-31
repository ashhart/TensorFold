//! A shared round's attention in one launch a kernel (flashnext_attn_lanes.cuh): measured resources for all five SM targets.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_attn fn_attn_prep_lanes", .symbol = "fn_attn_prep_lanes", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 128, .bound = 256 }, .{ .regs = 38, .static = 128, .bound = 256 }, .{ .regs = 38, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_pool_lanes", .symbol = "fn_qsa_pool_lanes", .image = "flashnext_attn", .threads = 128 }, .usage = .{ .{ .regs = 32, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_scores_lanes", .symbol = "fn_qsa_scores_lanes", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 225, .static = 16384, .bound = 256 }, .{ .regs = 225, .static = 16384, .bound = 256 }, .{ .regs = 225, .static = 16384, .bound = 256 }, .{ .regs = 220, .static = 16384, .bound = 256 }, .{ .regs = 220, .static = 16384, .bound = 256 }, } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks_lanes", .symbol = "fn_attn_chunks_lanes", .image = "flashnext_attn", .threads = 128, .dynamic = 77312, .opt_in = true }, .usage = .{ .{ .regs = 95, .static = 768, .bound = 128 }, .{ .regs = 95, .static = 768, .bound = 128 }, .{ .regs = 95, .static = 768, .bound = 128 }, .{ .regs = 72, .static = 768, .bound = 128 }, .{ .regs = 72, .static = 768, .bound = 128 }, } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_merge_lanes", .symbol = "fn_attn_merge_lanes", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 30, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, } },
    .{ .launch = .{ .name = "flashnext_batch fn_qsa_select_lanes", .symbol = "fn_qsa_select_lanes", .image = "flashnext_batch", .threads = 1024 }, .usage = .{ .{ .regs = 32, .static = 1168, .bound = 1024 }, .{ .regs = 50, .static = 1168, .bound = 1024 }, .{ .regs = 50, .static = 1168, .bound = 1024 }, .{ .regs = 44, .static = 1168, .bound = 1024 }, .{ .regs = 44, .static = 1168, .bound = 1024 }, } },
};
test "Flash Next one-launch attention: measured resources satisfy every SM launch limit" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
        for (r.usage) |u| {
            try t.expect(u.regs > 0);
            try t.expectEqual(r.launch.threads, u.bound);
        }
    }
}
