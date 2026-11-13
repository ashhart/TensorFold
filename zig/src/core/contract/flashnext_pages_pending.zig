//! Paged Flash Next cache launches: measured nvcc 13.0 resources for all five SM targets.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "flashnext_attn fn_attn_prep_paged", .symbol = "fn_attn_prep_paged", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 30, .static = 128, .bound = 256 }, .{ .regs = 30, .static = 128, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_prep_media_paged", .symbol = "fn_attn_prep_media_paged", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 32, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 36, .static = 128, .bound = 256 }, .{ .regs = 30, .static = 128, .bound = 256 }, .{ .regs = 30, .static = 128, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_pool_paged", .symbol = "fn_qsa_pool_paged", .image = "flashnext_attn", .threads = 128 }, .usage = .{ .{ .regs = 32, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_pool_media_paged", .symbol = "fn_qsa_pool_media_paged", .image = "flashnext_attn", .threads = 128 }, .usage = .{ .{ .regs = 32, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 }, .{ .regs = 40, .static = 1152, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_qsa_scores_paged", .symbol = "fn_qsa_scores_paged", .image = "flashnext_attn", .threads = 256 }, .usage = .{ .{ .regs = 215, .static = 16384, .bound = 256 }, .{ .regs = 216, .static = 16384, .bound = 256 }, .{ .regs = 216, .static = 16384, .bound = 256 }, .{ .regs = 218, .static = 16384, .bound = 256 }, .{ .regs = 218, .static = 16384, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks_ahead_paged", .symbol = "fn_attn_chunks_ahead_paged", .image = "flashnext_attn", .threads = 128, .dynamic = 77312, .opt_in = true }, .usage = .{ .{ .regs = 76, .static = 1280, .bound = 128 }, .{ .regs = 76, .static = 1280, .bound = 128 }, .{ .regs = 76, .static = 1280, .bound = 128 }, .{ .regs = 73, .static = 1280, .bound = 128 }, .{ .regs = 73, .static = 1280, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_chunks_one_paged", .symbol = "fn_attn_chunks_one_paged", .image = "flashnext_attn", .threads = 128, .dynamic = 26624, .opt_in = true }, .usage = .{ .{ .regs = 72, .static = 896, .bound = 128 }, .{ .regs = 72, .static = 896, .bound = 128 }, .{ .regs = 72, .static = 896, .bound = 128 }, .{ .regs = 96, .static = 896, .bound = 128 }, .{ .regs = 96, .static = 896, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_attn fn_attn_fused_paged", .symbol = "fn_attn_fused_paged", .image = "flashnext_attn", .threads = 128, .dynamic = 26624 }, .usage = .{ .{ .regs = 118, .static = 896, .bound = 128 }, .{ .regs = 118, .static = 896, .bound = 128 }, .{ .regs = 118, .static = 896, .bound = 128 }, .{ .regs = 123, .static = 896, .bound = 128 }, .{ .regs = 123, .static = 896, .bound = 128 } } },
};
test "Flash Next paged launches: measured resources satisfy every SM launch limit" {
    const t = @import("std").testing;
    for (records) |r| {
        try t.expectEqual(@as(usize, 0), try registry.checkCuda(t.allocator, &.{r}, true));
        for (r.usage) |u| {
            try t.expect(u.regs > 0);
            try t.expectEqual(r.launch.threads, u.bound);
        }
        for (registry.cuda) |measured| try t.expect(!@import("std").mem.eql(u8, r.launch.symbol, measured.launch.symbol));
    }
}
