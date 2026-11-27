//! Hy4-flash router (`hy4_route`), row helpers (`hy4_rows`), and DFly2 glue (`hy4_dfly2_glue`) 5-SM records.
const std = @import("std");
const registry = @import("registry.zig");
const U = @import("limits.zig").Usage;

fn sameS(b: u32, s: u32, r80: u32, r86: u32, r89: u32, r120: u32, r121: u32) [5]U {
    return .{
        .{ .regs = r80, .static = s, .bound = b },
        .{ .regs = r86, .static = s, .bound = b },
        .{ .regs = r89, .static = s, .bound = b },
        .{ .regs = r120, .static = s, .bound = b },
        .{ .regs = r121, .static = s, .bound = b },
    };
}

pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "hy4_route tf_hy4_route_m4", .symbol = "tf_hy4_route_m4", .image = "hy4_route", .threads = 128 }, .usage = sameS(128, 6144, 56, 56, 56, 56, 56) },
    .{ .launch = .{ .name = "hy4_route tf_hy4_route_m4_plan1", .symbol = "tf_hy4_route_m4_plan1", .image = "hy4_route", .threads = 128 }, .usage = sameS(128, 6144, 56, 56, 56, 56, 56) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_wide", .symbol = "hy4_rows_wide", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 125, 125, 125, 126, 126) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_narrow", .symbol = "hy4_rows_narrow", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 54, 56, 56, 62, 62) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_argmax", .symbol = "hy4_rows_argmax", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 96, 20, 20, 20, 29, 29) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_slice_bf16", .symbol = "hy4_rows_slice_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_interleave2_bf16", .symbol = "hy4_rows_interleave2_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 18, 18, 18, 24, 24) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_slice_f32", .symbol = "hy4_rows_slice_f32", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_comb_swiglu_bf16", .symbol = "hy4_rows_comb_swiglu_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 17, 17, 17, 17, 17) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_kv_write", .symbol = "hy4_rows_kv_write", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 16, 25, 24, 24, 24, 24) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_q_rope", .symbol = "hy4_rows_q_rope", .image = "hy4_rows", .threads = 32 }, .usage = sameS(32, 0, 26, 26, 26, 23, 23) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_f32_to_bf16", .symbol = "hy4_rows_f32_to_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 25, 25, 25, 24, 24) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_interleave_bf16", .symbol = "hy4_rows_interleave_bf16", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_kv_write_ex", .symbol = "hy4_rows_kv_write_ex", .image = "hy4_rows", .threads = 512 }, .usage = sameS(512, 16, 25, 23, 23, 24, 24) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_kv_write_paged", .symbol = "hy4_rows_kv_write_paged", .image = "hy4_rows", .threads = 512 }, .usage = sameS(512, 16, 26, 27, 27, 24, 24) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_q_rope_mrope", .symbol = "hy4_rows_q_rope_mrope", .image = "hy4_rows", .threads = 32 }, .usage = sameS(32, 0, 27, 27, 28, 22, 22) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_splice_streams", .symbol = "hy4_rows_splice_streams", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 16, 16, 16, 16, 16) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_vit_qkv", .symbol = "hy4_rows_vit_qkv", .image = "hy4_rows", .threads = 128 }, .usage = sameS(128, 0, 30, 29, 29, 30, 30) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_vit_merge_gather", .symbol = "hy4_rows_vit_merge_gather", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 26, 26, 26, 28, 28) },
    .{ .launch = .{ .name = "hy4_rows hy4_rows_vit_append_newline", .symbol = "hy4_rows_vit_append_newline", .image = "hy4_rows", .threads = 256 }, .usage = sameS(256, 0, 18, 18, 18, 22, 22) },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_dual_rmsnorm_cat_s1", .symbol = "hy4_dfly_dual_rmsnorm_cat_s1", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 48, 29, 31, 31, 31, 31) },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_fusion_gated_add_s1", .symbol = "hy4_dfly_fusion_gated_add_s1", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 48, 40, 40, 40, 40, 40) },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_sublayer_boundary_s1", .symbol = "hy4_dfly_sublayer_boundary_s1", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 96, 40, 40, 40, 40, 40) },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_extend_topk_k", .symbol = "hy4_dfly_extend_topk_k", .image = "hy4_dfly2_glue", .threads = 128 }, .usage = sameS(128, 0, 32, 40, 40, 40, 40) },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_pick_argmax", .symbol = "hy4_dfly_pick_argmax", .image = "hy4_dfly2_glue", .threads = 64 }, .usage = sameS(64, 0, 12, 12, 12, 14, 14) },
    .{ .launch = .{ .name = "hy4_dfly2_glue hy4_dfly_gather_hidden", .symbol = "hy4_dfly_gather_hidden", .image = "hy4_dfly2_glue", .threads = 384 }, .usage = sameS(384, 0, 16, 16, 16, 18, 18) },
};

test "Hy4 route, rows, and DFly2 glue CUDA: every launch is in the live registry and fits every SM" {
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
