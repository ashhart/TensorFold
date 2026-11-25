//! Hy4-flash fused iHC (`hy4_ihc_fused`) and HC (`hy4_hc`) CUDA launch records from `nvcc 13.0` across 5 SMs.
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
    .{ .launch = .{ .name = "hy4_hc hy4_hc_pre", .symbol = "hy4_hc_pre", .image = "hy4_hc", .threads = 256 }, .usage = sameS(256, 304, 80, 80, 80, 80, 80) },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_head", .symbol = "hy4_hc_head", .image = "hy4_hc", .threads = 256 }, .usage = sameS(256, 176, 48, 48, 48, 48, 48) },
    .{ .launch = .{ .name = "hy4_hc hy4_hc_merge", .symbol = "hy4_hc_merge", .image = "hy4_hc", .threads = 256 }, .usage = sameS(256, 0, 22, 22, 22, 22, 22) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_fused_boundary", .symbol = "hy4_ihc_fused_boundary", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 46672 }, .usage = sameS(256, 160, 147, 149, 149, 162, 162) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_entry", .symbol = "hy4_ihc_entry", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 46672 }, .usage = sameS(256, 16, 144, 144, 144, 162, 162) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_epilogue_head", .symbol = "hy4_ihc_epilogue_head", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 30288 }, .usage = sameS(256, 160, 150, 148, 148, 166, 166) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_head_only", .symbol = "hy4_ihc_head_only", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 30288 }, .usage = sameS(256, 16, 148, 148, 148, 160, 160) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_post_only", .symbol = "hy4_ihc_post_only", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 7696 }, .usage = sameS(256, 144, 80, 80, 80, 79, 79) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_fused_boundary_coop", .symbol = "hy4_ihc_fused_boundary_coop", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 46336, 138, 140, 140, 154, 154) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_entry_coop", .symbol = "hy4_ihc_entry_coop", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 45952, 139, 141, 141, 153, 153) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_epilogue_head_coop", .symbol = "hy4_ihc_epilogue_head_coop", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 42240, 102, 104, 104, 117, 117) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_head_only_coop", .symbol = "hy4_ihc_head_only_coop", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 41856, 107, 109, 109, 116, 116) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_post_vec", .symbol = "hy4_ihc_post_vec", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 12320, 32, 46, 46, 54, 54) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_pre_tiled_rb2", .symbol = "hy4_ihc_pre_tiled_rb2", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 7080 }, .usage = sameS(256, 0, 48, 48, 48, 56, 56) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_pre_tiled_rb4", .symbol = "hy4_ihc_pre_tiled_rb4", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 13904 }, .usage = sameS(256, 0, 48, 48, 48, 63, 63) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_head_tiled", .symbol = "hy4_ihc_head_tiled", .image = "hy4_ihc_fused", .threads = 256, .dynamic = 27552 }, .usage = sameS(256, 0, 48, 48, 48, 61, 61) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_fused_prefill_m8", .symbol = "hy4_ihc_fused_prefill_m8", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 18816, 38, 48, 48, 48, 48) },
    .{ .launch = .{ .name = "hy4_ihc_fused hy4_ihc_fused_prefill_m4", .symbol = "hy4_ihc_fused_prefill_m4", .image = "hy4_ihc_fused", .threads = 256 }, .usage = sameS(256, 18816, 38, 48, 48, 48, 48) },
};

test "Hy4 fused iHC and HC CUDA: every launch is in the live registry and fits every SM" {
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
