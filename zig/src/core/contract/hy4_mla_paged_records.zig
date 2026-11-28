//! Hy4-flash paged MLA (`hy4_mla_paged`, `hy4_mla_attn_paged`, `hy4_mla_c2`) 5-SM launch records (`nvcc 13.0`).
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
    .{ .launch = .{ .name = "hy4_mla_paged tf_hy4_mla_absorb", .symbol = "tf_hy4_mla_absorb", .image = "hy4_mla_paged", .threads = 128 }, .usage = sameS(128, 12288, 104, 106, 106, 88, 94) },
    .{ .launch = .{ .name = "hy4_mla_paged tf_hy4_mla_attn", .symbol = "tf_hy4_mla_attn", .image = "hy4_mla_paged", .threads = 256, .dynamic = 56672, .opt_in = true }, .usage = sameS(256, 0, 64, 64, 64, 78, 78) },
    .{ .launch = .{ .name = "hy4_mla_paged tf_hy4_mla_attn_rows", .symbol = "tf_hy4_mla_attn_rows", .image = "hy4_mla_paged", .threads = 256, .dynamic = 56672, .opt_in = true }, .usage = sameS(256, 0, 74, 77, 77, 80, 80) },
    .{ .launch = .{ .name = "hy4_mla_paged tf_hy4_mla_value", .symbol = "tf_hy4_mla_value", .image = "hy4_mla_paged", .threads = 128 }, .usage = sameS(128, 8320, 60, 56, 56, 56, 56) },
    .{ .launch = .{ .name = "hy4_mla_paged tf_hy4_mla_post", .symbol = "tf_hy4_mla_post", .image = "hy4_mla_paged", .threads = 256 }, .usage = sameS(256, 32, 40, 40, 40, 40, 40) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_dec_x16", .symbol = "hy4_mla_dec_x16", .image = "hy4_mla_attn_paged", .threads = 256, .dynamic = 81552, .opt_in = true }, .usage = sameS(256, 0, 125, 125, 125, 130, 130) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_dec_x16_rows", .symbol = "hy4_mla_dec_x16_rows", .image = "hy4_mla_attn_paged", .threads = 256, .dynamic = 81552, .opt_in = true }, .usage = sameS(256, 0, 127, 127, 127, 132, 132) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_dec_x16_qreg", .symbol = "hy4_mla_dec_x16_qreg", .image = "hy4_mla_attn_paged", .threads = 256, .dynamic = 48896 }, .usage = sameS(256, 0, 128, 128, 128, 128, 128) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_dec_x16_qreg_rows", .symbol = "hy4_mla_dec_x16_qreg_rows", .image = "hy4_mla_attn_paged", .threads = 256, .dynamic = 48896 }, .usage = sameS(256, 0, 128, 128, 128, 128, 128) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_fold", .symbol = "hy4_mla_fold", .image = "hy4_mla_attn_paged", .threads = 256 }, .usage = sameS(256, 16640, 64, 64, 64, 62, 62) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_dec_x16_paged", .symbol = "hy4_mla_dec_x16_paged", .image = "hy4_mla_attn_paged", .threads = 256, .dynamic = 81552, .opt_in = true }, .usage = sameS(256, 0, 125, 125, 125, 132, 132) },
    .{ .launch = .{ .name = "hy4_mla_attn_paged hy4_mla_dec_x16_qreg_paged", .symbol = "hy4_mla_dec_x16_qreg_paged", .image = "hy4_mla_attn_paged", .threads = 256, .dynamic = 48896 }, .usage = sameS(256, 0, 128, 128, 128, 128, 128) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_absorb_d1_x", .symbol = "hy4c2_absorb_d1_x", .image = "hy4_mla_c2", .threads = 128, .dynamic = 5632 }, .usage = sameS(128, 0, 82, 82, 82, 78, 78) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_absorb_d2_x", .symbol = "hy4c2_absorb_d2_x", .image = "hy4_mla_c2", .threads = 128, .dynamic = 8704 }, .usage = sameS(128, 0, 92, 92, 92, 88, 88) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_absorb_p_x", .symbol = "hy4c2_absorb_p_x", .image = "hy4_mla_c2", .threads = 256, .dynamic = 33792 }, .usage = sameS(256, 0, 124, 124, 124, 126, 126) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_d1_f32lat", .symbol = "hy4c2_expand_d1_f32lat", .image = "hy4_mla_c2", .threads = 128, .dynamic = 27136 }, .usage = sameS(128, 0, 109, 109, 109, 118, 118) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_d2_f32lat", .symbol = "hy4c2_expand_d2_f32lat", .image = "hy4_mla_c2", .threads = 128, .dynamic = 51712, .opt_in = true }, .usage = sameS(128, 0, 113, 113, 113, 125, 125) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_d4k64s2_f32lat", .symbol = "hy4c2_expand_d4k64s2_f32lat", .image = "hy4_mla_c2", .threads = 128, .dynamic = 27136 }, .usage = sameS(128, 0, 94, 94, 94, 86, 86) },
    .{ .launch = .{ .name = "hy4_mla_c2 hy4c2_expand_p_f32lat", .symbol = "hy4c2_expand_p_f32lat", .image = "hy4_mla_c2", .threads = 256, .dynamic = 46080 }, .usage = sameS(256, 0, 128, 128, 128, 124, 124) },
};

test "Hy4 paged MLA CUDA: every launch is in the live registry and fits every SM" {
    for (cuda) |r| {
        var found: usize = 0;
        for (registry.all_cuda) |active| found += @intFromBool(std.mem.eql(u8, active.launch.name, r.launch.name));
        try std.testing.expectEqual(@as(usize, 1), found);
        try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
    }
}
