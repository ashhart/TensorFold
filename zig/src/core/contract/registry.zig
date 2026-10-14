//! The registered launches gate 5 checks: each one's threads and memory, and the limits its compiles recorded.
const std = @import("std");
const limits = @import("limits.zig");

const M = limits.MetalLaunch;

/// Metal pipelines as the engine launches them; tg_memory is the most any chip reflects, m1 and m2 the compiled limits.
pub const metal = [_]M{
    .{ .name = "softmax tf_softmax_bf16", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "rms tf_rms_norm_bf16", .threads = 1024, .tg_memory = 144, .m1 = 1024, .m2 = 1024 },
    .{ .name = "top-k tf_bf16_topk", .threads = 256, .tg_memory = 4096, .m1 = 768, .m2 = 768 },
    .{ .name = "top-k tf_bf16_topk_max", .threads = 256, .tg_memory = 4112, .m1 = 1024, .m2 = 1024 },
    .{ .name = "top-k tf_bf16_topk_span", .threads = 256, .tg_memory = 4096, .m1 = 704, .m2 = 704 },
    .{ .name = "top-k tf_bf16_topk_merge", .threads = 256, .tg_memory = 4096, .m1 = 1024, .m2 = 1024 },
    .{ .name = "row projection q4 g64 tf_row_projection_q4_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q6 g32 tf_row_projection_q6_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q8 g64 tf_row_projection_q8_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "row projection q3 g32 tf_row_projection_q3_f32", .threads = 64, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "affine_mm q4 g64 tf_affine_row_sums", .threads = 64, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "affine_mm q4 g64 tf_affine_mm", .threads = 128, .tg_memory = 9728 },
    .{ .name = "affine_mm q8 g32 tf_affine_mm", .threads = 128, .tg_memory = 10240 },
    .{ .name = "simd_qmm tf_sq_mma_16_4_1", .threads = 512, .tg_memory = 16384, .m1 = 768, .m2 = 768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_16_2_2", .threads = 512, .tg_memory = 16384, .m1 = 704, .m2 = 704, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_1", .threads = 256, .tg_memory = 8192, .m1 = 768, .m2 = 768, .adapts = true },
    .{ .name = "simd_qmm tf_sq_mma_8_4_2", .threads = 256, .tg_memory = 16384, .m1 = 576, .m2 = 576, .adapts = true },
    .{ .name = "lane projection q6 g32 sk1 tf_lane", .threads = 32, .tg_memory = 3072 },
    .{ .name = "lane projection q6 g32 sk4 tf_lane", .threads = 128, .tg_memory = 10240 },
};

/// Gate 5's planted Metal controls: a launch past M1/M2's compiled limit (#708's class), 36 KiB of threadgroup memory.
pub const metal_controls = [_]M{
    .{ .name = "control: row projection q4 g64 at 1024 threads tf_row_projection_q4_f32", .threads = 1024, .tg_memory = 0, .m1 = 512, .m2 = 512 },
    .{ .name = "control: 36 KiB tf_ctl_big_tg", .threads = 256, .tg_memory = 36 * 1024 },
};

/// Every Metal launch's problems on every family we serve, printed; returns how many.
pub fn checkMetal(a: std.mem.Allocator, list: []const M, quiet: bool) !usize {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    var n: usize = 0;
    for (list) |l| for (limits.apple) |f| {
        n += try limits.metal(l, f, &out, a);
    };
    if (!quiet and out.items.len > 0) std.debug.print("{s}", .{out.items});
    return n;
}

test "the registered Metal launches fit every Apple family, and each planted control breaks one" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(usize, 0), try checkMetal(a, &metal, false));
    for (metal_controls) |l| try std.testing.expect(try checkMetal(a, &.{l}, true) > 0);
}
