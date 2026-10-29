//! Nemotron 3.5 Bolt's MLX-exact ops as nemotron/bolt.zig launches them (gate 5; gates 1-4 in tests/contract/metal_bolt.zig).
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "bolt gemv tf_gemv_bf16_bm4_tm4", .threads = 128, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt gemv tf_gemv_bf16_bm8_tm4", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // one threadgroup of E threads a row: Bolt's largest layer has 496 experts (the kernel's arrays hold 1,024)
    .{ .name = "bolt route tf_route_topk", .threads = 496, .tg_memory = 8448, .m1 = 1024, .m2 = 1024 },
    // a thread a latent column, 256 a group
    .{ .name = "bolt combine tf_moe_combine_rows_lanes8", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    // the gated group norm: a group of 1,024 a threadgroup of 256 (outv 2 KiB + 8 partials); one per MP
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_16", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_32", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_48", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_64", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_80", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_96", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_112", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt group norm tf_nemotron_group_norm_xs_bolt_128", .threads = 256, .tg_memory = 2080, .m1 = 1024, .m2 = 1024 },
    // the residual add + RMS norm at hidden 4,096: 512 threads, xb 8 KiB + 16 partials
    .{ .name = "bolt add norm tf_nemotron_add_norm_xs_bolt", .threads = 512, .tg_memory = 8256, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt add norm tf_nemotron_add_norm_moe_xs_bolt", .threads = 512, .tg_memory = 8256, .m1 = 1024, .m2 = 1024 },
    // the Mamba conv: a thread a channel and row, 256 a group; the scan: 32 lanes x 8 channels, a lane's states in registers
    .{ .name = "bolt mamba conv tf_tree_conv_bolt", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "bolt mamba scan tf_tree_scan_bolt", .threads = 256, .tg_memory = 0, .m1 = 1024, .m2 = 1024 },
};

test "Bolt's op launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
