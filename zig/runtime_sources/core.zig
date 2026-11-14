//! Shared projection, routing and drafting libraries loaded by the Metal families.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const Set = @import("set.zig").Set;
const lane = @import("../src/core/lane_projection.zig");
const affine = @import("../src/core/affine_mm.zig");
pub fn add(s: *Set, device: mtl.Device) !void {
    inline for (.{ "core_row_projection", "core_exl3_projection", "core_exlr8_projection", "core_exlr8_dense", "core_pair", "core_pair_rows", "core_lane_order", "core_deltanet", "core_simd_qmm", "core_simd_qmm_bits", "core_grammar_mask" }) |name|
        try s.raw("core", name, @field(ks, name));
    try s.fragments("core", "binned-projection", ks.core_binned_projection);

    try s.fragments("core", "exlr8-prompt", ks.core_exlr8_projection ++ ks.core_exlr8_prompt);
    try s.fragments("core", "exlr8-dense-prompt", ks.core_exlr8_dense ++ ks.core_exlr8_dense_prompt);
    for ([_]u32{ 13, 15 }) |gen| try s.raw("core", "tile-qmm", @import("../src/core/tile_qmm.zig").source(gen));
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u16{ 32, 64, 128 }) |group| {
        for ([_]bool{ false, true }) |float_out| for (std.enums.values(affine.Glu)) |glu|
            try s.raw("core-affine", try std.fmt.allocPrint(s.a, "b{d}-g{d}-{}-{t}", .{ bits, group, float_out, glu }), try affine.source(device, s.a, .{ .bits = bits, .group = group, .out_f32 = float_out, .glu = glu }));
    };
    try s.fragments("core", "fragment-layout-probe", @import("../src/core/frags.zig").check_source);
    try s.raw("core", "counter-probe", ks.counter_probe);
    try s.raw("nemotron", "simd-layout-probe", @import("../src/families/nemotron/simd_attention.zig").check_source);
    try s.raw("qwen27", "simd-layout-probe", @import("../src/families/qwen27/simd_attention.zig").check_source);
    try projections(s);
    try s.raw("core", "bf16-topk", @import("bf16_topk_sources").text);
    try s.raw("core", "tree-round", @import("tree_round_sources").text);
    try s.raw("core", "tree-commit", @import("tree_commit_sources").text);
    try s.raw("core", "draft-ops", @import("draft_ops_sources").text);
}

fn projections(s: *Set) !void {
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        var l = lane.Layout{ .n = 256, .k = 512, .format = .{ .bits = bits, .group = group } };
        try s.fragments("core-lane", "plain", try lane.source(s.a, l));
        l.sk = 2;
        l.pf = 2;
        l.output = .f32;
        l.groups = .{ 1, 2 };
        l.ranges = &.{.{ 1, 2 }};
        l.y_stride = 512;
        try s.fragments("core-lane", "split-strided", try lane.source(s.a, l));
    };
    for ([_]bool{ false, true }) |reg| for ([_]bool{ false, true }) |cooperative| {
        if (reg and cooperative) continue;
        const l = lane.Layout{ .n = 256, .k = 512, .format = .{ .bits = 4, .group = 64 }, .reg = reg, .cooperative = cooperative, .precompute_sums = true };
        try s.fragments("core-lane", "prepared", try lane.source(s.a, l));
    };
    const attention = @import("../src/core/shared_attention.zig");
    for (std.enums.values(attention.Output)) |out| {
        const text = try attention.source(s.a, .{ .query_heads = 8, .kv_heads = 2, .head_dim = 128, .capacity = 4096, .output = out });
        try s.add("core", "shared-attention", text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    }
    const route = @import("../src/core/moe_route.zig");
    for ([_]route.Shape{ .{ .hidden = 4096, .experts = 288, .topk = 8 }, .{ .hidden = 4096, .experts = 128, .topk = 8, .out_bf16 = true }, .{ .hidden = 6144, .experts = 128, .topk = 4 } }) |shape|
        try s.raw("core", "route", try route.source(s.a, shape));
    const hc = @import("../src/core/hc.zig");
    try s.raw("core", "hc", try hc.source(s.a, .{ .width = 4096, .sinkhorn = 20, .eps_e9 = 1000 }));
}
