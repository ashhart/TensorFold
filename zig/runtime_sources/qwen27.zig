//! Qwen27 shares the loader's grouped, paged and SIMD attention assemblers.
const std = @import("std");
const ks = @import("qwen_runtime_sources");
const Set = @import("set.zig").Set;
pub fn add(s: *Set) !void {
    inline for (.{ "glue", "copy", "moe" }) |name|
        try s.raw("qwen27", name, @field(ks, name));
    try s.rawFrom("qwen27", "moe-rows", &.{ @import("kernel_sources").core_row_projection, ks.moe_rows }, @import("../src/families/qwen27/moe_gpu.zig").rows_source);
    const quant = @import("../src/families/qwen27/quant_gpu.zig");
    for (std.enums.values(quant.Kind)) |kind| {
        if (kind == .projection) continue;
        const origin = switch (kind) {
            .projection => unreachable,
            .xsum => ks.xsum,
            .norm => ks.norm_xs,
            .input_norm => ks.norm_input,
            .mlp => ks.mlp_xs,
            .ane_input, .mlp_ane, .add_ane => ks.mlp_ane,
        };
        try s.rawFrom("qwen27", @tagName(kind), &.{origin}, try quant.specializedSource(s.a, .{ .kind = kind, .k = 512, .n = 1024, .tile = 256 }));
        if (kind == .norm or kind == .input_norm or kind == .mlp)
            try s.rawFrom("qwen27", try std.fmt.allocPrint(s.a, "{s}-no-xs", .{@tagName(kind)}), &.{origin}, try quant.specializedSource(s.a, .{ .kind = kind, .k = 512, .n = 1024, .xs = false }));
    }
    const attention = @import("../src/families/qwen27/simd_attention.zig");
    for ([_]bool{ false, true }) |paged| for ([_]bool{ false, true }) |prompt| for ([_][2]u32{ .{ 6, 4 }, .{ 4, 4 }, .{ 8, 4 } }) |g| {
        const origins = try attentionOrigins(s, prompt, paged);
        try s.rawFrom("qwen27", try std.fmt.allocPrint(s.a, "attention-paged{}-prompt{}-g{d}", .{ paged, prompt, g[0] }), origins, try attention.grouped(s.a, s.units, prompt, g[0], g[1], paged));
    };
    if (!s.units) {
        for ([_]bool{ false, true }) |paged| for ([_][2]u32{ .{ 6, 4 }, .{ 4, 4 }, .{ 8, 4 } }) |g| {
            const origins = try attentionOrigins(s, false, paged);
            try s.rawFrom("qwen27", try std.fmt.allocPrint(s.a, "attention-join-paged{}-g{d}", .{ paged, g[0] }), origins, try attention.joined(s.a, false, false, g[0], g[1], paged));
        };
    }
    const ksrc = @import("kernel_sources");
    try s.raw("sensenova", "mhc", ksrc.core_sensenova_mhc);
    try s.fragments("sensenova", "bf16_linear", ksrc.qwen27_bf16_linear);
    try s.fragments("sensenova", "awq_w4a16", ksrc.qwen27_awq_w4a16);
}

fn attentionOrigins(s: *Set, prompt: bool, paged: bool) ![]const []const u8 {
    const twins = @import("kernel_sources").qwen27_paged;
    var origins: std.ArrayList([]const u8) = .empty;
    try origins.append(s.a, ks.glue);
    try origins.append(s.a, if (paged) twins.attention_mpp else ks.attention_mpp);
    try origins.append(s.a, if (paged) twins.attention_io else ks.attention_io);
    if (paged) try origins.append(s.a, @import("kernel_sources").kv_pages);
    if (s.units or prompt) try origins.append(s.a, if (paged) twins.attention_prompt else ks.attention_prompt);
    if (!s.units and prompt) try origins.append(s.a, if (paged) twins.attention_prompt_sg else ks.attention_prompt_sg);
    return origins.items;
}
