//! Qwen27 shares the loader's grouped, paged and SIMD attention assemblers.
const std = @import("std");
const ks = @import("qwen_runtime_sources");
const Set = @import("set.zig").Set;
pub fn add(s: *Set) !void {
    inline for (.{ "glue", "copy", "moe" }) |name|
        try s.raw("qwen27", name, @field(ks, name));
    try s.raw("qwen27", "moe-rows", @import("../src/families/qwen27/moe_gpu.zig").rows_source);
    const quant = @import("../src/families/qwen27/quant_gpu.zig");
    for (std.enums.values(quant.Kind)) |kind| {
        if (kind == .projection) continue;
        try s.raw("qwen27", @tagName(kind), try quant.specializedSource(s.a, .{ .kind = kind, .k = 512, .n = 1024, .tile = 256 }));
        if (kind == .norm or kind == .input_norm or kind == .mlp)
            try s.raw("qwen27", try std.fmt.allocPrint(s.a, "{s}-no-xs", .{@tagName(kind)}), try quant.specializedSource(s.a, .{ .kind = kind, .k = 512, .n = 1024, .xs = false }));
    }
    const attention = @import("../src/families/qwen27/simd_attention.zig");
    for ([_]bool{ false, true }) |paged| for ([_]bool{ false, true }) |prompt| for ([_][2]u32{ .{ 6, 4 }, .{ 4, 4 }, .{ 8, 4 } }) |g|
        try s.raw("qwen27", try std.fmt.allocPrint(s.a, "attention-paged{}-prompt{}-g{d}", .{ paged, prompt, g[0] }), try attention.grouped(s.a, s.units, prompt, g[0], g[1], paged));
}
