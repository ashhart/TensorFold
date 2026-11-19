//! Prompt attention uses eight simdgroups and a 16 KiB score panel, in contiguous and paged libraries.
const M = @import("limits.zig").MetalLaunch;
const std = @import("std");

pub const forms = [_]struct { group: u32, kv_heads: u32, tag: []const u8 }{ .{ .group = 6, .kv_heads = 4, .tag = " g6h4 " }, .{ .group = 6, .kv_heads = 2, .tag = " g6h2 " }, .{ .group = 16, .kv_heads = 2, .tag = " g16h2 " }, .{ .group = 4, .kv_heads = 4, .tag = " g4h4 " } };

pub const records = build();

fn build() [32]M {
    @setEvalBranchQuota(100_000);
    var out: [32]M = undefined;
    var i: usize = 0;
    for ([_]bool{ false, true }) |units| for ([_]bool{ false, true }) |paged| for (forms) |form| {
        const prefix = if (units) "q27_attn_prompt" else "q27_attn_prompt_sg";
        const label = std.fmt.comptimePrint("qwen27 prompt {s} {s} g{d}h{d}", .{ if (units) "m5" else "sg", if (paged) "paged" else "contiguous", form.group, form.kv_heads });
        for ([_][]const u8{ std.fmt.comptimePrint("{s}_8", .{prefix}), std.fmt.comptimePrint("{s}_masked_8", .{prefix}) }) |fun| {
            out[i] = .{
                .name = std.fmt.comptimePrint("{s} {s}", .{ label, fun }),
                .threads = 256,
                .tg_memory = 16384,
                .offline = !units,
                .adapts = !units,
            };
            i += 1;
        }
    };
    return out;
}
