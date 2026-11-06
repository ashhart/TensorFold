//! Separate exceptional prompt pipelines, both P precision modes, and the GPU selector's actual launches.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;
pub const records = variants ++ [_]M{
    .{ .name = "nemotron masked prompt scan nemotron_prompt_scan", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "nemotron masked prompt scan paged nemotron_prompt_scan_paged", .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false },
    .{ .name = "nemotron masked prompt pick nemotron_prompt_pick", .threads = 1, .tg_memory = 0, .reserved = 1, .offline = false },
};
const variants = blk: {
    var out: [16]M = undefined;
    var at = 0;
    for ([_][]const u8{ "cut", "fp32" }) |mode| for ([_]bool{ false, true }) |page| {
        for ([_][]const u8{ "n_n", "n_t", "t_n", "t_t" }) |tag| {
            const name = "nemotron_attention_masked_" ++ (if (page) "paged_" else "") ++ tag ++
                (if (page) "" else "_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t");
            out[at] = .{ .name = "nemotron masked prompt " ++ mode ++ " " ++ name, .threads = 128, .tg_memory = 0, .reserved = 128, .offline = false };
            at += 1;
        }
    };
    break :blk out;
};
test "Nemotron masked prompts: both layouts, modes, alignments and every selector launch are recorded" {
    try std.testing.expectEqual(@as(usize, 19), records.len);
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
