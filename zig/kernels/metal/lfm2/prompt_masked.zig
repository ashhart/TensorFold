//! Exceptional twin of the original prompt entry, in its own library. Finite source stays unchanged.
const std = @import("std");
const guard = @import("../threadgroup_source.zig");
pub const original = @embedFile("lfm2_attn_nax.metal");
pub const source = build(original);

pub fn build(comptime input: []const u8) []const u8 {
    return comptime blk: {
        @setEvalBranchQuota(20_000_000);
        const begin = std.mem.indexOf(u8, input, "[[kernel]]").?;
        const reservation = "[[max_total_threads_per_threadgroup(128)]]\n";
        _ = guard.normalize(input[0..begin] ++ reservation ++ input[begin..]);
        var body = edit(input[begin..], "void lfm2_attn_nax(", "void lfm2_attn_masked(");
        body = edit(body, "mma_16x32<false, false>(o[0], o[1], pb[h], v0, v1);", "lfm2_masked_pv(o[0], o[1], pb[h], v0, v1, home, j0 + 16 * h, count0, count1);");
        body = edit(body, "mma_16x32<false, false>(o[2], o[3], pb[h], v2, v3);", "lfm2_masked_pv(o[2], o[3], pb[h], v2, v3, home, j0 + 16 * h, count0, count1);");
        break :blk guard.normalize(input ++ @embedFile("prompt_masked.h") ++ reservation ++ body ++ @embedFile("../core/prompt_pick.metal") ++ @embedFile("prompt_pick.metal"));
    };
}
fn edit(comptime text: []const u8, comptime from: []const u8, comptime to: []const u8) []const u8 {
    if (std.mem.count(u8, text, from) != 1) @compileError("LFM2 masked prompt source no longer matches");
    const at = std.mem.indexOf(u8, text, from).?;
    return text[0..at] ++ to ++ text[at + from.len ..];
}
test "finite LFM2 prompt source remains byte-identical" {
    try std.testing.expect(std.mem.startsWith(u8, source, original));
    try std.testing.expect(std.mem.indexOf(u8, source, "void lfm2_attn_masked(") != null);
}
