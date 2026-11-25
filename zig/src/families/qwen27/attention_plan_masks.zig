//! A test check of attention plans: whether the prefix kernel would read a key some live row must not see.
const std = @import("std");
const plan = @import("attention_plan.zig");
const Plan = plan.Plan;
const global_words = plan.global_words;
const stream_words = plan.stream_words;
const node_words = plan.node_words;

/// Whether q27_attn_prefix, reading this plan, has a key in its range that some live row of the tile must not see.
pub fn prefixMasks(p: Plan) bool {
    const g = p.geometry.group();
    for (0..p.tiles) |t| {
        const st: usize = @intCast(p.tile_stream[t]);
        const at = global_words + st * stream_words;
        const rows: usize = @intCast(p.meta[at + 2]);
        const lt = t - @as(usize, @intCast(p.meta[at + 3]));
        const first: usize = @intCast(p.meta[at + 6]);
        const start: u32 = @intCast(p.meta[at + 4]);
        const limit: u32 = @intCast(p.meta[at + 9]);
        for (0..@as(usize, @intCast(p.meta[at + 1]))) |c| {
            for ([_]bool{ false, true }) |widest| {
                var wa: u32 = std.math.maxInt(u32);
                var wb: u32 = 0;
                for (0..16) |i| {
                    const ck: u32 = @intCast(p.nodes[(first + @min(lt * 16 + i, g * rows - 1) / g) * node_words + 2]);
                    wa = @min(wa, ck);
                    wb = @max(wb, ck);
                }
                const w = if (widest) wb else wa;
                var seen: u32 = std.math.maxInt(u32);
                for (0..16) |i| {
                    const q = lt * 16 + i;
                    if (q >= g * rows) continue;
                    const ck: u32 = @intCast(p.nodes[(first + q / g) * node_words + 2]);
                    seen = @min(seen, if (ck == w) @min(limit, start + @as(u32, @intCast(q / g)) + 1) else 0);
                }
                const kbeg: u32 = @intCast(c * w);
                if (@max(seen, kbeg) < @min(kbeg + w, limit)) return true;
            }
        }
    }
    return false;
}
