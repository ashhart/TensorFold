//! The engine's GLM_RANKS diagnostics: where a missed draft's target ranked among the head's choices.
const std = @import("std");
const wts = @import("weights.zig");
const cfg = @import("config.zig");

/// Where `t` falls among MTP depth m + 1's choices (its logits row m): 1 for the 2nd, 2 the 3rd, 3 later or none.
pub fn rankAt(c: *const cfg.Config, draft_vocab: u32, m_logits: wts.Ref, m: usize, t: u32) usize {
    if (t >= draft_vocab) return 3;
    const v: [*]const u16 = @ptrCast(@alignCast(m_logits.addr()));
    const row = v[m * c.vocab ..][0..draft_vocab];
    const lt: f32 = @bitCast(@as(u32, row[t]) << 16);
    var above: usize = 0; // choices before t: higher logits, or equal ones at lower ids
    for (row, 0..) |h, i| {
        const f: f32 = @bitCast(@as(u32, h) << 16);
        if (i != t and (f > lt or (f == lt and i < t))) above += 1;
        if (above >= 3) return 3;
    }
    return @max(above, 1); // the draft holds the first place even when t ties it
}

/// The top two logits' difference in row `row` of the last head's logits (bf16).
pub fn margin(c: *const cfg.Config, logits: wts.Ref, row: u32) f32 {
    if (c.tp > 1) return std.math.nan(f32); // TP2: this Mac holds half of each row's logits
    const v: [*]const u16 = @ptrCast(@alignCast(logits.addr()));
    var top = [2]f32{ -std.math.inf(f32), -std.math.inf(f32) };
    for (v[@as(usize, row) * c.vocab ..][0..c.vocab]) |h| {
        const f: f32 = @bitCast(@as(u32, h) << 16);
        if (f > top[0]) {
            top[1] = top[0];
            top[0] = f;
        } else if (f > top[1]) top[1] = f;
    }
    return top[0] - top[1];
}
