//! Once-per-context suffix admission, with per-stream match bands and bounded source probes.
const std = @import("std");
const win = @import("windows.zig");
pub const evidence = 4;
pub const retry_every = 8;
pub const long_match = 20;
pub const Band = struct {
    judgments: u64 = 0,
    landed: f64 = 0,
    breaks: f64 = 0,
    choices: u64 = 0,
};
pub const State = struct {
    tokens: std.ArrayList(u32) = .empty,
    context: ?usize = null,
    round: u64 = 0,
    band: usize = 0,
    copied: bool = false,
    selected: bool = false,
    heads: u64 = 0,
    bands: [2]Band = .{ .{}, .{} },
};

pub fn enabled(e: anytype, s: anytype) bool {
    return e.cfg.copy_select and s.drafts and s.proposer != null and !s.proposer.?.vtable.priced and
        !e.cfg.root_copies and !e.cfg.context_copies and e.cfg.fill_lanes == 0 and s.grammar == null;
}

pub fn proposal(e: anytype, s: anytype, comptime query: anytype) !win.Copy {
    const st = &s.copy_select;
    if (st.context != s.context.items.len or st.round != s.rounds) {
        st.context = null;
        st.tokens.clearRetainingCapacity();
        const c = try query(e, s);
        try st.tokens.appendSlice(e.gpa, c.tokens);
        st.copied = !c.recalled and c.tokens.len > 0;
        st.selected = c.tokens.len > 0;
        if (st.copied) {
            st.band = @intFromBool(s.proposer.?.lastMatch() >= long_match);
            const b = &st.bands[st.band];
            b.choices +|= 1;
            if (b.choices % retry_every == 0) {
                // Probe the head while warming up, and retry a refused copy once the head is established.
                st.selected = if (b.judgments >= evidence and st.heads >= evidence) true else false;
            } else if (b.judgments >= evidence and st.heads >= evidence) {
                st.selected = !(try weak(e, s, c.tokens.len, b.*));
            }
        }
        st.context = s.context.items.len;
        st.round = s.rounds;
    }
    return .{ .tokens = if (st.selected) st.tokens.items else &.{}, .recalled = !st.copied };
}

// A copy is weak when its estimated kept tokens/ms loses to an available head depth; ties retain the copy.
fn weak(e: anytype, s: anytype, n: usize, b: Band) !bool {
    const costs = if (!e.alone and !e.cfg.shared_costs.empty()) &e.cfg.shared_costs else &e.cfg.family_costs;
    const copy_ms = costs.get(@intCast(n + 1)) orelse return false;
    if (copy_ms <= 0 or b.landed + b.breaks <= 0) return false;
    const rate = b.landed / (b.landed + b.breaks);
    var copy_expected: f64 = 1;
    var chance: f64 = 1;
    for (0..n) |_| {
        chance *= rate;
        copy_expected += chance;
    }
    const rates = try e.rule.rates(win.who(s));
    const cap = if (!e.alone and e.cfg.shared_drafts > 0) e.cfg.shared_drafts else e.cfg.most_drafts;
    const most = @min(cap, e.cfg.most_drafts, @as(u32, @intCast(@max(0, s.draftRoom() - 1))));
    const step = if (!e.alone) e.cfg.k.shared_step_ms orelse e.cfg.mtp_step_ms else e.cfg.mtp_step_ms;
    var expected: f64 = 1;
    chance = 1;
    for (0..most) |i| {
        chance *= rates[@min(i, rates.len - 1)];
        expected += chance;
        const forward = costs.get(@intCast(i + 2)) orelse continue;
        const ms = forward + step * @as(f64, @floatFromInt(i + 1));
        if (ms > 0 and expected / ms > copy_expected / copy_ms) return true;
    }
    return false;
}

pub fn observe(e: anytype, s: anytype, recalled: bool, kept: usize, offered: usize) void {
    if (!enabled(e, s) or recalled or offered == 0 or !s.copy_select.copied) return;
    const b = &s.copy_select.bands[s.copy_select.band];
    b.judgments +|= 1;
    b.landed = 0.9 * b.landed + @as(f64, @floatFromInt(kept));
    b.breaks = 0.9 * b.breaks + @as(f64, @floatFromInt(@intFromBool(kept < offered)));
}
