//! Lower-only warm spot checks rotate across the table and around kernel tile boundaries.
const std = @import("std");
pub const budget_ms: f64 = 200;
pub const Stats = struct { reads: u32 = 0, ms: f64 = 0, lowered: u32 = 0 };
pub const Local = struct {
    pub fn decide(_: @This(), mine: bool) !bool { return mine; }
};

pub fn valid(ms: []const f64, widths: []const u32) bool {
    if (ms.len == 0 or ms.len != widths.len) return false;
    for (ms, widths, 0..) |m, w, i| {
        if (!(m > 0) or !std.math.isFinite(m) or w == 0) return false;
        if (i > 0 and w <= widths[i - 1]) return false;
    }
    return true;
}

const Plan = struct {
    at: [3]usize = undefined,
    n: usize = 0,
    fn add(p: *Plan, i: usize) void {
        for (p.at[0..p.n]) |old| if (old == i) return;
        if (p.n < p.at.len) { p.at[p.n] = i; p.n += 1; }
    }
};

fn plan(widths: []const u32, cursor: u64) Plan {
    var p: Plan = .{};
    p.add(@intCast(cursor % widths.len)); // progress gets the first share of the time budget
    const tile = 16 * (1 + cursor % 3);
    var edge = widths.len - 1;
    for (widths, 0..) |w, i| if (w >= tile) { edge = i; break; };
    p.add(edge);
    if (edge > 0) p.add(edge - 1);
    return p;
}

fn charge(s: *Stats, ms: f64) void {
    s.reads += 1;
    s.ms += if (ms > 0 and std.math.isFinite(ms)) ms else budget_ms;
}

/// A single warm-up and at most two reads at three entries; only the leader chooses a paired schedule.
pub fn synced(timer: anytype, ms: []f64, widths: []const u32, cursor: *u64, agree: anytype, lead: bool) !Stats {
    var stats: Stats = .{};
    if (!try agree.decide(lead and valid(ms, widths))) return stats;
    // Estimate before the warm-up too: an expensive table may consume no extra GPU work on a warm load.
    if (!try agree.decide(lead and ms[ms.len - 1] < budget_ms)) return stats;
    charge(&stats, try timer.time(ms.len - 1));
    const selected = if (lead) plan(widths, cursor.*) else Plan{};
    for (0..3) |slot| {
        // Broadcasting the chosen indices as flags keeps both ranks on the same collectives.
        for (0..ms.len) |i| {
            if (!try agree.decide(lead and slot < selected.n and selected.at[slot] == i)) continue;
            var lowered = false;
            for (0..2) |_| {
                if (!try agree.decide(lead and stats.ms + ms[i] <= budget_ms)) break;
                const measured = try timer.time(i);
                charge(&stats, measured);
                if (measured > 0 and std.math.isFinite(measured) and measured < ms[i]) {
                    ms[i] = measured;
                    lowered = true;
                }
            }
            stats.lowered += @intFromBool(lowered);
        }
    }
    if (lead) cursor.* +%= 1;
    return stats;
}

pub fn local(timer: anytype, ms: []f64, widths: []const u32, cursor: *u64) !Stats {
    return synced(timer, ms, widths, cursor, Local{}, true);
}
