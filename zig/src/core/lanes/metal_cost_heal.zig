//! Plan warm probes before scratch setup; retain only measured, consistent lower prices.
const std = @import("std");
pub const budget_ms: f64 = 200;
pub const Stats = struct { reads: u32 = 0, ms: f64 = 0, lowered: u32 = 0 };
pub const Local = struct {
    pub fn decide(_: @This(), mine: bool) !bool {
        return mine;
    }
};

pub fn valid(ms: []const f64, widths: []const u32) bool {
    if (ms.len == 0 or ms.len != widths.len) return false;
    for (ms, widths, 0..) |m, w, i| {
        if (!(m > 0) or !std.math.isFinite(m) or w == 0) return false;
        if (i > 0 and w <= widths[i - 1]) return false;
    }
    return true;
}

pub const Plan = struct {
    at: [3]usize = undefined,
    n: usize = 0,
    fn add(p: *Plan, i: usize) void {
        for (p.at[0..p.n]) |old| if (old == i) return;
        if (p.n < p.at.len) {
            p.at[p.n] = i;
            p.n += 1;
        }
    }
    pub fn widest(p: Plan, widths: []const u32) u32 {
        var rows: u32 = 0;
        for (p.at[0..p.n]) |i| rows = @max(rows, widths[i]);
        return rows;
    }
};

/// Empty means no scratch setup; one useful read per selected entry, in increasing width order.
pub fn select(ms: []const f64, widths: []const u32, cursor: u64) Plan {
    var p: Plan = .{};
    if (!valid(ms, widths) or ms[ms.len - 1] >= budget_ms) return p;
    p.add(@intCast(cursor % widths.len));
    const tile = 16 * (1 + cursor % 3);
    var below: usize = 0;
    var above = widths.len - 1;
    for (widths, 0..) |w, i| {
        if (w <= tile) below = i;
        if (w > tile) {
            above = i;
            break;
        }
    }
    p.add(below);
    p.add(above);
    std.mem.sort(usize, p.at[0..p.n], {}, std.sort.asc(usize));
    var fits: Plan = .{};
    var estimate: f64 = 0;
    for (p.at[0..p.n]) |i| if (estimate + ms[i] <= budget_ms) {
        fits.add(i);
        estimate += ms[i];
    };
    return fits;
}

fn kind(comptime T: type) type {
    return switch (@typeInfo(T)) { .pointer => |p| p.child, else => T };
}

fn charge(timer: anytype, stats: *Stats, measured: f64) void {
    if (comptime @hasDecl(kind(@TypeOf(timer)), "elapsedMs")) {
        stats.ms = timer.elapsedMs();
        if (!(stats.ms >= 0) or !std.math.isFinite(stats.ms)) stats.ms = budget_ms;
    } else stats.ms += if (measured > 0 and std.math.isFinite(measured)) measured else budget_ms;
}

/// Three bit-encoded indices replace one collective per table entry per slot.
fn broadcast(p: Plan, len: usize, agree: anytype, lead: bool) !Plan {
    var selected: Plan = .{};
    for (0..3) |slot| {
        const mine = if (lead and slot < p.n) p.at[slot] + 1 else 0;
        var value: usize = 0;
        var bit: usize = 1;
        while (bit <= len) : (bit *= 2) {
            if (try agree.decide(lead and mine & bit != 0)) value |= bit;
        }
        if (value > 0 and value <= len) selected.add(value - 1);
    }
    return selected;
}

/// Setup and each complete read count against elapsed wall time; one unexpected slow operation can overrun it.
pub fn prepared(timer: anytype, ms: []f64, widths: []const u32, cursor: *u64, floor: f64, agree: anytype, lead: bool) !Stats {
    var stats: Stats = .{};
    if (comptime @hasDecl(kind(@TypeOf(timer)), "start")) timer.start();
    const mine = if (lead) select(ms, widths, cursor.*) else Plan{};
    if (!try agree.decide(lead and mine.n > 0)) return stats;
    const selected = try broadcast(mine, ms.len, agree, lead);
    if (comptime @hasDecl(kind(@TypeOf(timer)), "prepare")) try timer.prepare(selected, widths);
    if (comptime @hasDecl(kind(@TypeOf(timer)), "elapsedMs")) charge(timer, &stats, 0);
    for (selected.at[0..selected.n]) |i| {
        if (comptime @hasDecl(kind(@TypeOf(timer)), "elapsedMs")) charge(timer, &stats, 0);
        if (!try agree.decide(lead and stats.ms + ms[i] <= budget_ms)) break;
        const measured = try timer.time(i);
        stats.reads += 1;
        charge(timer, &stats, measured);
        if (!lead or !(measured > 0) or !std.math.isFinite(measured) or measured >= ms[i]) continue;
        var predecessor = floor;
        for (ms[0..i]) |m| predecessor = @max(predecessor, m);
        if (measured < predecessor) continue;
        ms[i] = measured;
        stats.lowered += 1;
    }
    if (lead and stats.reads > 0) cursor.* +%= 1;
    return stats;
}

pub fn synced(timer: anytype, ms: []f64, widths: []const u32, cursor: *u64, agree: anytype, lead: bool) !Stats {
    return prepared(timer, ms, widths, cursor, 0, agree, lead);
}

pub fn local(timer: anytype, ms: []f64, widths: []const u32, cursor: *u64) !Stats {
    return synced(timer, ms, widths, cursor, Local{}, true);
}
