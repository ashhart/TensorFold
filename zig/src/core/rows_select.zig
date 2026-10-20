//! Select a compatible disk restore by feasible memory and estimated time saved over the RAM prefix.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const rows = @import("rows_store.zig");

pub const Choice = struct {
    disk: ?usize = null,
    restore: ?usize = null,
    read_seconds: f64 = 0,
};

pub fn choose(s: *const pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, have: ?*pc.Entry) Choice {
    const r = s.disk orelse return .{};
    const room = capacity(s, have) orelse return .{};
    const from = if (have) |e| e.at else 0;
    var out: Choice = .{};
    var best_saved: f64 = 0;
    for (r.states.items, 0..) |*state, i| {
        if (s.family.vtable.bytes(s.family.ptr, state.at) > room) continue;
        if (!r.matches(state, prompt, starts, spans)) continue;
        if (out.disk == null or state.at > r.states.items[out.disk.?].at) out.disk = i;
        if (state.at <= from) continue;
        if (savedSeconds(r, i, from)) |saved| {
            if (!(saved > 0)) continue;
            if (out.restore != null and (saved < best_saved or
                (saved == best_saved and state.at <= r.states.items[out.restore.?].at))) continue;
            best_saved = saved;
        } else if (out.restore != null and state.at <= r.states.items[out.restore.?].at) continue;
        out.restore = i;
        out.read_seconds = readSeconds(r, i) orelse 0;
    }
    if (out.restore) |i| out.disk = i;
    return out;
}

fn capacity(s: *const pc.Store, have: ?*pc.Entry) ?u64 {
    var reclaimable: u64 = 0;
    var protected: usize = 0;
    const f = s.family.vtable;
    for (s.entries.items) |e| {
        if (e == have or e.borrowers > 0) protected += 1 else if (f.spare == null or f.trim != null) {
            reclaimable +|= e.bytes;
        }
    }
    const fixed = (s.held -| reclaimable) +| if (f.trim == null) s.spare() else 0;
    if (protected >= s.most or fixed > s.budget) return null;
    return s.budget - fixed;
}

pub fn knownRate(value: f64) bool {
    return std.math.isFinite(value) and value > 0;
}

pub fn readSeconds(r: *const rows.Rows, i: usize) ?f64 {
    if (!knownRate(r.read_rate)) return null;
    const state = r.states.items[i];
    const seconds = @as(f64, @floatFromInt(state.own +| state.rows)) / r.read_rate;
    return if (std.math.isFinite(seconds)) seconds else null;
}

pub fn savedSeconds(r: *const rows.Rows, i: usize, from: u32) ?f64 {
    if (!knownRate(r.prefill_rate) or !knownRate(r.read_rate)) return null;
    const read = readSeconds(r, i) orelse return -std.math.inf(f64);
    const prefill = @as(f64, @floatFromInt(r.states.items[i].at -| from)) / r.prefill_rate;
    return if (std.math.isFinite(prefill)) prefill - read else -std.math.inf(f64);
}

test {
    _ = @import("rows_select_test.zig");
}
