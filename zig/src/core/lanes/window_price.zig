//! Generic post-block chain probabilities and surplus price state; target verification never reads this state.
const std = @import("std");
const Table = @import("table.zig").Table;
const Bin = struct { predicted: f64 = 0, landed: f64 = 0, latest: f64 = 0, observations: usize = 0 };
pub const State = struct {
    bins: std.ArrayList(Bin) = .empty,
    offered: usize = 0,
    token_rate: f64 = 0,
    modeled_ms: f64 = 0,
    fn deinit(s: *State, a: std.mem.Allocator) void {
        s.bins.deinit(a);
    }
};
fn probability(p: f64) f64 {
    return if (std.math.isFinite(p)) std.math.clamp(p, 0, 1) else 0;
}

/// Backend chances are cumulative; calibrate their conditional edges and rebuild each prefix's survival.
pub fn probabilities(e: anytype, stream: anytype, raw: []const f64) ![]const f64 {
    const slot = try e.priced_states.getOrPut(e.gpa, stream);
    if (!slot.found_existing) slot.value_ptr.* = .{};
    const s = slot.value_ptr;
    while (s.bins.items.len < raw.len) try s.bins.append(e.gpa, .{});
    s.offered = raw.len;
    s.modeled_ms = 0;
    const out = try e.arena.allocator().alloc(f64, raw.len);
    var previous: f64 = 1;
    var reach: f64 = 1;
    for (raw, out, s.bins.items[0..raw.len]) |p, *v, *bin| {
        const cumulative = @min(previous, probability(p));
        const conditional = if (previous > 0) cumulative / previous else 0;
        bin.latest = conditional;
        const factor = (8 + bin.landed) / (8 + bin.predicted);
        reach *= std.math.clamp(conditional * factor, 0, 0.995);
        v.* = reach;
        previous = cumulative;
    }
    return out;
}

/// The stream's modeled throughput, never below the table's plain one-row rate.
pub fn tokenRate(e: anytype, stream: anytype, costs: *const Table) ?f64 {
    const s = e.priced_states.getPtr(stream) orelse return null;
    const plain = costs.get(1) orelse return null;
    if (!(plain > 0) or !std.math.isFinite(plain)) return null;
    return @max(1 / plain, s.token_rate);
}

/// Save the selected verification price before a live observation changes its table.
pub fn selected(e: anytype, stream: anytype, count: usize, costs: *const Table) void {
    const s = e.priced_states.getPtr(stream) orelse return;
    s.modeled_ms = costs.get(@intCast(count + 1)) orelse 0;
}

/// A conditional edge is observed only after its parent landed, and only within the actually verified prefix.
pub fn observe(e: anytype, stream: anytype, offered: usize, kept: usize, may_reject: bool) void {
    const s = e.priced_states.getPtr(stream) orelse return;
    const n = @min(offered, s.offered);
    const accepted = @min(kept, n);
    for (s.bins.items[0..n], 0..) |*bin, j| {
        if (j > accepted or (j == accepted and !may_reject)) break;
        bin.predicted = 0.95 * bin.predicted + bin.latest;
        bin.landed = 0.95 * bin.landed + @as(f64, @floatFromInt(@intFromBool(j < accepted)));
        bin.observations += 1;
    }
    if (s.modeled_ms > 0 and std.math.isFinite(s.modeled_ms)) {
        const rate = @as(f64, @floatFromInt(accepted + 1)) / s.modeled_ms;
        s.token_rate = if (s.token_rate > 0) s.token_rate + 0.25 * (rate - s.token_rate) else rate;
    }
    s.offered = 0;
    s.modeled_ms = 0;
}
pub fn release(e: anytype, stream: anytype) void {
    if (e.priced_states.fetchRemove(stream)) |entry| {
        var s = entry.value;
        s.deinit(e.gpa);
    }
}
pub fn deinit(e: anytype) void {
    var it = e.priced_states.valueIterator();
    while (it.next()) |s| s.deinit(e.gpa);
    e.priced_states.deinit(e.gpa);
}
