//! The boot calibration over a CPU twin on a virtual clock: the pass is timed the way a round runs it, warm.
const std = @import("std");
const lanes = @import("lanes");
const iface = @import("draft/iface.zig");
const twin = @import("draft/twin.zig");
const cm = @import("draft/calib_measure.zig");
const costs_mod = @import("draft/costs.zig");
const lanes_calib = @import("calib_lanes.zig");

const gpa = std.testing.allocator;
const t_ = std.testing;

/// The twin on a virtual clock: windows cost by rows, the first ingest after one pays a cold start.
const Fake = struct {
    t: *twin.Twin,
    d: *twin.Drafter,
    now: u64 = 0,
    cold: bool = false,
    cold_ms: f64,
    plen: [4]u64 = @splat(0),

    fn ns(ms: f64) u64 {
        return @intFromFloat(ms * 1e6);
    }

    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const f = of(ptr);
        f.plen[slot] = ids.len;
        const ring = f.d.shape.ring();
        for (0..ids.len) |pos| f.d.tags[slot][pos % ring] = @intCast(pos); // the prompt's rows are in the drafter's ring
        return f.t.target().prefill(slot, ids, smp, draw);
    }

    fn window(ptr: *anyopaque, segs: []const iface.Segment, out: [][]u32) anyerror!void {
        const f = of(ptr);
        var rows: u64 = 0;
        for (segs) |s| rows += s.tokens.len;
        f.now += ns(28.0 + 6.6 * @as(f64, @floatFromInt(rows - 1)) + (if (rows > 16) @as(f64, 28.0) else 0.0));
        f.cold = true;
        return f.t.target().window(segs, out);
    }

    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        return of(ptr).t.target().keep(slot, path);
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        return of(ptr).t.target().taps(slot);
    }

    fn release(ptr: *anyopaque, slot: u32) void {
        of(ptr).t.target().release(slot);
    }

    fn ingest(ptr: *anyopaque, slot: u32, start: u64, tp: iface.Taps, rows: []const u32) anyerror!void {
        const f = of(ptr);
        f.now += ns(0.7 + (if (f.cold) f.cold_ms else 0.0));
        f.cold = false;
        return f.d.pass().ingest(slot, start, tp, rows);
    }

    fn propose(ptr: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const f = of(ptr);
        f.now += ns(6.0);
        return f.d.pass().propose(asks, out);
    }

    fn reset(ptr: *anyopaque, slot: u32) void {
        of(ptr).d.pass().reset(slot);
    }

    fn markov(ptr: *anyopaque) ?@import("draft/dspark.zig").Markov {
        return of(ptr).d.pass().markov();
    }

    fn mark(ptr: *anyopaque) anyerror!u64 {
        return of(ptr).now;
    }

    fn gather(_: *anyopaque, ours: []const i64, all: []i64) anyerror!void {
        @memcpy(all, ours);
    }

    fn pending(_: *anyopaque, _: u32) bool {
        return false;
    }

    fn drop(ptr: *anyopaque, slot: u32) anyerror!void {
        of(ptr).t.drop(slot);
    }

    fn feed(ptr: *anyopaque, slot: u32) anyerror!void {
        const f = of(ptr);
        const tp = f.t.target().taps(slot);
        const n: u32 = @intCast(@min(@min(tp.rows, f.plen[slot]), 3));
        if (n == 0) return;
        var rows: [3]u32 = undefined;
        for (rows[0..n], 0..) |*r, i| r.* = tp.rows - n + @as(u32, @intCast(i));
        try ingest(ptr, slot, f.plen[slot] - n, tp, rows[0..n]);
    }
};

fn measured(cold_ms: f64) !lanes_calib.Measured {
    const t = try twin.Twin.init(gpa, .{}, 4);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, 4);
    defer d.deinit();
    var f: Fake = .{ .t = t, .d = d, .cold_ms = cold_ms };
    const dev: lanes_calib.Device = .{
        .ctx = &f,
        .target = .{ .ptr = &f, .vtable = &.{ .prefill = Fake.prefill, .window = Fake.window, .keep = Fake.keep, .taps = Fake.taps, .release = Fake.release } },
        .pass = .{ .ptr = &f, .vtable = &.{ .ingest = Fake.ingest, .propose = Fake.propose, .reset = Fake.reset, .markov = Fake.markov } },
        .slots = 4,
        .limit = 1000,
        .rows = 16,
        .world = 1,
        .vocab = 48,
        .shape = d.shape,
        .mark = Fake.mark,
        .gather = Fake.gather,
        .has_pending = Fake.pending,
        .drop_window = Fake.drop,
        .ingest = Fake.feed,
        .wide = &cm.launch_wide,
        .base = cm.launch_bound,
    };
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1 + i % 30);
    return lanes_calib.measure(gpa, dev, &ids, cm.launch_bound, 64);
}

test "the pass is timed warm: a cold first ingest after the windows is not the table's pass" {
    var cold = try measured(55.0);
    defer cold.costs.deinit(gpa);
    var warm = try measured(0.0);
    defer warm.costs.deinit(gpa);
    // an ingest of 0.7 ms and a propose of 6 ms, however cold the first ingest after the windows is
    try t_.expectApproxEqAbs(@as(f64, 6.7), warm.costs.draft, 0.01);
    try t_.expectApproxEqAbs(warm.costs.draft, cold.costs.draft, 0.01);
    try t_.expectApproxEqAbs(@as(f64, 0.7), cold.ingest_ms, 0.01);
    try t_.expect(cold.ingest_first_ms > 50.0); // the cold start is seen and reported, not priced
    try t_.expect(cold.costs.plausible());
}

test "a pass dearer than twice a 1-row round is not a table to use" {
    var v = [_]f64{ 28.4, 37.5 };
    try t_.expect((costs_mod.Costs{ .verify = &v, .draft = 6.42 }).plausible());
    try t_.expect(!(costs_mod.Costs{ .verify = &v, .draft = 60.31 }).plausible());
    try t_.expect(!(costs_mod.Costs{ .verify = &v, .draft = std.math.nan(f64) }).plausible());
    try t_.expectError(error.ImplausibleCosts, costs_mod.Costs.parseJson(gpa, "{\"verify\": [28.4, 37.5], \"draft\": 105.0, \"slot\": 8.0}"));
    var ok = try costs_mod.Costs.parseJson(gpa, "{\"verify\": [28.4, 37.5], \"draft\": 6.4, \"slot\": 8.0}");
    ok.deinit(gpa);
}
