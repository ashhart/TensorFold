//! The target's forward ms by round rows, timed before any stream opens: the depth rule's verify prices for a drafter.
const std = @import("std");
const lanes = @import("lanes");
const be = @import("backend.zig");
const fwd = @import("forward.zig");

const Cost = lanes.config.Cost;

/// Round widths timed (the lane core interpolates between them and scales past the widest).
pub const widths = [_]u32{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 };

/// Chains a round of `rows` rows takes: windows of at most be.max_window rows, as even as they split.
pub fn chainsFor(rows: u32) u32 {
    return (rows + be.max_window - 1) / be.max_window;
}

const Map = struct {
    m: *fwd.Model,
    x: *be.Kolibri,
    slot: usize,
    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const p: *const Map = @ptrCast(@alignCast(ctx));
        try p.m.mapPages(p.x.s, p.slot, ids[from..], from);
    }
};

/// Entry `i` is a round of widths[i] rows over free slots at position 0, timed once (cost_rule's timer shape).
const Timer = struct {
    x: *be.Kolibri,
    io: std.Io,
    tokens: []const u32,
    pub fn time(t: Timer, i: usize) !f64 {
        const w = widths[i];
        const n = chainsFor(w);
        var chains: [chainsFor(widths[widths.len - 1])]fwd.Chain = undefined;
        var at: usize = 0;
        for (chains[0..n], 0..) |*ch, c| {
            const rows = w / n + @intFromBool(c < w % n);
            ch.* = .{ .slot = c, .p0 = 0, .tokens = t.tokens[at..][0..rows] };
            at += rows;
        }
        var rows: [fwd.round_rows]usize = undefined;
        for (&rows, 0..) |*r, k| r.* = k;
        const t0 = std.Io.Clock.awake.now(t.io);
        try t.x.m.forward(t.x.ops(), chains[0..n], false, rows[0..w], t.x.logits.ptr);
        try t.x.taps.kept(t.x.gpa, t.x.m, t.x.s, chains[0..n], false);
        try t.x.s.synchronize();
        return @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(t.io)).toNanoseconds())) / 1e6;
    }
};

/// Widths a round over `slots` streams can take: each stream's window at most be.max_window rows.
pub fn widthsFor(slots: usize) usize {
    var n: usize = 0;
    while (n < widths.len and chainsFor(widths[n]) <= slots) n += 1;
    return n;
}

/// Every width the target's slots hold, fastest run each (lanes.cost_rule), pages returned after; the count timed.
pub fn time(x: *be.Kolibri, io: std.Io, out: *[widths.len]Cost) !usize {
    if (x.lanes.count() != 0) return error.TimingWithStreams;
    const n = widthsFor(x.m.slots);
    if (n == 0) return error.TimingSlots;
    const most = chainsFor(widths[n - 1]);
    var leases: [chainsFor(widths[widths.len - 1])]@import("core").kv_pool.Lease = @splat(.{});
    defer for (leases[0..most], 0..) |*l, slot| if (x.pool) |*p| {
        p.finish(l);
        x.m.unmapPages(slot);
    };
    if (x.pool) |*p| for (leases[0..most], 0..) |*l, slot| {
        try p.promise(l, 1);
        var map: Map = .{ .m = x.m, .x = x, .slot = slot };
        try p.cover(l, 1, .{ .ctx = &map, .write = Map.write });
    };
    var ids: [fwd.round_rows]u32 = undefined; // spread ids, as qwen27_costs.zig times its windows
    for (&ids, 0..) |*t, i| t.* = @intCast((i * 7919 + 13) % x.m.c.vocab);
    const timer: Timer = .{ .x = x, .io = io, .tokens = &ids };
    for (0..5) |_| _ = try timer.time(n - 1); // the widest round first, while the GPU's clocks ramp up
    var ms: [widths.len]f64 = undefined;
    for (ms[0..n], 0..) |*m, i| m.* = try lanes.cost_rule.fastest(timer, i);
    try lanes.cost_rule.smooth(timer, ms[0..n]);
    for (out[0..n], widths[0..n], ms[0..n]) |*c, w, m| c.* = .{ .width = w, .ms = m };
    std.log.info("Kolibri round costs over {d} slot{s}: 1 row {d:.2} ms, {d} rows {d:.2}", .{ x.m.slots, if (x.m.slots == 1) "" else "s", ms[0], widths[n - 1], ms[n - 1] });
    for (x.taps.written[0..@min(most, x.taps.written.len)]) |*w| w.* = 0; // no stream wrote these rows
    return n;
}

test "a round splits into windows of at most max_window rows" {
    try std.testing.expectEqual(@as(u32, 1), chainsFor(1));
    try std.testing.expectEqual(@as(u32, 1), chainsFor(16));
    try std.testing.expectEqual(@as(u32, 2), chainsFor(17));
    try std.testing.expectEqual(@as(u32, 4), chainsFor(64));
    for (widths) |w| try std.testing.expect(chainsFor(w) * be.max_window >= w);
}

test "calibration times the widths 1, 2 and 3 streams can take, without asking four slots" {
    try std.testing.expectEqual(@as(usize, 0), widthsFor(0));
    for ([_]usize{ 1, 2, 3, 4, 8 }, [_]u32{ 16, 32, 48, 64, 64 }, [_]usize{ 8, 10, 11, 12, 12 }) |slots, widest, n| {
        try std.testing.expectEqual(n, widthsFor(slots));
        try std.testing.expectEqual(widest, widths[widthsFor(slots) - 1]);
        try std.testing.expect(chainsFor(widths[n - 1]) <= slots);
    }
}
