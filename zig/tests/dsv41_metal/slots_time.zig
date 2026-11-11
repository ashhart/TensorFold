//! The tiny forward's decode round at 1, 2 and 4 slots of one row (and one slot of 4): dispatches, barriers, time.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");
const recorder = @import("recorder.zig");

const testing = std.testing;
const fwd = pkg.forward;
const limit = 1280;
const look = 3;
const most = 4;
const reps = 100;

/// Dispatches and barriers of an encode.
const Count = struct { d: usize = 0, b: usize = 0 };

fn tally(rec: *const recorder.Recording) Count {
    var n: Count = .{};
    for (rec.ops.items) |op| switch (op) {
        .groups, .threads => n.d += 1,
        .barrier => n.b += 1,
        else => {},
    };
    return n;
}

fn lessThan(_: void, x: f64, y: f64) bool {
    return x < y;
}

/// A launch of `k` slots of `n` rows each over the slots' pasts; the ids and records written, the context returned.
fn launch(f: *ff.Fwd, slots: []pkg.kv.Slot, hist: []const []u32, n: u32, segs: *[most]pkg.ctx.Seg) !pkg.ctx.Ctx {
    const r = f.r;
    const c = &r.c;
    var pieces: [most]fwd.Piece = undefined;
    var spans: [most]pkg.scratch.Span = undefined;
    const k = slots.len;
    for (0..k) |i| {
        const p0: u32 = @intCast(1101 + 3 * i);
        pieces[i] = .{ .ids = hist[i][look..][0..n], .lookback = hist[i][0..look] };
        spans[i] = .{ .p0 = p0, .n = n };
        segs[i] = .{ .slot = &slots[i], .p0 = p0, .r0 = @as(u32, @intCast(i)) * n, .n = n };
    }
    try fwd.prepareSegs(&f.b, c, pieces[0..k], f.lookup());
    try r.s.writeSpans(c, spans[0..k]);
    var x = r.ctx(&slots[0], segs[0].p0, @as(u32, @intCast(k)) * n);
    if (k > 1) x.segs = segs[0..k];
    return x;
}

/// Each layer's dispatches and barriers, and the round's median wall time over `reps` runs.
fn measure(a: std.mem.Allocator, f: *ff.Fwd, slots: []pkg.kv.Slot, hist: []const []u32, n: u32, per: []Count) !f64 {
    const r = f.r;
    var segs: [most]pkg.ctx.Seg = undefined;
    const x = try launch(f, slots, hist, n, &segs);
    var rec: recorder.Recording = .{ .a = a };
    defer rec.deinit();
    for (per, 0..) |*cnt, i| {
        const re = recorder.start(&rec);
        const ok = fwd.encodeLayer(&x, re, f.model(), &f.b, i);
        recorder.stop(re);
        try ok;
        cnt.* = tally(&rec);
    }
    var t: [reps]f64 = undefined;
    for (0..reps + 5) |i| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const t0 = mtl.clock.seconds();
        try fwd.run(&x, r.queue, f.model(), &f.b);
        if (i >= 5) t[i - 5] = mtl.clock.seconds() - t0;
    }
    std.mem.sort(f64, &t, {}, lessThan);
    return t[reps / 2] * 1e3;
}

test "dsv41 metal slots timing: a round's dispatches, barriers and time at 1, 2 and 4 slots and 1 slot of 4 rows" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 81);
    defer f.deinit();
    const c = &f.r.c;
    var slots: [most]pkg.kv.Slot = undefined;
    var hist: [most][]u32 = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        slots[i].deinit();
        a.free(hist[i]);
    };
    for (0..most) |i| {
        slots[i] = try pkg.kv.Slot.init(a, f.r.device, c, limit);
        hist[i] = try ff.ids(a, c, look + most, 500 + i);
        made = i + 1;
        fixture.fillPast(c, &slots[i], @intCast(1101 + 3 * i), 600 + i);
    }
    const per = try a.alloc(Count, c.layers);
    defer a.free(per);
    for ([_][2]u32{ .{ 1, 1 }, .{ 2, 1 }, .{ 4, 1 }, .{ 1, 4 } }) |kn| {
        const k = kn[0];
        const ms = try measure(a, f, slots[0..k], hist[0..k], kn[1], per);
        var tot: Count = .{};
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(a);
        for (per, 0..) |n, i| {
            tot.d += n.d;
            tot.b += n.b;
            try line.print(a, " L{d} {d}/{d}", .{ i, n.d, n.b });
        }
        std.debug.print("dsv41 metal slots timing: {d} slot(s) x {d} row(s): {d} dispatches, {d} barriers in the layers; median round {d:.3} ms; per layer d/b:{s}\n", .{ k, kn[1], tot.d, tot.b, ms, line.items });
    }
}
