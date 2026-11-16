//! Rows of several slots in one decode launch equal each slot's rows launched alone, bit for bit.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");
const fs = @import("fixture_serve.zig");

const testing = std.testing;
const fwd = pkg.forward;
const limit = 1280;
const look = 3;

/// One slot's share of a launch: where it stands and the ids its rows carry (after `look` ids of lookback).
const Part = struct { p0: u32, hist: []u32, n: usize };

fn finish(r: *fixture.Rig, f: *ff.Fwd, x: *const pkg.ctx.Ctx, rows: u32) !void {
    const pass = fixture.Pass.begin(r.queue);
    try fwd.encode(x, pass.e, f.model(), &f.b);
    try pass.finish();
    _ = rows;
}

/// Each part's rows alone on its slot (the single-stream path): logits [rows][vocab] and taps per part, concatenated.
fn alone(a: std.mem.Allocator, f: *ff.Fwd, slots: []pkg.kv.Slot, parts: []const Part, logits: []f32, taps: []u16) !void {
    const r = f.r;
    const c = &r.c;
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    _ = a;
    var r0: usize = 0;
    for (parts, slots) |pt, *sl| {
        try fwd.prepare(&f.b, c, pt.hist[look..][0..pt.n], pt.hist[0..look], f.lookup());
        try r.s.writeRows(c, pt.p0, @intCast(pt.n));
        const x = r.ctx(sl, pt.p0, @intCast(pt.n));
        try finish(r, f, &x, @intCast(pt.n));
        @memcpy(logits[r0 * V ..][0 .. pt.n * V], f.b.logitsOf(c, pt.n));
        @memcpy(taps[r0 * TH ..][0 .. pt.n * TH], sl.taps.?.slice(u16, pt.n * TH));
        r0 += pt.n;
    }
}

/// All parts' rows in one launch; the taps are read back from each slot.
fn together(f: *ff.Fwd, slots: []pkg.kv.Slot, parts: []const Part, logits: []f32, taps: []u16) !void {
    const r = f.r;
    const c = &r.c;
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    var pieces: [16]fwd.Piece = undefined;
    var spans: [16]pkg.scratch.Span = undefined;
    var segs: [16]pkg.ctx.Seg = undefined;
    var r0: u32 = 0;
    for (parts, slots, 0..) |pt, *sl, i| {
        pieces[i] = .{ .ids = pt.hist[look..][0..pt.n], .lookback = pt.hist[0..look] };
        spans[i] = .{ .p0 = pt.p0, .n = @intCast(pt.n) };
        segs[i] = .{ .slot = sl, .p0 = pt.p0, .r0 = r0, .n = @intCast(pt.n) };
        r0 += @intCast(pt.n);
    }
    const k = parts.len;
    try fwd.prepareSegs(&f.b, c, pieces[0..k], f.lookup());
    try r.s.writeSpans(c, spans[0..k]);
    var x = r.ctx(&slots[0], parts[0].p0, r0);
    x.segs = segs[0..k];
    try finish(r, f, &x, r0);
    @memcpy(logits[0 .. r0 * V], f.b.logitsOf(c, r0));
    var at: usize = 0;
    for (parts, slots) |pt, *sl| {
        @memcpy(taps[at * TH ..][0 .. pt.n * TH], sl.taps.?.slice(u16, pt.n * TH));
        at += pt.n;
    }
}

fn reset(r: *fixture.Rig, to: *const pkg.kv.Slot, from: *const pkg.kv.Slot) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    to.copyFrom(cb, from);
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CommandBufferFailed;
}

/// Bytes that differ between two slots' every region: window, pool, keys, carry and the carry's inputs.
fn stateDiff(x: *const pkg.kv.Slot, y: *const pkg.kv.Slot) usize {
    var n: usize = 0;
    for (x.regions, y.regions) |p, q| {
        const pairs = [_][2]?mtl.Buffer{ .{ p.window, q.window }, .{ p.pool, q.pool }, .{ p.keys, q.keys }, .{ p.carry, q.carry }, .{ p.inputs, q.inputs } };
        for (pairs) |pr| if (pr[0]) |bx| {
            const len = bx.length();
            for (bx.contents()[0..len], pr[1].?.contents()[0..len]) |s, t| n += @intFromBool(s != t);
        };
    }
    return n;
}

/// Slots at `starts` with `rows` rows each: one launch of all vs one launch each, bit for bit.
fn compare(a: std.mem.Allocator, f: *ff.Fwd, starts: []const u32, rows: []const usize, seed: u64) !usize {
    const r = f.r;
    const c = &r.c;
    const k = starts.len;
    var pristine: [4]pkg.kv.Slot = undefined;
    var one: [4]pkg.kv.Slot = undefined;
    var many: [4]pkg.kv.Slot = undefined;
    var parts: [4]Part = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        pristine[i].deinit();
        one[i].deinit();
        many[i].deinit();
        a.free(parts[i].hist);
    };
    var total: usize = 0;
    for (0..k) |i| {
        pristine[i] = try pkg.kv.Slot.init(a, r.device, c, limit);
        one[i] = try pkg.kv.Slot.init(a, r.device, c, limit);
        many[i] = try pkg.kv.Slot.init(a, r.device, c, limit);
        parts[i] = .{ .p0 = starts[i], .hist = try ff.ids(a, c, look + rows[i], seed + 10 * i), .n = rows[i] };
        made = i + 1;
        fixture.fillPast(c, &pristine[i], starts[i], seed + 100 * i);
        try reset(r, &one[i], &pristine[i]);
        try reset(r, &many[i], &pristine[i]);
        total += rows[i];
    }
    const TH = c.dspark_targets.len * c.hidden;
    const l1 = try a.alloc(f32, total * c.vocab);
    defer a.free(l1);
    const ln = try a.alloc(f32, total * c.vocab);
    defer a.free(ln);
    const t1 = try a.alloc(u16, total * TH);
    defer a.free(t1);
    const tn = try a.alloc(u16, total * TH);
    defer a.free(tn);
    try alone(a, f, one[0..k], parts[0..k], l1, t1);
    try together(f, many[0..k], parts[0..k], ln, tn);
    var bad: usize = 0;
    for (l1, ln) |s, t| bad += @intFromBool(@as(u32, @bitCast(s)) != @as(u32, @bitCast(t)));
    var tb: usize = 0;
    for (t1, tn) |s, t| tb += @intFromBool(s != t);
    var st: usize = 0;
    for (0..k) |i| st += stateDiff(&one[i], &many[i]);
    const word = if (bad + tb + st == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41 metal multi-slot forward: {d} slots, {d} rows in one launch vs one launch a slot: {d} logits, {d} tap values, {d} state bytes differ\n", .{ word, k, total, bad, tb, st });
    return bad + tb + st;
}

test "dsv41 metal multi-slot forward: rows of 2, 3 and 4 slots in one launch equal each slot's launch alone" {
    const a = testing.allocator;
    const f = try ff.Fwd.init(a, testing.io, limit, 61);
    defer f.deinit();
    var bad: usize = 0;
    // positions of both parities, a slot near the start and one past the window; ratio 2 pools complete on odd rows
    bad += try compare(a, f, &.{ 1101, 1104 }, &.{ 3, 5 }, 300);
    bad += try compare(a, f, &.{ 1103, 640, 1100 }, &.{ 1, 4, 2 }, 310);
    bad += try compare(a, f, &.{ 1101, 1104, 1099, 97 }, &.{ 4, 4, 4, 4 }, 320);
    bad += try compare(a, f, &.{ 1101, 1102, 1103, 1104 }, &.{ 1, 1, 1, 1 }, 330);
    bad += try compare(a, f, &.{ 7, 128, 1101, 1279 - 6 }, &.{ 2, 6, 1, 7 }, 340);
    try testing.expectEqual(@as(usize, 0), bad);
}

/// A model with the tiny drafter's config and prompts of different lengths on its slots.
fn rounds(m: *pkg.serve.Metal, packed_: bool, out: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const tg = m.t.target();
    const lens = [4]usize{ 150, 211, 64, 333 };
    var next: [4]u32 = undefined;
    var prompts: [4][]u32 = undefined;
    for (&prompts, 0..) |*p, i| p.* = try ff.ids(a, &fs.config(), lens[i], 400 + i);
    defer for (prompts) |p| a.free(p);
    for (0..4) |i| next[i] = try tg.prefill(@intCast(i), prompts[i], null, lens[i]);
    var pos = lens;
    // verify rounds of one anchor and three drafts a slot; the keep cuts back a different number each time
    for (0..4) |round| {
        var toks: [4][4]u32 = undefined;
        var draws: [4][4]u64 = undefined;
        var chs: [4][4]u32 = undefined;
        var segs: [4]host.iface.Segment = undefined;
        var cs: [4][]u32 = undefined;
        for (0..4) |i| {
            toks[i][0] = next[i];
            for (1..4) |j| toks[i][j] = @intCast((next[i] * 7 + j * 13 + round * 5 + i) % 200);
            for (&draws[i], 0..) |*d, j| d.* = pos[i] + 1 + j;
            segs[i] = .{ .slot = @intCast(i), .start = pos[i], .tokens = &toks[i], .parents = null, .draws = &draws[i], .sampling = null };
            cs[i] = &chs[i];
        }
        if (packed_) try tg.window(&segs, &cs) else for (0..4) |i| try tg.window(segs[i .. i + 1], cs[i .. i + 1]);
        for (0..4) |i| {
            const kept: u32 = @intCast(1 + (i + round) % 4);
            const path = [_]u32{ 0, 1, 2, 3 };
            try tg.keep(@intCast(i), path[0..kept]);
            try out.appendSlice(a, &chs[i]);
            pos[i] += kept;
            next[i] = chs[i][kept - 1];
        }
    }
    for (0..4) |i| tg.release(@intCast(i));
}

test "dsv41 metal multi-slot forward: a 4-slot DSpark verify round in one launch equals per-slot launches" {
    const a = testing.allocator;
    const m1 = try fs.open(a, testing.io, 71, .{});
    defer fs.close(m1);
    const m2 = try fs.open(a, testing.io, 71, .{});
    defer fs.close(m2);
    var o1: std.ArrayList(u32) = .empty;
    defer o1.deinit(a);
    var o2: std.ArrayList(u32) = .empty;
    defer o2.deinit(a);
    try rounds(m1, true, &o1);
    try rounds(m2, false, &o2);
    var st: usize = 0;
    for (m1.t.slots, m2.t.slots) |*x, *y| st += stateDiff(&x.kv, &y.kv);
    const same = std.mem.eql(u32, o1.items, o2.items);
    const word = if (same and st == 0) "PASS" else "FAIL";
    std.debug.print("{s} dsv41 metal multi-slot verify rounds: {d} choices, {d} launches in one vs {d} apart, {d} state bytes differ\n", .{ word, o1.items.len, m1.t.stats.launches, m2.t.stats.launches, st });
    try testing.expect(same);
    try testing.expectEqual(@as(usize, 0), st);
    try testing.expect(m1.t.stats.launches < m2.t.stats.launches);
}
