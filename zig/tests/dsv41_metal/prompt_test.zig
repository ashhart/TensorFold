//! DeepSeek-V4.1's Metal prompt path: prompt then decode == decode only, chunk sizes, the cone, resumes, skipped rows.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const fixture = @import("fixture.zig");
const ff = @import("fixture_fwd.zig");

const testing = std.testing;
const fwd = pkg.forward;
const pr = pkg.prompt;
pub const limit = 1280;
/// the prompt: much longer than the tiny model's cone (3 decoder layers: 128 + 3 * 127 rows)
const n_prompt = 900;
const steps = 8;

/// Logits [rows][vocab] and taps [rows][targets][hidden] of consecutive rows.
pub const Out = struct {
    logits: []f32,
    taps: []u16,

    pub fn init(a: std.mem.Allocator, c: anytype, rows: usize) !Out {
        return .{ .logits = try a.alloc(f32, rows * c.vocab), .taps = try a.alloc(u16, rows * c.dspark_targets.len * c.hidden) };
    }

    pub fn deinit(o: Out, a: std.mem.Allocator) void {
        a.free(o.logits);
        a.free(o.taps);
    }
};

/// Positions [from, to) one row a launch through the decode forward; row i of `out` is position from + i.
pub fn decode(f: *ff.Fwd, slot: *const pkg.kv.Slot, hist: []const u32, from: usize, to: usize, out: ?Out) !void {
    const r = f.r;
    const c = &r.c;
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    const look = c.engram_max_ngram - 1;
    for (from..to) |p| {
        try fwd.prepare(&f.b, c, hist[p..][0..1], hist[p -| look..p], f.lookup());
        try r.s.writeRows(c, @intCast(p), 1);
        const x = r.ctx(slot, @intCast(p), 1);
        const pass = fixture.Pass.begin(r.queue);
        try fwd.encode(&x, pass.e, f.model(), &f.b);
        try pass.finish();
        if (out) |o| {
            @memcpy(o.logits[(p - from) * V ..][0..V], f.b.logitsOf(c, 1));
            @memcpy(o.taps[(p - from) * TH ..][0..TH], slot.taps.?.slice(u16, TH));
        }
    }
}

pub fn env(f: *ff.Fwd) pr.Env {
    return .{ .c = &f.r.c, .p = &f.r.p, .tables = &f.r.tables, .model = f.model(), .lookup = f.lookup(), .queue = f.r.queue };
}

/// A fresh slot and tail with a prompt's state, and the run's outputs.
pub const Run = struct {
    slot: pkg.kv.Slot,
    tail: pr.Tail,
    p: pr.Prompt,

    pub fn init(a: std.mem.Allocator, f: *ff.Fwd, o: pr.Opts, cap: u32) !Run {
        const c = &f.r.c;
        var slot = try pkg.kv.Slot.init(a, f.r.device, c, limit);
        errdefer slot.deinit();
        var tail = try pr.Tail.init(f.r.device, c, @max(cap, try pr.tailRows(c, o)));
        errdefer tail.deinit();
        return .{ .slot = slot, .tail = tail, .p = try pr.Prompt.init(f.r.device, c, o, limit) };
    }

    pub fn deinit(r: *Run) void {
        r.p.deinit();
        r.tail.deinit();
        r.slot.deinit();
    }
};

pub fn bitsDiff(x: []const f32, y: []const f32) usize {
    var n: usize = 0;
    for (x, y) |s, t| n += @intFromBool(@as(u32, @bitCast(s)) != @as(u32, @bitCast(t)));
    return n;
}

pub fn halfDiff(x: []const u16, y: []const u16) usize {
    var n: usize = 0;
    for (x, y) |s, t| n += @intFromBool(s != t);
    return n;
}

/// Bytes that differ between two slots: every layer's window ring, pool, index keys and carry, and the taps.
pub fn stateDiff(x: *const pkg.kv.Slot, y: *const pkg.kv.Slot) usize {
    var n: usize = 0;
    for (x.regions, y.regions) |p, q| {
        const pairs = [_][2]?mtl.Buffer{ .{ p.window, q.window }, .{ p.pool, q.pool }, .{ p.keys, q.keys }, .{ p.carry, q.carry } };
        for (pairs) |pp| if (pp[0]) |bx| {
            const len = bx.length();
            for (bx.contents()[0..len], pp[1].?.contents()[0..len]) |s, t| n += @intFromBool(s != t);
        };
    }
    const len = x.taps.?.length();
    for (x.taps.?.contents()[0..len], y.taps.?.contents()[0..len]) |s, t| n += @intFromBool(s != t);
    return n;
}

pub fn tailDiff(x: *const pr.Tail, y: *const pr.Tail) usize {
    var n: usize = 0;
    for ([2][2]mtl.Buffer{ .{ x.x, y.x }, .{ x.pre, y.pre } }) |b| {
        for (b[0].contents()[0..b[0].length()], b[1].contents()[0..b[1].length()]) |s, t| n += @intFromBool(s != t);
    }
    return n;
}

/// A run's head logits, taps and slot against another's: values that differ.
pub fn runDiff(c: anytype, x: *const Run, y: *const Run) [3]usize {
    return .{ bitsDiff(x.p.logits(c), y.p.logits(c)), halfDiff(x.p.tapRows(c), y.p.tapRows(c)), stateDiff(&x.slot, &y.slot) };
}

fn say(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} dsv41 metal prompt: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(fmt ++ "\n", args);
}

pub const head_opts: pr.Opts = .{ .chunk = 64, .head_rows = 128, .taps_rows = 128 };

test "dsv41 metal prompt: a prompt then 8 decode steps equal the same tokens decoded one at a time; the cone's rows" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), limit, 61);
    defer f.deinit();
    const c = &f.r.c;
    const hist = try ff.ids(a, c, n_prompt + steps, 62);
    defer a.free(hist);
    // decode only: every position one row a launch
    var dslot = try pkg.kv.Slot.init(a, f.r.device, c, limit);
    defer dslot.deinit();
    const dref = try Out.init(a, c, n_prompt + steps);
    defer dref.deinit(a);
    try decode(f, &dslot, hist, 0, n_prompt + steps, dref);
    // the prompt, then the same decode steps
    var run = try Run.init(a, f, head_opts, 0);
    defer run.deinit();
    try run.p.run(env(f), &run.slot, &run.tail, hist[0..n_prompt]);
    const V: usize = c.vocab;
    const TH = c.dspark_targets.len * c.hidden;
    const first = n_prompt - 128;
    const lg = bitsDiff(run.p.logits(c), dref.logits[first * V .. n_prompt * V]);
    const tp = halfDiff(run.p.tapRows(c), dref.taps[first * TH .. n_prompt * TH]);
    const dout = try Out.init(a, c, steps);
    defer dout.deinit(a);
    try decode(f, &run.slot, hist, n_prompt, n_prompt + steps, dout);
    try run.tail.advance(n_prompt + steps);
    const dl = bitsDiff(dout.logits, dref.logits[n_prompt * V ..]);
    const dt = halfDiff(dout.taps, dref.taps[n_prompt * TH ..]);
    const st = stateDiff(&run.slot, &dslot);
    const bad = lg + tp + dl + dt + st;
    say(bad == 0, "(a) {d} tokens + {d} steps vs decode only: {d} head logits ({d} rows), {d} tap values, {d} step logits, {d} step taps, {d} state bytes differ", .{ n_prompt, steps, lg, run.p.head, tp, dl, dt, st });
    try testing.expectEqual(@as(usize, 0), bad);
    // (e) decoder layers ran only their cone: keys kept + (layers - L) * 127, rows kept + (layers - 1 - L) * 127
    var skipped: usize = 0;
    for (4..7) |L| {
        const outs: u32 = 128 + @as(u32, @intCast(6 - L)) * 127;
        std.debug.print("     layer {d}: window keys {d}, whole rows {d} of {d}\n", .{ L, run.p.keys[L], run.p.outs[L], n_prompt });
        skipped += @intFromBool(run.p.outs[L] == outs and run.p.keys[L] == outs + 127 and outs < n_prompt);
    }
    say(skipped == 3, "(e) cone rows: {d} of 3 decoder layers ran exactly their cone", .{skipped});
    try testing.expectEqual(@as(usize, 3), skipped);
}

test "dsv41 metal prompt: chunk sizes 1, 7, 64 and whole give the same logits, taps, slot and tail" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), limit, 63);
    defer f.deinit();
    const c = &f.r.c;
    const hist = try ff.ids(a, c, n_prompt, 64);
    defer a.free(hist);
    var base = try Run.init(a, f, head_opts, 0);
    defer base.deinit();
    try base.p.run(env(f), &base.slot, &base.tail, hist);
    var bad: usize = 0;
    for ([_]u32{ 1, 7, n_prompt }) |chunk| {
        var o = head_opts;
        o.chunk = chunk;
        var run = try Run.init(a, f, o, 0);
        defer run.deinit();
        try run.p.run(env(f), &run.slot, &run.tail, hist);
        const d = runDiff(c, &run, &base);
        const td = tailDiff(&run.tail, &base.tail);
        say(d[0] + d[1] + d[2] + td == 0, "(b) chunk {d} vs 64: {d} logits, {d} taps, {d} state bytes, {d} tail bytes differ", .{ chunk, d[0], d[1], d[2], td });
        bad += d[0] + d[1] + d[2] + td;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal prompt: the suffix cone equals every decoder layer on every row" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), limit, 65);
    defer f.deinit();
    const c = &f.r.c;
    const hist = try ff.ids(a, c, n_prompt, 66);
    defer a.free(hist);
    var bad: usize = 0;
    for ([_]u32{ 128, 1 }) |kept| {
        var o = head_opts;
        o.kept = kept;
        o.head_rows = @min(128, pr.keptOf(c, o));
        var cone = try Run.init(a, f, o, 0);
        defer cone.deinit();
        try cone.p.run(env(f), &cone.slot, &cone.tail, hist);
        o.full = true;
        var full = try Run.init(a, f, o, n_prompt);
        defer full.deinit();
        try full.p.run(env(f), &full.slot, &full.tail, hist);
        const d = runDiff(c, &cone, &full);
        say(d[0] + d[1] + d[2] == 0, "(c) kept {d}: cone ({d}/{d}/{d} rows at layers 4/5/6) vs full ({d}): {d} logits, {d} taps, {d} state bytes differ", .{ kept, cone.p.outs[4], cone.p.outs[5], cone.p.outs[6], full.p.outs[4], d[0], d[1], d[2] });
        bad += d[0] + d[1] + d[2];
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal prompt: resumes — a slot saved at a chunk start, and a prompt in two turns, equal a fresh prompt" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), limit, 67);
    defer f.deinit();
    const c = &f.r.c;
    const hist = try ff.ids(a, c, n_prompt, 68);
    defer a.free(hist);
    var fresh = try Run.init(a, f, head_opts, 0);
    defer fresh.deinit();
    try fresh.p.run(env(f), &fresh.slot, &fresh.tail, hist);
    var bad: usize = 0;
    // mid-prompt: the encoder to a chunk start, the regions and tail copied, the rest on the copy (other chunks)
    {
        var part = try Run.init(a, f, head_opts, 0);
        defer part.deinit();
        try part.p.encode(env(f), &part.slot, &part.tail, hist, 448);
        var o = head_opts;
        o.chunk = 7;
        var rest = try Run.init(a, f, o, 0);
        defer rest.deinit();
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = f.r.queue.commandBuffer();
            rest.slot.copyFrom(cb, &part.slot);
            try rest.tail.copyFrom(cb, &part.tail);
            cb.commit();
            cb.wait();
        }
        try rest.p.run(env(f), &rest.slot, &rest.tail, hist);
        const d = runDiff(c, &rest, &fresh);
        say(d[0] + d[1] + d[2] == 0, "(d) saved at 448, resumed in chunks of 7: {d} logits, {d} taps, {d} state bytes differ", .{ d[0], d[1], d[2] });
        bad += d[0] + d[1] + d[2];
    }
    // two turns: a finished prompt of 600, then 300 more (the decoder resumes from the first's rings)
    {
        var two = try Run.init(a, f, head_opts, 0);
        defer two.deinit();
        try two.p.run(env(f), &two.slot, &two.tail, hist[0..600]);
        try two.p.run(env(f), &two.slot, &two.tail, hist);
        const d = runDiff(c, &two, &fresh);
        say(d[0] + d[1] + d[2] == 0, "(d) turns of 600 + 300 vs one: {d} logits, {d} taps, {d} state bytes differ", .{ d[0], d[1], d[2] });
        bad += d[0] + d[1] + d[2];
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "dsv41 metal prompt: the grouped routed experts equal the decode row kernels over a chunk" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), limit, 69);
    defer f.deinit();
    const c = &f.r.c;
    const hist = try ff.ids(a, c, n_prompt, 70);
    defer a.free(hist);
    var bad: usize = 0;
    for ([_]u32{ 64, 300 }) |chunk| {
        var o = head_opts;
        o.chunk = chunk;
        o.expert_loop = true;
        var loop = try Run.init(a, f, o, 0);
        defer loop.deinit();
        try loop.p.run(env(f), &loop.slot, &loop.tail, hist);
        o.expert_loop = false;
        var grouped = try Run.init(a, f, o, 0);
        defer grouped.deinit();
        try grouped.p.run(env(f), &grouped.slot, &grouped.tail, hist);
        const d = runDiff(c, &grouped, &loop);
        const td = tailDiff(&grouped.tail, &loop.tail);
        say(d[0] + d[1] + d[2] + td == 0, "(f) chunk {d}, grouped experts vs row kernels: {d} logits, {d} taps, {d} state bytes, {d} tail bytes differ", .{ chunk, d[0], d[1], d[2], td });
        bad += d[0] + d[1] + d[2] + td;
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

// Wall time of a 2,048-row chunk through the encoder layers (median of 3), row kernels then grouped experts.
test "dsv41 metal prompt: timing of a 2048-row chunk, row kernels vs grouped experts" {
    const a = testing.allocator;
    const f = try ff.Fwd.initWith(a, testing.io, fixture.promptConfig(), 2304, 71);
    defer f.deinit();
    const c = &f.r.c;
    const hist = try ff.ids(a, c, 2048, 72);
    defer a.free(hist);
    for ([_]bool{ true, false }) |loop| {
        var us: [3]f64 = undefined;
        for (&us) |*u| {
            var slot = try pkg.kv.Slot.init(a, f.r.device, c, 2304);
            defer slot.deinit();
            var o = head_opts;
            o.chunk = 2048;
            o.expert_loop = loop;
            var tail = try pr.Tail.init(f.r.device, c, try pr.tailRows(c, o));
            defer tail.deinit();
            var p = try pr.Prompt.init(f.r.device, c, o, 2304);
            defer p.deinit();
            const t0 = std.Io.Clock.awake.now(testing.io);
            try p.run(env(f), &slot, &tail, hist);
            u.* = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(testing.io)).toNanoseconds())) / 1000.0;
        }
        std.mem.sort(f64, &us, {}, std.sort.asc(f64));
        std.debug.print("TIME dsv41 metal prompt: 2048 rows, experts on {s}: {d:.0} us\n", .{ if (loop) "row kernels" else "grouped kernels", us[1] });
    }
}
