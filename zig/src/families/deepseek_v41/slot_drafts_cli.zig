//! `tf-dsv41 slot-drafts PACK ASSETS TRACE...` (rank 0): DSpark drafting over several live slots, batched == alone.

const std = @import("std");
const api = @import("engine_api");
const se = @import("serve_engine.zig");

const Keyed = struct { seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64 = 0 };
const Line = struct { prompt: []const u32, tokens: []const u32, sampling: ?Keyed = null };

const Box = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(b.gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn finished(b: *Box) ?api.Reason {
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        return b.done;
    }
};

/// `got` equals `want`, or `want` cut at its first end-of-sequence id (with or without it).
fn same(got: []const u32, want: []const u32, eos: u32) bool {
    if (std.mem.eql(u32, got, want)) return true;
    const cut = std.mem.indexOfScalar(u32, want, eos) orelse return false;
    return std.mem.eql(u32, got, want[0 .. cut + 1]) or std.mem.eql(u32, got, want[0..cut]);
}

fn firstDiff(got: []const u32, want: []const u32) i64 {
    return @intCast(std.mem.indexOfDiff(u32, got, want) orelse @min(got.len, want.len));
}

fn submitAll(eng: api.Engine, io: std.Io, lines: []const Line, which: []const usize, boxes: []Box, reqs: []api.Request, id0: api.Id) !void {
    for (which, 0..) |i, k| {
        const l = lines[i];
        const smp: ?api.Sampling = if (l.sampling) |x| .{ .seed = x.seed, .temperature = x.temperature, .top_k = x.top_k, .top_p = x.top_p, .min_p = x.min_p } else null;
        reqs[i] = .{ .prompt = l.prompt, .max_tokens = @intCast(l.tokens.len), .drafts = true, .sampling = smp };
        try eng.submit(id0 + k, &reqs[i], .{ .ctx = &boxes[i], .event = Box.event });
    }
    while (true) {
        var all = true;
        for (which) |i| all = all and boxes[i].finished() != null;
        if (all) return;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
}

/// A run's drafting from the lanes' counters (deltas) and the slot passes'.
const Tally = struct {
    drafted: u64 = 0,
    accepted: u64 = 0,
    passes: u64 = 0,
    pass_slots: u64 = 0,
    /// the slot passes' drafts digest after the run (SlotPass.digest: every proposal so far, in order)
    digest: u64 = 0,

    fn now(s: *const se.Served) Tally {
        const p = s.m.slot_pass;
        return .{ .drafted = s.core.drafted, .accepted = s.core.accepted, .passes = if (p) |x| x.stats.passes else 0, .pass_slots = if (p) |x| x.stats.slots else 0, .digest = if (p) |x| x.digest else 0 };
    }

    fn since(t: Tally, s: *const se.Served) Tally {
        const n = now(s);
        return .{ .drafted = n.drafted - t.drafted, .accepted = n.accepted - t.accepted, .passes = n.passes - t.passes, .pass_slots = n.pass_slots - t.pass_slots, .digest = n.digest };
    }

    /// Each stream's first token comes from its prefill; every round after commits its accepted drafts + 1.
    fn print(t: Tally, name: []const u8, tokens: usize, streams: usize, ms: i128) void {
        const rounds = tokens -| (t.accepted + streams);
        const acc = if (t.drafted > 0) @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.drafted)) else 0;
        const tpr = if (rounds > 0) @as(f64, @floatFromInt(tokens - streams)) / @as(f64, @floatFromInt(rounds)) else 0;
        const spp = if (t.passes > 0) @as(f64, @floatFromInt(t.pass_slots)) / @as(f64, @floatFromInt(t.passes)) else 0;
        std.debug.print("slot drafts stats {s}: tokens {d}, streams {d}, drafted {d}, accepted {d}, acceptance {d:.3}, stream rounds {d}, tokens/round {d:.3}, passes {d}, slots/pass {d:.2}, {d} ms\n", .{ name, tokens, streams, t.drafted, t.accepted, acc, rounds, tpr, t.passes, spp, @divTrunc(ms, 1_000_000) });
        std.debug.print("slot drafts drafts {s}: digest {x:0>16}\n", .{ name, t.digest });
    }
};

pub fn main(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8, traces: []const []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lines: std.ArrayList(Line) = .empty;
    for (traces) |trace| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, trace, a, .limited(1 << 30));
        var it = std.mem.tokenizeScalar(u8, text, '\n');
        while (it.next()) |l| try lines.append(a, try std.json.parseFromSliceLeaky(Line, a, l, .{ .ignore_unknown_fields = true }));
    }
    const n = lines.items.len;
    if (n == 0 or n > 16) return error.BadTrace;
    const s = try se.Served.open(gpa, io, pack, assets, true, null);
    defer se.Served.close(s);
    if (s.m.slot_pass == null) return error.SlotDraftsOff; // TF_DSV41_SLOTS > 1 and TF_DSV41_SLOT_DRAFTS=1
    const eng = s.engine();
    const eos = s.m.cfg.eos;
    std.debug.print("slot drafts check: {d} live slots, DSpark passes of up to {d} slots, {d} requests\n", .{ s.m.slots, s.m.slot_pass.?.group, n });
    const batched = try a.alloc(Box, n);
    const alone = try a.alloc(Box, n);
    const reqs = try a.alloc(api.Request, n);
    for (batched, alone) |*x, *y| {
        x.* = .{ .io = io, .gpa = a };
        y.* = .{ .io = io, .gpa = a };
    }
    const all = try a.alloc(usize, n);
    for (all, 0..) |*x, i| x.* = i;
    var t0 = std.Io.Clock.awake.now(io);
    var tb = Tally.now(s);
    try submitAll(eng, io, lines.items, all, batched, reqs, 1);
    const ms_b = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const db = tb.since(s);
    t0 = std.Io.Clock.awake.now(io);
    tb = Tally.now(s);
    for (0..n) |i| try submitAll(eng, io, lines.items, &.{i}, alone, reqs, 100 + i);
    const ms_a = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const da = tb.since(s);
    var ok_b: usize = 0;
    var ok_a: usize = 0;
    var ok_ba: usize = 0;
    var tok_b: usize = 0;
    var tok_a: usize = 0;
    for (lines.items, batched, alone, 0..) |l, *b, *x, i| {
        const eb = same(b.tokens.items, l.tokens, eos);
        const ea = same(x.tokens.items, l.tokens, eos);
        const eba = std.mem.eql(u32, b.tokens.items, x.tokens.items);
        ok_b += @intFromBool(eb);
        ok_a += @intFromBool(ea);
        ok_ba += @intFromBool(eba);
        tok_b += b.tokens.items.len;
        tok_a += x.tokens.items.len;
        std.debug.print("{{\"line\": {d}, \"prompt\": {d}, \"sampled\": {}, \"recorded\": {d}, \"batched\": {d}, \"alone\": {d}, \"batched_eq_ref\": {}, \"alone_eq_ref\": {}, \"batched_eq_alone\": {}, \"first_diff_batched\": {d}, \"first_diff_alone\": {d}, \"finished\": \"{t}\"}}\n", .{ i, l.prompt.len, l.sampling != null, l.tokens.len, b.tokens.items.len, x.tokens.items.len, eb, ea, eba, if (eb) -1 else firstDiff(b.tokens.items, l.tokens), if (ea) -1 else firstDiff(x.tokens.items, l.tokens), b.done.? });
    }
    db.print("batched", tok_b, n, ms_b);
    da.print("alone", tok_a, n, ms_a);
    // drafted with several slots: some batched pass held more than one slot
    const shared = db.drafted > 0 and db.pass_slots > db.passes;
    const pass = ok_b == n and ok_a == n and ok_ba == n and shared;
    std.debug.print("{s} slot drafts batched == Python: {d}/{d}\n", .{ if (ok_b == n) "PASS" else "FAIL", ok_b, n });
    std.debug.print("{s} slot drafts alone == Python: {d}/{d}\n", .{ if (ok_a == n) "PASS" else "FAIL", ok_a, n });
    std.debug.print("{s} slot drafts batched == alone: {d}/{d}\n", .{ if (ok_ba == n) "PASS" else "FAIL", ok_ba, n });
    std.debug.print("{s} slot drafts several slots a pass: drafted {d}, {d} passes over {d} slots\n", .{ if (shared) "PASS" else "FAIL", db.drafted, db.passes, db.pass_slots });
    if (s.m.rows) |r| std.debug.print("row windows: {d}, rows {d}, padded {d}\n", .{ r.stats.windows, r.stats.rows, r.stats.padded });
    return if (pass) 0 else 1;
}
