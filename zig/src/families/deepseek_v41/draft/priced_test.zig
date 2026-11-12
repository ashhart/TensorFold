//! Priced plain rounds over the twin: no pass where no draft pays, drafts where they do, probes, replies == serial.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const ln = @import("lanes.zig");
const iface = @import("iface.zig");
const costs_mod = @import("costs.zig");
const priced = @import("priced.zig");

const gpa = std.testing.allocator;
const t_ = std.testing;
const dims: twin.Dims = .{ .bigram = 8.0, .head = 0.08 };

/// A stream's drafts' chance of landing at a position (the scripted drafter reports it as its confidence).
pub const Chance = *const fn (stream: usize, pos: u64) f64;

/// The twin's pass (ring checks kept), each draft right with its stream's chance, which it reports as confidence.
const Script = struct {
    inner: iface.Pass,
    refs: []const []const u32, // slot -> prompt then its serial reply
    chance: Chance,
    vocab: u32,
    passes: u64 = 0,
    asks: u64 = 0,
    lone: u64 = 0, // passes that asked for one slot

    fn pass(s: *Script) iface.Pass {
        return .{ .ptr = s, .vtable = &.{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov } };
    }

    fn self(ptr: *anyopaque) *Script {
        return @ptrCast(@alignCast(ptr));
    }

    fn ingest(ptr: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        return self(ptr).inner.ingest(slot, start, taps, rows);
    }

    fn reset(ptr: *anyopaque, slot: u32) void {
        self(ptr).inner.reset(slot);
    }

    fn markov(ptr: *anyopaque) ?@import("dspark.zig").Markov {
        return self(ptr).inner.markov();
    }

    fn propose(ptr: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const s = self(ptr);
        try s.inner.propose(asks, out);
        s.passes += 1;
        s.asks += asks.len;
        s.lone += @intFromBool(asks.len == 1);
        for (asks, out) |a, o| for (o.drafts, o.conf, 0..) |*dr, *cf, j| {
            const pos = a.start + 1 + j;
            const ref = s.refs[a.slot];
            const want: u32 = if (pos < ref.len) ref[pos] else 0;
            const p = s.chance(a.slot, pos);
            var h = std.hash.Wyhash.init(pos);
            h.update(std.mem.asBytes(&a.slot));
            const u = @as(f64, @floatFromInt(h.final() >> 11)) / @as(f64, @floatFromInt(@as(u64, 1) << 53));
            dr.* = if (u < p) want else (want + 1) % s.vocab;
            cf.* = @floatCast(p);
        };
    }
};

/// A run's settings: chances, arrivals, sampling, the priced setting (off: the plain switch at 3), the table.
pub const Opts = struct { chance: Chance, stagger: bool = false, sampling: ?lanes.Sampling = null, priced: bool = true, costs: ?costs_mod.Costs = null };

pub const Run = struct {
    out: [][]u32,
    drafted: []u64, // each stream's verified drafts
    passes: u64,
    asks: u64,
    lone: u64,
    ingests: u64,
    stats: priced.Stats,
    steps: usize,

    pub fn deinit(r: Run) void {
        for (r.out) |o| gpa.free(o);
        gpa.free(r.out);
        gpa.free(r.drafted);
    }
};

/// The M3 Ultra's shape: its measured verify windows and a 12 ms pass.
pub fn m3Costs(buf: *[32]f64) costs_mod.Costs {
    buf.* = priced.m3Table();
    return .{ .verify = buf, .draft = 12.0 };
}

/// Each prompt's serial reply at its length plus a block past it (what a draft can reach), prompt first.
fn references(cases: []const []const u32, max_news: []const u32, sampling: ?lanes.Sampling) ![][]u32 {
    const t = try twin.Twin.init(gpa, dims, 1);
    defer t.deinit();
    const out = try gpa.alloc([]u32, cases.len);
    for (out, cases, max_news) |*o, p, n| {
        const reply = try t.serial(0, p, n + dims.block + 2, sampling, null);
        defer gpa.free(reply);
        o.* = try std.mem.concat(gpa, u32, &.{ p, reply });
    }
    return out;
}

pub fn run(cases: []const []const u32, max_news: []const u32, o: Opts) !Run {
    const refs = try references(cases, max_news, o.sampling);
    defer {
        for (refs) |r| gpa.free(r);
        gpa.free(refs);
    }
    const t = try twin.Twin.init(gpa, dims, cases.len + 1);
    defer t.deinit();
    const d = try twin.Drafter.init(gpa, t, cases.len + 1);
    defer d.deinit();
    d.check_ring = true;
    var script: Script = .{ .inner = d.pass(), .refs = refs, .chance = o.chance, .vocab = dims.vocab };
    var buf: [32]f64 = undefined;
    var c = try (o.costs orelse m3Costs(&buf)).clone(gpa);
    defer c.deinit(gpa);
    const x = try ln.Lanes.init(gpa, t.target(), script.pass(), c, .{ .shape = d.shape, .slots = 4, .priced_default = o.priced, .plain_streams_default = 3 });
    defer x.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var cfg = try lanes.Config.init(gpa, try x.model(arena.allocator(), 64), 16, 15);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var e = lanes.Engine.init(gpa, &cfg, x.backend(), clock.clock());
    defer e.deinit();
    x.attach(&e);
    const streams = try gpa.alloc(lanes.Stream, cases.len);
    defer gpa.free(streams);
    for (cases, streams, max_news) |p, *s, n| s.* = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = p, .max_new = n, .sampling = o.sampling });
    defer for (streams) |*s| s.deinit(gpa);
    var added: usize = 0;
    var steps: usize = 0;
    while (e.activeCount() > 0 or added < streams.len) : (steps += 1) {
        if (added < streams.len and (!o.stagger or steps % 3 == 0)) {
            try e.addStream(&streams[added]);
            added += 1;
            if (!o.stagger) continue;
        }
        try e.step();
        if (steps > 4000) return error.TooManySteps;
    }
    const out = try gpa.alloc([]u32, cases.len);
    const drafted = try gpa.alloc(u64, cases.len);
    for (out, drafted, streams) |*q, *dr, *s| {
        q.* = try gpa.dupe(u32, s.emitted());
        dr.* = s.drafted;
    }
    return .{ .out = out, .drafted = drafted, .passes = script.passes, .asks = script.asks, .lone = script.lone, .ingests = d.ingests, .stats = x.policy.priced, .steps = steps };
}

/// Each reply is its prompt's serial decode (one token a step, no drafts) at its own length.
pub fn expectSerial(r: Run, cases: []const []const u32, max_news: []const u32, sampling: ?lanes.Sampling) !void {
    const t = try twin.Twin.init(gpa, dims, 1);
    defer t.deinit();
    for (r.out, cases, max_news) |got, p, n| {
        const want = try t.serial(0, p, n, sampling, null);
        defer gpa.free(want);
        try t_.expectEqualSlices(u32, want, got);
    }
}

const a1 = @import("plain_streams_test.zig").a1;
const b1 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0, 4, 5, 2, 3, 5, 3 };
const c1 = [_]u32{ 1, 4, 1, 4, 2, 1, 3, 5, 6, 2, 3, 7, 3, 0, 9, 5, 0, 4, 8, 8 };
const d1 = [_]u32{ 1, 7, 3, 2, 0, 5, 0, 8, 0, 7, 5, 6, 8, 8, 7, 7, 2, 9, 3, 5 };
pub const four = [_][]const u32{ &a1, &b1, &c1, &d1 };

fn weak(_: usize, _: u64) f64 {
    return 0.3;
}
fn strong(_: usize, _: u64) f64 {
    return 0.95;
}
fn mixed(stream: usize, _: u64) f64 {
    return if (stream == 0) 0.95 else 0.1;
}

test {
    _ = @import("priced_probe_test.zig");
}

test "(a) weak drafts at 3 and 4 streams: plain rounds, no pass and no ingest in them, priced at the plain rate" {
    for ([_]usize{ 3, 4 }) |n| {
        const lens = [_]u32{ 60, 60, 60, 60 };
        const r = try run(four[0..n], lens[0..n], .{ .chance = weak });
        defer r.deinit();
        try expectSerial(r, four[0..n], lens[0..n], null);
        const s = r.stats;
        // the opening asks (each stream's first, no confidences yet) and the probes run a pass; no other round does
        try t_.expectEqual(s.passes, r.passes);
        try t_.expect(s.plain > 4 * s.passes);
        try t_.expect(s.probes >= 1 and s.passes <= s.fresh + s.probes);
        // ingests: each stream's prompt, then one catch-up and one window a slot asked, never one in a plain round
        try t_.expect(r.ingests <= n + 2 * r.asks);
        try t_.expectEqual(@as(u64, 0), s.below);
        // the probes' passes stay within their share of the plain rounds' time (the last one may be under way)
        try t_.expect(s.probe_ms <= priced.probe_share * s.plain_total_ms + 12.0);
        try t_.expect(!s.last.?.pass and s.last.?.rate >= s.last.?.plain);
    }
}

test "(b) strong drafts at 3 and 4 streams: every round drafts and its price beats plain" {
    for ([_]usize{ 3, 4 }) |n| {
        const lens = [_]u32{ 60, 60, 60, 60 };
        const r = try run(four[0..n], lens[0..n], .{ .chance = strong });
        defer r.deinit();
        try expectSerial(r, four[0..n], lens[0..n], null);
        const s = r.stats;
        // every round drafts but the last ones, whose streams have no room left for a draft
        try t_.expect(s.rounds > 0 and s.plain <= 2 and s.passes + s.plain == s.rounds);
        for (r.drafted) |dr| try t_.expect(dr > 0);
        try t_.expect(s.low > 1.0 and s.below == 0);
    }
}

test "(c) one strong stream beside three weak ones: it alone drafts, the others get none, the pass asks one slot" {
    const lens = [_]u32{ 60, 60, 60, 60 };
    const r = try run(&four, &lens, .{ .chance = mixed });
    defer r.deinit();
    try expectSerial(r, &four, &lens, null);
    try t_.expect(r.drafted[0] > 0);
    for (r.drafted[1..]) |dr| try t_.expectEqual(@as(u64, 0), dr);
    // the opening pass asks all four and a probe the plain three with it; every other pass asks the strong slot
    try t_.expect(r.lone > 0 and r.lone + 1 + r.stats.probes >= r.passes);
}

test "(e) sampled streams: weak, strong and mixed chances leave every keyed reply its serial decode" {
    const smp: lanes.Sampling = .{ .temperature = 0.8, .top_k = 0, .top_p = 1.0, .seed = 5 };
    const lens = [_]u32{ 50, 44, 38, 56 };
    for ([_]Chance{ weak, strong, mixed }) |ch| for ([_]bool{ false, true }) |stagger| {
        const r = try run(&four, &lens, .{ .chance = ch, .sampling = smp, .stagger = stagger });
        defer r.deinit();
        try expectSerial(r, &four, &lens, smp);
        const g = try run(&four, &lens, .{ .chance = ch, .stagger = stagger });
        defer g.deinit();
        try expectSerial(g, &four, &lens, null);
        // both kinds of round ran: plain ones where drafts lose, drafted ones where they win
        for ([_]Run{ r, g }) |x| try t_.expect(if (ch == strong) x.stats.passes > x.stats.plain else x.stats.plain > x.stats.passes);
    };
}

test "with the setting off: the plain switch, so four streams never draft and two pay a pass every round" {
    const lens = [_]u32{ 60, 60, 60, 60 };
    const strong4 = try run(&four, &lens, .{ .chance = strong, .priced = false });
    defer strong4.deinit();
    try expectSerial(strong4, &four, &lens, null);
    for (strong4.drafted) |dr| try t_.expectEqual(@as(u64, 0), dr);
    const weak2 = try run(four[0..2], lens[0..2], .{ .chance = weak, .priced = false });
    defer weak2.deinit();
    try expectSerial(weak2, four[0..2], lens[0..2], null);
    try t_.expectEqual(@as(u64, 0), weak2.stats.rounds);
    try t_.expect(weak2.passes + 4 >= weak2.steps);
}
