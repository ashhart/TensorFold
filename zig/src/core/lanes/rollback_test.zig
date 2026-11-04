//! Rollback conformance (#2138): heads landing no draft, some or all, and cancels at each boundary, against fresh runs.
const std = @import("std");
const Config = @import("config.zig").Config;
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const SuffixLookup = @import("proposer.zig").SuffixLookup;
const Sampling = @import("sampling.zig").Sampling;
const rounds = @import("engine_test.zig");
const p1 = rounds.p1;
const p2 = rounds.p2;
const p3 = [_]u32{ 6, 2, 8, 3, 1, 8, 5, 3, 0, 7 };

const gpa = std.testing.allocator;
const expect = std.testing.expect;

/// `Fake.head_accepts` past any depth: every draft is the target's own.
const every: usize = 1000;
const sampled: Sampling = .{ .seed = 9, .temperature = 0.8, .top_k = 0, .top_p = 0.95 };
const other: Sampling = .{ .seed = 17, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };

const Req = struct { prompt: []const u32, max_new: u32 = 40, sampling: ?Sampling = null, drafts: bool = true, cancel: ?sm.CancelCheck = null };

/// The fake target's one-token decode after `q.prompt`, to `q.max_new` tokens or eos 96: the fresh plain reply.
fn decode(q: Req) ![]u32 {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(gpa);
    try h.appendSlice(gpa, q.prompt);
    while (h.items.len - q.prompt.len < q.max_new) {
        const t = fake.next(h.items, q.sampling, h.items.len);
        try h.append(gpa, t);
        if (t == 96) break;
    }
    return gpa.dupe(u32, h.items[q.prompt.len..]);
}

fn expectFresh(q: Req, s: *const sm.Stream) !void {
    const want = try decode(q);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u32, want, s.emitted());
}

/// One engine over the fake target, its stream slots reused in place as a host reuses its jobs' memory.
const Rig = struct {
    cfg: Config,
    target: fake.Fake,
    clock: fake.FixedClock = .{},
    engine: Engine = undefined,
    streams: [3]sm.Stream = undefined,
    proposers: [3]SuffixLookup = undefined,
    used: [3]bool = @splat(false),
    filling: ?*sm.Stream = null, // a stream part way through its stepped prompt pass: a cache, not yet live

    fn create(accepts: ?usize, chunks: usize) !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.* = .{ .cfg = try rounds.model(), .target = .{ .gpa = gpa, .head_accepts = accepts, .prefill_chunks = chunks } };
        r.engine = Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        return r;
    }

    fn destroy(r: *Rig) void {
        r.engine.deinit();
        r.target.deinit();
        for (r.used, &r.streams, &r.proposers) |u, *s, *p| if (u) {
            s.deinit(gpa);
            p.deinit();
        };
        r.cfg.deinit(gpa);
        gpa.destroy(r);
    }

    /// Slot `i` holding a new stream for `q` (no copies: every draft is the head's).
    fn open(r: *Rig, i: usize, q: Req) !*sm.Stream {
        if (r.used[i]) {
            r.streams[i].deinit(gpa);
            r.proposers[i].deinit();
            r.used[i] = false;
        }
        r.proposers[i] = try SuffixLookup.init(gpa, .{ .min_match = 1000 });
        r.streams[i] = sm.Stream.init(gpa, .{ .id = "s", .prompt = q.prompt, .max_new = q.max_new, .eos = &.{96}, .sampling = q.sampling, .drafts = q.drafts, .proposer = r.proposers[i].proposer(), .cancel_check = q.cancel }) catch |err| {
            r.proposers[i].deinit();
            return err;
        };
        r.used[i] = true;
        return &r.streams[i];
    }

    fn step(r: *Rig) !void {
        try r.engine.step();
        try r.expectStates();
    }

    fn finish(r: *Rig) !usize {
        var n: usize = 0;
        while (r.engine.activeCount() > 0) : (n += 1) try r.step();
        return n;
    }

    /// One cache a live stream, its digest a fresh run's: the prompt and committed tokens, the pending one fed or not.
    fn expectStates(r: *Rig) !void {
        const partial: usize = if (r.filling) |s| @intFromBool(r.target.cache(s) != null) else 0;
        try std.testing.expectEqual(r.engine.live.items.len + partial, r.target.lanes.count());
        for (r.engine.live.items) |s| {
            const rows = r.target.cache(s) orelse return error.NoLane;
            const fresh = s.context.items;
            try expect(rows.len == fresh.len or rows.len + 1 == fresh.len);
            try std.testing.expectEqual(fake.Fake.digest(fresh[0..rows.len]), fake.Fake.digest(rows));
        }
    }
};

const Run = struct { tokens: []u32, drafted: u64, accepted: u64 };

fn run(accepts: ?usize, reqs: []const Req) ![]Run {
    const r = try Rig.create(accepts, 0);
    defer r.destroy();
    for (reqs, 0..) |q, i| try r.engine.addStream(try r.open(i, q));
    try r.expectStates();
    _ = try r.finish();
    const out = try gpa.alloc(Run, reqs.len);
    for (out, r.streams[0..reqs.len]) |*o, *s| o.* = .{ .tokens = try gpa.dupe(u32, s.emitted()), .drafted = s.drafted, .accepted = s.accepted };
    return out;
}

fn free(runs: []Run) void {
    for (runs) |x| gpa.free(x.tokens);
    gpa.free(runs);
}

test "rollback: heads landing no draft, some or all commit the one-token decode, alone and shared, greedy and sampled" {
    for ([_]?usize{ 0, 1, 2, every, null }) |k| for ([_]?Sampling{ null, sampled }) |smp| {
        const a: Req = .{ .prompt = &p1, .sampling = smp };
        const layouts = [_][]const Req{
            &.{a},
            &.{ a, .{ .prompt = &p2, .sampling = other, .max_new = 30 } },
            &.{ .{ .prompt = &p2, .drafts = false, .max_new = 30 }, a },
        };
        for (layouts) |reqs| {
            const got = try run(k, reqs);
            defer free(got);
            for (reqs, got) |q, g| {
                const want = try decode(q);
                defer gpa.free(want);
                try std.testing.expectEqualSlices(u32, want, g.tokens);
                var plain_q = q;
                plain_q.drafts = false;
                const plain = try run(k, &.{plain_q});
                defer free(plain);
                try std.testing.expectEqualSlices(u32, plain[0].tokens, g.tokens);
                if (!q.drafts) continue;
                // the head did as told: it drafted, and kept none, some or all of what it drafted
                try expect(g.drafted > 0);
                if (k == 0) try expect(g.accepted == 0) else if (k == every) try expect(g.accepted == g.drafted) else try expect(g.accepted > 0 and g.accepted < g.drafted);
            }
        }
    };
}

/// Steps of the uncancelled run of `reqs`, which bound where a cancel can land.
fn stepsOf(accepts: ?usize, reqs: []const Req) !usize {
    const r = try Rig.create(accepts, 0);
    defer r.destroy();
    for (reqs, 0..) |q, i| try r.engine.addStream(try r.open(i, q));
    return r.finish();
}

test "rollback: a stream cancelled between any two rounds leaves its peer and the next request in its slot as fresh runs" {
    const victims = [_]Req{ .{ .prompt = &p1 }, .{ .prompt = &p1, .sampling = sampled }, .{ .prompt = &p1, .drafts = false } };
    const peer: Req = .{ .prompt = &p2, .sampling = other, .max_new = 30 };
    const next: Req = .{ .prompt = &p3, .max_new = 24, .sampling = sampled };
    var cuts: usize = 0;
    for ([_]?usize{ 0, 1, every, null }) |k| for (victims) |v| {
        const steps = try stepsOf(k, &.{ v, peer });
        for (0..steps + 1) |cut| {
            const r = try Rig.create(k, 0);
            defer r.destroy();
            const a = try r.open(0, v);
            const b = try r.open(1, peer);
            try r.engine.addStream(a);
            try r.engine.addStream(b);
            try r.expectStates();
            for (0..cut) |_| if (!a.finished) try r.step();
            if (a.finished) try expectFresh(v, a) else {
                const before = a.emitted().len;
                r.engine.discard(a); // the host's cancel between rounds
                cuts += 1;
                try expect(a.finished and a.reason == .cancelled and a.emitted().len == before);
            }
            try expect(r.target.cache(a) == null);
            try r.expectStates();
            const c = try r.open(0, next); // the cancelled stream's memory, as a reused job's
            try r.engine.addStream(c);
            try r.expectStates();
            _ = try r.finish();
            try expectFresh(peer, b);
            try expectFresh(next, c);
            try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
        }
    };
    try expect(cuts > 100);
}

/// Answers the `at`-th question whether the request is cancelled, and every one after it, with yes.
const Asks = struct {
    at: usize,
    n: usize = 0,

    fn check(a: *Asks) sm.CancelCheck {
        return .{ .ptr = a, .check = asked };
    }

    fn asked(ptr: *anyopaque) bool {
        const a: *Asks = @ptrCast(@alignCast(ptr));
        a.n += 1;
        return a.n >= a.at;
    }
};

/// The victim's prompt pass (whole, or a chunk between the peer's rounds), cancelled at its `at`-th ask; its asks.
fn cancelInPass(k: ?usize, stepped: bool, v: Req, at: usize) !struct { asks: usize, cancelled: bool } {
    const chunks = 3;
    const peer: Req = .{ .prompt = &p2, .sampling = other, .max_new = 30 };
    const next: Req = .{ .prompt = &p3, .max_new = 24, .sampling = sampled };
    const r = try Rig.create(k, chunks);
    defer r.destroy();
    if (stepped) r.engine.backend = r.target.stepped();
    const b = try r.open(1, peer);
    try r.engine.addStream(b);
    try r.step();
    var asks: Asks = .{ .at = at };
    var q = v;
    q.cancel = asks.check();
    const a = try r.open(0, q);
    var cancelled = false;
    if (stepped) {
        var first = true;
        r.filling = a;
        defer r.filling = null;
        while (true) {
            const done = r.engine.fillStream(a, first) catch |err| {
                try std.testing.expectEqual(error.Cancelled, err);
                cancelled = true;
                break;
            };
            first = false;
            if (done) break;
            if (r.engine.activeCount() > 0) try r.step(); // the peer decodes between the victim's chunks
        }
    } else r.engine.addStream(a) catch |err| {
        try std.testing.expectEqual(error.Cancelled, err);
        cancelled = true;
    };
    try expect(cancelled or a.emitted().len > 0);
    if (cancelled) try expect(r.target.cache(a) == null and a.emitted().len == 0);
    try r.expectStates();
    if (!cancelled) {
        a.cancel_check = null;
        _ = try r.finish();
        try expectFresh(v, a);
    }
    const seen = asks.n;
    const c = try r.open(0, next);
    try r.engine.addStream(c);
    _ = try r.finish();
    try expectFresh(peer, b);
    try expectFresh(next, c);
    try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
    return .{ .asks = seen, .cancelled = cancelled };
}

test "rollback: a cancel at each ask of a prompt pass, whole or between a peer's rounds, leaves the next request fresh" {
    const victims = [_]Req{ .{ .prompt = &p1 }, .{ .prompt = &p1, .sampling = sampled }, .{ .prompt = &p1, .drafts = false } };
    var cut: usize = 0;
    for ([_]?usize{ 0, null }) |k| for ([_]bool{ false, true }) |stepped| for (victims) |v| {
        const whole = try cancelInPass(k, stepped, v, std.math.maxInt(usize));
        try expect(!whole.cancelled and whole.asks >= 3); // a chunk's ask each, then the open's
        for (1..whole.asks + 1) |at| {
            const got = try cancelInPass(k, stepped, v, at);
            try expect(got.cancelled);
            cut += 1;
        }
    };
    try expect(cut >= 24);
}
