//! Reply-scoped host timing for paired jobs, with release gaps retained across reply boundaries.
const std = @import("std");

const names = [_][]const u8{ "exchange", "reduce", "pick", "send", "wait" };
const ticks_per_us = 24;

pub const Metric = struct {
    count: u64 = 0,
    sum_ticks: u64 = 0,
    max_ticks: u64 = 0,
    buckets: [65]u64 = @splat(0),

    pub fn bucket(ticks: u64) usize {
        const us = ticks / ticks_per_us;
        return if (us == 0) 0 else 1 + @as(usize, std.math.log2_int(u64, us));
    }

    pub fn add(m: *Metric, ticks: u64) void {
        m.count +|= 1;
        m.sum_ticks +|= ticks;
        m.max_ticks = @max(m.max_ticks, ticks);
        m.buckets[bucket(ticks)] +|= 1;
    }

    fn sumUs(m: Metric) f64 {
        return @as(f64, @floatFromInt(m.sum_ticks)) / ticks_per_us;
    }

    fn maxUs(m: Metric) f64 {
        return @as(f64, @floatFromInt(m.max_ticks)) / ticks_per_us;
    }

    fn meanUs(m: Metric) f64 {
        return if (m.count == 0) 0 else m.sumUs() / @as(f64, @floatFromInt(m.count));
    }
};

pub const Kind = struct {
    held: Metric = .{},
    release_to_post: Metric = .{},
    send: Metric = .{},
    arrival: Metric = .{},
};

pub const Reply = struct {
    first: u64 = 0,
    last: u64 = 0,
    observed: u64 = 0,
    partial: u64 = 0,
    held: Metric = .{},
    kinds: [names.len]Kind = @splat(.{}),

    pub fn print(r: *const Reply, rank: u32) void {
        std.debug.print("TP rank{d} stats jobs {d}..{d}: host held {d:.1} us a job, max {d:.1} us ({d} jobs, {d} partial)\n", .{ rank, r.first, r.last, r.held.meanUs(), r.held.maxUs(), r.observed, r.partial });
        for (r.kinds, names) |k, name| {
            if (k.held.count == 0) continue;
            std.debug.print("TP rank{d} stats {s}: host held {d:.1} us a job ({d} jobs), max {d:.1} us\n", .{ rank, name, k.held.meanUs(), k.held.count, k.held.maxUs() });
            printMetric(rank, name, "release to next post", k.release_to_post);
            std.debug.print("TP rank{d} stats {s} release to next post log2_us (<1,1..2,2..4,...): {any}\n", .{ rank, name, k.release_to_post.buckets });
            printMetric(rank, name, "send", k.send);
            printMetric(rank, name, "peer arrival wait", k.arrival);
        }
    }

    fn printMetric(rank: u32, kind: []const u8, label: []const u8, m: Metric) void {
        std.debug.print("TP rank{d} stats {s} {s}: count {d} sum {d:.1} us max {d:.1} us\n", .{ rank, kind, label, m.count, m.sumUs(), m.maxUs() });
    }
};

pub const Phase = enum { send, arrival };
const Span = struct {
    begin: ?u64 = null,
    end: ?u64 = null,

    fn duration(s: Span, now: u64) u64 {
        return if (s.begin) |begin| (s.end orelse now) -% begin else 0;
    }
};
const Active = struct {
    seq: u64,
    kind: usize,
    seen: u64,
    send: Span = .{},
    arrival: Span = .{},
};

pub const State = struct {
    lock: std.atomic.Mutex = .unlocked,
    previous_release: ?u64 = null,
    closed_through: u64 = 0,
    active: ?Active = null,
    reply: Reply = .{},

    fn acquire(s: *State) void {
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
    }

    pub fn observe(s: *State, seq: u64, kind: anytype, seen: u64) void {
        s.acquire();
        defer s.lock.unlock();
        if (seq <= s.closed_through) return;
        std.debug.assert(s.active == null);
        const index = @backingInt(kind);
        s.active = .{ .seq = seq, .kind = index, .seen = seen };
        if (s.reply.observed == 0) s.reply.first = seq;
        s.reply.last = seq;
        s.reply.observed +|= 1;
        if (s.previous_release) |previous| s.reply.kinds[index].release_to_post.add(seen -% previous);
    }

    pub fn beginPhase(s: *State, seq: u64, phase: Phase, now: u64) void {
        s.mark(seq, phase, now, false);
    }

    pub fn endPhase(s: *State, seq: u64, phase: Phase, now: u64) void {
        s.mark(seq, phase, now, true);
    }

    fn mark(s: *State, seq: u64, phase: Phase, now: u64, end: bool) void {
        s.acquire();
        defer s.lock.unlock();
        const a = if (s.active) |*active| active else return;
        if (a.seq != seq) return;
        const span = switch (phase) {
            .send => &a.send,
            .arrival => &a.arrival,
        };
        if (end) span.end = now else span.begin = now;
    }

    pub fn publish(s: *State, owner: anytype, seq: u64) bool {
        s.acquire();
        defer s.lock.unlock();
        var released: u64 = undefined;
        if (!owner.handoff.publishTimed(seq, &released)) return false;
        if (s.active) |a| if (a.seq == seq) {
            s.complete(released, false);
            s.previous_release = released;
        };
        return true;
    }

    fn complete(s: *State, now: u64, partial: bool) void {
        const a = s.active.?;
        const k = &s.reply.kinds[a.kind];
        const held = now -% a.seen;
        s.reply.held.add(held);
        k.held.add(held);
        k.send.add(a.send.duration(now));
        k.arrival.add(a.arrival.duration(now));
        if (partial) s.reply.partial +|= 1;
        s.active = null;
    }

    pub fn takeReply(s: *State, now: u64) Reply {
        s.acquire();
        defer s.lock.unlock();
        if (s.active != null) s.complete(now, true);
        const reply = s.reply;
        s.closed_through = @max(s.closed_through, reply.last);
        s.reply = .{};
        return reply;
    }

    pub fn finishReply(s: *State, rank: u32, now: u64) void {
        const reply = s.takeReply(now);
        reply.print(rank);
    }
};

const JobKind = enum { exchange, reduce, pick, send, wait };
const Fake = struct {
    released: u64 = 240,
    seq: u64 = 0,
    succeeds: bool = true,

    pub fn publishTimed(f: *Fake, seq: u64, released: *u64) bool {
        f.seq = seq;
        released.* = f.released;
        return f.succeeds;
    }
};

test "pair stats bucket microseconds at powers of two" {
    const ticks = [_]u64{ 0, 1, 23, 24, 47, 48, 95, 96, 191, 192 };
    const buckets = [_]usize{ 0, 0, 0, 1, 1, 2, 2, 3, 3, 4 };
    var metric: Metric = .{};
    for (ticks, buckets) |t, b| {
        try std.testing.expectEqual(b, Metric.bucket(t));
        metric.add(t);
    }
    try std.testing.expectEqual(@as(u64, ticks.len), metric.count);
    try std.testing.expectEqual(@as(u64, 717), metric.sum_ticks);
    try std.testing.expectEqual(@as(u64, 192), metric.max_ticks);
    try std.testing.expectEqualSlices(u64, &.{ 3, 2, 2, 2, 1 }, metric.buckets[0..5]);
}

test "pair stats reset each reply and retain the preceding release" {
    var state: State = .{};
    var owner = struct { handoff: Fake = .{} }{};
    const seq: u64 = (1 << 32) + 7;
    state.observe(seq, JobKind.exchange, 24);
    state.beginPhase(seq, .send, 48);
    state.endPhase(seq, .send, 96);
    state.beginPhase(seq, .arrival, 120);
    state.endPhase(seq, .arrival, 216);
    try std.testing.expect(state.publish(&owner, seq));
    const first = state.takeReply(300);
    try std.testing.expectEqual(seq, first.first);
    try std.testing.expectEqual(seq, first.last);
    try std.testing.expectEqual(seq, owner.handoff.seq);
    try std.testing.expectEqual(@as(u64, 216), first.kinds[0].held.sum_ticks);
    try std.testing.expectEqual(@as(u64, 48), first.kinds[0].send.sum_ticks);
    try std.testing.expectEqual(@as(u64, 96), first.kinds[0].arrival.sum_ticks);
    try std.testing.expectEqual(@as(u64, 0), first.kinds[0].release_to_post.count);
    owner.handoff.released = 600;
    state.observe(seq + 1, JobKind.pick, 480);
    try std.testing.expect(state.publish(&owner, seq + 1));
    const second = state.takeReply(700);
    try std.testing.expectEqual(seq + 1, second.first);
    try std.testing.expectEqual(seq + 1, second.last);
    try std.testing.expectEqual(@as(u64, 1), second.held.count);
    try std.testing.expectEqual(@as(u64, 0), second.kinds[0].held.count);
    try std.testing.expectEqual(@as(u64, 240), second.kinds[2].release_to_post.sum_ticks);
    try std.testing.expectEqual(@as(u64, 1), second.kinds[2].release_to_post.buckets[4]);
    try std.testing.expectEqual(@as(u64, 0), second.kinds[2].send.sum_ticks);
    const empty = state.takeReply(800);
    try std.testing.expectEqual(@as(u64, 0), empty.first);
    try std.testing.expectEqual(@as(u64, 0), empty.last);
    try std.testing.expectEqual(@as(u64, 0), empty.held.count);
}

test "pair stats close partial sends at reset and reject late completion" {
    var state: State = .{ .previous_release = 24 };
    var handoff: Fake = .{ .succeeds = false };
    var owner = .{ .handoff = &handoff };
    state.observe(8, JobKind.exchange, 48);
    state.beginPhase(8, .send, 72);
    try std.testing.expect(!state.publish(&owner, 8));
    const first = state.takeReply(168);
    try std.testing.expectEqual(@as(u64, 1), first.observed);
    try std.testing.expectEqual(@as(u64, 1), first.partial);
    try std.testing.expectEqual(@as(u64, 120), first.held.sum_ticks);
    try std.testing.expectEqual(@as(u64, 24), first.kinds[0].release_to_post.sum_ticks);
    try std.testing.expectEqual(@as(u64, 96), first.kinds[0].send.sum_ticks);
    try std.testing.expectEqual(@as(u64, 0), first.kinds[0].arrival.sum_ticks);
    state.endPhase(8, .send, 192);
    handoff.succeeds = true;
    try std.testing.expect(state.publish(&owner, 8));
    state.observe(8, JobKind.exchange, 216);
    const empty = state.takeReply(264);
    try std.testing.expectEqual(@as(u64, 0), empty.observed);
    try std.testing.expectEqual(@as(?u64, 24), state.previous_release);
}

test "pair stats partial arrival cannot leak into the following reply" {
    var state: State = .{};
    var owner = struct { handoff: Fake = .{} }{};
    state.observe(1, JobKind.reduce, 24);
    state.beginPhase(1, .send, 48);
    state.endPhase(1, .send, 72);
    state.beginPhase(1, .arrival, 96);
    const first = state.takeReply(192);
    try std.testing.expectEqual(@as(u64, 24), first.kinds[1].send.sum_ticks);
    try std.testing.expectEqual(@as(u64, 96), first.kinds[1].arrival.sum_ticks);
    state.observe(2, JobKind.pick, 216);
    state.endPhase(1, .arrival, 240);
    try std.testing.expect(state.publish(&owner, 1));
    owner.handoff.released = 288;
    try std.testing.expect(state.publish(&owner, 2));
    const second = state.takeReply(312);
    try std.testing.expectEqual(@as(u64, 2), second.first);
    try std.testing.expectEqual(@as(u64, 2), second.last);
    try std.testing.expectEqual(@as(u64, 1), second.observed);
    try std.testing.expectEqual(@as(u64, 0), second.partial);
    try std.testing.expectEqual(@as(u64, 0), second.kinds[1].held.count);
    try std.testing.expectEqual(@as(u64, 72), second.kinds[2].held.sum_ticks);
    try std.testing.expectEqual(@as(u64, 0), second.kinds[2].send.sum_ticks);
    try std.testing.expectEqual(@as(u64, 0), second.kinds[2].arrival.sum_ticks);
}
