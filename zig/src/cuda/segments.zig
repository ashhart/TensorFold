//! Staggered prompt segments on CUDA streams for any family: segment k's mixer waits on k-1's event, the rest overlaps.
const std = @import("std");
const stagger = @import("stagger");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const Event = @import("stream.zig").Event;

pub const MAX = stagger.MAX;
pub const Wait = stagger.Wait;
pub const Call = stagger.Call;
pub const chunked = stagger.chunked;
pub const chunkRows = stagger.chunkRows;

/// A segment and the stream its work goes on.
pub const Lane = struct { k: usize, s: Stream };

/// Side streams and events for up to `n` segments a call, made once and reused by every call.
pub const Runner = struct {
    side: [MAX - 1]Stream = undefined,
    handoff: [MAX - 1]Event = undefined, // handoff[k]: segment k's latest mixer, which segment k+1 waits for
    join: [MAX - 1]Event = undefined, // join[k]: side segment k+1's end, which the caller's stream waits for
    fork: Event, // the caller's work before a call
    n: usize = 1,

    pub fn init(d: *const Driver, n: usize) !Runner {
        if (n == 0 or n > MAX) return error.Segments;
        var r: Runner = .{ .fork = try Event.init(d, false) };
        errdefer r.deinit();
        while (r.n < n) : (r.n += 1) {
            const j = r.n - 1;
            r.side[j] = try Stream.init(d, true);
            r.handoff[j] = Event.init(d, false) catch |err| {
                r.side[j].deinit();
                return err;
            };
            r.join[j] = Event.init(d, false) catch |err| {
                r.handoff[j].deinit();
                r.side[j].deinit();
                return err;
            };
        }
        return r;
    }

    pub fn deinit(r: *Runner) void {
        for (0..r.n - 1) |j| {
            r.join[j].deinit();
            r.handoff[j].deinit();
            r.side[j].deinit();
        }
        r.fork.deinit();
        r.* = undefined;
    }

    /// The stream segment k runs on beside the caller's.
    pub fn stream(r: *const Runner, caller: Stream, k: usize) Stream {
        return if (k == 0) caller else r.side[k - 1];
    }

    /// One call: `parts` segments through core/segments.zig's hooks on `fam`, forked from and joined back to `caller`.
    pub fn run(r: *const Runner, caller: Stream, parts: usize, layers: usize, fam: anytype) !void {
        if (parts == 0 or parts > r.n) return error.Segments;
        var lanes: [MAX]Lane = undefined;
        for (0..parts) |k| lanes[k] = .{ .k = k, .s = r.stream(caller, k) };
        const ops: Ops = .{ .r = r, .lanes = lanes[0..parts] };
        return stage(ops, fam, lanes[0..parts], layers);
    }
};

/// The events between real streams: lanes[0] is the caller's stream.
const Ops = struct {
    r: *const Runner,
    lanes: []const Lane,

    pub fn fork(o: Ops) !void {
        try o.r.fork.record(o.lanes[0].s);
        for (o.lanes[1..]) |l| try l.s.wait(o.r.fork);
    }
    pub fn signal(o: Ops, k: usize) !void {
        try o.r.handoff[k].record(o.lanes[k].s);
    }
    pub fn waitFor(o: Ops, k: usize) !void {
        try o.lanes[k].s.wait(o.r.handoff[k - 1]);
    }
    pub fn join(o: Ops) !void {
        for (o.lanes[1..], 0..) |l, j| {
            try o.r.join[j].record(l.s);
            try o.lanes[0].s.wait(o.r.join[j]);
        }
    }
    /// Blocks until every segment's queued work has ended (the error path, before buffers are reused).
    pub fn drain(o: Ops) void {
        for (o.lanes) |l| l.s.synchronize() catch {};
    }
};

/// The schedule over `lanes`, with `ops` placing the fork, handoff and join events (real streams or the test's).
fn stage(ops: anytype, fam: anytype, lanes: []Lane, layers: usize) !void {
    const n = lanes.len;
    const X = struct {
        ops: @TypeOf(ops),
        fam: @TypeOf(fam),
        lanes: []Lane,

        pub fn begin(x: *const @This(), k: usize) !void {
            try x.fam.begin(&x.lanes[k]);
        }
        pub fn wait(x: *const @This(), i: usize) Wait {
            return x.fam.wait(i);
        }
        pub fn waitFor(x: *const @This(), k: usize, i: usize) !void {
            try x.ops.waitFor(k);
            try x.fam.handoff(&x.lanes[k], i);
        }
        pub fn signal(x: *const @This(), k: usize, _: usize) !void {
            try x.ops.signal(k);
        }
        pub fn pre(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.pre(&x.lanes[k], i);
        }
        pub fn mixer(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.mixer(&x.lanes[k], i);
        }
        pub fn post(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.post(&x.lanes[k], i);
        }
        pub fn finish(x: *const @This(), k: usize) !void {
            try x.fam.finish(&x.lanes[k]);
        }
    };
    const x: X = .{ .ops = ops, .fam = fam, .lanes = lanes };
    // one handoff event a pair serves every layer: a CUDA wait takes the latest record, and drive queues it in order
    if (n > 1) try ops.fork();
    stagger.drive(n, layers, &x) catch |err| { // join what was queued and let it end before the buffers are reused
        if (n > 1) ops.join() catch {};
        ops.drain();
        return err;
    };
    if (n > 1) try ops.join();
}

/// CUDA's ordering in the tests: a record marks its stream's position; a wait takes the event's latest record when queued.
const Sim = struct {
    const Kind = enum { before, begin, pre, mixer, post, finish, after };
    const Work = struct { call: usize, k: usize, kind: Kind, i: usize };
    const Op = union(enum) { work: Work, record: usize, wait: ?[2]usize };
    const events = 2 * MAX;
    ops: [MAX][160]Op = undefined,
    len: [MAX]usize = @splat(0),
    latest: [events]?[2]usize = @splat(null),

    fn put(s: *Sim, stream: usize, op: Op) void {
        s.ops[stream][s.len[stream]] = op;
        s.len[stream] += 1;
    }
    fn record(s: *Sim, ev: usize, stream: usize) void {
        s.put(stream, .{ .record = ev });
        s.latest[ev] = .{ stream, s.len[stream] };
    }
    fn waitOn(s: *Sim, stream: usize, ev: usize) void {
        s.put(stream, .{ .wait = s.latest[ev] });
    }

    /// Runs the streams one op at a time in an order `seed` picks, each work op through `check`; false on a deadlock.
    fn execute(s: *const Sim, n: usize, seed: u64, check: anytype) !bool {
        var prng = std.Random.DefaultPrng.init(seed);
        var pc: [MAX]usize = @splat(0);
        while (true) {
            var ready: [MAX]usize = undefined;
            var m: usize = 0;
            for (0..n) |k| {
                if (pc[k] == s.len[k]) continue;
                const op = s.ops[k][pc[k]];
                if (op == .wait) if (op.wait) |w| if (pc[w[0]] < w[1]) continue;
                ready[m] = k;
                m += 1;
            }
            if (m == 0) break;
            const k = ready[prng.random().uintLessThan(usize, m)];
            const op = s.ops[k][pc[k]];
            if (op == .work) try check.see(op.work);
            pc[k] += 1;
        }
        for (0..n) |k| if (pc[k] < s.len[k]) return false;
        return true;
    }
};

/// The simulated runner's events: fork 0, handoff 1 + k, join MAX + k.
const SimOps = struct {
    sim: *Sim,
    n: usize,

    pub fn fork(o: SimOps) !void {
        o.sim.record(0, 0);
        for (1..o.n) |k| o.sim.waitOn(k, 0);
    }
    pub fn signal(o: SimOps, k: usize) !void {
        o.sim.record(1 + k, k);
    }
    pub fn waitFor(o: SimOps, k: usize) !void {
        o.sim.waitOn(k, 1 + (k - 1));
    }
    pub fn join(o: SimOps) !void {
        for (1..o.n) |k| {
            o.sim.record(MAX + k, k);
            o.sim.waitOn(0, MAX + k);
        }
    }
    pub fn drain(_: SimOps) void {}
};

/// A family that queues one work op per hook on its segment's stream.
const SimFam = struct {
    sim: *Sim,
    call: usize = 0,
    early: usize,

    fn mark(f: *SimFam, k: usize, kind: Sim.Kind, i: usize) void {
        f.sim.put(k, .{ .work = .{ .call = f.call, .k = k, .kind = kind, .i = i } });
    }
    pub fn begin(f: *SimFam, l: *const Lane) !void {
        f.mark(l.k, .begin, 0);
    }
    pub fn wait(f: *SimFam, i: usize) Wait {
        return if (i == f.early) .pre else .mixer;
    }
    pub fn handoff(_: *SimFam, _: *const Lane, _: usize) !void {}
    pub fn pre(f: *SimFam, l: *const Lane, i: usize) !void {
        f.mark(l.k, .pre, i);
    }
    pub fn mixer(f: *SimFam, l: *const Lane, i: usize) !void {
        f.mark(l.k, .mixer, i);
    }
    pub fn post(f: *SimFam, l: *const Lane, i: usize) !void {
        f.mark(l.k, .post, i);
    }
    pub fn finish(f: *SimFam, l: *const Lane) !void {
        f.mark(l.k, .finish, 0);
    }
};

/// What must hold whatever order the streams run in.
const Check = struct {
    n: usize,
    early: usize,
    before: [2]bool = @splat(false),
    mixed: [2][MAX]usize = @splat(@splat(0)), // mixers run, per call and segment
    ended: [2]usize = @splat(0),

    fn see(c: *Check, w: Sim.Work) !void {
        const t = std.testing;
        switch (w.kind) {
            .before => c.before[w.call] = true,
            .begin => { // side segments start after the caller's earlier work, and after the last call ended
                try t.expect(c.before[w.call]);
                if (w.call > 0) try t.expectEqual(c.n, c.ended[w.call - 1]);
            },
            .pre => if (w.k > 0 and w.i == c.early) try t.expect(c.mixed[w.call][w.k - 1] > w.i),
            .mixer => {
                if (w.k > 0) try t.expect(c.mixed[w.call][w.k - 1] > w.i);
                c.mixed[w.call][w.k] = w.i + 1;
            },
            .post => {},
            .finish => c.ended[w.call] += 1,
            .after => try t.expectEqual(c.n, c.ended[w.call]), // the caller's later work sees every segment
        }
    }
};

// Two calls with reused events, in many random stream orders: every dependency in Check holds and no stream deadlocks.
test "the CUDA schedule keeps every dependency with reused events" {
    const layers = 6;
    for (1..MAX + 1) |n| {
        var sim: Sim = .{};
        var fam: SimFam = .{ .sim = &sim, .early = 1 };
        var lanes: [MAX]Lane = undefined;
        for (0..n) |k| lanes[k] = .{ .k = k, .s = undefined };
        for (0..2) |call| {
            fam.call = call;
            sim.put(0, .{ .work = .{ .call = call, .k = 0, .kind = .before, .i = 0 } });
            try stage(SimOps{ .sim = &sim, .n = n }, &fam, lanes[0..n], layers);
            sim.put(0, .{ .work = .{ .call = call, .k = 0, .kind = .after, .i = 0 } });
        }
        for (0..200) |seed| {
            var check: Check = .{ .n = n, .early = fam.early };
            try std.testing.expect(try sim.execute(n, seed, &check));
            try std.testing.expectEqual(@as(usize, n), check.ended[1]);
        }
    }
}

// The same streams without the handoff waits let some order run a mixer before the segment before's: the check bites.
test "the simulated check catches a missing handoff" {
    const NoWait = struct {
        inner: SimOps,
        pub fn fork(o: @This()) !void {
            try o.inner.fork();
        }
        pub fn signal(o: @This(), k: usize) !void {
            try o.inner.signal(k);
        }
        pub fn waitFor(_: @This(), _: usize) !void {}
        pub fn join(o: @This()) !void {
            try o.inner.join();
        }
        pub fn drain(_: @This()) void {}
    };
    var sim: Sim = .{};
    var fam: SimFam = .{ .sim = &sim, .early = 1 };
    var lanes: [2]Lane = .{ .{ .k = 0, .s = undefined }, .{ .k = 1, .s = undefined } };
    sim.put(0, .{ .work = .{ .call = 0, .k = 0, .kind = .before, .i = 0 } });
    try stage(NoWait{ .inner = .{ .sim = &sim, .n = 2 } }, &fam, &lanes, 4);
    var caught = false;
    for (0..200) |seed| {
        var check: Check = .{ .n = 2, .early = fam.early };
        _ = sim.execute(2, seed, &check) catch {
            caught = true;
            break;
        };
    }
    try std.testing.expect(caught);
}
