//! Nemotron-H's lane costs, timed at load on scratch lanes: every rank runs the same rounds, rank 0 keeps the times.

const std = @import("std");
const serve = @import("hip_serve");
const lanes = @import("lanes.zig");
const head = @import("head.zig");

const shared = serve.costs;
const draw = serve.draw;
const Hip = lanes.Hip;
pub const Costs = shared.Costs;

/// A scratch lane's kept length in the timings, and the positions it covers past it (the head steps' too).
const at = 16;
const room = 2 * shared.window * 4;

const Rig = struct {
    h: *Hip,
    ls: [shared.most_streams]*lanes.Lane = undefined,
    made: usize = 0,
    tokens: [shared.most_streams][lanes.max_window]u32 = @splat(@splat(1)),
    reqs: [shared.most_streams * lanes.max_window]draw.Request = undefined,
    drawn: [shared.most_streams * lanes.max_window]u32 = undefined,

    fn deinit(r: *Rig) void {
        for (r.ls[0..r.made]) |l| r.h.free(l);
    }

    /// Mean ms of a verify's work on this rank (the round, the head's logits, rank 0's draw) of the given widths.
    pub fn time(r: *Rig, widths: []const usize) !f64 {
        var toks: [shared.most_streams][]const u32 = undefined;
        var total: usize = 0;
        for (widths, 0..) |w, i| {
            toks[i] = r.tokens[i][0..w];
            for (0..w) |j| r.reqs[total + j] = .{ .sampling = null, .position = at + j + 1 };
            total += w;
        }
        const run: Run = .{ .r = r, .ls = r.ls[0..widths.len], .toks = toks[0..widths.len], .total = total };
        return shared.mean(run, Run.go);
    }

    const Run = struct {
        r: *Rig,
        ls: []const *lanes.Lane,
        toks: []const []const u32,
        total: usize,

        fn go(x: Run) anyerror!void {
            const h = x.r.h;
            // each round from the same kept length: the window's rows are dropped, as a keep of none does
            for (x.ls) |l| l.s.len = at;
            const y = try h.windows(x.ls, x.toks);
            for (x.ls) |l| l.pending = null;
            if (h.e.o.rank == 0) try h.sampler.draw(
                h.e.ops(&h.e.arena),
                h.e.stream,
                y,
                x.r.reqs[0..x.total],
                x.r.drawn[0..x.total],
                false,
            );
            try h.e.stream.synchronize();
        }
    };

    /// One chained head step: a lone chain of the deepest depth from a scratch lane's (garbage) row.
    fn step(r: *Rig, hd: *head.Head) !f64 {
        const run: Chain = .{ .r = r, .hd = hd };
        return try shared.mean(run, Chain.go) / head.max_depth;
    }

    const Chain = struct {
        r: *Rig,
        hd: *head.Head,

        fn go(x: Chain) anyerror!void {
            const h = x.r.h;
            const l = x.r.ls[0];
            l.s.len = at;
            l.s.mtp_len = at;
            var out: [head.max_depth]u32 = undefined;
            var jobs = [_]head.Job{.{ .s = &l.s, .hidden = l.hidden.base(), .next = x.r.tokens[0][0..1], .out = &out }};
            try h.e.fresh();
            try x.hd.chains(h.e.ops(&h.e.arena), &jobs);
        }
    };
};

/// The forwards and the head timed on scratch lanes of `h` (before any request and before rank 0 links its group).
pub fn measure(h: *Hip, out: *Costs) !void {
    var r: Rig = .{ .h = h };
    defer r.deinit();
    const e = h.e;
    const streams = @min(shared.most_streams, e.plan.slots);
    while (r.made < streams) : (r.made += 1) {
        const l = try h.make();
        r.ls[r.made] = l;
        try l.s.cover(&e.pool, at + room);
        l.s.len = at;
    }
    try shared.forwards(&r, @min(lanes.max_window, e.plan.rows), streams, e.plan.rows, out);
    if (h.head) |hd| out.step_ms = try r.step(hd);
}
