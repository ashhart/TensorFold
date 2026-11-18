//! DSpark's pass for the lanes (iface.Pass) over the Metal drafter: the target's taps in, blocks of drafts out.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const dspark = @import("dspark.zig");
const early_ = @import("early.zig");
const target_ = @import("target.zig");

const iface = host.iface;

/// How the lanes' begun pass runs: behind its window, queued at `begin`, or whole at `collect`.
const Begun = enum { none, chain, queued, later };

pub const Pass = struct {
    x: *dspark.Pass,
    /// the target whose taps the ingests name (by address)
    t: *const target_.Target,
    /// drafts behind the verify (null: every pass after its window)
    early: ?*early_.Early = null,
    passes: u64 = 0,
    ingests: u64 = 0,
    begun: Begun = .none,
    asks: [16]dspark.Ask = undefined,
    n_asks: usize = 0,

    pub fn pass(p: *Pass) iface.Pass {
        return .{ .ptr = p, .vtable = &.{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov, .begin = begin, .collect = collect, .arm = arm } };
    }

    /// The target's view of the drafter's queued work: a chain after a window, every queued buffer settled before one.
    pub fn behind(p: *Pass) target_.Behind {
        return .{ .ctx = p, .queue = queueChain, .drain = drain };
    }

    fn self(ptr: *anyopaque) *Pass {
        return @ptrCast(@alignCast(ptr));
    }

    /// Right after a window: each slot's next passes wide or narrow by its rows, then the chain behind it.
    fn queueChain(ctx: *anyopaque, group: []const iface.Segment, logits: mtl.Buffer, off: usize, taps: mtl.Buffer) void {
        const p = self(ctx);
        for (group) |g| p.x.note(g.slot, @intCast(g.tokens.len));
        if (p.early) |e| e.queue(group, logits, off, taps);
    }

    /// Every queued drafter buffer done and settled (before the target writes what one of them reads).
    fn drain(ctx: *anyopaque) anyerror!void {
        const p = self(ctx);
        if (p.early) |e| try e.drain();
        try p.x.settle();
    }

    /// Before other drafter work: a queued chain done, its rows the host's or put back (the drafter's own queue stays).
    fn chainDone(p: *Pass) !void {
        if (p.early) |e| try e.drain();
    }

    fn ingest(ptr: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const p = self(ptr);
        if (rows.len == 0) return;
        const at = p.t.bufferOf(taps.address) orelse return error.UnknownTaps;
        p.ingests += 1;
        if (p.early) |e| if (e.covers(slot, start, at.buf, at.off, rows)) return;
        try p.chainDone();
        // left queued: the target drains the drafter before it writes any taps again (Behind.drain)
        try p.x.ingestQueued(slot, start, .{ .buf = at.buf, .off = at.off, .rows = taps.rows }, rows);
    }

    fn convert(p: *Pass, asks: []const iface.Ask, out: []iface.Proposal, ys: []dspark.Proposal) !void {
        if (asks.len > p.asks.len or out.len != asks.len) return error.TooManySlots;
        for (asks, out, p.asks[0..asks.len], ys[0..asks.len]) |a, o, *x, *y| {
            x.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .params = a.params };
            y.* = .{ .drafts = o.drafts, .conf = o.conf, .cand = o.cand, .base = o.base };
        }
        p.n_asks = asks.len;
    }

    fn propose(ptr: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const p = self(ptr);
        if (p.begun != .none) return error.PassBegun;
        var ys: [16]dspark.Proposal = undefined;
        try p.convert(asks, out, &ys);
        try p.chainDone();
        p.passes += 1;
        try p.x.propose(p.asks[0..asks.len], ys[0..asks.len]);
    }

    /// The lanes' speculative pass: the chain queued behind its window, else queued now (one launch), else at collect.
    fn begin(ptr: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const p = self(ptr);
        if (p.begun != .none) return error.PassBegun;
        if (asks.len == 0 or asks.len > p.asks.len) return error.TooManySlots;
        for (asks, p.asks[0..asks.len]) |a, *x| x.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .params = a.params };
        p.n_asks = asks.len;
        p.passes += 1;
        if (p.early) |e| if (e.begin(asks)) {
            p.begun = .chain;
            return;
        };
        try p.chainDone();
        p.begun = if (try p.x.begin(p.asks[0..asks.len])) .queued else .later;
    }

    fn collect(ptr: *anyopaque, out: []iface.Proposal) anyerror!void {
        const p = self(ptr);
        const how = p.begun;
        p.begun = .none;
        var ys: [16]dspark.Proposal = undefined;
        if (out.len != p.n_asks) return error.BadOutput;
        for (out, ys[0..out.len]) |o, *y| y.* = .{ .drafts = o.drafts, .conf = o.conf, .cand = o.cand, .base = o.base };
        const asks = p.asks[0..p.n_asks];
        switch (how) {
            .none => return error.NothingBegun,
            .queued => return p.x.collect(ys[0..out.len]),
            .later => return p.x.propose(asks, ys[0..out.len]),
            .chain => {
                const e = p.early.?;
                // a chain settled since (the target drained it): its rows are the host's now, the pass runs here
                if (e.chain != null and e.chain.?.ask != null) return e.collect(ys[0..out.len]);
                try p.chainDone();
                return p.x.propose(asks, ys[0..out.len]);
            },
        }
    }

    fn arm(ptr: *anyopaque, slot: u32, start: u64, rows: u32, sampled: bool) void {
        const p = self(ptr);
        if (p.early) |e| e.arm(slot, start, rows, sampled);
    }

    fn reset(ptr: *anyopaque, slot: u32) void {
        const p = self(ptr);
        drain(p) catch |err| std.log.err("dsv41 drafter: queued work failed before a reset ({t})", .{err});
        p.x.reset(slot);
    }

    fn markov(ptr: *anyopaque) ?host.dspark.Markov {
        return self(ptr).x.markov();
    }
};

test "the pass's asks and proposals are the drafter's, field for field" {
    try std.testing.expectEqual(@sizeOf(iface.Ask), @sizeOf(dspark.Ask));
    try std.testing.expectEqual(@sizeOf(iface.Proposal), @sizeOf(dspark.Proposal));
}
