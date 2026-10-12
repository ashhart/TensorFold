//! Phase timers around the lanes' contract: a `Target` and a `Pass` that time their inner calls and pass them through.

const std = @import("std");
const lanes = @import("lanes");
const iface = @import("draft/iface.zig");
const ph = @import("phases.zig");

pub const TimedTarget = struct {
    inner: iface.Target,
    p: *ph.Phases,
    gpa: std.mem.Allocator,
    io: std.Io,
    /// rows of the last window not kept yet
    pending: usize = 0,

    pub fn target(t: *TimedTarget) iface.Target {
        return .{ .ptr = t, .vtable = &vtable };
    }

    const vtable: iface.Target.VTable = .{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .admit = admit, .begin = begin, .piece = piece, .finish = finish, .tail = tail, .pieces = pieces, .replays = replays, .warm = warm };

    fn admit(p: *anyopaque, slot: u32, max_new: u64) void {
        self(p).inner.admit(slot, max_new);
    }

    fn begin(p: *anyopaque, slot: u32, ids: []const u32) anyerror!iface.Target.Begun {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.begin(slot, ids);
    }

    fn replays(p: *anyopaque, list: []const iface.Target.Final) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.runReplays(list);
    }

    fn pieces(p: *anyopaque, list: []const iface.Target.Piece) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.runPieces(list);
    }

    fn tail(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!?u32 {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.tail(slot, ids, sampling, draw);
    }

    fn warm(p: *anyopaque, asks: []const iface.Warm) void {
        self(p).inner.warm(asks);
    }

    fn piece(p: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.piece(slot, ids, start, end, save);
    }

    fn finish(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.finish(slot, ids, sampling, draw);
    }

    fn self(p: *anyopaque) *TimedTarget {
        return @ptrCast(@alignCast(p));
    }

    fn prefill(p: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(p);
        const tm = ph.start(t.p, .prefill);
        defer tm.stop();
        return t.inner.prefill(slot, ids, sampling, draw);
    }

    fn window(p: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self(p);
        if (t.pending > 0) t.p.round(t.gpa, t.io, t.pending);
        var rows: usize = 0;
        for (segments) |s| rows += s.tokens.len;
        t.pending = rows;
        const tm = ph.start(t.p, .window);
        defer tm.stop();
        return t.inner.window(segments, choices);
    }

    fn keep(p: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self(p);
        {
            const tm = ph.start(t.p, .commit);
            defer tm.stop();
            try t.inner.keep(slot, path);
        }
        t.p.round(t.gpa, t.io, path.len);
        t.pending = 0;
    }

    fn taps(p: *anyopaque, slot: u32) iface.Taps {
        return self(p).inner.taps(slot);
    }

    fn release(p: *anyopaque, slot: u32) void {
        const t = self(p);
        t.inner.release(slot);
        t.pending = 0;
        t.p.flush(t.gpa, t.io);
    }
};

pub const TimedPass = struct {
    inner: iface.Pass,
    p: *ph.Phases,

    pub fn pass(t: *TimedPass) iface.Pass {
        return .{ .ptr = t, .vtable = if (t.inner.vtable.begin != null) &vtable_begin else &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov };
    const vtable_begin: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = markov, .begin = begin, .collect = collect, .arm = arm };

    fn self(p: *anyopaque) *TimedPass {
        return @ptrCast(@alignCast(p));
    }

    fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .draft);
        defer tm.stop();
        return t.inner.ingest(slot, start, taps, rows);
    }

    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .@"draft.pass");
        defer tm.stop();
        return t.inner.propose(asks, out);
    }

    /// The speculative pass as the inner pass gives it: launched under `spec`, collected under `draft.pass`.
    fn begin(p: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .spec);
        defer tm.stop();
        const f = t.inner.vtable.begin orelse return error.NoBegin;
        return f(t.inner.ptr, asks);
    }

    fn collect(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const t = self(p);
        const tm = ph.start(t.p, .@"draft.pass");
        defer tm.stop();
        const f = t.inner.vtable.collect orelse return error.NoBegin;
        return f(t.inner.ptr, out);
    }

    fn reset(p: *anyopaque, slot: u32) void {
        self(p).inner.reset(slot);
    }

    fn arm(p: *anyopaque, slot: u32, start: u64, rows: u32, sampled: bool) void {
        const t = self(p);
        if (t.inner.vtable.arm) |f| f(t.inner.ptr, slot, start, rows, sampled);
    }

    fn markov(p: *anyopaque) ?@import("draft/dspark.zig").Markov {
        return self(p).inner.markov();
    }
};
