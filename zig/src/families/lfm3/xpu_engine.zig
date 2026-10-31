//! LFM3 behind the Intel GPU lane host: a sequence a stream, windows verified bit for bit the steps.
const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("config.zig");
const model = @import("xpu_model.zig");

pub const Options = struct { context: usize = 4096, window_rows: u32 = 64 };

pub const Engine = struct {
    pub const Sequence = model.Seq;
    pub const Snapshot = model.Snap;
    /// The lane core drafts by copying the context (no head): windows of up to 16 rows, bit for bit the steps.
    pub const copy_only = true;
    head: ?u8 = null,
    gpa: std.mem.Allocator,
    stream: xpu.Stream,
    r: xpu.rt.Runtime,
    loader: xpu.loader.Loader,
    c: cfg.Config,
    m: model.Model,
    max_len: usize,
    window_rows: u32,
    load_seconds: f64,
    /// The first stream's state, held here so the lane host can tell it from the streams it adds (newSeq).
    own: Sequence,
    /// Each slot's sequence (0 is `own`); verifyMulti names streams by slot.
    slots: [model.max_streams]?*Sequence,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const xpu.Context, dir: []const u8, o: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.c = try cfg.Config.read(gpa, io, dir);
        errdefer e.c.deinit();
        if (e.c.kind != .text) return error.UnsupportedFormat;
        const q = e.c.quant.global;
        if (q.bits != 4 or q.group != 64 or e.c.quant.modules.count() != 0) return error.UnsupportedFormat;
        e.stream = try xpu.Stream.init(ctx);
        errdefer e.stream.deinit();
        e.r = try xpu.rt.Runtime.init(ctx, e.stream);
        errdefer e.r.deinit();
        e.loader = try xpu.loader.Loader.init(gpa, &e.r, dir);
        e.window_rows = o.window_rows;
        e.m = try model.Model.load(gpa, &e.r, &e.loader, &e.c, @intCast(o.context), o.window_rows);
        e.own = e.m.own.*;
        gpa.destroy(e.m.own);
        e.m.own = &e.own;
        e.m.seq = &e.own;
        e.head = null;
        e.slots = @splat(null);
        e.slots[0] = &e.own;
        e.max_len = o.context;
        e.load_seconds = xpu.decode.seconds(io, t0);
        return e;
    }

    /// The process exits right after: device memory goes with it once the queue is drained.
    pub fn deinit(e: *Engine) void {
        e.r.deinit();
        e.stream.deinit();
        e.c.deinit();
        e.gpa.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) u64 {
        return e.loader.total;
    }

    pub fn vocab(_: *const Engine) usize {
        return cfg.vocab;
    }

    pub fn reset(e: *Engine) !void {
        try e.m.reset();
    }

    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try e.m.forward(&.{token}, logits, null);
    }

    pub fn sync(e: *Engine) !void {
        try e.r.sync();
    }

    pub fn argmax(e: *Engine) !u32 {
        return e.m.argmax();
    }

    pub fn fetchLogits(e: *Engine, out: []f32) !void {
        try e.m.fetchLogits(out);
    }

    pub fn newSeq(e: *Engine) !*Sequence {
        const slot = for (e.slots[1..], 1..) |q, i| {
            if (q == null) break i;
        } else return error.NoFreeSlot;
        const q = try e.m.newSeq();
        q.slot = @intCast(slot);
        e.slots[slot] = q;
        return q;
    }

    pub fn freeSeq(e: *Engine, q: *Sequence) void {
        e.slots[q.slot] = null;
        e.m.freeSeq(q);
    }

    pub fn slotOf(_: *const Engine, q: *const Sequence) u32 {
        return q.slot;
    }

    pub fn bind(e: *Engine, q: *Sequence) void {
        e.m.bind(q);
    }

    /// Prompt tokens [from, to) of the bound stream in windows of at most `rows`; the last window keeps logits.
    pub fn prefillSpan(e: *Engine, ids: []const u32, from: usize, to: usize, rows: u32) !void {
        if (e.m.pos != from) return error.PositionMismatch;
        const w = @min(rows, e.window_rows);
        var t = from;
        while (t < to) {
            const n = @min(w, to - t);
            try e.m.forward(ids[t .. t + n], t + n == ids.len, null);
            t += n;
        }
    }

    /// The pending token and drafts of the bound stream as one window; every row keeps its logits and argmax.
    pub fn verifyRows(e: *Engine, ids: []const u32) !void {
        try e.m.forwardSegs(&.{.{ .q = e.m.seq, .ids = ids }}, .rows, true, null);
    }

    /// Windows of several streams (by slot) in one forward over the weights, each row at its stream's position.
    pub fn verifyMulti(e: *Engine, slots: []const u32, windows: []const []const u32) !void {
        var segs: [model.max_streams]model.Seg = undefined;
        if (slots.len > segs.len) return error.WindowTooWide;
        for (slots, windows, segs[0..slots.len]) |sl, ids, *g| g.* = .{ .q = e.slots[sl] orelse return error.UnknownStream, .ids = ids };
        try e.m.forwardSegs(segs[0..slots.len], .rows, true, null);
    }

    /// Keeps the first c rows of the bound stream's last window.
    pub fn keepRows(e: *Engine, c: u32) !void {
        try e.m.keepRows(c);
    }

    /// The last window's greedy tokens, a row each (all streams' rows in order after verifyMulti).
    pub fn rowTokens(e: *Engine, out: []i32) !void {
        try e.m.rowTokens(out);
    }

    pub fn rowLogits(e: *Engine, out: []f32) !void {
        try e.m.rowLogits(out);
    }

    /// No head: the lane core's copies are the drafts.
    pub fn draftKept(_: *Engine, _: []const u32, _: u32, _: u32, _: []u32) !usize {
        return 0;
    }

    pub fn draftFirst(_: *Engine, _: u32, _: u32, _: []u32) !usize {
        return 0;
    }

    /// Window ms by rows on a scratch sequence: 1, 2, 4, 8, 16 timed (best of 3), the rest interpolated; no head: 0.
    pub fn timeRounds(e: *Engine, ms: []f64) !f64 {
        const prev = e.m.seq;
        const q = try e.m.newSeq();
        e.m.bind(q);
        defer {
            e.m.bind(prev);
            e.m.freeSeq(q);
        }
        const ids: [model.window_rows]u32 = @splat(1);
        const timed = [_]usize{ 1, 2, 4, 8, model.window_rows };
        std.debug.assert(ms.len == model.window_rows);
        for (timed) |w| {
            var best: f64 = std.math.inf(f64);
            for (0..3) |_| {
                try e.m.reset();
                try e.r.sync();
                var ts0: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.MONOTONIC, &ts0);
                try e.m.forwardSegs(&.{.{ .q = q, .ids = ids[0..w] }}, .rows, true, null);
                try e.r.sync();
                var ts1: std.c.timespec = undefined;
                _ = std.c.clock_gettime(.MONOTONIC, &ts1);
                best = @min(best, @as(f64, @floatFromInt(ts1.sec - ts0.sec)) * 1e3 + @as(f64, @floatFromInt(ts1.nsec - ts0.nsec)) / 1e6);
            }
            ms[w - 1] = best;
        }
        for (timed[0 .. timed.len - 1], timed[1..]) |a, b| for (a + 1..b) |w| {
            const f = @as(f64, @floatFromInt(w - a)) / @as(f64, @floatFromInt(b - a));
            ms[w - 1] = ms[a - 1] + f * (ms[b - 1] - ms[a - 1]);
        };
        try e.m.reset();
        return 0;
    }

    /// One decode step of the bound stream, with its logits.
    pub fn stepRow(e: *Engine, token: u32) !void {
        try e.m.forward(&.{token}, true, null);
    }

    pub fn snapBytes(e: *const Engine, at: u32) u64 {
        return e.m.snapBytes(at);
    }

    /// The bound stream's state at prompt position `at`, which it has just reached.
    pub fn snapSave(e: *Engine, at: u32) !*Snapshot {
        if (at != e.m.pos) return error.PositionMismatch;
        return e.m.snapSave();
    }

    pub fn snapRestore(e: *Engine, sn: *const Snapshot) !void {
        try e.m.snapRestore(sn);
    }

    pub fn snapDrop(e: *Engine, sn: *Snapshot) void {
        e.m.snapDrop(sn);
    }

    /// The prompt head in windows of at most `rows` tokens, without logits.
    pub fn prefillWindows(e: *Engine, head: []const u32, rows: u32) !void {
        var t: usize = 0;
        const w = @min(rows, e.window_rows);
        while (t < head.len) {
            const n = @min(w, head.len - t);
            try e.m.forward(head[t .. t + n], false, null);
            t += n;
        }
        try e.r.sync();
    }
};
