//! Gemma 4 E4B as an engine of the XPU lane host (native/xpu_lane.zig): one stream, card argmax or top-k, host draws.
const std = @import("std");
const fwd = @import("xpu_forward.zig");
const cfg = @import("xpu_config.zig");
const dev = @import("xpu_dev.zig");

pub const Runner = fwd.Runner;
pub const config = cfg;

/// A kept prompt state: the global layers' K/V rows [0, at) and the sliding layers' ring slots (first `min(at, 1536)`).
pub const Snap = struct {
    at: u32,
    k: [24]*anyopaque,
    v: [24]*anyopaque,
    bytes: u64,
};

fn rowBytes(layer: usize) u64 {
    return if (cfg.isGlobal(layer)) cfg.kv_heads * cfg.dim_g * 2 else cfg.kv_heads * cfg.dim_s * 2;
}

fn rowsKept(layer: usize, at: u32) u64 {
    return if (cfg.isGlobal(layer)) at else @min(at, cfg.ring);
}

/// The lane core drafts by copying the context (no head): windows of up to 16 rows are verified, bit for bit the steps.
const OneSeq = struct {};

pub const Engine = struct {
    pub const Snapshot = Snap;
    pub const Sequence = OneSeq;
    /// The lane core drafts by copying the context (no head): windows of up to 16 rows, bit for bit the steps.
    pub const copy_only = true;
    own: OneSeq = .{},
    head: ?u8 = null,
    r: *fwd.Runner,
    /// The lane backend reads the position here (the runner's, kept in step).
    m: struct { pos: u32 = 0 } = .{},
    max_len: usize,
    sc: ?*fwd.Runner = null, // the decisions' sequence (a fork), made at the first score
    sc_ids: std.ArrayList(u32) = .empty, // the tokens its K/V holds

    pub fn init(dir: []const u8, context: u32) !*Engine {
        const e = try std.heap.page_allocator.create(Engine);
        errdefer std.heap.page_allocator.destroy(e);
        e.* = .{ .r = try fwd.Runner.init(dir, context), .max_len = context };
        return e;
    }

    pub fn deinit(e: *Engine) void {
        if (e.sc) |f| f.deinitFork();
        e.sc_ids.deinit(std.heap.page_allocator);
        e.r.deinit();
        std.heap.page_allocator.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) u64 {
        return e.r.d.allocated;
    }

    pub fn vocab(_: *const Engine) usize {
        return cfg.vocab;
    }

    pub fn reset(e: *Engine) !void {
        e.r.reset();
        e.m.pos = 0;
    }

    /// One token in; with `logits`, the next one's logits and greedy token kept.
    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try e.r.feed(&.{token}, logits);
        e.m.pos = e.r.pos;
    }

    pub fn sync(e: *Engine) !void {
        try e.r.d.sync();
    }

    pub fn argmax(e: *Engine) !u32 {
        return e.r.last;
    }

    /// The kept row's logits after the softcap, fp32.
    pub fn fetchLogits(e: *Engine, out: []f32) !void {
        @memcpy(out, try e.r.logits(0));
    }

    /// The `k` (<= 64) largest logits of the kept row, chosen on the card; softcapped, descending, ties to lower id.
    pub fn candidates(e: *Engine, k: u32, vals: []f32, ids: []u32) !void {
        try e.r.topK(0, k, vals, ids);
    }

    /// Prompt tokens in windows (the runner cuts them in 16 rows), no logits.
    pub fn prefillWindows(e: *Engine, ids: []const u32, _: usize) !void {
        try e.r.feed(ids, false);
        e.m.pos = e.r.pos;
        try e.r.d.sync();
    }

    /// Prompt tokens [from, to) at the position `from`; the last row's logits are kept when `to` ends the prompt.
    pub fn prefillSpan(e: *Engine, ids: []const u32, from: usize, to: usize, _: usize) !void {
        if (e.m.pos != from) return error.PositionMismatch;
        try e.r.feed(ids[from..to], to == ids.len);
        e.m.pos = e.r.pos;
    }

    pub fn snapBytes(_: *const Engine, at: u32) u64 {
        var n: u64 = 0;
        for (0..24) |l| n += 2 * rowsKept(l, at) * rowBytes(l);
        return n;
    }

    pub fn snapSave(e: *Engine, at: u32) !*Snap {
        if (e.m.pos != at) return error.SnapshotOutOfStep;
        const d = e.r.d;
        const s = try std.heap.page_allocator.create(Snap);
        errdefer std.heap.page_allocator.destroy(s);
        s.at = at;
        s.bytes = 0;
        for (0..24) |l| {
            const n = rowsKept(l, at) * rowBytes(l);
            s.k[l] = try d.alloc_(n);
            s.v[l] = try d.alloc_(n);
            try d.copy(s.k[l], e.r.kc[l], n);
            try d.copy(s.v[l], e.r.vc[l], n);
            s.bytes += 2 * n;
        }
        return s;
    }

    /// The runner's state becomes the snapshot's: K/V rows and ring slots copied back, the position `at`.
    pub fn snapRestore(e: *Engine, s: *const Snap) !void {
        const d = e.r.d;
        for (0..24) |l| {
            const n = rowsKept(l, s.at) * rowBytes(l);
            try d.copy(e.r.kc[l], s.k[l], n);
            try d.copy(e.r.vc[l], s.v[l], n);
        }
        e.r.pos = s.at;
        e.m.pos = s.at;
    }

    pub fn snapDrop(e: *Engine, s: *Snap) void {
        e.r.d.settle();
        for (0..24) |l| {
            e.r.d.g.free(s.k[l]);
            e.r.d.g.free(s.v[l]);
        }
        std.heap.page_allocator.destroy(s);
    }

    /// A decision: the labels' logits after `prompt` (softcapped) and the row's log-sum-exp, on the second sequence.
    pub fn score(e: *Engine, prompt: []const u32, labels: []const u32, logits: []f64, _: bool) !f64 {
        if (prompt.len == 0 or prompt.len + 1 > e.max_len) return error.PromptTooLong;
        const r = e.sc orelse blk: {
            e.sc = try e.r.fork();
            break :blk e.sc.?;
        };
        const gpa = std.heap.page_allocator;
        var l: usize = 0;
        while (l < e.sc_ids.items.len and l + 1 < prompt.len and e.sc_ids.items[l] == prompt[l]) l += 1;
        // the ring holds positions back to (end - 1536): resume only if the old end is not too far past the prefix
        if (e.sc_ids.items.len > l + 1024) l = 0;
        e.sc_ids.clearRetainingCapacity();
        r.pos = @intCast(l);
        try r.feed(prompt[l..], true);
        try e.sc_ids.appendSlice(gpa, prompt);
        const row = try r.logits(0);
        var peak: f64 = -std.math.inf(f64);
        for (row) |x| peak = @max(peak, x);
        var sum: f64 = 0;
        for (row) |x| sum += @exp(@as(f64, x) - peak);
        for (labels, logits) |t, *o| o.* = row[t];
        return peak + @log(sum);
    }

    // -- windows for the lane core's copy drafts --------------------------------------------------------------------

    pub fn newSeq(_: *Engine) !*OneSeq {
        return error.OneStream;
    }

    pub fn freeSeq(_: *Engine, _: *OneSeq) void {}

    pub fn bind(_: *Engine, _: *OneSeq) void {}

    /// One token in, committed; the next one's logits kept.
    pub fn stepRow(e: *Engine, token: u32) !void {
        try e.feed(token, true);
    }

    /// A window at the position, not committed: every row's greedy token and logits are kept.
    pub fn verifyRows(e: *Engine, ids: []const u32) !void {
        try e.r.verify(ids);
    }

    pub fn rowTokens(e: *Engine, out: []i32) !void {
        try e.r.rowTokens(out);
    }

    /// The `k` largest logits of the last window's row `row` (card, softcapped, descending, ties to the lower id).
    pub fn rowCandidates(e: *Engine, row: u32, k: u32, vals: []f32, ids: []u32) !void {
        try e.r.topK(row, k, vals, ids);
    }

    pub fn rowLogits(e: *Engine, out: []f32) !void {
        try e.r.rowLogits(out);
    }

    /// Commit the first `c` rows of the last window.
    pub fn keepRows(e: *Engine, c: u32) !void {
        e.r.pos += c;
        e.m.pos = e.r.pos;
    }

    pub fn draftKept(_: *Engine, _: []const u32, _: u32, _: u32, _: []u32) !usize {
        return error.NoDraftHead;
    }

    pub fn draftFirst(_: *Engine, _: u32, _: u32, _: []u32) !usize {
        return error.NoDraftHead;
    }

    /// Forward ms of windows of 1 to 16 rows (the core prices copy drafts with them); returns the head step (none: 0).
    pub fn timeRounds(e: *Engine, ms: []f64) !f64 {
        const start = e.r.pos;
        defer e.r.pos = start;
        var ids: [16]u32 = @splat(100);
        for (ms, 1..) |*m, w| {
            try e.r.verify(ids[0..w]);
            const t0 = dev.now();
            for (0..8) |_| try e.r.verify(ids[0..w]);
            m.* = (dev.now() - t0) / 8.0 * 1e3;
        }
        return 0;
    }
};
