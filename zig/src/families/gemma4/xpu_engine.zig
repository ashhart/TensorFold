//! Gemma 4 E4B as an engine of the XPU lane host (native/xpu_lane.zig): one stream, card argmax or top-k, host draws.
const std = @import("std");
const fwd = @import("xpu_forward.zig");
const cfg = @import("xpu_config.zig");

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

pub const Engine = struct {
    pub const Snapshot = Snap;
    r: *fwd.Runner,
    /// The lane backend reads the position here (the runner's, kept in step).
    m: struct { pos: u32 = 0 } = .{},
    max_len: usize,

    pub fn init(dir: []const u8, context: u32) !*Engine {
        const e = try std.heap.page_allocator.create(Engine);
        errdefer std.heap.page_allocator.destroy(e);
        e.* = .{ .r = try fwd.Runner.init(dir, context), .max_len = context };
        return e;
    }

    pub fn deinit(e: *Engine) void {
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
};
