//! LFM3 behind the Intel GPU lane host: one sequence on the device, the prompt in windows, greedy argmax on the device.
const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("config.zig");
const model = @import("xpu_model.zig");

pub const Options = struct { context: usize = 4096, window_rows: u32 = 64 };

pub const Engine = struct {
    pub const Sequence = model.Seq;
    pub const Snapshot = model.Snap;
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
        return e.m.newSeq();
    }

    pub fn freeSeq(e: *Engine, q: *Sequence) void {
        e.m.freeSeq(q);
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

    /// The bound stream's next token after its last forward with logits (one row).
    pub fn rowTokens(e: *Engine, out: []i32) !void {
        if (out.len != 1) return error.WindowTooWide;
        out[0] = @intCast(try e.m.argmax());
    }

    pub fn rowLogits(e: *Engine, out: []f32) !void {
        if (out.len != cfg.vocab) return error.WindowTooWide;
        try e.m.fetchLogits(out);
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
