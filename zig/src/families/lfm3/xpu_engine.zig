//! LFM3 behind the Intel GPU lane host: one sequence on the device, the prompt in windows, greedy argmax on the device.
const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("config.zig");
const model = @import("xpu_model.zig");

pub const Options = struct { context: usize = 4096, window_rows: u32 = 64 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    stream: xpu.Stream,
    r: xpu.rt.Runtime,
    loader: xpu.loader.Loader,
    c: cfg.Config,
    m: model.Model,
    max_len: usize,
    window_rows: u32,
    load_seconds: f64,

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
