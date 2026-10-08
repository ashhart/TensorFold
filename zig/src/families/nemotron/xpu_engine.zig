//! Nemotron-H serial decode on an Intel GPU as one engine value: device checkpoint, one sequence.

const std = @import("std");
const xpu = @import("xpu");
const cfgm = @import("xpu_config.zig");
const model = @import("xpu_model.zig");
const nw = @import("xpu_win.zig");

const ld = xpu.loader;

pub const Options = struct {
    /// Positions of the KV caches (prompt and generated tokens).
    context: usize = 4096,
    progress: bool = false,
    /// Rows of the prompt windows when more than 16 (`--prefill`): the prefill GEMMs.
    window_rows: u32 = 0,
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    ctx: *const xpu.Context,
    stream: xpu.Stream,
    r: xpu.rt.Runtime,
    loader: ld.Loader,
    cfg: cfgm.Config,
    m: model.Model,
    w: nw.Win,
    max_len: usize,
    load_seconds: f64,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const xpu.Context, dir: []const u8, o: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.stream = try xpu.Stream.init(ctx);
        errdefer e.stream.deinit();
        e.r = try xpu.rt.Runtime.init(ctx, e.stream);
        errdefer e.r.deinit(); // drains the queue and forgets it, so the exit handler never calls into a closed driver
        e.loader = try ld.Loader.init(gpa, &e.r, dir);
        try model.checkFormat(&e.loader); // before the config: another format's config.json may not parse
        const text = try ld.readFile(gpa, dir, "config.json");
        e.cfg = (try cfgm.parse(gpa, text)).value;
        e.m = try model.Model.load(gpa, &e.r, &e.loader, e.cfg, @intCast(o.context));
        e.m.bf16_logits = true;
        if (o.window_rows > 16) nw.default_rows = o.window_rows;
        e.w = try nw.Win.init(gpa, &e.m);
        e.max_len = o.context;
        e.load_seconds = xpu.decode.seconds(io, t0);
        return e;
    }

    /// The process exits right after: device memory goes with it once the queue is drained.
    pub fn deinit(e: *Engine) void {
        e.r.deinit();
        e.stream.deinit();
        e.gpa.destroy(e);
    }

    pub fn deviceBytes(e: *const Engine) u64 {
        return e.loader.total;
    }

    pub fn vocab(e: *const Engine) usize {
        return e.cfg.vocab_size;
    }

    pub fn isEos(e: *const Engine, token: u32) bool {
        for (e.cfg.eos_token_id) |x| if (x == token) return true;
        return false;
    }

    pub fn reset(e: *Engine) !void {
        try e.w.reset(&e.m);
    }

    pub fn feed(e: *Engine, token: u32, logits: bool) !void {
        try nw.stopCheck(&e.r);
        try e.m.forward(token, logits);
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

    /// The prompt head in windows of `rows` tokens (above 16: prefill GEMMs), in place; last token fed as decode.
    pub fn prefillWindows(e: *Engine, head: []const u32, rows: u32) !void {
        var t: usize = 0;
        while (t < head.len) {
            try nw.stopCheck(&e.r);
            const n = @min(rows, head.len - t);
            try e.w.forward(&e.m, head[t .. t + n], true, 0);
            t += n;
        }
        try e.r.sync();
    }
};
