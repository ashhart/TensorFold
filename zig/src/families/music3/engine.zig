//! MiniMax Music 3 on one GPU: the three stages behind one load (AR frames, the DiT, the DAV), timed apart.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const prompt = @import("prompt.zig");
const ar_ = @import("ar.zig");
const dit_ = @import("dit.zig");
const dav_ = @import("dav.zig");

pub const Options = struct {
    format: wts.Format = .int8,
    depth_group: usize = 128, // int8 depth decoder and heads: a scale a group of 128 inputs (0: a row)
    load_ar: bool = true,
    load_dit: bool = true,
    load_dav: bool = true,
    dit: dit_.Options = .{},
    dav_tf32: bool = false,
    graphs: bool = true,
    sync_every: usize = 16,
    log: bool = true,
};

pub const Song = struct {
    audio: []f32, // [2, samples]
    samples: usize,
    frames: usize,
    stopped: bool,
    prompt_tokens: usize,
    prefill_ms: f64,
    ar_ms: f64,
    frames_per_s: f64,
    dit_ms: f64,
    dav_ms: f64,
    total_ms: f64,
    latent_frames: usize,
    peak_device_gb: f64,
};

pub const Progress = ar_.Ar.Progress;

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: *kern.Kernels,
    store: wts.Store,
    arw: ?wts.Ar = null,
    ditw: ?wts.Dit = null,
    davw: ?wts.Dav = null,
    tok: ?core.tokenizer.Tokenizer = null,
    ar: ?ar_.Ar = null,
    load_ms: f64 = 0,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const t0 = ar_.nowNs(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .o = o, .driver = undefined, .ctx = undefined, .stream = undefined, .k = undefined, .store = undefined };
        e.driver = try cuda.Driver.open();
        e.ctx = try cuda.Context.init(&e.driver, 0);
        e.stream = try cuda.Stream.init(&e.driver, true);
        e.k = try gpa.create(kern.Kernels);
        e.k.* = try kern.Kernels.load(gpa, &e.driver, &e.ctx);
        e.store = .{ .gpa = gpa, .d = &e.driver };
        if (o.load_ar) {
            e.arw = try wts.loadAr(gpa, io, &e.store, model_dir, o.format, o.depth_group, .{ .ctx = e, .f = quantize }, o.log);
            e.tok = try core.tokenizer.parse(gpa, e.arw.?.tokenizer_json);
            try prompt.checkTokenizer(&e.tok.?);
            e.ar = try ar_.Ar.init(gpa, &e.driver, e.stream, e.k, &e.arw.?);
        }
        if (o.load_dit) e.ditw = try wts.loadDit(gpa, io, &e.store, model_dir, o.log);
        if (o.load_dav) e.davw = try wts.loadDav(gpa, io, &e.store, model_dir);
        try e.stream.synchronize();
        e.load_ms = ar_.ms(t0, ar_.nowNs(io));
        if (o.log) std.log.info("music3: weights on the device: {d:.2} GB, loaded in {d:.1} s", .{ @as(f64, @floatFromInt(e.store.bytes)) / 1e9, e.load_ms / 1000 });
        return e;
    }

    fn quantize(ctx: *anyopaque, src: u64, n: usize, k: usize, group: usize, q: u64, sc: u64) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        try e.k.quantRows(e.stream, src, n, k, group, q, sc);
        try e.stream.synchronize();
    }

    pub fn destroy(e: *Engine) void {
        if (e.ar) |*a| a.deinit();
        if (e.tok) |*t| t.deinit();
        if (e.arw) |a| e.gpa.free(a.tokenizer_json);
        e.store.deinit();
        e.k.unload();
        e.gpa.destroy(e.k);
        e.stream.deinit();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn dit(e: *Engine) dit_.Dit {
        return .{ .gpa = e.gpa, .k = e.k, .w = &e.ditw.?, .d = &e.driver, .s = e.stream, .o = e.o.dit };
    }

    pub fn dav(e: *Engine) dav_.Dav {
        return .{ .gpa = e.gpa, .k = e.k, .w = &e.davw.?, .d = &e.driver, .s = e.stream, .o = .{ .tf32 = e.o.dav_tf32 } };
    }

    /// The conditional prompt's ids and the unconditional row (caller frees both).
    pub fn ids(e: *Engine, caption: []const u8, lyrics: []const u8) !struct { cond: []u32, unc: []u32 } {
        const c = try prompt.encode(e.gpa, &e.tok.?, caption, lyrics);
        errdefer e.gpa.free(c);
        return .{ .cond = c, .unc = try prompt.unconditional(e.gpa, c) };
    }

    /// Weight bytes a frame reads: body and lm_head once, the depth decoder 7 times, each head once.
    pub fn frameBytes(e: *const Engine) f64 {
        const a = &e.arw.?;
        const per: f64 = if (a.format == .int8) 1 else 2;
        var n: f64 = 0;
        const add = struct {
            fn m(x: wts.Mat) f64 {
                return @floatFromInt(x.n * x.k);
            }
        }.m;
        for (a.body) |L| n += add(L.qkv) + add(L.o) + add(L.gate_up) + add(L.down);
        n += add(a.lm_head);
        var dn: f64 = add(a.projection);
        for (a.depth) |L| dn += add(L.qkv) + add(L.o) + add(L.gate_up) + add(L.down);
        n += 7 * dn;
        for (a.heads) |h| n += add(h);
        return n * per;
    }

    pub fn generate(e: *Engine, io: std.Io, caption: []const u8, lyrics: []const u8, s: cfg.Settings, progress: ?Progress) !Song {
        _ = cuda.usage(true);
        const t0 = ar_.nowNs(io);
        const pr = try e.ids(caption, lyrics);
        defer e.gpa.free(pr.cond);
        defer e.gpa.free(pr.unc);
        const smp: ar_.Sampling = .{ .ar_cfg = s.ar_cfg, .top_k = s.top_k, .seed = cfg.deriveSeed(s.seed, "ar") };
        const r = try e.ar.?.run(io, pr.cond, pr.unc, smp, s.frameLimit(), e.o.graphs, e.o.sync_every, null, null, progress);
        defer e.gpa.free(r.states);
        const t1 = ar_.nowNs(io);
        var d = e.dit();
        const dp: ?dit_.Dit.Progress = if (progress) |p| .{ .ctx = p.ctx, .f = p.f } else null;
        const latents = try d.denoise(r.states, r.frames, s.steps, s.dit_cfg, cfg.deriveSeed(s.seed, "dit"), null, null, dp);
        defer e.gpa.free(latents);
        const t2 = ar_.nowNs(io);
        const L = cfg.framesToLatents(r.frames);
        if (progress) |p| p.f(p.ctx, "dav", 0, 1);
        var v = e.dav();
        const audio = try v.decode(latents, L);
        const t3 = ar_.nowNs(io);
        if (progress) |p| p.f(p.ctx, "dav", 1, 1);
        const u = cuda.usage(false);
        return .{ .audio = audio, .samples = L * 512, .frames = r.frames, .stopped = r.stopped, .prompt_tokens = r.prompt_tokens, .prefill_ms = r.prefill_ms, .ar_ms = ar_.ms(t0, t1), .frames_per_s = r.frames_per_s, .dit_ms = ar_.ms(t1, t2), .dav_ms = ar_.ms(t2, t3), .total_ms = ar_.ms(t0, t3), .latent_frames = L, .peak_device_gb = @as(f64, @floatFromInt(u.peak)) / 1e9 };
    }
};
