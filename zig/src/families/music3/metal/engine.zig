//! MiniMax Music 3 on Metal: the three stages behind one load (AR frames, the DiT, the DAV), timed apart.
const std = @import("std");
const core = @import("core");
const cfg = @import("../config.zig");
const wts = @import("../weights.zig");
const prompt = @import("../prompt.zig");
const common = @import("../common.zig");
const gpu = @import("gpu.zig");
const k = @import("kernels.zig");
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
    sync_every: usize = 16,
    log: bool = true,
    graphs: bool = true, // CUDA graphs: a CUDA-only knob, ignored here
    dav_tf32: bool = false, // CUDA-only, ignored here (the DAV's GEMMs are fp32 on the tensor units)
};

pub const Song = common.Song;
pub const Progress = common.Progress;

/// The store weights.zig's loaders fill: shared Metal buffers, and one staging buffer for the int8 quantizer.
pub const Store = struct {
    g: *gpu.Gpu,
    bytes: usize = 0,
    staging: u64 = 0,

    const staging_bytes: usize = 24576 * 4096 * 2;

    pub fn alloc(s: *Store, bytes: usize) !u64 {
        s.bytes += bytes;
        return s.g.alloc(bytes);
    }

    pub fn upload(s: *Store, bytes: []const u8) !u64 {
        const p = try s.alloc(bytes.len);
        @memcpy(s.g.host(p, bytes.len), bytes);
        return p;
    }

    pub fn stage(s: *Store, bytes: []const u8) !u64 {
        if (bytes.len > staging_bytes) return error.StagingTooSmall;
        if (s.staging == 0) s.staging = try s.g.alloc(staging_bytes);
        try s.g.put(s.staging, bytes);
        return s.staging;
    }

    pub fn unstage(s: *Store) void {
        s.g.flush() catch {};
        s.g.free(s.staging);
        s.staging = 0;
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    o: Options,
    g: *gpu.Gpu,
    store: Store,
    arw: ?wts.Ar = null,
    ditw: ?wts.Dit = null,
    davw: ?wts.Dav = null,
    tok: ?core.tokenizer.Tokenizer = null,
    ar: ?ar_.Ar = null,
    load_ms: f64 = 0,

    pub fn create(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8, o: Options) !*Engine {
        const t0 = common.nowNs(io);
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const g = try gpu.Gpu.open(gpa, @import("music3_metallib").bytes);
        errdefer g.close();
        e.* = .{ .gpa = gpa, .o = o, .g = g, .store = .{ .g = g } };
        if (o.load_ar) {
            e.arw = try wts.loadAr(gpa, io, &e.store, model_dir, o.format, o.depth_group, .{ .ctx = e, .f = quantize }, o.log);
            e.tok = try core.tokenizer.parse(gpa, e.arw.?.tokenizer_json);
            try prompt.checkTokenizer(&e.tok.?);
            e.ar = try ar_.Ar.init(gpa, g, &e.arw.?);
        }
        if (o.load_dit) {
            e.ditw = try wts.loadDit(gpa, io, &e.store, model_dir, o.log);
            try dit_.pairGlu(gpa, g, &e.ditw.?);
        }
        if (o.load_dav) {
            e.davw = try wts.loadDav(gpa, io, &e.store, model_dir);
            try dav_.splitWeights(g, &e.store, &e.davw.?);
        }
        try g.flush();
        e.load_ms = common.ms(t0, common.nowNs(io));
        if (o.log) std.log.info("music3: weights in Metal buffers: {d:.2} GB, loaded in {d:.1} s", .{ @as(f64, @floatFromInt(e.store.bytes)) / 1e9, e.load_ms / 1000 });
        return e;
    }

    fn quantize(ctx: *anyopaque, src: u64, n: usize, kk: usize, group: usize, q: u64, sc: u64) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        if (group != 0 and group != 128) return error.UnsupportedGroup;
        k.quant(e.g, src, n, kk, group, q, sc);
        try e.g.flush();
    }

    pub fn destroy(e: *Engine) void {
        if (e.tok) |*t| t.deinit();
        if (e.arw) |a| e.gpa.free(a.tokenizer_json);
        e.g.close();
        e.gpa.destroy(e);
    }

    pub fn dit(e: *Engine) dit_.Dit {
        return .{ .gpa = e.gpa, .g = e.g, .w = &e.ditw.?, .o = e.o.dit };
    }

    pub fn dav(e: *Engine) dav_.Dav {
        return .{ .gpa = e.gpa, .g = e.g, .w = &e.davw.? };
    }

    /// The conditional prompt's ids and the unconditional row (caller frees both).
    pub fn ids(e: *Engine, caption: []const u8, lyrics: []const u8) !struct { cond: []u32, unc: []u32 } {
        const c = try prompt.encode(e.gpa, &e.tok.?, caption, lyrics);
        errdefer e.gpa.free(c);
        return .{ .cond = c, .unc = try prompt.unconditional(e.gpa, c) };
    }

    pub fn frameBytes(e: *const Engine) f64 {
        return common.frameBytes(&e.arw.?);
    }

    /// Weight bytes in Metal buffers.
    pub fn deviceBytes(e: *const Engine) usize {
        return e.store.bytes;
    }

    pub fn generate(e: *Engine, io: std.Io, caption: []const u8, lyrics: []const u8, s: cfg.Settings, progress: ?Progress) !Song {
        e.g.peak = e.g.device.allocated();
        const t0 = common.nowNs(io);
        const pr = try e.ids(caption, lyrics);
        defer e.gpa.free(pr.cond);
        defer e.gpa.free(pr.unc);
        const smp: ar_.Sampling = .{ .ar_cfg = s.ar_cfg, .top_k = s.top_k, .seed = cfg.deriveSeed(s.seed, "ar") };
        const r = try e.ar.?.run(io, pr.cond, pr.unc, smp, s.frameLimit(), false, e.o.sync_every, null, null, progress);
        defer e.gpa.free(r.states);
        const t1 = common.nowNs(io);
        var d = e.dit();
        const latents = try d.denoise(r.states, r.frames, s.steps, s.dit_cfg, cfg.deriveSeed(s.seed, "dit"), null, null, progress);
        defer e.gpa.free(latents);
        const t2 = common.nowNs(io);
        const L = cfg.framesToLatents(r.frames);
        if (progress) |p| p.f(p.ctx, "dav", 0, 1);
        var v = e.dav();
        const audio = try v.decode(latents, L);
        const t3 = common.nowNs(io);
        if (progress) |p| p.f(p.ctx, "dav", 1, 1);
        return .{ .audio = audio, .samples = L * 512, .frames = r.frames, .stopped = r.stopped, .prompt_tokens = r.prompt_tokens, .prefill_ms = r.prefill_ms, .ar_ms = common.ms(t0, t1), .frames_per_s = r.frames_per_s, .dit_ms = common.ms(t1, t2), .dav_ms = common.ms(t2, t3), .total_ms = common.ms(t0, t3), .latent_frames = L, .peak_device_gb = @as(f64, @floatFromInt(e.g.peak)) / 1e9 };
    }
};
