//! The AR stage on Metal: the prompt's CFG rows in tensor-unit GEMMs, then frames (code 0, 7 depth codes, states).
const std = @import("std");
const cfg = @import("../config.zig");
const wts = @import("../weights.zig");
const common = @import("../common.zig");
const gpu = @import("gpu.zig");
const k = @import("kernels.zig");

const H = cfg.hidden;
const Gpu = gpu.Gpu;
const decode_splits = 16;

pub const Sampling = common.Sampling;
pub const Trace = common.Trace;
pub const Result = common.Result;
pub const Progress = common.Progress;

pub const Ar = struct {
    gpa: std.mem.Allocator,
    g: *Gpu,
    w: *const wts.Ar,
    ctx: usize = 0,
    cache: u64 = 0,
    kc: [cfg.layers]u64 = @splat(0),
    vc: [cfg.layers]u64 = @splat(0),
    // device scalars
    pos: u64 = 0,
    step: u64 = 0,
    count: u64 = 0,
    codes: u64 = 0, // i32 [8]
    pick: u64 = 0,
    tokens: u64 = 0, // i32 [max_frames + 2]
    forced: u64 = 0, // i32 [max_frames + 2, 8]
    states: u64 = 0, // bf16 [max_frames + 2, 8, 4096]
    // frame scratch
    x: u64 = 0,
    xn: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    att: u64 = 0,
    gu: u64 = 0,
    ml: u64 = 0,
    acc: u64 = 0,
    last: u64 = 0, // [2, 4096] the body's normed last state
    d_in: u64 = 0, // [4, 4096]
    dx: u64 = 0,
    dxn: u64 = 0,
    dqkv: u64 = 0,
    datt: u64 = 0,
    dgu: u64 = 0,
    dout: u64 = 0,
    dkc: [cfg.depth_layers]u64 = @splat(0),
    dvc: [cfg.depth_layers]u64 = @splat(0),
    c0_logits: u64 = 0, // f32 [2, 16385]
    code_logits: u64 = 0, // f32 [2, 1024]

    pub fn init(gpa: std.mem.Allocator, g: *Gpu, w: *const wts.Ar) !Ar {
        var a: Ar = .{ .gpa = gpa, .g = g, .w = w };
        a.pos = try g.alloc(16);
        a.step = try g.alloc(16);
        a.count = try g.alloc(16);
        a.codes = try g.alloc(64);
        a.pick = try g.alloc(16);
        a.tokens = try g.alloc((cfg.max_frames + 2) * 4);
        a.forced = try g.alloc((cfg.max_frames + 2) * 8 * 4);
        a.states = try g.alloc((cfg.max_frames + 2) * 8 * H * 2);
        a.x = try g.alloc(2 * H * 2);
        a.xn = try g.alloc(2 * H * 2);
        a.qkv = try g.alloc(2 * cfg.qkv_width * 2);
        a.q = try g.alloc(2 * H * 2);
        a.att = try g.alloc(2 * H * 2);
        a.gu = try g.alloc(2 * 2 * cfg.mlp * 2);
        a.ml = try g.alloc(2 * 32 * decode_splits * 2 * 4);
        a.acc = try g.alloc(2 * 32 * decode_splits * 128 * 4);
        a.last = try g.alloc(2 * H * 2);
        a.d_in = try g.alloc(4 * H * 2);
        a.dx = try g.alloc(4 * H * 2);
        a.dxn = try g.alloc(4 * H * 2);
        a.dqkv = try g.alloc(4 * 3 * H * 2);
        a.datt = try g.alloc(4 * H * 2);
        a.dgu = try g.alloc(4 * 2 * cfg.depth_mlp * 2);
        a.dout = try g.alloc(2 * H * 2);
        for (0..cfg.depth_layers) |i| {
            a.dkc[i] = try g.alloc(2 * 16 * 8 * 256 * 2);
            a.dvc[i] = try g.alloc(2 * 16 * 8 * 256 * 2);
        }
        a.c0_logits = try g.alloc(2 * (cfg.c0_codes + 1) * 4);
        a.code_logits = try g.alloc(2 * cfg.codebook_size * 4);
        return a;
    }

    fn setI32(a: *Ar, dst: u64, v: i32) !void {
        try a.g.put(dst, std.mem.asBytes(&v));
    }

    /// KV caches for `need` positions (both rows, every layer).
    pub fn ensureCache(a: *Ar, need: usize) !void {
        if (a.ctx >= need) return;
        try a.g.flush();
        if (a.cache != 0) a.g.free(a.cache);
        const ctxn = (need + 255) / 256 * 256;
        const per = 2 * cfg.kv_heads * ctxn * cfg.head_dim * 2;
        a.cache = try a.g.alloc(per * 2 * cfg.layers);
        for (0..cfg.layers) |i| {
            a.kc[i] = a.cache + (2 * i) * per;
            a.vc[i] = a.cache + (2 * i + 1) * per;
        }
        a.ctx = ctxn;
    }

    /// The prompt rows (conditional ids, then the unconditional ones) through the body; leaves `last` and the caches.
    pub fn prefill(a: *Ar, ids: []const u32, unc: []const u32) !void {
        const n = ids.len;
        const t = 2 * n;
        const g = a.g;
        const w = a.w;
        var scratch: [11]u64 = @splat(0);
        defer {
            g.flush() catch {};
            for (scratch) |p| g.free(p);
        }
        const sizes = [_]usize{ t * 4, t * H * 2, t * H * 2, t * H * 2, t * cfg.qkv_width * 2, t * H * 2, t * 2 * cfg.mlp * 2, t * cfg.mlp * 2, t * 32 * 2 * 4, t * 32 * 128 * 4, if (w.format == .int8) 2 * cfg.mlp * H * 2 else 0 };
        for (&scratch, sizes) |*p, sz| p.* = if (sz > 0) try g.alloc(sz) else 0;
        const ids_d, const x, const xn, const tmp, const qkv, const q, const gu, const act, const ml, const acc, const wtmp = scratch;
        const all = std.mem.bytesAsSlice(i32, g.host(ids_d, t * 4));
        for (ids, 0..) |v, i| all[i] = @intCast(v);
        for (unc, 0..) |v, i| all[n + i] = @intCast(v);
        try a.setI32(a.pos, 0);
        k.embed(g, ids_d, w.embed_prefill, x, t);
        for (0..cfg.layers) |li| {
            const L = &w.body[li];
            k.rmsnorm(g, x, H, L.in_norm, xn, H, t, cfg.eps);
            k.linearRows(g, L.qkv, xn, qkv, t, wtmp);
            k.qkRope(g, qkv, L.q_norm, L.k_norm, w.inv_freq, a.pos, n, t, q, a.kc[li], a.vc[li], a.ctx);
            k.attention(g, q, a.kc[li], a.vc[li], a.pos, n, t, a.ctx, 1, ml, acc, xn);
            k.linearRows(g, L.o, xn, tmp, t, wtmp);
            k.add(g, x, tmp, t * H);
            k.rmsnorm(g, x, H, L.post_norm, xn, H, t, cfg.eps);
            k.linearRows(g, L.gate_up, xn, gu, t, wtmp);
            k.swiglu(g, gu, act, cfg.mlp, t);
            k.linearRows(g, L.down, act, tmp, t, wtmp);
            k.add(g, x, tmp, t * H);
            try g.commit();
        }
        // the last row of each: the normed state code 0 is drawn from
        k.rmsnorm(g, x + (n - 1) * H * 2, n * H, w.norm, a.last, H, 2, cfg.eps);
    }

    pub fn depthPass(a: *Ar, rows: usize, p0: usize) void {
        const g = a.g;
        const w = a.w;
        for (0..cfg.depth_layers) |li| {
            const L = &w.depth[li];
            k.rmsnorm(g, a.dx, H, L.in_norm, a.dxn, H, rows, cfg.eps);
            k.gemv(g, L.qkv, a.dxn, rows, .{ .y = a.dqkv });
            k.depthAttention(g, a.dqkv, p0, rows, a.dkc[li], a.dvc[li], a.datt);
            k.gemv(g, L.o, a.datt, rows, .{ .y = a.dx, .res = a.dx });
            k.rmsnorm(g, a.dx, H, L.post_norm, a.dxn, H, rows, cfg.eps);
            k.gemv(g, L.gate_up, a.dxn, rows, .{ .glu = true, .y = a.dgu });
            k.gemv(g, L.down, a.dgu, rows, .{ .y = a.dx, .res = a.dx });
        }
        // the last position's two rows, normed
        k.rmsnorm(g, a.dx + (rows - 2) * H * 2, H, w.depth_norm, a.dout, H, 2, cfg.eps);
    }

    /// One frame: code 0, the 7 depth codes, the states, the frame fed back (`forced`: codes from the forced table).
    fn enqueueFrame(a: *Ar, smp: Sampling, forced: bool, trace: ?*Trace) !void {
        const g = a.g;
        const w = a.w;
        k.gemv(g, w.lm_head, a.last, 2, .{ .yf = a.c0_logits });
        if (trace) |tr| try a.pull(&tr.c0, a.c0_logits, 2 * (cfg.c0_codes + 1));
        if (forced) {
            k.forceCodes(g, a.forced, a.count, a.codes, a.tokens);
        } else {
            k.sample(g, a.c0_logits, cfg.c0_codes + 1, true, .{ .cfg = smp.ar_cfg, .top_k = smp.top_k, .seed = smp.seed, .step = a.step, .slot = 0, .pick = a.pick, .code = a.codes, .tokens = a.tokens, .count = a.count });
        }
        k.statePart(g, a.last, a.states, a.count, 0);
        // depth pass 0: [hidden, code-0 embedding] at positions 0 and 1, both rows
        g.copy(a.last, a.d_in, 2 * H * 2);
        k.embedCode(g, a.codes, 0, w.embed_audio, a.d_in + 2 * H * 2, 2);
        k.gemv(g, w.projection, a.d_in, 4, .{ .y = a.dx, .res = w.pos_emb, .res_div = 2 });
        a.depthPass(4, 0);
        for (1..cfg.codebooks) |ci| {
            if (ci > 1) {
                // the code drawn last, embedded, at position ci
                k.embedCode(g, a.codes + (ci - 1) * 4, (ci - 2) * cfg.codebook_size, w.extra, a.d_in, 2);
                k.gemv(g, w.projection, a.d_in, 2, .{ .y = a.dx, .res = w.pos_emb + ci * H * 2, .res_div = 2 });
                a.depthPass(2, ci);
            }
            k.statePart(g, a.dout, a.states, a.count, ci);
            k.gemv(g, w.heads[ci - 1], a.dout, 2, .{ .yf = a.code_logits });
            if (trace) |tr| try a.pull(&tr.depth, a.code_logits, 2 * cfg.codebook_size);
            if (!forced) k.sample(g, a.code_logits, cfg.codebook_size, false, .{ .cfg = smp.ar_cfg, .top_k = smp.top_k, .seed = smp.seed, .step = a.step, .slot = @intCast(ci), .pick = a.codes + ci * 4 });
        }
        a.bodyRow();
        try g.commit();
    }

    /// The frame fed back: one body row a CFG row at pos, then the normed state the next code 0 is drawn from.
    pub fn bodyRow(a: *Ar) void {
        const g = a.g;
        const w = a.w;
        k.frameEmbed(g, a.codes, w.embed_audio, w.extra, a.x, 2);
        for (0..cfg.layers) |li| {
            const L = &w.body[li];
            k.rmsnorm(g, a.x, H, L.in_norm, a.xn, H, 2, cfg.eps);
            k.gemv(g, L.qkv, a.xn, 2, .{ .y = a.qkv });
            k.qkRope(g, a.qkv, L.q_norm, L.k_norm, w.inv_freq, a.pos, 1, 2, a.q, a.kc[li], a.vc[li], a.ctx);
            k.attention(g, a.q, a.kc[li], a.vc[li], a.pos, 1, 2, a.ctx, decode_splits, a.ml, a.acc, a.att);
            k.gemv(g, L.o, a.att, 2, .{ .y = a.x, .res = a.x });
            k.rmsnorm(g, a.x, H, L.post_norm, a.xn, H, 2, cfg.eps);
            k.gemv(g, L.gate_up, a.xn, 2, .{ .glu = true, .y = a.gu });
            k.gemv(g, L.down, a.gu, 2, .{ .y = a.x, .res = a.x });
        }
        k.rmsnorm(g, a.x, H, w.norm, a.last, H, 2, cfg.eps);
        k.frameDone(g, a.pos, a.step, a.count);
    }

    fn pull(a: *Ar, list: *std.ArrayList(f32), src: u64, n: usize) !void {
        const at = list.items.len;
        try list.resize(a.gpa, at + n);
        try a.g.get(src, std.mem.sliceAsBytes(list.items[at..]));
    }

    /// Prompt, then frames to the stop code or `limit` kept; `forced_codes` replaces draws; `graphs` is CUDA's.
    pub fn run(a: *Ar, io: std.Io, ids: []const u32, unc: []const u32, smp: Sampling, limit: usize, graphs: bool, sync_every: usize, forced_codes: ?[]const [8]i32, trace: ?*Trace, progress: ?Progress) !Result {
        _ = graphs;
        const g = a.g;
        const t0 = common.nowNs(io);
        const frames_max = if (forced_codes) |fc| @min(limit + 1, fc.len) else limit + 1;
        try a.ensureCache(ids.len + frames_max + 2);
        try a.prefill(ids, unc);
        try g.flush();
        const t1 = common.nowNs(io);
        try a.setI32(a.pos, @intCast(ids.len));
        try a.setI32(a.count, 0);
        @memset(g.host(a.step, 8), 0);
        if (forced_codes) |fc| try g.put(a.forced, std.mem.sliceAsBytes(fc[0..frames_max]));
        const tokens = try a.gpa.alloc(i32, frames_max);
        defer a.gpa.free(tokens);
        var made: usize = 0;
        var stop_at: ?usize = null;
        const every = @max(1, sync_every);
        var reported: usize = 0;
        while (made < frames_max and stop_at == null) {
            const burst = @min(every, frames_max - made);
            for (0..burst) |_| try a.enqueueFrame(smp, forced_codes != null, trace);
            try g.get(a.tokens + made * 4, std.mem.sliceAsBytes(tokens[made .. made + burst]));
            for (tokens[made .. made + burst], made..) |tok, fi| if (tok == 0 and forced_codes == null) {
                stop_at = fi;
                break;
            };
            made += burst;
            if (progress) |p| if (made - reported >= 25 or stop_at != null or made == frames_max) {
                reported = made;
                const kept_now = @min(if (stop_at) |st| st else made, limit + 1) -| 1;
                p.f(p.ctx, "ar", kept_now, limit);
            };
        }
        const end = if (stop_at) |st| st else made; // frames 1 .. end - 1 are kept
        const kept = @min(end -| 1, limit);
        if (kept == 0) return error.NoAudioFrames;
        const states = try a.gpa.alloc(u16, kept * 8 * H);
        errdefer a.gpa.free(states);
        try g.get(a.states + 8 * H * 2, std.mem.sliceAsBytes(states));
        const t2 = common.nowNs(io);
        const dec_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
        return .{ .states = states, .frames = kept, .stopped = stop_at != null, .prompt_tokens = ids.len, .prefill_ms = common.ms(t0, t1), .ar_ms = common.ms(t0, t2), .frames_per_s = @as(f64, @floatFromInt(made)) / @max(dec_s, 1e-9) };
    }
};
