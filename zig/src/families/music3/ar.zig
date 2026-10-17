//! The AR stage: the prompt's CFG rows in cuBLAS GEMMs, then a CUDA-graph frame step (code 0, 7 depth codes, states).
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const common = @import("common.zig");

const H = cfg.hidden;
const Buf = cuda.DeviceBuffer;
const decode_splits = 16;

pub const Sampling = common.Sampling;
pub const Trace = common.Trace;
pub const Result = common.Result;

pub const Ar = struct {
    gpa: std.mem.Allocator,
    k: *kern.Kernels,
    w: *const wts.Ar,
    d: *const cuda.Driver,
    s: cuda.Stream,
    bufs: std.ArrayList(Buf) = .empty,
    // the cache, sized per request
    ctx: usize = 0,
    kc: [cfg.layers]u64 = @splat(0),
    vc: [cfg.layers]u64 = @splat(0),
    cache_buf: ?Buf = null,
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
    graph: ?cuda.graph.Exec = null,
    graph_smp: Sampling = .{},
    graph_ctx: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *kern.Kernels, w: *const wts.Ar) !Ar {
        var a: Ar = .{ .gpa = gpa, .k = k, .w = w, .d = d, .s = s };
        errdefer a.deinit();
        a.pos = try a.alloc(16);
        a.step = try a.alloc(16);
        a.count = try a.alloc(16);
        a.codes = try a.alloc(64);
        a.pick = try a.alloc(16);
        a.tokens = try a.alloc((cfg.max_frames + 2) * 4);
        a.forced = try a.alloc((cfg.max_frames + 2) * 8 * 4);
        a.states = try a.alloc((cfg.max_frames + 2) * 8 * H * 2);
        a.x = try a.alloc(2 * H * 2);
        a.xn = try a.alloc(2 * H * 2);
        a.qkv = try a.alloc(2 * cfg.qkv_width * 2);
        a.q = try a.alloc(2 * H * 2);
        a.att = try a.alloc(2 * H * 2);
        a.gu = try a.alloc(2 * 2 * cfg.mlp * 2);
        a.ml = try a.alloc(2 * 32 * decode_splits * 2 * 4);
        a.acc = try a.alloc(2 * 32 * decode_splits * 128 * 4);
        a.last = try a.alloc(2 * H * 2);
        a.d_in = try a.alloc(4 * H * 2);
        a.dx = try a.alloc(4 * H * 2);
        a.dxn = try a.alloc(4 * H * 2);
        a.dqkv = try a.alloc(4 * 3 * H * 2);
        a.datt = try a.alloc(4 * H * 2);
        a.dgu = try a.alloc(4 * 2 * cfg.depth_mlp * 2);
        a.dout = try a.alloc(2 * H * 2);
        for (0..cfg.depth_layers) |i| {
            a.dkc[i] = try a.alloc(2 * 16 * 8 * 256 * 2);
            a.dvc[i] = try a.alloc(2 * 16 * 8 * 256 * 2);
        }
        a.c0_logits = try a.alloc(2 * (cfg.c0_codes + 1) * 4);
        a.code_logits = try a.alloc(2 * cfg.codebook_size * 4);
        return a;
    }

    pub fn deinit(a: *Ar) void {
        if (a.graph) |*g| g.deinit();
        a.graph = null;
        if (a.cache_buf) |*b| b.free();
        a.cache_buf = null;
        for (a.bufs.items) |*b| b.free();
        a.bufs.deinit(a.gpa);
    }

    fn alloc(a: *Ar, bytes: usize) !u64 {
        const b = try Buf.alloc(a.d, @max(bytes, 256));
        try a.bufs.append(a.gpa, b);
        return b.ptr;
    }

    fn view(a: *Ar, ptr: u64) Buf {
        return .{ .d = a.d, .ptr = ptr, .len = std.math.maxInt(usize) / 2, .borrowed = true };
    }

    fn put(a: *Ar, dst: u64, bytes: []const u8) !void {
        try a.view(dst).uploadAsync(0, bytes, a.s.handle);
    }

    fn get(a: *Ar, src: u64, out: []u8) !void {
        try a.view(src).downloadAsync(0, out, a.s.handle);
        try a.s.synchronize();
    }

    fn setI32(a: *Ar, dst: u64, v: i32) !void {
        try a.put(dst, std.mem.asBytes(&v));
        try a.s.synchronize();
    }

    /// KV caches for `need` positions (both rows, every layer); a new size drops the captured graph.
    fn ensureCache(a: *Ar, need: usize) !void {
        if (a.ctx >= need) return;
        if (a.graph) |*g| g.deinit();
        a.graph = null;
        if (a.cache_buf) |*b| b.free();
        a.cache_buf = null;
        const ctxn = (need + 255) / 256 * 256;
        const per = 2 * cfg.kv_heads * ctxn * cfg.head_dim * 2;
        a.cache_buf = try Buf.alloc(a.d, per * 2 * cfg.layers);
        for (0..cfg.layers) |i| {
            a.kc[i] = a.cache_buf.?.ptr + (2 * i) * per;
            a.vc[i] = a.cache_buf.?.ptr + (2 * i + 1) * per;
        }
        a.ctx = ctxn;
    }

    /// The prompt rows (conditional ids, then the unconditional ones) through the body; leaves `last` and the caches.
    pub fn prefill(a: *Ar, ids: []const u32, unc: []const u32) !void {
        const n = ids.len;
        const t = 2 * n;
        const k = a.k;
        const s = a.s;
        const w = a.w;
        // per-request prompt scratch
        var scratch: std.ArrayList(Buf) = .empty;
        defer {
            for (scratch.items) |*b| b.free();
            scratch.deinit(a.gpa);
        }
        const P = struct {
            fn mk(list: *std.ArrayList(Buf), gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) !u64 {
                const b = try Buf.alloc(d, @max(bytes, 256));
                try list.append(gpa, b);
                return b.ptr;
            }
        };
        const ids_d = try P.mk(&scratch, a.gpa, a.d, t * 4);
        const x = try P.mk(&scratch, a.gpa, a.d, t * H * 2);
        const xn = try P.mk(&scratch, a.gpa, a.d, t * H * 2);
        const tmp = try P.mk(&scratch, a.gpa, a.d, t * H * 2);
        const qkv = try P.mk(&scratch, a.gpa, a.d, t * cfg.qkv_width * 2);
        const q = try P.mk(&scratch, a.gpa, a.d, t * H * 2);
        const gu = try P.mk(&scratch, a.gpa, a.d, t * 2 * cfg.mlp * 2);
        const act = try P.mk(&scratch, a.gpa, a.d, t * cfg.mlp * 2);
        const ml = try P.mk(&scratch, a.gpa, a.d, t * 32 * 2 * 4);
        const acc = try P.mk(&scratch, a.gpa, a.d, t * 32 * 128 * 4);
        const wtmp: u64 = if (w.format == .int8) try P.mk(&scratch, a.gpa, a.d, 2 * cfg.mlp * H * 2) else 0;
        const all = try a.gpa.alloc(i32, t);
        defer a.gpa.free(all);
        for (ids, 0..) |v, i| all[i] = @intCast(v);
        for (unc, 0..) |v, i| all[n + i] = @intCast(v);
        try a.put(ids_d, std.mem.sliceAsBytes(all));
        try a.setI32(a.pos, 0);
        try k.embed(s, ids_d, w.embed_prefill, x, t);
        for (0..cfg.layers) |li| {
            const L = &w.body[li];
            try k.rmsnorm(s, x, H, L.in_norm, xn, H, t, cfg.eps);
            try k.linearRows(s, L.qkv, xn, qkv, t, wtmp);
            try k.qkRope(s, qkv, L.q_norm, L.k_norm, w.inv_freq, a.pos, n, t, q, a.kc[li], a.vc[li], a.ctx);
            try k.attention(s, q, a.kc[li], a.vc[li], a.pos, n, t, a.ctx, 1, ml, acc, xn);
            try k.linearRows(s, L.o, xn, tmp, t, wtmp);
            try k.add(s, x, tmp, t * H);
            try k.rmsnorm(s, x, H, L.post_norm, xn, H, t, cfg.eps);
            try k.linearRows(s, L.gate_up, xn, gu, t, wtmp);
            try k.swiglu(s, gu, act, cfg.mlp, t);
            try k.linearRows(s, L.down, act, tmp, t, wtmp);
            try k.add(s, x, tmp, t * H);
        }
        // the last row of each: the normed state code 0 is drawn from
        try k.rmsnorm(s, x + (n - 1) * H * 2, n * H, w.norm, a.last, H, 2, cfg.eps);
        try a.s.synchronize();
    }

    fn depthPass(a: *Ar, rows: usize, p0: usize) !void {
        const k = a.k;
        const s = a.s;
        const w = a.w;
        for (0..cfg.depth_layers) |li| {
            const L = &w.depth[li];
            try k.rmsnorm(s, a.dx, H, L.in_norm, a.dxn, H, rows, cfg.eps);
            try k.gemv(s, L.qkv, a.dxn, rows, .{ .y = a.dqkv });
            try k.depthAttention(s, a.dqkv, p0, rows, a.dkc[li], a.dvc[li], a.datt);
            try k.gemv(s, L.o, a.datt, rows, .{ .y = a.dx, .res = a.dx });
            try k.rmsnorm(s, a.dx, H, L.post_norm, a.dxn, H, rows, cfg.eps);
            try k.gemv(s, L.gate_up, a.dxn, rows, .{ .glu = true, .y = a.dgu });
            try k.gemv(s, L.down, a.dgu, rows, .{ .y = a.dx, .res = a.dx });
        }
        // the last position's two rows, normed
        try k.rmsnorm(s, a.dx + (rows - 2) * H * 2, H, w.depth_norm, a.dout, H, 2, cfg.eps);
    }

    /// One frame: code 0, the 7 depth codes, the states, the frame fed back (`forced`: codes from the forced table).
    fn enqueueFrame(a: *Ar, smp: Sampling, forced: bool, trace: ?*Trace) !void {
        const k = a.k;
        const s = a.s;
        const w = a.w;
        try k.gemv(s, w.lm_head, a.last, 2, .{ .yf = a.c0_logits });
        if (trace) |tr| try a.pull(&tr.c0, a.c0_logits, 2 * (cfg.c0_codes + 1));
        if (forced) {
            try k.forceCodes(s, a.forced, a.count, a.codes, a.pick);
            try a.copyI32(a.pick, a.tokens, a.count);
        } else {
            try k.sample(s, a.c0_logits, cfg.c0_codes + 1, true, .{ .cfg = smp.ar_cfg, .top_k = smp.top_k, .seed = smp.seed, .step = a.step, .slot = 0, .pick = a.pick, .code = a.codes, .tokens = a.tokens, .count = a.count });
        }
        try k.statePart(s, a.last, a.states, a.count, 0);
        // depth pass 0: [hidden, code-0 embedding] at positions 0 and 1, both rows
        try a.copy(a.d_in, a.last, 2 * H * 2);
        try k.embedCode(s, a.codes, 0, w.embed_audio, a.d_in + 2 * H * 2, 2);
        try k.gemv(s, w.projection, a.d_in, 4, .{ .y = a.dx, .res = w.pos_emb, .res_div = 2 });
        try a.depthPass(4, 0);
        for (1..cfg.codebooks) |ci| {
            if (ci > 1) {
                // the code drawn last, embedded, at position ci
                try k.embedCode(s, a.codes + (ci - 1) * 4, (ci - 2) * cfg.codebook_size, w.extra, a.d_in, 2);
                try k.gemv(s, w.projection, a.d_in, 2, .{ .y = a.dx, .res = w.pos_emb + ci * H * 2, .res_div = 2 });
                try a.depthPass(2, ci);
            }
            try k.statePart(s, a.dout, a.states, a.count, ci);
            try k.gemv(s, w.heads[ci - 1], a.dout, 2, .{ .yf = a.code_logits });
            if (trace) |tr| try a.pull(&tr.depth, a.code_logits, 2 * cfg.codebook_size);
            if (!forced) try k.sample(s, a.code_logits, cfg.codebook_size, false, .{ .cfg = smp.ar_cfg, .top_k = smp.top_k, .seed = smp.seed, .step = a.step, .slot = @intCast(ci), .pick = a.codes + ci * 4 });
        }
        // the frame fed back: one body row a CFG row at pos
        try k.frameEmbed(s, a.codes, w.embed_audio, w.extra, a.x, 2);
        for (0..cfg.layers) |li| {
            const L = &w.body[li];
            try k.rmsnorm(s, a.x, H, L.in_norm, a.xn, H, 2, cfg.eps);
            try k.gemv(s, L.qkv, a.xn, 2, .{ .y = a.qkv });
            try k.qkRope(s, a.qkv, L.q_norm, L.k_norm, w.inv_freq, a.pos, 1, 2, a.q, a.kc[li], a.vc[li], a.ctx);
            try k.attention(s, a.q, a.kc[li], a.vc[li], a.pos, 1, 2, a.ctx, decode_splits, a.ml, a.acc, a.att);
            try k.gemv(s, L.o, a.att, 2, .{ .y = a.x, .res = a.x });
            try k.rmsnorm(s, a.x, H, L.post_norm, a.xn, H, 2, cfg.eps);
            try k.gemv(s, L.gate_up, a.xn, 2, .{ .glu = true, .y = a.gu });
            try k.gemv(s, L.down, a.gu, 2, .{ .y = a.x, .res = a.x });
        }
        try k.rmsnorm(s, a.x, H, w.norm, a.last, H, 2, cfg.eps);
        try k.frameDone(s, a.pos, a.step, a.count);
    }

    fn copy(a: *Ar, dst: u64, src: u64, bytes: usize) !void {
        try a.view(dst).copyFrom(0, src, bytes, a.s.handle);
    }

    /// tokens[*count] = *pick (forced frames), as a kernel-free device copy is not count-indexed: through the host.
    fn copyI32(a: *Ar, src: u64, table: u64, count: u64) !void {
        var c: i32 = 0;
        try a.get(count, std.mem.asBytes(&c));
        try a.copy(table + @as(u64, @intCast(c)) * 4, src, 4);
    }

    fn pull(a: *Ar, list: *std.ArrayList(f32), src: u64, n: usize) !void {
        const at = list.items.len;
        try list.resize(a.gpa, at + n);
        try a.get(src, std.mem.sliceAsBytes(list.items[at..]));
    }

    fn frame(a: *Ar, smp: Sampling, graphs: bool) !void {
        if (!graphs) return a.enqueueFrame(smp, false, null);
        if (a.graph == null or !std.meta.eql(a.graph_smp, smp) or a.graph_ctx != a.ctx) {
            if (a.graph) |*g| g.deinit();
            a.graph = null;
            try cuda.graph.beginCapture(a.s, .thread_local);
            a.enqueueFrame(smp, false, null) catch |e| {
                var g = cuda.graph.endCapture(a.s) catch return e;
                g.deinit();
                return e;
            };
            var g = try cuda.graph.endCapture(a.s);
            defer g.deinit();
            a.graph = try g.instantiate();
            a.graph_smp = smp;
            a.graph_ctx = a.ctx;
        }
        try a.graph.?.launchOn(a.s);
    }

    pub const Progress = common.Progress;

    /// Prompt, then frames to the stop code or `limit` kept; `forced_codes` replaces every draw, `trace` keeps logits.
    pub fn run(a: *Ar, io: std.Io, ids: []const u32, unc: []const u32, smp: Sampling, limit: usize, graphs: bool, sync_every: usize, forced_codes: ?[]const [8]i32, trace: ?*Trace, progress: ?Progress) !Result {
        const t0 = nowNs(io);
        const frames_max = if (forced_codes) |fc| @min(limit + 1, fc.len) else limit + 1;
        try a.ensureCache(ids.len + frames_max + 2);
        try a.prefill(ids, unc);
        const t1 = nowNs(io);
        try a.setI32(a.pos, @intCast(ids.len));
        try a.setI32(a.count, 0);
        const one: u64 = 0;
        try a.put(a.step, std.mem.asBytes(&one));
        if (forced_codes) |fc| try a.put(a.forced, std.mem.sliceAsBytes(fc[0..frames_max]));
        try a.s.synchronize();
        const tokens = try a.gpa.alloc(i32, frames_max);
        defer a.gpa.free(tokens);
        var made: usize = 0;
        var stop_at: ?usize = null;
        const every = @max(1, sync_every);
        var reported: usize = 0;
        while (made < frames_max and stop_at == null) {
            const burst = @min(every, frames_max - made);
            for (0..burst) |_| {
                if (forced_codes != null) try a.enqueueFrame(smp, true, trace) else try a.frame(smp, graphs);
            }
            try a.get(a.tokens + made * 4, std.mem.sliceAsBytes(tokens[made .. made + burst]));
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
        try a.get(a.states + 8 * H * 2, std.mem.sliceAsBytes(states));
        const t2 = nowNs(io);
        const dec_s = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
        return .{ .states = states, .frames = kept, .stopped = stop_at != null, .prompt_tokens = ids.len, .prefill_ms = ms(t0, t1), .ar_ms = ms(t0, t2), .frames_per_s = @as(f64, @floatFromInt(made)) / @max(dec_s, 1e-9) };
    }
};

pub const nowNs = common.nowNs;
pub const ms = common.ms;
