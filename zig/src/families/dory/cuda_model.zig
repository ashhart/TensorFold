//! Dory's forward on CUDA for one stream: per_loop KV caches, prompt chunks and decode windows on shared operators.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const schedule = @import("schedule.zig");
const wts = @import("cuda_weights.zig");

const Config = config.Config;
const DeviceBuffer = cuda.DeviceBuffer;
const WindowAttention = cuda.window_attention.WindowAttention;

pub const eps: f32 = 1e-6;
/// Prompt rows a forward takes at most.
pub const prompt_rows = 2048;
/// Decode rows (a verify window) a forward takes at most.
pub const window_rows = 16;
const key_block = 64;

/// Keys a sliding cache keeps: a chunk's rows written before its first row reads back its window.
pub fn ringSize(rows: usize, window: usize) usize {
    return (rows + window + key_block - 1) / key_block * key_block;
}

/// The shared operators the forward launches.
pub const Kernels = struct {
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    embed: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    act: cuda.act_rows.ActRows,
    dense: cuda.dense_rows.DenseRows,
    win: WindowAttention,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        const d = ctx.d;
        var k: Kernels = undefined;
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        k.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer k.prompt.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.act = try cuda.act_rows.ActRows.load(d);
        errdefer k.act.unload();
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        k.win = try WindowAttention.load(d);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.win.unload();
        k.dense.unload();
        k.act.unload();
        k.qk.unload();
        k.norms.unload();
        k.embed.unload();
        k.prompt.unload();
        k.lane.unload();
    }
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    c: Config,
    sched: schedule.Schedule,
    context: usize,
    ring: usize,
    k: []u64, // per cache: [len, HK, D] bf16 (len: context for full caches, the ring for sliding ones)
    v: []u64,
    inv: [2]u64, // fp32 [D / 2] inverse frequencies: sliding, full
    s: Scratch = undefined,
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    const Scratch = struct { ids: u64, pos: u64, h: u64, h2: u64, state: u64, input: u64, out: u64, q: u64, kk: u64, vv: u64, g: u64, qn: u64, kn: u64, vn: u64, att: u64, gated: u64, mg: u64, mu: u64, m: u64, part: u64, win: u64, sel: u64 };

    fn alloc(m: *Model, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, k: *const Kernels, context: usize) !Model {
        const c = w.config;
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .c = c, .sched = schedule.build(c), .context = context, .ring = ringSize(prompt_rows, c.window), .k = &.{}, .v = &.{}, .inv = .{ 0, 0 } };
        errdefer m.deinit();
        m.k = try gpa.alloc(u64, m.sched.caches);
        m.v = try gpa.alloc(u64, m.sched.caches);
        for (m.k, m.v, m.sched.cache_kind[0..m.sched.caches]) |*kc, *vc, kind| {
            const len = if (kind == .full) context else m.ring;
            const bytes = len * c.kvDim() * 2;
            kc.* = try m.alloc(bytes);
            vc.* = try m.alloc(bytes);
            try d.check(d.api.cuMemsetD8Async(kc.*, 0, bytes, s.handle), "cuMemsetD8Async");
            try d.check(d.api.cuMemsetD8Async(vc.*, 0, bytes, s.handle), "cuMemsetD8Async");
        }
        for ([_]f64{ c.theta_sliding, c.theta_full }, &m.inv) |theta, *p| {
            var inv: [256]f32 = undefined; // vLLM's inv_freq in fp32: 1 / theta^(2j / D)
            const half = c.head_dim / 2;
            for (inv[0..half], 0..) |*v, j| v.* = 1 / std.math.pow(f32, @floatCast(theta), @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(c.head_dim)));
            p.* = try m.alloc(half * 4);
            try m.upload(s, p.*, std.mem.sliceAsBytes(inv[0..half]));
        }
        try m.scratch(k);
        try s.synchronize();
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.k);
        m.gpa.free(m.v);
    }

    fn upload(m: *Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }

    fn copy(m: *Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    fn window(m: *const Model, full: bool) cuda.window_attention.Window {
        const c = m.c;
        return .{ .h = c.heads, .hk = c.kv_heads, .d = c.head_dim, .win = if (full) 0 else c.window, .ring = if (full) m.context else m.ring, .scale = c.factors().attn };
    }

    fn scratch(m: *Model, k: *const Kernels) !void {
        const c = m.c;
        const r = prompt_rows;
        const d = c.hidden;
        var part: usize = 0;
        for (m.w.subs) |sub| switch (sub) {
            .attn => |a| for ([_]cuda.qlinear.Weight{ a.q, a.k, a.v, a.g, a.o }) |q| for (1..window_rows + 1) |rows| {
                part = @max(part, k.lane.partBytes(rows, q.nvfp4));
            },
            .mlp => |p| for ([_]cuda.qlinear.Weight{ p.gate, p.up, p.down }) |q| for (1..window_rows + 1) |rows| {
                part = @max(part, k.lane.partBytes(rows, q.nvfp4));
            },
        };
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .h = try m.alloc(r * d * 2),
            .h2 = try m.alloc(r * d * 2),
            .state = try m.alloc(r * d * 2),
            .input = try m.alloc(r * d * 2),
            .out = try m.alloc(r * d * 2),
            .q = try m.alloc(r * c.qDim() * 2),
            .kk = try m.alloc(r * c.kvDim() * 2),
            .vv = try m.alloc(r * c.kvDim() * 2),
            .g = try m.alloc(r * c.qDim() * 2),
            .qn = try m.alloc(r * c.qDim() * 2),
            .kn = try m.alloc(r * c.kvDim() * 2),
            .vn = try m.alloc(r * c.kvDim() * 2),
            .att = try m.alloc(r * c.qDim() * 2),
            .gated = try m.alloc(r * c.qDim() * 2),
            .mg = try m.alloc(r * c.intermediate * 2),
            .mu = try m.alloc(r * c.intermediate * 2),
            .m = try m.alloc(r * c.intermediate * 2),
            .part = try m.alloc(part),
            .win = try m.alloc(@max(WindowAttention.scratchBytes(m.window(true), window_rows), WindowAttention.scratchBytes(m.window(false), window_rows))),
            .sel = try m.alloc(r * d * 2),
        };
    }

    /// fp32 logits [rows.len, V] at `out` of `tokens` at positions p0.. (a prompt chunk, or decode rows: a window).
    pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, tokens: []const u32, p0: usize, prompt: bool, rows: []const usize, out: u64) !void {
        const c = m.c;
        const n = tokens.len;
        if (n == 0 or n > @as(usize, if (prompt) prompt_rows else window_rows) or p0 + n > m.context) return error.TooManyRows;
        if (n > m.ring - c.window) return error.ChainOverrunsRing;
        var host: [2 * prompt_rows]i32 = undefined;
        for (tokens, 0..) |t, i| {
            host[i] = @intCast(t);
            host[n + i] = @intCast(p0 + i);
        }
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(host[0..n]));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(host[n .. 2 * n]));
        const lin: cuda.qlinear.Linear = .{ .lane = &k.lane, .nvfp4_prompt = &k.prompt };
        const d = c.hidden;
        const f = c.factors();
        const x = m.s;
        try k.embed.bf16(s, x.ids, m.w.embed, x.h, n, d);
        var h = x.h;
        var h2 = x.h2;
        for (m.sched.slice()) |step| {
            if (step.enter) {
                if (step.loop == 0) {
                    try m.copy(s, x.input, h, n * d * 2);
                    try m.copy(s, x.state, h, n * d * 2);
                }
                try k.norms.l2(s, x.state, x.input, h, n, d, eps);
            }
            const alpha = switch (m.w.subs[step.sub]) {
                .attn => |a| blk: {
                    try m.attention(k, s, lin, a, step, h, n, p0, prompt);
                    break :blk a.alpha;
                },
                .mlp => |p| blk: {
                    try m.proj(lin, s, p.gate, h, x.mg, n, prompt);
                    try m.proj(lin, s, p.up, h, x.mu, n, prompt);
                    try k.act.swigluL2(s, x.mg, c.intermediate, x.mu, c.intermediate, p.su_g, p.su_u, x.m, n, c.intermediate, eps);
                    try m.proj(lin, s, p.down, x.m, x.out, n, prompt);
                    break :blk p.alpha;
                },
            };
            try k.norms.slerp(s, h, x.out, alpha, f.alpha, h2, n, d, eps);
            std.mem.swap(u64, &h, &h2);
            if (step.leave) {
                try k.norms.l2(s, h, null, x.state, n, d, eps);
                try m.copy(s, h, x.state, n * d * 2);
            }
        }
        for (rows, 0..) |r, j| try m.copy(s, x.sel + j * d * 2, h + r * d * 2, d * 2);
        try k.dense.rowsScaled(s, x.sel, d, m.w.head, m.w.logit, out, c.vocab, rows.len, c.vocab, d);
    }

    fn attention(m: *Model, k: *const Kernels, s: cuda.Stream, lin: cuda.qlinear.Linear, a: wts.Attn, step: schedule.Step, h: u64, n: usize, p0: usize, prompt: bool) !void {
        const c = m.c;
        const x = m.s;
        try m.proj(lin, s, a.q, h, x.q, n, prompt);
        try m.proj(lin, s, a.k, h, x.kk, n, prompt);
        try m.proj(lin, s, a.v, h, x.vv, n, prompt);
        try m.proj(lin, s, a.g, h, x.g, n, prompt);
        const full = step.kind == .full;
        const cache = step.cache.?;
        const dh: i64 = @intCast(c.head_dim);
        const src: cuda.qk_rows.Src = .{ .q = x.q, .q_ld = @intCast(c.qDim()), .q_hs = dh, .k = x.kk, .k_ld = @intCast(c.kvDim()), .k_hs = dh, .v = x.vv, .v_ld = @intCast(c.kvDim()), .v_hs = dh };
        const shape: cuda.qk_rows.Shape = .{ .l2 = true, .q_scale = a.qs, .inv = m.inv[@intFromBool(full)], .rot = c.head_dim, .eps = eps };
        const len = if (full) m.context else m.ring;
        try k.qk.rowsFrom(s, src, .{ .h = c.heads, .hk = c.kv_heads, .dim = c.head_dim }, shape, x.pos, null, .{ .q = x.qn, .k = x.kn, .v = x.vn, .kc = m.k[cache], .vc = m.v[cache], .len = len }, n);
        const w = m.window(full);
        if (prompt) try k.win.promptRows(s, x.qn, m.k[cache], m.v[cache], x.att, p0, n, w, null) else try k.win.decode(s, x.qn, m.k[cache], m.v[cache], x.att, x.pos, null, w, x.win, n);
        try k.act.gateL2(s, x.att, x.g, c.qDim(), a.sg, x.gated, n, c.qDim(), eps);
        try m.proj(lin, s, a.o, x.gated, x.out, n, prompt);
    }

    fn proj(m: *const Model, lin: cuda.qlinear.Linear, s: cuda.Stream, w: cuda.qlinear.Weight, in: u64, out: u64, n: usize, prompt: bool) !void {
        const rows: cuda.qlinear.Rows = .{ .x = in, .part = m.s.part };
        try (if (prompt) lin.prompt(s, w, rows, out, n) else lin.decode(s, w, rows, out, n));
    }
};

test "the forward's launches type-check against the shared operators" {
    std.testing.refAllDecls(Model);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(wts);
}
