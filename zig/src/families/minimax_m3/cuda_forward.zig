//! MiniMax-M3.1-Flash's TP forward on CUDA: the Metal family's layer loop with every rank's partials gathered.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const kern = @import("cuda_kernels.zig");

pub const State = state.State;
pub const Buffers = state.Buffers;

/// The EXL3 kernels: dense projections (their scratch grows with each weight's reserve), the grouped experts.
pub const Exl3 = struct {
    dense: cuda.exl3.Dense,
    experts: cuda.exl3_experts.Experts,
    grouped: cuda.exl3_grouped.Grouped,
};

/// A forward's kernels: EXL3 projections and experts, the lane (bf16 indexer, dspark/'s MXFP8), the router, our glue.
pub const Modules = struct {
    gpa: std.mem.Allocator,
    experts_mod: cuda.Module, // the grouped plan's router
    x3: *Exl3,
    lane: cuda.qmmf.Lane,
    router: cuda.grouped.Router,
    k: kern.Kernels,

    pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        var m: Modules = .{ .gpa = gpa, .experts_mod = try cuda.Module.load(d, cuda.kernels.experts), .x3 = undefined, .lane = undefined, .router = undefined, .k = undefined };
        errdefer m.experts_mod.unload();
        m.x3 = try gpa.create(Exl3);
        errdefer gpa.destroy(m.x3);
        m.x3.dense = try cuda.exl3.Dense.load(d);
        errdefer m.x3.dense.unload();
        m.x3.experts = try cuda.exl3_experts.Experts.load(d);
        errdefer m.x3.experts.unload();
        m.x3.grouped = .{ .d = d, .kernels = &m.x3.experts };
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.router = try cuda.grouped.Router.resolve(m.experts_mod);
        m.k = try kern.Kernels.load(d);
        return m;
    }

    pub fn deinit(m: *Modules) void {
        m.k.unload();
        m.lane.unload();
        m.x3.grouped.deinit();
        m.x3.experts.unload();
        m.x3.dense.unload();
        m.gpa.destroy(m.x3);
        m.experts_mod.unload();
        m.* = undefined;
    }

    /// What the loader uploads with.
    pub fn tools(m: *const Modules, d: *const cuda.Driver, s: cuda.Stream) weights.Tools {
        return .{ .d = d, .stream = s, .dense = &m.x3.dense };
    }

    /// The grouped experts' scratch for a window of `rows` (allocated before serving, never mid-forward).
    pub fn reserve(m: *const Modules, c: *const cfg.Config, share: anytype, rows: usize, prefill: bool) !void {
        const items = cuda.grouped.maxItems(rows * c.topk, c.experts, if (prefill) 64 else 16);
        const ni = share.moe.len();
        const up = try cuda.exl3_grouped.Grouped.bytes(c.hidden, ni, items, try cuda.exl3_experts.plan(c.hidden, ni, true), 2);
        const down = try cuda.exl3_grouped.Grouped.bytes(ni, c.hidden, items, try cuda.exl3_experts.plan(ni, c.hidden, false), 1);
        const need = @max(up, down);
        const g = &m.x3.grouped;
        if (g.storage != null and g.storage.?.len >= need) return;
        const replacement = try cuda.DeviceBuffer.alloc(g.d, need);
        if (g.storage) |*old| old.free();
        g.storage = replacement;
    }

    fn grouped(m: *const Modules) cuda.experts.Grouped {
        return .{ .exl3 = &m.x3.grouped };
    }
};

/// How the ranks' partials meet: one rank alone, or every rank's slot gathered over NCCL (RoCE for small ones).
pub const Exchange = union(enum) {
    local,
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, group: ?*const cuda.tp_group.Group = null },
};

/// Called after each layer with the residual rows (a reference check's dumps).
pub const Watch = struct { ctx: *anyopaque, call: *const fn (*anyopaque, usize, u64, usize) anyerror!void };

/// The engine's image rows and the sequence the forward runs on (they splice into that sequence's rows only).
pub const ImageRows = struct { rows: u64, spans: []const @import("cuda_vision.zig").Span };

/// One stream's rows in a shared forward: rows [row0, row0 + rows) on its sequence.
pub const Seg = struct { st: *State, row0: usize, rows: usize };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *const Modules,
    w: *const weights.Weights,
    stream: cuda.Stream,
    b: *Buffers,
    exchange: Exchange = .local,
    watch: ?Watch = null,
    taps: ?u64 = null, // a drafter's feature rows [rows][taps][hidden] bf16
    tap_at: []const i32 = &.{}, // each layer's tap slot, or -1
    images: ?ImageRows = null, // a prompt pass's image rows over their positions (its sequence only)

    fn c(f: *const Forward) *const cfg.Config {
        return &f.w.config;
    }

    /// Every rank's partial of `rows` rows (`slots` a row) into every rank's gathered buffer, rank order.
    pub fn gather(f: *const Forward, rows: usize, slots: usize) !void {
        const bytes = rows * slots * f.c().hidden * 4;
        switch (f.exchange) {
            .local => {},
            .nccl => |n| {
                const mine = f.b.gathered + f.w.share.rank * bytes;
                if (n.group) |g| return g.allGather(f.stream, mine, f.b.gathered, bytes);
                try n.lib.check(n.lib.api.ncclAllGather(mine, f.b.gathered, rows * slots * f.c().hidden, .f32, n.comm, f.stream.handle), "ncclAllGather");
            },
        }
    }

    /// This rank's slot of the gathered buffer for a site of `slots` a row.
    pub fn slot(f: *const Forward, rows: usize, slots: usize) u64 {
        return f.b.gathered + f.w.share.rank * rows * slots * f.c().hidden * 4;
    }

    /// y [rows][w.n] (bf16, or fp32 a TP rank's partial) = packed bf16 x [rows][w.k] through an EXL3 projection.
    pub fn x3(f: *const Forward, w: weights.Linear, x: u64, rows: usize, y: u64, out: cuda.qmmf.Out) !void {
        const dt: cuda.exl3.DType = if (out == .f32) .f32 else .bf16;
        const dense = &f.m.x3.dense;
        if (f.b.prefill) return dense.prompt(f.stream, w, x, .bf16, rows, y, dt);
        try dense.decode(f.stream, w, x, .bf16, rows, y, dt);
    }

    /// The same through a lane weight, x rows `ldx` apart.
    pub fn mx(f: *const Forward, w: weights.Mx, x: u64, ldx: usize, rows: usize, y: u64, out: cuda.qmmf.Out) !void {
        try f.m.lane.matmulOut(f.stream, x, ldx, rows, w, y, out, f.b.part);
    }

    /// `ids` at each segment's positions: the last row's logits (a prompt chunk) or every row's, the caches written.
    pub fn run(f: *const Forward, ids: []const i32, segs: []const Seg) !void {
        const R = ids.len;
        const b = f.b;
        const co = f.c();
        if (R == 0 or R > b.rows or segs.len == 0) return error.BadWindow;
        var total: usize = 0;
        for (segs) |g| {
            if (g.rows == 0 or g.st.pos + g.rows > g.st.capacity or g.st.capacity > b.context) return error.ContextFull;
            total += g.rows;
        }
        if (total != R) return error.BadWindow;
        try cuda.copy_source.enqueue(f.d, b.ids, std.mem.sliceAsBytes(ids), f.stream);
        try kern.embed(&f.m.k, f.stream, b.ids, f.w.embed, b.resid, R, co.hidden);
        if (f.images) |im| if (segs.len == 1) try @import("cuda_vision.zig").splice(&f.m.k, f.stream, im.rows, im.spans, segs[0].st.pos, segs[0].st.pos + R, b.resid);
        try kern.norm(&f.m.k, f.stream, b.resid, f.w.layers[0].in_norm, b.x, R, co.hidden, co.eps);
        for (f.w.layers, 0..) |l, j| {
            try f.layer(l, j, R, segs, if (j + 1 < f.w.layers.len) f.w.layers[j + 1].in_norm else 0);
            if (f.taps) |t| if (j < f.tap_at.len and f.tap_at[j] >= 0) try kern.tap(&f.m.k, f.stream, b.resid, t, R, co.hidden, 5, @intCast(f.tap_at[j]));
            if (f.watch) |wt| try wt.call(wt.ctx, j, b.resid, R);
        }
        for (segs) |g| g.st.pending = g.rows;
        const first: usize = if (b.prefill) R - 1 else 0;
        const n = R - first;
        try kern.norm(&f.m.k, f.stream, b.resid + first * co.hidden * 4, f.w.norm, b.fnormed, n, co.hidden, co.eps);
        try f.x3(f.w.head, b.fnormed, n, b.logits, .bf16);
    }

    /// Layer `l` on every segment's rows (b.x their normed input); the last add norms the residual by `next` (0: none).
    fn layer(f: *const Forward, l: weights.Layer, j: usize, R: usize, segs: []const Seg, next: u64) !void {
        const b = f.b;
        const co = f.c();
        const h: usize = co.hidden;
        const world = f.w.share.world;
        const in = f.projections(R);
        try f.x3(l.attn.q, b.x, R, in.q, .bf16);
        try f.x3(l.attn.k, b.x, R, in.k, .bf16);
        try f.x3(l.attn.v, b.x, R, in.v, .bf16);
        try f.mx(l.attn.index, b.x, h, R, in.i, .bf16);
        for (segs) |g| try f.attention(l.attn, j, g, R);
        try f.x3(l.attn.o, b.o, R, f.slot(R, 1), .f32);
        try f.gather(R, 1);
        try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, world, 1, 1.0, l.post_norm, b.x, R, h, co.eps);
        switch (l.ffn) {
            .dense => |m| {
                try f.mlp(m, f.w.share.dense.len(), R, f.slot(R, 1), .f32);
                try f.gather(R, 1);
                try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, world, 1, 1.0, next, b.x, R, h, co.eps);
            },
            .moe => |m| {
                try f.moe(m, R);
                try f.gather(R, 2);
                try kern.addNorm(&f.m.k, f.stream, b.resid, b.gathered, world, 2, co.routed_scale, next, b.x, R, h, co.eps);
            },
        }
    }

    /// Where a window of `rows` keeps its projections in b.qkv: q, k, v and index q | index k, each [rows][width].
    pub fn projections(f: *const Forward, rows: usize) struct { q: u64, k: u64, v: u64, i: u64 } {
        const co = f.c();
        const s = f.w.share;
        const D: usize = co.head_dim;
        const q = f.b.qkv;
        const k = q + rows * s.queries(co).len() * D * 2;
        const v = k + rows * s.kv.len() * D * 2;
        return .{ .q = q, .k = k, .v = v, .i = v + rows * s.kv.len() * D * 2 };
    }

    /// gate and up of b.x, SwiGLU-OAI, down into `out` (fp32 partial rows).
    pub fn mlp(f: *const Forward, m: weights.Mlp, inter: usize, R: usize, out: u64, kind: cuda.qmmf.Out) !void {
        const b = f.b;
        const co = f.c();
        const up = b.gu + R * inter * 2;
        try f.x3(m.gate, b.x, R, b.gu, .bf16);
        try f.x3(m.up, b.x, R, up, .bf16);
        try kern.swigluOai(&f.m.k, f.stream, b.gu, up, b.act, R, inter, inter, co.swiglu_alpha, co.swiglu_limit);
        try f.x3(m.down, b.act, R, out, kind);
    }

    /// dspark/'s MLP: gate | up stacked on the lane, SwiGLU-OAI, down into `out`.
    pub fn mlpMx(f: *const Forward, m: weights.MxMlp, inter: usize, R: usize, out: u64, kind: cuda.qmmf.Out) !void {
        const b = f.b;
        const co = f.c();
        try f.mx(m.gu, b.x, co.hidden, R, b.gu, .bf16);
        try kern.swigluOai(&f.m.k, f.stream, b.gu, b.gu + inter * 2, b.act, R, inter, 2 * inter, co.swiglu_alpha, co.swiglu_limit);
        try f.mx(m.down, b.act, inter, R, out, kind);
    }

    /// The routed and shared experts on b.x: this rank's partial rows (routed weighted, shared) into its slot.
    fn moe(f: *const Forward, m: weights.Moe, R: usize) !void {
        const b = f.b;
        const co = f.c();
        const h: usize = co.hidden;
        const topk: usize = co.topk;
        const pairs = R * topk;
        try kern.router(&f.m.k, f.stream, b.x, m.router, b.logits32, R, h, co.experts);
        try kern.route(&f.m.k, f.stream, b.logits32, m.bias, b.pick, b.wts, R, co.experts, topk);
        cuda.route_tap_copy.afterPick(f.stream, b.pick, b.wts, R, topk);
        const tile: usize = if (b.prefill) 64 else 16;
        try f.m.router.route(f.stream, b.pick, pairs, co.experts, tile, b.plan());
        const g = f.m.grouped();
        const in_up: cuda.experts.Rows = .{ .x = b.x, .stride = h, .slots = topk };
        const in_down: cuda.experts.Rows = .{ .x = b.eact, .stride = m.routed.width };
        // one path for a prompt chunk and a window: every pick's down rows in fp32, so a row's bits never depend on it
        try g.decode(f.stream, .up, in_up, m.routed, b.plan(), b.eact, b.max_items, -1);
        try g.decode(f.stream, .down_f32, in_down, m.routed, b.plan(), b.ey, b.max_items, -1);
        try f.mlp(m.shared, f.w.share.shared.len(), R, b.sy, .f32);
        try kern.moePartial(&f.m.k, f.stream, b.ey, b.wts, b.sy, f.slot(R, 2), R, topk, h);
    }

    /// One segment's rows: q, k, v and the index pair prepared, the block selection, attention into b.o.
    fn attention(f: *const Forward, a: weights.Attention, j: usize, g: Seg, window: usize) !void {
        const b = f.b;
        const co = f.c();
        const s = f.w.share;
        const hq = s.queries(co).len();
        const hk = s.kv.len();
        const hi = hk;
        const D: usize = co.head_dim;
        const st = g.st;
        const R = g.rows;
        const pos = st.pos;
        const base = g.row0;
        const q = b.q + base * hq * D * 2;
        const iq = b.iq + base * hi * co.index_dim * 2;
        const o = b.o + base * hq * D * 2;
        const in = f.projections(window); // the window's parts, this segment's rows from row0
        const iw = (hi + 1) * co.index_dim;
        try kern.prep(&f.m.k, f.stream, .{
            .xq = kern.Rows.of(in.q + base * hq * D * 2, hq * D),
            .xk = kern.Rows.of(in.k + base * hk * D * 2, hk * D),
            .xv = kern.Rows.of(in.v + base * hk * D * 2, hk * D),
            .xi = kern.Rows.of(in.i + base * iw * 2, iw),
            .qw = a.q_norm,
            .kw = a.k_norm,
            .iqw = a.iq_norm,
            .ikw = a.ik_norm,
            .q = q,
            .iq = iq,
            .kc = st.keys(j),
            .vc = st.values(co, j),
            .ik = st.indexKeys(co, j),
            .rows = R,
            .pos = pos,
            .hq = hq,
            .hk = hk,
            .hi = hi,
            .rotary = co.rotary,
            .eps = co.eps,
            .theta = co.theta,
        });
        const nb = co.blocks(pos + R);
        const topk: usize = co.topk_blocks;
        const all = nb <= topk;
        const sel_at = base * hi;
        const scores = b.scores + sel_at * co.blocks(b.context) * 4;
        const sel = b.sel + sel_at * topk * 4;
        // every row on the tensor units, a prompt chunk's and a window's alike: a row's bits never depend on its window
        const words = b.selWords(co);
        const bits = b.bits + sel_at * words * 4;
        if (!all) {
            try kern.indexScores(&f.m.k, f.stream, iq, st.indexKeys(co, j), scores, R, pos, hi, nb);
            try kern.indexSelect(&f.m.k, f.stream, scores, sel, bits, R, pos, hi, nb, topk, words);
        }
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
        try kern.attnPrompt(&f.m.k, f.stream, q, st.keys(j), st.values(co, j), bits, o, R, pos, hq, hk, if (all) 0 else words, scale);
    }

    /// The first `keep` rows of the last forward committed on each segment's sequence.
    pub fn commit(segs: []const Seg, keep: []const usize) !void {
        for (segs, keep) |g, k| {
            if (k == 0 or k > g.st.pending) return error.BadCommit;
            g.st.pos += k;
            g.st.pending = 0;
        }
    }
};
