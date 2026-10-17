//! Nemotron-H's forwards on CUDA in the Python engine's kernel order: a verify window, and a prompt chunk.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const glue = @import("cuda_glue.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const Dump = @import("cuda_dump.zig").Dump;
const train_ops = @import("cuda_train_ops.zig");
const lesson_route = @import("lesson_route.zig");
const kept = @import("cuda_marks.zig");
const latent = @import("cuda_latent.zig");

pub const Delta = enum { none, dense, moe };

/// A second stream the shared expert runs on while the routed experts are chosen; fork and join are events.
pub const Side = struct { s: cuda.Stream, fork: cuda.Event, join: cuda.Event };

/// A verify window supplies its stream buffers and starting row.
pub const Seg = struct { b: *const state.Buffers, row0: usize, rows: usize, sampled: bool, mask: Mask = .{}, slot: u32 = 0, untaught: bool = false };

/// Structured output: a segment's rows' grammar bits (device-readable, `words` a row); bits 0: no grammar.
pub const Mask = struct { bits: u64 = 0, words: usize = 0 };

/// A prompt chunk carries rows, position, residual delta and mixer positions between blocks.
pub const Walk = struct { rows: usize, pos: usize, x: u64 = 0, delta: Delta = .none, mj: usize = 0, aj: usize = 0 };

/// A window's kernel classes as a profile names them: each mark closes the span since the one before.
pub const Class = enum { embed, norm, mamba_in, conv_scan, gnorm, mamba_out, qkv, attention, attn_o, route, plan, experts_up, experts_down, shared_wait, head, sample };

/// Timing events recorded between kernel classes (the window profile's; null in every other forward).
pub const Marks = struct {
    ev: []cuda.Event,
    class: []Class,
    n: usize = 0,

    fn add(m: *Marks, s: cuda.Stream, c: Class) !void {
        if (m.n == m.ev.len) return error.TooManyMarks;
        try m.ev[m.n].record(s);
        m.class[m.n] = c;
        m.n += 1;
    }
};

pub const Forward = struct {
    c: Config,
    w: *const weights.Weights,
    b: *const state.Buffers,
    ops: kern.Ops,
    glue: glue.Glue,
    max_len: usize,
    nch: usize,
    sampled: bool, // the target draws by the bound sequence's rule (sample.cu); false: torch.argmax
    dump: ?*Dump = null,
    side: ?Side = null, // decode MoE layers run the shared expert here (null: one stream)
    marks: ?*Marks = null, // tf-cuda-test window-profile's events; null costs one check a class, no GPU work
    route: lesson_route.Route = .{}, // which rows carry the lessons (none set: no row does)
    slot: u32 = 0, // the bound sequence's lesson latch slot, which a window's rows read
    untaught: bool = false, // the pocket leaves rows (the model as shipped): `segs`' untaught sequences', else all
    segs: []const Seg = &.{}, // a window's sequences, as windowSegs runs them
    captures: []const kept.Mark = &.{}, // marks inside the prompt chunk: their Mamba states are copied out as it runs
    tp: ?*const cuda.tp_group.Group = null, // a latent MoE's ranks: their expert rows meet every MoE layer
    vision: ?*const @import("cuda_vision.zig").Vision = null, // a prompt pass's image rows, spliced after its embeds
    page_elems: u64 = 0, // elements between a paged sequence's pages (its layers' K and V of state.chunk_keys rows)

    pub fn init(c: Config, w: *const weights.Weights, b: *const state.Buffers, ops: kern.Ops, max_len: usize, nch: usize, sampled: bool) Forward {
        return .{
            .c = c,
            .w = w,
            .b = b,
            .ops = .{ .k = ops.k, .s = ops.s, .ck8 = b.ck8, .part = b.lane_part }, // FP8 rows and lane partials in these buffers
            .glue = .{
                .set = if (ops.k.triton) |*t| t else null,
                .f = &ops.k.glue,
                .s = ops.s,
                .emb = &ops.k.emb,
                .norms = &ops.k.norms,
            },
            .max_len = max_len,
            .nch = nch,
            .sampled = sampled,
        };
    }

    /// Rows of logits draw their next tokens into `out`, row r at position META[0] + r + 1, by `rule` when sampled.
    pub fn sample(f: *const Forward, logits: u64, rows: usize, meta: u64, out: u64, sampled: bool, rule: u64) !void {
        const v = f.c.vocab;
        if (!sampled) return f.ops.torch().argmax(logits, v, v, out, rows);
        try f.ops.draw(logits, v, rule, meta, 0, out, rows, null, null);
    }

    pub fn mark(f: *const Forward, c: Class) !void {
        if (f.marks) |m| try m.add(f.ops.s, c);
    }

    fn mshape(f: *const Forward) glue.Shape {
        const c = f.c;
        return .{ .proj = c.projDim(), .xd = c.inner(), .cd = c.convDim(), .heads = c.mamba_heads, .dh = c.mamba_head_dim, .groups = c.groups, .state = c.state, .rmax = state.windowRows(c) };
    }

    pub fn ashape(f: *const Forward) glue.Attn {
        return .{ .nqkv = f.c.qkvDim(), .heads = f.c.heads, .kv_heads = f.c.kv_heads, .dim = f.c.head_dim, .nch = f.nch, .pt = f.b.table, .ps = f.page_elems };
    }

    /// Attention layer `layer`'s plane in `b`'s cache at `base`: max_len rows a layer, or a page's K and V a layer.
    fn cache(f: *const Forward, b: *const state.Buffers, base: u64, layer: usize) u64 {
        const row: u64 = state.kvRowBytes(f.c);
        return base + layer * (if (b.table != 0) 2 * state.chunk_keys else f.max_len) * row;
    }

    /// Engine.norm: h = x + delta (dense, or the experts' slot-order sum), then the block's RMSNorm into y and xs.
    pub fn norm(f: *const Forward, x: *u64, delta: Delta, w: u64, rows: usize, moe_f32: bool) !void {
        const b = f.b;
        const c = f.c;
        const next = if (x.* == b.h[0]) b.h[1] else b.h[0];
        switch (delta) {
            .none => try f.glue.addRmsnorm(x.*, null, w, x.*, b.y, b.xs, rows, c.hidden, c.eps),
            .dense => try f.glue.addRmsnorm(x.*, b.delta, w, next, b.y, b.xs, rows, c.hidden, c.eps),
            .moe => try f.glue.addMoeNorm(x.*, b.ymoe, moe_f32, b.wts, w, next, b.y, b.xs, rows, c.hidden, c.eps, c.top_k, c.slots()),
        }
        if (delta != .none) x.* = next;
        if (f.dump) |d| try d.norm(f.ops, x.*, b.y, b.xs, rows, c.hidden);
        try f.mark(.norm);
    }

    /// Engine._forward: rows tokens at meta's position; every row samples its next token into `sampled`.
    pub fn window(f: *const Forward, rows: usize) !void {
        try f.windowSegs(&.{.{ .b = f.b, .row0 = 0, .rows = rows, .sampled = f.sampled, .slot = f.slot, .untaught = f.untaught }});
    }

    /// Streams share matmuls, norms and experts but keep their mixer state; each row's latch alone picks its lessons.
    pub fn windowSegs(f: *const Forward, segs: []const Seg) !void {
        const routed = f.routedWindow(segs);
        try routed.windowRows(segs);
    }

    /// Selects the window schedule while retaining each segment's lesson latch and state.
    pub fn routedWindow(f: *const Forward, segs: []const Seg) Forward {
        var routed = f.*;
        routed.segs = segs;
        if (f.c.latent > 0 and (segs.len > 1 or f.tp != null)) routed.side = null;
        routed.untaught = for (segs) |s| {
            if (s.untaught) break true;
        } else false;
        if (routed.route.count == 0) for (segs[0..@min(segs.len, lesson_route.max_segments)]) |s| routed.route.later(s.row0, s.rows, s.slot);
        return routed;
    }

    fn windowRows(f: *const Forward, segs: []const Seg) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const t = f.glue;
        const ms = f.mshape();
        const at = f.ashape();
        const D: u64 = c.hidden;
        var rows: usize = 0;
        for (segs) |s| rows += s.rows;
        if (rows > ms.rmax) return error.WindowTooWide;
        for (segs) |s| try f.embed(s.b.ids, b.emb + s.row0 * D * 2, s.rows);
        try f.mark(.embed);
        var x = b.emb;
        var delta: Delta = .none;
        var mj: usize = 0;
        var aj: usize = 0;
        for (f.w.blocks) |blk| {
            try f.norm(&x, delta, blk.norm, rows, true);
            switch (blk.kind) {
                .mamba => {
                    const m = blk.mamba;
                    const W: u64 = ms.rmax;
                    try o.dense(b.y, b.xs, m.in_proj, b.proj, rows);
                    try f.mark(.mamba_in);
                    var k0: usize = 0;
                    while (k0 < segs.len) : (k0 += glue.max_segs) {
                        const part = segs[k0..@min(segs.len, k0 + glue.max_segs)];
                        var ms_segs: [glue.max_segs]glue.MambaSeg = undefined;
                        for (part, ms_segs[0..part.len]) |s, *out| {
                            const sb = s.b;
                            out.* = .{
                                .proj = b.proj + s.row0 * c.projDim() * 2,
                                .base = sb.conv_base + @as(u64, mj) * 3 * c.convDim() * 2,
                                .raw = sb.raw + mj * 2 * W * c.convDim() * 2,
                                .xc = sb.xc + mj * 2 * W * c.convDim() * 2,
                                .dt = sb.dt + mj * 2 * W * c.mamba_heads * 4,
                                .state = sb.ssm + @as(u64, mj) * c.mamba_heads * c.mamba_head_dim * c.state * 4,
                                .meta = sb.meta,
                                .y = b.sy + s.row0 * c.inner() * 2,
                                .rows = s.rows,
                            };
                        }
                        try t.mambaSegs(ms_segs[0..part.len], m.conv_w, m.conv_b, m.a, m.d, m.dt_bias, c.dt_min, c.dt_max, ms);
                    }
                    try f.mark(.conv_scan);
                    try t.groupRmsnorm(b.sy, m.gnorm, b.g, b.gxs, rows, c.inner(), c.groups, c.eps);
                    try f.mark(.gnorm);
                    try o.dense(b.g, b.gxs, m.out_proj, b.delta, rows);
                    if (m.out_rest != 0) try o.restRows(b.g, c.inner(), m.out_rest, b.delta, c.hidden, false, rows, 1, 0, c.hidden, c.inner());
                    if (m.adapter) |ad| try adapt(o, ad, b.g, c.inner(), b.delta, c.hidden, false, rows, &f.route);
                    try f.mark(.mamba_out);
                    delta = .dense;
                    mj += 1;
                },
                .attention => {
                    const a = blk.attn;
                    const qd: u64 = c.heads * c.head_dim;
                    try o.dense(b.y, b.xs, a.qkv, b.qkv, rows);
                    try f.mark(.qkv);
                    var k0: usize = 0;
                    while (k0 < segs.len) : (k0 += glue.max_segs) {
                        const part = segs[k0..@min(segs.len, k0 + glue.max_segs)];
                        var at_segs: [glue.max_segs]glue.AttnSeg = undefined;
                        for (part, at_segs[0..part.len]) |s, *out| out.* = .{ .kc = f.cache(s.b, s.b.k_cache, aj), .vc = f.cache(s.b, s.b.v_cache, aj), .meta = s.b.meta, .row0 = s.row0, .rows = s.rows, .pt = s.b.table };
                        try t.attentionSegs(at_segs[0..part.len], b.qkv, b.po, b.pm, b.pl, b.att, b.axs, at);
                    }
                    try f.mark(.attention);
                    try o.dense(b.att, b.axs, a.o, b.delta, rows);
                    if (a.o_rest != 0) try o.restRows(b.att, qd, a.o_rest, b.delta, c.hidden, false, rows, 1, 0, c.hidden, qd);
                    if (a.adapter) |ad| try adapt(o, ad, b.att, qd, b.delta, c.hidden, false, rows, &f.route);
                    try f.mark(.attn_o);
                    delta = .dense;
                    aj += 1;
                },
                .moe => if (blk.moe.latent != null) {
                    try latent.run(f, blk.moe, rows);
                    delta = .dense;
                } else {
                    try f.experts(blk.moe, rows, false);
                    delta = if (blk.moe.bolt != null) .dense else .moe;
                },
            }
        }
        try f.norm(&x, delta, f.w.norm_f, rows, true);
        for (segs) |s| try o.copy(s.b.hidden, b.y + s.row0 * D * 2, s.rows * c.hidden * 2);
        try o.dense(b.y, b.xs, f.w.head, b.logits, rows);
        try f.mark(.head);
        for (segs) |s| if (s.mask.bits != 0) try o.grammarMask(b.logits + s.row0 * @as(u64, c.vocab) * 2, c.vocab, s.mask.bits, s.mask.words, s.rows);
        for (segs) |s| try f.sample(b.logits + s.row0 * @as(u64, c.vocab) * 2, s.rows, s.b.meta, s.b.sampled, s.sampled, s.b.rule);
        try f.mark(.sample);
        if (f.dump) |d| try d.tail(f.ops, b.logits, b.sampled, rows, c.vocab);
    }

    /// Token rows of the embedding table as the checkpoint stores it: MLX 4-bit words, or bf16.
    pub fn embed(f: *const Forward, ids: u64, out: u64, rows: usize) !void {
        const e = f.w.embed;
        if (e.bf16) return f.glue.embed16(ids, e.w, out, rows, f.c.hidden);
        return f.glue.embed(ids, e.w, e.s, e.b, out, rows, f.c.hidden);
    }

    /// The token rows up projections read, and the pair rows down projections read.
    pub fn rowsIn(f: *const Forward, ex: kern.Experts) [2]cuda.experts.Rows {
        return .{ .{ .x = f.b.y, .stride = f.c.hidden, .slots = f.c.slots() }, .{ .x = f.b.act, .stride = ex.width } };
    }

    /// Engine.moe / moe_rows: route each row's experts, group pairs by expert, then up (relu^2) and down.
    pub fn experts(f: *const Forward, m: weights.MoE, rows: usize, prompt: bool) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const ex = m.experts;
        const pairs = rows * c.slots();
        if (m.bolt) |lt| return f.latentExperts(m, lt, rows, prompt);
        if (!prompt) if (f.side) |sd| return f.forked(m, rows, sd);
        try f.glue.route(b.y, m.router, m.bias, b.part, b.pick, b.wts, rows, c.hidden, c.experts, c.top_k, c.slots(), c.routed_scaling, c.norm_topk);
        try f.mark(.route);
        const tile: usize = if (prompt) cuda.experts.promptTile(ex.format) else 16;
        try o.plan(b.pick, pairs, ex.experts, tile, b.plan);
        try f.mark(.plan);
        const items = kern.maxItems(pairs, ex.experts, tile);
        const g = o.grouped();
        const in = f.rowsIn(ex);
        if (prompt) {
            try g.prompt(o.s, .up, in[0], ex, b.plan, b.act, items, -1);
            try g.prompt(o.s, .down_bf16, in[1], ex, b.plan, b.ymoe, items, -1);
            if (f.dump) |d| for ([_][]const u8{ "pick", "act", "ymoe" }, [_]u64{ b.pick, b.act, b.ymoe }, [_]usize{ 4, ex.width * 2, ex.dims * 2 }) |l, p, w| try d.input(o, l, p, pairs * w);
        } else {
            try g.decode(o.s, .up, in[0], ex, b.plan, b.act, items, -1);
            try f.mark(.experts_up);
            try g.decode(o.s, .down_f32, in[1], ex, b.plan, b.ymoe, items, -1);
            try f.mark(.experts_down);
        }
        try f.sharedRest(o, m, rows, !prompt);
    }

    /// LatentMoE (Bolt), into b.delta: fc2_latent(sum of the routed experts on fc1_latent(y)) + the shared expert on y.
    fn latentExperts(f: *const Forward, m: weights.MoE, lt: weights.BoltLatent, rows: usize, prompt: bool) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const ex = m.experts;
        const K = c.top_k;
        const pairs = rows * K;
        try f.glue.route(b.y, m.router, m.bias, b.part, b.pick, b.wts, rows, c.hidden, ex.experts, K, K, c.routed_scaling, c.norm_topk);
        try f.mark(.route);
        if (prompt) try o.prefillDense(b.y, lt.fc1, b.lat, rows) else try o.dense(b.y, b.xs, lt.fc1, b.lat, rows);
        const tile: usize = if (prompt) cuda.experts.promptTile(ex.format) else 16;
        const g = o.grouped();
        try o.plan(b.pick, pairs, ex.experts, tile, b.plan);
        try f.mark(.plan);
        var items = kern.maxItems(pairs, ex.experts, tile);
        const up_in: cuda.experts.Rows = .{ .x = b.lat, .stride = c.latent, .slots = K };
        const down_in: cuda.experts.Rows = .{ .x = b.act, .stride = ex.width };
        if (prompt) {
            try g.prompt(o.s, .up, up_in, ex, b.plan, b.act, items, -1);
            try g.prompt(o.s, .down_f32, down_in, ex, b.plan, b.ymoe, items, -1);
        } else {
            try g.decode(o.s, .up, up_in, ex, b.plan, b.act, items, -1);
            try f.mark(.experts_up);
            try g.decode(o.s, .down_f32, down_in, ex, b.plan, b.ymoe, items, -1);
            try f.mark(.experts_down);
        }
        try o.latentSum(b.ymoe, b.wts, b.comb, rows, K, c.latent);
        if (prompt) try o.prefillDense(b.comb, lt.fc2, b.delta, rows) else try o.dense(b.comb, 0, lt.fc2, b.delta, rows);
        // the shared expert: every row's pair with expert 0 of a one-expert layer, relu^2 between as the routed ones
        const sh = lt.shared;
        try o.plan(b.sids, rows, 1, tile, b.plan);
        items = kern.maxItems(rows, 1, tile);
        const sh_up: cuda.experts.Rows = .{ .x = b.y, .stride = c.hidden, .slots = 1 };
        const sh_down: cuda.experts.Rows = .{ .x = b.act, .stride = sh.width };
        if (prompt) {
            try g.prompt(o.s, .up, sh_up, sh, b.plan, b.act, items, -1);
            try g.prompt(o.s, .down_f32, sh_down, sh, b.plan, b.ymoe, items, -1);
        } else {
            try g.decode(o.s, .up, sh_up, sh, b.plan, b.act, items, -1);
            try g.decode(o.s, .down_f32, sh_down, sh, b.plan, b.ymoe, items, -1);
        }
        try o.addBf16(b.delta, b.delta, b.ymoe, true, rows * c.hidden);
        try f.mark(.shared_wait);
    }

    /// The pocket over the rows of sequences that read what was learned: every row, or the runs between untaught ones'.
    fn pocket(f: *const Forward, o: kern.Ops, p: weights.Pocket, d: usize, y_f32: bool, rows: usize) !void {
        if (!f.untaught) return f.pocketRun(o, p, d, y_f32, .{ 0, rows });
        var runs: lesson_route.Runs = .{};
        for (f.segs) |s| if (runs.next(s.rows, s.untaught)) |r| try f.pocketRun(o, p, d, y_f32, r);
        if (runs.last()) |r| try f.pocketRun(o, p, d, y_f32, r);
    }

    /// The pocket's neurons over rows r[0]..r[0] + r[1]: their normed rows in, their shared expert's first slot out.
    fn pocketRun(f: *const Forward, o: kern.Ops, p: weights.Pocket, d: usize, y_f32: bool, r: [2]usize) !void {
        const c = f.c;
        const out = f.b.ymoe + @as(u64, r[0] * c.slots() * d) * @as(u64, if (y_f32) 4 else 2);
        try o.pocketRows(f.b.y + @as(u64, r[0] * d) * 2, d, p, f.b.pocket, out, y_f32, r[1], c.slots(), c.top_k);
    }

    /// A lesson's rest into the shared expert's halves' pair rows, a live change (--slide, 2W wide) into the first's.
    fn sharedRest(f: *const Forward, o: kern.Ops, m: weights.MoE, rows: usize, y_f32: bool) !void {
        const c = f.c;
        const ex = m.experts;
        for (m.rest, 0..) |r, h| if (r != 0) try o.restRows(f.b.act, ex.width, r, f.b.ymoe, ex.dims, y_f32, rows, c.slots(), c.top_k + h, ex.dims, ex.width);
        // the pocket's learned neurons read the normed rows as the shared expert's own do, into its first half's output
        if (m.pocket) |p| try f.pocket(o, p, ex.dims, y_f32, rows);
        // the learned change reads both halves (the whole shared activation) into the first half's output
        const ad = m.adapter orelse return;
        const x = f.b.act + @as(u64, c.top_k) * ex.width * 2;
        const y = f.b.ymoe + @as(u64, c.top_k) * ex.dims * @as(u64, if (y_f32) 4 else 2);
        try adapt(o, ad, x, c.slots() * ex.width, y, c.slots() * ex.dims, y_f32, rows, &f.route);
    }

    /// The decode MoE with its shared halves on the side stream: their pairs are fixed, so they need no routing.
    fn forked(f: *const Forward, m: weights.MoE, rows: usize, sd: Side) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const ex = m.experts;
        const so: kern.Ops = .{ .k = o.k, .s = sd.s };
        const g = o.grouped();
        const in = f.rowsIn(ex);
        try sd.fork.record(o.s);
        try sd.s.wait(sd.fork);
        const sp = b.sharedPlan(rows);
        try g.decode(sd.s, .up, in[0], ex, sp, b.act, 2, -1);
        try g.decode(sd.s, .down_f32, in[1], ex, sp, b.ymoe, 2, -1);
        try f.sharedRest(so, m, rows, true);
        try sd.join.record(sd.s);
        try f.glue.route(b.y, m.router, m.bias, b.part, b.pick, b.wts, rows, c.hidden, c.experts, c.top_k, c.slots(), c.routed_scaling, c.norm_topk);
        try f.mark(.route);
        try o.planRouted(b.pick, rows, c.slots(), c.top_k, c.experts, 16, b.plan);
        try f.mark(.plan);
        const items = kern.maxItems(rows * c.top_k, c.experts, 16);
        try g.decode(o.s, .up, in[0], ex, b.plan, b.act, items, -1);
        try f.mark(.experts_up);
        try g.decode(o.s, .down_f32, in[1], ex, b.plan, b.ymoe, items, -1);
        try f.mark(.experts_down);
        try o.s.wait(sd.join);
        try f.mark(.shared_wait);
    }

    /// Engine.prefill_chunk: `rows` tokens in p_ids at positions pos..; commits them and samples the next token.
    pub fn chunk(f: *const Forward, rows: usize, pos: usize) !void {
        var w: Walk = .{ .rows = rows, .pos = pos };
        try f.chunkBegin(&w);
        for (0..f.w.blocks.len) |i| {
            try f.chunkPre(&w, i);
            try f.chunkMixer(&w, i);
            try f.chunkPost(&w, i);
        }
        try f.chunkFinish(&w);
    }

    /// A chunk's embedding, from p_ids.
    pub fn chunkBegin(f: *const Forward, w: *Walk) !void {
        const b = f.b;
        try f.embed(b.p_ids, b.emb, w.rows);
        if (f.vision) |v| try v.splice(f.ops.s, w.pos, w.pos + w.rows, b.emb);
        w.x = b.emb;
    }

    /// Block i up to its mixer: the norm, then the Mamba input projection or attention's q, k and v.
    pub fn chunkPre(f: *const Forward, w: *Walk, i: usize) !void {
        const blk = f.w.blocks[i];
        const b = f.b;
        try f.norm(&w.x, w.delta, blk.norm, w.rows, false);
        switch (blk.kind) {
            .mamba => try f.ops.prefillDense(b.y, blk.mamba.in_proj, b.proj, w.rows),
            .attention => {
                try f.ops.prefillDense(b.y, blk.attn.qkv, b.qkv, w.rows);
                try f.ops.fill32(b.p_meta, @intCast(w.pos), 4);
            },
            .moe => {},
        }
    }

    /// A block mixer reads its Mamba state or attention cache from earlier rows.
    pub fn chunkMixer(f: *const Forward, w: *Walk, i: usize) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        switch (f.w.blocks[i].kind) {
            .mamba => {
                const m = f.w.blocks[i].mamba;
                const layer: u64 = c.mamba_heads * c.mamba_head_dim * c.state * 4;
                const base = b.conv_base + @as(u64, w.mj) * 3 * c.convDim() * 2;
                var snaps: kern.Snaps = .{}; // the scan stores its state at each mark too, the bits a chunk ending there stores
                for (f.captures, 0..) |cap, k| {
                    try f.keepTaps(cap, w, base); // before the commit below moves BASE on
                    snaps.p[k] = cap.ssm + w.mj * layer;
                    snaps.r[k] = @intCast(cap.at - w.pos);
                }
                try f.glue.convRows(b.proj, base, b.p_xc, m.conv_w, m.conv_b, w.rows, f.mshape());
                try o.scanRows(b.proj, b.p_xc, b.ssm + w.mj * layer, m.a, m.d, m.dt_bias, b.sy, w.rows, c.projDim(), c.mamba_heads, c.mamba_head_dim, c.convDim(), c.groups, c.dt_min, c.dt_max, snaps);
            },
            .attention => {
                const kc = f.cache(b, b.k_cache, w.aj);
                const vc = f.cache(b, b.v_cache, w.aj);
                const qd = c.heads * c.head_dim;
                try f.glue.kvWrite(b.qkv, kc, vc, b.p_meta, w.rows, f.ashape());
                try o.torch().copyRows(b.qkv, c.qkvDim() * 2, b.q, qd * 2, qd * 2, w.rows);
                const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
                try o.prefillAttention(b.q, kc, vc, b.att, w.pos, w.rows, c.heads, c.kv_heads, scale, b.table, f.page_elems);
            },
            .moe => {},
        }
    }

    /// A mark's conv taps in this layer: the raw inputs of the three rows before it, BASE's last rows then the chunk's.
    fn keepTaps(f: *const Forward, cap: kept.Mark, w: *const Walk, base: u64) !void {
        const cd: u64 = f.c.convDim();
        const dst = cap.conv + w.mj * 3 * cd * 2;
        const old = 3 -| (cap.at - w.pos); // taps from before the chunk; a mark inside it has at least one row
        try f.ops.copy(dst, base + (3 - old) * cd * 2, old * cd * 2);
        const first = f.b.proj + (cap.at - w.pos - (3 - old)) * f.c.projDim() * 2 + f.c.inner() * 2;
        try f.ops.copyRows(dst + old * cd * 2, cd * 2, first, f.c.projDim() * 2, cd * 2, 3 - old);
    }

    /// Block i after its mixer: Mamba gate norm and out projection, attention's out projection, or the experts.
    pub fn chunkPost(f: *const Forward, w: *Walk, i: usize) !void {
        const c = f.c;
        const b = f.b;
        const blk = f.w.blocks[i];
        switch (blk.kind) {
            .mamba => {
                try f.glue.groupRmsnorm(b.sy, blk.mamba.gnorm, b.g, b.gxs, w.rows, c.inner(), c.groups, c.eps);
                if (f.dump) |d| try d.input(f.ops, "g", b.g, w.rows * c.inner() * 2);
                try f.ops.prefillDense(b.g, blk.mamba.out_proj, b.delta, w.rows);
                if (blk.mamba.out_rest != 0) try f.ops.restRows(b.g, c.inner(), blk.mamba.out_rest, b.delta, c.hidden, false, w.rows, 1, 0, c.hidden, c.inner());
                if (blk.mamba.adapter) |ad| try adapt(f.ops, ad, b.g, c.inner(), b.delta, c.hidden, false, w.rows, &f.route);
                w.delta = .dense;
                w.mj += 1;
            },
            .attention => {
                if (f.dump) |d| try d.input(f.ops, "att", b.att, w.rows * c.heads * c.head_dim * 2);
                try f.ops.prefillDense(b.att, blk.attn.o, b.delta, w.rows);
                const qd = c.heads * c.head_dim;
                if (blk.attn.o_rest != 0) try f.ops.restRows(b.att, qd, blk.attn.o_rest, b.delta, c.hidden, false, w.rows, 1, 0, c.hidden, qd);
                if (blk.attn.adapter) |ad| try adapt(f.ops, ad, b.att, qd, b.delta, c.hidden, false, w.rows, &f.route);
                w.delta = .dense;
                w.aj += 1;
            },
            .moe => if (blk.moe.latent != null) {
                try latent.run(f, blk.moe, w.rows);
                w.delta = .dense;
            } else {
                try f.experts(blk.moe, w.rows, true);
                w.delta = if (blk.moe.bolt != null) .dense else .moe;
            },
        }
    }

    /// The chunk's last norm alone: w.x then holds the residual it read, b.y every row's normed hidden state.
    pub fn finalNorm(f: *const Forward, w: *Walk) !void {
        try f.norm(&w.x, w.delta, f.w.norm_f, w.rows, false);
    }

    /// The chunk's last norm, its rows' hidden states for the MTP head, and the token after its last row.
    pub fn chunkFinish(f: *const Forward, w: *Walk) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const rows = w.rows;
        try f.norm(&w.x, w.delta, f.w.norm_f, rows, false);
        try o.copy(b.p_hidden, b.y, @as(usize, rows) * c.hidden * 2);
        try o.prefillDense(b.y + @as(u64, rows - 1) * c.hidden * 2, f.w.head, b.p_logits, 1);
        try o.fill32(b.p_meta, @intCast(w.pos + rows - 1), 4); // sample_last: the chunk's last row is at pos + rows - 1
        try f.sample(b.p_logits, 1, b.p_meta, b.p_sampled, f.sampled, b.rule);
        if (f.dump) |d| try d.tail(f.ops, b.p_logits, b.p_sampled, 1, c.vocab);
    }
};

/// y += scale (x a^T) b over the blocks `route` opens on each row (x and y rows their strides apart, y bf16 or fp32).
fn adapt(o: kern.Ops, ad: weights.Adapter, x: u64, x_stride: usize, y: u64, y_stride: usize, y_f32: bool, rows: usize, route: *const lesson_route.Route) !void {
    const t: train_ops.Ops = .{ .k = &o.k.train, .s = o.s };
    try t.loraIn(x, x_stride, ad.a, ad.xa, rows, ad.in, ad.rank, ad.xn);
    var r = route.*;
    if (ad.whole) r.whole = 1;
    try t.gate(ad.xa, ad.xn, ad.tau, ad.gates, rows, ad.rank, ad.unit, &r, ad.latch, ad.score);
    try t.loraOut(ad.xa, ad.b, y, y_stride, y_f32, rows, ad.out, ad.rank, ad.scale);
}
