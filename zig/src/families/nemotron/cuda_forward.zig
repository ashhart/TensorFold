//! Nemotron-H's forwards on CUDA in the Python engine's kernel order: a verify window, and a prompt chunk.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const Tri = @import("cuda_triton.zig").Tri;
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const Dump = @import("cuda_dump.zig").Dump;

const Delta = enum { none, dense, moe };

/// A second stream the shared expert runs on while the routed experts are chosen; fork and join are events.
pub const Side = struct { s: cuda.Stream, fork: cuda.Event, join: cuda.Event };

/// A verify window supplies its stream buffers and starting row.
pub const Seg = struct { b: *const state.Buffers, row0: usize, rows: usize, sampled: bool };

/// A prompt chunk carries rows, position, residual delta and mixer positions between blocks.
pub const Walk = struct { rows: usize, pos: usize, x: u64 = 0, delta: Delta = .none, mj: usize = 0, aj: usize = 0 };

pub const Forward = struct {
    c: Config,
    w: *const weights.Weights,
    b: *const state.Buffers,
    ops: kern.Ops,
    tri: Tri,
    max_len: usize,
    nch: usize,
    sampled: bool, // the target draws by the bound sequence's rule (sample.cu); false: torch.argmax
    dump: ?*Dump = null,
    side: ?Side = null, // decode MoE layers run the shared expert here (null: one stream)

    pub fn init(c: Config, w: *const weights.Weights, b: *const state.Buffers, ops: kern.Ops, max_len: usize, nch: usize, sampled: bool) Forward {
        return .{ .c = c, .w = w, .b = b, .ops = ops, .tri = .{ .set = &ops.k.triton, .s = ops.s }, .max_len = max_len, .nch = nch, .sampled = sampled };
    }

    /// Rows of logits draw their next tokens into `out`, row r at position META[0] + r + 1, by `rule` when sampled.
    fn sample(f: *const Forward, logits: u64, rows: usize, meta: u64, out: u64, sampled: bool, rule: u64) !void {
        const v = f.c.vocab;
        if (!sampled) return f.ops.torch().argmax(logits, v, v, out, rows);
        try f.ops.draw(logits, v, rule, meta, 0, out, rows, null, null);
    }

    fn mshape(f: *const Forward, rmax: usize) Tri.Shape {
        const c = f.c;
        return .{ .proj = c.projDim(), .xd = c.inner(), .cd = c.convDim(), .heads = c.mamba_heads, .dh = c.mamba_head_dim, .groups = c.groups, .state = c.state, .rmax = rmax };
    }

    pub fn ashape(f: *const Forward) Tri.Attn {
        return .{ .nqkv = f.c.qkvDim(), .heads = f.c.heads, .kv_heads = f.c.kv_heads, .dim = f.c.head_dim, .nch = f.nch };
    }

    fn cache(f: *const Forward, base: u64, layer: usize) u64 {
        return base + layer * f.max_len * f.c.kv_heads * f.c.head_dim * 2;
    }

    /// Engine.norm: h = x + delta (dense, or the experts' slot-order sum), then the block's RMSNorm into y and xs.
    fn norm(f: *const Forward, x: *u64, delta: Delta, w: u64, rows: usize, moe_f32: bool) !void {
        const b = f.b;
        const c = f.c;
        const next = if (x.* == b.h[0]) b.h[1] else b.h[0];
        switch (delta) {
            .none => try f.tri.addRmsnorm(x.*, null, w, x.*, b.y, b.xs, rows, c.hidden, c.eps),
            .dense => try f.tri.addRmsnorm(x.*, b.delta, w, next, b.y, b.xs, rows, c.hidden, c.eps),
            .moe => try f.tri.addMoeNorm(x.*, b.ymoe, moe_f32, b.wts, w, next, b.y, b.xs, rows, c.hidden, c.eps, c.top_k, c.slots()),
        }
        if (delta != .none) x.* = next;
        if (f.dump) |d| try d.norm(f.ops, x.*, b.y, b.xs, rows, c.hidden);
    }

    /// Engine._forward: rows tokens at meta's position; every row samples its next token into `sampled`.
    pub fn window(f: *const Forward, rows: usize) !void {
        try f.windowSegs(&.{.{ .b = f.b, .row0 = 0, .rows = rows, .sampled = f.sampled }});
    }

    /// Shared matmuls, norms and experts retain each stream's mixer state.
    pub fn windowSegs(f: *const Forward, segs: []const Seg) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const t = f.tri;
        const ms = f.mshape(state.max_rows);
        const at = f.ashape();
        const D: u64 = c.hidden;
        var rows: usize = 0;
        for (segs) |s| rows += s.rows;
        if (rows > state.max_rows) return error.WindowTooWide;
        for (segs) |s| try t.embed(s.b.ids, f.w.embed.w, f.w.embed.s, f.w.embed.b, b.emb + s.row0 * D * 2, s.rows, c.hidden);
        var x = b.emb;
        var delta: Delta = .none;
        var mj: usize = 0;
        var aj: usize = 0;
        for (f.w.blocks) |blk| {
            try f.norm(&x, delta, blk.norm, rows, true);
            switch (blk.kind) {
                .mamba => {
                    const m = blk.mamba;
                    const W: u64 = state.max_rows;
                    try o.dense(b.y, b.xs, m.in_proj, b.proj, rows);
                    for (segs) |s| {
                        const sb = s.b;
                        const raw = sb.raw + mj * 2 * W * c.convDim() * 2;
                        const xc = sb.xc + mj * 2 * W * c.convDim() * 2;
                        const dt = sb.dt + mj * 2 * W * c.mamba_heads * 4;
                        const ssm = sb.ssm + @as(u64, mj) * c.mamba_heads * c.mamba_head_dim * c.state * 4;
                        const base = sb.conv_base + @as(u64, mj) * 3 * c.convDim() * 2;
                        const proj = b.proj + s.row0 * c.projDim() * 2;
                        try t.conv(proj, base, raw, xc, m.conv_w, m.conv_b, sb.meta, s.rows, ms);
                        try t.scan(proj, xc, dt, ssm, m.a, m.d, m.dt_bias, sb.meta, b.sy + s.row0 * c.inner() * 2, s.rows, c.dt_min, c.dt_max, ms);
                    }
                    try t.groupRmsnorm(b.sy, m.gnorm, b.g, b.gxs, rows, c.inner(), c.groups, c.eps);
                    try o.dense(b.g, b.gxs, m.out_proj, b.delta, rows);
                    delta = .dense;
                    mj += 1;
                },
                .attention => {
                    const a = blk.attn;
                    const qd: u64 = c.heads * c.head_dim;
                    try o.dense(b.y, b.xs, a.qkv, b.qkv, rows);
                    for (segs) |s| {
                        const kc = f.cache(s.b.k_cache, aj);
                        const vc = f.cache(s.b.v_cache, aj);
                        const qkv = b.qkv + s.row0 * c.qkvDim() * 2;
                        const r0: u64 = s.row0;
                        try t.kvWrite(qkv, kc, vc, s.b.meta, s.rows, at);
                        try t.attention(qkv, kc, vc, s.b.meta, b.po + r0 * f.nch * qd * 4, b.pm + r0 * f.nch * c.heads * 4, b.pl + r0 * f.nch * c.heads * 4, b.att + r0 * qd * 2, b.axs + r0 * (qd / 64) * 4, s.rows, at);
                    }
                    try o.dense(b.att, b.axs, a.o, b.delta, rows);
                    delta = .dense;
                    aj += 1;
                },
                .moe => {
                    try f.experts(blk.moe, rows, false);
                    delta = .moe;
                },
            }
        }
        try f.norm(&x, delta, f.w.norm_f, rows, true);
        for (segs) |s| try o.copy(s.b.hidden, b.y + s.row0 * D * 2, s.rows * c.hidden * 2);
        try o.dense(b.y, b.xs, f.w.head, b.logits, rows);
        for (segs) |s| try f.sample(b.logits + s.row0 * @as(u64, c.vocab) * 2, s.rows, s.b.meta, s.b.sampled, s.sampled, s.b.rule);
        if (f.dump) |d| try d.tail(f.ops, b.logits, b.sampled, rows, c.vocab);
    }

    /// Engine.moe / moe_rows: route each row's experts, group pairs by expert, then up (relu^2) and down.
    pub fn experts(f: *const Forward, m: weights.MoE, rows: usize, prompt: bool) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const ex = m.experts;
        const pairs = rows * c.slots();
        if (!prompt) if (f.side) |sd| return f.forked(m, rows, sd);
        try f.tri.route(b.y, m.router, m.bias, b.part, b.pick, b.wts, rows, c.hidden, c.experts, c.top_k, c.routed_scaling, c.norm_topk);
        const tile: usize = if (prompt) 64 else 16;
        try o.plan(b.pick, pairs, ex.count, tile, b.plan);
        const items = kern.maxItems(pairs, ex.count, tile);
        if (prompt) {
            try o.expertsPrefill(true, b.y, c.hidden, c.slots(), ex.up, ex.dims / 64, ex.width / 32, b.plan, b.act, ex.width, items);
            try o.expertsPrefill(false, b.act, ex.width, 0, ex.down, ex.width / 64, ex.dims / 32, b.plan, b.ymoe, ex.dims, items);
        } else {
            try o.experts(true, b.y, c.hidden, c.slots(), ex.up, ex.dims / 64, ex.width / 32, b.plan, b.act, ex.width, items * (ex.width / 32));
            try o.experts(false, b.act, ex.width, 0, ex.down, ex.width / 64, ex.dims / 32, b.plan, b.ymoe, ex.dims, items * (ex.dims / 32));
        }
    }

    /// The decode MoE with its shared halves on the side stream: their pairs are fixed, so they need no routing.
    fn forked(f: *const Forward, m: weights.MoE, rows: usize, sd: Side) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const ex = m.experts;
        const so: kern.Ops = .{ .k = o.k, .s = sd.s };
        try sd.fork.record(o.s);
        try sd.s.wait(sd.fork);
        const sp = b.sharedPlan(rows);
        try so.experts(true, b.y, c.hidden, c.slots(), ex.up, ex.dims / 64, ex.width / 32, sp, b.act, ex.width, 2 * (ex.width / 32));
        try so.experts(false, b.act, ex.width, 0, ex.down, ex.width / 64, ex.dims / 32, sp, b.ymoe, ex.dims, 2 * (ex.dims / 32));
        try sd.join.record(sd.s);
        try f.tri.route(b.y, m.router, m.bias, b.part, b.pick, b.wts, rows, c.hidden, c.experts, c.top_k, c.routed_scaling, c.norm_topk);
        try o.planRouted(b.pick, rows, c.slots(), c.top_k, c.experts, 16, b.plan);
        const items = kern.maxItems(rows * c.top_k, c.experts, 16);
        try o.experts(true, b.y, c.hidden, c.slots(), ex.up, ex.dims / 64, ex.width / 32, b.plan, b.act, ex.width, items * (ex.width / 32));
        try o.experts(false, b.act, ex.width, 0, ex.down, ex.width / 64, ex.dims / 32, b.plan, b.ymoe, ex.dims, items * (ex.dims / 32));
        try o.s.wait(sd.join);
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
        try f.tri.embed(b.p_ids, f.w.embed.w, f.w.embed.s, f.w.embed.b, b.emb, w.rows, f.c.hidden);
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
                const ssm = b.ssm + @as(u64, w.mj) * c.mamba_heads * c.mamba_head_dim * c.state * 4;
                const base = b.conv_base + @as(u64, w.mj) * 3 * c.convDim() * 2;
                try f.tri.convRows(b.proj, base, b.p_xc, m.conv_w, m.conv_b, w.rows, f.mshape(state.max_rows));
                try o.scanRows(b.proj, b.p_xc, ssm, m.a, m.d, m.dt_bias, b.sy, w.rows, c.projDim(), c.mamba_heads, c.mamba_head_dim, c.convDim(), c.groups, c.dt_min, c.dt_max);
            },
            .attention => {
                const kc = f.cache(b.k_cache, w.aj);
                const vc = f.cache(b.v_cache, w.aj);
                const qd = c.heads * c.head_dim;
                try f.tri.kvWrite(b.qkv, kc, vc, b.p_meta, w.rows, f.ashape());
                try o.torch().copyRows(b.qkv, c.qkvDim() * 2, b.q, qd * 2, qd * 2, w.rows);
                const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
                try o.prefillAttention(b.q, kc, vc, b.att, w.pos, w.rows, c.heads, c.kv_heads, scale);
            },
            .moe => {},
        }
    }

    /// Block i after its mixer: Mamba gate norm and out projection, attention's out projection, or the experts.
    pub fn chunkPost(f: *const Forward, w: *Walk, i: usize) !void {
        const c = f.c;
        const b = f.b;
        const blk = f.w.blocks[i];
        switch (blk.kind) {
            .mamba => {
                try f.tri.groupRmsnorm(b.sy, blk.mamba.gnorm, b.g, b.gxs, w.rows, c.inner(), c.groups, c.eps);
                try f.ops.prefillDense(b.g, blk.mamba.out_proj, b.delta, w.rows);
                w.delta = .dense;
                w.mj += 1;
            },
            .attention => {
                try f.ops.prefillDense(b.att, blk.attn.o, b.delta, w.rows);
                w.delta = .dense;
                w.aj += 1;
            },
            .moe => {
                try f.experts(blk.moe, w.rows, true);
                w.delta = .moe;
            },
        }
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
