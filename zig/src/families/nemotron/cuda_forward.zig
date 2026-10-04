//! Nemotron-H's forwards on CUDA in the Python engine's kernel order: a verify window, and a prompt chunk.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const Tri = @import("cuda_triton.zig").Tri;
const Keyed = @import("cuda_triton.zig").Keyed;
const sampler = @import("cuda_sampler.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const Dump = @import("cuda_dump.zig").Dump;

const Delta = enum { none, dense, moe };

pub const Forward = struct {
    c: Config,
    w: *const weights.Weights,
    b: *const state.Buffers,
    ops: kern.Ops,
    tri: Tri,
    max_len: usize,
    nch: usize,
    keyed: ?Keyed, // the target's draw: null is torch.argmax
    dump: ?*Dump = null,

    pub fn init(c: Config, w: *const weights.Weights, b: *const state.Buffers, ops: kern.Ops, max_len: usize, nch: usize, keyed: ?Keyed) Forward {
        return .{ .c = c, .w = w, .b = b, .ops = ops, .tri = .{ .set = &ops.k.triton, .s = ops.s }, .max_len = max_len, .nch = nch, .keyed = keyed };
    }

    /// sampler.sample: rows of logits draw their next tokens into `out`, row r at position META[0] + r + 1.
    fn sample(f: *const Forward, logits: u64, rows: usize, meta: u64, out: u64) !void {
        const b = f.b;
        const t = f.ops.torch();
        const v = f.c.vocab;
        const k = f.keyed orelse return t.argmax(logits, v, v, out, rows);
        const n = sampler.count(k.k, v);
        try t.toF32(logits, b.flog, rows * v);
        try t.topk(b.flog, v, rows, n, b.vals, b.cols, b.topk);
        try f.tri.keyed(b.vals, b.cols, meta, out, b.seed, b.fp, null, 0, rows, n, k);
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
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const t = f.tri;
        const ms = f.mshape(state.max_rows);
        const at = f.ashape();
        try t.embed(b.ids, f.w.embed.w, f.w.embed.s, f.w.embed.b, b.emb, rows, c.hidden);
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
                    const raw = b.raw + mj * 2 * W * c.convDim() * 2;
                    const xc = b.xc + mj * 2 * W * c.convDim() * 2;
                    const dt = b.dt + mj * 2 * W * c.mamba_heads * 4;
                    const ssm = b.ssm + @as(u64, mj) * c.mamba_heads * c.mamba_head_dim * c.state * 4;
                    const base = b.conv_base + @as(u64, mj) * 3 * c.convDim() * 2;
                    try o.dense(b.y, b.xs, m.in_proj, b.proj, rows);
                    try t.conv(b.proj, base, raw, xc, m.conv_w, m.conv_b, b.meta, rows, ms);
                    try t.scan(b.proj, xc, dt, ssm, m.a, m.d, m.dt_bias, b.meta, b.sy, rows, c.dt_min, c.dt_max, ms);
                    try t.groupRmsnorm(b.sy, m.gnorm, b.g, b.gxs, rows, c.inner(), c.groups, c.eps);
                    try o.dense(b.g, b.gxs, m.out_proj, b.delta, rows);
                    delta = .dense;
                    mj += 1;
                },
                .attention => {
                    const a = blk.attn;
                    const kc = f.cache(b.k_cache, aj);
                    const vc = f.cache(b.v_cache, aj);
                    try o.dense(b.y, b.xs, a.qkv, b.qkv, rows);
                    try t.kvWrite(b.qkv, kc, vc, b.meta, rows, at);
                    try t.attention(b.qkv, kc, vc, b.meta, b.po, b.pm, b.pl, b.att, b.axs, rows, at);
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
        try o.copy(b.hidden, b.y, @as(usize, rows) * c.hidden * 2);
        try o.dense(b.y, b.xs, f.w.head, b.logits, rows);
        try f.sample(b.logits, rows, b.meta, b.sampled);
        if (f.dump) |d| try d.tail(f.ops, b.logits, b.sampled, rows, c.vocab);
    }

    /// Engine.moe / moe_rows: route each row's experts, group pairs by expert, then up (relu^2) and down.
    pub fn experts(f: *const Forward, m: weights.MoE, rows: usize, prompt: bool) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const ex = m.experts;
        const pairs = rows * c.slots();
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

    /// Engine.prefill_chunk: `rows` tokens in p_ids at positions pos..; commits them and samples the next token.
    pub fn chunk(f: *const Forward, rows: usize, pos: usize) !void {
        const c = f.c;
        const b = f.b;
        const o = f.ops;
        const t = f.tri;
        const ms = f.mshape(state.max_rows);
        const at = f.ashape();
        const qd = c.heads * c.head_dim;
        try t.embed(b.p_ids, f.w.embed.w, f.w.embed.s, f.w.embed.b, b.emb, rows, c.hidden);
        var x = b.emb;
        var delta: Delta = .none;
        var mj: usize = 0;
        var aj: usize = 0;
        for (f.w.blocks) |blk| {
            try f.norm(&x, delta, blk.norm, rows, false);
            switch (blk.kind) {
                .mamba => {
                    const m = blk.mamba;
                    const ssm = b.ssm + @as(u64, mj) * c.mamba_heads * c.mamba_head_dim * c.state * 4;
                    const base = b.conv_base + @as(u64, mj) * 3 * c.convDim() * 2;
                    try o.prefillDense(b.y, m.in_proj, b.proj, rows);
                    try t.convRows(b.proj, base, b.p_xc, m.conv_w, m.conv_b, rows, ms);
                    try o.scanRows(b.proj, b.p_xc, ssm, m.a, m.d, m.dt_bias, b.sy, rows, c.projDim(), c.mamba_heads, c.mamba_head_dim, c.convDim(), c.groups, c.dt_min, c.dt_max);
                    try t.groupRmsnorm(b.sy, m.gnorm, b.g, b.gxs, rows, c.inner(), c.groups, c.eps);
                    try o.prefillDense(b.g, m.out_proj, b.delta, rows);
                    delta = .dense;
                    mj += 1;
                },
                .attention => {
                    const a = blk.attn;
                    const kc = f.cache(b.k_cache, aj);
                    const vc = f.cache(b.v_cache, aj);
                    try o.prefillDense(b.y, a.qkv, b.qkv, rows);
                    try o.fill32(b.p_meta, @intCast(pos), 4);
                    try t.kvWrite(b.qkv, kc, vc, b.p_meta, rows, at);
                    try o.torch().copyRows(b.qkv, c.qkvDim() * 2, b.q, qd * 2, qd * 2, rows);
                    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.head_dim), -0.5));
                    try o.prefillAttention(b.q, kc, vc, b.att, pos, rows, c.heads, c.kv_heads, scale);
                    try o.prefillDense(b.att, a.o, b.delta, rows);
                    delta = .dense;
                    aj += 1;
                },
                .moe => {
                    try f.experts(blk.moe, rows, true);
                    delta = .moe;
                },
            }
        }
        try f.norm(&x, delta, f.w.norm_f, rows, false);
        try o.copy(b.p_hidden, b.y, @as(usize, rows) * c.hidden * 2);
        try o.prefillDense(b.y + @as(u64, rows - 1) * c.hidden * 2, f.w.head, b.p_logits, 1);
        try o.fill32(b.p_meta, @intCast(pos + rows - 1), 4); // sample_last: the chunk's last row is at pos + rows - 1
        try f.sample(b.p_logits, 1, b.p_meta, b.p_sampled);
        if (f.dump) |d| try d.tail(f.ops, b.p_logits, b.p_sampled, 1, c.vocab);
    }
};
