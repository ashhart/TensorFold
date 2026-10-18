//! Rows through every layer in one command buffer: row kernels to 16 rows, tensor-unit matmuls for prompt chunks.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

/// The glue kernels' constants (glue.metal's s5_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, span: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, theta: f32 = 0, limit: f32 = 0, ring: u32 = 0 };
/// The sparse kernels' constants (sparse.metal's s5_sparse_args).
const SArgs = extern struct { rows: u32 = 0, pos: u32 = 0, heads: u32 = 0, dim: u32 = 0, span: u32 = 0, ring: u32 = 0, window: u32 = 0, regions: u32 = 0, groups: u32 = 0, topk: u32 = 0, first: u32 = 0, norm: u32 = 0, eps: f32 = 0, theta: f32 = 0, scale: f32 = 0 };

const Enc = struct {
    e: mtl.ComputeEncoder,
    m: *const Model,
    s: *st.Scratch,

    fn glue(x: Enc, g: Glue, args: anytype, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        x.e.setPipeline(x.m.pipe(g));
        var slot: usize = 0;
        for (bufs) |r| {
            if (r.args) x.e.setValue(args, slot) else x.e.setBuffer(r.buf, r.off, slot);
            slot += 1;
        }
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    fn project(x: Enc, w: wts.Linear, in: mtl.Buffer, in_off: usize, out: mtl.Buffer, out_off: usize, n_rows: usize) !void {
        if (n_rows > st.window_rows) if (x.m.mmFor(w)) |p| {
            // a prompt chunk: each group's row sums, then the tensor-unit product (each weight read once a chunk)
            affine_mm.rowSums(x.e, p, @intCast(w.group), bo(in, in_off), b(x.s.sums), @intCast(n_rows), @intCast(w.k));
            affine_mm.dense(x.e, p, bo(in, in_off), b(x.s.sums), mat(w), bo(out, out_off), @intCast(n_rows));
            return;
        };
        const call = try rows.call(&x.m.rows, w, n_rows, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, in_off, 0);
        x.e.setBuffer(w.w, w.w_off, 1);
        x.e.setBuffer(w.scales, w.s_off, 2);
        x.e.setBuffer(w.biases, w.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, out_off, 5);
        x.e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
    }

    fn experts(x: Enc, w: rows.Experts, in: mtl.Buffer, out: mtl.Buffer, ids: mtl.Buffer, slots: usize) !void {
        const call = try rows.callIndexed(&x.m.rows, w, slots, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, 0, 0);
        x.e.setBuffer(w.one.w, w.one.w_off, 1);
        x.e.setBuffer(w.one.scales, w.one.s_off, 2);
        x.e.setBuffer(w.one.biases, w.one.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, 0, 5);
        x.e.setBuffer(ids, 0, 6);
        x.e.dispatchGroups(mtl.Size.of(call.groups, slots, 1), mtl.Size.of(call.threads, 1, 1));
    }
};

pub const B = struct { buf: mtl.Buffer = undefined, off: usize = 0, args: bool = false };
fn b(buf: mtl.Buffer) B {
    return .{ .buf = buf };
}
fn bo(buf: mtl.Buffer, off: usize) B {
    return .{ .buf = buf, .off = off };
}
fn t(x: wts.Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}
const A = B{ .args = true };

/// A matrix as affine_mm takes it.
fn mat(w: wts.Linear) struct { w: B, s: B, b: B, n: u32, k: u32 } {
    return .{ .w = bo(w.w, w.w_off), .s = bo(w.scales, w.s_off), .b = bo(w.biases, w.b_off), .n = @intCast(w.n), .k = @intCast(w.k) };
}

/// Beyond the cache: every row's logits (s.logits_all) or the last's, and rows' fp32 residuals (`hidden`).
pub const Out = struct { all: bool = false, hidden: ?B = null, images: []const Span = &.{} };

/// Rows that replace placeholder embeddings: positions [at, at + n) take bf16 rows from `rows` ([n, hidden]).
pub const Span = struct { at: usize, n: usize, rows: B };

/// `ids` at positions cache.len.. through the model: keys and values into the cache (len unmoved), logits out.
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    const c = &m.config;
    const n = ids.len;
    const start = cache.len;
    if (n == 0 or n > s.rows or (out.all and n > st.window_rows)) return error.Step5RowsExceeded;
    if (start + n > cache.capacity) return error.Step5ContextFull;
    for (ids) |id| if (id >= c.vocab) return error.InvalidStep5Token;
    @memcpy(s.ids.slice(u32, n), ids);
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const W = m.weights;
    embed(x, n);
    for (out.images) |sp| { // image rows over the placeholders this window holds
        const lo = @max(sp.at, start);
        const hi = @min(sp.at + sp.n, start + n);
        if (lo >= hi) continue;
        const args = Args{ .rows = @intCast(hi - lo), .width = c.hidden };
        x.glue(.widen, args, &.{ bo(s.resid, (lo - start) * h * 4), bo(sp.rows.buf, sp.rows.off + (lo - sp.at) * h * 2), A }, .{ h, hi - lo, 1 }, .{ 256, 1, 1 }, false);
    }
    for (W.layers[0..c.layers], 0..) |l, i| try layer(x, cache, l, i, start, n);
    if (out.hidden) |dst| x.glue(.copy32, Args{ .rows = nr, .width = c.hidden, .stride = c.hidden }, &.{ b(s.resid), A, dst }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    const normed = Args{ .rows = nr, .width = c.hidden, .eps = c.eps };
    if (out.all) {
        x.glue(.final_norm, normed, &.{ b(s.resid), t(W.norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(W.head, s.x, 0, s.logits_all, 0, n);
    } else {
        x.glue(.final_norm, normed, &.{ bo(s.resid, (n - 1) * h * 4), t(W.norm), A, b(s.x) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(W.head, s.x, 0, cache.logits, 0, 1);
    }
    try finish(cb, x);
}

/// MTP layer `k` on `tokens` and the `hidden` rows before them: residuals into s.hid, last logits in s.logits_all.
pub fn mtp(m: *const Model, s: *st.Scratch, cache: *st.Cache, k: usize, hidden: B, tokens: []const u32, start: usize) !void {
    const c = &m.config;
    const n = tokens.len;
    if (n == 0 or n > s.rows or k >= c.mtp) return error.Step5RowsExceeded;
    if (start + n > cache.capacity) return error.Step5ContextFull;
    @memcpy(s.ids.slice(u32, n), tokens);
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const head = m.weights.mtp[k];
    embed(x, n);
    const cat = Args{ .rows = nr, .width = c.hidden, .eps = c.eps, .stride = 2 * c.hidden };
    x.glue(.norm, cat, &.{ b(s.resid), t(head.enorm), A, b(s.cat) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    x.glue(.norm, cat, &.{ hidden, t(head.hnorm), A, bo(s.cat, h * 2) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    try x.project(head.eh, s.cat, 0, s.y, 0, n);
    x.glue(.widen, Args{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    const i = c.layers + k;
    try layer(x, cache, m.weights.layers[i], i, start, n);
    x.glue(.copy32, Args{ .rows = nr, .width = c.hidden, .stride = c.hidden }, &.{ b(s.resid), A, b(s.hid) }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    x.glue(.final_norm, Args{ .rows = 1, .width = c.hidden, .eps = c.eps }, &.{ bo(s.resid, (n - 1) * h * 4), t(head.norm), A, b(s.x) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, true);
    try x.project(head.head, s.x, 0, s.logits_all, 0, 1);
    try finish(cb, x);
}

/// Embeddings of s.ids's first n into the fp32 residual.
fn embed(x: Enc, n: usize) void {
    const c = &x.m.config;
    const E = x.m.weights.embed;
    const args = Args{ .rows = @intCast(n), .width = c.hidden, .bits = E.bits, .group = @intCast(E.group) };
    x.glue(.embed, args, &.{ b(x.s.ids), bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(x.s.resid) }, .{ c.hidden, n, 1 }, .{ 256, 1, 1 }, false);
}

fn finish(cb: mtl.CommandBuffer, x: Enc) !void {
    x.e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("step5 forward: {s}", .{msg});
        return error.GpuFailed;
    }
}

/// Decoder layer `i` (its cache slot too) over rows at positions start..: attention, then the MLP or the experts.
fn layer(x: Enc, cache: *st.Cache, l: wts.Layer, i: usize, start: usize, n: usize) !void {
    const m = x.m;
    const s = x.s;
    const c = &m.config;
    const h: usize = c.hidden;
    const qw: usize = @as(usize, c.heads) * c.head_dim;
    const nr: u32 = @intCast(n);
    const normed = Args{ .rows = nr, .width = c.hidden, .eps = c.eps };
    const lay = c.layer[i];
    x.glue(.norm, normed, &.{ b(s.resid), t(l.input_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    try x.project(l.attn.q, s.x, 0, s.q, 0, n);
    try x.project(l.attn.k, s.x, 0, s.k, 0, n);
    try x.project(l.attn.v, s.x, 0, s.v, 0, n);
    try x.project(l.attn.g, s.x, 0, s.g, 0, n);
    try attention(x, s, cache, l.attn, i, start, n);
    x.glue(.gate, Args{ .rows = nr, .heads = c.heads, .dim = c.head_dim }, &.{ b(s.o), b(s.g), A }, .{ qw, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(l.attn.o, s.o, 0, s.y, 0, n);
    x.glue(.add, Args{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    x.glue(.norm, normed, &.{ b(s.resid), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    switch (l.ffn) {
        .dense => |d| {
            try mlp(x, d, s, s.y, n, c.intermediate, lay.limit);
            x.glue(.add, Args{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
        .moe => |e| {
            var r0: usize = 0;
            while (r0 < n) : (r0 += st.window_rows) { // the router's launches take 16 rows each
                const blk: u32 = @intCast(@min(st.window_rows, n - r0));
                const at = r0 * c.topk * 4;
                route.logits(x.e, m.route[0], .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk }, bo(s.x, r0 * h * 2), b(e.router), b(s.logits32), blk);
                route.select(x.e, m.route[1], b(s.logits32), t(e.bias), c.routed_scale, blk, .{ 0, c.experts }, .{
                    .pick = bo(s.pick, at),
                    .wts = bo(s.wts, at),
                    .ids = b(s.route[0]),
                    .members = b(s.route[1]),
                    .count = b(s.route[2]),
                    .mine = b(s.route[3]),
                    .theirs = b(s.route[4]),
                    .counts = b(s.route[5]),
                    .word = b(s.route[6]),
                });
            }
            const slots = n * c.topk;
            if (n > st.window_rows) {
                try expertsByExpert(x, e, n, lay.limit);
                try mlp(x, e.shared, s, s.shared, n, c.shared_intermediate, lay.shared_limit);
                x.glue(.moe_add, Args{ .rows = nr, .width = c.hidden, .heads = c.topk }, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
                return;
            }
            var gate = e.gate;
            gate.repeat = c.topk;
            var up = e.up;
            up.repeat = c.topk;
            try x.experts(gate, s.x, s.eg, s.pick, slots);
            try x.experts(up, s.x, s.eu, s.pick, slots);
            x.glue(.swiglu, Args{ .rows = @intCast(slots), .width = c.moe_intermediate, .limit = lay.limit }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_intermediate, slots, 1 }, .{ 256, 1, 1 }, false);
            try x.experts(e.down, s.ea, s.ey, s.pick, slots);
            try mlp(x, e.shared, s, s.shared, n, c.shared_intermediate, lay.shared_limit);
            x.glue(.moe_add, Args{ .rows = nr, .width = c.hidden, .heads = c.topk }, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
    }
}

/// A chunk's routed experts by expert (each expert's weights read once): sorted, gate/up, SwiGLU, down back to s.ey.
fn expertsByExpert(x: Enc, e: wts.Moe, n: usize, limit: f32) !void {
    const c = &x.m.config;
    const s = x.s;
    const slots: u32 = @intCast(n * c.topk);
    const p = x.m.mmFor(e.gate.one) orelse return error.UnsupportedStep5Experts;
    const pd = x.m.mmFor(e.down.one) orelse return error.UnsupportedStep5Experts;
    const ex: u32 = c.experts;
    x.glue(.sort, Args{ .rows = slots, .width = ex }, &.{ b(s.pick), A, b(s.offsets), b(s.order) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
    x.glue(.rows_by_order, Args{ .rows = slots, .width = c.hidden, .heads = c.topk }, &.{ b(s.x), b(s.order), A, b(s.xs) }, .{ c.hidden, slots, 1 }, .{ 256, 1, 1 }, false);
    affine_mm.rowSums(x.e, p, @intCast(e.gate.one.group), b(s.xs), b(s.sums), slots, c.hidden);
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.gate.one), b(s.offsets), b(s.eg), slots, ex);
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.up.one), b(s.offsets), b(s.eu), slots, ex);
    x.glue(.swiglu, Args{ .rows = slots, .width = c.moe_intermediate, .limit = limit }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_intermediate, slots, 1 }, .{ 256, 1, 1 }, false);
    affine_mm.rowSums(x.e, pd, @intCast(e.down.one.group), b(s.ea), b(s.sums), slots, c.moe_intermediate);
    affine_mm.gatherTo(x.e, pd, b(s.ea), b(s.sums), mat(e.down.one), b(s.offsets), b(s.ey), b(s.order), slots, ex);
}

/// gate and up from s.x, the clamped SwiGLU, down into `out`.
fn mlp(x: Enc, w: wts.Mlp, s: *st.Scratch, out: mtl.Buffer, n: usize, inter: u32, limit: f32) !void {
    try x.project(w.gate, s.x, 0, s.gate, 0, n);
    try x.project(w.up, s.x, 0, s.up, 0, n);
    x.glue(.swiglu, Args{ .rows = @intCast(n), .width = inter, .limit = limit }, &.{ b(s.gate), b(s.up), A, b(s.act) }, .{ inter, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(w.down, s.act, 0, out, 0, n);
}

/// q and k normed and rotated, k and v into the layer's cache (a ring when sliding), the indexer, attention into s.o.
fn attention(x: Enc, s: *st.Scratch, cache: *st.Cache, w: wts.Attention, i: usize, start: usize, n: usize) !void {
    const c = &x.m.config;
    const lay = c.layer[i];
    const kvw: usize = c.kvWidth();
    const nr: u32 = @intCast(n);
    const ring = cache.ring(c, i);
    const rot = Args{ .rows = nr, .dim = c.head_dim, .span = lay.span, .pos = @intCast(start), .eps = c.eps, .theta = lay.theta };
    var qa = rot;
    qa.heads = c.heads;
    qa.stride = c.heads * c.head_dim;
    x.glue(.qk, qa, &.{ b(s.q), t(w.q_norm), A, b(s.qn) }, .{ c.heads, n, 1 }, .{ 32, 1, 1 }, true);
    // keys and values land at slot (start + r) % ring in a ring, at position start + r otherwise
    const at = if (ring > 0) 0 else start * kvw * 2;
    var ka = rot;
    ka.heads = c.kv_heads;
    ka.stride = @intCast(kvw);
    ka.ring = @intCast(ring);
    x.glue(.qk, ka, &.{ b(s.k), t(w.k_norm), A, bo(cache.kv, cache.keys(i) + at) }, .{ c.kv_heads, n, 1 }, .{ 32, 1, 1 }, true);
    x.glue(.copy, Args{ .rows = nr, .width = @intCast(kvw), .stride = @intCast(kvw), .pos = @intCast(start), .ring = @intCast(ring) }, &.{ b(s.v), A, bo(cache.kv, cache.values(c, i) + at) }, .{ kvw, n, 1 }, .{ 256, 1, 1 }, false);
    if (w.indexer) |ix| try indexer(x, s, cache, ix, i, start, n);
    const args = SArgs{ .rows = nr, .pos = @intCast(start), .heads = c.heads, .dim = c.head_dim, .ring = @intCast(ring), .window = if (lay.kind == .sliding) c.window else 0, .groups = c.kv_heads, .topk = c.index_topk, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))) };
    const bufs = [_]B{ b(s.qn), bo(cache.kv, cache.keys(i)), bo(cache.kv, cache.values(c, i)), b(s.sel), A, b(s.o) };
    if (n > st.window_rows) if (x.m.attend_nax) |nax| { // a prompt chunk: a KV group's heads on the tensor units
        x.e.setPipeline(nax);
        for (bufs, 0..) |r, slot| if (r.args) x.e.setValue(args, slot) else x.e.setBuffer(r.buf, r.off, slot);
        x.e.dispatchGroups(mtl.Size.of(c.kv_heads, n, 1), mtl.Size.of(96, 1, 1));
        return;
    };
    x.glue(.attend, args, &bufs, .{ c.heads, n, 1 }, .{ 256, 1, 1 }, true);
}

/// Full layer `i`'s indexer: k and z into rings, completed regions summarized, then past topk regions: scores, top-k.
fn indexer(x: Enc, s: *st.Scratch, cache: *st.Cache, ix: wts.Indexer, i: usize, start: usize, n: usize) !void {
    const c = &x.m.config;
    const lay = c.layer[i];
    const nr: u32 = @intCast(n);
    const end = start + n;
    try x.project(ix.q, s.x, 0, s.iq, 0, n);
    try x.project(ix.k, s.x, 0, s.ik, 0, n);
    try x.project(ix.z, s.x, 0, s.iz, 0, n);
    x.glue(.index_weights, SArgs{ .rows = nr, .heads = c.index_heads, .dim = c.hidden, .groups = c.kv_heads }, &.{ b(s.x), t(ix.w), A, b(s.iw) }, .{ c.index_heads, n, 1 }, .{ 32, 1, 1 }, true);
    const rot = SArgs{ .rows = nr, .pos = @intCast(start), .dim = c.proxy, .span = c.index_span, .eps = c.eps, .theta = lay.theta };
    var qa = rot;
    qa.heads = c.index_heads;
    x.glue(.index_norm, qa, &.{ b(s.iq), t(ix.q_norm), t(ix.q_norm), A, b(s.iqn) }, .{ c.index_heads, n, 1 }, .{ 32, 1, 1 }, true);
    var ka = rot;
    ka.heads = 1;
    ka.ring = st.index_ring;
    ka.norm = 1;
    x.glue(.index_norm, ka, &.{ b(s.ik), t(ix.k_norm), t(ix.k_bias), A, bo(cache.index, cache.indexKeys(c, i)) }, .{ 1, n, 1 }, .{ 32, 1, 1 }, true);
    x.glue(.copy, Args{ .rows = nr, .width = c.proxy, .stride = c.proxy, .pos = @intCast(start), .ring = st.index_ring }, &.{ b(s.iz), A, bo(cache.index, cache.indexZ(c, i)) }, .{ c.proxy, n, 1 }, .{ 256, 1, 1 }, false);
    const first = start / c.region;
    const done = end / c.region;
    if (done > first) x.glue(.summarize, SArgs{ .dim = c.proxy, .ring = st.index_ring, .first = @intCast(first) }, &.{ bo(cache.index, cache.indexKeys(c, i)), bo(cache.index, cache.indexZ(c, i)), A, bo(cache.index, cache.summaries(i)) }, .{ done - first, 1, 1 }, .{ c.proxy, 1, 1 }, true);
    // the last row sees the most regions; below the budget every row reads all of them and nothing is chosen
    var r0: usize = 0;
    while (r0 < n) : (r0 += st.score_rows) { // the scores buffer holds score_rows rows by every region
        const blk = @min(st.score_rows, n - r0);
        const regions = (start + r0 + blk - 1) / c.region;
        if (regions <= c.index_topk) continue;
        const args = SArgs{ .rows = @intCast(blk), .pos = @intCast(start + r0), .heads = c.index_heads, .dim = c.proxy, .regions = @intCast(regions), .groups = c.kv_heads, .topk = c.index_topk };
        x.glue(.score, args, &.{ bo(s.iqn, r0 * c.index_heads * c.proxy * 2), bo(s.iw, r0 * c.index_heads * 4), bo(cache.index, cache.summaries(i)), A, b(s.scores) }, .{ (regions + 255) / 256, blk, c.kv_heads }, .{ 256, 1, 1 }, true);
        x.glue(.select, args, &.{ b(s.scores), A, bo(s.sel, r0 * c.kv_heads * c.index_topk * 4) }, .{ blk, c.kv_heads, 1 }, .{ 1024, 1, 1 }, true);
    }
}
