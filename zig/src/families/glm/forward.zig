//! A window of 1-16 rows at consecutive positions through GLM-5.3-Flash on one serial encoder, each op in the Python
//! family's decode arithmetic: HC boundaries, KDA or MLA, dense MLP or MoE, the final norm, the LM head and argmax.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const Kernels = @import("kernels.zig").Kernels;
const Ep = @import("ep.zig").Ep;
const Ref = wts.Ref;

pub const Ctx = struct {
    k: *const Kernels,
    c: *const cfg.Config,
    w: *const wts.Weights,
    s: *st.State,
    sc: *const st.Scratch,
    xi: u1 = 0, // which stream buffer holds the streams now
    dump: ?Ref = null, // a capture: each sublayer's input and output appended here (glm_ref.py's order)
    dump_at: usize = 0,
    ep: ?*Ep = null, // expert parallel: this Mac computes its routed experts' picks and swaps them with the peer's
};

/// Append `bytes` of `src` to the capture (a u32 copy).
pub fn snap(x: *Ctx, e: mtl.ComputeEncoder, src: Ref, bytes: usize) void {
    const d = x.dump orelse return;
    const n: u32 = @intCast(bytes / 4);
    e.setPipeline(x.k.copy_u32);
    bind(e, 0, .{ src, d.at(x.dump_at) });
    e.setValue(n, 2);
    e.dispatchThreads(size(n, 1, 1), size(256, 1, 1));
    x.dump_at += bytes;
}

pub const Rows = extern struct { rows: i32, width: i32, x_stride: i32, y_stride: i32, eps: f32 };
const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32 };
const SelectArgs = extern struct { p0: u32, top: u32, width: u32, s_stride: u32, i_stride: u32 };

pub fn bind(e: mtl.ComputeEncoder, first: usize, refs: anytype) void {
    inline for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, first + i);
}

pub fn shape(e: mtl.ComputeEncoder, index: usize, dims: anytype) void {
    var v: [dims.len]i32 = undefined;
    inline for (dims, 0..) |d, i| v[i] = @intCast(d);
    e.setBytes(std.mem.asBytes(&v), index);
}

pub fn size(w: usize, h: usize, d: usize) mtl.Size {
    return mtl.Size.of(w, h, d);
}

/// x [rows, K] through a 4-bit matrix: qmv_rows, MLX's one-row qmv_fast sums for every row.
pub fn qmv(x: *const Ctx, e: mtl.ComputeEncoder, pipe: mtl.Pipeline, in: Ref, q: wts.Q4, out: Ref, rows: u32) void {
    e.setPipeline(pipe);
    bind(e, 0, .{ in, q.w, q.s, q.b, out });
    _ = x;
    e.dispatchThreads(size(32 * rows, q.n / 4, 1), size(32 * rows, 1, 1));
}

/// MLX's RMSNorm on rows of `width` (its threadgroup: 4 values a thread, whole simdgroups).
pub fn rms(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, w: Ref, out: Ref, rows: u32, width: u32, x_stride: u32, y_stride: u32, eps: f32) void {
    e.setPipeline(x.k.rms);
    bind(e, 0, .{ in, w, out });
    e.setValue(Rows{ .rows = @intCast(rows), .width = @intCast(width), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .eps = eps }, 3);
    e.dispatchGroups(size(rows, 1, 1), size(((width + 3) / 4 + 31) / 32 * 32, 1, 1));
}

pub fn scale(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, out: Ref, rows: u32, width: u32, x_stride: u32, y_stride: u32, factor: f32) void {
    e.setPipeline(x.k.scale);
    bind(e, 0, .{ in, out });
    e.setValue(Rows{ .rows = @intCast(rows), .width = @intCast(width), .x_stride = @intCast(x_stride), .y_stride = @intCast(y_stride), .eps = factor }, 2);
    e.dispatchThreads(size(width, rows, 1), size(@min(width, 256), 1, 1));
}

/// The window's tokens (`ids`, u32) as embedding rows in all four streams.
pub fn embed(x: *Ctx, e: mtl.ComputeEncoder, ids: Ref, rows: u32) void {
    const sc = x.sc;
    const D = x.c.hidden;
    embedRows(x, e, ids, sc.h, rows);
    e.setPipeline(x.k.streams);
    bind(e, 0, .{ sc.h, sc.x[0] });
    e.setValue([2]u32{ D, rows }, 2);
    e.dispatchThreads(size(D, rows, 1), size(256, 1, 1));
    x.xi = 0;
}

pub fn embedRows(x: *const Ctx, e: mtl.ComputeEncoder, ids: Ref, out: Ref, rows: u32) void {
    const q = x.w.embed;
    e.setPipeline(x.k.embed);
    bind(e, 0, .{ ids, q.w, q.s, q.b });
    shape(e, 4, .{x.c.hidden});
    bind(e, 5, .{out});
    e.dispatchThreads(size(x.c.hidden / 2, rows, 1), size(256, 1, 1));
}

/// A block boundary (model.boundary): the pending branch written back into the streams, then the next block's
/// mix, Sinkhorn split and RMSNorm into `normed`, `post` and `comb` (no `hc`: the write-back only).
pub fn boundary(x: *Ctx, e: mtl.ComputeEncoder, rows: u32, pending: bool, hc: ?wts.Hc, norm: ?Ref) void {
    const sc = x.sc;
    const k = x.k;
    e.setPipeline(if (pending and hc != null) k.hc_expand_11 else if (pending) k.hc_expand_10 else k.hc_expand_01);
    bind(e, 0, .{ sc.x[x.xi], sc.branch, sc.post, sc.comb });
    e.setValue(x.c.eps, 4);
    bind(e, 5, .{ sc.x[1 - x.xi], sc.inv, sc.z });
    e.dispatchThreads(size(1024 * rows, 1, 1), size(1024, 1, 1));
    if (pending) x.xi = 1 - x.xi;
    const h = hc orelse return;
    e.setPipeline(k.hc_mix);
    bind(e, 0, .{ sc.x[x.xi], sc.inv, h.fnp, sc.mixes });
    e.dispatchThreads(size(6 * 256, rows, 1), size(256, 1, 1));
    e.setPipeline(k.hc_split_norm);
    bind(e, 0, .{ sc.x[x.xi], sc.mixes, h.scale, h.base, norm.? });
    e.setValue(x.c.eps, 5);
    bind(e, 6, .{ sc.normed, sc.post, sc.comb });
    e.dispatchThreads(size(1024 * rows, 1, 1), size(1024, 1, 1));
}

/// KDA layer `ki` (its index among KDA layers) on `normed`: the stacked projection kept for a replay, the fused step
/// from the state at `cur` into the other slot, the out-projection into `branch`.
fn kda(x: *Ctx, e: mtl.ComputeEncoder, ki: usize, w: *const wts.Kda, rows: u32) void {
    const sc = x.sc;
    const L = &x.s.kda[ki];
    qmv(x, e, x.k.qmv_kda_in, sc.normed, w.in_proj, L.proj, rows);
    kdaStep(x, e, ki, w, L.proj, rows, sc.y);
    qmv(x, e, x.k.qmv_kda_out, sc.y, w.o_proj, sc.branch, rows);
}

/// The fused KDA step over `rows` of stacked projections `proj`: state and conv window from slot cur to the other.
pub fn kdaStep(x: *const Ctx, e: mtl.ComputeEncoder, ki: usize, w: *const wts.Kda, proj: Ref, rows: u32, y: Ref) void {
    const c = x.c;
    const L = &x.s.kda[ki];
    const cur = L.cur;
    e.setPipeline(x.k.kda_rows);
    bind(e, 0, .{proj});
    shape(e, 1, .{ rows, c.kdaProj() });
    bind(e, 2, .{ L.cs[cur], w.conv_w, w.f_b.w, w.f_b.s, w.f_b.b, w.g_b.w, w.g_b.s, w.g_b.b, w.a, w.dt_bias, L.st[cur], w.o_norm });
    e.setValue(c.lower_bound, 14);
    e.setValue(c.eps, 15);
    bind(e, 16, .{ y, L.st[1 - cur], L.cs[1 - cur] });
    e.dispatchThreads(size(32, 32, c.kda_heads), size(32, 32, 1));
}

/// MLA layer `mi` (its index among MLA caches) for rows at positions pos.. on `x_in`, into `branch`.
pub fn mla(x: *Ctx, e: mtl.ComputeEncoder, mi: usize, w: *const wts.Mla, x_in: Ref, rows: u32, pos: u32) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const C = &x.s.mla[mi];
    const H = c.mla_heads;
    const RANK = c.kv_lora;
    qmv(x, e, k.qmv_x, x_in, w.x_proj, sc.xp, rows);
    rms(x, e, sc.xp, w.q_norm, sc.qr, rows, c.q_lora, c.xProj(), c.q_lora, c.eps);
    qmv(x, e, k.qmv_qr, sc.qr, w.qr_proj, sc.qp, rows);
    mlaCache(x, e, mi, w, x_in, sc.xp, c.xProj(), sc.iw, rows, pos);
    absorb(x, e, w, sc.qp, sc.ql, rows);
    var dense: u32 = 0; // rows whose keys all fit index_topk attend every key (MLX's unfused attention)
    while (dense < rows and pos + dense + 1 <= c.i_topk) dense += 1;
    if (dense > 0) scale(x, e, sc.ql, sc.qls, dense * H, RANK, RANK, RANK, 1.0 / 16.0);
    for (0..dense) |ri| {
        const r: u32 = @intCast(ri);
        const n = pos + r + 1;
        const plane = @as(usize, H) * c.i_topk * 2;
        attendDense(x, e, C.keys, sc.qls.at(@as(usize, r) * H * RANK * 2), sc.scores.at(r * plane), sc.probs.at(r * plane), sc.att.at(@as(usize, r) * H * RANK * 2), n);
    }
    if (dense < rows) attendSparse(x, e, mi, rows, pos, dense);
    unabsorb(x, e, w, sc.att, sc.vals, rows);
    qmv(x, e, k.qmv_mla_out, sc.vals, w.o_proj, sc.branch, rows);
}

/// Rows' latent keys, indexer keys and gates into MLA cache `mi` at positions pos.., their indexer head weights into
/// `iw`, and the pooled blocks they complete (`xp`: the x_proj rows, `xp_stride` apart).
pub fn mlaCache(x: *const Ctx, e: mtl.ComputeEncoder, mi: usize, w: *const wts.Mla, x_in: Ref, xp: Ref, xp_stride: u32, iw: Ref, rows: u32, pos: u32) void {
    const c = x.c;
    const k = x.k;
    const C = &x.s.mla[mi];
    const RANK = c.kv_lora;
    rms(x, e, xp.at(c.q_lora * 2), w.kv_norm, C.keys.at(@as(usize, pos) * RANK * 2), rows, RANK, xp_stride, RANK, c.eps);
    e.setPipeline(k.layer_norm);
    bind(e, 0, .{ xp.at((c.q_lora + RANK) * 2), w.k_norm_w, w.k_norm_b, C.ik.at(@as(usize, pos) * c.i_dim * 2) });
    e.setValue(Rows{ .rows = @intCast(rows), .width = @intCast(c.i_dim), .x_stride = @intCast(xp_stride), .y_stride = @intCast(c.i_dim), .eps = 1e-6 }, 4);
    e.dispatchGroups(size(rows, 1, 1), size(32, 1, 1));
    e.setPipeline(k.gemv_t_igate);
    bind(e, 0, .{x_in});
    shape(e, 1, .{ rows, c.hidden });
    bind(e, 2, .{w.igate});
    shape(e, 3, .{ c.hidden, c.i_dim });
    bind(e, 4, .{C.ig.at(@as(usize, pos) * c.i_dim * 2)});
    e.dispatchThreads(size(4 * 64, 1, rows), size(64, 1, 1));
    scale(x, e, xp.at((c.q_lora + RANK + c.i_dim) * 2), iw, rows, c.i_heads, xp_stride, c.i_heads, 1.0 / 64.0);
    const first = pos / c.kpool;
    const last = (pos + rows) / c.kpool;
    if (last > first) {
        e.setPipeline(k.pool);
        bind(e, 0, .{ C.ik, C.ig, w.ape, C.pool });
        e.setValue([2]u32{ first, last - first }, 4);
        e.dispatchThreads(size(c.i_dim, last - first, 1), size(c.i_dim, 1, 1));
    }
}

/// Every row's 64 heads of q_nope (in `qp`, a row qrProj apart) into the latent: MLX's qvm on kv_b's key half.
pub fn absorb(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Mla, qp: Ref, ql: Ref, rows: u32) void {
    const c = x.c;
    e.setPipeline(x.k.absorb);
    bind(e, 0, .{ w.kv_b.w, w.kv_b.s, w.kv_b.b, qp, ql });
    e.setValue(c.qrProj(), 5);
    e.dispatchGroups(size(1, c.kv_lora / 64, rows * c.mla_heads), size(64, 1, 1));
}

/// Every row's 64 heads' latent outputs to values [rows, heads * v]: MLX's batched qmv_fast on kv_b's value half.
pub fn unabsorb(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Mla, att: Ref, vals: Ref, rows: u32) void {
    const c = x.c;
    e.setPipeline(x.k.unabsorb);
    bind(e, 0, .{ w.kv_b.w, w.kv_b.s, w.kv_b.b, att, vals });
    e.dispatchGroups(size(1, c.v_dim / 4, rows * c.mla_heads), size(32, 1, 1));
}

/// One query row's 64 heads over keys [0, n): MLX's fallback for 64 heads on one latent (gemv, precise softmax,
/// gemv_t), the scores and probabilities bf16 [heads, n].
fn attendDense(x: *const Ctx, e: mtl.ComputeEncoder, keys: Ref, q: Ref, scores: Ref, probs: Ref, out: Ref, n: u32) void {
    const k = x.k;
    const H = x.c.mla_heads;
    const RANK = x.c.kv_lora;
    // MLX 0.32's gemv tiling for in 512, out n: (BM 1, BN 8, TM 1) below 4, (1, 8, 4) to 32, (4, 1, 4) beyond
    const pipe, const per, const threads = if (n < 4) .{ k.gemv_scores_lt4, @as(u32, 1), @as(u32, 256) } else if (n <= 32) .{ k.gemv_scores_le32, @as(u32, 4), @as(u32, 256) } else .{ k.gemv_scores, @as(u32, 16), @as(u32, 128) };
    e.setPipeline(pipe);
    bind(e, 0, .{q});
    shape(e, 1, .{ H, RANK });
    bind(e, 2, .{keys});
    shape(e, 3, .{ n, RANK });
    bind(e, 4, .{scores});
    e.dispatchThreads(size((n + per - 1) / per * threads, 1, H), size(threads, 1, 1));
    e.setPipeline(k.softmax);
    bind(e, 0, .{scores});
    shape(e, 1, .{n});
    bind(e, 2, .{probs});
    e.dispatchGroups(size(H, 1, 1), size(((n + 3) / 4 + 31) / 32 * 32, 1, 1));
    e.setPipeline(k.gemv_t_values);
    bind(e, 0, .{probs});
    shape(e, 1, .{ H, n });
    bind(e, 2, .{keys});
    shape(e, 3, .{ n, RANK });
    bind(e, 4, .{out});
    e.dispatchThreads(size(RANK / 64 * 128, 1, H), size(128, 1, 1));
}

/// Rows [first, rows) past index_topk keys: fp32 block scores, the best blocks in block order plus the tail,
/// then the sparse kernel over those keys (Python's indexed attention).
fn attendSparse(x: *const Ctx, e: mtl.ComputeEncoder, mi: usize, rows: u32, pos: u32, first: u32) void {
    const c = x.c;
    const sc = x.sc;
    const n = rows - first;
    selectKeys(x, e, mi, sc.qp.at((@as(usize, first) * c.qrProj() + c.mla_heads * c.nope) * 2), c.qrProj(), sc.iw.at(@as(usize, first) * c.i_heads * 2), sc.sscore, sc.indices, n, pos + first);
    attendIndexed(x, e, mi, sc.ql.at(@as(usize, first) * c.mla_heads * c.kv_lora * 2), sc.indices, sc.att.at(@as(usize, first) * c.mla_heads * c.kv_lora * 2), n, pos + rows);
}

/// Key lists for `n` rows at positions p0.. past index_topk keys (indexer queries `iq` a row `q_stride` apart).
pub fn selectKeys(x: *const Ctx, e: mtl.ComputeEncoder, mi: usize, iq: Ref, q_stride: u32, iw: Ref, scores: Ref, indices: Ref, n: u32, p0: u32) void {
    const c = x.c;
    const k = x.k;
    const s_stride = x.s.cap / c.kpool + 1;
    const most = (p0 + n) / c.kpool; // the last row's blocks
    e.setPipeline(k.index_scores);
    bind(e, 0, .{ iq, iw, x.s.mla[mi].pool, scores });
    e.setValue(ScoreArgs{ .p0 = p0, .q_stride = q_stride, .w_stride = c.i_heads, .s_stride = s_stride }, 4);
    e.dispatchGroups(size((most + 7) / 8, n, 1), size(256, 1, 1));
    e.setPipeline(k.index_select);
    bind(e, 0, .{ scores, indices });
    e.setValue(SelectArgs{ .p0 = p0, .top = c.i_topk / c.kpool, .width = c.keyWidth(), .s_stride = s_stride, .i_stride = c.keyWidth() }, 2);
    e.dispatchGroups(size(n, 1, 1), size(1024, 1, 1));
}

/// The sparse kernel for `n` rows' 64 heads over their listed keys (`key_length` keys written so far).
pub fn attendIndexed(x: *const Ctx, e: mtl.ComputeEncoder, mi: usize, ql: Ref, indices: Ref, out: Ref, n: u32, key_length: u32) void {
    const c = x.c;
    e.setPipeline(x.k.sparse_attention);
    bind(e, 0, .{ ql, x.s.mla[mi].keys, indices });
    e.setValue(@as(f32, 1.0 / 16.0), 3);
    e.setValue(@as(i32, @intCast(key_length)), 4);
    bind(e, 5, .{out});
    e.dispatchThreads(size(1024, n * c.mla_heads, 1), size(1024, 1, 1));
}

fn denseMlp(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Dense, x_in: Ref, rows: u32) void {
    const c = x.c;
    const sc = x.sc;
    qmv(x, e, x.k.qmv_dense_gu, x_in, w.gate_up, sc.gu, rows);
    e.setPipeline(x.k.swiglu);
    bind(e, 0, .{ sc.gu, sc.actd });
    e.setValue(Rows{ .rows = @intCast(rows), .width = @intCast(c.dense_inter), .x_stride = @intCast(2 * c.dense_inter), .y_stride = @intCast(c.dense_inter), .eps = c.swiglu_limit }, 2);
    e.dispatchThreads(size(c.dense_inter, rows, 1), size(256, 1, 1));
    qmv(x, e, x.k.qmv_dense_down, sc.actd, w.down, sc.branch, rows);
}

/// The MoE block (moe.moe_rows): the shared expert, the router and route, routed gate/up and down, the combine.
/// Expert parallel: the route, this Mac's routed experts, their outputs sent, the shared expert, the peer's received.
pub fn moe(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    std.debug.assert(rows <= st.max_rows);
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const top = c.topk;
    if (x.ep) |ep| {
        route(x, e, w, x_in, rows);
        ep.localize(e, sc.pick, sc.uids, sc.umem, sc.ucount, rows);
        experts(x, e, w, x_in, rows, 2, ep.group());
        ep.send(e, sc.ye, rows);
        experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        ep.receive(e, sc.ye, rows);
    } else {
        experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        route(x, e, w, x_in, rows);
        experts(x, e, w, x_in, rows, 2, .{ sc.uids, sc.umem, sc.ucount });
    }
    e.setPipeline(k.moe_combine);
    bind(e, 0, .{ sc.ys, sc.ye, sc.wts });
    shape(e, 3, .{ rows, top });
    bind(e, 4, .{sc.branch});
    e.dispatchThreads(size(rows * c.hidden, 1, 1), size(256, 1, 1));
}

/// The router's fp32 logits and the route: picks, weights and the window's unique experts with their members.
fn route(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    e.setPipeline(k.cast_f32);
    bind(e, 0, .{ x_in, sc.xf });
    e.setValue(rows * c.hidden, 2);
    e.dispatchThreads(size(rows * c.hidden, 1, 1), size(256, 1, 1));
    e.setPipeline(k.router[rows - 1]);
    bind(e, 0, .{ sc.xf, w.router, sc.logits_r });
    e.dispatchThreads(size(1024 * c.experts / 16, 1, 1), size(1024, 1, 1));
    e.setPipeline(k.moe_route);
    bind(e, 0, .{sc.logits_r});
    shape(e, 1, .{ rows, c.experts });
    bind(e, 2, .{w.bias});
    e.setValue(c.routed_scale, 3);
    bind(e, 4, .{ sc.pick, sc.wts, sc.uids, sc.umem, sc.ucount });
    e.dispatchThreads(size(512, 1, 1), size(512, 1, 1));
}

/// Part 1: the shared expert into `ys`; part 2: the routed experts of `group` (unique ids, members, count) into `ye`.
fn experts(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32, part: u32, group: [3]Ref) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const D = c.hidden;
    const N = c.moe_inter;
    const zs: u32 = if (part == 1) 1 else rows * c.topk;
    const slots: u32 = if (part == 1) 1 else c.topk;
    const act = if (part == 1) sc.acts else sc.act;
    e.setPipeline(if (part == 1) k.moe_gateup_1 else k.moe_gateup_2);
    bind(e, 0, .{x_in});
    shape(e, 1, .{ rows, D });
    bind(e, 2, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b, group[0], group[1], group[2] });
    e.setValue(c.swiglu_limit, 14);
    bind(e, 15, .{act});
    e.dispatchThreads(size(32 * rows, N / 4, zs), size(32 * rows, 1, 1));
    e.setPipeline(if (part == 1) k.moe_down_1 else k.moe_down_2);
    bind(e, 0, .{act});
    shape(e, 1, .{ rows, slots, N });
    bind(e, 2, .{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b, group[0], group[1], group[2], if (part == 1) sc.ys else sc.ye });
    e.dispatchThreads(size(32 * rows, D / 4, zs), size(32 * rows, 1, 1));
}

/// The backbone over the window (its tokens in `ids`) at positions pos..: final-normed rows into `hidden`.
pub fn backbone(x: *Ctx, e: mtl.ComputeEncoder, ids: Ref, rows: u32, pos: u32) void {
    const c = x.c;
    const sc = x.sc;
    embed(x, e, ids, rows);
    const plane = @as(usize, rows) * c.hidden * 2;
    snap(x, e, sc.h, plane);
    var pending = false;
    var ki: usize = 0;
    var mi: usize = 0;
    for (0..c.run) |li| {
        const L = &x.w.layers[li];
        const hcs = L.hc.?;
        boundary(x, e, rows, pending, hcs[0], L.in_norm);
        snap(x, e, sc.normed, plane);
        switch (L.attn) {
            .kda => |*a| {
                kda(x, e, ki, a, rows);
                ki += 1;
            },
            .mla => |*a| {
                mla(x, e, mi, a, sc.normed, rows, pos);
                mi += 1;
            },
        }
        snap(x, e, sc.branch, plane);
        boundary(x, e, rows, true, hcs[1], L.post_norm);
        snap(x, e, sc.normed, plane);
        switch (L.mlp) {
            .dense => |*d| denseMlp(x, e, d, sc.normed, rows),
            .moe => |*m| moe(x, e, m, sc.normed, rows),
        }
        snap(x, e, sc.branch, plane);
        pending = true;
    }
    boundary(x, e, rows, true, null, null);
    e.setPipeline(x.k.stream_mean);
    bind(e, 0, .{ sc.x[x.xi], sc.raw });
    e.setValue([2]u32{ c.hidden, rows }, 2);
    e.dispatchThreads(size(c.hidden, rows, 1), size(256, 1, 1));
    rms(x, e, sc.raw, x.w.norm, sc.hidden, rows, c.hidden, c.hidden, c.hidden, c.eps);
    snap(x, e, sc.hidden, plane);
}

/// LM head logits and argmax for `rows` rows of `in` into `logits` and `picks` (u32).
pub fn head(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, logits: Ref, picks: Ref, rows: u32) void {
    qmv(x, e, x.k.qmv_head, in, x.w.head, logits, rows);
    e.setPipeline(x.k.argmax);
    bind(e, 0, .{ logits, picks });
    e.setValue(x.c.vocab, 2);
    e.dispatchGroups(size(rows, 1, 1), size(1024, 1, 1));
}

/// After a window's rows are judged: keep its first `keep` of `rows` in every KDA layer (a replay of the kept rows
/// from the round's entry state when some were rejected), then make the result current.
pub fn keepKda(x: *Ctx, e: mtl.ComputeEncoder, rows: u32, keep: u32) void {
    var ki: usize = 0;
    for (0..x.c.run) |li| {
        const a = switch (x.w.layers[li].attn) {
            .kda => |*a| a,
            .mla => continue,
        };
        if (keep < rows) kdaStep(x, e, ki, a, x.s.kda[ki].proj, keep, x.sc.y);
        ki += 1;
    }
}

/// Flip every KDA layer's current slot (after keepKda's work is encoded).
pub fn flipKda(x: *Ctx) void {
    for (0..x.c.countKind(.kda)) |ki| x.s.kda[ki].cur = 1 - x.s.kda[ki].cur;
}

test {
    std.testing.refAllDecls(@This());
}
