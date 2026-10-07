//! A window of 1-16 consecutive rows through GLM-5.3-Flash on one serial encoder, each op in the Python family's decode arithmetic.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const Kernels = @import("kernels.zig").Kernels;
const Ep = @import("ep.zig").Ep;
const moe_route = @import("../../core/moe_route.zig");
const route_shape = @import("kernels.zig").route_shape;
const hc_shape = @import("kernels.zig").hc_shape;
const hc_core = @import("../../core/hc.zig");
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
    dump_row: ?u32 = null, // a one-row trace: each capture point appends only this row
    ep: ?*Ep = null, // expert parallel: this Mac computes its routed experts' picks and swaps them with the peer's
    skip: u32 = 0, // a profile's knock-outs: launch classes left out (Class bits)
    pskip: u32 = 0, // a profile's knock-outs by launch (Part bits)
    fused_route: bool = true, // the core route (two launches); false: the Python family's cast, router and top-k
    draft_vocab: u32 = 154880, // the MTP head's tokens: the vocabulary's first this many
    m_row: u32 = 0, // the MTP head's logits row in m_logits (a rank log keeps each depth's in its own row)
};

/// Single launches (or tight groups) for a profile that times each alone.
pub const Part = struct {
    pub const names = [_][]const u8{ "hc_expand", "hc_mix", "kda_in", "kda_step", "kda_out", "mla_proj", "mla_cache", "mla_absorb", "mla_select", "mla_attn", "mla_unabs", "mla_out", "r_cast", "r_router", "r_topk", "x_locpost", "x_pack", "x_unpack", "e_gateup", "e_down", "s_gateup", "s_down", "combine", "dense", "head_qmv", "head_argmax" };
    pub fn bit(comptime name: []const u8) u32 {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return @as(u32, 1) << i;
        @compileError("no part " ++ name);
    }
};

pub fn on(x: *const Ctx, comptime part: []const u8) bool {
    return x.pskip & Part.bit(part) == 0;
}

/// Launch classes for a profile's knock-outs.
pub const Class = struct {
    pub const hc: u32 = 1 << 0;
    pub const kda: u32 = 1 << 1;
    pub const mla: u32 = 1 << 2;
    pub const dense: u32 = 1 << 3;
    pub const route: u32 = 1 << 4;
    pub const routed: u32 = 1 << 5;
    pub const shared: u32 = 1 << 6;
    pub const exchange: u32 = 1 << 7;
    pub const combine: u32 = 1 << 8;
    pub const head: u32 = 1 << 9;
    pub const mtp: u32 = 1 << 10;
    pub const ends: u32 = 1 << 11; // embedding, final mean and norm
    pub const names = [_][]const u8{ "hc", "kda", "mla", "dense", "route", "routed", "shared", "exchange", "combine", "head", "mtp", "ends" };
};

/// Append `bytes` of `src` to the capture (a u32 copy).
pub fn snap(x: *Ctx, e: mtl.ComputeEncoder, src: Ref, plane: usize) void {
    const d = x.dump orelse return;
    const row_bytes = @as(usize, x.c.hidden) * 2; // every capture point is a [rows, hidden] bf16 plane
    const from = if (x.dump_row) |r| src.at(r * row_bytes) else src;
    const bytes = if (x.dump_row != null) row_bytes else plane;
    const n: u32 = @intCast(bytes / 4);
    e.setPipeline(x.k.copy_u32);
    bind(e, 0, .{ from, d.at(x.dump_at) });
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

/// A block boundary: the pending branch into the streams, then (with `hc`) the next block's mix, split and norm (core/hc.zig).
pub fn boundary(x: *Ctx, e: mtl.ComputeEncoder, rows: u32, pending: bool, hc: ?wts.Hc, norm: ?Ref) void {
    const sc = x.sc;
    const k = x.k;
    const h = hc orelse { // the last boundary: the pending branch written into the streams alone
        if (pending and on(x, "hc_expand")) {
            e.setPipeline(k.hc_expand_10);
            bind(e, 0, .{ sc.x[x.xi], sc.branch, sc.post, sc.comb });
            e.setValue(x.c.eps, 4);
            bind(e, 5, .{ sc.x[1 - x.xi], sc.inv, sc.z });
            e.dispatchThreads(size(1024 * rows, 1, 1), size(1024, 1, 1));
        }
        if (pending) x.xi = 1 - x.xi;
        return;
    };
    if (on(x, "hc_mix")) hc_core.boundary(e, k.hc_core, hc_shape, pending, rows, x.c.eps, .{ .x_old = sc.x[x.xi], .branch = sc.branch, .post = sc.post, .comb = sc.comb, .fn_packed = h.fnp, .x_new = sc.x[1 - x.xi], .part = sc.mixes, .scale = h.scale, .base = h.base, .norm = norm.?, .normed = sc.normed });
    if (pending) x.xi = 1 - x.xi;
}

/// KDA layer `ki` on `normed`: the stacked projection (kept for a replay), the fused step into the other slot, the out-projection.
fn kda(x: *Ctx, e: mtl.ComputeEncoder, ki: usize, w: *const wts.Kda, rows: u32) void {
    const sc = x.sc;
    const L = &x.s.kda[ki];
    if (on(x, "kda_in")) qmv(x, e, x.k.qmv_kda_in, sc.normed, w.in_proj, L.proj, rows);
    if (on(x, "kda_step")) kdaStep(x, e, ki, w, L.proj, rows, sc.y);
    if (on(x, "kda_out")) qmv(x, e, x.k.qmv_kda_out, sc.y, w.o_proj, sc.branch, rows);
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
    if (on(x, "mla_proj")) {
        qmv(x, e, k.qmv_x, x_in, w.x_proj, sc.xp, rows);
        rms(x, e, sc.xp, w.q_norm, sc.qr, rows, c.q_lora, c.xProj(), c.q_lora, c.eps);
        qmv(x, e, k.qmv_qr, sc.qr, w.qr_proj, sc.qp, rows);
    }
    if (on(x, "mla_cache")) mlaCache(x, e, mi, w, x_in, sc.xp, c.xProj(), sc.iw, rows, pos);
    var dense: u32 = 0; // rows whose keys all fit index_topk attend every key (MLX's unfused attention)
    while (dense < rows and pos + dense + 1 <= c.i_topk) dense += 1;
    if (on(x, "mla_absorb")) {
        absorb(x, e, w, sc.qp, sc.ql, rows);
        if (dense > 0) scale(x, e, sc.ql, sc.qls, dense * H, RANK, RANK, RANK, 1.0 / 16.0);
    }
    if (on(x, "mla_attn")) {
        for (0..dense) |ri| {
            const r: u32 = @intCast(ri);
            const n = pos + r + 1;
            const plane = @as(usize, H) * c.i_topk * 2;
            attendDense(x, e, C.keys, sc.qls.at(@as(usize, r) * H * RANK * 2), sc.scores.at(r * plane), sc.probs.at(r * plane), sc.att.at(@as(usize, r) * H * RANK * 2), n);
        }
        if (dense < rows) attendSparse(x, e, mi, rows, pos, dense);
    }
    if (on(x, "mla_unabs")) unabsorb(x, e, w, sc.att, sc.vals, rows);
    if (on(x, "mla_out")) qmv(x, e, k.qmv_mla_out, sc.vals, w.o_proj, sc.branch, rows);
}

/// Rows' latent keys, indexer keys and gates into MLA cache `mi` at pos.., their indexer weights into `iw`, the blocks they complete.
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
    moe_route.logits(e, k.igate_logits, @import("kernels.zig").igate_shape, x_in, w.igate, C.ig.at(@as(usize, pos) * c.i_dim * 2), rows); // MLX's gemv_t sums, more threadgroups
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

/// One row's 64 heads over keys [0, n) as one 64-row matrix against the latent keys: scores, a precise softmax, values.
pub fn attendDense(x: *const Ctx, e: mtl.ComputeEncoder, keys: Ref, q: Ref, scores: Ref, probs: Ref, out: Ref, n: u32) void {
    const k = x.k;
    const H = x.c.mla_heads;
    e.setPipeline(k.latent_scores);
    bind(e, 0, .{ q, keys, scores });
    e.setValue([2]i32{ @intCast(n), @intFromBool(n < 256) }, 3); // fewer than 256 keys: the 512 dims in two halves
    e.dispatchGroups(size((n + 31) / 32, 1, 1), size(128, 1, 1));
    e.setPipeline(k.softmax);
    bind(e, 0, .{scores});
    shape(e, 1, .{n});
    bind(e, 2, .{probs});
    e.dispatchGroups(size(H, 1, 1), size(((n + 3) / 4 + 31) / 32 * 32, 1, 1));
    e.setPipeline(k.latent_values);
    bind(e, 0, .{ probs, keys, out });
    e.setValue([2]i32{ @intCast(n), if (n > 1024) 1024 else 0 }, 3); // past 1,024 keys: the first 1,024 apart
    e.dispatchGroups(size(x.c.kv_lora / 32, 1, 1), size(128, 1, 1));
}

/// Rows [first, rows) past index_topk keys: fp32 block scores, the best blocks in block order plus the tail, the sparse kernel.
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
    if (!on(x, "dense")) return;
    const c = x.c;
    const sc = x.sc;
    qmv(x, e, x.k.qmv_dense_gu, x_in, w.gate_up, sc.gu, rows);
    e.setPipeline(x.k.swiglu);
    bind(e, 0, .{ sc.gu, sc.actd });
    e.setValue(Rows{ .rows = @intCast(rows), .width = @intCast(c.dense_inter), .x_stride = @intCast(2 * c.dense_inter), .y_stride = @intCast(c.dense_inter), .eps = c.swiglu_limit }, 2);
    e.dispatchThreads(size(c.dense_inter, rows, 1), size(256, 1, 1));
    qmv(x, e, x.k.qmv_dense_down, sc.actd, w.down, sc.branch, rows);
}

/// The MoE block (shared expert, route, routed experts, combine); expert parallel: this Mac's experts, sent, the peer's received.
pub fn moe(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    std.debug.assert(rows <= st.max_rows);
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const top = c.topk;
    const s = x.skip;
    if (x.ep) |ep| if (c.byRows()) { // every expert's half on each Mac: both compute every pick, then swap their sums
        if (s & Class.exchange == 0 and on(x, "x_locpost")) ep.begin();
        if (s & Class.route == 0) route(x, e, w, x_in, rows);
        if (s & Class.routed == 0) halfExperts(x, e, w, x_in, rows);
        if (s & Class.exchange == 0 and on(x, "x_locpost")) ep.sendRows(e, sc.yp, sc.wts, rows); // with its begin
        if (s & Class.shared == 0) experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        if (s & Class.combine == 0) ep.receiveRows(e, sc.ys, sc.branch, rows);
        return;
    };
    if (x.ep) |ep| {
        if (s & Class.exchange == 0 and on(x, "x_locpost")) ep.begin();
        if (s & Class.route == 0) route(x, e, w, x_in, rows);
        if (!x.fused_route and s & Class.exchange == 0 and on(x, "x_locpost")) ep.localize(e, sc.pick, sc.uids, sc.umem, sc.ucount, rows);
        if (s & Class.routed == 0) experts(x, e, w, x_in, rows, 2, ep.group());
        if (s & Class.exchange == 0) ep.send(e, sc.ye, rows, on(x, "x_pack") and on(x, "x_locpost"), on(x, "x_locpost"));
        if (s & Class.shared == 0) experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        if (s & Class.exchange == 0 and on(x, "x_unpack")) ep.receive(e, sc.ye, rows);
    } else {
        if (s & Class.shared == 0) experts(x, e, w, x_in, rows, 1, .{ sc.none, sc.none, sc.none });
        if (s & Class.route == 0) route(x, e, w, x_in, rows);
        if (s & Class.routed == 0) experts(x, e, w, x_in, rows, 2, .{ sc.uids, sc.umem, sc.ucount });
    }
    if (s & Class.combine != 0 or !on(x, "combine")) return;
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
    if (x.fused_route) {
        if (on(x, "r_router")) moe_route.logits(e, k.route_logits, route_shape, x_in, w.router, sc.logits_r, rows);
        if (!on(x, "r_topk")) return;
        if (x.ep) |ep| if (!c.byRows()) return moe_route.select(e, k.route_select, sc.logits_r, w.bias, c.routed_scale, rows, c.own, ep.outputs(sc.pick, sc.wts));
        const rl = sc.rl;
        const T: usize = c.topk * st.max_rows * 4;
        return moe_route.select(e, k.route_select, sc.logits_r, w.bias, c.routed_scale, rows, .{ 0, c.experts }, .{ .pick = sc.pick, .wts = sc.wts, .ids = sc.uids, .members = sc.umem, .count = sc.ucount, .mine = rl, .theirs = rl.at(T), .counts = rl.at(2 * T), .word = rl.at(2 * T + 64) });
    }
    if (on(x, "r_cast")) {
        e.setPipeline(k.cast_f32);
        bind(e, 0, .{ x_in, sc.xf });
        e.setValue(rows * c.hidden, 2);
        e.dispatchThreads(size(rows * c.hidden, 1, 1), size(256, 1, 1));
    }
    if (on(x, "r_router")) {
        e.setPipeline(k.router[rows - 1]);
        bind(e, 0, .{ sc.xf, w.router, sc.logits_r });
        e.dispatchThreads(size(1024 * c.experts / 16, 1, 1), size(1024, 1, 1));
    }
    if (!on(x, "r_topk")) return;
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
    if (if (part == 1) on(x, "s_gateup") else on(x, "e_gateup")) gateUp(x, e, w, x_in, rows, part, group, zs, act);
    if (!(if (part == 1) on(x, "s_down") else on(x, "e_down"))) return;
    e.setPipeline(if (part == 1) k.moe_down_1 else k.moe_down_2);
    bind(e, 0, .{act});
    shape(e, 1, .{ rows, slots, N });
    bind(e, 2, .{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b, group[0], group[1], group[2], if (part == 1) sc.ys else sc.ye });
    e.dispatchThreads(size(32 * rows, D / 4, zs), size(32 * rows, 1, 1));
}

/// By rows: every routed pick's half (this Mac's intermediate rows), down's fp32 partials into `yp`.
fn halfExperts(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32) void {
    const c = x.c;
    const k = x.k;
    const sc = x.sc;
    const D = c.hidden;
    const N: u32 = c.inter[1] - c.inter[0];
    const zs: u32 = rows * c.topk;
    if (on(x, "e_gateup")) {
        e.setPipeline(k.moe_gateup_2h);
        bind(e, 0, .{x_in});
        shape(e, 1, .{ rows, D });
        bind(e, 2, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b, sc.uids, sc.umem, sc.ucount });
        e.setValue(c.swiglu_limit, 14);
        bind(e, 15, .{sc.act});
        e.dispatchThreads(size(32 * rows, N / 4, zs), size(32 * rows, 1, 1));
    }
    if (!on(x, "e_down")) return;
    e.setPipeline(k.moe_down_2h);
    bind(e, 0, .{sc.act});
    shape(e, 1, .{ rows, c.topk, N });
    bind(e, 2, .{ w.down.w, w.down.s, w.down.b, w.sh_down.w, w.sh_down.s, w.sh_down.b, sc.uids, sc.umem, sc.ucount, sc.yp });
    e.dispatchThreads(size(32 * rows, D / 4, zs), size(32 * rows, 1, 1));
}

fn gateUp(x: *const Ctx, e: mtl.ComputeEncoder, w: *const wts.Moe, x_in: Ref, rows: u32, part: u32, group: [3]Ref, zs: u32, act: Ref) void {
    const c = x.c;
    const k = x.k;
    const D = c.hidden;
    const N = c.moe_inter;
    e.setPipeline(if (part == 1) k.moe_gateup_1 else k.moe_gateup_2);
    bind(e, 0, .{x_in});
    shape(e, 1, .{ rows, D });
    bind(e, 2, .{ w.gate.w, w.gate.s, w.gate.b, w.up.w, w.up.s, w.up.b, w.sh_gate_up.w, w.sh_gate_up.s, w.sh_gate_up.b, group[0], group[1], group[2] });
    e.setValue(c.swiglu_limit, 14);
    bind(e, 15, .{act});
    e.dispatchThreads(size(32 * rows, N / 4, zs), size(32 * rows, 1, 1));
}

/// The backbone over the window (its tokens in `ids`) at positions pos..: final-normed rows into `hidden`.
pub fn backbone(x: *Ctx, e: mtl.ComputeEncoder, ids: Ref, rows: u32, pos: u32) void {
    const c = x.c;
    const sc = x.sc;
    const k = x.skip;
    if (k & Class.ends == 0) embed(x, e, ids, rows);
    const plane = @as(usize, rows) * c.hidden * 2;
    snap(x, e, sc.h, plane);
    var pending = false;
    var ki: usize = 0;
    var mi: usize = 0;
    for (0..c.run) |li| {
        const L = &x.w.layers[li];
        const hcs = L.hc.?;
        if (k & Class.hc == 0) boundary(x, e, rows, pending, hcs[0], L.in_norm);
        snap(x, e, sc.normed, plane);
        switch (L.attn) {
            .kda => |*a| {
                if (k & Class.kda == 0) kda(x, e, ki, a, rows);
                ki += 1;
            },
            .mla => |*a| {
                if (k & Class.mla == 0) mla(x, e, mi, a, sc.normed, rows, pos);
                mi += 1;
            },
        }
        snap(x, e, sc.branch, plane);
        if (k & Class.hc == 0) boundary(x, e, rows, true, hcs[1], L.post_norm);
        snap(x, e, sc.normed, plane);
        switch (L.mlp) {
            .dense => |*d| if (k & Class.dense == 0) denseMlp(x, e, d, sc.normed, rows),
            .moe => |*m| moe(x, e, m, sc.normed, rows),
        }
        snap(x, e, sc.branch, plane);
        pending = true;
    }
    if (k & Class.hc == 0) boundary(x, e, rows, true, null, null);
    if (k & Class.ends == 0) {
        e.setPipeline(x.k.stream_mean);
        bind(e, 0, .{ sc.x[x.xi], sc.raw });
        e.setValue([2]u32{ c.hidden, rows }, 2);
        e.dispatchThreads(size(c.hidden, rows, 1), size(256, 1, 1));
        rms(x, e, sc.raw, x.w.norm, sc.hidden, rows, c.hidden, c.hidden, c.hidden, c.eps);
    }
    snap(x, e, sc.hidden, plane);
}

/// LM head logits and argmax for `rows` rows of `in` into `logits` and `picks` (u32).
pub fn head(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, logits: Ref, picks: Ref, rows: u32) void {
    headOver(x, e, in, logits, picks, rows, x.c.vocab);
}

/// `head` over the vocabulary's first `vocab` tokens (the MTP head's draft vocabulary: the most frequent BPE merges).
pub fn headOver(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, logits: Ref, picks: Ref, rows: u32, vocab: u32) void {
    if (x.skip & Class.head != 0) return;
    var q = x.w.head;
    q.n = vocab;
    qmv(x, e, x.k.qmv_head, in, q, logits, rows);
    e.setPipeline(x.k.argmax);
    bind(e, 0, .{ logits, picks });
    e.setValue(vocab, 2);
    e.dispatchGroups(size(rows, 1, 1), size(1024, 1, 1));
}

/// A judged window's first `keep` of `rows` in every KDA layer (replayed from the round's entry state when some were rejected).
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
