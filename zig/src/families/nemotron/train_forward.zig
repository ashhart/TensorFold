//! A training pass's layers over several sequences at once: each its own conv, scan and attention, the rest shared.
const std = @import("std");
const mtl = @import("metal");
const wts = @import("weights.zig");
const pre = @import("prefill.zig");
const pl = @import("prefill_launch.zig");
const layers = @import("layers.zig");
const Span = @import("living").train_mode.Span;
const Store = @import("train_prefix.zig").Store;

const At = pl.At;
const Chunk = pre.Chunk;

/// The open fact's pocket neurons scaled row by row: `rows` f32 [rows], at layer l the neurons from `from[l]` on.
pub const Scale = struct { rows: mtl.Buffer, from: []const u32 };

/// Where the spans' prompt prefixes are kept, and whether this pass keeps each span's end states there (prime).
pub const Prefix = struct { store: *const Store, prime: bool = false };

/// Layer i on s.h over every span (prefill.zig's layer): its input norm, its mixer into s.out, the residual add.
pub fn layer(x: Chunk, i: usize, spans: []const Span, scale: Scale, pf: Prefix) void {
    const p = x.p;
    const s = &p.s;
    const d = p.c.hidden;
    x.rms(s.h, pre.tensorAt(p.w.norms[i]), d, x.rows, s.x);
    switch (p.layers[i]) {
        .mamba => |m| mamba(x, m, p.w.layers[i].mamba, p.index[i], spans, pf),
        .attention => |a| attention(x, a, p.w.layers[i].attention, p.index[i], spans, pf),
        .moe => |m| moe(x, m, p.w.layers[i].moe, scale.rows, scale.from[i]),
    }
    x.add(s.h, s.out, x.rows * d, s.h);
}

/// The Mamba2 mixer (prefill_mixers.zig's) with each span's conv and SSD steps from its prefix's states, else zero.
fn mamba(x: Chunk, m: pre.Mamba, lin: wts.Mamba, index: usize, spans: []const Span, pf: Prefix) void {
    std.debug.assert(!x.carries());
    const c = x.p.c;
    const s = &x.p.s;
    const l = x.launch();
    const L = x.rows;
    const H = c.mamba_heads;
    const DH = c.mamba_head_dim;
    const G = c.groups;
    const N = c.state;
    const inner = H * DH;
    const C = c.convDim();
    const P = c.projDim();
    const taps = c.conv_kernel;
    const conv_bytes = (taps - 1) * C * 2;
    const ssm_bytes = H * DH * N * 4;
    const proj = At.of(s.proj);
    const pad = At.of(s.pad);
    const act = At.of(s.act);
    const dta = At.of(s.dta);
    const dtx = At.of(s.dtx);
    const ya = At.of(s.ya);
    const yb = At.of(s.yb);
    x.linear("in", "xsum_2688", lin.in_proj, m.in_proj, s.x, s.proj);
    const conv_in = At.of(x.pool.conv[index]).plus(x.cache.slot * conv_bytes);
    for (spans) |sp| {
        const carry = sp.skip > 0;
        l.glue("conv_pack_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ proj.plus(sp.at * P * 2), if (carry) pf.store.convAt(sp.slot, index) else conv_in }, &.{ sp.rows, C, P, inner, @intFromBool(carry) }, &.{pad}, .{ C, sp.rows + taps - 1, 1 }, .{ 256, 1, 1 });
        l.conv(pad, m.conv_w, At.of(s.conv).plus(sp.at * C * 2), sp.rows, C, taps);
        if (pf.prime) l.glue("conv_keep_bfloat16_t_int32_t_bfloat16_t", &.{pad}, &.{ sp.rows, C }, &.{pf.store.convAt(sp.slot, index)}, .{ C, taps - 1, 1 }, .{ 256, 1, 1 });
    }
    l.glue("conv_act_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ At.of(s.conv), m.conv_b }, &.{ L * C, C }, &.{act}, .{ L * C, 1, 1 }, .{ 256, 1, 1 });
    l.glue("mamba_dt_bfloat16_t_bfloat16_t_float_bfloat16_t_int32_t_float_float_float", &.{ proj, m.dt_bias, m.a, act }, &.{ L, P, inner + C, C, H, DH }, &.{ At.of(s.dt), dta, dtx }, .{ H * DH, L, 1 }, .{ 256, 1, 1 });
    var flip = false;
    for (spans) |sp| {
        // SSD steps of 256 rows, a state carrying in from the span's prefix and into its own later steps
        var old: ?At = if (sp.skip > 0) pf.store.ssmAt(sp.slot, index) else null;
        var at = sp.at;
        const end = sp.at + sp.rows;
        while (at < end) : (at += pre.ssd_step) {
            const r: usize = @min(pre.ssd_step, end - at);
            const last_step = if (pf.prime) pf.store.ssmAt(sp.slot, index) else At.of(x.pool.ssm[index]).plus(x.fresh * ssm_bytes);
            const state = if (at + r == end) last_step else At.of(if (flip) s.ssm_b else s.ssm_a);
            flip = !flip;
            const cb = At.of(s.cb);
            const sur = At.of(s.sur);
            const last = At.of(s.last);
            const y = At.of(s.y);
            const bh = At.of(s.bh);
            const dd = At.of(s.dd);
            l.gemm(false, act, act, cb, r, r, N, C, C, false, true, G, .{ N, N }, .{ at * C + inner + G * N, at * C + inner });
            l.glue("segsum_in_float_int32_t_float", &.{dta}, &.{ at, r, H }, &.{At.of(s.segin)}, .{ r, r, H }, .{ 32, 8, 1 });
            l.scan(At.of(s.segin), At.of(s.seg), H, r, r, 0);
            l.glue("ssd_decay_float_bfloat16_t_int32_t_float_float", &.{ At.of(s.seg), cb }, &.{ r, H, G }, &.{ sur, last }, .{ r, r, H }, .{ 32, 8, 1 });
            l.gemm(true, sur, dtx, y, r, DH, r, r, H * DH, false, false, H, .{ r * r, DH }, .{ 0, at * H * DH });
            l.glue("ssd_b_heads_bfloat16_t_int32_t_float", &.{act}, &.{ at, r, C, inner, H, G, N }, &.{bh}, .{ r, N, H }, .{ 32, 8, 1 });
            l.glue("ssd_dtx_decay_float_float_int32_t_float", &.{ dtx, last }, &.{ at, r, H, DH }, &.{dd}, .{ H * DH, r, 1 }, .{ 256, 1, 1 });
            l.gemm(true, dd, bh, if (old != null) At.of(s.nxt) else state, DH, N, r, H * DH, r, true, true, H, .{ DH, N * r }, .{ 0, 0 });
            var e = y;
            var prev = y;
            if (old) |o| {
                e = At.of(s.e);
                prev = At.of(s.prev);
                l.scan(dta, At.of(s.cs), 1, r, H, at * H);
                l.glue("exp_f32_float_int32_t_float", &.{At.of(s.cs)}, &.{r * H}, &.{e}, .{ r * H, 1, 1 }, .{ 256, 1, 1 });
                l.glue("ssd_c_f32_bfloat16_t_int32_t_float", &.{act}, &.{ at, r, C, inner + G * N, G * N }, &.{At.of(s.cf)}, .{ G * N, r, 1 }, .{ 256, 1, 1 });
                l.gemv(o, At.of(s.cf), prev, r, G, H, DH, N);
                l.glue("ssd_state_carry_float_float_float_int32_t_float", &.{ At.of(s.nxt), e, o }, &.{ r, H, DH * N }, &.{state}, .{ H * DH * N, 1, 1 }, .{ 256, 1, 1 });
            }
            l.glue("ssd_y_out_float_float_float_bfloat16_t_int32_t_bfloat16_t", &.{ y, e, prev, proj }, &.{ at, r, H, DH, @intFromBool(old != null), L, 0 }, &.{ya}, .{ inner, r, 1 }, .{ 256, 1, 1 });
            old = state;
        }
    }
    l.glue("mamba_skip_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ ya, act, m.d }, &.{ L, C, H, DH }, &.{yb}, .{ inner, L, 1 }, .{ 256, 1, 1 });
    l.glue("mamba_gate_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ proj, yb }, &.{ L, inner, P }, &.{ya}, .{ inner, L, 1 }, .{ 256, 1, 1 });
    x.rms(s.ya, At.of(x.p.ones), inner / G, L * G, s.yb);
    l.glue("scale_cols_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ yb, pre.tensorAt(lin.norm) }, &.{ L * inner, inner }, &.{ya}, .{ L * inner, 1, 1 }, .{ 256, 1, 1 });
    x.linear("out", "xsum_4096", lin.out_proj, m.out_proj, s.ya, s.out);
    if (lin.slide) |r| layers.rest(x.f, x.e, r, s.ya, s.out, L, inner);
    if (lin.adapter) |ad| layers.adapt(x.f, x.e, ad, s.ya, s.out, L);
}

/// Causal attention (prefill_mixers.zig's) of each span's rows on its prefix's keys and its own.
fn attention(x: Chunk, a: pre.Attention, lin: wts.Attention, index: usize, spans: []const Span, pf: Prefix) void {
    std.debug.assert(!x.carries());
    const c = x.p.c;
    const s = &x.p.s;
    const l = x.launch();
    const e = x.e;
    const L = x.rows;
    const D = c.head_dim;
    const row = c.heads * D;
    const kv_row = c.kv_heads * D;
    const kv = x.cache.kv[index];
    const cap = kv.capacity;
    const qkv = At.of(s.qkv);
    const stacked = L <= pre.lane_rows;
    const q_row = if (stacked) c.qkvDim() else row;
    if (stacked) {
        // lane_qmm's q, k and v columns stacked in one call: a column's bits follow only K and its K slices
        x.f.xsum(x.e, "xsum_2688", c.hidden, s.x, 0, s.xs, L);
        x.f.coop(x.e, "qkv", lin.qkv, s.x, 0, s.xs, s.qkv, L);
    } else {
        l.qmm(At.of(s.x), a.q, qkv, At.of(s.parts), L, row, c.hidden);
        l.qmm(At.of(s.x), a.k, qkv.plus(L * row * 2), At.of(s.parts), L, kv_row, c.hidden);
        l.qmm(At.of(s.x), a.v, qkv.plus(L * (row + kv_row) * 2), At.of(s.parts), L, kv_row, c.hidden);
    }
    const kvh = c.kv_heads * cap * D;
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(D), -0.5));
    for (spans) |sp| {
        // the prefix's keys and values first, then the span's own after them
        if (sp.skip > 0) for ([2]At{ At.of(kv.k), At.of(kv.v) }, 0..) |dst, v| {
            const src = pf.store.keysAt(sp.slot, index, v == 1);
            l.glue("kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ src, src }, &.{ sp.skip, c.kv_heads, D, cap, 0, cap, 0 }, &.{dst}, .{ D, sp.skip, c.kv_heads }, .{ 128, 2, 1 });
        };
        if (stacked) {
            e.pipe(x.f.k.get("tf_kv_write"));
            e.buf(s.qkv, sp.at * q_row * 2, 0);
            e.buf(kv.k, 0, 1);
            e.buf(kv.v, 0, 2);
            e.bytes([4]u32{ @intCast(c.qkvDim()), @intCast(row), @intCast(cap), @intCast(sp.skip) }, 3);
            e.run(.{ D, c.kv_heads, sp.rows }, .{ D, 1, 1 });
        } else {
            const k_at = qkv.plus((L * row + sp.at * kv_row) * 2);
            const v_at = qkv.plus((L * (row + kv_row) + sp.at * kv_row) * 2);
            for ([2]At{ k_at, v_at }, [2]At{ At.of(kv.k), At.of(kv.v) }) |src, dst| {
                l.glue("kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t", &.{ src, src }, &.{ sp.rows, c.kv_heads, D, cap, sp.skip, cap, 0 }, &.{dst}, .{ D, sp.rows, c.kv_heads }, .{ 128, 2, 1 });
            }
        }
        if (pf.prime) for (0..2) |v| {
            // the span's own keys and values, row by row, as a later pass's prefix
            const src = if (stacked) qkv.plus((sp.at * q_row + row + v * kv_row) * 2) else qkv.plus((L * (row + v * kv_row) + sp.at * kv_row) * 2);
            copyCols(x, src, if (stacked) q_row else kv_row, pf.store.keysAt(sp.slot, index, v == 1), kv_row, sp.rows);
        };
        const keys = sp.skip + sp.rows;
        const strides = [12]usize{ sp.rows * q_row, D, q_row, kvh, cap * D, D, kvh, cap * D, D, sp.rows * row, D, row };
        l.attention(qkv.plus(sp.at * q_row * 2), At.of(kv.k), At.of(kv.v), At.of(s.ya).plus(sp.at * row * 2), sp.rows, keys, c.heads, c.kv_heads, strides, scale);
    }
    x.linear("out", "xsum_4096", lin.o_proj, a.o, s.ya, s.out);
    if (lin.slide) |r| layers.rest(x.f, x.e, r, s.ya, s.out, L, row);
    if (lin.adapter) |ad| layers.adapt(x.f, x.e, ad, s.ya, s.out, L);
}

/// The routed experts plus the shared expert (prefill_mixers.zig's), the pocket's open neurons scaled row by row.
fn moe(x: Chunk, m: pre.Moe, w: wts.Moe, scale: mtl.Buffer, from: u32) void {
    const c = x.p.c;
    const s = &x.p.s;
    const l = x.launch();
    const e = x.e;
    const L = x.rows;
    const D = c.hidden;
    const k = c.top_k;
    const E = c.experts;
    const W = c.expert_width;
    const n = L * k;
    const order = At.of(s.order);
    l.matmulNt(At.of(s.x), pre.tensorAt(w.gate), At.of(s.parts), At.of(s.logits), L, E, D);
    e.pipe(x.p.k.get("tf_route_topk"));
    e.buf(s.logits, 0, 0);
    e.buf(w.gate_bias.buffer, w.gate_bias.offset, 1);
    e.bytes([2]f32{ 1e-20, c.routed_scaling }, 2);
    e.bytes([2]i32{ @intCast(E), @intCast(k) }, 3);
    e.buf(s.ids, 0, 4);
    e.buf(s.wt, 0, 5);
    e.run(.{ 128, L, 1 }, .{ 128, 1, 1 });
    const blocks = (n + 255) / 256;
    l.glue("sort_count_uint32_t_int32_t_int32_t", &.{At.of(s.ids)}, &.{ n, E }, &.{At.of(s.counts)}, .{ blocks * 256, 1, 1 }, .{ 256, 1, 1 });
    l.glue("sort_starts_int32_t_int32_t_int32_t_int32_t", &.{At.of(s.counts)}, &.{ n, E }, &.{ At.of(s.starts), At.of(s.offsets) }, .{ E, 1, 1 }, .{ E, 1, 1 });
    l.glue("sort_place_uint32_t_int32_t_int32_t_uint32_t_uint32_t", &.{ At.of(s.ids), At.of(s.starts) }, &.{ n, E }, &.{ order, At.of(s.sorted) }, .{ blocks * 256, 1, 1 }, .{ 256, 1, 1 });
    const fc1 = [3]At{ pre.tensorAt(w.fc1[0]), pre.tensorAt(w.fc1[1]), pre.tensorAt(w.fc1[2]) };
    const fc2 = [3]At{ pre.tensorAt(w.fc2[0]), pre.tensorAt(w.fc2[1]), pre.tensorAt(w.fc2[2]) };
    if (n >= 16 and n / E >= 4) {
        l.glue("rows_take_bfloat16_t_uint32_t_int32_t_bfloat16_t", &.{ At.of(s.x), order }, &.{ n, D, k }, &.{At.of(s.xr)}, .{ D, n, 1 }, .{ 256, 1, 1 });
        l.gatherQmm(At.of(s.xr), fc1, At.of(s.offsets), At.of(s.y1), n, W, D, E);
        x.relu2(s.y1, n * W, s.y1r);
        l.gatherQmm(At.of(s.y1r), fc2, At.of(s.offsets), At.of(s.y2), n, D, W, E);
    } else {
        l.glue("rows_of_uint32_t_int32_t_uint32_t", &.{order}, &.{ n, k }, &.{At.of(s.xids)}, .{ n, 1, 1 }, .{ 256, 1, 1 });
        l.glue("rows_of_uint32_t_int32_t_uint32_t", &.{order}, &.{ n, 0 }, &.{At.of(s.rowids)}, .{ n, 1, 1 }, .{ 256, 1, 1 });
        gatherQmv(x, fc1, s.x, s.xids, D, W, n, s.y1);
        x.relu2(s.y1, n * W, s.y1r);
        gatherQmv(x, fc2, s.y1r, s.rowids, W, D, n, s.y2);
    }
    l.glue("sort_inverse_uint32_t_int32_t_uint32_t", &.{order}, &.{n}, &.{At.of(s.inv)}, .{ n, 1, 1 }, .{ 256, 1, 1 });
    l.glue("rows_take_bfloat16_t_uint32_t_int32_t_bfloat16_t", &.{ At.of(s.y2), At.of(s.inv) }, &.{ n, D, 1 }, &.{At.of(s.xr)}, .{ D, n, 1 }, .{ 256, 1, 1 });
    e.pipe(x.p.k.get("tf_moe_combine_rows"));
    e.buf(s.xr, 0, 0);
    e.buf(s.wt, 0, 1);
    e.bytes([2]i32{ @intCast(k), @intCast(D) }, 2);
    e.buf(s.out, 0, 3);
    e.run(.{ D, L, 1 }, .{ 256, 1, 1 });
    x.linear("up", "xsum_2688", w.shared_up, m.up, s.x, s.up);
    x.relu2(s.up, L * c.shared_width, s.upr);
    x.linear("down", "xsum_3712", w.shared_down, m.down, s.upr, s.sh);
    if (w.slide) |r| layers.rest(x.f, e, r, s.upr, s.sh, L, c.shared_width);
    if (w.adapter) |ad| layers.adapt(x.f, e, ad, s.upr, s.sh, L);
    if (w.pocket) |p| if (p.used > 0) pocket(x, p, s.x, s.sh, L, scale, from);
    x.add(s.out, s.sh, L * D, s.out);
}

/// Each sorted pair's row of `rows` (by `xids`) times its expert's 4-bit W^T (prefill_mixers.zig's), into y [n, N].
fn gatherQmv(x: Chunk, fc: [3]At, rows: mtl.Buffer, xids: mtl.Buffer, K: usize, N: usize, n: usize, y: mtl.Buffer) void {
    const e = x.e;
    e.pipe(x.p.k.get("tf_gather_qmv_b4_g64"));
    for (fc, 0..) |t, i| e.buf(t.b, t.off, i);
    e.buf(rows, 0, 3);
    e.buf(x.p.s.sorted, 0, 4);
    e.buf(xids, 0, 5);
    e.bytes([2]i32{ @intCast(K), @intCast(N) }, 6);
    e.buf(y, 0, 7);
    e.run(.{ 32, N / 8 * 2, n }, .{ 32, 2, 1 });
}

/// `rows` rows of `width` bf16 values from `src` (rows `src_stride` apart) to `dst` (rows `dst_stride` apart).
fn copyCols(x: Chunk, src: At, src_stride: usize, dst: At, dst_stride: usize, rows: usize) void {
    const e = x.e;
    e.pipe(x.f.k.get("tf_train_copy_cols"));
    e.buf(src.b, src.off, 0);
    e.buf(dst.b, dst.off, 1);
    e.bytes([4]u32{ @intCast(src_stride), @intCast(dst_stride), @intCast(dst_stride), 0 }, 2);
    e.run(.{ dst_stride, rows, 1 }, .{ 256, 1, 1 });
}

/// y [rows, D] += relu(x up^T)^2 down over a pocket's neurons (layers.zig's), from `from` on times the row's scale.
fn pocket(x: Chunk, p: wts.Pocket, in: mtl.Buffer, y: mtl.Buffer, rows: usize, scale: mtl.Buffer, from: u32) void {
    const e = x.e;
    const k = x.f.k;
    const d: u32 = @intCast(x.p.c.hidden);
    const n: u32 = @intCast(p.used);
    e.pipe(k.get("tf_train_lora_in"));
    e.buf(in, 0, 0);
    e.buf(p.up, 0, 1);
    e.buf(p.z, 0, 2);
    e.bytes([4]u32{ @intCast(rows), d, n, 0 }, 3);
    e.buf(p.zn, 0, 4);
    e.run(.{ 32 * p.used, rows, 1 }, .{ 256, 1, 1 });
    e.pipe(k.get("tf_pocket_relu2_rows"));
    e.buf(p.z, 0, 0);
    e.bytes([4]u32{ @as(u32, @intCast(rows)) * n, n, from, 0 }, 1);
    e.buf(scale, 0, 2);
    e.run(.{ rows * p.used, 1, 1 }, .{ 256, 1, 1 });
    e.pipe(k.get("tf_train_lora_out"));
    e.buf(p.z, 0, 0);
    e.buf(p.down, 0, 1);
    e.buf(y, 0, 2);
    e.bytes([4]u32{ @intCast(rows), d, n, 0 }, 3);
    e.bytes(@as(f32, 1), 4);
    e.run(.{ x.p.c.hidden, rows, 1 }, .{ 256, 1, 1 });
}
