//! Qwen3.8 Flash Next device sub-block kernels and batched multi-stream verifyMulti on Intel Xe2.
const std = @import("std");
const ld = @import("xpu_load.zig");
const st = @import("xpu_state.zig");
const fwd = @import("xpu_forward.zig");

const Runner = fwd.Runner;
const RankSeq = st.RankSeq;
const Seq = st.Seq;
const RankActs = st.RankActs;
const layers_n = ld.layers_n;
const dims = ld.dims;
const streams = ld.streams;
const wide = ld.wide;
const eps = ld.eps;
const vocab = ld.vocab;
const moe_w = ld.moe_w;
const head_dim = ld.head_dim;
const index_heads = ld.index_heads;
const index_dim = ld.index_dim;
const index_ratio = ld.index_ratio;
const index_idw = ld.index_idw;
const max_attn_slices = ld.max_attn_slices;
const ple_tail_n = ld.ple_tail_n;
const max_tp = ld.max_tp;
const max_verify_rows = ld.max_verify_rows;
const Dev = ld.Dev;
const Q4 = ld.Q4;
const HcW = ld.HcW;
const LayerW = ld.LayerW;
const off = ld.off;

pub fn checkIds(ids: []const u32) !void {
    for (ids) |tok| if (tok >= vocab) return error.InvalidTokenId;
}

pub fn linearBf16(d: *Dev, q: Q4, x: *anyopaque, y: *anyopaque, rows: usize) !void {
    if (q.n == 0 or rows == 0) return;
    const k_i: i32 = @intCast(q.k);
    const n_i: i32 = @intCast(q.n);
    const r_i: i32 = @intCast(rows);
    if (rows <= 16) {
        try d.launch(d.k.qmv_bf16, .{ .x = @intCast((q.n + 15) / 16), .y = @intCast((rows + 15) / 16) }, .{ q.w.?, q.s.?, q.b.?, x, y, k_i, n_i, r_i });
    } else {
        try d.launch(d.k.qmm_bf16, .{ .x = @intCast((q.n + 63) / 64), .y = @intCast((rows + 31) / 32) }, .{ q.w.?, q.s.?, q.b.?, x, y, k_i, n_i, r_i });
    }
}

pub fn linearF32(d: *Dev, q: Q4, x: *anyopaque, y: *anyopaque, rows: usize) !void {
    if (q.n == 0 or rows == 0) return;
    const k_i: i32 = @intCast(q.k);
    const n_i: i32 = @intCast(q.n);
    const r_i: i32 = @intCast(rows);
    if (rows <= 16) {
        try d.launch(d.k.qmv_f32, .{ .x = @intCast((q.n + 15) / 16), .y = @intCast((rows + 15) / 16) }, .{ q.w.?, q.s.?, q.b.?, x, y, k_i, n_i, r_i });
    } else {
        try d.launch(d.k.qmm_f32, .{ .x = @intCast((q.n + 63) / 64), .y = @intCast((rows + 31) / 32) }, .{ q.w.?, q.s.?, q.b.?, x, y, k_i, n_i, r_i });
    }
}

pub fn hcReadout(d: *Dev, a: *RankActs, hc: HcW, h_ptr: *anyopaque, br_ptr: *anyopaque, inj_in: *anyopaque, has_br: i32, rows: usize) !void {
    try d.launch(d.k.hc_writeback_norm, .{ .x = @intCast(rows) }, .{ h_ptr, br_ptr, inj_in, hc.scale, a.norm, eps, has_br });
    const ndn_i: i32 = @intCast(hc.n_down);
    const r_i: i32 = @intCast(rows);
    if (rows <= 16) {
        try d.launch(d.k.hc_down_reduce, .{ .x = @intCast(hc.n_down), .y = @intCast((rows + 15) / 16) }, .{
            a.norm, hc.down.w.?, hc.down.s.?, hc.down.b.?, a.act, a.inj, r_i, ndn_i, hc.has_inj,
        });
    } else {
        try d.launch(d.k.hc_down_dpas, .{ .x = @intCast((hc.n_down + 63) / 64), .y = @intCast((rows + 31) / 32) }, .{
            a.norm, hc.down.w.?, hc.down.s.?, hc.down.b.?, a.act, a.inj, r_i, ndn_i, hc.has_inj,
        });
    }
    try d.launch(d.k.hc_upmix, .{ .x = dims / 16, .y = @intCast((rows + 15) / 16) }, .{
        a.act, a.norm, hc.up.w.?, hc.up.s.?, hc.up.b.?, a.mixed, r_i,
    });
}

pub fn gdnRank(r: *Runner, rk: usize, lw: *const LayerW, li: usize, rs: *RankSeq, rows: usize, inplace: bool, out_f32: *anyopaque) !void {
    const d = &r.devs[rk];
    const a = &r.acts[rk];
    const shp = r.shp;
    const nk_i: i32 = @intCast(shp.nk);
    const nv_i: i32 = @intCast(shp.nv);
    const r_i: i32 = @intCast(rows);

    const proj_dst = if (inplace) a.proj else rs.saved_proj[li];
    try linearBf16(d, lw.proj, a.mixed, proj_dst, rows);

    try d.launch(d.k.gdn_front, .{ .x = @intCast(rows), .y = @intCast(shp.nk + shp.nv) }, .{
        proj_dst, rs.conv[li], lw.cw.?, lw.a_log.?, lw.dt.?, a.gq, a.gk, a.gv, a.ggb, nk_i, nv_i,
    });
    const st_out = if (inplace) rs.state[li] else rs.tmp_state[li];
    try d.launch(d.k.gdn_rows, .{ .x = 16, .y = @intCast(shp.nv) }, .{
        a.gq, a.gk, a.gv, a.ggb, rs.state[li], st_out, r_i, a.ys, nk_i, nv_i,
    });
    try d.launch(d.k.gdn_norm, .{ .x = @intCast(rows), .y = @intCast(shp.nv) }, .{
        a.ys, proj_dst, lw.norm.?, eps, a.gout, nk_i, nv_i,
    });
    if (inplace) {
        try d.launch(d.k.shift_window, .{ .x = @intCast((shp.gdn_conv + 255) / 256) }, .{
            rs.conv[li], proj_dst, @as(i64, @intCast(shp.gdn_pw)), r_i, @as(i32, 3), @as(i32, @intCast(shp.gdn_conv)),
        });
    }
    try linearF32(d, lw.out, a.gout, out_f32, rows);
}

pub fn attnRank(r: *Runner, rk: usize, lw: *const LayerW, kc: *anyopaque, vc: *anyopaque, ikc: *anyopaque, pooled: *anyopaque, pos0: usize, rows: usize, out_f32: *anyopaque) !void {
    const d = &r.devs[rk];
    const a = &r.acts[rk];
    const shp = r.shp;
    const nq_i: i32 = @intCast(shp.nq);
    const nkv_i: i32 = @intCast(shp.nkv);
    const pw_i: i32 = @intCast(shp.attn_pw);
    const pos0_i: i32 = @intCast(pos0);
    const r_i: i32 = @intCast(rows);

    try linearBf16(d, lw.proj, a.mixed, a.proj, rows);
    try d.launch(d.k.attn_prep, .{ .x = @intCast(rows), .y = @intCast(shp.nq + shp.nkv + 5) }, .{
        a.proj, pos0_i, lw.q_scale.?, lw.k_scale.?, lw.i_scale.?, r.weights[rk].inv_freq,
        a.q,    kc,     vc,           a.iq,         ikc,          eps,
        pw_i,   nq_i,   nkv_i,
    });

    const pool_groups = (pos0 + rows) / index_ratio - pos0 / index_ratio;
    if (pool_groups > 0) {
        try d.launch(d.k.qsa_pool, .{ .x = @intCast(pool_groups) }, .{
            ikc, pooled, pos0_i, lw.ik_scale.?, r.weights[rk].inv_freq, eps, r_i,
        });
    }

    const nb_max: i32 = @intCast(@max((r.max_len + index_ratio - 1) / index_ratio, 64));
    const complete_max = (pos0 + rows) / index_ratio;
    if (complete_max > ld.index_top) {
        try d.launch(d.k.qsa_scores, .{ .x = @intCast(rows), .y = @intCast((complete_max + 15) / 16) }, .{
            a.iq, pooled, pos0_i, a.qsa_sc, nb_max,
        });
    }

    try d.launch(d.k.qsa_select, .{ .x = @intCast(rows) }, .{
        a.qsa_sc, pos0_i, a.qsa_ids, a.qsa_nkr, a.qsa_spr, nb_max,
    });

    const max_nkeys = @min(pos0 + rows, 2051);
    const ns = @max((max_nkeys + 31) / 32, 1);
    const ns_i: i32 = @intCast(ns);

    try d.launch(d.k.attn_sparse, .{ .x = @intCast(ns), .y = @intCast(shp.nq), .z = @intCast(rows) }, .{
        a.q, kc, vc, a.qsa_ids, a.qsa_nkr, a.qsa_spr, a.pm, a.pl, a.po, ns_i, nq_i, nkv_i,
    });
    try d.launch(d.k.attn_merge, .{ .x = @intCast(shp.nq), .y = @intCast(rows) }, .{
        a.pm, a.pl, a.po, a.qsa_nkr, a.ao, ns_i, nq_i,
    });
    try d.launch(d.k.attn_gate, .{ .x = @intCast(rows), .y = @intCast(shp.nq) }, .{
        a.ao, a.proj, a.agated, pw_i, nq_i,
    });
    try linearF32(d, lw.out, a.agated, out_f32, rows);
}

pub fn moeRank(r: *Runner, rk: usize, lw: *const LayerW, rows: usize, out_f32: *anyopaque) !void {
    const d = &r.devs[rk];
    const a = &r.acts[rk];
    const shp = r.shp;
    const r_i: i32 = @intCast(rows);
    const first_e: i32 = @intCast(rk * shp.local_e);
    const local_e: i32 = @intCast(shp.local_e);
    const sh_w_i: i32 = @intCast(shp.sh_w);

    try d.launch(d.k.router_logits, .{ .x = (513 + 3) / 4, .y = @intCast(rows) }, .{ a.mixed, lw.router, a.router_l, r_i });
    try d.launch(d.k.route_topk, .{ .x = @intCast(rows) }, .{ a.router_l, a.picks, a.wts });
    try d.launch(d.k.moe_compact, .{ .x = 1 }, .{ a.picks, a.active, a.exp_grp, a.n_act, a.row_list, r_i, first_e, local_e });
    try d.launch(d.k.moe_routed_up, .{ .x = moe_w / 16, .y = @intCast(rows * 10) }, .{
        a.mixed,            a.active,           a.exp_grp,          a.n_act,
        lw.routed_gate.w.?, lw.routed_gate.s.?, lw.routed_gate.b.?, lw.routed_up.w.?,
        lw.routed_up.s.?,   lw.routed_up.b.?,   a.moe_act,
    });
    if (shp.sh_w > 0) {
        try d.launch(d.k.moe_shared_up, .{ .x = @intCast((shp.sh_w + 15) / 16), .y = @intCast((rows + 15) / 16) }, .{
            a.mixed,
            lw.shared_gate.w.?,
            lw.shared_gate.s.?,
            lw.shared_gate.b.?,
            lw.shared_up.w.?,
            lw.shared_up.s.?,
            lw.shared_up.b.?,
            a.sh_act,
            sh_w_i,
            r_i,
        });
    }
    try d.launch(d.k.moe_down_combine, .{ .x = dims / 16, .y = @intCast(rows) }, .{
        a.moe_act,          a.sh_act,           a.active,           a.row_list,         a.wts,
        lw.routed_down.w.?, lw.routed_down.s.?, lw.routed_down.b.?, lw.shared_down.w.?, lw.shared_down.s.?,
        lw.shared_down.b.?, out_f32,            sh_w_i,             r_i,
    });
}

pub fn pleRank(r: *Runner, rk: usize, lw: *const LayerW, rs: *RankSeq, rows: usize, inplace: bool) !void {
    const d = &r.devs[rk];
    const a = &r.acts[rk];
    const r_i: i32 = @intCast(rows);
    try linearBf16(d, lw.ple_key, a.ple_emb, a.ple_keys, rows);
    try linearBf16(d, lw.ple_val, a.ple_emb, a.ple_vals, rows);
    try d.launch(d.k.ple_gate, .{ .x = @intCast(rows) }, .{
        a.ple_keys, a.ple_vals, a.h, lw.ple_nk.?, lw.ple_nq.?, a.ple_gated, a.ple_pss, eps,
    });
    const nrow_dst = if (inplace) a.ple_nrow else rs.saved_ple_nrow;
    try d.launch(d.k.ple_conv, .{ .x = @intCast(rows), .y = 20 }, .{
        a.ple_gated, a.ple_pss, lw.ple_nc.?, rs.ple_tail, lw.ple_cw.?, a.h, nrow_dst, eps,
    });
    if (inplace) {
        try d.launch(d.k.shift_window, .{ .x = (wide + 255) / 256 }, .{
            rs.ple_tail, nrow_dst, @as(i64, @intCast(wide)), r_i, @as(i32, @intCast(ple_tail_n)), @as(i32, @intCast(wide)),
        });
    }
}

// Full 48-Layer Forward Pass (Prompt Chunks, Single Step, Verify Windows)

pub fn gdnMultiRank(r: *Runner, rk: usize, lw: *const LayerW, li: usize, seqs: []*Seq, wins: []const []const u32, total: usize, out_f32: *anyopaque) !void {
    const d = &r.devs[rk];
    const a = &r.acts[rk];
    const shp = r.shp;
    const nk_i: i32 = @intCast(shp.nk);
    const nv_i: i32 = @intCast(shp.nv);

    try linearBf16(d, lw.proj, a.mixed, a.proj, total);

    var row0: usize = 0;
    for (seqs, wins) |sq, w| {
        try checkIds(w);
        const m = w.len;
        const m_i: i32 = @intCast(m);
        const rs = &sq.ranks[rk];
        const p_s = off(a.proj, row0 * shp.gdn_pw * 2);
        try d.copy(rs.saved_proj[li], p_s, m * shp.gdn_pw * 2);
        const gq_s = off(a.gq, row0 * shp.nk * 128 * 4);
        const gk_s = off(a.gk, row0 * shp.nk * 128 * 4);
        const gv_s = off(a.gv, row0 * shp.nv * 128 * 4);
        const ggb_s = off(a.ggb, row0 * shp.nv * 2 * 4);
        const ys_s = off(a.ys, row0 * shp.nv * 128 * 4);
        const gout_s = off(a.gout, row0 * shp.gdn_vdim * 2);
        try d.launch(d.k.gdn_front, .{ .x = @intCast(m), .y = @intCast(shp.nk + shp.nv) }, .{
            p_s, rs.conv[li], lw.cw.?, lw.a_log.?, lw.dt.?, gq_s, gk_s, gv_s, ggb_s, nk_i, nv_i,
        });
        try d.launch(d.k.gdn_rows, .{ .x = 16, .y = @intCast(shp.nv) }, .{
            gq_s, gk_s, gv_s, ggb_s, rs.state[li], rs.tmp_state[li], m_i, ys_s, nk_i, nv_i,
        });
        try d.launch(d.k.gdn_norm, .{ .x = @intCast(m), .y = @intCast(shp.nv) }, .{
            ys_s, p_s, lw.norm.?, eps, gout_s, nk_i, nv_i,
        });
        row0 += m;
    }
    try linearF32(d, lw.out, a.gout, out_f32, total);
}

pub fn attnMultiRank(r: *Runner, rk: usize, lw: *const LayerW, ai: usize, seqs: []*Seq, wins: []const []const u32, total: usize, out_f32: *anyopaque) !void {
    const d = &r.devs[rk];
    const a = &r.acts[rk];
    const shp = r.shp;
    const nq_i: i32 = @intCast(shp.nq);
    const nkv_i: i32 = @intCast(shp.nkv);
    const pw_i: i32 = @intCast(shp.attn_pw);
    const nb_max: i32 = @intCast(@max((r.max_len + index_ratio - 1) / index_ratio, 64));

    try linearBf16(d, lw.proj, a.mixed, a.proj, total);

    var row0: usize = 0;
    for (seqs, wins) |sq, w| {
        const m = w.len;
        const m_i: i32 = @intCast(m);
        const pos0: usize = sq.pos;
        const pos0_i: i32 = @intCast(pos0);
        const rs = &sq.ranks[rk];
        const p_s = off(a.proj, row0 * shp.attn_pw * 2);
        const q_s = off(a.q, row0 * shp.nq * head_dim * 2);
        const iq_s = off(a.iq, row0 * index_heads * index_dim * 2);
        const sc_s = off(a.qsa_sc, row0 * @as(usize, @intCast(nb_max)) * 4);
        const ids_s = off(a.qsa_ids, row0 * index_idw * 4);
        const nkr_s = off(a.qsa_nkr, row0 * 4);
        const spr_s = off(a.qsa_spr, row0 * 4);
        const ao_s = off(a.ao, row0 * shp.nq * head_dim * 2);
        const ag_s = off(a.agated, row0 * shp.attn_odim * 2);

        try d.launch(d.k.attn_prep, .{ .x = @intCast(m), .y = @intCast(shp.nq + shp.nkv + 5) }, .{
            p_s,  pos0_i,    lw.q_scale.?, lw.k_scale.?, lw.i_scale.?, r.weights[rk].inv_freq,
            q_s,  rs.kc[ai], rs.vc[ai],    iq_s,         rs.ikc[ai],   eps,
            pw_i, nq_i,      nkv_i,
        });
        const pool_groups = (pos0 + m) / index_ratio - pos0 / index_ratio;
        if (pool_groups > 0) {
            try d.launch(d.k.qsa_pool, .{ .x = @intCast(pool_groups) }, .{
                rs.ikc[ai], rs.pooled[ai], pos0_i, lw.ik_scale.?, r.weights[rk].inv_freq, eps, m_i,
            });
        }
        const complete_max = (pos0 + m) / index_ratio;
        if (complete_max > ld.index_top) {
            try d.launch(d.k.qsa_scores, .{ .x = @intCast(m), .y = @intCast((complete_max + 15) / 16) }, .{
                iq_s, rs.pooled[ai], pos0_i, sc_s, nb_max,
            });
        }
        try d.launch(d.k.qsa_select, .{ .x = @intCast(m) }, .{ sc_s, pos0_i, ids_s, nkr_s, spr_s, nb_max });

        const max_nkeys = @min(pos0 + m, 2051);
        const ns = @max((max_nkeys + 31) / 32, 1);
        const ns_i: i32 = @intCast(ns);
        const pm_s = off(a.pm, row0 * shp.nq * max_attn_slices * 4);
        const pl_s = off(a.pl, row0 * shp.nq * max_attn_slices * 4);
        const po_s = off(a.po, row0 * shp.nq * max_attn_slices * head_dim * 4);

        try d.launch(d.k.attn_sparse, .{ .x = @intCast(ns), .y = @intCast(shp.nq), .z = @intCast(m) }, .{
            q_s, rs.kc[ai], rs.vc[ai], ids_s, nkr_s, spr_s, pm_s, pl_s, po_s, ns_i, nq_i, nkv_i,
        });
        try d.launch(d.k.attn_merge, .{ .x = @intCast(shp.nq), .y = @intCast(m) }, .{
            pm_s, pl_s, po_s, nkr_s, ao_s, ns_i, nq_i,
        });
        try d.launch(d.k.attn_gate, .{ .x = @intCast(m), .y = @intCast(shp.nq) }, .{
            ao_s, p_s, ag_s, pw_i, nq_i,
        });
        row0 += m;
    }
    try linearF32(d, lw.out, a.agated, out_f32, total);
}

pub fn verifyMulti(r: *Runner, seqs: []*Seq, wins: []const []const u32) !void {
    var total: usize = 0;
    for (seqs, wins) |sq, w| {
        if (sq.win_open) try r.keepRowsSeq(sq, sq.win_rows);
        if (w.len == 0 or w.len > max_verify_rows) return error.WindowTooWide;
        if (@as(usize, sq.pos) + w.len > sq.cap) return error.PromptTooLong;
        sq.win_pos0 = sq.pos;
        sq.win_rows = @intCast(w.len);
        sq.win_history = sq.history;
        @memcpy(sq.win_ids[0..w.len], w);
        for (w, 0..) |tok, j| r.host_ids[total + j] = @intCast(tok);
        total += w.len;
    }
    if (total == 0 or total > max_verify_rows) return error.WindowTooWide;

    var br_dst: [max_tp]*anyopaque = undefined;
    for (0..r.tp) |rk| {
        br_dst[rk] = r.acts[rk].branch;
        const d = &r.devs[rk];
        const a = &r.acts[rk];
        try d.copy(a.ids, r.host_ids.ptr, total * 4);
        const ew = r.weights[rk].embed;
        if (ew.dense) {
            try d.launch(d.k.embed_bf16, .{ .x = @intCast(total) }, .{ a.ids, ew.w, a.h, @as(i32, dims), @as(i32, streams) });
        } else {
            try d.launch(d.k.embed_q4, .{ .x = @intCast(total) }, .{ a.ids, ew.w, ew.s.?, ew.b.?, a.h, @as(i32, dims), @as(i32, streams) });
        }
    }

    // Layer 0 across all streams at once!
    {
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const lw = &r.weights[rk].layers[0];
            try hcReadout(d, a, lw.attn_hc, a.h, a.dummy, a.dummy, 0, total);
            const out_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
            try r.gdnMultiRank(rk, lw, 0, seqs, wins, total, out_dst);
        }
        try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], total * dims);
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const lw = &r.weights[rk].layers[0];
            try hcReadout(d, a, lw.mlp_hc, a.h, a.branch, a.inj, 1, total);
            const moe_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
            try r.moeRank(rk, lw, total, moe_dst);
        }
        try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], total * dims);
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            try d.launch(d.k.hc_writeback_only, .{ .x = @intCast(total), .y = 10 }, .{ a.h, a.branch, a.inj, @as(i32, 1) });
        }
    }

    // Gather Layer 1 PLE rows per stream on CPU while GPUs execute Layer 0!
    {
        var row0: usize = 0;
        for (seqs, wins) |sq, w| {
            try r.ngram.rows(&sq.history, w, r.host_ple[row0 * dims ..][0 .. w.len * dims]);
            row0 += w.len;
        }
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const lw1 = &r.weights[rk].layers[1];
            try d.copy(a.ple_emb, r.host_ple.ptr, total * dims * 2);
            try linearBf16(d, lw1.ple_key, a.ple_emb, a.ple_keys, total);
            try linearBf16(d, lw1.ple_val, a.ple_emb, a.ple_vals, total);
            try d.launch(d.k.ple_gate, .{ .x = @intCast(total) }, .{
                a.ple_keys, a.ple_vals, a.h, lw1.ple_nk.?, lw1.ple_nq.?, a.ple_gated, a.ple_pss, eps,
            });
            var r_at: usize = 0;
            for (seqs, wins) |sq, w| {
                const m = w.len;
                const rs = &sq.ranks[rk];
                try d.launch(d.k.ple_conv, .{ .x = @intCast(m), .y = 20 }, .{
                    off(a.ple_gated, r_at * wide * 2),
                    off(a.ple_pss, r_at * streams * 4),
                    lw1.ple_nc.?,
                    rs.ple_tail,
                    lw1.ple_cw.?,
                    off(a.h, r_at * wide * 2),
                    rs.saved_ple_nrow,
                    eps,
                });
                r_at += m;
            }
        }
    }

    // Layers 1..47 across all streams in ONE weight pass!
    var pending_br: i32 = 0;
    for (1..layers_n) |l| {
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const lw = &r.weights[rk].layers[l];
            try hcReadout(d, a, lw.attn_hc, a.h, a.branch, a.inj, pending_br, total);
            const out_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
            if (lw.linear) {
                try r.gdnMultiRank(rk, lw, ld.linearIndex(l), seqs, wins, total, out_dst);
            } else {
                try r.attnMultiRank(rk, lw, ld.attnIndex(l), seqs, wins, total, out_dst);
            }
        }
        try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], total * dims);
        for (0..r.tp) |rk| {
            const d = &r.devs[rk];
            const a = &r.acts[rk];
            const lw = &r.weights[rk].layers[l];
            try hcReadout(d, a, lw.mlp_hc, a.h, a.branch, a.inj, 1, total);
            const moe_dst = if (r.tp > 1) r.comm.currentSendBuf(rk) else a.branch;
            try r.moeRank(rk, lw, total, moe_dst);
        }
        try r.comm.allReduceF32(r.gpus[0..r.tp], r.lists[0..r.tp], null, br_dst[0..r.tp], total * dims);
        pending_br = 1;
    }

    // Final writeback, per-stream stream saving, and `total`-row logits head + argmax!
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        const a = &r.acts[rk];
        try d.launch(d.k.hc_writeback_only, .{ .x = @intCast(total), .y = 10 }, .{ a.h, a.branch, a.inj, @as(i32, 1) });
        var r_at: usize = 0;
        for (seqs, wins) |sq, w| {
            const m = w.len;
            try d.copy(sq.ranks[rk].verify_streams, off(a.h, r_at * wide * 2), m * wide * 2);
            try d.copy(sq.ranks[rk].last_streams, off(a.h, (r_at + m - 1) * wide * 2), wide * 2);
            r_at += m;
        }
        try hcReadout(d, a, r.weights[rk].mixer, a.h, a.dummy, a.dummy, 0, total);
        try linearBf16(d, r.weights[rk].head, a.mixed, a.shard_logits, total);
    }
    try r.syncAll();

    if (r.tp > 1) {
        const d0 = &r.devs[0];
        const dst_base = r.acts[0].logits_bf16.?;
        const vp = r.shp.vocab_per;
        for (0..total) |row_i| {
            for (0..r.tp) |src_rk| {
                const dst = off(dst_base, (row_i * vocab + src_rk * vp) * 2);
                const src = off(r.acts[src_rk].shard_logits, row_i * vp * 2);
                try d0.copy(dst, src, vp * 2);
            }
        }
    }
    const d0 = &r.devs[0];
    const a0 = &r.acts[0];
    try d0.launch(d0.k.logits_part, .{ .x = 256, .y = @intCast(total) }, .{
        a0.logits_bf16.?, @as(i32, vocab), a0.part_v.?, a0.part_i.?, a0.part_s.?,
    });
    try d0.launch(d0.k.logits_best, .{ .x = @intCast(total) }, .{
        a0.part_v.?, a0.part_i.?, a0.part_s.?, a0.out_tok.?, a0.out_best.?,
    });
    try d0.copy(r.host_tok.ptr, a0.out_tok.?, total * 4);
    try d0.sync();
    r.last_rows = @intCast(total);
    for (seqs) |sq| sq.win_open = true;
}
