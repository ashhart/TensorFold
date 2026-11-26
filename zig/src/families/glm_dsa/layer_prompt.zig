//! A layer for a prompt window's rows at once: projections and experts on the tensor units, MLA 16 rows at a time.
const mtl = @import("metal");
const ex = @import("../../core/exlr8.zig");
const op = @import("../../core/exlr8_projection.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const ly = @import("layer.zig");
const wt = @import("weights.zig");
const pp = ly.pp;
const pack = ex.pack;
const Ref = kn.Ref;
const L_ = ly.Layer;

fn ref(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

fn at(b: mtl.Buffer, row: u32, width: usize) Ref {
    return .{ .buf = b, .off = @as(usize, row) * width * 4 };
}

fn encodeDense(K: *const ly.Kernels, L: *const L_, e: mtl.ComputeEncoder, g: gpu.Groups, x: Ref, y: Ref, rows: u32, x_stride: usize, y_stride: usize) !void {
    if (L.k.fast_dense) {
        try pp.encodeDenseFast(&K.pp, e, x, L.k.dense64(g), L.k.dense(g), y, rows, x_stride, y_stride, true);
    } else {
        try pp.encodeDense(&K.pp, e, x, L.k.dense(g), y, rows, x_stride, y_stride, true);
    }
}

/// y = x times the linear over the window's rows: row-stored on the dense kernel, trellis rotated into `rot` first.
pub fn linear(K: *const ly.Kernels, L: *const L_, e: mtl.ComputeEncoder, l: pack.Linear, g: gpu.Groups, x: Ref, x_stride: usize, rot: Ref, y: Ref, y_stride: usize, rows: u32) !void {
    switch (l) {
        .rows => try encodeDense(K, L, e, g, x, y, rows, x_stride, y_stride),
        .trellis => |t| {
            try op.encodeRotate(&K.op, e, x, L.k.signs, L.k.halves, rot, .{ .rows = rows, .x_stride = @intCast(x_stride), .x_f32 = 1, .r_stride = t.in, .sign = t.sign, .scale = t.scale }, t.in / 128, 1, null);
            e.barrier();
            try pp.encode(&K.pp, e, rot, L.k.trellis(t), y, rows, t.in, y_stride, true, null);
        },
    }
}

/// Attention for the window's normed rows `x` (positions in `table`) into `out`; full layers list keys in q.keys.
pub fn attention(K: *const ly.Kernels, L: *const L_, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, pos0: u32, table: mtl.Buffer, x: Ref, out: Ref) !void {
    const nh = c.localHeads();
    const a = L.w.attn;
    const hid = c.hidden;
    const t = ref(table);
    try encodeDense(K, L, e, L.groups[L_.x_g], x, ref(q.proj), rows, hid, wt.proj_width);
    e.barrier();
    const ma: kn.MlaArgs = .{ .rows = rows, .heads = nh, .slot_keys = s.context, .proj_stride = wt.proj_width, .q_stride = nh * c.qHead(), .kvb = a.kvb_w, .kvb_scale = a.kvb_s };
    var mp = ma;
    mp.q_stride = c.q_lora;
    const ia: kn.IndexArgs = .{ .rows = rows, .slot_keys = s.context, .proj_stride = wt.proj_width, .k_col = wt.wk_col, .w_col = wt.wp_col, .q_stride = c.i_heads * c.i_dim, .score_stride = s.context, .pad0 = 1, .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_dim))), .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_heads))) };
    const need_idx = L.w.full and (pos0 + rows - 1 >= c.i_topk);
    if (L.w.full) {
        kn.prepFull(&K.k, e, ref(q.proj), L.k.at(a.q_norm), L.k.at(a.kv_norm), ref(K.inv), t, mp, ref(q.q_resid), ref(L.cache), L.k.at(a.k_norm), L.k.at(a.k_bias), ref(L.index_cache.?), null);
        if (need_idx and a.wp != null) try encodeDense(K, L, e, L.groups[L_.wp_g], x, ref(q.proj), rows, hid, wt.proj_width);
    } else {
        kn.prep(&K.k, e, ref(q.proj), L.k.at(a.q_norm), L.k.at(a.kv_norm), ref(K.inv), t, mp, ref(q.q_resid), ref(L.cache));
    }
    e.barrier();
    if (need_idx) {
        try encodeDense(K, L, e, L.groups[L_.wq_b_g], ref(q.q_resid), ref(q.qidx), rows, c.q_lora, c.i_heads * c.i_dim);
        e.barrier();
        kn.indexScoresBlock(&K.k, e, ref(q.qidx), ref(q.proj), ref(L.index_cache.?), t, ia, ref(q.score), ref(K.inv), pos0 + rows - 1, null);
    }
    try linear(K, L, e, a.q_b, L.groups[L_.q_b_g], ref(q.q_resid), c.q_lora, ref(q.rot), ref(q.qq), nh * c.qHead(), rows);
    e.barrier();
    if (need_idx) kn.indexTopkOnly(&K.k, e, t, ia, ref(q.score), ref(q.keys), null);
    kn.absorbPrompt(&K.k, e, ref(q.qq), L.k.bytes, t, ma, ref(q.qlat), ref(q.qrope), ref(K.inv));
    e.barrier();
    if (rows >= 32 and nh >= 8 and (nh % 8 == 0)) {
        kn.attendFlash(&K.k, e, ref(q.qlat), ref(q.qrope), t, ref(q.keys), ref(L.cache), ma, ref(q.olat));
        e.barrier();
    } else {
        const AR = gpu.prompt_attend_rows;
        var k: u32 = 0;
        while (k < rows) : (k += AR) {
            var mk = ma;
            mk.rows = @min(AR, rows - k);
            const tk: Ref = .{ .buf = table, .off = @as(usize, k) * @sizeOf(kn.Row) };
            kn.attend(&K.k, e, at(q.qlat, k, nh * c.kv_lora), at(q.qrope, k, nh * c.rope), tk, ref(q.keys), ref(L.cache), mk, ref(q.partial), at(q.olat, k, nh * c.kv_lora));
            e.barrier();
        }
    }
    try encodeDense(K, L, e, L.groups[L_.uv_g], ref(q.olat), ref(q.o), rows, nh * c.kv_lora, nh * c.v_dim);
    e.barrier();
    try linear(K, L, e, a.o, L.groups[L_.o_g], ref(q.o), nh * c.v_dim, ref(q.rot), out, hid, rows);
}

/// The dense MLP of the window's normed rows `x` into `out` (this Mac's gate and up rows, its down inputs).
pub fn denseMlp(K: *const ly.Kernels, L: *const L_, q: *const gpu.Prompt, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, out: Ref) !void {
    const m = L.w.mlp.dense;
    const n = c.localDense();
    try linear(K, L, e, m.gate, L.groups[L_.dense_gate_g], x, c.hidden, ref(q.rot), ref(q.gu), 2 * n, rows);
    if (m.gate == .trellis or m.up == .trellis) e.barrier();
    try linear(K, L, e, m.up, L.groups[L_.dense_up_g], x, c.hidden, ref(q.rot), .{ .buf = q.gu, .off = n * 4 }, 2 * n, rows);
    e.barrier();
    kn.siluMul(&K.k, e, ref(q.gu), .{ .width = n, .x_stride = 2 * n, .y_stride = n }, ref(q.act), rows);
    e.barrier();
    try linear(K, L, e, m.down, L.groups[L_.dense_down_g], ref(q.act), n, ref(q.rot), out, c.hidden, rows);
}

/// The router's logits for the window's rows at once, then single-launch top-k and weights into q.pick and q.wts8.
pub fn route(K: *const ly.Kernels, L: *const L_, s: *const gpu.Scratch, q: *const gpu.Prompt, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref) !void {
    _ = s;
    const m = L.w.mlp.moe;
    if (L.k.fast_dense and L.fp8_moe != null) {
        try pp.encodeDenseBf16Fast(&K.pp, e, x, L.k.dense64(L.groups[L_.router_g]), L.k.dense(L.groups[L_.router_g]), ref(q.logits), rows, c.hidden, c.experts, true);
    } else {
        try encodeDense(K, L, e, L.groups[L_.router_g], x, ref(q.logits), rows, c.hidden, c.experts);
    }
    e.barrier();
    kn.routePrompt(&K.k, e, ref(q.logits), L.k.at(m.bias), .{ .rows = rows, .experts = c.experts, .topk = c.topk, .scale = c.routed_scale }, ref(q.pick), ref(q.wts8));
    e.barrier();
}

/// The window's experts of normed rows `x` into `out` from q's picks (EXLR8 trellis or native FP8 MMA).
pub fn experts(K: *const ly.Kernels, L: *const L_, q: *const gpu.Prompt, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, out: Ref) !void {
    const hid = c.hidden;
    const loc = c.localHalves() * 128;
    if (L.fp8_moe) |fm| {
        kn.fp8MoePrompt(
            &K.k,
            e,
            ref(q.pick),
            ref(q.wts8),
            x,
            fm.gate_w,
            fm.gate_s,
            fm.up_w,
            fm.up_s,
            fm.down_w,
            fm.down_s,
            .{
                .rows = rows,
                .loc = loc,
                .hidden = hid,
                .topk = c.topk,
                .slots = gpu.slots,
                .experts = c.experts,
                .loc_blocks = c.localHalves(),
                .hid_blocks = hid / 128,
            },
            ref(q.jobs),
            ref(q.members),
            ref(q.wts),
            ref(q.dispatch),
            ref(q.act),
            ref(q.z),
            out,
        );
        return;
    }
    const m = L.w.mlp.moe;
    kn.moeJobsPrompt(&K.k, e, ref(q.pick), ref(q.wts8), .{ .buf = L.desc.? }, .{ .rows = rows, .halves = c.localHalves(), .topk = c.topk, .slots = gpu.slots, .gu_cap = q.gu_cap, .experts = c.experts }, ref(q.jobs), ref(q.members), ref(q.wts), ref(q.sv), ref(q.suh), ref(q.dispatch));
    try op.encodeRotate(&K.op, e, x, L.k.signs, L.k.halves, ref(q.rot), .{ .rows = rows, .x_stride = hid, .x_f32 = 1, .r_stride = hid, .sign = m.signs }, hid / 128, 1, null);
    e.barrier();
    const t: op.Tables = .{ .weights = L.k.bytes, .bases = L.k.bases, .halves = L.k.halves, .signs = L.k.signs, .jobs = q.jobs, .count = 0 };
    pp.encodeIndirect(&K.pp, e, ref(q.rot), t, ref(q.gu), hid, 2 * loc, true, ref(q.members), ref(q.dispatch));
    e.barrier();
    kn.siluMul(&K.k, e, ref(q.gu), .{ .width = loc, .x_stride = 2 * loc, .y_stride = loc }, ref(q.act), rows * gpu.slots);
    e.barrier();
    try op.encodeRotate(&K.op, e, ref(q.act), L.k.signs, L.k.halves, ref(q.arot), .{ .rows = rows, .x_stride = gpu.slots * loc, .x_f32 = 1, .r_stride = gpu.slots * loc, .slot_cols = loc, .scales = 1, .row_slots = gpu.slots }, loc / 128, gpu.slots, ref(q.suh));
    e.barrier();
    var td = t;
    td.jobs_off = @as(usize, q.gu_cap) * @sizeOf(ex.job.Job);
    pp.encodeIndirect(&K.pp, e, ref(q.arot), td, ref(q.z), loc, hid, true, ref(q.members), .{ .buf = q.dispatch, .off = 16 });
    e.barrier();
    try op.encodeCombine(&K.op, e, ref(q.z), ref(q.wts), ref(q.sv), L.k.halves, L.k.signs, out, .{ .rows = rows, .slots = gpu.slots, .z_stride = gpu.slots * hid, .y_stride = hid, .y_f32 = 1, .sign = m.signs + hid, .z_slot = hid, .per_row = 1 }, hid / 128);
}
