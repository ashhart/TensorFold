//! One GLM-5.3 layer for a window of lane rows on one Mac: attention with its indexer, then the MLP or the experts.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("../../core/exlr8.zig");
const op = @import("../../core/exlr8_projection.zig");
const dg = @import("../../core/exlr8_dense_gpu.zig");
pub const pp = @import("../../core/exlr8_prompt.zig");
const cfg = @import("config.zig");
const kn = @import("kernels.zig");
const gpu = @import("gpu.zig");
const wt = @import("weights.zig");
const hfw = @import("hf_weights.zig");
const pack = ex.pack;
const Ref = kn.Ref;

pub const Kernels = struct {
    k: kn.Pipelines,
    op: op.Pipelines,
    dg: dg.Pipelines,
    pp: pp.Pipelines,
    /// The rope's 32 inverse frequencies, fp32.
    inv: mtl.Buffer,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const cfg.Config) !Kernels {
        const inv = try device.buffer(c.rope / 2 * 4, mtl.ResourceOptions.shared);
        for (inv.slice(f32, c.rope / 2), 0..) |*v, i| {
            v.* = @floatCast(1.0 / std.math.pow(f64, c.rope_theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(c.rope))));
        }
        return .{ .k = try kn.Pipelines.load(gpa, device, c.experts, c.hidden, c.topk), .op = try op.Pipelines.load(device), .dg = try dg.Pipelines.load(device), .pp = try pp.Pipelines.load(gpa, device), .inv = inv };
    }

    pub fn deinit(k: *Kernels) void {
        k.k.deinit();
        k.op.deinit();
        k.dg.deinit();
        k.pp.deinit();
        k.inv.deinit();
    }
};

/// Constant-arg descriptor for a contiguous FP8 block-128 dense matrix inside `Layer.k.bytes`.
pub const Fp8Mat = struct { w: u64, s: u64, outs: u32, ins: u32, y_col: u32, sb_cols: u32 };

/// A layer on the GPU with its dense dispatch groups, caches, and optional FP8 MoE expert buffers.
pub const Layer = struct {
    w: wt.Layer,
    k: gpu.Packed,
    groups: [11]gpu.Groups,
    cache: mtl.Buffer,
    index_cache: ?mtl.Buffer,
    /// The experts' tables for the EXLR8 jobs kernel (null for native FP8 checkpoints).
    desc: ?mtl.Buffer = null,
    /// Native HF FP8 E4M3 128x128 block-scaled expert weights (null for EXLR8 checkpoints).
    fp8_moe: ?hfw.Fp8Moe = null,
    fp8_x: [3]?Fp8Mat = @splat(null),
    fp8_x_n: u8 = 0,
    fp8_q_b: ?Fp8Mat = null,
    fp8_wq_b: ?Fp8Mat = null,
    fp8_o: ?Fp8Mat = null,
    fp8_dgate: ?Fp8Mat = null,
    fp8_dup: ?Fp8Mat = null,
    fp8_ddown: ?Fp8Mat = null,

    pub const x_g = 0;
    pub const uv_g = 1;
    pub const q_b_g = 2;
    pub const wq_b_g = 3;
    pub const o_g = 4;
    pub const router_g = 5;
    pub const eh_g = 6;
    pub const dense_gate_g = 7;
    pub const dense_up_g = 8;
    pub const dense_down_g = 9;
    pub const wp_g = 10;

    fn one(r: anytype) pack.Range {
        return switch (r) {
            .rows => |x| .{ .lo = x.job, .hi = x.job + 1 },
            .trellis => .{ .lo = 0, .hi = 0 },
        };
    }

    fn extractFp8(p: *const pack.Pack, job_idx: u32) ?Fp8Mat {
        const dj = p.dense_jobs.items[job_idx];
        if (dj.format != 4 or dj.out0 != 0 or dj.in0 != 0 or dj.x_col != 0 or dj.ins != dj.in_total) return null;
        if ((dj.ins % 512 != 0) or (dj.outs % 16 != 0) or (dj.w % 16 != 0)) return null;
        return .{ .w = dj.w, .s = dj.s, .outs = dj.outs, .ins = dj.ins, .y_col = dj.y_col, .sb_cols = (dj.in_total + 127) / 128 };
    }

    fn extractLin(p: *const pack.Pack, lin: pack.Linear) ?Fp8Mat {
        return switch (lin) {
            .rows => |r| extractFp8(p, r.job),
            .trellis => null,
        };
    }

    /// `storage`: private for serving; shared lets a check read the caches back.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, w: wt.Layer, p: *const pack.Pack, slots: u32, context: u32, storage: usize) !Layer {
        return initWithFp8Queue(gpa, device, null, w, p, null, slots, context, storage);
    }

    pub fn initWithFp8(gpa: std.mem.Allocator, device: mtl.Device, w: wt.Layer, p: *const pack.Pack, fp8_moe: ?hfw.Fp8Moe, slots: u32, context: u32, storage: usize) !Layer {
        return initWithFp8Queue(gpa, device, null, w, p, fp8_moe, slots, context, storage);
    }

    pub fn initWithFp8Queue(gpa: std.mem.Allocator, device: mtl.Device, bq: ?mtl.Queue, w: wt.Layer, p: *const pack.Pack, fp8_moe: ?hfw.Fp8Moe, slots: u32, context: u32, storage: usize) !Layer {
        const a = w.attn;
        var d: [11]pack.Range = @splat(.{ .lo = 0, .hi = 0 });
        d[x_g] = a.x_jobs;
        d[uv_g] = a.uv_jobs;
        d[q_b_g] = one(a.q_b);
        if (a.wq_b) |r| d[wq_b_g] = .{ .lo = r.job, .hi = r.job + 1 };
        if (a.wp) |r| d[wp_g] = .{ .lo = r.job, .hi = r.job + 1 };
        d[o_g] = one(a.o);
        switch (w.mlp) {
            .moe => |m| d[router_g] = .{ .lo = m.router.job, .hi = m.router.job + 1 },
            .dense => |dm| {
                d[dense_gate_g] = one(dm.gate);
                d[dense_up_g] = one(dm.up);
                d[dense_down_g] = one(dm.down);
            },
        }
        if (w.mtp) |t| d[eh_g] = one(t.eh);
        var l: Layer = .{ .w = w, .k = undefined, .groups = undefined, .cache = undefined, .index_cache = null, .fp8_moe = fp8_moe };
        const x_cnt = a.x_jobs.hi - a.x_jobs.lo;
        if (x_cnt > 0 and x_cnt <= 3) {
            var all_ok = true;
            for (0..x_cnt) |i| {
                if (extractFp8(p, a.x_jobs.lo + @as(u32, @intCast(i)))) |m| {
                    l.fp8_x[i] = m;
                } else all_ok = false;
            }
            if (all_ok) l.fp8_x_n = @intCast(x_cnt);
        }
        l.fp8_q_b = extractLin(p, a.q_b);
        if (a.wq_b) |r| l.fp8_wq_b = extractFp8(p, r.job);
        l.fp8_o = extractLin(p, a.o);
        if (w.mlp == .dense) {
            l.fp8_dgate = extractLin(p, w.mlp.dense.gate);
            l.fp8_dup = extractLin(p, w.mlp.dense.up);
            l.fp8_ddown = extractLin(p, w.mlp.dense.down);
        }
        l.k = try gpu.Packed.initWithQueue(gpa, device, bq, p, &d, &l.groups);
        l.cache = try device.buffer(@as(usize, slots) * context * 576 * 2, storage);
        if (w.full) l.index_cache = try device.buffer(@as(usize, slots) * context * 128 * 2, storage);
        if (w.mlp == .moe and fp8_moe == null) l.desc = try descs(device, w.mlp.moe);
        return l;
    }

    fn descs(device: mtl.Device, m: wt.Moe) !mtl.Buffer {
        const b = try device.buffer(m.experts.len * @sizeOf(kn.ExpertDesc), mtl.ResourceOptions.shared);
        for (b.slice(kn.ExpertDesc, m.experts.len), m.experts) |*d, x| {
            d.* = .{ .down_k = x.down[0].k, .down_base = x.down_base, .down_suh = x.down_suh, .down_block = x.down[0].block };
            for (0..wt.max_halves) |h| {
                d.gate_k[h] = x.gate[h].k;
                d.gate_base[h] = x.gate_base[h];
                d.gate_chan[h] = x.gate[h].chan;
                d.gate_block[h] = x.gate[h].block;
                d.up_k[h] = x.up[h].k;
                d.up_base[h] = x.up_base[h];
                d.up_chan[h] = x.up[h].chan;
                d.up_block[h] = x.up[h].block;
            }
        }
        return b;
    }

    pub fn deinit(l: *Layer, gpa: std.mem.Allocator) void {
        if (l.w.mlp == .moe and l.w.mlp.moe.experts.len > 0) gpa.free(l.w.mlp.moe.experts);
        l.k.deinit(gpa);
        l.cache.deinit();
        if (l.index_cache) |b| b.deinit();
        if (l.desc) |b| b.deinit();
        if (l.fp8_moe) |*fm| fm.deinit();
    }
};

fn ref(b: mtl.Buffer) Ref {
    return .{ .buf = b };
}

fn rowsOf(s: *const gpu.Scratch) Ref {
    return .{ .buf = s.rows, .off = s.rows_off };
}

/// y (y_stride apart) = x (x_stride apart) times the linear's slice; a trellis one rotates x into `rot` first.
pub fn linear(K: *const Kernels, L: *const Layer, e: mtl.ComputeEncoder, l: pack.Linear, g: gpu.Groups, x: Ref, x_stride: usize, rot: Ref, y: Ref, y_stride: usize, rows: u32) !void {
    switch (l) {
        .rows => try dg.encodeProject(&K.dg, e, x, L.k.denseFor(g, rows), y, rows, x_stride, y_stride, true),
        .trellis => |t| {
            try op.encodeRotate(&K.op, e, x, L.k.signs, L.k.halves, rot, .{ .rows = rows, .x_stride = @intCast(x_stride), .x_f32 = 1, .r_stride = t.in, .sign = t.sign, .scale = t.scale }, t.in / 128, 1, null);
            e.barrier();
            try op.encodeProject(&K.op, e, rot, L.k.trellis(t), y, rows, t.in, y_stride, true, null);
        },
    }
}

fn matArgs(m: Fp8Mat, rows: u32, x_stride: u32, y_stride: u32, y_col: u32) kn.fast.Fp8MatArgs {
    return .{
        .outs = m.outs,
        .ins = m.ins,
        .rows = rows,
        .x_stride = x_stride,
        .y_stride = y_stride,
        .y_col = y_col,
        .sb_cols = m.sb_cols,
        .w_off = m.w,
        .s_off = m.s,
    };
}

fn mat3Args(m0: Fp8Mat, m1: Fp8Mat, m2: ?Fp8Mat, rows: u32, x_stride: u32, y_stride: u32, yc0: u32, yc1: u32, yc2: u32) kn.fast.Fp8Mat3Args {
    const per_tg: u32 = if (rows >= 3 and rows <= 5) 4 else if (rows == 1) 8 else 16;
    const g0 = (m0.outs + per_tg - 1) / per_tg;
    const g1 = (m1.outs + per_tg - 1) / per_tg;
    const g2: u32 = if (m2) |m| (m.outs + per_tg - 1) / per_tg else 0;
    return .{
        .o0 = m0.outs,
        .o1 = m1.outs,
        .o2 = if (m2) |m| m.outs else 0,
        .ins = m0.ins,
        .rows = rows,
        .x_stride = x_stride,
        .y_stride = y_stride,
        .sb_cols = m0.sb_cols,
        .yc0 = yc0,
        .yc1 = yc1,
        .yc2 = yc2,
        .g0 = g0,
        .g1 = g1,
        .g2 = g2,
        .w0 = m0.w,
        .s0 = m0.s,
        .w1 = m1.w,
        .s1 = m1.s,
        .w2 = if (m2) |m| m.w else 0,
        .s2 = if (m2) |m| m.s else 0,
    };
}

/// h += y's `parts`, s.x = rms(h) w_in (parts null: `x` is normed), attention into `out`; optionally BF16-packs peers.
pub fn attention(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, last_pos: u32, parts: ?u32, x: Ref, out: Ref, pack_info: [4]u32, pack_base: Ref) !bool {
    return attentionGated(K, L, s, e, c, rows, last_pos, parts, x, out, pack_info, pack_base, null, .{});
}

pub fn attentionGated(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, last_pos: u32, parts: ?u32, x: Ref, out: Ref, pack_info: [4]u32, pack_base: Ref, gate: ?mtl.Buffer, tree_args: kn.TreeArgs) !bool {
    const nh = c.localHeads();
    const a = L.w.attn;
    const hid = c.hidden;
    const use_xh = rows >= 2 and rows <= 8;
    if (parts) |n| {
        kn.addRms(&K.k, e, ref(s.h), ref(s.y), L.k.at(L.w.in_norm), .{ .width = hid, .x_stride = hid, .y_stride = hid, .parts = n, .part_stride = gpu.max_rows * hid, .pad0 = if (use_xh) @intCast(gpu.max_rows * (hid / 4)) else 0, .eps = c.eps }, ref(s.x), rows);
        e.barrier();
    }
    const x_in = if (parts != null) ref(s.x) else x;
    const x_fp8 = if (use_xh) s.xhRef(hid) else x_in;
    if (rows <= 8 and L.fp8_x_n >= 2) {
        const m0 = L.fp8_x[0].?;
        const m1 = L.fp8_x[1].?;
        const m2 = if (L.fp8_x_n >= 3) L.fp8_x[2] else null;
        kn.fast.fp8Gemv3(&K.k.fast_p, e, x_fp8, L.k.bytes, mat3Args(m0, m1, m2, rows, hid, wt.proj_width, m0.y_col, m1.y_col, if (m2) |m| m.y_col else 0), ref(s.proj), gate);
    } else if (rows <= 8 and L.fp8_x_n == 1) {
        const fm = L.fp8_x[0].?;
        kn.fast.fp8Gemv(&K.k.fast_p, e, x_fp8, L.k.bytes, matArgs(fm, rows, hid, wt.proj_width, fm.y_col), ref(s.proj), gate, 1);
    } else {
        try dg.encodeProject(&K.dg, e, x_in, L.k.denseFor(L.groups[Layer.x_g], rows), ref(s.proj), rows, hid, wt.proj_width, true);
    }
    e.barrier();
    const ma: kn.MlaArgs = .{ .rows = rows, .heads = nh, .slot_keys = s.context, .proj_stride = wt.proj_width, .q_stride = nh * c.qHead(), .pad1 = if (use_xh) 1.0 else 0.0, .kvb = a.kvb_w, .kvb_scale = a.kvb_s, .tree = tree_args };
    var mp = ma;
    mp.q_stride = c.q_lora;
    mp.pad2 = if (use_xh) gpu.max_rows * c.q_lora else 0;
    const ia: kn.IndexArgs = .{ .rows = rows, .slot_keys = s.context, .proj_stride = wt.proj_width, .k_col = wt.wk_col, .w_col = wt.wp_col, .q_stride = c.i_heads * c.i_dim, .score_stride = s.context, .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_dim))), .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_heads))), .tree = tree_args };
    const need_idx = L.w.full and last_pos >= c.i_topk;
    if (L.w.full) {
        kn.prepFull(&K.k, e, ref(s.proj), L.k.at(a.q_norm), L.k.at(a.kv_norm), ref(K.inv), rowsOf(s), mp, ref(s.q_resid), ref(L.cache), L.k.at(a.k_norm), L.k.at(a.k_bias), ref(L.index_cache.?), gate);
        if (need_idx) {
            if (a.wp_bf16_off != 0 and rows <= 16 and (hid % 2048 == 0)) {
                kn.fast.wpBf16Decode(&K.k.fast_p, e, x_in, L.k.at(a.wp_bf16_off), .{ .rows = rows, .cols = c.i_heads, .hidden = hid, .proj_stride = wt.proj_width, .wp_col = wt.wp_col }, ref(s.proj), gate);
            } else if (a.wp != null) {
                try dg.encodeProject(&K.dg, e, x_in, L.k.denseFor(L.groups[Layer.wp_g], rows), ref(s.proj), rows, hid, wt.proj_width, true);
            }
        }
    } else {
        kn.prep(&K.k, e, ref(s.proj), L.k.at(a.q_norm), L.k.at(a.kv_norm), ref(K.inv), rowsOf(s), mp, ref(s.q_resid), ref(L.cache));
    }
    e.barrier();
    const q_fp8 = if (use_xh) s.qhRef(c.q_lora) else ref(s.q_resid);
    if (need_idx) {
        if (rows <= 8 and L.fp8_wq_b != null) {
            kn.fast.fp8Gemv(&K.k.fast_p, e, q_fp8, L.k.bytes, matArgs(L.fp8_wq_b.?, rows, c.q_lora, c.i_heads * c.i_dim, 0), ref(s.qidx), gate, 3);
        } else {
            try dg.encodeProject(&K.dg, e, ref(s.q_resid), L.k.denseFor(L.groups[Layer.wq_b_g], rows), ref(s.qidx), rows, c.q_lora, c.i_heads * c.i_dim, true);
        }
        e.barrier();
        kn.indexScoresBlock(&K.k, e, ref(s.qidx), ref(s.proj), ref(L.index_cache.?), rowsOf(s), ia, ref(s.score), ref(K.inv), last_pos, gate);
    }
    if (rows <= 8 and L.fp8_q_b != null) {
        kn.fast.fp8Gemv(&K.k.fast_p, e, q_fp8, L.k.bytes, matArgs(L.fp8_q_b.?, rows, c.q_lora, nh * c.qHead(), 0), ref(s.q), gate, 2);
    } else {
        try linear(K, L, e, a.q_b, L.groups[Layer.q_b_g], ref(s.q_resid), c.q_lora, ref(s.rot), ref(s.q), nh * c.qHead(), rows);
    }
    e.barrier();
    if (need_idx) kn.indexTopkOnly(&K.k, e, rowsOf(s), ia, ref(s.score), ref(s.keys), gate);
    kn.absorbGated(&K.k, e, ref(s.q), L.k.bytes, rowsOf(s), ma, ref(s.qlat), ref(s.qrope), ref(K.inv), gate);
    e.barrier();
    const fuse_uv = L.k.fast_dense and c.nope == 192 and c.v_dim == 256 and c.kv_lora == 512;
    const fused_uv = kn.attendDecode(&K.k, e, ref(s.qlat), ref(s.qrope), rowsOf(s), ref(s.keys), ref(L.cache), ma, ref(s.partial), ref(s.olat), last_pos, if (fuse_uv) L.k.bytes else null, if (fuse_uv) ref(s.o) else null, gate);
    e.barrier();
    if (!fused_uv) {
        try dg.encodeProject(&K.dg, e, ref(s.olat), L.k.denseFor(L.groups[Layer.uv_g], rows), ref(s.o), rows, nh * c.kv_lora, nh * c.v_dim, true);
        e.barrier();
        if (use_xh and L.fp8_o != null) {
            kn.fast.castXh(&K.k.fast_p, e, ref(s.o), s.ohRef(nh, c.v_dim), (rows * nh * c.v_dim) / 4);
            e.barrier();
        }
    }
    if (rows <= 8 and L.fp8_o != null and (hid % 4 == 0) and (rows > 1 or hid % 16 == 0)) {
        kn.fast.fp8OPack(&K.k.fast_p, e, if (use_xh) s.ohRef(nh, c.v_dim) else ref(s.o), L.k.bytes, matArgs(L.fp8_o.?, rows, nh * c.v_dim, hid, 0), out, pack_info, pack_base, gate);
        return pack_info[1] > 1;
    }
    if (rows <= 6 and pack_info[1] > 1 and L.k.fast_dense and a.o == .rows and (nh * c.v_dim % 512 == 0) and (hid % 4 == 0) and (rows > 1 or hid % 16 == 0)) {
        try dg.encodeProjectPackBf16(&K.dg, e, ref(s.o), L.k.dense16(L.groups[Layer.o_g]), out, rows, nh * c.v_dim, hid, pack_info, pack_base);
        return true;
    }
    try linear(K, L, e, a.o, L.groups[Layer.o_g], ref(s.o), nh * c.v_dim, ref(s.rot), out, hid, rows);
    return false;
}

/// h += y's `parts` (attention's partial sums, in rank order), x = rms(h) w_post.
pub fn postNorm(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, parts: u32) void {
    const hid = c.hidden;
    kn.addRms(&K.k, e, ref(s.h), ref(s.y), L.k.at(L.w.post_norm), .{ .width = hid, .x_stride = hid, .y_stride = hid, .parts = parts, .part_stride = gpu.max_rows * hid, .eps = c.eps }, ref(s.x), rows);
}

/// The dense MLP (layers before the first MoE layer) of x into `out`: this Mac's gate and up rows, its down inputs.
pub fn denseMlp(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, out: Ref) !void {
    _ = try denseMlpPack(K, L, s, e, c, rows, x, out, .{ 0, 1, 0, 0 }, out);
}

/// Dense MLP with optional fused peer BF16 pack on `down` (returns true if peer slots were packed).
pub fn denseMlpPack(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, out: Ref, pack_info: [4]u32, pack_base: Ref) !bool {
    const m = L.w.mlp.dense;
    const n = c.localDense();
    const use_xh = rows >= 2 and rows <= 8;
    if (rows <= 8 and L.fp8_dgate != null and L.fp8_dup != null) {
        kn.fast.fp8Gemv3(&K.k.fast_p, e, if (use_xh) s.xhRef(c.hidden) else x, L.k.bytes, mat3Args(L.fp8_dgate.?, L.fp8_dup.?, null, rows, c.hidden, 2 * n, 0, n, 0), ref(s.gu), null);
    } else {
        try linear(K, L, e, m.gate, L.groups[Layer.dense_gate_g], x, c.hidden, ref(s.rot), ref(s.gu), 2 * n, rows);
        if (m.gate == .trellis or m.up == .trellis) e.barrier();
        try linear(K, L, e, m.up, L.groups[Layer.dense_up_g], x, c.hidden, ref(s.rot), .{ .buf = s.gu, .off = n * 4 }, 2 * n, rows);
    }
    e.barrier();
    kn.siluMul(&K.k, e, ref(s.gu), .{ .width = n, .x_stride = 2 * n, .y_stride = n, .pad0 = if (use_xh and L.fp8_ddown != null) gpu.max_rows * n else 0 }, ref(s.act), rows);
    e.barrier();
    if (rows <= 8 and L.fp8_ddown != null and (c.hidden % 4 == 0) and (rows > 1 or c.hidden % 16 == 0)) {
        kn.fast.fp8OPack(&K.k.fast_p, e, if (use_xh) s.acthRef(n) else ref(s.act), L.k.bytes, matArgs(L.fp8_ddown.?, rows, n, c.hidden, 0), out, pack_info, pack_base, null);
        return pack_info[1] > 1;
    }
    try linear(K, L, e, m.down, L.groups[Layer.dense_down_g], ref(s.act), n, ref(s.rot), out, c.hidden, rows);
    return false;
}

/// Router logits of x (1-16 rows), top-k into `pick`/`wts`; returns true if MoE jobs were also built in-kernel.
pub fn route(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, pick: ?Ref, wts: ?Ref) !bool {
    return routeGated(K, L, s, e, c, rows, x, pick, wts, null, null);
}

pub fn routeGated(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, pick: ?Ref, wts: ?Ref, gate: ?mtl.Buffer, keep_meta: ?mtl.Buffer) !bool {
    const m = L.w.mlp.moe;
    if (L.fp8_moe) |fm| {
        const r = s.route;
        if (fm.router_bf16_off != 0 and rows <= 8 and c.experts == 256 and c.topk == 8 and (c.hidden % 2048 == 0)) {
            const ra: kn.fast.RouterFusedArgs = .{
                .rows = rows,
                .experts = c.experts,
                .hidden = c.hidden,
                .topk = c.topk,
                .loc = c.localHalves() * 128,
                .slots = gpu.slots,
                .scale = c.routed_scale,
            };
            if (rows == 1) {
                kn.fast.routerRoute1(&K.k.fast_p, e, x, L.k.at(fm.router_bf16_off), L.k.at(m.bias), ra, ref(s.logits), ref(s.route_done), pick orelse ref(r[0]), wts orelse ref(r[1]), gate);
                return false;
            }
            if (pick == null and wts == null) {
                kn.fast.routerRouteJobs(&K.k.fast_p, e, x, L.k.at(fm.router_bf16_off), L.k.at(m.bias), ra, ref(s.logits), ref(s.route_done), ref(r[0]), ref(r[1]), ref(s.moe_jobs), ref(s.members), ref(s.wts), ref(s.dispatch), keep_meta);
                return true;
            }
            kn.routerBf16Decode(&K.k, e, x, L.k.at(fm.router_bf16_off), .{ .rows = rows, .experts = c.experts, .hidden = c.hidden }, ref(s.logits));
        } else if (fm.router_bf16_off != 0 and rows <= 16 and (c.hidden % 512 == 0)) {
            kn.routerBf16Decode(&K.k, e, x, L.k.at(fm.router_bf16_off), .{ .rows = rows, .experts = c.experts, .hidden = c.hidden }, ref(s.logits));
        } else {
            try dg.encodeProject(&K.dg, e, x, L.k.denseFor(L.groups[Layer.router_g], rows), ref(s.logits), rows, c.hidden, c.experts, true);
        }
        e.barrier();
        kn.routePrompt(&K.k, e, ref(s.logits), L.k.at(m.bias), .{ .rows = rows, .experts = c.experts, .topk = c.topk, .scale = c.routed_scale }, pick orelse ref(r[0]), wts orelse ref(r[1]));
        return false;
    }
    try dg.encodeProject(&K.dg, e, x, L.k.denseFor(L.groups[Layer.router_g], rows), ref(s.logits), rows, c.hidden, c.experts, true);
    e.barrier();
    const r = s.route;
    kn.route_select(e, K.k.select, ref(s.logits), L.k.at(m.bias), c.routed_scale, rows, .{ 0, c.experts }, kn.RouteOutputs(Ref){ .pick = pick orelse ref(r[0]), .wts = wts orelse ref(r[1]), .ids = ref(r[2]), .members = ref(r[3]), .count = ref(r[4]), .mine = ref(r[5]), .theirs = ref(r[6]), .counts = ref(r[7]), .word = ref(r[8]) });
    return false;
}

/// The experts of x into s.y from jobs already in s.moe_jobs (gate/up first, then down) and their tables.
pub fn experts(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, gu_jobs: usize, down_jobs: usize, out: Ref) !void {
    const m = L.w.mlp.moe;
    const hid = c.hidden;
    const loc = c.localHalves() * 128;
    const P = rows * gpu.slots;
    const job_size = @sizeOf(ex.job.Job);
    try op.encodeRotate(&K.op, e, ref(s.x), L.k.signs, L.k.halves, ref(s.rot), .{ .rows = rows, .x_stride = hid, .x_f32 = 1, .r_stride = hid, .sign = m.signs }, hid / 128, 1, null);
    e.barrier();
    const t: op.Tables = .{ .weights = L.k.bytes, .bases = L.k.bases, .halves = L.k.halves, .signs = L.k.signs, .jobs = s.moe_jobs, .count = gu_jobs };
    try op.encodeProject(&K.op, e, ref(s.rot), t, ref(s.gu), rows, hid, 2 * loc, true, ref(s.members));
    e.barrier();
    kn.siluMul(&K.k, e, ref(s.gu), .{ .width = loc, .x_stride = 2 * loc, .y_stride = loc }, ref(s.act), P);
    e.barrier();
    try op.encodeRotate(&K.op, e, ref(s.act), L.k.signs, L.k.halves, ref(s.arot), .{ .rows = rows, .x_stride = gpu.slots * loc, .x_f32 = 1, .r_stride = gpu.slots * loc, .slot_cols = loc, .scales = 1, .row_slots = gpu.slots }, loc / 128, gpu.slots, ref(s.suh));
    e.barrier();
    var td = t;
    td.count = down_jobs;
    td.jobs_off = gu_jobs * job_size;
    try op.encodeProject(&K.op, e, ref(s.arot), td, ref(s.z), rows, loc, hid, true, ref(s.members));
    e.barrier();
    try op.encodeCombine(&K.op, e, ref(s.z), ref(s.wts), ref(s.sv), L.k.halves, L.k.signs, out, .{ .rows = rows, .slots = gpu.slots, .z_stride = gpu.slots * hid, .y_stride = hid, .y_f32 = 1, .sign = m.signs + hid, .z_slot = hid, .per_row = 1 }, hid / 128);
}

/// The experts of x into `out` with jobs the GPU writes from the route (returns true if peer BF16 slots were packed).
pub fn expertsGpu(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, out: Ref, pack_info: [4]u32, pack_base: Ref, jobs_ready: bool) !bool {
    return expertsGpuGated(K, L, s, e, c, rows, out, pack_info, pack_base, jobs_ready, null);
}

/// Concurrent router + shared-expert MoE overlap for R=2..8 HF FP8 (falls back cleanly when gated).
pub fn moeGpuGated(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, out: Ref, pack_info: [4]u32, pack_base: Ref, gate: ?mtl.Buffer, keep_meta: ?mtl.Buffer) !bool {
    const hid = c.hidden;
    const loc = c.localHalves() * 128;
    if (rows >= 2 and rows <= 8 and gate == null and c.experts == 256 and c.topk == 8 and (hid % 2048 == 0) and (loc % 512 == 0)) {
        if (L.fp8_moe) |fm| if (fm.router_bf16_off != 0) {
            const ra: kn.fast.RouterFusedArgs = .{
                .rows = rows,
                .experts = c.experts,
                .hidden = hid,
                .topk = c.topk,
                .loc = loc,
                .slots = gpu.slots,
                .pad1 = 1 | (if (rows <= 5) @as(u32, 2) else 0),
                .scale = c.routed_scale,
            };
            const fa: kn.Fp8MoeArgs = .{
                .rows = rows,
                .loc = loc,
                .hidden = hid,
                .topk = c.topk,
                .slots = gpu.slots,
                .experts = c.experts,
                .loc_blocks = c.localHalves(),
                .hid_blocks = hid / 128,
            };
            kn.fast.routerGemv(&K.k.fast_p, e, ref(s.x), L.k.at(fm.router_bf16_off), ra, ref(s.logits));
            kn.fast.fp8MoeSharedGu(&K.k.fast_p, e, s.xhRef(hid), fm.gate_w, fm.gate_s, fm.up_w, fm.up_s, fa, ref(s.act));
            e.barrier();
            kn.fast.routerRouteJobsOnly(&K.k.fast_p, e, ref(s.logits), L.k.at(L.w.mlp.moe.bias), ra, ref(s.route[0]), ref(s.route[1]), ref(s.moe_jobs), ref(s.members), ref(s.wts), ref(s.dispatch), keep_meta);
            kn.fast.fp8MoeSharedDown(&K.k.fast_p, e, ref(s.act), fm.down_w, fm.down_s, fa, ref(s.z));
            e.barrier();
            kn.fp8MoeDecode(&K.k, e, s.route, ref(s.x), s.xhRef(hid), fm.gate_w, fm.gate_s, fm.up_w, fm.up_s, fm.down_w, fm.down_s, fa, ref(s.moe_jobs), ref(s.members), ref(s.wts), ref(s.dispatch), ref(s.act), ref(s.z), out, pack_info, pack_base, true, null);
            return pack_info[1] > 1 and (hid % 4 == 0);
        };
    }
    const jobs_ready = try routeGated(K, L, s, e, c, rows, ref(s.x), null, null, gate, keep_meta);
    return expertsGpuGated(K, L, s, e, c, rows, out, pack_info, pack_base, jobs_ready, gate);
}

pub fn expertsGpuGated(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, out: Ref, pack_info: [4]u32, pack_base: Ref, jobs_ready: bool, gate: ?mtl.Buffer) !bool {
    const hid = c.hidden;
    const loc = c.localHalves() * 128;
    e.barrier();
    if (L.fp8_moe) |fm| {
        kn.fp8MoeDecode(
            &K.k,
            e,
            s.route,
            ref(s.x),
            s.xhRef(hid),
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
            ref(s.moe_jobs),
            ref(s.members),
            ref(s.wts),
            ref(s.dispatch),
            ref(s.act),
            ref(s.z),
            out,
            pack_info,
            pack_base,
            jobs_ready,
            gate,
        );
        return rows <= 16 and pack_info[1] > 1 and (hid % 4 == 0);
    }
    const m = L.w.mlp.moe;
    const P = rows * gpu.slots;
    kn.moeJobs(&K.k, e, s.route, .{ .buf = L.desc.? }, .{ .rows = rows, .halves = c.localHalves(), .topk = c.topk, .slots = gpu.slots, .gu_cap = gpu.gu_cap }, ref(s.moe_jobs), ref(s.members), ref(s.wts), ref(s.sv), ref(s.suh), ref(s.dispatch));
    try op.encodeRotate(&K.op, e, ref(s.x), L.k.signs, L.k.halves, ref(s.rot), .{ .rows = rows, .x_stride = hid, .x_f32 = 1, .r_stride = hid, .sign = m.signs }, hid / 128, 1, null);
    e.barrier();
    // The split rule's count: at most each pick its own expert, the shared one too, gate and up, this Mac's halves.
    const gu_most = (@min(rows * c.topk, c.experts) + 1) * 2 * c.localHalves();
    const t: op.Tables = .{ .weights = L.k.bytes, .bases = L.k.bases, .halves = L.k.halves, .signs = L.k.signs, .jobs = s.moe_jobs, .count = gu_most, .blocks = hid / 128 };
    try op.encodeProjectIndirect(&K.op, e, ref(s.rot), t, ref(s.gu), rows, hid, 2 * loc, true, ref(s.members), ref(s.dispatch));
    e.barrier();
    kn.siluMul(&K.k, e, ref(s.gu), .{ .width = loc, .x_stride = 2 * loc, .y_stride = loc }, ref(s.act), P);
    e.barrier();
    try op.encodeRotate(&K.op, e, ref(s.act), L.k.signs, L.k.halves, ref(s.arot), .{ .rows = rows, .x_stride = gpu.slots * loc, .x_f32 = 1, .r_stride = gpu.slots * loc, .slot_cols = loc, .scales = 1, .row_slots = gpu.slots }, loc / 128, gpu.slots, ref(s.suh));
    e.barrier();
    var td = t;
    td.jobs_off = gpu.gu_cap * @sizeOf(ex.job.Job);
    td.blocks = loc / 128;
    try op.encodeProjectIndirect(&K.op, e, ref(s.arot), td, ref(s.z), rows, loc, hid, true, ref(s.members), .{ .buf = s.dispatch, .off = 16 });
    e.barrier();
    try op.encodeCombine(&K.op, e, ref(s.z), ref(s.wts), ref(s.sv), L.k.halves, L.k.signs, out, .{ .rows = rows, .slots = gpu.slots, .z_stride = gpu.slots * hid, .y_stride = hid, .y_f32 = 1, .sign = m.signs + hid, .z_slot = hid, .per_row = 1 }, hid / 128);
    return false;
}
