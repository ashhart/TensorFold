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

/// A layer on the GPU with its dense dispatch groups and its caches (latent and rope keys; index keys when full).
pub const Layer = struct {
    w: wt.Layer,
    k: gpu.Packed,
    groups: [8]gpu.Groups,
    cache: mtl.Buffer,
    index_cache: ?mtl.Buffer,
    /// The experts' tables for the jobs kernel (MoE layers).
    desc: ?mtl.Buffer = null,

    pub const x_g = 0;
    pub const uv_g = 1;
    pub const q_b_g = 2;
    pub const wq_b_g = 3;
    pub const o_g = 4;
    pub const router_g = 5;
    pub const eh_g = 6;

    fn one(r: anytype) pack.Range {
        return switch (r) {
            .rows => |x| .{ .lo = x.job, .hi = x.job + 1 },
            .trellis => .{ .lo = 0, .hi = 0 },
        };
    }

    /// `storage`: private for serving; shared lets a check read the caches back.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, w: wt.Layer, p: *const pack.Pack, slots: u32, context: u32, storage: usize) !Layer {
        const a = w.attn;
        var d: [8]pack.Range = @splat(.{ .lo = 0, .hi = 0 });
        d[x_g] = a.x_jobs;
        d[uv_g] = a.uv_jobs;
        d[q_b_g] = one(a.q_b);
        if (a.wq_b) |r| d[wq_b_g] = .{ .lo = r.job, .hi = r.job + 1 };
        d[o_g] = one(a.o);
        if (w.mlp == .moe) d[router_g] = .{ .lo = w.mlp.moe.router.job, .hi = w.mlp.moe.router.job + 1 };
        if (w.mtp) |t| d[eh_g] = one(t.eh);
        var l: Layer = .{ .w = w, .k = undefined, .groups = undefined, .cache = undefined, .index_cache = null };
        l.k = try gpu.Packed.init(gpa, device, p, &d, &l.groups);
        l.cache = try device.buffer(@as(usize, slots) * context * 576 * 2, storage);
        if (w.full) l.index_cache = try device.buffer(@as(usize, slots) * context * 128 * 2, storage);
        if (w.mlp == .moe) l.desc = try descs(device, w.mlp.moe);
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
        if (l.w.mlp == .moe) gpa.free(l.w.mlp.moe.experts);
        l.k.deinit(gpa);
        l.cache.deinit();
        if (l.index_cache) |b| b.deinit();
        if (l.desc) |b| b.deinit();
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

/// h += y's `parts`, s.x = rms(h) w_in (parts null: `x` is normed), attention into `out`; full layers list keys.
pub fn attention(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, last_pos: u32, parts: ?u32, x: Ref, out: Ref) !void {
    const nh = c.localHeads();
    const a = L.w.attn;
    const hid = c.hidden;
    if (parts) |n| {
        kn.addRms(&K.k, e, ref(s.h), ref(s.y), L.k.at(L.w.in_norm), .{ .width = hid, .x_stride = hid, .y_stride = hid, .parts = n, .part_stride = gpu.max_rows * hid, .eps = c.eps }, ref(s.x), rows);
        e.barrier();
    }
    try dg.encodeProject(&K.dg, e, if (parts != null) ref(s.x) else x, L.k.denseFor(L.groups[Layer.x_g], rows), ref(s.proj), rows, hid, wt.proj_width, true);
    e.barrier();
    const ma: kn.MlaArgs = .{ .rows = rows, .heads = nh, .slot_keys = s.context, .proj_stride = wt.proj_width, .q_stride = nh * c.qHead(), .kvb = a.kvb_w, .kvb_scale = a.kvb_s };
    var mp = ma;
    mp.q_stride = c.q_lora;
    kn.prep(&K.k, e, ref(s.proj), L.k.at(a.q_norm), L.k.at(a.kv_norm), ref(K.inv), rowsOf(s), mp, ref(s.q_resid), ref(L.cache));
    const ia: kn.IndexArgs = .{ .rows = rows, .slot_keys = s.context, .proj_stride = wt.proj_width, .k_col = wt.wk_col, .w_col = wt.wp_col, .q_stride = c.i_heads * c.i_dim, .score_stride = s.context, .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_dim))), .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.i_heads))) };
    if (L.w.full) kn.indexKey(&K.k, e, ref(s.proj), L.k.at(a.k_norm), L.k.at(a.k_bias), ref(K.inv), rowsOf(s), ia, ref(L.index_cache.?));
    e.barrier();
    try linear(K, L, e, a.q_b, L.groups[Layer.q_b_g], ref(s.q_resid), c.q_lora, ref(s.rot), ref(s.q), nh * c.qHead(), rows);
    if (L.w.full) try dg.encodeProject(&K.dg, e, ref(s.q_resid), L.k.denseFor(L.groups[Layer.wq_b_g], rows), ref(s.qidx), rows, c.q_lora, c.i_heads * c.i_dim, true);
    e.barrier();
    if (L.w.full) kn.indexTopk(&K.k, e, ref(s.qidx), ref(s.proj), ref(L.index_cache.?), rowsOf(s), ia, ref(s.score), ref(K.inv), ref(s.keys), last_pos);
    kn.absorb(&K.k, e, ref(s.q), L.k.bytes, rowsOf(s), ma, ref(s.qlat), ref(s.qrope), ref(K.inv));
    e.barrier();
    kn.attend(&K.k, e, ref(s.qlat), ref(s.qrope), rowsOf(s), ref(s.keys), ref(L.cache), ma, ref(s.partial), ref(s.olat));
    e.barrier();
    try dg.encodeProject(&K.dg, e, ref(s.olat), L.k.denseFor(L.groups[Layer.uv_g], rows), ref(s.o), rows, nh * c.kv_lora, nh * c.v_dim, true);
    e.barrier();
    try linear(K, L, e, a.o, L.groups[Layer.o_g], ref(s.o), nh * c.v_dim, ref(s.rot), out, hid, rows);
}

/// h += y's `parts` (attention's partial sums, in rank order), x = rms(h) w_post.
pub fn postNorm(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, parts: u32) void {
    const hid = c.hidden;
    kn.addRms(&K.k, e, ref(s.h), ref(s.y), L.k.at(L.w.post_norm), .{ .width = hid, .x_stride = hid, .y_stride = hid, .parts = parts, .part_stride = gpu.max_rows * hid, .eps = c.eps }, ref(s.x), rows);
}

/// The dense MLP (layers before the first MoE layer) of x into `out`: this Mac's gate and up rows, its down inputs.
pub fn denseMlp(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, out: Ref) !void {
    const m = L.w.mlp.dense;
    const n = c.localDense();
    const none: gpu.Groups = .{ .off = 0, .count = 0 };
    try linear(K, L, e, m.gate, none, x, c.hidden, ref(s.rot), ref(s.gu), 2 * n, rows);
    e.barrier();
    try linear(K, L, e, m.up, none, x, c.hidden, ref(s.rot), .{ .buf = s.gu, .off = n * 4 }, 2 * n, rows);
    e.barrier();
    kn.siluMul(&K.k, e, ref(s.gu), .{ .width = n, .x_stride = 2 * n, .y_stride = n }, ref(s.act), rows);
    e.barrier();
    try linear(K, L, e, m.down, none, ref(s.act), n, ref(s.rot), out, c.hidden, rows);
}

/// The router's logits of x (1-16 rows), the top-k with weights into `pick` and `wts` (null: the scratch's route).
pub fn route(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, x: Ref, pick: ?Ref, wts: ?Ref) !void {
    const m = L.w.mlp.moe;
    try dg.encodeProject(&K.dg, e, x, L.k.denseFor(L.groups[Layer.router_g], rows), ref(s.logits), rows, c.hidden, c.experts, true);
    e.barrier();
    const r = s.route;
    kn.route_select(e, K.k.select, ref(s.logits), L.k.at(m.bias), c.routed_scale, rows, .{ 0, c.experts }, kn.RouteOutputs(Ref){ .pick = pick orelse ref(r[0]), .wts = wts orelse ref(r[1]), .ids = ref(r[2]), .members = ref(r[3]), .count = ref(r[4]), .mine = ref(r[5]), .theirs = ref(r[6]), .counts = ref(r[7]), .word = ref(r[8]) });
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

/// The experts of x into s.y with jobs the GPU writes from the route (after `route`, on the same encoder).
pub fn expertsGpu(K: *const Kernels, L: *const Layer, s: *const gpu.Scratch, e: mtl.ComputeEncoder, c: *const cfg.Config, rows: u32, out: Ref) !void {
    const m = L.w.mlp.moe;
    const hid = c.hidden;
    const loc = c.localHalves() * 128;
    const P = rows * gpu.slots;
    e.barrier();
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
}
