//! The DSpark drafter's encoders: the main rows into the key rings, a draft block's stages, and its head.
const std = @import("std");
const mtl = @import("metal");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const dw = @import("dspark_weights.zig");
const plan_ = @import("dspark_plan.zig");
const pipelines = @import("pipelines.zig");
const layer = @import("layer.zig");
const layer_attn = @import("attn.zig");
const moe = @import("moe.zig");
const forward = @import("forward.zig");

const Config = @import("dsv41").config.Config;
const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const g = pipelines.glue;
const cl = @import("calls.zig");
const attn = pipelines.attn;
const plus = ctx_.plus;

/// Concurrent ingest scratch sized for up to `max(16, window)` rows across all stages plus a 4-slot ring plan.
pub const IngestBuf = struct {
    buf: mtl.Buffer,
    plan_buf: mtl.Buffer,
    cap: u32,
    stages: u32,
    quant: Ref,
    h: Ref,
    hn: Ref,
    hq: Ref,
    kv: Ref,
    kvn: Ref,
    kvr: Ref,
    kvq: Ref,
    kv_stride: usize,

    pub const ring_slots = 4;

    pub fn init(device: mtl.Device, c: *const Config) !IngestBuf {
        const cap: u32 = @max(pipelines.max_rows, c.window);
        const stages: u32 = @max(1, c.mtp_layers);
        const taps_k: usize = @as(usize, @intCast(c.dspark_targets.len)) * c.hidden;
        const H: usize = c.hidden;
        const D: usize = c.head_dim;
        const kv_stride = std.mem.alignForward(usize, @as(usize, cap) * D * 2, 256);

        var off: usize = 0;
        const take = struct {
            fn next(o: *usize, len: usize) usize {
                const at = std.mem.alignForward(usize, o.*, 256);
                o.* = at + @max(len, 256);
                return at;
            }
        }.next;

        const q_off = take(&off, @as(usize, cap) * taps_k * 2);
        const h_off = take(&off, @as(usize, cap) * H * 2);
        const hn_off = take(&off, @as(usize, cap) * H * 2);
        const hq_off = take(&off, @as(usize, cap) * H * 2);
        const kv_off = take(&off, @as(usize, stages) * kv_stride);
        const kvn_off = take(&off, @as(usize, stages) * kv_stride);
        const kvr_off = take(&off, @as(usize, stages) * kv_stride);
        const kvq_off = take(&off, @as(usize, stages) * kv_stride);

        const buf = try device.buffer(off, mtl.ResourceOptions.shared);
        errdefer buf.deinit();
        @memset(buf.contents()[0..off], 0);

        const plan_bytes = ring_slots * 2 * @as(usize, cap) * 4;
        const plan_buf = try device.buffer(plan_bytes, mtl.ResourceOptions.shared);
        errdefer plan_buf.deinit();
        @memset(plan_buf.contents()[0..plan_bytes], 0);

        return .{
            .buf = buf,
            .plan_buf = plan_buf,
            .cap = cap,
            .stages = stages,
            .quant = .{ .buf = buf, .off = q_off },
            .h = .{ .buf = buf, .off = h_off },
            .hn = .{ .buf = buf, .off = hn_off },
            .hq = .{ .buf = buf, .off = hq_off },
            .kv = .{ .buf = buf, .off = kv_off },
            .kvn = .{ .buf = buf, .off = kvn_off },
            .kvr = .{ .buf = buf, .off = kvr_off },
            .kvq = .{ .buf = buf, .off = kvq_off },
            .kv_stride = kv_stride,
        };
    }

    pub fn deinit(ib: *IngestBuf) void {
        ib.plan_buf.deinit();
        ib.buf.deinit();
    }

    pub fn stageRef(ib: *const IngestBuf, base: Ref, stage_idx: usize) Ref {
        return plus(base, stage_idx * ib.kv_stride);
    }

    /// Write `n` ingest positions and destination ring rows into ring slot `ring_idx` (disjoint from queued CBs).
    pub fn writePlan(ib: *const IngestBuf, c: *const Config, ring_idx: u32, slot: u32, start: u64, n: u32) !struct { pos: Ref, dest: Ref } {
        if (n == 0 or n > ib.cap) return error.BadRows;
        const stride_ints: usize = 2 * @as(usize, ib.cap);
        const base_ints: usize = @as(usize, ring_idx % ring_slots) * stride_ints;
        const all = ib.plan_buf.slice(i32, ring_slots * stride_ints);
        const ipos = all[base_ints..][0..ib.cap];
        const idest = all[base_ints + ib.cap ..][0..ib.cap];
        for (0..n) |r| {
            ipos[r] = @intCast(start + r);
            idest[r] = @intCast(plan_.ringRow(c, slot, start + r));
        }
        return .{
            .pos = .{ .buf = ib.plan_buf, .off = base_ints * 4 },
            .dest = .{ .buf = ib.plan_buf, .off = (base_ints + ib.cap) * 4 },
        };
    }
};

/// A key of rows (a block's or main rows): wkv, kv_norm, RoPE, act_quant of `hq` [rows][hidden], into pool rows `dest`.
pub fn keys(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, hq: Ref, ring: mtl.Buffer, pos: Ref, dest: Ref) !void {
    try x.proj(e, hq, w.wkv, x.s.kv, false);
    e.barrier();
    try keyStore(x, e, w, ring, pos, dest);
}

/// The key's second half: kv_norm, RoPE and act_quant of s.kv into the ring's pool rows `dest`.
fn keyStore(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ring: mtl.Buffer, pos: Ref, dest: Ref) !void {
    const s = x.s;
    const f = pipelines.fused;
    const cv = ctx_.cv;
    const k: f.Kv = .{ .kv = cv(f.Ref, s.kv), .w = cv(f.Ref, w.kv_norm), .kvn = cv(f.Ref, s.kvn), .kvr = cv(f.Ref, s.kvr), .kvq = cv(f.Ref, s.kvq), .ring = .{ .buf = ring }, .pos = cv(f.Ref, pos), .table = cv(f.Ref, x.table(w.layer)), .dest = cv(f.Ref, dest), .reps = 1 };
    try cl.key(x.p, e, k, x.rows, x.c.head_dim, x.c.eps);
}

/// The main rows' shared part: act_quant (into `quant`), main_proj, main_norm and act_quant again, into s.hq.
pub fn mainRows(x: *const Ctx, e: mtl.ComputeEncoder, d: *const dw.Weights, taps: Ref, quant: Ref) !void {
    const c = x.c;
    const s = x.s;
    try cl.actqRows(x.p, e, taps, quant, x.rows, d.main_proj.k);
    try x.proj(e, quant, d.main_proj, s.h, false);
    try cl.normq(x.p, e, s.h, d.main_norm, s.hn, s.hq, x.rows, c.hidden, c.eps);
}

/// Main rows into every stage's ring: `mainRows`, then each stage's key of the result.
pub fn ingest(x: *const Ctx, e: mtl.ComputeEncoder, d: *const dw.Weights, taps: Ref, quant: Ref, rings: []const mtl.Buffer, pos: Ref, dest: Ref) !void {
    try mainRows(x, e, d, taps, quant);
    for (d.stages, rings) |*w, ring| try keys(x, e, w, x.s.hq, ring, pos, dest);
}

/// Dense FP8 projection for ingest: decode kernels up to 16 rows, bit-equal prompt MMA above 16 rows.
fn projIngest(x: *const Ctx, e: mtl.ComputeEncoder, in: Ref, w: weights.Fp8, out: Ref) !void {
    if (x.rows > pipelines.max_rows) {
        const m = pipelines.fp8_mma;
        return m.encode(&x.p.mma, e, ctx_.cv(m.Ref, in), ctx_.cv(m.Ref, w.w), ctx_.cv(m.Ref, w.s), ctx_.cv(m.Ref, out), .{ .k = w.k, .n = w.n, .rows = x.rows, .out = .bf16 });
    }
    return x.proj(e, in, w, out, false);
}

/// Concurrent batched ingest (1..window rows): mainRows, parallel stages' wkv, then parallel stages' keyStore.
pub fn ingestConcurrent(x: *const Ctx, e: mtl.ComputeEncoder, d: *const dw.Weights, ib: *const IngestBuf, taps: Ref, rings: []const mtl.Buffer, pos: Ref, dest: Ref) !void {
    const c = x.c;
    const f = pipelines.fused;
    const cv = ctx_.cv;
    try cl.actqRows(x.p, e, taps, ib.quant, x.rows, d.main_proj.k);
    e.barrier();
    try projIngest(x, e, ib.quant, d.main_proj, ib.h);
    e.barrier();
    try cl.normq(x.p, e, ib.h, d.main_norm, ib.hn, ib.hq, x.rows, c.hidden, c.eps);
    e.barrier();
    for (d.stages, 0..) |*w, i| {
        try projIngest(x, e, ib.hq, w.wkv, ib.stageRef(ib.kv, i));
    }
    e.barrier();
    for (d.stages, rings, 0..) |*w, ring, i| {
        const k: f.Kv = .{
            .kv = cv(f.Ref, ib.stageRef(ib.kv, i)),
            .w = cv(f.Ref, w.kv_norm),
            .kvn = cv(f.Ref, ib.stageRef(ib.kvn, i)),
            .kvr = cv(f.Ref, ib.stageRef(ib.kvr, i)),
            .kvq = cv(f.Ref, ib.stageRef(ib.kvq, i)),
            .ring = .{ .buf = ring },
            .pos = cv(f.Ref, pos),
            .table = cv(f.Ref, x.table(w.layer)),
            .dest = cv(f.Ref, dest),
            .reps = 1,
        };
        try cl.key(x.p, e, k, x.rows, c.head_dim, c.eps);
    }
}

/// Encode one concurrent ingest command buffer: optional GPU blit undo settle+save, then `ingestConcurrent`.
pub fn encodeIngestCb(cb: mtl.CommandBuffer, x: *const Ctx, d: *const dw.Weights, ib: *const IngestBuf, undo: anytype, gpu_undo: bool, slot: u32, end: u64, start: u64, taps: Ref, rings: []const mtl.Buffer, pos: Ref, dest: Ref) !void {
    if (gpu_undo) {
        undo.encodeSettleAndSave(cb, slot, end, start, x.rows);
    } else {
        undo.settle(slot, end);
        undo.save(slot, start, x.rows);
    }
    const e = cb.compute(.concurrent);
    ingestConcurrent(x, e, d, ib, taps, rings, pos, dest) catch |err| {
        e.end();
        return err;
    };
    e.end();
}

/// A draft launch's rows: their positions, and where their keys, attention records and selected keys are.
pub const Rows = struct {
    pos: Ref,
    dest: Ref,
    meta: Ref,
    sel: Ref,
};

/// A stage's attention: q and the block's keys, then each row over its context and the block, barriers between.
fn attention(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, ring: mtl.Buffer, rows: Rows) !void {
    const c = x.c;
    const s = x.s;
    const gp = &x.p.glue;
    const R = x.rows;
    const D = c.head_dim;
    const table = x.table(w.layer);
    // wq_a and the key's wkv both read hq; q_norm beside the key's norm, RoPE and ring write
    try x.proj(e, s.hq, w.wq_a, s.qa, false);
    try x.proj(e, s.hq, w.wkv, s.kv, false);
    e.barrier();
    try cl.normq(x.p, e, s.qa, w.q_norm, s.qr, s.qrq, R, c.q_lora, c.eps);
    try keyStore(x, e, w, ring, rows.pos, rows.dest);
    e.barrier();
    try x.proj(e, s.qrq, w.wq_b, s.q, false);
    e.barrier();
    try g.rope(gp, e, s.q, s.qrope, rows.pos, table, R, c.heads, D, @intCast(x.tables.positions));
    e.barrier();
    const b: attn.Bind = .{
        .q = ctx_.cv(attn.Ref, s.qrope),
        .w = .{ .buf = ring },
        .c = .{ .buf = ring },
        .i = ctx_.cv(attn.Ref, rows.sel),
        .m = ctx_.cv(attn.Ref, rows.meta),
        .sk = ctx_.cv(attn.Ref, w.sink),
        .cs = ctx_.cv(attn.Ref, table),
        .o = ctx_.cv(attn.Ref, s.o),
    };
    try attn.encodeDecode(&x.p.attn, e, b, .{ .rows = R, .heads = c.heads, .out = .bf16 });
    e.barrier();
    try layer_attn.output(x, e, w);
}

/// The route's shape of a stage: the drafter's own experts, picks, and fused gate+route+plan kernel.
pub fn shape(x: *const Ctx) !moe.Shape {
    return .{
        .experts = x.c.dspark_experts,
        .topk = x.c.dspark_topk,
        .route = x.p.route_dspark orelse return error.NoDspark,
        .plan = x.p.gate_plan_dspark,
    };
}

/// One stage as layer.encode runs a layer, with the draft attention and the drafter's MoE.
pub fn stage(x: *const Ctx, e: mtl.ComputeEncoder, w: *const weights.Layer, io: layer.Io, ring: mtl.Buffer, rows: Rows) !void {
    const s = x.s;
    const ma = try layer.mixNorm(x, e, io.x, w.hc_attn, s.mix_a, io.pre, w.attn_norm, .{ s.h, s.hn, s.hq });
    e.barrier();
    layer.mixFin(x, e, ma);
    try attention(x, e, w, ring, rows);
    e.barrier();
    try layer.ffn(x, e, w, io, try shape(x));
}

/// The block's token ids into all four streams of streams[0] (the shared embedding), the pre mix one-hot.
pub fn embed(x: *const Ctx, e: mtl.ComputeEncoder, top: *const weights.Top, b: *const forward.Bufs) !void {
    try g.embed(&x.p.glue, e, top.embed, b.ids, b.streams[0], b.pres[0], x.rows, x.c.hidden, x.c.vocab);
}

/// The last stage's streams collapsed (hf, the confidence head's input), the drafter's norm, its FP8 or the bf16 head.
pub fn head(x: *const Ctx, e: mtl.ComputeEncoder, top: *const weights.Top, d: *const dw.Weights, b: *const forward.Bufs) !void {
    const c = x.c;
    const gp = &x.p.glue;
    const last = d.stages.len % 2;
    cl.hcpre(x.p, e, b.streams[last], b.pres[last], b.hf, x.rows, c.hidden);
    e.barrier();
    g.rmsnorm(gp, e, b.hf, d.norm, b.hfn, x.rows, c.hidden, c.eps);
    e.barrier();
    if (d.head) |h| return x.proj(e, b.hfn, h.fp8, b.logits, true);
    try g.mv(gp, e, b.hfn, top.head, b.logits, x.rows, c.hidden, c.vocab, true);
}
