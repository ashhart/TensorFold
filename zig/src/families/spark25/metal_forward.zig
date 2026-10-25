//! Spark2.5's rows through every layer in one command buffer: row kernels to 32 rows, tensor-unit matmuls past them.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("metal_state.zig");
const wts = @import("metal_weights.zig");
const rows = @import("../../core/row_projection.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Model = @import("metal_model.zig").Model;
const Glue = @import("metal_model.zig").Glue;

const geo = cfg.spark4b;

/// The glue kernels' constants (glue.metal's sp_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0 };
/// The q, k and v preparation's constants (glue.metal's sp_prep_args).
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos: u32, ring: u32, rot: u32, stride: u32 };
/// The decode attention's constants (glue.metal's sp_vec_args).
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, ring: u32, window: u32, blocks: u32, chunk: u32, first: u32, scale: f32 };

const Enc = struct {
    e: mtl.ComputeEncoder,
    m: *const Model,
    s: *st.Scratch,

    fn glue(x: Enc, g: Glue, args: anytype, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        x.e.setPipeline(x.m.pipe(g));
        for (bufs, 0..) |r, slot| {
            if (r.args) x.e.setValue(args, slot) else x.e.setBuffer(r.buf, r.off, slot);
        }
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    fn project(x: Enc, w: wts.Linear, in: mtl.Buffer, out: mtl.Buffer, out_off: usize, n_rows: usize) !void {
        if (n_rows > st.batch_rows) {
            // a prompt chunk: each group's row sums, then the tensor-unit product (each weight read once a chunk)
            const p = x.m.mm;
            affine_mm.rowSums(x.e, p, @intCast(w.group), b(in), b(x.s.sums), @intCast(n_rows), @intCast(w.k));
            affine_mm.dense(x.e, p, b(in), b(x.s.sums), mat(w), bo(out, out_off), @intCast(n_rows));
            return;
        }
        const call = try rows.call(&x.m.rows, w, n_rows, false);
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, 0, 0);
        x.e.setBuffer(w.w, w.w_off, 1);
        x.e.setBuffer(w.scales, w.s_off, 2);
        x.e.setBuffer(w.biases, w.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, out_off, 5);
        x.e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
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

/// All rows' logits or the last's; ids on the GPU; the last row drawn on the GPU; committed unwaited into `flight`.
pub const Out = struct { all: bool = false, timing: ?*Timing = null, ids: ?B = null, draw: ?Draw = null, flight: ?*mtl.CommandBuffer = null };

/// A GPU draw of the last row's logits: argmax, or the keyed sampler at `position` (lanes' gpu_rule and gpu_full).
pub const Draw = struct {
    how: union(enum) { greedy, sampled: struct { sampling: lanes.Sampling, position: u64 } },
    out: B,
};

/// A forward's host encoding and GPU seconds (tf-spark-run's bench).
pub const Timing = struct { encode: f64 = 0, gpu: f64 = 0 };

/// A stream's rows in a forward: the next `rows` of ids at positions cache.len.., its keys and values into its cache.
pub const Segment = struct { cache: *st.Cache, rows: usize };

/// One stream's `ids` at positions cache.len.. (len unmoved).
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    return runShared(m, s, &.{.{ .cache = cache, .rows = ids.len }}, ids, out);
}

/// Several streams' rows in one forward, each segment's attention over its own cache: every row keeps its solo bits.
pub fn runShared(m: *const Model, s: *st.Scratch, segments: []const Segment, ids: []const u32, out: Out) !void {
    const n = ids.len;
    if (n == 0 or n > s.rows or (out.all and n > st.batch_rows) or (!out.all and segments.len != 1)) return error.SparkRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.SparkContextFull;
        if (segments.len > 1 and seg.rows > st.window_rows) return error.SparkRowsExceeded;
        total += seg.rows;
    }
    if (total != n) return error.InvalidSparkSegments;
    if (out.ids == null) {
        for (ids) |id| if (id >= cfg.vocab) return error.InvalidSparkToken;
        @memcpy(s.ids.slice(u32, n), ids);
    }
    if (out.draw != null and (out.all or segments.len != 1)) return error.InvalidSparkSegments;
    const h = geo.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const W = m.weights;
    embed(x, out.ids orelse b(s.ids), n);
    const normed = Args{ .rows = nr, .width = h, .eps = cfg.eps };
    x.glue(.norm, normed, &.{ b(s.resid), t(W.layers[0].input_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    for (W.layers, 0..) |l, i| try layer(x, segments, l, i, n, if (i + 1 < W.layers.len) W.layers[i + 1].input_norm else null);
    if (out.all) {
        x.glue(.norm, normed, &.{ b(s.resid), t(W.norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(W.embed, s.x, s.logits_all, 0, n);
    } else {
        x.glue(.norm, normed, &.{ bo(s.resid, (n - 1) * h * 4), t(W.norm), A, b(s.x) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(W.embed, s.x, segments[0].cache.logits, 0, 1);
        if (out.draw) |d| pick(x, segments[0].cache.logits, d);
    }
    x.e.end();
    if (out.timing) |tm| tm.encode += seconds() - t0;
    cb.commit();
    if (out.flight) |f| {
        f.* = cb.retain();
        return;
    }
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("spark2_5 forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (out.timing) |tm| tm.gpu += cb.gpuSeconds();
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// The draw of one logits row on the GPU into `d.out`: sp_argmax, or the keyed sampler one 1024-thread threadgroup.
fn pick(x: Enc, logits: mtl.Buffer, d: Draw) void {
    const e = x.e;
    const vocab: u32 = cfg.vocab;
    switch (d.how) {
        .greedy => {
            e.setPipeline(x.m.pipe(.argmax));
            e.setBuffer(logits, 0, 0);
            e.setBuffer(d.out.buf, d.out.off, 1);
            e.setValue(vocab, 2);
        },
        .sampled => |sd| {
            // the kernels index these by row: one row, padded as Nemotron's launches pad them
            const r = lanes.gpu_rule.rule(sd.sampling);
            var seeds: [16]u32 = @splat(0);
            var positions: [8]u32 = @splat(0);
            var cfgs: [32]f32 = @splat(0);
            var caps: [8]u32 = @splat(0);
            seeds[0..2].* = r.seedHalves();
            positions[0] = @intCast(sd.position);
            cfgs[0..4].* = r.cfg();
            caps[0] = r.top_k;
            const whole = lanes.gpu_full.fullVocabulary(sd.sampling);
            e.setPipeline(if (whole) x.m.sample_full else x.m.sample);
            e.setBuffer(logits, 0, 0);
            e.setBytes(std.mem.sliceAsBytes(&seeds), 1);
            e.setBytes(std.mem.sliceAsBytes(&positions), 2);
            e.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
            e.setBytes(std.mem.sliceAsBytes(&caps), 4);
            e.setBuffer(d.out.buf, d.out.off, 5);
            if (whole) {
                e.setBuffer(d.out.buf, d.out.off, 6);
                e.setValue(vocab, 7);
            }
        },
    }
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// Embeddings of the token ids at `ids` (n of them) into the residual.
fn embed(x: Enc, ids: B, n: usize) void {
    const E = x.m.weights.embed;
    const args = Args{ .rows = @intCast(n), .width = geo.hidden, .bits = E.bits, .group = @intCast(E.group) };
    x.glue(.embed, args, &.{ ids, bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(x.s.resid) }, .{ geo.hidden, n, 1 }, .{ 256, 1, 1 }, false);
}

/// Layer `i` on every segment's rows (s.x their input norm); its last add norms the residual by `next`.
fn layer(x: Enc, segments: []const Segment, l: wts.Layer, i: usize, n: usize, next: ?wts.Tensor) !void {
    const s = x.s;
    const h = geo.hidden;
    const nr: u32 = @intCast(n);
    const normed = Args{ .rows = nr, .width = h, .eps = cfg.eps };
    try x.project(l.attn.qkvg, s.x, s.q, 0, n);
    var base: usize = 0;
    for (segments) |seg| {
        try attention(x, seg.cache, i, base, seg.rows);
        base += seg.rows;
    }
    const gated = Args{ .rows = nr, .width = @intCast(st.qkvg_rows), .heads = geo.query_heads };
    x.glue(.gate, gated, &.{ b(s.att), bo(s.q, st.gate_col * 2), A, b(s.o) }, .{ geo.queryWidth(), n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(l.attn.out, s.o, s.y, 0, n);
    x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    try x.project(l.mlp.gu, s.x, s.gu, 0, n);
    x.glue(.gelu_gu, Args{ .rows = nr, .width = geo.intermediate }, &.{ b(s.gu), A, b(s.act) }, .{ geo.intermediate, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(l.mlp.down, s.act, s.y, 0, n);
    if (next) |w| {
        x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    } else x.glue(.add, Args{ .rows = nr, .width = h }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
}

/// One stream's scratch rows base..: q and k rotated, k and v into its cache (a ring if sliding), attention to s.att.
fn attention(x: Enc, cache: *st.Cache, i: usize, base: usize, n: usize) !void {
    const s = x.s;
    const full = cfg.full(i);
    const start = cache.len;
    const qrow: usize = geo.queryWidth() * 2; // a row's qn and att bytes
    const qn = bo(s.qn, base * qrow);
    const o = bo(s.att, base * qrow);
    const nr: u32 = @intCast(n);
    const ring = cache.ring(i);
    const pa = PArgs{ .rows = nr, .heads = geo.query_heads, .kv_heads = geo.kv_heads, .pos = @intCast(start), .ring = @intCast(ring), .rot = if (full) cfg.full_rotated else cfg.head_dim, .stride = @intCast(st.qkvg_rows) };
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(i));
    x.glue(.prep, pa, &.{ bo(s.q, base * st.qkvg_rows * 2), b(x.m.inv[@intFromBool(full)]), A, qn, keys, values }, .{ geo.query_heads + geo.kv_heads, n, 1 }, .{ 32, 1, 1 }, true);
    const e = x.e;
    if (n > st.window_rows) { // a prompt chunk: 32 rows of a head a tile, over every key the rows see
        e.setPipeline(x.m.prompt_attn[@intFromBool(full)]);
        for ([_]B{ qn, keys, values, o }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue([2]i32{ @intCast(n), @intCast(start) }, 4);
        e.dispatchGroups(mtl.Size.of((n + 31) / 32, geo.query_heads, 1), mtl.Size.of(cfg.head_dim / 64 * 32, 1, 1));
        return;
    }
    // decode rows: a key head's query heads a threadgroup over fixed key blocks at absolute places, then the merge
    const chunk: usize = if (full) st.fullChunk(cache.capacity) else st.sliding_chunk;
    const first = (if (full) 0 else (start + 1) -| cfg.window) / chunk;
    const blocks = (start + n - 1) / chunk - first + 1;
    if (blocks > st.max_blocks) return error.SparkTooManyBlocks;
    const va = VArgs{ .rows = nr, .pos = @intCast(start), .heads = geo.query_heads, .groups = geo.kv_heads, .ring = @intCast(ring), .window = if (full) 0 else cfg.window, .blocks = @intCast(blocks), .chunk = @intCast(chunk), .first = @intCast(first), .scale = cfg.scale };
    e.setPipeline(x.m.pipe(.decode_vec));
    for ([_]B{ qn, keys, values }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
    e.setValue(va, 3);
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 4..) |r, slot| e.setBuffer(r, 0, slot);
    e.dispatchGroups(mtl.Size.of(geo.kv_heads, n, blocks), mtl.Size.of(32, geo.query_heads / geo.kv_heads, 1));
    e.setPipeline(x.m.pipe(.merge_vec));
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 0..) |r, slot| e.setBuffer(r, 0, slot);
    e.setValue(va, 3);
    e.setBuffer(o.buf, o.off, 4);
    e.dispatchGroups(mtl.Size.of(geo.query_heads, n, 1), mtl.Size.of(1024, 1, 1));
}
