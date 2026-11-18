//! Rows through every layer in one command buffer: row kernels to 16 rows, tensor-unit matmuls for prompt chunks.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const rows = @import("../../core/row_projection.zig");
const route = @import("../../core/moe_route.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

/// The glue kernels' constants (glue.metal's fl_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, ring: u32 = 0, rope: u32 = 0, eps: f32 = 0, theta: f32 = 0 };

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
        if (n_rows > st.batch_rows) if (x.m.mmFor(w)) |p| {
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

/// All rows' logits or the last's; ids on the GPU; the last row drawn on the GPU; committed unwaited into `flight`.
pub const Out = struct { all: bool = false, timing: ?*Timing = null, ids: ?B = null, draw: ?Draw = null, flight: ?*mtl.CommandBuffer = null };

/// A GPU draw of the last row's logits: argmax, or the keyed sampler at `position` (lanes' gpu_rule and gpu_full).
pub const Draw = struct {
    how: union(enum) { greedy, sampled: struct { sampling: lanes.Sampling, position: u64 } },
    out: B,
};

/// The q, k and v preparation's constants (glue.metal's fl_prep_args).
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, dim: u32, pos: u32, ring: u32, rope: u32, stride: u32, eps: f32, theta: f32 };
/// The decode attention's constants (glue.metal's fl_vec_args).
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, ring: u32, window: u32, blocks: u32, chunk: u32, first: u32, scale: f32 };

/// A forward's host encoding and GPU seconds (tf-flint-run's bench).
pub const Timing = struct { encode: f64 = 0, gpu: f64 = 0 };

/// A stream's rows in a forward: the next `rows` of ids at positions cache.len.., its keys and values into its cache.
pub const Segment = struct { cache: *st.Cache, rows: usize };

/// One stream's `ids` at positions cache.len.. (len unmoved).
pub fn run(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, out: Out) !void {
    return runShared(m, s, &.{.{ .cache = cache, .rows = ids.len }}, ids, out);
}

/// Several streams' rows in one forward, each segment's attention over its own cache: every row keeps its solo bits.
pub fn runShared(m: *const Model, s: *st.Scratch, segments: []const Segment, ids: []const u32, out: Out) !void {
    const c = &m.config;
    const n = ids.len;
    if (n == 0 or n > s.rows or (out.all and n > st.batch_rows) or (!out.all and segments.len != 1)) return error.FlintRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.FlintContextFull;
        if (segments.len > 1 and seg.rows > st.window_rows) return error.FlintRowsExceeded;
        total += seg.rows;
    }
    if (total != n) return error.InvalidFlintSegments;
    if (out.ids == null) {
        for (ids) |id| if (id >= c.vocab) return error.InvalidFlintToken;
        @memcpy(s.ids.slice(u32, n), ids);
    }
    if (out.draw != null and (out.all or segments.len != 1)) return error.InvalidFlintSegments;
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const W = m.weights;
    embed(x, out.ids orelse b(s.ids), n);
    const normed = Args{ .rows = nr, .width = c.hidden, .eps = c.eps };
    x.glue(.norm, normed, &.{ b(s.resid), t(W.layers[0].input_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    for (W.layers[0..c.layers], 0..) |l, i| try layer(x, segments, l, i, n, if (i + 1 < c.layers) W.layers[i + 1].input_norm else null);
    if (out.all) {
        x.glue(.norm, normed, &.{ b(s.resid), t(W.norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(W.head, s.x, 0, s.logits_all, 0, n);
    } else {
        x.glue(.norm, normed, &.{ bo(s.resid, (n - 1) * h * 4), t(W.norm), A, b(s.x) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, true);
        try x.project(W.head, s.x, 0, segments[0].cache.logits, 0, 1);
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
        std.log.err("flint forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (out.timing) |tm| tm.gpu += cb.gpuSeconds();
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// The draw of one logits row on the GPU into `d.out`: fl_argmax, or the keyed sampler one 1024-thread threadgroup.
fn pick(x: Enc, logits: mtl.Buffer, d: Draw) void {
    const e = x.e;
    const vocab: u32 = x.m.config.vocab;
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
    const c = &x.m.config;
    const E = x.m.weights.embed;
    const args = Args{ .rows = @intCast(n), .width = c.hidden, .bits = E.bits, .group = @intCast(E.group) };
    x.glue(.embed, args, &.{ ids, bo(E.w, E.w_off), bo(E.scales, E.s_off), bo(E.biases, E.b_off), A, b(x.s.resid) }, .{ c.hidden, n, 1 }, .{ 256, 1, 1 }, false);
}

/// Layer `i` on every segment's rows (s.x their input norm); its last add norms the residual by `next`.
fn layer(x: Enc, segments: []const Segment, l: wts.Layer, i: usize, n: usize, next: ?wts.Tensor) !void {
    const m = x.m;
    const s = x.s;
    const c = &m.config;
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    const normed = Args{ .rows = nr, .width = c.hidden, .eps = c.eps };
    try x.project(l.attn.qkv, s.x, 0, s.q, 0, n);
    var base: usize = 0;
    for (segments) |seg| {
        try attention(x, s, seg.cache, l.attn, i, base, seg.rows);
        base += seg.rows;
    }
    try x.project(l.attn.o, s.o, 0, s.y, 0, n);
    x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    switch (l.ffn) {
        .dense => |d| {
            try mlp(x, d, s, s.y, n, c.intermediate);
            if (next) |w| {
                x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
            } else x.glue(.add, Args{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
        .moe => |e| {
            var r0: usize = 0;
            while (r0 < n) : (r0 += st.window_rows) { // the router's launches take 16 rows each
                const blk: u32 = @intCast(@min(st.window_rows, n - r0));
                const at = r0 * c.topk * 4;
                route.logits(x.e, m.route, .{ .hidden = c.hidden, .experts = c.experts, .topk = c.topk }, bo(s.x, r0 * h * 2), b(e.router), b(s.logits32), blk);
                x.glue(.route_pick, Args{ .rows = blk, .width = c.experts, .heads = c.topk }, &.{ b(s.logits32), A, bo(s.pick, at), bo(s.wts, at) }, .{ blk, 1, 1 }, .{ 32, 1, 1 }, true);
            }
            const slots = n * c.topk;
            if (n > st.batch_rows) {
                try expertsByExpert(x, e, n);
            } else {
                var gate = e.gate;
                gate.repeat = c.topk;
                var up = e.up;
                up.repeat = c.topk;
                try x.experts(gate, s.x, s.eg, s.pick, slots);
                try x.experts(up, s.x, s.eu, s.pick, slots);
                x.glue(.swiglu, Args{ .rows = @intCast(slots), .width = c.moe_intermediate }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_intermediate, slots, 1 }, .{ 256, 1, 1 }, false);
                try x.experts(e.down, s.ea, s.ey, s.pick, slots);
            }
            try mlp(x, e.shared, s, s.shared, n, c.shared_intermediate);
            const madd = Args{ .rows = nr, .width = c.hidden, .heads = c.topk, .eps = c.eps };
            if (next) |w| {
                x.glue(.moe_add_norm, madd, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
            } else x.glue(.moe_add, madd, &.{ b(s.resid), b(s.ey), b(s.wts), b(s.shared), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
    }
}

/// A chunk's routed experts by expert (each expert's weights read once): sorted, gate/up, SwiGLU, down back to s.ey.
fn expertsByExpert(x: Enc, e: wts.Moe, n: usize) !void {
    const c = &x.m.config;
    const s = x.s;
    const slots: u32 = @intCast(n * c.topk);
    const p = x.m.mmFor(e.gate.one) orelse return error.UnsupportedFlintExperts;
    const pd = x.m.mmFor(e.down.one) orelse return error.UnsupportedFlintExperts;
    const ex: u32 = c.experts;
    x.glue(.sort, Args{ .rows = slots, .width = ex }, &.{ b(s.pick), A, b(s.offsets), b(s.order) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
    x.glue(.rows_by_order, Args{ .rows = slots, .width = c.hidden, .heads = c.topk }, &.{ b(s.x), b(s.order), A, b(s.xs) }, .{ c.hidden, slots, 1 }, .{ 256, 1, 1 }, false);
    affine_mm.rowSums(x.e, p, @intCast(e.gate.one.group), b(s.xs), b(s.sums), slots, c.hidden);
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.gate.one), b(s.offsets), b(s.eg), slots, ex);
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.up.one), b(s.offsets), b(s.eu), slots, ex);
    x.glue(.swiglu, Args{ .rows = slots, .width = c.moe_intermediate }, &.{ b(s.eg), b(s.eu), A, b(s.ea) }, .{ c.moe_intermediate, slots, 1 }, .{ 256, 1, 1 }, false);
    affine_mm.rowSums(x.e, pd, @intCast(e.down.one.group), b(s.ea), b(s.sums), slots, c.moe_intermediate);
    affine_mm.gatherTo(x.e, pd, b(s.ea), b(s.sums), mat(e.down.one), b(s.offsets), b(s.ey), b(s.order), slots, ex);
}

/// gate|up from s.x, SwiGLU, down into `out`.
fn mlp(x: Enc, w: wts.Mlp, s: *st.Scratch, out: mtl.Buffer, n: usize, inter: u32) !void {
    try x.project(w.gu, s.x, 0, s.gate, 0, n);
    x.glue(.swiglu_gu, Args{ .rows = @intCast(n), .width = inter }, &.{ b(s.gate), A, b(s.act) }, .{ inter, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(w.down, s.act, 0, out, 0, n);
}

/// One stream's scratch rows base..: q, k normed (and rotated), k, v into its cache (a ring), attention to s.o's rows.
fn attention(x: Enc, s: *st.Scratch, cache: *st.Cache, w: wts.Attention, i: usize, base: usize, n: usize) !void {
    const c = &x.m.config;
    const lay = c.layer[i];
    const start = cache.len;
    const qrow: usize = @as(usize, c.heads) * c.head_dim * 2; // a row's qn and o bytes
    const xrow: usize = (@as(usize, c.heads) + 2 * c.kv_heads) * c.head_dim; // a row's q|k|v values
    const qn = bo(s.qn, base * qrow);
    const o = bo(s.o, base * qrow);
    const nr: u32 = @intCast(n);
    const ring = cache.ring(c, i);
    // q normed and rotated into s.qn; k (normed, rotated) and v into the cache at slot (start + r) % ring, or start + r
    const pa = PArgs{ .rows = nr, .heads = c.heads, .kv_heads = c.kv_heads, .dim = c.head_dim, .pos = @intCast(start), .ring = @intCast(ring), .rope = @intFromBool(lay.rope), .stride = @intCast(xrow), .eps = c.eps, .theta = c.theta };
    x.glue(.prep, pa, &.{ bo(s.q, base * xrow * 2), t(w.q_norm), t(w.k_norm), A, qn, bo(cache.kv, cache.keys(i)), bo(cache.kv, cache.values(c, i)) }, .{ c.heads + c.kv_heads, n, 1 }, .{ 32, 1, 1 }, true);
    const k: usize = if (lay.kind == .sliding) 0 else 1;
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(c, i));
    const e = x.e;
    if (n > st.window_rows) { // a prompt chunk: 32 rows of a head a tile, over every key the rows see
        e.setPipeline(x.m.prompt_attn[k]);
        for ([_]B{ qn, keys, values, o }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue([2]i32{ @intCast(n), @intCast(start) }, 4);
        e.setBuffer(w.sinks.buffer, w.sinks.offset, 5);
        e.dispatchGroups(mtl.Size.of((n + 31) / 32, c.heads, 1), mtl.Size.of(@as(usize, c.head_dim) / 64 * 32, 1, 1));
        return;
    }
    // decode rows: a key head's query heads a threadgroup over fixed key blocks at absolute places, then the merge
    const chunk: usize = if (lay.kind == .sliding) st.sliding_chunk else st.fullChunk(cache.capacity);
    const first = (if (lay.kind == .sliding) (start + 1) -| c.window else 0) / chunk;
    const blocks = (start + n - 1) / chunk - first + 1;
    if (blocks > st.max_blocks) return error.FlintTooManyBlocks;
    const va = VArgs{ .rows = nr, .pos = @intCast(start), .heads = c.heads, .groups = c.kv_heads, .ring = @intCast(ring), .window = if (lay.kind == .sliding) c.window else 0, .blocks = @intCast(blocks), .chunk = @intCast(chunk), .first = @intCast(first), .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))) };
    e.setPipeline(x.m.pipe(.decode_vec));
    for ([_]B{ qn, keys, values }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
    e.setValue(va, 3);
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 4..) |r, slot| e.setBuffer(r, 0, slot);
    e.dispatchGroups(mtl.Size.of(c.kv_heads, n, blocks), mtl.Size.of(32, c.heads / c.kv_heads, 1));
    e.setPipeline(x.m.pipe(.merge_vec));
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 0..) |r, slot| e.setBuffer(r, 0, slot);
    e.setValue(va, 3);
    e.setBuffer(w.sinks.buffer, w.sinks.offset, 4);
    e.setBuffer(o.buf, o.off, 5);
    e.dispatchGroups(mtl.Size.of(c.heads, n, 1), mtl.Size.of(1024, 1, 1));
}
