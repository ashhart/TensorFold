//! Rows through every layer in one command buffer: row kernels to 32 rows, tensor-unit matmuls for prompt chunks.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const vision = @import("vision.zig");
const wts = @import("weights.zig");
const rows = @import("../../core/row_projection.zig");
const affine_mm = @import("../../core/affine_mm.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

/// The glue kernels' constants (glue.metal's mm_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, dim: u32 = 0, pos: u32 = 0, stride: u32 = 0, bits: u32 = 0, group: u32 = 0, eps: f32 = 0, alpha: f32 = 0, limit: f32 = 0, scale: f32 = 0 };
/// The q, k and index preparation's constants (mm_prep_args).
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, index_heads: u32, dim: u32, pos: u32, rotary: u32, stride: u32, eps: f32, theta: f32 };
/// The indexer's constants (mm_index_args).
const IArgs = extern struct { rows: u32, pos: u32, heads: u32, blocks: u32, first_row: u32 = 0 };
/// The decode attention's constants (mm_vec_args).
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, topk: u32, scale: f32, sel: u32, all: u32 };

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
        return projectWith(x.e, x.m, x.s.sums, w, bo(in, in_off), bo(out, out_off), n_rows);
    }

    fn experts(x: Enc, w: rows.Experts, in: mtl.Buffer, out: mtl.Buffer, ids: mtl.Buffer, slots: usize) !void {
        const call = try rows.callIndexed(&x.m.rows, w, slots, false);
        if (w.one.bits == 3 or w.one.bits == 4) { // expert by expert: a weight read once for every row that picked it
            x.e.setPipeline(x.m.grouped[if (w.one.bits == 3) 0 else 1]);
            for ([_]mtl.Buffer{ in, w.one.w, w.one.scales, w.one.biases }, [_]usize{ 0, w.one.w_off, w.one.s_off, w.one.b_off }, 0..) |r, off, slot| x.e.setBuffer(r, off, slot);
            x.e.setValue(call.dims, 4);
            x.e.setBuffer(out, 0, 5);
            x.e.setBuffer(x.s.groups, 0, 6);
            x.e.setBuffer(x.s.members, 0, 7);
            x.e.dispatchGroups(mtl.Size.of(call.groups, slots, 1), mtl.Size.of(call.threads, 1, 1));
            return;
        }
        _ = ids;
        x.e.setPipeline(call.pipeline);
        x.e.setBuffer(in, 0, 0);
        x.e.setBuffer(w.one.w, w.one.w_off, 1);
        x.e.setBuffer(w.one.scales, w.one.s_off, 2);
        x.e.setBuffer(w.one.biases, w.one.b_off, 3);
        x.e.setValue(call.dims, 4);
        x.e.setBuffer(out, 0, 5);
        x.e.setBuffer(x.s.pick, 0, 6);
        x.e.dispatchGroups(mtl.Size.of(call.groups, slots, 1), mtl.Size.of(call.threads, 1, 1));
    }
};

/// `w` on `n_rows` rows of `in` into `out`: row kernels to batch_rows, else the tensor units (group sums in `sums`).
pub fn projectWith(e: mtl.ComputeEncoder, m: *const Model, sums: mtl.Buffer, w: wts.Linear, in: B, out: B, n_rows: usize) !void {
    return projectRows(e, m, sums, w, in, out, n_rows, st.batch_rows + 1);
}

/// projectWith with the tensor units from `tensor_from` rows (a drafter's proposals need not keep one-row bits).
pub fn projectRows(e: mtl.ComputeEncoder, m: *const Model, sums: mtl.Buffer, w: wts.Linear, in: B, out: B, n_rows: usize, tensor_from: usize) !void {
    if (n_rows >= tensor_from) if (m.mmFor(w)) |p| {
        // a prompt chunk: each group's row sums, then the tensor-unit product (each weight read once a chunk)
        affine_mm.rowSums(e, p, @intCast(w.group), in, b(sums), @intCast(n_rows), @intCast(w.k));
        affine_mm.dense(e, p, in, b(sums), mat(w), out, @intCast(n_rows));
        return;
    };
    const call = try rows.call(&m.rows, w, n_rows, false);
    e.setPipeline(call.pipeline);
    e.setBuffer(in.buf, in.off, 0);
    e.setBuffer(w.w, w.w_off, 1);
    e.setBuffer(w.scales, w.s_off, 2);
    e.setBuffer(w.biases, w.b_off, 3);
    e.setValue(call.dims, 4);
    e.setBuffer(out.buf, out.off, 5);
    e.dispatchGroups(mtl.Size.of(call.groups, 1, 1), mtl.Size.of(call.threads, 1, 1));
}

pub const B = struct { buf: mtl.Buffer = undefined, off: usize = 0, args: bool = false };
pub fn b(buf: mtl.Buffer) B {
    return .{ .buf = buf };
}
pub fn bo(buf: mtl.Buffer, off: usize) B {
    return .{ .buf = buf, .off = off };
}
pub fn t(x: wts.Tensor) B {
    return .{ .buf = x.buffer, .off = x.offset };
}
pub const A = B{ .args = true };

/// A matrix as affine_mm takes it.
fn mat(w: wts.Linear) struct { w: B, s: B, b: B, n: u32, k: u32 } {
    return .{ .w = bo(w.w, w.w_off), .s = bo(w.scales, w.s_off), .b = bo(w.biases, w.b_off), .n = @intCast(w.n), .k = @intCast(w.k) };
}

/// All rows' logits or the last's; ids on the GPU; the last row drawn on the GPU; committed unwaited into `flight`.
pub const Out = struct { all: bool = false, timing: ?*Timing = null, ids: ?B = null, draw: ?Draw = null, flight: ?*mtl.CommandBuffer = null, taps: ?B = null, images: []const Span = &.{} };

/// An image's rows (bf16) over prompt positions [at, at + n); attention is causal, so a chunk may end inside one.
pub const Span = struct { at: usize, n: u32, rows: vision.Ref };

/// A GPU draw of the last row's logits: argmax, or the keyed sampler at `position` (lanes' gpu_rule and gpu_full).
pub const Draw = struct {
    how: union(enum) { greedy, sampled: struct { sampling: lanes.Sampling, position: u64 } },
    out: B,
};

/// A forward's host encoding and GPU seconds (tf-minimax-run's bench).
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
    if (n == 0 or n > s.rows or (out.all and n > st.batch_rows) or (!out.all and segments.len != 1)) return error.MiniMaxRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity or seg.cache.capacity > s.context) return error.MiniMaxContextFull;
        if (segments.len > 1 and seg.rows > st.window_rows) return error.MiniMaxRowsExceeded;
        total += seg.rows;
    }
    if (total != n) return error.InvalidMiniMaxSegments;
    if (out.ids == null) {
        for (ids) |id| if (id >= c.vocab) return error.InvalidMiniMaxToken;
        @memcpy(s.ids.slice(u32, n), ids);
    }
    if (out.draw != null and (out.all or segments.len != 1)) return error.InvalidMiniMaxSegments;
    const h: usize = c.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const W = m.weights;
    embed(x, out.ids orelse b(s.ids), n);
    if (out.images.len > 0) {
        if (segments.len != 1) return error.InvalidMiniMaxSegments;
        const p0 = segments[0].cache.len;
        for (out.images) |span| {
            const lo = @max(span.at, p0);
            const hi = @min(span.at + span.n, p0 + n);
            if (lo >= hi) continue;
            const v = m.vision orelse return error.NoMiniMaxVision;
            v.inject(x.e, span.rows.at((lo - span.at) * h * 2), .{ .buf = s.resid, .off = (lo - p0) * h * 4 }, @intCast(hi - lo));
        }
    }
    const normed = Args{ .rows = nr, .width = c.hidden, .eps = c.eps };
    x.glue(.norm, normed, &.{ b(s.resid), t(W.layers[0].input_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
    for (W.layers[0..c.layers], 0..) |l, i| {
        try layer(x, segments, l, i, n, if (i + 1 < c.layers) W.layers[i + 1].input_norm else null);
        // the residual after a tapped layer, every row, into the drafter's features
        if (out.taps) |tb| if (m.tap_at[i] >= 0) {
            x.e.setPipeline(m.tap);
            x.e.setBuffer(s.resid, 0, 0);
            x.e.setValue([4]u32{ nr, c.hidden, m.taps, @intCast(m.tap_at[i]) }, 1);
            x.e.setBuffer(tb.buf, tb.off, 2);
            x.e.dispatchThreads(mtl.Size.of(c.hidden, n, 1), mtl.Size.of(256, 1, 1));
        };
    }
    if (out.all) {
        x.glue(.norm, normed, &.{ b(s.resid), t(W.norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
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
        std.log.err("minimax_m3 forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (out.timing) |tm| tm.gpu += cb.gpuSeconds();
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// The draw of one logits row on the GPU into `d.out`: mm_argmax, or the keyed sampler one 1024-thread threadgroup.
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
        try attention(x, seg.cache, l.attn, i, base, seg.rows);
        base += seg.rows;
    }
    try x.project(l.attn.o, s.o, 0, s.y, 0, n);
    x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
    switch (l.ffn) {
        .dense => |d| {
            try mlp(x, d, s.y, n, c.dense_intermediate);
            if (next) |w| {
                x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
            } else x.glue(.add, Args{ .rows = nr, .width = c.hidden }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
        .moe => |e| {
            x.glue(.router, Args{ .rows = nr, .width = c.hidden, .heads = c.experts }, &.{ b(s.x), t(e.router), A, b(s.logits32) }, .{ c.experts, (n + 15) / 16, 1 }, .{ 256, 1, 1 }, true);
            // a decode window's picks and its slots by expert in one threadgroup; a prompt chunk's picks a row each
            if (n > st.batch_rows) {
                x.glue(.route_pick, Args{ .rows = nr, .width = c.experts, .heads = c.topk }, &.{ b(s.logits32), t(e.bias), A, b(s.pick), b(s.wts) }, .{ n, 1, 1 }, .{ 32, 1, 1 }, true);
            } else x.glue(.route_groups, Args{ .rows = nr, .width = c.experts, .heads = c.topk }, &.{ b(s.logits32), t(e.bias), A, b(s.pick), b(s.wts), b(s.groups), b(s.members) }, .{ 1, 1, 1 }, .{ 32 * n, 1, 1 }, true);
            const slots = n * c.topk;
            const inter: u32 = c.moe_intermediate;
            const act = Args{ .rows = @intCast(slots), .width = inter, .stride = 2 * inter, .alpha = c.swiglu_alpha, .limit = c.swiglu_limit };
            if (n > st.batch_rows) {
                try expertsByExpert(x, e, n, act);
            } else {
                var gu = e.gu;
                gu.repeat = c.topk;
                try x.experts(gu, s.x, s.egu, s.pick, slots);
                x.glue(.swiglu, act, &.{ b(s.egu), A, b(s.ea) }, .{ inter, slots, 1 }, .{ 256, 1, 1 }, false);
                try x.experts(e.down, s.ea, s.ey, s.pick, slots);
            }
            try mlp(x, e.shared, s.shared, n, c.shared_intermediate);
            const madd = Args{ .rows = nr, .width = c.hidden, .heads = c.topk, .eps = c.eps, .scale = c.routed_scale };
            if (next) |w| {
                x.glue(.moe_add_norm, madd, &.{ b(s.resid), b(s.ey), b(s.pick), b(s.wts), b(s.shared), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ 1024, 1, 1 }, true);
            } else x.glue(.moe_add, madd, &.{ b(s.resid), b(s.ey), b(s.pick), b(s.wts), b(s.shared), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
        },
    }
}

/// A chunk's routed experts by expert (each expert's weights read once): sorted, gate|up, SwiGLU, down back to s.ey.
fn expertsByExpert(x: Enc, e: wts.Moe, n: usize, act: Args) !void {
    const c = &x.m.config;
    const s = x.s;
    const slots: u32 = @intCast(n * c.topk);
    const p = x.m.mmFor(e.gu.one) orelse return error.UnsupportedMiniMaxExperts;
    const pd = x.m.mmFor(e.down.one) orelse return error.UnsupportedMiniMaxExperts;
    const ex: u32 = c.experts;
    x.glue(.sort, Args{ .rows = slots, .width = ex }, &.{ b(s.pick), A, b(s.offsets), b(s.order) }, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, true);
    x.glue(.rows_by_order, Args{ .rows = slots, .width = c.hidden, .heads = c.topk }, &.{ b(s.x), b(s.order), A, b(s.xs) }, .{ c.hidden, slots, 1 }, .{ 256, 1, 1 }, false);
    affine_mm.rowSums(x.e, p, @intCast(e.gu.one.group), b(s.xs), b(s.sums), slots, c.hidden);
    affine_mm.gather(x.e, p, b(s.xs), b(s.sums), mat(e.gu.one), b(s.offsets), b(s.egu), slots, ex);
    x.glue(.swiglu, act, &.{ b(s.egu), A, b(s.ea) }, .{ c.moe_intermediate, slots, 1 }, .{ 256, 1, 1 }, false);
    affine_mm.rowSums(x.e, pd, @intCast(e.down.one.group), b(s.ea), b(s.sums), slots, c.moe_intermediate);
    affine_mm.gatherTo(x.e, pd, b(s.ea), b(s.sums), mat(e.down.one), b(s.offsets), b(s.ey), b(s.order), slots, ex);
}

/// gate|up from s.x, SwiGLU, down into `out`.
fn mlp(x: Enc, w: wts.Mlp, out: mtl.Buffer, n: usize, inter: u32) !void {
    const c = &x.m.config;
    const s = x.s;
    try x.project(w.gu, s.x, 0, s.gate, 0, n);
    const act = Args{ .rows = @intCast(n), .width = inter, .stride = 2 * inter, .alpha = c.swiglu_alpha, .limit = c.swiglu_limit };
    x.glue(.swiglu, act, &.{ b(s.gate), A, b(s.act) }, .{ inter, n, 1 }, .{ 256, 1, 1 }, false);
    try x.project(w.down, s.act, 0, out, 0, n);
}

/// One stream's rows base..: q, k, the index pair normed and rotated, the block selection, attention to s.o.
fn attention(x: Enc, cache: *st.Cache, w: wts.Attention, i: usize, base: usize, n: usize) !void {
    const c = &x.m.config;
    const s = x.s;
    const start = cache.len;
    const qrow: usize = @as(usize, c.heads) * c.head_dim * 2; // a row's qn and o bytes
    const irow: usize = @as(usize, c.index_heads) * c.index_dim * 2;
    const qn = bo(s.qn, base * qrow);
    const iq = bo(s.iq, base * irow);
    const o = bo(s.o, base * qrow);
    const nr: u32 = @intCast(n);
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(c, i));
    const ikeys = bo(cache.kv, cache.indexKeys(c, i));
    // an int8 cache's scales (a bf16 cache's kernels never read them: the keys stand in)
    const kscale = if (c.kv8) bo(cache.kv, cache.keyScales(c, i)) else keys;
    const vscale = if (c.kv8) bo(cache.kv, cache.valueScales(c, i)) else keys;
    const pa = PArgs{ .rows = nr, .heads = c.heads, .kv_heads = c.kv_heads, .index_heads = c.index_heads, .dim = c.head_dim, .pos = @intCast(start), .rotary = c.rotary, .stride = c.qkvRows(), .eps = c.eps, .theta = c.theta };
    x.glue(if (c.kv8) .prep8 else .prep, pa, &.{ bo(s.q, base * c.qkvRows() * 2), t(w.q_norm), t(w.k_norm), t(w.iq_norm), t(w.ik_norm), A, qn, iq, keys, values, ikeys, kscale, vscale }, .{ c.heads + c.kv_heads + c.index_heads + 1, n, 1 }, .{ 32, 1, 1 }, true);
    // the selection: every block while a row sees at most topk_blocks of them, else the indexer's best and its own
    const nb = c.blocks(start + n);
    const prompt = n > st.window_rows;
    const sel_at = base * c.kv_heads;
    const scores = bo(s.scores, sel_at * c.blocks(s.context) * 4);
    const sel = bo(s.sel, sel_at * c.topk_blocks * 4);
    const ia = IArgs{ .rows = nr, .pos = @intCast(start), .heads = c.kv_heads, .blocks = @intCast(nb) };
    if (nb <= c.topk_blocks) {} else if (n > st.batch_rows) { // a prompt chunk: the tensor units
        x.e.setPipeline(x.m.index_nax);
        for ([_]B{ iq, ikeys, scores }, 0..) |r, slot| x.e.setBuffer(r.buf, r.off, slot);
        x.e.setValue([3]i32{ @intCast(n), @intCast(start), @intCast(nb) }, 3);
        x.e.dispatchGroups(mtl.Size.of((n + 31) / 32, c.index_heads, 1), mtl.Size.of(@as(usize, c.index_dim) / 64 * 32, 1, 1));
    } else x.glue(.index_scores, ia, &.{ iq, ikeys, A, scores }, .{ nb, (n + 15) / 16, 1 }, .{ 128, 1, 1 }, true);
    const words: u32 = if (prompt) @intCast(st.selWords(c)) else 0;
    const bits = bo(s.bits, if (prompt) sel_at * words * 4 else 0);
    const e = x.e;
    // rows that see no more than topk_blocks blocks keep them all: decode needs no selection then
    const all = !prompt and nb <= c.topk_blocks;
    if (!all) {
        e.setPipeline(x.m.pipe(.index_select));
        e.setBuffer(scores.buf, scores.off, 0);
        e.setValue(ia, 1);
        e.setValue(c.topk_blocks, 2);
        e.setBuffer(sel.buf, sel.off, 3);
        e.setBuffer(bits.buf, bits.off, 4);
        e.setValue(words, 5);
        e.dispatchGroups(mtl.Size.of(c.kv_heads, n, 1), mtl.Size.of(32, 1, 1));
    }
    if (prompt) { // a prompt chunk: a key head's query heads on two rows a tile, over the blocks either row keeps
        e.setPipeline(x.m.prompt_attn);
        for ([_]B{ qn, keys, values, o, A, bits, kscale, vscale }, 0..) |r, slot| {
            if (r.args) e.setValue([2]i32{ @intCast(n), @intCast(start) }, slot) else e.setBuffer(r.buf, r.off, slot);
        }
        const per: usize = 32 / (c.heads / c.kv_heads); // rows a tile
        e.dispatchGroups(mtl.Size.of((n + per - 1) / per, c.kv_heads, 1), mtl.Size.of(@as(usize, c.head_dim) / 64 * 32, 1, 1));
        return;
    }
    // decode rows: a key head's query heads a threadgroup over each kept block, then the merge in block order
    const va = VArgs{ .rows = nr, .pos = @intCast(start), .heads = c.heads, .groups = c.kv_heads, .topk = c.topk_blocks * 4, .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))), .sel = c.topk_blocks, .all = @intFromBool(all) };
    e.setPipeline(x.m.pipe(if (c.kv8) .decode_vec8 else .decode_vec));
    for ([_]B{ qn, keys, values }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
    e.setValue(va, 3);
    e.setBuffer(sel.buf, sel.off, 4);
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 5..) |r, slot| e.setBuffer(r, 0, slot);
    e.setBuffer(kscale.buf, kscale.off, 8);
    e.setBuffer(vscale.buf, vscale.off, 9);
    e.dispatchGroups(mtl.Size.of(c.kv_heads, n, c.topk_blocks * 4), mtl.Size.of(32, c.heads / c.kv_heads, 1));
    e.setPipeline(x.m.pipe(.merge_vec));
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 0..) |r, slot| e.setBuffer(r, 0, slot);
    e.setValue(va, 3);
    e.setBuffer(o.buf, o.off, 4);
    e.dispatchGroups(mtl.Size.of(c.heads, n, 1), mtl.Size.of(32, 1, 1));
}
