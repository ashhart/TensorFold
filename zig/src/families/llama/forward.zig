//! Llama's rows through every layer in one command buffer: row kernels to 32 rows, tensor-unit matmuls past them.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Model = @import("model.zig").Model;
const Glue = @import("model.zig").Glue;

/// The glue kernels' constants (glue.metal's ll_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, heads: u32 = 0, eps: f32 = 0 };
/// The q, k and v preparation's constants (glue.metal's ll_prep_args).
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos: u32, stride: u32 };
/// The bf16 row projection's constants (glue.metal's ll_mv_args).
const MArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
/// The decode attention's constants (glue.metal's ll_vec_args).
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, blocks: u32, chunk: u32, first: u32, scale: f32 };

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

    /// y [rows, d.n] (rows `ldy` apart, at byte `out_off`) = x W^T: the GEMM a prompt chunk, the row kernels a window.
    fn dense(x: Enc, d: wts.Dense, in: mtl.Buffer, out: mtl.Buffer, out_off: usize, n_rows: usize, ldy: usize) void {
        const e = x.e;
        if (d.int4) |q| {
            const ma = MArgs{ .rows = @intCast(n_rows), .n = @intCast(d.n), .k = @intCast(d.k), .ldy = @intCast(ldy) };
            if (n_rows <= st.batch_rows) {
                const variant: Glue = if (n_rows <= 1) .q4_matvec_1 else if (n_rows <= 2) .q4_matvec_2 else if (n_rows <= 4) .q4_matvec_4 else if (n_rows <= 8) .q4_matvec_8 else if (n_rows <= 16) .q4_matvec_16 else .q4_matvec_32;
                x.glue(variant, ma, &.{ b(in), bo(d.buf, d.off), t(q.scales), t(q.zeros), A, bo(out, out_off) }, .{ (d.n + 7) / 8, 1, 1 }, .{ 256, 1, 1 }, true);
                return;
            }
            // a prompt chunk: W as bf16 (as the decompressor makes it) into the scratch, then the bf16 GEMM on it
            x.glue(.q4_dequant, ma, &.{ bo(d.buf, d.off), t(q.scales), t(q.zeros), A, b(x.s.wq) }, .{ d.k / 8, d.n, 1 }, .{ 128, 2, 1 }, false);
            return x.gemmOn(.{ .buf = x.s.wq, .off = 0, .n = d.n, .k = d.k }, in, out, out_off, n_rows, ldy);
        }
        if (n_rows <= st.batch_rows) {
            // the variant holding the rows in registers: 1, 2, 4, 8, 16 or 32
            const variant: Glue = if (n_rows <= 1) .matvec_1 else if (n_rows <= 2) .matvec_2 else if (n_rows <= 4) .matvec_4 else if (n_rows <= 8) .matvec_8 else if (n_rows <= 16) .matvec_16 else .matvec_32;
            e.setPipeline(x.m.pipe(variant));
            e.setBuffer(in, 0, 0);
            e.setBuffer(d.buf, d.off, 1);
            e.setValue(MArgs{ .rows = @intCast(n_rows), .n = @intCast(d.n), .k = @intCast(d.k), .ldy = @intCast(ldy) }, 2);
            e.setBuffer(out, out_off, 3);
            e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        x.gemmOn(d, in, out, out_off, n_rows, ldy);
    }

    /// The tensor-unit GEMM of a prompt chunk on bf16 d.
    fn gemmOn(x: Enc, d: wts.Dense, in: mtl.Buffer, out: mtl.Buffer, out_off: usize, n_rows: usize, ldy: usize) void {
        const e = x.e;
        const M: usize = n_rows;
        const tn = (d.n + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(x.m.gemm[@intFromBool(M % 64 == 0)][@intFromBool(d.n % 128 == 0)]);
        e.setBuffer(in, 0, 0);
        e.setBuffer(d.buf, d.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, d.n, d.k, d.k, d.k, ldy, tn, tm, 2, d.k / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(out, out_off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
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

/// All rows' logits or the last's; ids on the GPU; the last row drawn on the GPU; committed unwaited into `flight`.
pub const Out = struct { all: bool = false, timing: ?*Timing = null, ids: ?B = null, draw: ?Draw = null, flight: ?*mtl.CommandBuffer = null };

/// A GPU draw of the last row's logits: argmax, or the keyed sampler at `position` (lanes' gpu_rule and gpu_full).
pub const Draw = struct {
    how: union(enum) { greedy, sampled: struct { sampling: lanes.Sampling, position: u64 } },
    out: B,
};

/// A forward's host encoding and GPU seconds (tf-llama-run's bench).
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
    const c = &m.config;
    if (n == 0 or n > s.rows or (out.all and n > st.batch_rows) or (!out.all and segments.len != 1)) return error.LlamaRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.LlamaContextFull;
        if (segments.len > 1 and seg.rows > st.window_rows) return error.LlamaRowsExceeded;
        total += seg.rows;
    }
    if (total != n) return error.InvalidLlamaSegments;
    if (out.ids == null) {
        for (ids) |id| if (id >= c.vocab) return error.InvalidLlamaToken;
        @memcpy(s.ids.slice(u32, n), ids);
    }
    if (out.draw != null and (out.all or segments.len != 1)) return error.InvalidLlamaSegments;
    const h = c.hidden;
    const nr: u32 = @intCast(n);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const W = m.weights;
    x.glue(.embed, Args{ .rows = nr, .width = @intCast(h) }, &.{ out.ids orelse b(s.ids), bo(W.embed.buf, W.embed.off), A, b(s.resid) }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    const normed = Args{ .rows = nr, .width = @intCast(h), .eps = c.eps };
    x.glue(.norm, normed, &.{ b(s.resid), t(W.layers[0].input_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    for (W.layers, 0..) |l, i| try layer(x, segments, l, i, n, if (i + 1 < W.layers.len) W.layers[i + 1].input_norm else null);
    if (out.all) {
        x.glue(.norm, normed, &.{ b(s.resid), t(W.norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
        x.dense(W.head, s.x, s.logits_all, 0, n, c.vocab);
    } else {
        x.glue(.norm, normed, &.{ bo(s.resid, (n - 1) * h * 2), t(W.norm), A, b(s.x) }, .{ 1, 1, 1 }, .{ 256, 1, 1 }, true);
        x.dense(W.head, s.x, segments[0].cache.logits, 0, 1, c.vocab);
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
        std.log.err("llama forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (out.timing) |tm| tm.gpu += cb.gpuSeconds();
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}

/// The draw of one logits row on the GPU into `d.out`: ll_argmax, or tf_sample_full one 1024-thread threadgroup.
fn pick(x: Enc, logits: mtl.Buffer, d: Draw) void {
    const e = x.e;
    const vocab: u32 = @intCast(x.m.config.vocab);
    switch (d.how) {
        .greedy => {
            e.setPipeline(x.m.pipe(.argmax));
            e.setBuffer(logits, 0, 0);
            e.setBuffer(d.out.buf, d.out.off, 1);
            e.setValue(vocab, 2);
        },
        .sampled => |sd| {
            // the kernel indexes these by row: one row, padded as Nemotron's launches pad them
            const r = lanes.gpu_rule.rule(sd.sampling);
            var seeds: [16]u32 = @splat(0);
            var positions: [8]u32 = @splat(0);
            var cfgs: [32]f32 = @splat(0);
            var caps: [8]u32 = @splat(0);
            seeds[0..2].* = r.seedHalves();
            positions[0] = @intCast(sd.position);
            cfgs[0..4].* = r.cfg();
            caps[0] = r.top_k;
            e.setPipeline(x.m.sample_full);
            e.setBuffer(logits, 0, 0);
            e.setBytes(std.mem.sliceAsBytes(&seeds), 1);
            e.setBytes(std.mem.sliceAsBytes(&positions), 2);
            e.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
            e.setBytes(std.mem.sliceAsBytes(&caps), 4);
            e.setBuffer(d.out.buf, d.out.off, 5);
            e.setBuffer(d.out.buf, d.out.off, 6);
            e.setValue(vocab, 7);
        },
    }
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// Layer `i` on every segment's rows (s.x their input norm); its last add norms the residual by `next`.
fn layer(x: Enc, segments: []const Segment, l: wts.Layer, i: usize, n: usize, next: ?wts.Tensor) !void {
    const s = x.s;
    const c = &x.m.config;
    const h = c.hidden;
    const nr: u32 = @intCast(n);
    const normed = Args{ .rows = nr, .width = @intCast(h), .eps = c.eps };
    const qkvw = c.qkvWidth();
    x.dense(l.attn.q, s.x, s.q, 0, n, qkvw);
    x.dense(l.attn.k, s.x, s.q, c.queryWidth() * 2, n, qkvw);
    x.dense(l.attn.v, s.x, s.q, (c.queryWidth() + c.kvWidth()) * 2, n, qkvw);
    var base: usize = 0;
    for (segments) |seg| {
        try attention(x, seg.cache, i, base, seg.rows);
        base += seg.rows;
    }
    x.dense(l.attn.o, s.att, s.y, 0, n, h);
    x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(l.post_norm), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    x.dense(l.mlp.gate, s.x, s.gu, 0, n, 2 * c.intermediate);
    x.dense(l.mlp.up, s.x, s.gu, c.intermediate * 2, n, 2 * c.intermediate);
    x.glue(.silu_gu, Args{ .rows = nr, .width = @intCast(c.intermediate) }, &.{ b(s.gu), A, b(s.act) }, .{ c.intermediate, n, 1 }, .{ 256, 1, 1 }, false);
    x.dense(l.mlp.down, s.act, s.y, 0, n, h);
    if (next) |w| {
        x.glue(.add_norm, normed, &.{ b(s.resid), b(s.y), t(w), A, b(s.x) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    } else x.glue(.add, Args{ .rows = nr, .width = @intCast(h) }, &.{ b(s.resid), b(s.y), A }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
}

/// One stream's scratch rows base..: q and k rotated, k and v into its cache at their positions, attention to s.att.
fn attention(x: Enc, cache: *st.Cache, i: usize, base: usize, n: usize) !void {
    const s = x.s;
    const c = &x.m.config;
    const start = cache.len;
    const qrow: usize = c.queryWidth() * 2; // a row's qn and att bytes
    const qn = bo(s.qn, base * qrow);
    const o = bo(s.att, base * qrow);
    const nr: u32 = @intCast(n);
    const pa = PArgs{ .rows = nr, .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .pos = @intCast(start), .stride = @intCast(c.qkvWidth()) };
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(i));
    x.glue(.prep, pa, &.{ bo(s.q, base * c.qkvWidth() * 2), b(x.m.inv), A, qn, keys, values }, .{ c.heads + c.kv_heads, n, 1 }, .{ 32, 1, 1 }, true);
    const e = x.e;
    if (n > st.window_rows) { // a prompt chunk: 32 rows of a head a tile, over every key the rows see
        e.setPipeline(x.m.prompt_attn);
        for ([_]B{ qn, keys, values, o }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue([2]i32{ @intCast(n), @intCast(start) }, 4);
        e.dispatchGroups(mtl.Size.of((n + 31) / 32, c.heads, 1), mtl.Size.of(cfg.head_dim / 64 * 32, 1, 1));
        return;
    }
    // decode rows: a key head's query heads a threadgroup over fixed key blocks at absolute places, then the merge
    const chunk = st.chunk(cache.capacity);
    const blocks = (start + n - 1) / chunk + 1;
    if (blocks > st.max_blocks) return error.LlamaTooManyBlocks;
    const va = VArgs{ .rows = nr, .pos = @intCast(start), .heads = @intCast(c.heads), .groups = @intCast(c.kv_heads), .blocks = @intCast(blocks), .chunk = @intCast(chunk), .first = 0, .scale = c.scale() };
    e.setPipeline(x.m.pipe(.decode_vec));
    for ([_]B{ qn, keys, values }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
    e.setValue(va, 3);
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 4..) |r, slot| e.setBuffer(r, 0, slot);
    e.dispatchGroups(mtl.Size.of(c.kv_heads, n, blocks), mtl.Size.of(32, c.group(), 1));
    e.setPipeline(x.m.pipe(.merge_vec));
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 0..) |r, slot| e.setBuffer(r, 0, slot);
    e.setValue(va, 3);
    e.setBuffer(o.buf, o.off, 4);
    e.dispatchGroups(mtl.Size.of(c.heads, n, 1), mtl.Size.of(1024, 1, 1));
}
