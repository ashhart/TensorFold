//! Ripple's rows through every layer in one command buffer: row kernels to 32 rows, tensor-unit matmuls past them.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Model = @import("model.zig").Model;

/// The glue kernels' constants (glue.metal's rc_args).
const Args = extern struct { rows: u32 = 0, width: u32 = 0, flags: u32 = 0, eps: f32 = 0, scale: f32 = 0, blend: f32 = 0 };
const add_flag: u32 = 1; // rc_add_norm: the sublayer's output normed and added first
const lerp_flag: u32 = 2; // rc_add_norm: v's input blended from the embeddings too
/// q and k's norm and RoPE, into the cache (glue.metal's rc_prep_args).
const PArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, pos: u32, stride: u32, ring: u32, rope: u32, eps: f32 };
/// The NVFP4 row kernels' and dequantization's constants (fp4.metal's rc_fp4_args).
const Fp4Args = extern struct { rows: u32, n: u32, k: u32, ldy: u32, global: f32 };
/// The bf16 row projection's constants (llama's ll_mv_args).
const MArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };
/// The decode attention's constants (rc_vec_args); its first eight are llama's ll_vec_args, which the merge reads.
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, blocks: u32, chunk: u32, first: u32, scale: f32, window: u32 = 0, ring: u32 = 0 };
/// A global layer's window: every key before.
const every_key = std.math.maxInt(i32);

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

const Enc = struct {
    e: mtl.ComputeEncoder,
    m: *const Model,
    s: *st.Scratch,

    fn glue(x: Enc, p: mtl.Pipeline, args: anytype, bufs: []const B, grid: [3]usize, group: [3]usize, groups: bool) void {
        x.e.setPipeline(p);
        for (bufs, 0..) |r, slot| {
            if (r.args) x.e.setValue(args, slot) else x.e.setBuffer(r.buf, r.off, slot);
        }
        const g3 = mtl.Size.of(grid[0], grid[1], grid[2]);
        const t3 = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) x.e.dispatchGroups(g3, t3) else x.e.dispatchThreads(g3, t3);
    }

    /// y [rows, d.n] (rows `ldy` apart, at byte `out_off`) = x W^T: the GEMM a prompt chunk, the row kernels a window.
    fn dense(x: Enc, d: wts.Dense, in: B, out: mtl.Buffer, out_off: usize, n_rows: usize, ldy: usize) void {
        const e = x.e;
        if (n_rows <= st.batch_rows) {
            // the variant holding the rows in registers: 1, 2, 4, 8, 16 or 32
            const v: @import("model.zig").Llama = if (n_rows <= 1) .matvec_1 else if (n_rows <= 2) .matvec_2 else if (n_rows <= 4) .matvec_4 else if (n_rows <= 8) .matvec_8 else if (n_rows <= 16) .matvec_16 else .matvec_32;
            e.setPipeline(x.m.ll(v));
            e.setBuffer(in.buf, in.off, 0);
            e.setBuffer(d.buf, d.off, 1);
            e.setValue(MArgs{ .rows = @intCast(n_rows), .n = @intCast(d.n), .k = @intCast(d.k), .ldy = @intCast(ldy) }, 2);
            e.setBuffer(out, out_off, 3);
            e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        const tn = (d.n + 127) / 128;
        const tm = (n_rows + 63) / 64;
        e.setPipeline(x.m.gemm[@intFromBool(n_rows % 64 == 0)][@intFromBool(d.n % 128 == 0)]);
        e.setBuffer(in.buf, in.off, 0);
        e.setBuffer(d.buf, d.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ n_rows, d.n, d.k, d.k, d.k, ldy, tn, tm, 2, d.k / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |v, i| p[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(out, out_off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }

    /// An MLP projection: bf16, or NVFP4 rows; a prompt chunk dequantizes exactly, then the GEMM applies the global.
    fn proj(x: Enc, p: wts.Proj, in: B, out: mtl.Buffer, out_off: usize, n_rows: usize, ldy: usize) void {
        const d = switch (p) {
            .bf16 => |d| return x.dense(d, in, out, out_off, n_rows, ldy),
            .fp4 => |d| d,
        };
        const f = &(x.m.fp4 orelse unreachable);
        const e = x.e;
        const fa = Fp4Args{ .rows = @intCast(n_rows), .n = @intCast(d.n), .k = @intCast(d.k), .ldy = @intCast(ldy), .global = d.global };
        if (n_rows <= st.batch_rows) {
            const v: usize = if (n_rows <= 1) 0 else if (n_rows <= 2) 1 else if (n_rows <= 4) 2 else if (n_rows <= 8) 3 else if (n_rows <= 16) 4 else 5;
            e.setPipeline(f.rows[v]);
            e.setBuffer(in.buf, in.off, 0);
            e.setBuffer(d.w, d.w_off, 1);
            e.setBuffer(d.s, d.s_off, 2);
            e.setValue(fa, 3);
            e.setBuffer(out, out_off, 4);
            e.dispatchGroups(mtl.Size.of((d.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            return;
        }
        e.setPipeline(f.rows[6]);
        e.setBuffer(d.w, d.w_off, 0);
        e.setBuffer(d.s, d.s_off, 1);
        e.setValue(fa, 2);
        e.setBuffer(x.s.wq, 0, 3);
        e.dispatchThreads(mtl.Size.of(d.k / 16, d.n, 1), mtl.Size.of(64, 4, 1));
        const tn = (d.n + 127) / 128;
        const tm = (n_rows + 63) / 64;
        e.setPipeline(f.gemm[@intFromBool(n_rows % 64 == 0)][@intFromBool(d.n % 128 == 0)]);
        e.setBuffer(in.buf, in.off, 0);
        e.setBuffer(x.s.wq, 0, 1);
        var pa: [16]i32 = @splat(0);
        const vals = [_]usize{ n_rows, d.n, d.k, d.k, d.k, ldy, tn, tm, 2, d.k / 256, 0, 0, 0, 0, 0 };
        for (vals, 0..) |v, i| pa[i] = @intCast(v);
        e.setBytes(std.mem.asBytes(&pa), 2);
        e.setBuffer(out, out_off, 3);
        e.setValue(d.global, 4);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, 2), mtl.Size.of(32, 4, 2));
    }

    /// rc_add_norm over n rows: with `post`, R += norm(y, post); X = norm(R, next); with `blend`, VX too.
    fn addNorm(x: Enc, n: usize, post: ?wts.Tensor, next: wts.Tensor, blend: ?f32) void {
        const s = x.s;
        const c = &x.m.config;
        const flags = (if (post != null) add_flag else 0) | (if (blend != null) lerp_flag else 0);
        const a = Args{ .rows = @intCast(n), .width = @intCast(c.hidden), .flags = flags, .eps = c.eps, .blend = blend orelse 0 };
        const p = if (post) |w| t(w) else t(next); // unread without the add
        x.glue(x.m.pipe(.add_norm), a, &.{ b(s.resid), b(s.y), p, t(next), b(s.v0), A, b(s.x), b(s.vx) }, .{ n, 1, 1 }, .{ 256, 1, 1 }, true);
    }
};

/// All rows' logits or the last's; ids on the GPU; the last row drawn on the GPU; committed unwaited into `flight`.
pub const Out = struct { all: bool = false, timing: ?*Timing = null, ids: ?B = null, draw: ?Draw = null, flight: ?*mtl.CommandBuffer = null };

/// A GPU draw of the last row's logits: argmax, or the keyed sampler at `position` (lanes' gpu_rule and gpu_full).
pub const Draw = struct {
    how: union(enum) { greedy, sampled: struct { sampling: lanes.Sampling, position: u64 } },
    out: B,
};

/// A forward's host encoding and GPU seconds (tf-ricursive-run's bench).
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
    if (n == 0 or n > s.rows or (out.all and n > st.batch_rows) or (!out.all and segments.len != 1)) return error.RicursiveRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.RicursiveContextFull;
        if (segments.len > 1 and seg.rows > st.window_rows) return error.RicursiveRowsExceeded;
        total += seg.rows;
    }
    if (total != n) return error.InvalidRicursiveSegments;
    if (out.ids == null) {
        for (ids) |id| if (id >= c.vocab) return error.InvalidRicursiveToken;
        @memcpy(s.ids.slice(u32, n), ids);
    }
    if (out.draw != null and (out.all or segments.len != 1)) return error.InvalidRicursiveSegments;
    const h = c.hidden;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const t0 = if (out.timing != null) seconds() else 0;
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    const W = m.weights;
    const ea = Args{ .rows = @intCast(n), .width = @intCast(h), .scale = c.embed_scale };
    x.glue(m.pipe(.embed), ea, &.{ out.ids orelse b(s.ids), bo(W.embed.buf, W.embed.off), A, b(s.resid), b(s.v0) }, .{ h, n, 1 }, .{ 256, 1, 1 }, false);
    x.addNorm(n, null, W.layers[0].input_norm, W.layers[0].attn.blend);
    for (W.layers, 0..) |l, i| try layer(x, segments, l, i, n, if (i + 1 < W.layers.len) W.layers[i + 1] else null);
    if (out.all) {
        x.dense(W.head, b(s.x), s.logits_all, 0, n, c.vocab);
    } else {
        x.dense(W.head, bo(s.x, (n - 1) * h * 2), segments[0].cache.logits, 0, 1, c.vocab);
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
        std.log.err("ricursive forward: {s}", .{msg});
        return error.GpuFailed;
    }
    if (out.timing) |tm| tm.gpu += cb.gpuSeconds();
}

/// The head alone on `rows` final-normed rows in s.x, into s.logits_all (a prompt chunk's every row, for the check).
pub fn head(m: *const Model, s: *st.Scratch, rows: usize) !void {
    if (rows == 0 or rows > st.batch_rows or rows > s.rows) return error.RicursiveRowsExceeded;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const x = Enc{ .e = cb.compute(.serial), .m = m, .s = s };
    x.dense(m.weights.head, b(s.x), s.logits_all, 0, rows, m.config.vocab);
    x.e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("ricursive head: {s}", .{msg});
        return error.GpuFailed;
    }
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
            e.setPipeline(x.m.ll(.argmax));
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

/// Layer `i` on every segment's rows (s.x and s.vx its inputs); its last add norms the residual for `next`.
fn layer(x: Enc, segments: []const Segment, l: wts.Layer, i: usize, n: usize, next: ?wts.Layer) !void {
    const s = x.s;
    const c = &x.m.config;
    const h = c.hidden;
    const qkvw = c.qkvWidth();
    x.dense(l.attn.q, b(s.x), s.q, 0, n, qkvw);
    x.dense(l.attn.k, b(s.x), s.q, c.queryWidth() * 2, n, qkvw);
    x.dense(l.attn.v, b(s.vx), s.q, (c.queryWidth() + c.kvWidth()) * 2, n, qkvw);
    x.dense(l.attn.g, b(s.x), s.g, 0, n, c.queryWidth());
    var base: usize = 0;
    for (segments) |seg| {
        try attention(x, seg.cache, l, i, base, seg.rows);
        base += seg.rows;
    }
    const ga = Args{ .rows = @intCast(n), .width = @intCast(c.queryWidth()) };
    x.glue(x.m.pipe(.gate), ga, &.{ b(s.att), b(s.g), A }, .{ c.queryWidth(), n, 1 }, .{ 256, 1, 1 }, false);
    x.dense(l.attn.o, b(s.att), s.y, 0, n, h);
    x.addNorm(n, l.post_attn_norm, l.pre_ffn_norm, null);
    x.proj(l.mlp.gate, b(s.x), s.gu, 0, n, 2 * c.intermediate);
    x.proj(l.mlp.up, b(s.x), s.gu, c.intermediate * 2, n, 2 * c.intermediate);
    const fa = Args{ .rows = @intCast(n), .width = @intCast(c.intermediate), .scale = c.clamp };
    x.glue(x.m.pipe(.silu_clamp), fa, &.{ b(s.gu), A, b(s.act) }, .{ c.intermediate, n, 1 }, .{ 256, 1, 1 }, false);
    x.proj(l.mlp.down, b(s.act), s.y, 0, n, h);
    if (next) |nl| {
        x.addNorm(n, l.post_ffn_norm, nl.input_norm, nl.attn.blend);
    } else x.addNorm(n, l.post_ffn_norm, x.m.weights.norm, null);
}

/// One stream's scratch rows base..: q and k normed (rotated on sliding layers), k and v into its cache, attention.
fn attention(x: Enc, cache: *st.Cache, l: wts.Layer, i: usize, base: usize, n: usize) !void {
    const s = x.s;
    const c = &x.m.config;
    const start = cache.len;
    const global = c.global[i];
    const ring = cache.layer_slots[i];
    const qrow: usize = c.queryWidth() * 2; // a row's qn and att bytes
    const qn = bo(s.qn, base * qrow);
    const o = bo(s.att, base * qrow);
    const pa = PArgs{ .rows = @intCast(n), .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .pos = @intCast(start), .stride = @intCast(c.qkvWidth()), .ring = @intCast(ring), .rope = @intFromBool(!global), .eps = c.eps };
    const keys = bo(cache.kv, cache.keys(i));
    const values = bo(cache.kv, cache.values(i));
    x.glue(x.m.pipe(.prep), pa, &.{ bo(s.q, base * c.qkvWidth() * 2), b(x.m.inv), t(l.attn.q_norm), t(l.attn.k_norm), A, qn, keys, values }, .{ c.heads + c.kv_heads, n, 1 }, .{ 32, 1, 1 }, true);
    const e = x.e;
    const window: usize = if (global) every_key else c.window;
    if (n > st.window_rows) { // a prompt chunk: 32 rows of a head a tile, over every key in the rows' windows
        e.setPipeline(x.m.prompt_attn);
        for ([_]B{ qn, keys, values, o }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
        e.setValue([4]i32{ @intCast(n), @intCast(start), @intCast(window), @intCast(ring) }, 4);
        e.dispatchGroups(mtl.Size.of((n + 31) / 32, c.heads, 1), mtl.Size.of(cfg.head_dim / 64 * 32, 1, 1));
        return;
    }
    // decode rows: a key head's query heads a threadgroup over fixed key blocks at absolute places, then the merge
    const chunk = if (global) st.chunk(cache.capacity) else st.ring_chunk;
    const low = if (start + 1 > window) start + 1 - window else 0; // the first row's first key
    const first = low / chunk;
    const blocks = (start + n - 1) / chunk + 1 - first;
    if (blocks > st.max_blocks) return error.RicursiveTooManyBlocks;
    const va = VArgs{ .rows = @intCast(n), .pos = @intCast(start), .heads = @intCast(c.heads), .groups = @intCast(c.kv_heads), .blocks = @intCast(blocks), .chunk = @intCast(chunk), .first = @intCast(first), .scale = c.scale(), .window = @intCast(window), .ring = @intCast(ring) };
    e.setPipeline(x.m.pipe(.decode_vec));
    for ([_]B{ qn, keys, values }, 0..) |r, slot| e.setBuffer(r.buf, r.off, slot);
    e.setValue(va, 3);
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 4..) |r, slot| e.setBuffer(r, 0, slot);
    e.dispatchGroups(mtl.Size.of(c.kv_heads, n, blocks), mtl.Size.of(32, c.group(), 1));
    e.setPipeline(x.m.ll(.merge_vec));
    for ([_]mtl.Buffer{ s.pm, s.pl, s.po }, 0..) |r, slot| e.setBuffer(r, 0, slot);
    e.setValue(va, 3);
    e.setBuffer(o.buf, o.off, 4);
    e.dispatchGroups(mtl.Size.of(c.heads, n, 1), mtl.Size.of(1024, 1, 1));
}
