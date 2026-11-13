//! The Qwen3.5-2B pass on the CPU: the Metal forward's rows, segments and commit points, each op over the worker pool.
const std = @import("std");
const c = @import("config.zig");
const st = @import("cpu_state.zig");
const k = @import("cpu_kernels.zig");
const ops = @import("cpu_ops.zig");
const mx = @import("cpu_mixers.zig");
const Model = @import("cpu_model.zig").Model;
const Linear = k.Linear;

pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { last, all };

/// Columns a projection task covers (eight tiles of eight).
const block = 64;

const Project = struct {
    m: *const Model,
    s: *st.Scratch,
    rows: usize,
    linears: []const Linear,
    outs: []const []u16,
    x: []const u16,
    inputs: usize,

    fn prepare(ctx: *anyopaque, index: usize, _: usize) void {
        const p: *Project = @ptrCast(@alignCast(ctx));
        const groups = p.inputs / k.group;
        k.prepareRow(p.x[index * p.inputs ..][0..p.inputs], p.s.xp[index * p.inputs ..][0..p.inputs], p.s.xs[index * groups ..][0..groups]);
    }

    fn columns(ctx: *anyopaque, index: usize, _: usize) void {
        const p: *Project = @ptrCast(@alignCast(ctx));
        var at = index;
        for (p.linears, p.outs) |l, y| {
            const blocks = (l.outputs + block - 1) / block;
            if (at < blocks) {
                k.project(l, p.s.xp, p.s.xs, p.rows, y, at * block, @min(l.outputs, (at + 1) * block));
                return;
            }
            at -= blocks;
        }
    }
};

/// y = x · Wᵀ for several projections of the same input rows in one pass over the pool.
fn project(m: *const Model, s: *st.Scratch, x: []const u16, rows: usize, linears: []const Linear, outs: []const []u16) void {
    var p: Project = .{ .m = m, .s = s, .rows = rows, .linears = linears, .outs = outs, .x = x, .inputs = linears[0].inputs };
    m.pool.run(&p, Project.prepare, rows);
    var tasks: usize = 0;
    for (linears) |l| tasks += (l.outputs + block - 1) / block;
    m.pool.run(&p, Project.columns, tasks);
}

/// A per-row op over the pool: `f(ctx, row)`.
fn perRow(m: *const Model, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) void, count: usize) void {
    const C = @TypeOf(ctx);
    const Wrap = struct {
        fn task(raw: *anyopaque, index: usize, _: usize) void {
            f(@as(C, @ptrCast(@alignCast(raw))), index);
        }
    };
    m.pool.run(@ptrCast(@constCast(ctx)), Wrap.task, count);
}

/// The residual add into h when `residual`, then the registered RMS norm of h into x.
const Norm = struct {
    s: *st.Scratch,
    weight: []const u16,
    residual: bool,
    fn row(n: *const Norm, r: usize) void {
        const h = n.s.h[r * c.hidden ..][0..c.hidden];
        if (n.residual) ops.residual(h, n.s.r[r * c.hidden ..][0..c.hidden]);
        k.rmsnorm(h, n.weight, c.eps, n.s.x[r * c.hidden ..][0..c.hidden]);
    }
};

const Embed = struct {
    m: *const Model,
    s: *st.Scratch,
    fn row(e: *const Embed, r: usize) void {
        ops.embed(e.m.embedding, e.s.ids[r], e.s.h[r * c.hidden ..][0..c.hidden]);
    }
};

const Act = struct {
    s: *st.Scratch,
    fn row(a: *const Act, r: usize) void {
        const at = r * c.intermediate;
        ops.mlpAct(a.s.gate[at..][0..c.intermediate], a.s.up[at..][0..c.intermediate], a.s.act[at..][0..c.intermediate]);
    }
};

pub fn run(m: *const Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head) !void {
    if (tokens.len == 0 or tokens.len > s.rows or (record and tokens.len > st.batch_rows)) return error.QwenRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.QwenContextFull;
        total += seg.rows;
    }
    if (total != tokens.len or (head == .all and total > st.batch_rows) or (!record and segments.len != 1)) return error.InvalidQwenSegments;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidQwenToken;
    @memcpy(s.ids[0..total], tokens);
    perRow(m, &Embed{ .m = m, .s = s }, Embed.row, total);
    for (m.blocks, 0..) |blk, i| {
        perRow(m, &Norm{ .s = s, .weight = blk.input_norm, .residual = i != 0 }, Norm.row, total);
        switch (blk.mixer) {
            .delta => |d| delta(m, s, d, i, segments, record, total),
            .attention => |a| attention(m, s, a, i, segments, total),
        }
        perRow(m, &Norm{ .s = s, .weight = blk.post_norm, .residual = true }, Norm.row, total);
        project(m, s, s.x, total, &.{ blk.gate, blk.up }, &.{ s.gate, s.up });
        perRow(m, &Act{ .s = s }, Act.row, total);
        project(m, s, s.act, total, &.{blk.down}, &.{s.r});
    }
    perRow(m, &Norm{ .s = s, .weight = m.norm, .residual = true }, Norm.row, total);
    if (head == .all) project(m, s, s.x, total, &.{m.head()}, &.{s.logits}) else project(m, s, s.x[(total - 1) * c.hidden ..], 1, &.{m.head()}, &.{s.logits});
    var base: usize = 0;
    for (segments) |seg| {
        const logit_row = if (head == .all) base + seg.rows - 1 else 0;
        @memcpy(seg.cache.logits, s.logits[logit_row * c.vocab ..][0..c.vocab]);
        seg.cache.commit(s, base, seg.rows, record);
        base += seg.rows;
    }
}

const Pre = struct {
    s: *st.Scratch,
    d: @import("cpu_model.zig").Delta,
    cache: *st.Cache,
    layer: usize,
    base: usize,
    fn row(p: *const Pre, w: usize) void {
        const s = p.s;
        const r = p.base + w;
        const snapshot = s.snapshots[p.layer].?;
        const in: mx.ConvInput = .{ .saved = p.cache.blocks[p.layer].delta.conv, .qkv = s.qkv[p.base * c.conv_dim ..] };
        const heads = c.linear_heads;
        mx.gdnPre(in, w, p.d.conv, s.a[r * heads ..][0..heads], s.b[r * heads ..][0..heads], p.d.a_log, p.d.dt_bias, s.q[r * c.hidden ..][0..c.hidden], s.k[r * c.hidden ..][0..c.hidden], s.v[r * c.hidden ..][0..c.hidden], s.g[r * heads ..][0..heads], s.beta[r * heads ..][0..heads], snapshot.conv[r * st.conv_elems ..][0..st.conv_elems]);
    }
};

const Chain = struct {
    s: *st.Scratch,
    cache: *st.Cache,
    layer: usize,
    base: usize,
    count: usize,
    record: bool,
    fn head(ch: *const Chain, h: usize) void {
        const s = ch.s;
        const hd = c.linear_dim;
        var state: [hd * hd]f32 = undefined;
        @memcpy(&state, ch.cache.blocks[ch.layer].delta.recurrence[h * hd * hd ..][0 .. hd * hd]);
        const snapshot = s.snapshots[ch.layer].?;
        const saved = if (ch.record) snapshot.recurrence[ch.base * st.delta_floats ..] else snapshot.recurrence;
        const at = ch.base * c.hidden;
        mx.gdnChain(&state, s.q[at..], s.k[at..], s.v[at..], s.g[ch.base * c.linear_heads ..], s.beta[ch.base * c.linear_heads ..], ch.count, h, s.y[at..], saved, ch.record);
    }
};

const Post = struct {
    s: *st.Scratch,
    norm: []const u16,
    fn row(p: *const Post, r: usize) void {
        const at = r * c.hidden;
        mx.gdnPost(p.s.y[at..][0..c.hidden], p.s.z[at..][0..c.hidden], p.norm, p.s.mix[at..][0..c.hidden]);
    }
};

fn delta(m: *const Model, s: *st.Scratch, d: @import("cpu_model.zig").Delta, layer: usize, segments: []const Segment, record: bool, total: usize) void {
    project(m, s, s.x, total, &.{ d.qkv, d.z, d.a, d.b }, &.{ s.qkv, s.z, s.a, s.b });
    var base: usize = 0;
    for (segments) |seg| {
        perRow(m, &Pre{ .s = s, .d = d, .cache = seg.cache, .layer = layer, .base = base }, Pre.row, seg.rows);
        perRow(m, &Chain{ .s = s, .cache = seg.cache, .layer = layer, .base = base, .count = seg.rows, .record = record }, Chain.head, c.linear_heads);
        base += seg.rows;
    }
    perRow(m, &Post{ .s = s, .norm = d.norm }, Post.row, total);
    project(m, s, s.mix, total, &.{d.out}, &.{s.r});
}

const HeadNorm = struct {
    s: *st.Scratch,
    a: @import("cpu_model.zig").Attention,
    fn row(hn: *const HeadNorm, r: usize) void {
        const s = hn.s;
        for (0..c.query_heads) |h| mx.headNorm(s.qraw[r * 2 * c.hidden + h * 2 * c.head_dim ..][0..c.head_dim], hn.a.q_norm, s.q[(r * c.query_heads + h) * c.head_dim ..][0..c.head_dim]);
        for (0..c.kv_heads) |h| mx.headNorm(s.k[(r * c.kv_heads + h) * c.head_dim ..][0..c.head_dim], hn.a.k_norm, s.knorm[(r * c.kv_heads + h) * c.head_dim ..][0..c.head_dim]);
    }
};

const Rotate = struct {
    s: *st.Scratch,
    cache: *st.Cache,
    layer: usize,
    base: usize,
    fn row(ro: *const Rotate, w: usize) void {
        const s = ro.s;
        const r = ro.base + w;
        const position = ro.cache.len + w;
        for (0..c.query_heads) |h| ops.rotate(s.q[(r * c.query_heads + h) * c.head_dim ..][0..c.head_dim], position, s.queries[(r * c.query_heads + h) * c.head_dim ..][0..c.head_dim]);
        const kv = ro.cache.blocks[ro.layer].attention;
        for (0..c.kv_heads) |h| {
            const dst = (h * ro.cache.capacity + position) * c.head_dim;
            ops.rotate(s.knorm[(r * c.kv_heads + h) * c.head_dim ..][0..c.head_dim], position, kv.keys[dst..][0..c.head_dim]);
            @memcpy(kv.values[dst..][0..c.head_dim], s.v[(r * c.kv_heads + h) * c.head_dim ..][0..c.head_dim]);
        }
    }
};

const Attend = struct {
    s: *st.Scratch,
    cache: *st.Cache,
    layer: usize,
    base: usize,
    fn task(ctx: *anyopaque, index: usize, worker: usize) void {
        const at: *const Attend = @ptrCast(@alignCast(ctx));
        const s = at.s;
        const w = index / c.query_heads;
        const qh = index % c.query_heads;
        const r = at.base + w;
        const kv = at.cache.blocks[at.layer].attention;
        const chunks = mx.chunksFor(at.cache.capacity);
        const partials = s.partials[worker * chunks ..][0..chunks];
        const q = s.queries[(r * c.query_heads + qh) * c.head_dim ..][0..c.head_dim];
        mx.attention(q, kv.keys, kv.values, at.cache.capacity, qh / (c.query_heads / c.kv_heads), at.cache.len + w, partials, s.y[(r * c.query_heads + qh) * c.head_dim ..][0..c.head_dim]);
    }
};

const Gate = struct {
    s: *st.Scratch,
    fn row(g: *const Gate, r: usize) void {
        ops.attentionGate(g.s.y[r * c.hidden ..][0..c.hidden], g.s.qraw[r * 2 * c.hidden ..][0 .. 2 * c.hidden], g.s.mix[r * c.hidden ..][0..c.hidden]);
    }
};

fn attention(m: *const Model, s: *st.Scratch, a: @import("cpu_model.zig").Attention, layer: usize, segments: []const Segment, total: usize) void {
    project(m, s, s.x, total, &.{ a.q, a.k, a.v }, &.{ s.qraw, s.k, s.v });
    perRow(m, &HeadNorm{ .s = s, .a = a }, HeadNorm.row, total);
    var base: usize = 0;
    for (segments) |seg| {
        perRow(m, &Rotate{ .s = s, .cache = seg.cache, .layer = layer, .base = base }, Rotate.row, seg.rows);
        var at: Attend = .{ .s = s, .cache = seg.cache, .layer = layer, .base = base };
        m.pool.run(&at, Attend.task, seg.rows * c.query_heads);
        base += seg.rows;
    }
    perRow(m, &Gate{ .s = s }, Gate.row, total);
    project(m, s, s.mix, total, &.{a.out}, &.{s.r});
}
