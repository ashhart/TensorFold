//! A training pass as one command buffer: its sequences forward through every layer, their losses, back to each change.
const std = @import("std");
const train_mode = @import("train_mode.zig");
const mtl = @import("metal");
const cfg = @import("config.zig");
const fwd = @import("forward.zig");
const pre = @import("prefill.zig");
const pl = @import("prefill_launch.zig");
const ops = @import("train_ops.zig");
const back = @import("train_back.zig");
const tf = @import("train_forward.zig");
const adapters = @import("adapters.zig");
const backend = @import("backend.zig");
const lesson_route = @import("lesson_route.zig");
const train = @import("train.zig");
const Trainer = train.Trainer;
const pack_rows = train.pack_rows;

const Metal = backend.Metal;
const At = pl.At;
const Buffer = mtl.Buffer;
const Mode = train_mode.Mode;
const Seq = train_mode.Seq;
const reads = train_mode.reads;

/// Where sequence s's vector for key i after layer l sits in `delta` and `push`, in floats.
pub fn vec(c: cfg.Config, s: usize, i: usize, l: usize) usize {
    return ((s * 3 + i) * c.layers + l) * c.hidden;
}

/// The lowest layer any sequence's nudge names (layers when none does).
fn lowest(c: cfg.Config, seqs: []const Seq) usize {
    var mask: u64 = 0;
    for (seqs) |sq| mask |= sq.nudge.layers[0] | sq.nudge.layers[1] | sq.nudge.layers[2];
    return if (mask == 0) c.layers else @ctz(mask);
}

pub const Job = struct {
    t: *Trainer,
    rows: usize,
    seqs: []const Seq,
    mode: Mode,
    route: lesson_route.Route,

    pub fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
        const t = j.t;
        const rows = j.rows;
        const spans = t.spans[0..j.seqs.len];
        const m = b.m;
        const c = m.config;
        const d = c.hidden;
        const w = &b.wide.?;
        var f = b.forward();
        f.route = j.route;
        const x = w.context(e, f, &t.cache, rows, t.fresh);
        const o: ops.Ops = .{ .k = &m.kernels, .pk = &m.prefill, .e = e, .tmp = &t.tmp, .route = &j.route };
        const scale: tf.Scale = .{ .rows = t.scale, .from = &t.from };
        const low = if (j.mode == .target) lowest(c, j.seqs) else 0;
        const one = j.seqs[0];
        w.embed(x, b.prompt, 0);
        for (0..c.layers) |l| {
            copy(e, m, At.of(w.s.h), t.savedAt(l), rows);
            tf.layer(x, l, spans, scale, .{ .store = &t.prefixes, .prime = j.mode == .prime });
            if (j.mode == .target) for (j.seqs, spans, 0..) |sq, sp, s| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| if (mask >> @intCast(l) & 1 != 0) {
                e.pipe(m.kernels.get("tf_train_nudge"));
                e.buf(w.s.h, sp.row(row) * d * 2, 0);
                e.buf(t.delta, vec(c, s, i, l) * 4, 1);
                e.run(.{ d, 1, 1 }, .{ 256, 1, 1 });
            };
            switch (j.mode) {
                .loss, .prime => {},
                .grad, .learn => keepActs(t, e, &w.s, l, rows, .out),
                // a target pass keeps each MoE layer's normed input too; its backward reads none below the lowest nudge
                .target => {
                    if (l >= low) keepActs(t, e, &w.s, l, rows, .out);
                    if (c.kinds[l] == .moe) copy(e, m, At.of(w.s.x), At.of(t.inputs).plus(l * pack_rows * d * 2), rows);
                },
                .keys => if (c.kinds[l] == .moe) {
                    keepActs(t, e, &w.s, l, rows, .out);
                    copy(e, m, At.of(w.s.x), At.of(t.inputs).plus(l * pack_rows * d * 2), rows);
                },
                .avoid, .seek => sketchSite(t, o, &w.s, l, rows, one.start, one.decide, j.mode),
                .project => {
                    const site = &t.sites.list[l];
                    o.project(input(c, &w.s, l), site.seek, At.of(t.proj).plus(l * train.max_rows * adapters.candidates * 4), rows, site.in, adapters.candidates);
                },
            }
        }
        if (reads(j.mode)) return;
        copy(e, m, At.of(w.s.h), t.savedAt(c.layers), rows);
        w.final(x);
        // only the rows that predict an answer reach the head: any other row's loss and gradient there is zero
        const answers = t.answers[0..j.seqs.len];
        for (answers) |a| copy(e, m, At.of(w.s.x).plus(a.first * d * 2), At.of(t.ans).plus(a.at * d * 2), a.rows);
        const lp: pl.Launch = .{ .k = &m.prefill, .e = e };
        lp.qmm(At.of(t.ans), t.head, At.of(t.logits), At.of(w.s.parts), t.answered, c.vocab, d);
        e.pipe(m.kernels.get("tf_train_softmax"));
        e.buf(t.logits, 0, 0);
        e.buf(t.targets, 0, 1);
        e.buf(t.weights, 0, 2);
        e.buf(t.stats, 0, 3);
        e.bytes(@as(u32, @intCast(c.vocab)), 4);
        e.run(.{ 1024 * t.answered, 1, 1 }, .{ 1024, 1, 1 });
        if (j.mode == .loss or (j.mode == .target and low == c.layers)) return;
        ops.mm(lp, At.of(t.logits), At.of(t.head_t), At.of(t.dhn), t.answered, d, c.vocab, c.vocab, c.vocab, d);
        o.zero(t.bufs.dx, rows * d);
        for (answers) |a| o.widenAt(At.of(t.dhn).plus(a.at * d * 2), At.of(t.bufs.dx).plus(a.first * d * 4), a.rows * d);
        o.zero(t.bufs.g, rows * d);
        o.rmsBack(t.savedAt(c.layers), pre.tensorAt(m.weights.norm_f), t.bufs.dx, t.bufs.g, rows, d, c.eps);
        var l = c.layers;
        while (l > 0) {
            l -= 1;
            keepActs(t, e, &w.s, l, rows, .in);
            const site = &t.sites.list[l];
            const first = if (t.sites.rank >= adapters.block) t.sites.first() else 0;
            const layer: back.Layer = .{ .o = o, .c = c, .s = &w.s, .p = w, .w = &m.weights, .ad = site.adapter(&t.sites), .gb = site.gb, .first = first, .rows = rows, .spans = spans, .store = &t.prefixes };
            switch (c.kinds[l]) {
                .moe => back.moe(layer, &t.bufs, l),
                .mamba => back.mamba(layer, &t.bufs, l),
                .attention => back.attention(layer, &t.bufs, l),
            }
            o.rmsBack(t.savedAt(l), pre.tensorAt(m.weights.norms[l]), t.bufs.dx, t.bufs.g, rows, d, c.eps);
            // a target pass wants only the gradients at the nudged rows of the residuals its layers left
            if (j.mode == .target and l > 0) {
                for (j.seqs, spans, 0..) |sq, sp, s| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| if (mask >> @intCast(l - 1) & 1 != 0) {
                    e.pipe(m.kernels.get("tf_copy_u32"));
                    e.buf(t.bufs.g, sp.row(row) * d * 4, 0);
                    e.buf(t.push, vec(c, s, i, l - 1) * 4, 1);
                    e.run(.{ d, 1, 1 }, .{ 256, 1, 1 });
                };
                if (l - 1 == low) return;
            }
        }
        if (j.mode == .learn) t.sites.adam(o);
    }
};

/// The scratch each kind of layer's backward reads, with how much of it a row (or the layer) fills.
const Part = enum { shared, routed, pair, experts, outs, route, proj, conv, inner, heads, hidden };
const Act = struct { field: []const u8, part: Part };
const moe_acts = [_]Act{ .{ .field = "up", .part = .shared }, .{ .field = "upr", .part = .shared }, .{ .field = "y1", .part = .routed }, .{ .field = "order", .part = .pair }, .{ .field = "wt", .part = .pair }, .{ .field = "inv", .part = .pair }, .{ .field = "offsets", .part = .experts }, .{ .field = "xr", .part = .outs }, .{ .field = "ids", .part = .pair }, .{ .field = "logits", .part = .route } };
const mamba_acts = [_]Act{ .{ .field = "proj", .part = .proj }, .{ .field = "conv", .part = .conv }, .{ .field = "act", .part = .conv }, .{ .field = "ya", .part = .inner } };
const attention_acts = [_]Act{ .{ .field = "x", .part = .hidden }, .{ .field = "ya", .part = .heads } };

fn acts(comptime kind: cfg.Kind) []const Act {
    return switch (kind) {
        .moe => &moe_acts,
        .mamba => &mamba_acts,
        .attention => &attention_acts,
    };
}

pub fn bytes(c: cfg.Config, part: Part, rows: usize) usize {
    return switch (part) {
        .shared => rows * c.shared_width * 2,
        .routed => rows * c.top_k * c.expert_width * 2,
        .pair => rows * c.top_k * 4,
        .experts => c.experts * 4,
        .outs => rows * c.top_k * c.hidden * 2,
        .route => rows * c.experts * 2,
        .proj => rows * c.projDim() * 2,
        .conv => rows * c.convDim() * 2,
        .inner => rows * c.inner() * 2,
        .heads => rows * c.heads * c.head_dim * 2,
        .hidden => rows * c.hidden * 2,
    };
}

/// A layer's slot: each part at `rows` rows, 256-byte aligned.
pub fn slot(c: cfg.Config, kind: cfg.Kind, rows: usize) usize {
    var n: usize = 0;
    switch (kind) {
        inline else => |k| inline for (comptime acts(k)) |f| {
            n += std.mem.alignForward(usize, bytes(c, f.part, rows), 256);
        },
    }
    return n;
}

/// Into the layer's slot from the scratch (out, after its forward), or back (in, before its backward).
const Way = enum { out, in };

fn keepActs(t: *Trainer, e: *fwd.Enc, s: *const pre.Scratch, l: usize, rows: usize, way: Way) void {
    const m = t.b.m;
    const c = m.config;
    var off = t.slots[l];
    switch (c.kinds[l]) {
        inline else => |k| inline for (comptime acts(k)) |f| {
            const here = At.of(@field(s, f.field));
            const there = At.of(t.acts).plus(off);
            const from = if (way == .out) here else there;
            const to = if (way == .out) there else here;
            e.pipe(m.kernels.get("tf_copy_u32"));
            e.buf(from.b, from.off, 0);
            e.buf(to.b, to.off, 1);
            e.run(.{ bytes(c, f.part, rows) / 4, 1, 1 }, .{ 256, 1, 1 });
            off += std.mem.alignForward(usize, bytes(c, f.part, pack_rows), 256);
        },
    }
}

/// Layer l's site input: what its change reads.
fn input(c: cfg.Config, s: *const pre.Scratch, l: usize) Buffer {
    return if (c.kinds[l] == .moe) s.upr else s.ya;
}

/// Layer l's site input into the sketch a block avoids or reads; the answer's first row again, weighed as its answer.
fn sketchSite(t: *Trainer, o: ops.Ops, s: *const pre.Scratch, l: usize, rows: usize, start: usize, decide: usize, mode: Mode) void {
    const site = &t.sites.list[l];
    const x = At.of(input(t.b.m.config, s, l));
    const y = if (mode == .avoid) site.avoid else site.seek;
    const k: usize = if (mode == .avoid) adapters.avoid_dims else adapters.candidates;
    const seed: u32 = if (mode == .avoid) 1 else 2;
    // an edit of the weights is keyed on the question's rows, where its subject is read, not on how answers open
    if (mode == .seek and t.sites.whole) return o.sketch(x, y, decide + 1, site.in, k, t.sketched, seed, 1);
    const first = start - 1;
    o.sketch(x, y, rows, site.in, k, t.sketched, seed, 1);
    o.sketch(x.plus(first * site.in * 2), y, 1, site.in, k, (1 << 30) + t.sketched + first, seed, @floatFromInt(rows - first));
}

/// rows rows of the hidden width from one bf16 place to another.
fn copy(e: *fwd.Enc, m: anytype, src: At, dst: At, rows: usize) void {
    e.pipe(m.kernels.get("tf_copy_rows"));
    e.buf(src.b, src.off, 0);
    e.buf(dst.b, dst.off, 1);
    e.run(.{ m.config.hidden, rows, 1 }, .{ 256, 1, 1 });
}
