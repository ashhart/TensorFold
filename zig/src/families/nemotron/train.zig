//! One step of learning in the weights: a sequence forward through every layer, its answer's loss, back to each change.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const pre = @import("prefill.zig");
const pl = @import("prefill_launch.zig");
const ops = @import("train_ops.zig");
const back = @import("train_back.zig");
const adapters = @import("adapters.zig");
const learned = @import("learned.zig");
const backend = @import("backend.zig");

const Metal = backend.Metal;
const At = pl.At;
const Buffer = mtl.Buffer;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// What the learner runs on: the Metal engine.
pub const Backend = Metal;

/// Rows a step learns from at most: a question and its answer (attention's backward reads at most 256).
pub const max_rows = 256;

/// A step's loss alone, with the open block's gradient, or with that and Adam; or site inputs sketched or projected.
pub const Mode = enum { loss, grad, learn, avoid, seek, project };

/// The answer's mean loss, and whether each of its tokens was the likeliest at its row.
pub const Result = struct { loss: f32, recalled: bool };

pub const Trainer = struct {
    b: *Metal,
    sites: adapters.Sites,
    bufs: back.Bufs,
    tmp: ops.Tmp,
    saved: Buffer, // bf16 [layers + 1, max_rows, D]: the residual entering each layer, then the last one
    acts: Buffer, // what each layer's backward reads of its forward's scratch, from slots[layer]
    slots: [cfg.max_layers]usize,
    logits: Buffer, // bf16 [max_rows, V]: the logits, then their gradient
    stats: Buffer, // f32 [max_rows, 2]: each row's loss and target probability
    targets: Buffer, // u32 [max_rows]
    weights: Buffer, // f32 [max_rows]: 1 / answer tokens on the rows that predict the answer, 0 elsewhere
    dhn: Buffer, // bf16 [max_rows, D]
    head_t: Buffer, // bf16 [D, V]: the LM head transposed, for the loss's backward product
    head: [3]At,
    cache: st.Cache,
    fresh: u32,
    sketched: usize = 0, // rows sketched since the sketches were cleared, each row's place in them
    proj: Buffer, // f32 [layers, max_rows, candidates]: each row along each layer's candidate directions
    profile: ?*fwd.Profiler = null, // times each dispatch alone (a check's breakdown), else steps run as one command buffer

    /// Every buffer a step needs and a change at every layer, attached to the forward (zero, so nothing moves yet).
    pub fn init(gpa: std.mem.Allocator, b: *Metal) !*Trainer {
        const m = b.m;
        const c = m.config;
        if (b.wide == null or c.mamba_head_dim != 64 or c.state != 128) return error.UnsupportedShape;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        t.b = b;
        t.profile = null;
        t.sketched = 0;
        t.sites = try adapters.Sites.init(gpa, m.device, c, @max(b.o.chunk, 64));
        errdefer t.sites.deinit();
        t.bufs = try back.Bufs.init(m.device, c, max_rows);
        errdefer t.bufs.deinit();
        const widest = @max(c.projDim(), c.heads * c.head_dim, c.shared_width, c.hidden);
        t.tmp = .{ .xb = try m.device.buffer(max_rows * widest * 2, opts), .wt = try m.device.buffer(widest * c.hidden * 2, opts), .acc = try m.device.buffer(max_rows * widest * 4, opts), .part = try m.device.buffer(ops.slices * max_rows * c.top_k * @max(c.hidden, c.expert_width) * 4, opts) };
        errdefer inline for (.{ "xb", "wt", "acc", "part" }) |f| @field(t.tmp, f).deinit();
        const d = c.hidden;
        const sizes = [_]usize{ (c.layers + 1) * max_rows * d * 2, max_rows * c.vocab * 2, max_rows * 8, max_rows * 4, max_rows * 4, max_rows * d * 2, d * c.vocab * 2, c.layers * max_rows * adapters.candidates * 4 };
        const fields = [_]*Buffer{ &t.saved, &t.logits, &t.stats, &t.targets, &t.weights, &t.dhn, &t.head_t, &t.proj };
        var made: usize = 0;
        errdefer for (fields[0..made]) |buf| buf.deinit();
        for (fields, sizes) |buf, n| {
            buf.* = try m.device.buffer(n, opts);
            made += 1;
        }
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, i| {
            const tensor = try m.checkpoint.get("lm_head." ++ part);
            t.head[i] = .{ .b = tensor.buffer, .off = tensor.offset };
        }
        var at: usize = 0;
        for (0..c.layers) |l| {
            t.slots[l] = at;
            at += slot(c, c.kinds[l]);
        }
        t.acts = try m.device.buffer(at, opts);
        errdefer t.acts.deinit();
        t.cache = try st.Cache.init(m.device, c, max_rows + 1, false);
        errdefer t.cache.deinit(&b.pool);
        t.fresh = try b.pool.take();
        const Head = struct {
            t: *Trainer,
            pub fn encode(j: @This(), mb: *Metal, e: *fwd.Enc) !void {
                e.pipe(mb.m.kernels.get("tf_train_head_t"));
                for (j.t.head, 0..) |h, i| e.buf(h.b, h.off, i);
                e.buf(j.t.head_t, 0, 3);
                e.run(.{ mb.m.config.vocab, mb.m.config.hidden, 1 }, .{ 256, 1, 1 });
            }
        };
        try b.drain();
        try b.submit(.prefill, b.next, Head{ .t = t });
        try b.drain();
        t.sites.attach(&m.weights, true);
        return t;
    }

    pub fn deinit(t: *Trainer, gpa: std.mem.Allocator) void {
        t.sites.attach(&t.b.m.weights, false);
        t.sites.deinit();
        t.bufs.deinit();
        inline for (.{ "xb", "wt", "acc", "part" }) |f| @field(t.tmp, f).deinit();
        inline for (.{ "saved", "acts", "logits", "stats", "targets", "weights", "dhn", "head_t", "proj" }) |f| @field(t, f).deinit();
        t.b.pool.give(t.fresh);
        t.cache.deinit(&t.b.pool);
        gpa.destroy(t);
    }

    /// The mean loss of `ids`' answer from `start`, and what `mode` adds to it.
    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode) !Result {
        const b = t.b;
        if (ids.len < 2 or start < 1 or start >= ids.len or ids.len - 1 > max_rows) return error.BadExample;
        if ((mode == .grad or mode == .learn) and t.sites.rank == 0) return error.NoOpenBlock;
        const rows = ids.len - 1;
        @memcpy(b.prompt.slice(u32, rows), ids[0..rows]);
        const share = 1 / @as(f32, @floatFromInt(ids.len - start));
        for (t.targets.slice(u32, rows), t.weights.slice(f32, rows), 0..) |*tg, *w, r| {
            tg.* = ids[r + 1];
            w.* = if (r + 1 >= start) share else 0;
        }
        t.cache.len = 0;
        try b.drain();
        const job: Job = .{ .t = t, .rows = rows, .start = start, .mode = mode };
        if (t.profile) |p| {
            var e: fwd.Enc = .{ .e = p.begin(), .prof = p };
            try job.encode(b, &e);
            e.e.end();
            p.cb.?.commit();
            p.cb.?.wait();
        } else {
            try b.submit(.prefill, b.next, job);
            try b.drain();
        }
        if (mode == .avoid or mode == .seek) t.sketched += rows;
        if (mode == .avoid or mode == .seek or mode == .project) return .{ .loss = 0, .recalled = false };
        var out: Result = .{ .loss = 0, .recalled = true };
        for (t.stats.slice([2]f32, rows), t.weights.slice(f32, rows)) |s, w| if (w > 0) {
            out.loss += s[0] * w;
            out.recalled = out.recalled and s[1] > 0.5;
        };
        return out;
    }

    /// Every layer's forward applies the change's blocks in use (off: none of them).
    pub fn attach(t: *Trainer, on: bool) void {
        t.sites.attach(&t.b.m.weights, on);
    }

    /// The first `ranks` of the change folded into the output projections and written into the model's shards.
    pub fn save(t: *Trainer, gpa: std.mem.Allocator, io: std.Io, ranks: usize) !usize {
        const m = t.b.m;
        return learned.write(gpa, io, m.dir, &m.checkpoint, &m.weights, m.config, &t.sites, ranks);
    }

    /// Each row along layer l's candidate directions ([rows, candidates]), as the last project step left them.
    pub fn projected(t: *const Trainer, l: usize, rows: usize) []const f32 {
        const k = adapters.candidates;
        return t.proj.slice(f32, t.b.m.config.layers * max_rows * k)[l * max_rows * k ..][0 .. rows * k];
    }

    const Job = struct {
        t: *Trainer,
        rows: usize,
        start: usize,
        mode: Mode,

        pub fn encode(j: @This(), b: *Metal, e: *fwd.Enc) !void {
            const t = j.t;
            const rows = j.rows;
            const m = b.m;
            const c = m.config;
            const d = c.hidden;
            const w = &b.wide.?;
            const x = w.context(e, b.forward(), &t.cache, rows, t.fresh);
            const o: ops.Ops = .{ .k = &m.kernels, .pk = &m.prefill, .e = e, .tmp = &t.tmp };
            const at = struct {
                fn layer(tr: *Trainer, l: usize, dim: usize) At {
                    return At.of(tr.saved).plus(l * max_rows * dim * 2);
                }
            }.layer;
            w.embed(x, b.prompt, 0);
            for (0..c.layers) |l| {
                copy(e, m, At.of(w.s.h), at(t, l, d), rows);
                w.layer(x, l);
                switch (j.mode) {
                    .loss => {},
                    .grad, .learn => keepActs(t, e, &w.s, l, rows, .out),
                    .avoid, .seek => sketchSite(t, o, &w.s, l, rows, j.start, j.mode),
                    .project => {
                        const site = &t.sites.list[l];
                        o.project(input(c, &w.s, l), site.seek, At.of(t.proj).plus(l * max_rows * adapters.candidates * 4), rows, site.in, adapters.candidates);
                    },
                }
            }
            if (j.mode == .avoid or j.mode == .seek or j.mode == .project) return;
            copy(e, m, At.of(w.s.h), at(t, c.layers, d), rows);
            w.final(x);
            const lp: pl.Launch = .{ .k = &m.prefill, .e = e };
            lp.qmm(At.of(w.s.x), t.head, At.of(t.logits), At.of(w.s.parts), rows, c.vocab, d);
            e.pipe(m.kernels.get("tf_train_softmax"));
            e.buf(t.logits, 0, 0);
            e.buf(t.targets, 0, 1);
            e.buf(t.weights, 0, 2);
            e.buf(t.stats, 0, 3);
            e.bytes(@as(u32, @intCast(c.vocab)), 4);
            e.run(.{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 });
            if (j.mode == .loss) return;
            ops.mm(lp, At.of(t.logits), At.of(t.head_t), At.of(t.dhn), rows, d, c.vocab, c.vocab, c.vocab, d);
            o.widen(At.of(t.dhn), t.bufs.dx, rows * d);
            o.zero(t.bufs.g, rows * d);
            o.rmsBack(at(t, c.layers, d), pre.tensorAt(m.weights.norm_f), t.bufs.dx, t.bufs.g, rows, d, c.eps);
            var l = c.layers;
            while (l > 0) {
                l -= 1;
                keepActs(t, e, &w.s, l, rows, .in);
                const site = &t.sites.list[l];
                const first = if (t.sites.rank >= adapters.block) t.sites.first() else 0;
                const layer: back.Layer = .{ .o = o, .c = c, .s = &w.s, .p = w, .w = &m.weights, .ad = site.adapter(&t.sites), .gb = site.gb, .first = first, .rows = rows };
                switch (c.kinds[l]) {
                    .moe => back.moe(layer, &t.bufs, l),
                    .mamba => back.mamba(layer, &t.bufs, l),
                    .attention => back.attention(layer, &t.bufs, l),
                }
                o.rmsBack(at(t, l, d), pre.tensorAt(m.weights.norms[l]), t.bufs.dx, t.bufs.g, rows, d, c.eps);
            }
            if (j.mode == .learn) t.sites.adam(o);
        }
    };
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

fn bytes(c: cfg.Config, part: Part, rows: usize) usize {
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

/// A layer's slot: each part at max_rows, 256-byte aligned.
fn slot(c: cfg.Config, kind: cfg.Kind) usize {
    var n: usize = 0;
    switch (kind) {
        inline else => |k| inline for (comptime acts(k)) |f| {
            n += std.mem.alignForward(usize, bytes(c, f.part, max_rows), 256);
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
            off += std.mem.alignForward(usize, bytes(c, f.part, max_rows), 256);
        },
    }
}

/// Layer l's site input: what its change reads.
fn input(c: cfg.Config, s: *const pre.Scratch, l: usize) Buffer {
    return if (c.kinds[l] == .moe) s.upr else s.ya;
}

/// Layer l's site input into the sketch a block avoids or reads; the answer's first row again, weighed as its answer.
fn sketchSite(t: *Trainer, o: ops.Ops, s: *const pre.Scratch, l: usize, rows: usize, start: usize, mode: Mode) void {
    const site = &t.sites.list[l];
    const x = At.of(input(t.b.m.config, s, l));
    const y = if (mode == .avoid) site.avoid else site.seek;
    const k: usize = if (mode == .avoid) adapters.avoid_dims else adapters.candidates;
    const seed: u32 = if (mode == .avoid) 1 else 2;
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
