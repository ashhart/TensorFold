//! One step of learning in the weights on CUDA: a sequence through a prompt chunk, its loss, back to each change.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const glue = @import("cuda_glue.zig");
const fwd = @import("cuda_forward.zig");
const ops = @import("cuda_train_ops.zig");
const back = @import("cuda_train_back.zig");
const sites = @import("cuda_sites.zig");
const learned = @import("cuda_learned.zig");
const dims = @import("slide_dims.zig");

/// What the learner runs on: the CUDA engine, whose change's sites --slide made at load.
pub const Backend = engine.Engine;

/// Rows a step learns from at most: a question and its answer (attention's backward reads at most 256).
pub const max_rows = 256;

/// A step's loss alone, with the open block's gradient, or with that and Adam; or site inputs sketched or projected.
pub const Mode = enum { loss, grad, learn, avoid, seek, project };

/// The answer's mean loss, and whether each of its tokens was the likeliest at its row.
pub const Result = struct { loss: f32, recalled: bool };

/// The trainer's own caches hold one attention chunk of keys, past max_rows.
const keys = state.chunk_keys;

/// The scratch each kind of layer's backward reads, with how much of it a row (or the layer) fills.
const Part = enum { proj, conv, inner, qkv, heads, pairs, router, act, ys };
const Act = struct { field: []const u8, part: Part };
const moe_acts = [_]Act{ .{ .field = "pick", .part = .pairs }, .{ .field = "wts", .part = .pairs }, .{ .field = "part", .part = .router }, .{ .field = "act", .part = .act }, .{ .field = "ymoe", .part = .ys } };
const mamba_acts = [_]Act{ .{ .field = "proj", .part = .proj }, .{ .field = "p_xc", .part = .conv }, .{ .field = "g", .part = .inner } };
const attention_acts = [_]Act{ .{ .field = "qkv", .part = .qkv }, .{ .field = "att", .part = .heads } };

fn acts(comptime kind: cfg.Kind) []const Act {
    return switch (kind) {
        .moe => &moe_acts,
        .mamba => &mamba_acts,
        .attention => &attention_acts,
    };
}

fn bytes(c: cfg.Config, part: Part, rows: usize) usize {
    return switch (part) {
        .proj => rows * c.projDim() * 2,
        .conv => rows * c.convDim() * 2,
        .inner => rows * c.inner() * 2,
        .qkv => rows * c.qkvDim() * 2,
        .heads => rows * c.heads * c.head_dim * 2,
        .pairs => rows * c.slots() * 4,
        .router => glue.routerShape(c.hidden).sk * rows * c.experts * 4,
        .act => rows * c.slots() * c.expert_width * 2,
        .ys => rows * c.slots() * c.hidden * 2,
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

pub const Trainer = struct {
    e: *engine.Engine,
    sites: sites.Sites,
    seq: state.Seq, // the trainer's own caches and Mamba state: a step leaves every stream's alone
    b: state.Buffers, // the engine's scratch, the trainer's sequence in place of the bound one's
    bufs: back.Bufs,
    saved: cuda.DeviceBuffer, // bf16 [layers + 1, max_rows, D]: the residual entering each layer, then the last one
    acts: cuda.DeviceBuffer, // what each layer's backward reads of its forward, from slots[layer]
    slots: [cfg.max_layers]usize,
    logits: cuda.DeviceBuffer, // bf16 [max_rows, V]: the logits, then their gradient
    head_t: cuda.DeviceBuffer, // bf16 [D, V]: the LM head dequantized transposed, for the loss's backward product
    io: sites.Shared, // u32 ids [max_rows], targets [max_rows], f32 weights [max_rows], stats [max_rows, 2]
    proj: sites.Shared, // f32 [layers, max_rows, candidates]: each row along each layer's candidate directions
    sketched: usize = 0, // rows sketched since the sketches were cleared, each row's place in them
    bases: learned.Bases = .{}, // each saved layer's projection as loaded, which every later save folds onto

    /// What `init` allocates for `c`, device and mapped: the first lesson's memory, which the budget keeps at load.
    pub fn memory(c: cfg.Config) usize {
        var parts_bytes: usize = 0;
        for (0..c.layers) |l| parts_bytes += slot(c, c.kinds[l]);
        const device = state.Seq.bytes(c, keys, &.{}) + back.Bufs.total(c, max_rows) + (c.layers + 1) * max_rows * c.hidden * 2 + parts_bytes + max_rows * c.vocab * 2 + c.hidden * c.vocab * 2;
        return device + 5 * max_rows * 4 + c.layers * max_rows * dims.candidates * 4;
    }

    /// Every buffer a step needs, the change's sites taken from the engine (still attached: nothing moves yet).
    pub fn init(gpa: std.mem.Allocator, e: *engine.Engine) !*Trainer {
        const c = e.c;
        if (c.mamba_head_dim != 64 or c.state != 128 or c.conv_kernel != 4 or c.head_dim > 256 or !c.norm_topk) return error.UnsupportedShape;
        const owned = e.slide orelse return error.NotLearning;
        const d = e.ctx.d;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        t.* = .{ .e = e, .sites = owned, .seq = undefined, .b = e.b, .bufs = undefined, .saved = undefined, .acts = undefined, .slots = undefined, .logits = undefined, .head_t = undefined, .io = undefined, .proj = undefined };
        t.seq = try state.Seq.init(d, c, keys, &.{}, e.stream, null);
        errdefer t.seq.deinit();
        inline for (state.seq_fields, t.seq.ptr) |name, p| @field(t.b, name) = p;
        t.bufs = try back.Bufs.init(d, c, max_rows);
        errdefer t.bufs.deinit();
        t.io = try sites.Shared.init(d, u32, 5 * max_rows);
        errdefer t.io.free();
        t.b.p_ids = t.io.dev;
        t.proj = try sites.Shared.init(d, f32, c.layers * max_rows * dims.candidates);
        errdefer t.proj.free();
        var at: usize = 0;
        for (0..c.layers) |l| {
            t.slots[l] = at;
            at += slot(c, c.kinds[l]);
        }
        const sizes = [_]usize{ (c.layers + 1) * max_rows * c.hidden * 2, at, max_rows * c.vocab * 2, c.hidden * c.vocab * 2 };
        const fields = [_]*cuda.DeviceBuffer{ &t.saved, &t.acts, &t.logits, &t.head_t };
        var made: usize = 0;
        errdefer for (fields[0..made]) |f| f.free();
        for (fields, sizes) |f, n| {
            f.* = try cuda.DeviceBuffer.alloc(d, n);
            made += 1;
        }
        try e.ops().train().dequantT(e.w.head, t.head_t.ptr);
        try e.stream.synchronize();
        e.slide = null;
        return t;
    }

    pub fn deinit(t: *Trainer, gpa: std.mem.Allocator) void {
        t.e.stream.synchronize() catch {};
        t.sites.attach(&t.e.w, false);
        t.sites.deinit();
        t.seq.deinit();
        t.bufs.deinit();
        t.io.free();
        t.proj.free();
        inline for (.{ "saved", "acts", "logits", "head_t" }) |f| @field(t, f).free();
        t.bases.deinit(gpa);
        gpa.destroy(t);
    }

    /// The forward applies the change's ranks in use (off: none of them, its kernels idle).
    pub fn attach(t: *Trainer, on: bool) void {
        t.sites.attach(&t.e.w, on);
    }

    /// The first `ranks` of the change folded into the output projections and written into the model's shards.
    pub fn save(t: *Trainer, gpa: std.mem.Allocator, io: std.Io, ranks: usize) !usize {
        return learned.write(gpa, io, t.e.dir orelse return error.NotLearning, t.e.c, &t.sites, ranks, &t.bases);
    }

    /// The mean loss of `ids`' answer from `start`, and what `mode` adds to it.
    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode) !Result {
        const e = t.e;
        if (ids.len < 2 or start < 1 or start >= ids.len or ids.len - 1 > max_rows) return error.BadExample;
        if ((mode == .grad or mode == .learn) and t.sites.rank == 0) return error.NoOpenBlock;
        const rows = ids.len - 1;
        try e.stream.synchronize();
        const w = t.io.slice(u32, 5 * max_rows);
        const targets = w[max_rows..][0..rows];
        const weights = std.mem.bytesAsSlice(f32, std.mem.sliceAsBytes(w[2 * max_rows ..][0..rows]));
        @memcpy(w[0..rows], ids[0..rows]);
        const share = 1 / @as(f32, @floatFromInt(ids.len - start));
        for (targets, weights, 0..) |*tg, *x, r| {
            tg.* = ids[r + 1];
            x.* = if (r + 1 >= start) share else 0;
        }
        try t.forward(rows, start, mode);
        if (mode == .avoid or mode == .seek) t.sketched += rows;
        if (mode == .avoid or mode == .seek or mode == .project) {
            try e.stream.synchronize();
            return .{ .loss = 0, .recalled = false };
        }
        if (mode == .grad or mode == .learn) try t.backward(rows, mode);
        try e.stream.synchronize();
        const stats = std.mem.bytesAsSlice([2]f32, std.mem.sliceAsBytes(w[3 * max_rows ..][0 .. 2 * rows]));
        var out: Result = .{ .loss = 0, .recalled = true };
        for (stats, weights) |s, x| if (x > 0) {
            out.loss += s[0] * x;
            out.recalled = out.recalled and s[1] > 0.5;
        };
        return out;
    }

    /// Each row along layer l's candidate directions ([rows, candidates]), as the last project step left them.
    pub fn projected(t: *const Trainer, l: usize, rows: usize) []const f32 {
        const k = dims.candidates;
        return t.proj.slice(f32, t.e.c.layers * max_rows * k)[l * max_rows * k ..][0 .. rows * k];
    }

    /// The prompt chunk from position 0 on the trainer's fresh state, each layer's parts kept, sketched or projected.
    fn forward(t: *Trainer, rows: usize, start: usize, mode: Mode) !void {
        const e = t.e;
        const c = e.c;
        const tr = e.ops().train();
        const d: u64 = c.hidden;
        const nm = c.count(.mamba);
        try tr.zero(t.b.ssm, nm * c.mamba_heads * c.mamba_head_dim * c.state * 4);
        try tr.zero(t.b.conv_base, nm * 3 * c.convDim() * 2);
        const f = fwd.Forward.init(c, &e.w, &t.b, e.ops(), keys, e.nch, false);
        var w: fwd.Walk = .{ .rows = rows, .pos = 0 };
        try f.chunkBegin(&w);
        for (0..c.layers) |l| {
            try f.chunkPre(&w, l);
            try tr.copy(t.savedAt(l), w.x, rows * d * 2);
            try f.chunkMixer(&w, l);
            try f.chunkPost(&w, l);
            switch (mode) {
                .loss => {},
                .grad, .learn => try t.keep(l, rows),
                .avoid, .seek => try t.sketchSite(l, rows, start, mode),
                .project => {
                    const site = &t.sites.list[l];
                    const in = t.input(l);
                    try tr.project(in.x, in.stride, site.seek.dev, t.proj.dev + l * max_rows * dims.candidates * 4, rows, site.in, dims.candidates);
                },
            }
        }
        if (mode == .avoid or mode == .seek or mode == .project) return;
        try f.finalNorm(&w);
        try tr.copy(t.savedAt(c.layers), w.x, rows * d * 2);
        try e.ops().prefillDense(t.b.y, e.w.head, t.logits.ptr, rows);
        const io = t.io.dev;
        try tr.softmax(t.logits.ptr, io + max_rows * 4, io + 2 * max_rows * 4, io + 3 * max_rows * 4, rows, c.vocab);
    }

    /// The loss back through the head, the final norm and every layer to each change; Adam after, in `learn`.
    fn backward(t: *Trainer, rows: usize, mode: Mode) !void {
        const e = t.e;
        const c = e.c;
        const tr = e.ops().train();
        const b = &t.bufs;
        const d = c.hidden;
        try tr.zero(b.dx, rows * d * 4);
        try tr.gemm(t.logits.ptr, c.vocab, t.head_t.ptr, c.vocab, b.dx, d, rows, d, c.vocab, back.split(rows, d, c.vocab));
        try tr.zero(b.g, rows * d * 4);
        try tr.rmsBack(t.savedAt(c.layers), e.w.norm_f, b.dx, b.g, rows, d, c.eps);
        const sk = glue.routerShape(d).sk;
        var l = c.layers;
        while (l > 0) {
            l -= 1;
            const layer: back.Layer = .{ .t = tr, .o = e.ops(), .c = c, .blk = &e.w.blocks[l], .site = &t.sites.list[l], .sites = &t.sites, .acts = t.parts(l), .b = b, .rows = rows, .sk = sk, .plan = t.b.plan };
            switch (c.kinds[l]) {
                .moe => try back.moe(layer),
                .mamba => try back.mamba(layer),
                .attention => try back.attention(layer),
            }
            try tr.rmsBack(t.savedAt(l), e.w.blocks[l].norm, b.dx, b.g, rows, d, c.eps);
        }
        if (mode == .learn) try t.sites.adam(tr);
    }

    /// Where layer l's residual (or, at `layers`, the final norm's input) is kept.
    fn savedAt(t: *const Trainer, l: usize) u64 {
        return t.saved.ptr + l * max_rows * t.e.c.hidden * 2;
    }

    /// Layer l's kept parts, in its kind's order.
    pub fn parts(t: *const Trainer, l: usize) [5]u64 {
        const c = t.e.c;
        var out: [5]u64 = @splat(0);
        var off = t.slots[l];
        switch (c.kinds[l]) {
            inline else => |k| inline for (comptime acts(k), 0..) |f, i| {
                out[i] = t.acts.ptr + off;
                off += std.mem.alignForward(usize, bytes(c, f.part, max_rows), 256);
            },
        }
        return out;
    }

    /// Layer l's forward parts into its slot.
    fn keep(t: *Trainer, l: usize, rows: usize) !void {
        const c = t.e.c;
        const tr = t.e.ops().train();
        const at = t.parts(l);
        switch (c.kinds[l]) {
            inline else => |k| inline for (comptime acts(k), 0..) |f, i| {
                try tr.copy(at[i], @field(t.b, f.field), bytes(c, f.part, rows));
            },
        }
    }

    /// Layer l's site input as the forward left it: what its change reads, and its row stride.
    fn input(t: *const Trainer, l: usize) struct { x: u64, stride: usize } {
        const c = t.e.c;
        return switch (c.kinds[l]) {
            .moe => .{ .x = t.b.act + c.top_k * c.expert_width * 2, .stride = c.slots() * c.expert_width },
            .mamba => .{ .x = t.b.g, .stride = c.inner() },
            .attention => .{ .x = t.b.att, .stride = c.heads * c.head_dim },
        };
    }

    /// Layer l's site input into the sketch a block avoids or reads; the answer's first row again, as its answer.
    fn sketchSite(t: *Trainer, l: usize, rows: usize, start: usize, mode: Mode) !void {
        const site = &t.sites.list[l];
        const tr = t.e.ops().train();
        const in = t.input(l);
        const y = if (mode == .avoid) site.avoid.dev else site.seek.dev;
        const k: usize = if (mode == .avoid) dims.avoid_dims else dims.candidates;
        const seed: u32 = if (mode == .avoid) 1 else 2;
        const first = start - 1;
        try tr.sketch(in.x, in.stride, y, rows, site.in, k, t.sketched, seed, 1);
        try tr.sketch(in.x + first * in.stride * 2, in.stride, y, 1, site.in, k, (1 << 30) + t.sketched + first, seed, @floatFromInt(rows - first));
    }
};
