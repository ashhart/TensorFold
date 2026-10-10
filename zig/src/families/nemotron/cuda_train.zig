//! One step of learning in the weights on CUDA (train.zig on Metal): forward with activations kept, loss, and back.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const kern = @import("cuda_kernels.zig");
const state = @import("cuda_state.zig");
const fwd = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const ops = @import("cuda_train_ops.zig");
const sites_mod = @import("cuda_sites.zig");

/// Rows a step learns from at most: a question and its answer (attention's backward reads at most 256).
pub const max_rows = 256;

/// A step's loss (`base`: without the open block), with its gradient or Adam (`hold`: a steady row, pulled to nothing).
pub const Mode = enum { loss, base, grad, learn, hold, hold_learn, avoid, seek, project };

/// How hard a steady example pulls the open block's outputs on its rows toward nothing, over all its rows.
pub const hold_weight: f32 = 1;

/// The answer's mean loss, and whether each of its tokens was the likeliest at its row.
pub const Result = struct { loss: f32, recalled: bool };

/// Where the projections onto each layer's candidates start in the mapped host block (after targets, weights, stats).
const proj_at = max_rows * 16;

/// The trainer's own sequence: cache rows enough for a step, one attention chunk.
const seq_len = state.chunk_keys;

/// A matrix [n, k] as the checkpoint holds it, on the GPU: 4-bit words, scales and biases, or bf16 (a lesson's fold).
pub const Mat = struct { w: u64, s: u64 = 0, b: u64 = 0, bf16: bool = false, n: usize, k: usize };

/// What each layer's backward reads of the weights, in the checkpoint's own layout.
const Raw = union(enum) {
    mamba: struct { in_proj: Mat, out_proj: Mat },
    attention: struct { q: Mat, k: Mat, v: Mat, o: Mat },
    moe: struct { up: Mat, down: Mat, fc1: [3]u64, fc2: [3]u64 },
};

/// Where each layer keeps what its backward reads of its forward (bf16 unless named), at max_rows rows.
const Kept = struct {
    x: u64 = 0, // the normed input (attention, MoE)
    proj: u64 = 0, // Mamba: the in-projection's output
    ya: u64 = 0, // Mamba, attention: the output projection's input
    upr: u64 = 0, // MoE: the shared expert's whole activation [rows, F]
    act: u64 = 0, // MoE: every pair's up activation [rows * slots, W]
    ys: u64 = 0, // MoE: every pair's down output before its weight [rows * slots, D]
    ids: u64 = 0, // MoE: i32 [rows * slots]
    wts: u64 = 0, // MoE: f32 [rows * slots]
};

/// Sub-allocations of one device allocation, 256-byte aligned.
const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

pub const Trainer = struct {
    gpa: std.mem.Allocator,
    e: *engine.Engine,
    c: Config,
    sites: sites_mod.Sites,
    seq: state.Seq,
    buf: state.Buffers, // the engine's scratch shape, our own, bound to `seq`
    raw: []Raw,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty, // the raw weights and the LM head transposed
    head_t: u64 = 0, // bf16 [V, D]: the LM head dequantized, for the loss's backward product
    mem: Arena,
    kept: []Kept,
    saved: u64 = 0, // bf16 [layers + 1, max_rows, D]: the residual entering each layer, then the last one
    logits: u64 = 0, // bf16 [max_rows, V]: the logits, then their gradient
    host: sites_mod.Mapped, // targets u32, weights f32, stats [2]f32, each max_rows; then proj f32 [layers, max_rows, candidates]
    g: u64 = 0,
    dx: u64 = 0,
    gs: u64 = 0,
    dhn: u64 = 0,
    dxa: u64 = 0,
    xb: u64 = 0, // bf16 [max_rows, widest]: a product's left side
    wt: u64 = 0, // bf16 [widest n, k]: a product's dequantized matrix
    d_upr: u64 = 0,
    du: u64 = 0,
    dyp: u64 = 0,
    dxp: u64 = 0,
    d_y1r: u64 = 0,
    du1: u64 = 0,
    y1s: u64 = 0,
    part: u64 = 0,
    order: u64 = 0,
    inv: u64 = 0,
    offsets: u64 = 0,
    lg: u64 = 0,
    dyn: u64 = 0,
    ytot: u64 = 0,
    dytot: u64 = 0,
    dt: u64 = 0,
    ddt: u64 = 0,
    dact: u64 = 0,
    dproj: u64 = 0,
    ckpt: u64 = 0,
    states: u64 = 0,
    dbp: u64 = 0,
    dcp: u64 = 0,
    conv_out: u64 = 0,
    act_m: u64 = 0,
    qkv: u64 = 0,
    d_o: u64 = 0,
    dq: u64 = 0,
    dk: u64 = 0,
    dv: u64 = 0,
    probs: u64 = 0,
    dscores: u64 = 0,
    sketched: usize = 0, // rows sketched since the sketches were cleared, each row's place in them
    hold: f32 = 0, // this step's pull on the open block's outputs toward nothing (a steady example's), per row
    gram: u64 = 0, // f32 [16, 16]: the open block's inputs' Gram matrix, for that pull

    /// Every buffer a step needs and the raw weights its backward reads; `living`, saved blocks the trainer takes over.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *engine.Engine, dir: []const u8, living: *?sites_mod.Sites) !*Trainer {
        const c = e.c;
        if (c.mamba_head_dim != 64 or c.state != 128 or c.top_k > 16 or c.experts > 256) return error.UnsupportedShape;
        const d = e.ctx.d;
        const t = try gpa.create(Trainer);
        errdefer gpa.destroy(t);
        t.* = .{ .gpa = gpa, .e = e, .c = c, .sites = undefined, .seq = undefined, .buf = undefined, .raw = undefined, .mem = undefined, .kept = undefined, .host = undefined };
        const saved = living.* != null;
        t.sites = if (living.*) |s| s else try sites_mod.Sites.init(gpa, d, c, state.prefill_rows);
        living.* = null;
        errdefer if (saved) {
            living.* = t.sites;
        } else t.sites.deinit();
        t.seq = try state.Seq.init(d, c, seq_len, &.{}, e.stream);
        errdefer t.seq.deinit();
        t.buf = try e.b.sibling(d, c, e.nch);
        errdefer t.buf.deinit();
        inline for (state.seq_fields, 0..) |name, i| @field(t.buf, name) = t.seq.ptr[i];
        t.host = try sites_mod.Mapped.init(d, max_rows * 16 + c.layers * max_rows * sites_mod.candidates * 4);
        errdefer t.host.deinit();
        t.kept = try gpa.alloc(Kept, c.layers);
        errdefer gpa.free(t.kept);
        t.raw = try gpa.alloc(Raw, c.layers);
        errdefer gpa.free(t.raw);
        errdefer for (t.owned.items) |*b| b.free();
        try t.loadRaw(io, dir);
        t.mem = .{ .buf = try cuda.DeviceBuffer.alloc(d, t.layout(true)) };
        errdefer t.mem.buf.free();
        _ = t.layout(false);
        try e.stream.synchronize();
        return t;
    }

    pub fn deinit(t: *Trainer) void {
        const gpa = t.gpa;
        t.sites.attach(&t.e.w, false);
        t.sites.deinit();
        t.mem.buf.free();
        for (t.owned.items) |*b| b.free();
        t.owned.deinit(gpa);
        gpa.free(t.raw);
        gpa.free(t.kept);
        t.host.deinit();
        t.buf.deinit();
        t.seq.deinit();
        gpa.destroy(t);
    }

    fn ops_(t: *const Trainer) ops.Ops {
        return .{ .k = &t.e.k.train, .s = t.e.stream };
    }

    /// The raw weights the backward's products read, and the LM head dequantized once.
    fn loadRaw(t: *Trainer, io: std.Io, dir: []const u8) !void {
        const c = t.c;
        var ck = try core.Checkpoint.openModel(t.gpa, io, dir);
        defer ck.close();
        var name: [192]u8 = undefined;
        for (0..c.layers) |i| {
            const pre = try std.fmt.bufPrint(&name, "backbone.layers.{d}.mixer.", .{i});
            var buf: [256]u8 = undefined;
            const at = struct {
                fn f(b: *[256]u8, p: []const u8, leaf: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(b, "{s}{s}", .{ p, leaf });
                }
            }.f;
            t.raw[i] = switch (c.kinds[i]) {
                .mamba => .{ .mamba = .{ .in_proj = try t.mat(&ck, try at(&buf, pre, "in_proj")), .out_proj = try t.mat(&ck, try at(&buf, pre, "out_proj")) } },
                .attention => .{ .attention = .{ .q = try t.mat(&ck, try at(&buf, pre, "q_proj")), .k = try t.mat(&ck, try at(&buf, pre, "k_proj")), .v = try t.mat(&ck, try at(&buf, pre, "v_proj")), .o = try t.mat(&ck, try at(&buf, pre, "o_proj")) } },
                .moe => .{ .moe = .{ .up = try t.mat(&ck, try at(&buf, pre, "shared_experts.up_proj")), .down = try t.mat(&ck, try at(&buf, pre, "shared_experts.down_proj")), .fc1 = try t.four(&ck, try at(&buf, pre, "switch_mlp.fc1")), .fc2 = try t.four(&ck, try at(&buf, pre, "switch_mlp.fc2")) } },
            };
        }
        const head = try t.mat(&ck, "lm_head");
        if (head.bf16) {
            t.head_t = head.w;
        } else {
            t.head_t = try t.device(head.n * head.k * 2);
            try t.ops_().dequant(.{ head.w, head.s, head.b }, t.head_t, head.n * head.k);
        }
    }

    fn device(t: *Trainer, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(t.e.ctx.d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try t.owned.append(t.gpa, b);
        return b.ptr;
    }

    fn upload(t: *Trainer, tensor: core.checkpoint.Tensor) !u64 {
        const ptr = try t.device(tensor.bytes.len);
        try t.e.ops().upload(ptr, tensor.bytes);
        return ptr;
    }

    /// `base`.weight [n, k] as the checkpoint holds it: 4-bit with its scales and biases, or bf16.
    fn mat(t: *Trainer, ck: *core.Checkpoint, base: []const u8) !Mat {
        var buf: [256]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}));
        if (w.dtype == .bf16) return .{ .w = try t.upload(w), .bf16 = true, .n = w.dim(0), .k = w.dim(1) };
        if (w.dtype != .u32) return error.UnsupportedQuantization;
        const s = try ck.get(try std.fmt.bufPrint(&buf, "{s}.scales", .{base}));
        const b = try ck.get(try std.fmt.bufPrint(&buf, "{s}.biases", .{base}));
        return .{ .w = try t.upload(w), .s = try t.upload(s), .b = try t.upload(b), .n = w.dim(0), .k = w.dim(1) * 8 };
    }

    /// An expert table [E, n, k] as the checkpoint holds it, 4-bit.
    fn four(t: *Trainer, ck: *core.Checkpoint, base: []const u8) ![3]u64 {
        var buf: [256]u8 = undefined;
        var out: [3]u64 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, i| {
            const x = try ck.get(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, part }));
            if (i == 0 and x.dtype != .u32) return error.UnsupportedQuantization;
            out[i] = try t.upload(x);
        }
        return out;
    }

    /// The device arena's bytes (`count`), else its pointers handed out in the same order.
    fn layout(t: *Trainer, count: bool) usize {
        const c = t.c;
        const R: usize = max_rows;
        const D: usize = c.hidden;
        const S: usize = c.slots();
        const W: usize = c.expert_width;
        const F: usize = c.shared_width;
        const pairs = R * c.top_k;
        const qw = c.heads * c.head_dim;
        const kw = c.kv_heads * c.head_dim;
        const xd = c.inner();
        const cd = c.convDim();
        const pd = c.projDim();
        const H = c.mamba_heads;
        const size = c.mamba_head_dim * c.state;
        const widest_k = @max(pd, qw, F, D, xd);
        var widest_n: usize = 0;
        for (t.raw) |r| switch (r) {
            .mamba => |m| widest_n = @max(widest_n, m.in_proj.n * m.in_proj.k, m.out_proj.n * m.out_proj.k),
            .attention => |a| widest_n = @max(widest_n, a.q.n * a.q.k, a.o.n * a.o.k),
            .moe => |m| widest_n = @max(widest_n, m.up.n * m.up.k, m.down.n * m.down.k),
        };
        var a: Arena = if (count) .{ .buf = .{ .d = undefined, .ptr = 0, .len = std.math.maxInt(usize) } } else t.mem;
        var total: usize = 0;
        const Take = struct {
            fn f(ar: *Arena, n: usize, tot: *usize) u64 {
                tot.* += std.mem.alignForward(usize, n, 256) + 256;
                return ar.take(n);
            }
        };
        const T = Take.f;
        t.saved = T(&a, (c.layers + 1) * R * D * 2, &total);
        t.logits = T(&a, R * c.vocab * 2, &total);
        for (t.kept, 0..) |*k, l| {
            k.* = .{};
            switch (c.kinds[l]) {
                .mamba => {
                    k.proj = T(&a, R * pd * 2, &total);
                    k.ya = T(&a, R * xd * 2, &total);
                },
                .attention => {
                    k.x = T(&a, R * D * 2, &total);
                    k.ya = T(&a, R * qw * 2, &total);
                },
                .moe => {
                    k.x = T(&a, R * D * 2, &total);
                    k.upr = T(&a, R * F * 2, &total);
                    k.act = T(&a, R * S * W * 2, &total);
                    k.ys = T(&a, R * S * D * 2, &total);
                    k.ids = T(&a, R * S * 4, &total);
                    k.wts = T(&a, R * S * 4, &total);
                },
            }
        }
        for ([_]*u64{ &t.g, &t.dx, &t.gs, &t.dhn }) |p| p.* = T(&a, R * D * 4, &total);
        t.dxa = T(&a, R * sites_mod.max_rank * 4, &total);
        t.xb = T(&a, R * widest_k * 2, &total);
        t.wt = T(&a, widest_n * 2, &total);
        t.d_upr = T(&a, R * F * 4, &total);
        t.du = T(&a, R * F * 4, &total);
        t.dyp = T(&a, pairs * D * 4, &total);
        t.dxp = T(&a, pairs * D * 4, &total);
        t.d_y1r = T(&a, pairs * W * 4, &total);
        t.du1 = T(&a, pairs * W * 4, &total);
        t.y1s = T(&a, pairs * W * 2, &total);
        t.part = T(&a, ops.slices * pairs * @max(D, W) * 4, &total);
        t.order = T(&a, pairs * 4, &total);
        t.inv = T(&a, pairs * 4, &total);
        t.offsets = T(&a, c.experts * 4, &total);
        t.lg = T(&a, R * c.experts * 4, &total);
        for ([_]*u64{ &t.dyn, &t.ytot, &t.dytot }) |p| p.* = T(&a, R * xd * 4, &total);
        t.dt = T(&a, R * H * 4, &total);
        t.ddt = T(&a, R * H * 4, &total);
        t.dact = T(&a, R * cd * 4, &total);
        t.dproj = T(&a, R * pd * 4, &total);
        t.ckpt = T(&a, H * (R / 16 + 1) * size * 4, &total);
        t.states = T(&a, H * 16 * size * 4, &total);
        t.dbp = T(&a, R * H * c.state * 4, &total);
        t.dcp = T(&a, R * H * c.state * 4, &total);
        t.conv_out = T(&a, R * cd * 2, &total);
        t.act_m = T(&a, R * cd * 2, &total);
        t.qkv = T(&a, R * c.qkvDim() * 2, &total);
        t.d_o = T(&a, R * qw * 4, &total);
        t.dq = T(&a, R * qw * 4, &total);
        t.dk = T(&a, R * kw * 4, &total);
        t.dv = T(&a, R * kw * 4, &total);
        t.probs = T(&a, c.heads * R * R * 4, &total);
        t.dscores = T(&a, c.heads * R * R * 4, &total);
        t.gram = T(&a, 16 * 16 * 4, &total);
        if (!count) t.mem = a;
        return total;
    }

    fn targets(t: *const Trainer) []u32 {
        return t.host.slice(u32, max_rows);
    }

    fn weights(t: *const Trainer) []f32 {
        return t.host.slice(f32, 2 * max_rows)[max_rows..];
    }

    fn stats(t: *const Trainer) [][2]f32 {
        return t.host.slice([2]f32, 2 * max_rows)[max_rows..];
    }

    /// Layer i's output projection as loaded: the matrix its learned change sits after.
    pub fn outProj(t: *const Trainer, i: usize) Mat {
        return switch (t.raw[i]) {
            .mamba => |m| m.out_proj,
            .attention => |a| a.o,
            .moe => |m| m.down,
        };
    }

    /// The probability the last step gave row r's target token.
    pub fn targetProb(t: *const Trainer, r: usize) f32 {
        return t.stats()[r][1];
    }

    /// Each row along layer l's candidate directions ([rows, candidates]), as the last project step left them.
    pub fn projected(t: *const Trainer, l: usize, rows: usize) []const f32 {
        const k = sites_mod.candidates;
        const all = std.mem.bytesAsSlice(f32, t.host.host.bytes[proj_at..][0 .. t.c.layers * max_rows * k * 4]);
        return @alignCast(all[l * max_rows * k ..][0 .. rows * k]);
    }

    fn residual(t: *const Trainer, l: usize) u64 {
        return t.saved + @as(u64, l) * max_rows * t.c.hidden * 2;
    }

    /// Layer l's site input: what its change reads, rows `stride` apart, `width` wide (from the forward's scratch).
    fn siteInput(t: *const Trainer, l: usize) struct { x: u64, stride: usize } {
        const c = t.c;
        const b = &t.buf;
        return switch (c.kinds[l]) {
            .mamba => .{ .x = b.g, .stride = c.inner() },
            .attention => .{ .x = b.att, .stride = c.heads * c.head_dim },
            .moe => .{ .x = b.act + @as(u64, c.top_k) * c.expert_width * 2, .stride = c.slots() * c.expert_width },
        };
    }

    /// The mean loss of `ids`' answer from `start`, and what `mode` adds to it.
    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode) !Result {
        if (ids.len < 2 or start < 1 or start >= ids.len or ids.len - 1 > max_rows) return error.BadExample;
        if (mode != .loss and mode != .base and mode != .avoid and mode != .seek and mode != .project and t.sites.rank == 0) return error.NoOpenBlock;
        const full = t.sites.rank;
        if (mode == .base and full >= sites_mod.block) {
            t.sites.rank = full - sites_mod.block;
            t.sites.attach(&t.e.w, true);
        }
        defer if (mode == .base and full >= sites_mod.block) {
            t.sites.rank = full;
            t.sites.attach(&t.e.w, true);
        };
        const c = t.c;
        const e = t.e;
        const o = e.ops();
        const k = t.ops_();
        const b = &t.buf;
        const rows = ids.len - 1;
        const D: usize = c.hidden;
        const share = 1 / @as(f32, @floatFromInt(ids.len - start));
        for (t.targets()[0..rows], t.weights()[0..rows], 0..) |*tg, *w, r| {
            tg.* = ids[r + 1];
            w.* = if (r + 1 >= start) share else 0;
        }
        try o.upload(b.p_ids, std.mem.sliceAsBytes(ids[0..rows]));
        try t.seq.arena.?.buf.fill8(0, e.stream.handle);
        const f = fwd.Forward.init(c, &e.w, b, o, seq_len, e.nch, false);
        var w: fwd.Walk = .{ .rows = rows, .pos = 0 };
        try f.chunkBegin(&w);
        for (0..c.layers) |l| {
            try f.chunkPre(&w, l);
            try o.copy(t.residual(l), w.x, rows * D * 2);
            if (c.kinds[l] != .mamba) try o.copy(t.kept[l].x, b.y, rows * D * 2);
            try f.chunkMixer(&w, l);
            try f.chunkPost(&w, l);
            const in = t.siteInput(l);
            const s = &t.sites.list[l];
            switch (mode) {
                .loss, .base => {},
                .grad, .learn, .hold, .hold_learn => try t.keep(l, rows),
                .avoid, .seek => {
                    const y = if (mode == .avoid) s.avoid.dev else s.seek.dev;
                    const n: usize = if (mode == .avoid) sites_mod.avoid_dims else sites_mod.candidates;
                    const seed: u32 = if (mode == .avoid) 1 else 2;
                    const first = start - 1;
                    try k.sketch(in.x, in.stride, y, rows, s.in, n, t.sketched, seed, 1);
                    try k.sketch(in.x + @as(u64, first) * in.stride * 2, in.stride, y, 1, s.in, n, (1 << 30) + t.sketched + first, seed, @floatFromInt(rows - first));
                },
                .project => try k.project(in.x, in.stride, s.seek.dev, t.host.dev + proj_at + l * max_rows * sites_mod.candidates * 4, rows, s.in, sites_mod.candidates),
            }
        }
        if (mode == .avoid or mode == .seek or mode == .project) {
            try e.stream.synchronize();
            if (mode == .avoid or mode == .seek) t.sketched += rows;
            return .{ .loss = 0, .recalled = false };
        }
        try f.norm(&w.x, w.delta, e.w.norm_f, rows, false);
        try o.copy(t.residual(c.layers), w.x, rows * D * 2);
        try o.prefillDense(b.y, e.w.head, t.logits, rows);
        const tg = t.host.dev;
        try k.softmax(t.logits, tg, tg + max_rows * 4, tg + max_rows * 8, rows, c.vocab);
        t.hold = if (mode == .hold or mode == .hold_learn) hold_weight / @as(f32, @floatFromInt(rows)) else 0;
        if (mode != .loss and mode != .base) try t.backward(rows, mode == .learn or mode == .hold_learn);
        try e.stream.synchronize();
        var out: Result = .{ .loss = 0, .recalled = true };
        for (t.stats()[0..rows], t.weights()[0..rows]) |s, wgt| if (wgt > 0) {
            out.loss += s[0] * wgt;
            out.recalled = out.recalled and s[1] > 0.5;
        };
        return out;
    }

    /// Layer l's activations into its slot from the forward's scratch, for its backward.
    fn keep(t: *Trainer, l: usize, rows: usize) !void {
        const c = t.c;
        const o = t.e.ops();
        const b = &t.buf;
        const kp = t.kept[l];
        switch (c.kinds[l]) {
            .mamba => {
                try o.copy(kp.proj, b.proj, rows * c.projDim() * 2);
                try o.copy(kp.ya, b.g, rows * c.inner() * 2);
            },
            .attention => try o.copy(kp.ya, b.att, rows * c.heads * c.head_dim * 2),
            .moe => {
                const S = c.slots();
                const W = c.expert_width;
                try o.copy(kp.act, b.act, rows * S * W * 2);
                try t.ops_().copyRows(b.act + @as(u64, c.top_k) * W * 2, S * W, 0, kp.upr, c.shared_width, c.shared_width, rows);
                try o.copy(kp.ys, b.ymoe, rows * S * c.hidden * 2);
                try o.copy(kp.ids, b.pick, rows * S * 4);
                try o.copy(kp.wts, b.wts, rows * S * 4);
            },
        }
    }

    /// y [rows, m.k] (+)= x [rows, m.n] (fp32) times the matrix: its input's gradient from its output's.
    fn product(t: *Trainer, x: u64, m: Mat, y: u64, rows: usize, add: bool) !void {
        const k = t.ops_();
        try k.narrow(x, t.xb, rows * m.n);
        const right = if (m.bf16) m.w else blk: {
            try k.dequant(.{ m.w, m.s, m.b }, t.wt, m.n * m.k);
            break :blk t.wt;
        };
        try k.gemm(t.xb, right, y, rows, m.k, m.n, m.n, m.k, m.k, add);
    }

    /// The open block's gradient from its site's input x (rows `stride` apart) and output gradient gy; dx gains all.
    fn adapterBack(t: *Trainer, l: usize, x: u64, stride: usize, gy: u64, dx: u64, rows: usize) !void {
        const s = &t.sites;
        if (s.rank == 0) return;
        const site = &s.list[l];
        const k = t.ops_();
        const first = if (s.rank >= sites_mod.block) s.first() else 0;
        try k.loraIn(x, stride, site.a.dev, s.xa.dev, rows, site.in, s.rank, s.xn.dev);
        try k.gate(s.xa.dev, s.xn.dev, site.tau.dev, s.gates.dev, rows, s.rank, sites_mod.unit);
        try k.loraDb(s.xa.dev, gy, site.gb.dev, rows, site.out, s.rank, first, sites_mod.scale);
        if (t.hold > 0) try k.hold(s.xa.dev, t.gram, site.b.dev, site.gb.dev, rows, site.out, s.rank, first, t.hold * sites_mod.scale * sites_mod.scale);
        try k.loraDxa(gy, site.b.dev, t.dxa, rows, site.out, s.rank, sites_mod.scale, s.gates.dev);
        try k.loraDx(t.dxa, site.a.dev, dx, rows, site.in, s.rank);
    }

    /// The loss back through the head and every layer to each change, Adam after when `adam`.
    fn backward(t: *Trainer, rows: usize, adam: bool) !void {
        const c = t.c;
        const e = t.e;
        const k = t.ops_();
        const D: usize = c.hidden;
        try k.gemm(t.logits, t.head_t, t.dx, rows, D, c.vocab, c.vocab, D, D, false);
        try k.zero(t.g, rows * D);
        try k.rmsBack(t.residual(c.layers), e.w.norm_f, t.dx, t.g, rows, D, c.eps);
        var l = c.layers;
        while (l > 0) {
            l -= 1;
            switch (t.raw[l]) {
                .moe => |m| try t.moe(l, m, rows),
                .mamba => |m| try t.mamba(l, m, rows),
                .attention => |a| try t.attention(l, a, rows),
            }
            try k.rmsBack(t.residual(l), e.w.blocks[l].norm, t.dx, t.g, rows, D, c.eps);
        }
        if (adam) try t.sites.adam(k);
    }

    /// The MoE layer: the shared expert and its change (at its slot's weight), the routed experts, the routing weights.
    fn moe(t: *Trainer, l: usize, m: @FieldType(Raw, "moe"), rows: usize) !void {
        const c = t.c;
        const k = t.ops_();
        const kp = t.kept[l];
        const D: usize = c.hidden;
        const S = c.slots();
        const W = c.expert_width;
        const F = c.shared_width;
        try k.scaleRows(t.g, kp.wts, S, c.top_k, t.gs, rows, D);
        try t.product(t.gs, m.down, t.d_upr, rows, false);
        try t.adapterBack(l, kp.upr, F, t.gs, t.d_upr, rows);
        try k.relu2Back(kp.upr, t.d_upr, t.du, rows * F);
        try t.product(t.du, m.up, t.dx, rows, false);
        const count = rows * c.top_k;
        try k.sortPairs(kp.ids, rows, S, c.top_k, c.experts, t.order, t.inv, t.offsets);
        try k.pairsIn(t.order, kp.wts, t.g, t.dyp, D, S, count);
        try k.expertProducts(t.dyp, m.fc2, t.offsets, t.part, t.d_y1r, count, D, W, c.experts);
        try k.copyRows(kp.act, W, t.order, t.y1s, W, W, count);
        try k.relu2Back(t.y1s, t.d_y1r, t.du1, count * W);
        try k.expertProducts(t.du1, m.fc1, t.offsets, t.part, t.dxp, count, W, D, c.experts);
        try k.pairsOut(t.inv, t.dxp, t.dx, D, c.top_k, rows);
        const router = t.e.w.blocks[l].moe.router;
        try k.routerLogits(kp.x, router, t.lg, rows, c.experts, D);
        try k.routeBack(t.lg, kp.ids, S, kp.ys, t.g, router, t.dx, rows, c.experts, c.top_k, D, c.routed_scaling);
    }

    /// The Mamba-2 layer: out_proj and its change, the gated norm, the scan, dt, the conv, in_proj.
    fn mamba(t: *Trainer, l: usize, m: @FieldType(Raw, "mamba"), rows: usize) !void {
        const c = t.c;
        const k = t.ops_();
        const kp = t.kept[l];
        const mw = t.e.w.blocks[l].mamba;
        const s: ops.Shape = .{ .rows = @intCast(rows), .heads = @intCast(c.mamba_heads), .dh = @intCast(c.mamba_head_dim), .groups = @intCast(c.groups), .n = @intCast(c.state), .inner = @intCast(c.inner()), .conv = @intCast(c.convDim()), .proj = @intCast(c.projDim()) };
        try t.product(t.g, m.out_proj, t.dyn, rows, false);
        try t.adapterBack(l, kp.ya, c.inner(), t.g, t.dyn, rows);
        try k.convPre(kp.proj, mw.conv_w, mw.conv_b, t.conv_out, t.act_m, s, c.conv_kernel);
        try k.dt(kp.proj, mw.dt_bias, t.dt, s);
        try k.ssmFwd(t.act_m, t.dt, mw.a, mw.d, t.ytot, t.ckpt, s);
        try k.gateBack(t.ytot, kp.proj, mw.gnorm, t.dyn, t.dytot, t.dproj, s, c.eps);
        try k.ssmBack(.{ t.act_m, t.dt, mw.a, mw.d, t.dytot, t.ckpt, t.states, t.dact, t.dbp, t.dcp, t.ddt }, s);
        try k.ssmBc(t.dbp, t.dcp, t.dact, s);
        try k.convBack(t.dact, t.conv_out, mw.conv_w, mw.conv_b, t.dproj, s, c.conv_kernel);
        try k.dtBack(kp.proj, mw.dt_bias, t.ddt, t.dproj, s);
        try t.product(t.dproj, m.in_proj, t.dx, rows, false);
    }

    /// The attention layer: o_proj and its change, causal softmax attention with shared kv heads, q, k and v.
    fn attention(t: *Trainer, l: usize, a: @FieldType(Raw, "attention"), rows: usize) !void {
        const c = t.c;
        const k = t.ops_();
        const kp = t.kept[l];
        const qw = c.heads * c.head_dim;
        try t.e.ops().prefillDense(kp.x, t.e.w.blocks[l].attn.qkv, t.qkv, rows);
        try t.product(t.g, a.o, t.d_o, rows, false);
        try t.adapterBack(l, kp.ya, qw, t.g, t.d_o, rows);
        const h: ops.Heads = .{ .rows = @intCast(rows), .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .dim = @intCast(c.head_dim), .qkv = @intCast(c.qkvDim()), .scale = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(c.head_dim)))) };
        try k.attention(t.qkv, t.d_o, t.probs, t.dscores, t.dq, t.dk, t.dv, h);
        try t.product(t.dq, a.q, t.dx, rows, false);
        try t.product(t.dk, a.k, t.dx, rows, true);
        try t.product(t.dv, a.v, t.dx, rows, true);
    }
};
