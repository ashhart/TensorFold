//! Sage-A1.5 (qwen3_5_moe, 35B-A3B) on Intel Xe2 over Level Zero, first form: MLX affine 4/64 weights, decode kernels
//! only (prompts go through in windows of at most 16 rows), one sequence. It reuses the 27B's device layer, loaders and
//! kernels, built at Sage's shape (shape.h, sage_glue and sage_decode), plus the MoE ops of sage_moe.cl.
const std = @import("std");
const ze = @import("ze");
const kb = @import("xpu_qwen27_kernels");
const cfgm = @import("sage_config");
const sl = @import("sage_load");
const dev = @import("dev.zig");
const loader = @import("load.zig");
const fwd = @import("forward.zig");
const q = @import("qwen27.zig"); // the profiler state (XPU_PROFILE) the Dev records into
const Dev = dev.Dev;
const Checkpoint = dev.Checkpoint;
const Proj = loader.Proj;
const off = dev.off;
const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;

pub const EPS: f32 = 1e-6;
pub const SPLIT = 32; // keys per attention split (decode.cl)
/// Rows of one forward: the decode kernels' limit (the chain table and the serial GDN).
pub const WINDOW = 16;
/// Query heads an attn_decode subgroup holds at Sage's shape (sage_config.sage_defines: Q27_QPG).
const QPG = 4;

/// A layer's routed experts: gate|up rows of an expert stacked (2 ff rows), and down, MLX affine 4/64 as stored.
pub const Experts = struct { gu_w: *anyopaque, gu_s: *anyopaque, gu_b: *anyopaque, dn_w: *anyopaque, dn_s: *anyopaque, dn_b: *anyopaque };

pub const Layer = struct {
    in_norm: *anyopaque,
    post_norm: *anyopaque,
    attn: bool,
    mixer_in: Proj, // GDN: qkv | z | a | b ; attention: (q | gate) per head | k | v
    mixer_out: Proj,
    router: *anyopaque, // fp16 [experts, D]
    shared_gate: *anyopaque, // fp16 [D]
    experts: Experts,
    shared: Experts, // one expert, the same layout
    // GDN
    conv_w: ?*anyopaque = null,
    alog: ?*anyopaque = null,
    dt: ?*anyopaque = null,
    gnorm: ?*anyopaque = null,
    rec: ?*anyopaque = null, // fp32 [GV, 128, 128]
    conv: [2]?*anyopaque = .{ null, null }, // fp16 [3, NCONV], ping-pong
    conv_cur: u1 = 0,
    // attention
    qn: ?*anyopaque = null,
    kn: ?*anyopaque = null,
    kc: ?*anyopaque = null, // bf16 [ctx, AKV, 256]
    vc: ?*anyopaque = null,
};

pub const Model = struct {
    embed_w: *anyopaque,
    embed_s: *anyopaque,
    embed_b: *anyopaque,
    layers: []Layer,
    norm: *anyopaque,
    head: Proj,
    invf: *anyopaque,
};

/// The sizes derived from the shape, as the kernels compiled with sage_defines see them.
pub const Sizes = struct {
    s: cfgm.Shape,
    nconv: u32,
    vd: u32,
    od: u32,
    gdn_in: u32,
    att_in: u32,

    pub fn of(s: cfgm.Shape) Sizes {
        const nconv = 2 * s.gdn_k_heads * 128 + s.gdn_v_heads * 128;
        const vd = s.gdn_v_heads * 128;
        return .{ .s = s, .nconv = nconv, .vd = vd, .od = s.q_heads * 256, .gdn_in = (nconv + vd + 2 * s.gdn_v_heads + 255) / 256 * 256, .att_in = s.attIn() };
    }
};

pub const Acts = struct {
    x: *anyopaque, // fp32 [WINDOW, D]
    h: *anyopaque, // fp16 [WINDOW, widest GEMM input]
    ahi: *anyopaque,
    alo: *anyopaque,
    r: *anyopaque, // fp16 [WINDOW, D]
    p: *anyopaque, // fp16 [WINDOW, widest mixer input projection]
    q: *anyopaque,
    k: *anyopaque,
    v: *anyopaque,
    g: *anyopaque,
    beta: *anyopaque,
    y: *anyopaque,
    pm: *anyopaque,
    pl: *anyopaque,
    po: *anyopaque,
    pscr: *anyopaque,
    logits: *anyopaque, // fp16 [WINDOW, VOCAB]
    tok: *anyopaque,
    pv: *anyopaque,
    pi: *anyopaque,
    ids: *anyopaque,
    chain: *anyopaque,
    // MoE
    mlog: *anyopaque, // fp32 [WINDOW, experts + 1]
    mids: *anyopaque, // u32 [WINDOW, top_k]
    mwts: *anyopaque, // fp32 [WINDOW, top_k]
    msw: *anyopaque, // fp32 [WINDOW]: the shared expert's gate
    mact: *anyopaque, // fp16 [WINDOW (top_k + 1), ff]
    my: *anyopaque, // fp32 [WINDOW (top_k + 1), D]

    pub fn init(d: *Dev, z: Sizes, ctx: u32) !Acts {
        const s = z.s;
        const w: usize = WINDOW;
        const widest_in: usize = @max(s.hidden, @max(z.vd, z.od));
        const ns: usize = (ctx + SPLIT - 1) / SPLIT;
        const pairs: usize = w * (s.top_k + 1);
        return .{
            .x = try d.alloc_(w * s.hidden * 4),
            .h = try d.alloc_(w * widest_in * 2),
            .ahi = try d.alloc_(w * (widest_in / 64) * 2),
            .alo = try d.alloc_(w * (widest_in / 64) * 2),
            .r = try d.alloc_(w * s.hidden * 2),
            .p = try d.alloc_(w * @max(z.gdn_in, z.att_in) * 2),
            .q = try d.alloc_(w * @max(s.q_heads * 256, s.gdn_k_heads * 128) * 2),
            .k = try d.alloc_(w * s.gdn_k_heads * 128 * 2),
            .v = try d.alloc_(w * z.vd * 2),
            .g = try d.alloc_(w * s.gdn_v_heads * 4),
            .beta = try d.alloc_(w * s.gdn_v_heads * 4),
            .y = try d.alloc_(w * @max(z.vd, z.od) * 2),
            .pm = try d.alloc_(w * s.q_heads * ns * 4),
            .pl = try d.alloc_(w * s.q_heads * ns * 4),
            .po = try d.alloc_(w * s.q_heads * ns * 256 * 4),
            .pscr = try d.alloc_(w * s.hidden * 4),
            .logits = try d.alloc_(w * s.vocab * 2),
            .tok = try d.alloc_(w * 4),
            .pv = try d.alloc_(w * 1024),
            .pi = try d.alloc_(w * 1024),
            .ids = try d.alloc_(w * 4),
            .chain = try d.upload(std.mem.sliceAsBytes(&fwd.chainTable())),
            .mlog = try d.alloc_(w * (s.experts + 1) * 4),
            .mids = try d.alloc_(w * s.top_k * 4),
            .mwts = try d.alloc_(w * s.top_k * 4),
            .msw = try d.alloc_(w * 4),
            .mact = try d.alloc_(pairs * @max(s.expert_ff, s.shared_ff) * 2),
            .my = try d.alloc_(pairs * s.hidden * 4),
        };
    }
};

/// sage_moe.cl's kernels (the Dev's table is the 27B runner's, filled from the Sage images).
pub const MoeKernels = struct { router: ze.H, route: ze.H, swiglu: ze.H, down: ze.H, combine: ze.H };

// ---------------------------------------------------------------- loading

fn name(comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(alloc, f, args) catch @panic("out of memory");
}

fn halfTensor(d: *Dev, c: *const Checkpoint, p: []const u8) !*anyopaque {
    const h = try sl.halfRows(alloc, c, p);
    defer alloc.free(h);
    return d.upload(std.mem.sliceAsBytes(h));
}

/// Device copies of one expert table: sage_load packs it, this sink copies each piece into its buffer.
const TableSink = struct {
    d: *Dev,
    x: Experts,
    pub fn put(s: *TableSink, part: sl.Part, at: usize, b: []const u8) !void {
        const dst = switch (part) {
            .gu_w => s.x.gu_w,
            .gu_s => s.x.gu_s,
            .gu_b => s.x.gu_b,
            .dn_w => s.x.dn_w,
            .dn_s => s.x.dn_s,
            .dn_b => s.x.dn_b,
        };
        try s.d.copyIn(off(dst, at), b);
    }
};

fn experts(d: *Dev, c: *const Checkpoint, p: []const u8, count: u32, shared: bool, ff: u32, D: u32) !Experts {
    const z = sl.Sizes.of(ff, D);
    const n: usize = count;
    var s: TableSink = .{ .d = d, .x = .{
        .gu_w = try d.alloc_(n * z.gu_w),
        .gu_s = try d.alloc_(n * z.gu_g),
        .gu_b = try d.alloc_(n * z.gu_g),
        .dn_w = try d.alloc_(n * z.dn_w),
        .dn_s = try d.alloc_(n * z.dn_g),
        .dn_b = try d.alloc_(n * z.dn_g),
    } };
    try sl.packExperts(c, p, count, shared, ff, D, &s);
    return s.x;
}

/// MLX affine 4/64 only in this form; the scales of the experts must be BF16 (sage_moe.cl reads bf16). Checked on the
/// Sage MLX-4bit checkpoint by tools/xpu/sage_mlx_check.py: stacked switch_mlp experts, 8 bit router, conv1d [NCONV, 4, 1].
pub fn load(d: *Dev, c: *const Checkpoint, z: Sizes, ctx: u32) !Model {
    const s = z.s;
    if (c.has("in_proj_qkv.qweight") or c.has("in_proj_qkv.trellis")) return error.OnlyMlxInThisForm;
    const stage = try d.alloc_(800 << 20);
    var m: Model = undefined;
    m.embed_w = try d.upload((try c.get("embed_tokens.weight")).bytes);
    m.embed_s = try loader.bf16Tensor(d, c, "embed_tokens.scales");
    m.embed_b = try loader.bf16Tensor(d, c, "embed_tokens.biases");
    m.norm = try loader.bf16Tensor(d, c, "model.norm.weight");
    m.head = try loader.projection(d, c, "lm_head", &.{""}, true, false, stage);
    var invf: [32]f32 = undefined;
    const half_dims: f64 = @floatFromInt(s.rope_dims / 2);
    for (&invf, 0..) |*f, i| f.* = @floatCast(std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / half_dims));
    m.invf = try d.upload(std.mem.sliceAsBytes(invf[0 .. s.rope_dims / 2]));
    m.layers = try alloc.alloc(Layer, s.layers);
    const kv_bytes = @as(usize, ctx) * s.kv_heads * 256 * 2;
    for (m.layers, 0..) |*l, i| {
        const p = name("model.layers.{d}.", .{i});
        const attn = (i + 1) % 4 == 0;
        l.* = .{
            .in_norm = try loader.bf16Tensor(d, c, name("{s}input_layernorm.weight", .{p})),
            .post_norm = try loader.bf16Tensor(d, c, name("{s}post_attention_layernorm.weight", .{p})),
            .attn = attn,
            .mixer_in = undefined,
            .mixer_out = undefined,
            .router = try halfTensor(d, c, name("{s}mlp.gate", .{p})),
            .shared_gate = try halfTensor(d, c, name("{s}mlp.shared_expert_gate", .{p})),
            .experts = try experts(d, c, p, s.experts, false, s.expert_ff, s.hidden),
            .shared = try experts(d, c, p, 1, true, s.shared_ff, s.hidden),
        };
        if (attn) {
            l.mixer_in = try loader.projection(d, c, name("{s}self_attn.", .{p}), &.{ "q_proj", "k_proj", "v_proj" }, true, false, stage);
            l.mixer_out = try loader.projection(d, c, name("{s}self_attn.", .{p}), &.{"o_proj"}, false, false, stage);
            if (l.mixer_in.n != z.att_in) return error.UnexpectedShape;
            l.qn = try loader.bf16Tensor(d, c, name("{s}self_attn.q_norm.weight", .{p}));
            l.kn = try loader.bf16Tensor(d, c, name("{s}self_attn.k_norm.weight", .{p}));
            l.kc = try d.alloc_(kv_bytes);
            l.vc = try d.alloc_(kv_bytes);
        } else {
            l.mixer_in = try loader.projection(d, c, name("{s}linear_attn.", .{p}), &.{ "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b" }, true, false, stage);
            l.mixer_out = try loader.projection(d, c, name("{s}linear_attn.", .{p}), &.{"out_proj"}, false, false, stage);
            if (l.mixer_in.n != z.gdn_in) return error.UnexpectedShape;
            l.conv_w = try loader.bf16Tensor(d, c, name("{s}linear_attn.conv1d.weight", .{p}));
            l.alog = try loader.bf16Tensor(d, c, name("{s}linear_attn.A_log", .{p}));
            l.dt = try loader.bf16Tensor(d, c, name("{s}linear_attn.dt_bias", .{p}));
            l.gnorm = try loader.bf16Tensor(d, c, name("{s}linear_attn.norm.weight", .{p}));
            l.rec = try d.alloc_(@as(usize, s.gdn_v_heads) * 128 * 128 * 4);
            for (&l.conv) |*cv| cv.* = try d.alloc_(3 * @as(usize, z.nconv) * 2);
        }
        if (i % 8 == 7) std.debug.print("  layers loaded: {d}/{d}\n", .{ i + 1, s.layers });
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 800 << 20;
    return m;
}

// ---------------------------------------------------------------- the runner

pub const Runner = struct {
    d: Dev,
    mk: MoeKernels,
    z: Sizes,
    model: Model,
    acts: Acts,
    ckpt: Checkpoint,
    ctx: u32,
    pos: u32 = 0,
    last: i32 = -1, // the greedy token after the last fed row

    pub fn init(dir: []const u8, ctx: u32) !*Runner {
        const z = Sizes.of(try readShape(dir));
        var ckpt: Checkpoint = .{};
        errdefer ckpt.deinit();
        try addShards(&ckpt, dir);
        const r = try alloc.create(Runner);
        errdefer alloc.destroy(r);
        r.* = .{ .d = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined }, .mk = undefined, .z = z, .model = undefined, .acts = undefined, .ckpt = ckpt, .ctx = ctx };
        const d = &r.d;
        errdefer d.g.close();
        d.list = try d.g.immediate(0, 0);
        try kernels(d, &r.mk);
        if (std.c.getenv("XPU_PROFILE") != null) {
            q.prof.events = try d.g.events(4096);
            inline for (@typeInfo(@TypeOf(d.k)).@"struct".field_names, 0..) |n, i| q.prof.names[i] = .{ .k = @field(d.k, n), .name = n };
            const base = @typeInfo(@TypeOf(d.k)).@"struct".field_names.len;
            inline for (@typeInfo(MoeKernels).@"struct".field_names, 0..) |n, i| q.prof.names[base + i] = .{ .k = @field(r.mk, n), .name = "moe_" ++ n };
        }
        r.model = try load(d, &r.ckpt, z, ctx);
        q.prof.on = std.c.getenv("XPU_PROFILE") != null; // not during the load
        r.acts = try Acts.init(d, z, ctx);
        try r.reset();
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.d.deinit();
        r.ckpt.deinit();
        alloc.free(r.model.layers);
        alloc.destroy(r);
    }

    /// A fresh sequence: zero GDN and convolution states, position 0 (the K/V rows are overwritten as they are fed).
    pub fn reset(r: *Runner) !void {
        const s = r.z.s;
        for (r.model.layers) |*l| if (!l.attn) {
            try r.d.fill(l.rec.?, @as(u32, 0), @as(usize, s.gdn_v_heads) * 128 * 128 * 4);
            for (l.conv) |cv| try r.d.fill(cv.?, @as(u16, 0), 3 * @as(usize, r.z.nconv) * 2);
            l.conv_cur = 0;
        };
        r.pos = 0;
        r.last = -1;
    }

    /// Feeds tokens in windows of WINDOW rows; `last` afterwards is the greedy token after the final one.
    pub fn prompt(r: *Runner, ids: []const i32) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const n = @min(WINDOW, ids.len - at);
            r.last = try r.forward(ids[at..][0..n], at + n == ids.len);
            at += n;
        }
    }

    /// The fp16 logits of the last forward's final row (`out.len` is the vocabulary size).
    pub fn logits(r: *Runner, out: []f16) !void {
        if (out.len != r.z.s.vocab) return error.BadLength;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, out.ptr, r.acts.logits, out.len * 2, null, 0, null));
        try r.d.sync();
    }

    /// One serial decode token.
    pub fn step(r: *Runner, token: i32) !void {
        r.last = try r.forward(&.{token}, true);
    }

    /// One forward over at most WINDOW rows at r.pos; the last row's greedy token when `want`, else -1.
    fn forward(r: *Runner, ids: []const i32, want: bool) !i32 {
        if (ids.len == 0 or ids.len > WINDOW) return error.BadWindow;
        if (r.pos + ids.len > r.ctx) return error.ContextFull;
        const d = &r.d;
        const a = &r.acts;
        const m = &r.model;
        const z = r.z;
        const s = z.s;
        const T: u32 = @intCast(ids.len);
        const t32: i32 = @intCast(T);
        const p0: i32 = @intCast(r.pos);
        const D: i32 = @intCast(s.hidden);
        const gs: i32 = 64;
        const nul: ?*anyopaque = null;
        const zero: i32 = 0;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, a.ids, ids.ptr, ids.len * 4, null, 0, null));
        errdefer d.settle();
        try d.launch(d.k.embed, .{ .x = T }, .{ a.ids, m.embed_w, m.embed_s, m.embed_b, a.x, D });
        for (m.layers, 0..) |*l, i| {
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.in_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, if (i == 0) 0 else 1), gs, nul, nul, zero, zero });
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_in, a.p, T);
            if (l.attn) {
                const ns: u32 = (r.pos + T + SPLIT - 1) / SPLIT;
                try d.launch(d.k.attn_prep, .{ .x = T }, .{ a.p, l.qn.?, l.kn.?, m.invf, a.q, l.kc.?, l.vc.?, p0, EPS, nul, nul, nul, zero, zero });
                try d.launch(d.k.attn_decode, .{ .x = ns, .y = s.kv_heads * (s.q_heads / s.kv_heads / QPG), .z = T }, .{ a.q, l.kc.?, l.vc.?, a.pm, a.pl, a.po, p0, @as(i32, @intCast(ns)), @as(f32, 1.0 / 16.0), a.chain, nul, nul, zero });
                try d.launch(d.k.attn_merge, .{ .x = s.q_heads, .y = T }, .{ a.pm, a.pl, a.po, a.y, p0, @as(i32, @intCast(ns)), a.chain, nul });
                try d.launch(d.k.attn_gate, .{ .x = T }, .{ a.y, a.p, a.h, a.ahi, a.alo, gs, zero, nul, nul, zero });
            } else {
                const cin = l.conv[l.conv_cur].?;
                const cout = l.conv[1 - l.conv_cur].?;
                l.conv_cur = 1 - l.conv_cur;
                try d.launch(d.k.gdn_pre, .{ .x = T, .y = z.nconv / 256 }, .{ a.p, cin, cout, l.conv_w.?, l.alog.?, l.dt.?, a.q, a.k, a.v, a.g, a.beta, t32, nul, nul });
                try d.launch(d.k.gdn_step, .{ .x = 16, .y = s.gdn_v_heads, .z = 1 }, .{ a.q, a.k, a.v, a.g, a.beta, l.rec.?, l.rec.?, a.y, t32, nul, nul, nul, nul });
                try d.launch(d.k.gated_norm, .{ .x = T }, .{ a.y, a.p, l.gnorm.?, a.h, a.ahi, a.alo, EPS, gs, zero, nul, nul, zero });
            }
            try fwd.gemm(d, a.pscr, a.h, a.ahi, a.alo, l.mixer_out, a.r, T);
            try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, l.post_norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), gs, nul, nul, zero, zero });
            try r.moe(l, T);
        }
        r.pos += T;
        if (!want) {
            try d.sync();
            return -1;
        }
        const hgs: i32 = @intCast(m.head.gs);
        try d.launch(d.k.add_rmsnorm_dec, .{ .x = T }, .{ a.x, a.r, m.norm, a.h, a.ahi, a.alo, D, EPS, @as(i32, 1), hgs, nul, nul, zero, zero });
        const first = T - 1;
        const kg = s.hidden / m.head.gs;
        try fwd.gemm(d, a.pscr, off(a.h, first * s.hidden * 2), off(a.ahi, first * kg * 2), off(a.alo, first * kg * 2), m.head, a.logits, 1);
        try d.launch(d.k.argmax_part, .{ .x = 256, .y = 1 }, .{ a.logits, @as(i32, @intCast(s.vocab)), a.pv, a.pi });
        try d.launch(d.k.argmax, .{ .x = 1 }, .{ a.pv, a.pi, a.tok });
        var tok: [1]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &tok, a.tok, 4, null, 0, null));
        try d.sync();
        return tok[0];
    }

    /// The MoE block of `T` rows: h (the post-attention norm) to r (its output, the residual's next addend).
    fn moe(r: *Runner, l: *const Layer, T: u32) !void {
        const d = &r.d;
        const a = &r.acts;
        const s = r.z.s;
        const pairs = T * s.top_k;
        try d.launch(r.mk.router, .{ .x = s.experts + 1, .y = T }, .{ a.h, l.router, l.shared_gate, a.mlog, s.hidden });
        try d.launch(r.mk.route, .{ .x = T }, .{ a.mlog, a.mids, a.mwts, a.msw, s.experts, s.top_k });
        if (s.expert_ff != s.shared_ff) return error.SharedWidthNotWired; // one act stride for both (Sage: 512, 512)
        try d.launch(r.mk.swiglu, .{ .x = s.expert_ff, .y = pairs + T }, .{ l.experts.gu_w, l.experts.gu_s, l.experts.gu_b, l.shared.gu_w, l.shared.gu_s, l.shared.gu_b, a.h, a.mids, a.mact, s.hidden, s.expert_ff, s.top_k, pairs });
        try d.launch(r.mk.down, .{ .x = s.hidden, .y = pairs + T }, .{ l.experts.dn_w, l.experts.dn_s, l.experts.dn_b, l.shared.dn_w, l.shared.dn_s, l.shared.dn_b, a.mact, a.mids, a.my, s.hidden, s.expert_ff, pairs });
        try d.launch(r.mk.combine, .{ .x = s.hidden / 64, .y = T }, .{ a.my, a.mwts, a.msw, a.r, s.hidden, s.top_k, pairs });
    }
};

/// The Dev's kernel table from the Sage images; the 27B's prompt attention (6 query heads a K/V head) has no Sage
/// build, so its two slots hold attn_decode and this runner never launches them.
fn kernels(d: *Dev, mk: *MoeKernels) !void {
    const mg = try d.g.module(&kb.sage_glue);
    const md = try d.g.module(&kb.sage_decode);
    const mm = try d.g.module(&kb.gemm);
    const mgd = try d.g.module(&kb.gdn);
    const mw = try d.g.module(&kb.ngen_wide);
    const mn = try d.g.module(&kb.ngen_narrow);
    const mm128 = try d.g.module(&kb.gemm128);
    const mw128 = try d.g.module(&kb.ngen_wide128);
    const mn128 = try d.g.module(&kb.ngen_narrow128);
    const mo = try d.g.module(&kb.sage_moe);
    const attn_decode = try d.kernel(md, "attn_decode", 16);
    d.k = .{
        .embed = try d.kernel(mg, "embed", 256),
        .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 128),
        .add_rmsnorm_dec = try d.kernel(mg, "add_rmsnorm_dec", 512),
        .gdn_pre = try d.kernel(mg, "gdn_pre", 256),
        .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256),
        .gated_norm = try d.kernel(mg, "gated_norm", 256),
        .attn_prep = try d.kernel(mg, "attn_prep", 256),
        .attn_gate = try d.kernel(mg, "attn_gate", 256),
        .swiglu = try d.kernel(mg, "swiglu", 256),
        .argmax = try d.kernel(mg, "argmax", 256),
        .argmax_part = try d.kernel(mg, "argmax_part", 256),
        .argmax_part_masked = try d.kernel(mg, "argmax_part_masked", 256),
        .tap = try d.kernel(mg, "tap", 256),
        .conv_commit = try d.kernel(mg, "conv_commit", 256),
        .repack = try d.kernel(mg, "repack", 16),
        .gdn_step = try d.kernel(md, "gdn_step", 16),
        .attn_decode = attn_decode,
        .attn_merge = try d.kernel(md, "attn_merge", 16),
        .kv_move = try d.kernel(md, "kv_move", 16),
        .exl3_gemv = try d.kernel(md, "exl3_gemv", 16),
        .rot_heads = try d.kernel(mg, "rot_heads", 16),
        .attn_prompt = attn_decode, // not built at Sage's shape, never launched here
        .attn_prompt8 = attn_decode,
        .embed_dense = try d.kernel(mg, "embed_dense", 256),
        .dense_rows = try d.kernel(mg, "dense_rows", 256),
        .repack_gptq = try d.kernel(mg, "repack_gptq", 16),
        .quantize_mlx = try d.kernel(mg, "quantize_mlx", 16),
        .dec_wide_g128 = try d.kernel(mm128, "w4k1_decode", 64),
        .dec_wide64_g128 = try d.kernel(mm128, "w4k1_d64", 64),
        .dec_narrow_g128 = try d.kernel(mm128, "w4k_dsplit", 64),
        .ngen_wide_g128 = try d.kernel(mw128, "w4g_dpb", 256),
        .ngen_narrow_g128 = try d.kernel(mn128, "w4g_s4dpb", 256),
        .ngen_swiglu = try d.kernel(try d.g.module(&kb.ngen_swiglu), "w4g_dpb_swiglu", 256),
        .ngen_swiglu_g128 = try d.kernel(try d.g.module(&kb.ngen_swiglu128), "w4g_dpb_swiglu", 256),
        .rowsums_ff = try d.kernel(mg, "rowsums_ff", 128),
        .fakequant_rows = try d.kernel(mg, "fakequant_rows", 256),
        .a8 = try d.kernel(try d.g.module(&kb.ngen_a8), "w4a8", 256),
        .a8_swiglu = try d.kernel(try d.g.module(&kb.ngen_a8_swiglu), "w4a8_swiglu", 256),
        .quant_a8 = try d.kernel(mg, "quant_a8", 128),
        .quant_a4 = try d.kernel(mg, "quant_a4", 128),
        .a4 = try d.kernel(try d.g.module(&kb.ngen_a4), "w4a4", 256),
        .dec_wide = try d.kernel(mm, "w4k1_decode", 64),
        .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64),
        .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
        .gdn_prep = try d.kernel(mgd, "gdn_prep", 64),
        .gdn_state = try d.kernel(mgd, "gdn_state", 256),
        .gdn_out = try d.kernel(mgd, "gdn_out", 128),
        .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
        .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256),
    };
    mk.* = .{
        .router = try d.kernel(mo, "sage_router", 16),
        .route = try d.kernel(mo, "sage_route", 16),
        .swiglu = try d.kernel(mo, "sage_expert_swiglu", 16),
        .down = try d.kernel(mo, "sage_expert_down", 16),
        .combine = try d.kernel(mo, "sage_moe_combine", 64),
    };
}

fn readFile(path: []const u8) ![]u8 {
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);
    const fd = open(z.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    defer _ = close(fd);
    var out: std.ArrayList(u8) = .empty;
    var buf: [1 << 16]u8 = undefined;
    while (true) {
        const n = read(fd, &buf, buf.len);
        if (n < 0) return error.ReadFailed;
        if (n == 0) break;
        try out.appendSlice(alloc, buf[0..@intCast(n)]);
    }
    return out.toOwnedSlice(alloc);
}

fn readShape(dir: []const u8) !cfgm.Shape {
    const bytes = try readFile(name("{s}/config.json", .{dir}));
    defer alloc.free(bytes);
    const p = try cfgm.parse(alloc, bytes);
    defer p.deinit();
    const s = try cfgm.validate(p.value);
    var buf: [1024]u8 = undefined;
    // the images were built with sage_defines: a checkpoint of another shape would run the wrong kernels
    if (!std.mem.eql(u8, try cfgm.kernelDefines(&buf, s, QPG), cfgm.sage_defines)) return error.ShapeNotBuilt;
    return s;
}

fn addShards(c: *Checkpoint, dir: []const u8) !void {
    const bytes = try readFile(name("{s}/model.safetensors.index.json", .{dir}));
    defer alloc.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, bytes, .{}) catch return error.BadIndex;
    defer parsed.deinit();
    const map = if (parsed.value == .object) parsed.value.object.get("weight_map") else null;
    if (map == null or map.? != .object) return error.BadIndex;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(alloc);
    var it = map.?.object.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .string) return error.BadIndex;
        const f = e.value_ptr.string;
        if (seen.contains(f)) continue;
        try seen.put(alloc, f, {});
        const shard = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f }, 0);
        defer alloc.free(shard);
        try c.addShard(shard);
    }
}
