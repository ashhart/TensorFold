//! LFM3-VL's vision tower and projector on the Intel GPU: merge-order patches to the text model's image rows.

const std = @import("std");
const xpu = @import("xpu");
const rt = xpu.rt;
const ld = xpu.loader;

const Buf = rt.Buffer;
const spec = @import("vision_spec.zig");
const vs = spec.lfm3_vl_2b;
pub const width: u32 = @intCast(vs.hidden);
pub const intermediate: u32 = @intCast(vs.intermediate);
pub const depth = vs.layers;
pub const patch_width: u32 = @intCast(vs.patchWidth());
pub const out_width: u32 = @intCast(vs.text_hidden);
const heads: u32 = @intCast(vs.heads);
const kv_heads: u32 = @intCast(vs.kv_heads);
const head_dim: u32 = @intCast(vs.headDim());
const proj_hidden: u32 = @intCast(vs.projector_hidden);
const eps: f32 = 1e-5;
const ln_eps: f32 = 1e-6;

const Conv = struct { in: Buf, out: Buf, spatial: Buf };
const Attention = struct { q: Buf, k: Buf, v: Buf, out: Buf, q_norm: Buf, k_norm: Buf };
const Layer = struct { operator_norm: Buf, ffn_norm: Buf, w1: Buf, w3: Buf, w2: Buf, mixer: union(enum) { conv: Conv, attention: Attention } };

const Kernels = struct { gemm: rt.Kernel, rms: rt.Kernel, add_rms: rt.Kernel, ln: rt.Kernel, gelu: rt.Kernel, conv: rt.Kernel, rope: rt.Kernel, attn: rt.Kernel, swiglu: rt.Kernel };

/// Device scratch for a picture of up to `patches` patches.
const Scratch = struct { x0: Buf, h: Buf, xn: Buf, d: Buf, proj: Buf, q: Buf, k: Buf, v: Buf, att: Buf, a: Buf, b: Buf, act: Buf, p1: Buf, g: Buf };

/// Called after the patchifier's norm (-1), each layer (index) and the projector (depth) with the rows as fp32.
pub const Hook = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, stage: i32, rows: []const f32, cols: u32) void,
    /// The first conv layer's and the first attention layer's stages, and the projector's, by #1502's trace names.
    named: ?*const fn (ctx: *anyopaque, name: []const u8, rows: []const f32, cols: u32) void = null,
};

/// Rounding points (Tower.round): a set bit rounds them to bf16 values where transformers' bf16 tower does.
pub const Round = struct {
    pub const pixels: u32 = 1 << 0;
    pub const residual: u32 = 1 << 1;
    pub const norms: u32 = 1 << 2; // the embedding, operator and FFN norms' outputs
    pub const linear: u32 = 1 << 3; // the patchifier, conv in and out, q/k/v, the attention's out and w2
    pub const conv: u32 = 1 << 4; // B * x, the 3 x 3 sum, the gate
    pub const qk: u32 = 1 << 5; // the q/k head norms and RoPE
    pub const context: u32 = 1 << 6; // the attention's output
    pub const ffn: u32 = 1 << 7; // w1, w3 and SwiGLU
    pub const proj_norm: u32 = 1 << 8;
    pub const proj_linear: u32 = 1 << 9; // linear_1
    pub const proj_gelu: u32 = 1 << 10;
    pub const all: u32 = (1 << 11) - 1; // C11's tower: every activation bf16
};

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) {
            try k.setBuffer(i, v);
        } else if (T == f32) {
            try k.setF32(i, v);
        } else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

/// A bf16 tower or projector tensor by vision_spec's name and shape (module `m`, in `layer` when it sits in one).
fn get(l: *ld.Loader, seen: *u32, layer: ?usize, m: spec.Module) !Buf {
    var b: [160]u8 = undefined;
    const t = try spec.tensor(&b, vs, layer, m);
    const info = try l.info(t.name);
    if (!std.mem.eql(u8, info.dtype, "BF16") or info.len != t.bytes()) return error.UnexpectedLfm3VisionTensor;
    seen.* += 1;
    return l.load(t.name);
}

pub const Tower = struct {
    gpa: std.mem.Allocator,
    r: *rt.Runtime,
    patchifier: Buf,
    norm: Buf,
    layers: [depth]Layer,
    pn_w: Buf,
    pn_b: Buf,
    l1: Buf,
    b1: Buf,
    l2: Buf,
    b2: Buf,
    inv_freq: Buf,
    k: Kernels,
    s: Scratch,
    max_patches: u32,
    host: []f32,
    bytes: u64 = 0,
    /// Activations rounded to bf16 values (Round bits); 0: fp32 throughout, only the image rows bf16 for the decoder.
    round: u32 = 0,
    /// When set (tests): the tower's and the projector's ms of the last picture, a sync after each.
    split: ?*[2]f64 = null,
    t0: f64 = 0,

    /// The tower's 138 tensors from the checkpoint, and scratch for pictures of up to `max_patches` patches.
    pub fn load(gpa: std.mem.Allocator, r: *rt.Runtime, l: *ld.Loader, max_patches: u32) !*Tower {
        if (max_patches == 0 or max_patches % 4 != 0) return error.Invalid;
        const before = l.total;
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        var seen: u32 = 0;
        var m = try r.module(xpu.kernels.lfm3_vit);
        t.* = .{
            .gpa = gpa,
            .r = r,
            .patchifier = try get(l, &seen, null, .patchifier),
            .norm = try get(l, &seen, null, .embedding_norm),
            .layers = undefined,
            .pn_w = try get(l, &seen, null, .projector_norm),
            .pn_b = try get(l, &seen, null, .projector_norm_bias),
            .l1 = try get(l, &seen, null, .linear_1),
            .b1 = try get(l, &seen, null, .linear_1_bias),
            .l2 = try get(l, &seen, null, .linear_2),
            .b2 = try get(l, &seen, null, .linear_2_bias),
            .inv_freq = try l.empty(16 * 4),
            .k = .{
                .gemm = try m.kernel("vit_gemm", .{ 16, 16, 1 }),
                .rms = try m.kernel("vit_rms", .{ 64, 1, 1 }),
                .add_rms = try m.kernel("vit_add_rms", .{ 64, 1, 1 }),
                .ln = try m.kernel("vit_layernorm", .{ 64, 1, 1 }),
                .gelu = try m.kernel("vit_gelu", .{ 64, 1, 1 }),
                .conv = try m.kernel("vit_dwconv", .{ 64, 1, 1 }),
                .rope = try m.kernel("vit_qk_norm_rope", .{ 64, 1, 1 }),
                .attn = try m.kernel("vit_attn_qr", .{ 128, 1, 1 }), // vit_attn's bytes, q in registers: 1.7x
                .swiglu = try m.kernel("vit_swiglu", .{ 64, 1, 1 }),
            },
            .s = undefined,
            .max_patches = max_patches,
            .host = try gpa.alloc(f32, @as(usize, max_patches) * patch_width),
        };
        for (&t.layers, 0..) |*ly, i| {
            ly.* = .{
                .operator_norm = try get(l, &seen, i, .operator_norm),
                .ffn_norm = try get(l, &seen, i, .ffn_norm),
                .w1 = try get(l, &seen, i, .w1),
                .w3 = try get(l, &seen, i, .w3),
                .w2 = try get(l, &seen, i, .w2),
                .mixer = if (vs.mixers[i] == .attention) .{ .attention = .{
                    .q = try get(l, &seen, i, .q_proj),
                    .k = try get(l, &seen, i, .k_proj),
                    .v = try get(l, &seen, i, .v_proj),
                    .out = try get(l, &seen, i, .out_proj),
                    .q_norm = try get(l, &seen, i, .q_norm),
                    .k_norm = try get(l, &seen, i, .k_norm),
                } } else .{ .conv = .{
                    .in = try get(l, &seen, i, .conv_in),
                    .out = try get(l, &seen, i, .conv_out),
                    .spatial = try get(l, &seen, i, .spatial_conv),
                } },
            };
        }
        if (seen != spec.count(vs)) return error.UnexpectedLfm3VisionTensor;
        // axial RoPE's 16 inverse frequencies as transformers builds them in fp32: theta^(-2i/32)
        var inv: [16]f32 = undefined;
        for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, 1e6, @as(f32, @floatFromInt(2 * i)) / 32.0);
        try r.upload(t.inv_freq, std.mem.sliceAsBytes(&inv));
        const P: usize = max_patches;
        t.s = .{
            .x0 = try l.empty(P * patch_width * 4),
            .h = try l.empty(P * width * 4),
            .xn = try l.empty(P * width * 4),
            .d = try l.empty(P * width * 4),
            .proj = try l.empty(P * 3 * width * 4),
            .q = try l.empty(P * width * 4),
            .k = try l.empty(P * kv_heads * head_dim * 4),
            .v = try l.empty(P * kv_heads * head_dim * 4),
            .att = try l.empty(P * width * 4),
            .a = try l.empty(P * intermediate * 4),
            .b = try l.empty(P * intermediate * 4),
            .act = try l.empty(P * intermediate * 4),
            .p1 = try l.empty(P / 4 * proj_hidden * 4),
            .g = try l.empty(P / 4 * proj_hidden * 4),
        };
        try r.sync();
        t.bytes = l.total - before;
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.gpa.free(t.host);
        t.gpa.destroy(t);
    }

    /// y[m][n] = x[m] . w[n] + bias, fp32 (x rows ldx apart, y rows ldy apart); `bits` rounds y, out_bf stores it bf16.
    fn gemm(t: *Tower, x: Buf, ldx: u32, w: Buf, bias: ?Buf, y: Buf, ldy: u32, m: u32, n: u32, kdim: u32, bits: u32, out_bf: bool) !void {
        try run(&t.k.gemm, .{ n / 64, (m + 63) / 64, 1 }, .{ x, ldx, w, bias orelse w, @as(u32, @intFromBool(bias != null)), y, ldy, m, kdim, t.rd(bits), @as(u32, @intFromBool(out_bf)) });
    }

    /// 1 when the `bits` rounding points are on.
    fn rd(t: *const Tower, bits: u32) u32 {
        return @intFromBool(t.round & bits != 0);
    }

    fn trace(t: *Tower, h: ?Hook, name: []const u8, b: Buf, rows: u32, cols: u32) !void {
        const k = h orelse return;
        const f = k.named orelse return;
        const host = try t.gpa.alloc(f32, @as(usize, rows) * cols);
        defer t.gpa.free(host);
        try t.r.sync();
        try t.r.download(std.mem.sliceAsBytes(host), b);
        try t.r.sync();
        f(k.ctx, name, host, cols);
    }

    /// The rows of `b` (fp32, or bf16 when `bf`) to the hook as fp32.
    fn hook(t: *Tower, h: ?Hook, stage: i32, b: Buf, rows: u32, cols: u32, bf: bool) !void {
        const k = h orelse return;
        const n = @as(usize, rows) * cols;
        const host = try t.gpa.alloc(f32, n);
        defer t.gpa.free(host);
        try t.r.sync();
        if (bf) {
            const raw = try t.gpa.alloc(u16, n);
            defer t.gpa.free(raw);
            try t.r.download(std.mem.sliceAsBytes(raw), b);
            try t.r.sync();
            for (host, raw) |*o, v| o.* = @bitCast(@as(u32, v) << 16);
        } else {
            try t.r.download(std.mem.sliceAsBytes(host), b);
            try t.r.sync();
        }
        k.f(k.ctx, stage, host, cols);
    }

    /// One picture's merge-order patches (gh x gw, fp32 [patches][1536]) to its gh * gw / 4 image rows, bf16 [n][2048].
    pub fn image(t: *Tower, pixels: []const f32, gh: u32, gw: u32, out: Buf, h: ?Hook) !void {
        const P = gh * gw;
        if (gh == 0 or gw == 0 or gh % 2 != 0 or gw % 2 != 0 or P > t.max_patches or pixels.len != @as(usize, P) * patch_width) return error.BadLfm3ImageShape;
        const s = &t.s;
        if (t.split != null) t.t0 = try t.stamp();
        for (t.host[0..pixels.len], pixels) |*o, v| o.* = if (t.round & Round.pixels != 0) @bitCast(@as(u32, toBf(v)) << 16) else v;
        try t.r.upload(s.x0, std.mem.sliceAsBytes(t.host[0..pixels.len]));
        try t.gemm(s.x0, patch_width, t.patchifier, null, s.d, width, P, width, patch_width, Round.linear, false);
        try t.trace(h, "patchifier", s.d, P, width);
        try run(&t.k.rms, .{ P, 1, 1 }, .{ s.d, t.norm, s.h, width, eps, t.rd(Round.norms) });
        try t.hook(h, -1, s.h, P, width, false);
        try run(&t.k.rms, .{ P, 1, 1 }, .{ s.h, t.layers[0].operator_norm, s.xn, width, eps, t.rd(Round.norms) });
        for (&t.layers, 0..) |*ly, i| {
            const first = i == 0;
            if (first) try t.trace(h, "first_operator_norm", s.xn, P, width);
            if (i == 2) try t.trace(h, "first_attention_norm", s.xn, P, width);
            switch (ly.mixer) {
                .conv => |c| {
                    try t.gemm(s.xn, width, c.in, null, s.proj, 3 * width, P, 3 * width, width, Round.linear, false);
                    if (first) try t.trace(h, "first_conv_in", s.proj, P, 3 * width);
                    try run(&t.k.conv, .{ width / 64, P, 1 }, .{ s.proj, c.spatial, s.att, gh, gw, t.rd(Round.conv) });
                    if (first) try t.trace(h, "first_conv_gated", s.att, P, width);
                    try t.gemm(s.att, width, c.out, null, s.d, width, P, width, width, Round.linear, false);
                },
                .attention => |a| {
                    const kvw = kv_heads * head_dim;
                    try t.gemm(s.xn, width, a.q, null, s.q, width, P, width, width, Round.linear, false);
                    try t.gemm(s.xn, width, a.k, null, s.k, kvw, P, kvw, width, Round.linear, false);
                    try t.gemm(s.xn, width, a.v, null, s.v, kvw, P, kvw, width, Round.linear, false);
                    try run(&t.k.rope, .{ P * heads / 4, 1, 1 }, .{ s.q, width, s.q, a.q_norm, t.inv_freq, heads, gw, eps, t.rd(Round.qk) });
                    try run(&t.k.rope, .{ P * kv_heads / 4, 1, 1 }, .{ s.k, kvw, s.k, a.k_norm, t.inv_freq, kv_heads, gw, eps, t.rd(Round.qk) });
                    if (i == 2) {
                        try t.trace(h, "first_attention_q", s.q, P, width);
                        try t.trace(h, "first_attention_k", s.k, P, kvw);
                        try t.trace(h, "first_attention_v", s.v, P, kvw);
                    }
                    try run(&t.k.attn, .{ (P + 31) / 32, heads, 1 }, .{ s.q, s.k, s.v, s.att, P, heads, kv_heads, @as(f32, 0.125), t.rd(Round.context) });
                    if (i == 2) try t.trace(h, "first_attention_context", s.att, P, width);
                    try t.gemm(s.att, width, a.out, null, s.d, width, P, width, width, Round.linear, false);
                    if (i == 2) try t.trace(h, "first_attention_operator", s.d, P, width);
                },
            }
            if (first) try t.trace(h, "first_operator", s.d, P, width);
            try run(&t.k.add_rms, .{ P, 1, 1 }, .{ s.h, s.d, ly.ffn_norm, s.xn, width, eps, t.rd(Round.residual), t.rd(Round.norms) });
            if (first) try t.trace(h, "first_residual", s.h, P, width);
            if (first) try t.trace(h, "first_ffn_norm", s.xn, P, width);
            try t.gemm(s.xn, width, ly.w1, null, s.a, intermediate, P, intermediate, width, Round.ffn, false);
            try t.gemm(s.xn, width, ly.w3, null, s.b, intermediate, P, intermediate, width, Round.ffn, false);
            try run(&t.k.swiglu, .{ (P * intermediate + 63) / 64, 1, 1 }, .{ s.a, s.b, s.act, P * intermediate, t.rd(Round.ffn) });
            if (first) {
                try t.trace(h, "first_ffn_gate", s.a, P, intermediate);
                try t.trace(h, "first_ffn_up", s.b, P, intermediate);
                try t.trace(h, "first_ffn_act", s.act, P, intermediate);
            }
            try t.gemm(s.act, intermediate, ly.w2, null, s.d, width, P, width, intermediate, Round.linear, false);
            if (first) try t.trace(h, "first_ffn_down", s.d, P, width);
            // the residual add, and the next layer's operator norm (after the last: discarded)
            const next = if (i + 1 < depth) t.layers[i + 1].operator_norm else t.layers[0].operator_norm;
            try run(&t.k.add_rms, .{ P, 1, 1 }, .{ s.h, s.d, next, s.xn, width, eps, t.rd(Round.residual), t.rd(Round.norms) });
            try t.hook(h, @intCast(i), s.h, P, width, false);
        }
        const t_layers = if (t.split != null) try t.stamp() else 0;
        // projector: LayerNorm a patch, the 2 x 2 block's four rows as one 4096-wide row, linear_1, GELU, linear_2
        const R = P / 4;
        try run(&t.k.ln, .{ P, 1, 1 }, .{ s.h, t.pn_w, t.pn_b, s.xn, width, ln_eps, t.rd(Round.proj_norm) });
        try t.trace(h, "projector_norm", s.xn, P, width);
        try t.gemm(s.xn, proj_hidden, t.l1, t.b1, s.p1, proj_hidden, R, proj_hidden, proj_hidden, Round.proj_linear, false);
        try run(&t.k.gelu, .{ (R * proj_hidden + 63) / 64, 1, 1 }, .{ s.p1, s.g, R * proj_hidden, t.rd(Round.proj_gelu) });
        try t.trace(h, "projector_gelu", s.g, R, proj_hidden);
        // the image rows in bf16, as the decoder's embeddings are (#1502's rows buffer: hidden * 2 bytes a row)
        try t.gemm(s.g, proj_hidden, t.l2, t.b2, out, out_width, R, out_width, proj_hidden, 0, true);
        try t.hook(h, depth, out, R, out_width, true);
        if (t.split) |sp| {
            sp[1] = try t.stamp() - t_layers;
            sp[0] = t_layers - t.t0;
        }
    }

    /// ms on the monotonic clock once the queue is drained.
    fn stamp(t: *Tower) !f64 {
        try t.r.sync();
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
    }
};

/// Nearest bf16 of an fp32 value (ties to even), as torch's .to(bfloat16).
fn toBf(f: f32) u16 {
    var u: u32 = @bitCast(f);
    u +%= 0x7fff + ((u >> 16) & 1);
    return @truncate(u >> 16);
}
