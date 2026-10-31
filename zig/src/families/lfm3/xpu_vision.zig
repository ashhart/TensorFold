//! LFM3-VL's vision tower and projector on the Intel GPU: merge-order patches to the text model's image rows.

const std = @import("std");
const xpu = @import("xpu");
const rt = xpu.rt;
const ld = xpu.loader;

const Buf = rt.Buffer;
pub const width: u32 = 1024;
pub const intermediate: u32 = 2560;
pub const depth = 14;
pub const patch_width: u32 = 1536;
pub const out_width: u32 = 2048;
const heads: u32 = 16;
const kv_heads: u32 = 8;
const head_dim: u32 = 64;
const proj_hidden: u32 = 4096;
const eps: f32 = 1e-5;
const ln_eps: f32 = 1e-6;

const Conv = struct { in: Buf, out: Buf, spatial: Buf };
const Attention = struct { q: Buf, k: Buf, v: Buf, out: Buf, q_norm: Buf, k_norm: Buf };
const Layer = struct { operator_norm: Buf, ffn_norm: Buf, w1: Buf, w3: Buf, w2: Buf, mixer: union(enum) { conv: Conv, attention: Attention } };

const Kernels = struct { gemm: rt.Kernel, rms: rt.Kernel, add_rms: rt.Kernel, ln: rt.Kernel, gelu: rt.Kernel, conv: rt.Kernel, rope: rt.Kernel, attn: rt.Kernel, swiglu: rt.Kernel };

/// Device scratch for a picture of up to `patches` patches.
const Scratch = struct { x0: Buf, h: Buf, xn: Buf, d: Buf, proj: Buf, q: Buf, k: Buf, v: Buf, att: Buf, a: Buf, b: Buf, act: Buf, p1: Buf, g: Buf };

/// Called after the patchifier's norm (-1), each layer (index) and the projector (depth) with bf16 rows.
pub const Hook = struct { ctx: *anyopaque, f: *const fn (ctx: *anyopaque, stage: i32, rows: []const u16, cols: u32) void };

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

/// A bf16 tensor by name, checked against its shape (#1502's vision_weights.zig names and shapes).
fn get(l: *ld.Loader, seen: *u32, prefix: []const u8, suffix: []const u8, shape: []const usize) !Buf {
    var b: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&b, "{s}{s}", .{ prefix, suffix });
    const info = try l.info(name);
    var bytes: u64 = 2;
    for (shape) |d| bytes *= d;
    if (!std.mem.eql(u8, info.dtype, "BF16") or info.len != bytes) return error.UnexpectedLfm3VisionTensor;
    seen.* += 1;
    return l.load(name);
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
    host: []u16,
    bytes: u64 = 0,
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
            .patchifier = try get(l, &seen, "vision_tower.", "patchifier.weight", &.{ width, patch_width }),
            .norm = try get(l, &seen, "vision_tower.", "embedding_norm.weight", &.{width}),
            .layers = undefined,
            .pn_w = try get(l, &seen, "multi_modal_projector.", "layer_norm.weight", &.{width}),
            .pn_b = try get(l, &seen, "multi_modal_projector.", "layer_norm.bias", &.{width}),
            .l1 = try get(l, &seen, "multi_modal_projector.", "linear_1.weight", &.{ proj_hidden, proj_hidden }),
            .b1 = try get(l, &seen, "multi_modal_projector.", "linear_1.bias", &.{proj_hidden}),
            .l2 = try get(l, &seen, "multi_modal_projector.", "linear_2.weight", &.{ out_width, proj_hidden }),
            .b2 = try get(l, &seen, "multi_modal_projector.", "linear_2.bias", &.{out_width}),
            .inv_freq = try l.empty(16 * 4),
            .k = .{
                .gemm = try m.kernel("vit_gemm", .{ 16, 16, 1 }),
                .rms = try m.kernel("vit_rms", .{ 64, 1, 1 }),
                .add_rms = try m.kernel("vit_add_rms", .{ 64, 1, 1 }),
                .ln = try m.kernel("vit_layernorm", .{ 64, 1, 1 }),
                .gelu = try m.kernel("vit_gelu", .{ 64, 1, 1 }),
                .conv = try m.kernel("vit_dwconv", .{ 64, 1, 1 }),
                .rope = try m.kernel("vit_qk_norm_rope", .{ 64, 1, 1 }),
                .attn = try m.kernel("vit_attn", .{ 64, 1, 1 }),
                .swiglu = try m.kernel("vit_swiglu", .{ 64, 1, 1 }),
            },
            .s = undefined,
            .max_patches = max_patches,
            .host = try gpa.alloc(u16, @as(usize, max_patches) * patch_width),
        };
        for (&t.layers, 0..) |*ly, i| {
            var b: [64]u8 = undefined;
            const p = try std.fmt.bufPrint(&b, "vision_tower.layers.{d}.", .{i});
            ly.* = .{
                .operator_norm = try get(l, &seen, p, "operator_norm.weight", &.{width}),
                .ffn_norm = try get(l, &seen, p, "ffn_norm.weight", &.{width}),
                .w1 = try get(l, &seen, p, "feed_forward.w1.weight", &.{ intermediate, width }),
                .w3 = try get(l, &seen, p, "feed_forward.w3.weight", &.{ intermediate, width }),
                .w2 = try get(l, &seen, p, "feed_forward.w2.weight", &.{ width, intermediate }),
                .mixer = if (i >= 2 and i % 2 == 0) .{ .attention = .{
                    .q = try get(l, &seen, p, "self_attn.q_proj.weight", &.{ width, width }),
                    .k = try get(l, &seen, p, "self_attn.k_proj.weight", &.{ kv_heads * head_dim, width }),
                    .v = try get(l, &seen, p, "self_attn.v_proj.weight", &.{ kv_heads * head_dim, width }),
                    .out = try get(l, &seen, p, "self_attn.out_proj.weight", &.{ width, width }),
                    .q_norm = try get(l, &seen, p, "self_attn.q_layernorm.weight", &.{head_dim}),
                    .k_norm = try get(l, &seen, p, "self_attn.k_layernorm.weight", &.{head_dim}),
                } } else .{ .conv = .{
                    .in = try get(l, &seen, p, "conv.in_proj.weight", &.{ 3 * width, width }),
                    .out = try get(l, &seen, p, "conv.out_proj.weight", &.{ width, width }),
                    .spatial = try get(l, &seen, p, "conv.spatial_conv.weight", &.{ width, 3, 3, 1 }),
                } },
            };
        }
        if (seen != 138) return error.UnexpectedLfm3VisionTensor;
        // axial RoPE's 16 inverse frequencies as transformers builds them in fp32: theta^(-2i/32)
        var inv: [16]f32 = undefined;
        for (&inv, 0..) |*f, i| f.* = 1.0 / std.math.pow(f32, 1e6, @as(f32, @floatFromInt(2 * i)) / 32.0);
        try r.upload(t.inv_freq, std.mem.sliceAsBytes(&inv));
        const P: usize = max_patches;
        t.s = .{
            .x0 = try l.empty(P * patch_width * 2),
            .h = try l.empty(P * width * 2),
            .xn = try l.empty(P * width * 2),
            .d = try l.empty(P * width * 2),
            .proj = try l.empty(P * 3 * width * 2),
            .q = try l.empty(P * width * 2),
            .k = try l.empty(P * kv_heads * head_dim * 2),
            .v = try l.empty(P * kv_heads * head_dim * 2),
            .att = try l.empty(P * width * 2),
            .a = try l.empty(P * intermediate * 2),
            .b = try l.empty(P * intermediate * 2),
            .act = try l.empty(P * intermediate * 2),
            .p1 = try l.empty(P / 4 * proj_hidden * 2),
            .g = try l.empty(P / 4 * proj_hidden * 2),
        };
        try r.sync();
        t.bytes = l.total - before;
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.gpa.free(t.host);
        t.gpa.destroy(t);
    }

    /// y[m][n] = bf16(x[m] . w[n] + bias) (x rows ldx apart, y rows ldy apart).
    fn gemm(t: *Tower, x: Buf, ldx: u32, w: Buf, bias: ?Buf, y: Buf, ldy: u32, m: u32, n: u32, kdim: u32) !void {
        try run(&t.k.gemm, .{ n / 64, (m + 63) / 64, 1 }, .{ x, ldx, w, bias orelse w, @as(u32, @intFromBool(bias != null)), y, ldy, m, kdim });
    }

    fn hook(t: *Tower, h: ?Hook, stage: i32, b: Buf, rows: u32, cols: u32) !void {
        const k = h orelse return;
        const host = try t.gpa.alloc(u16, @as(usize, rows) * cols);
        defer t.gpa.free(host);
        try t.r.sync();
        try t.r.download(std.mem.sliceAsBytes(host), b);
        try t.r.sync();
        k.f(k.ctx, stage, host, cols);
    }

    /// One picture's merge-order patches (gh x gw, fp32 [patches][1536]) to its gh * gw / 4 image rows, bf16 [n][2048].
    pub fn image(t: *Tower, pixels: []const f32, gh: u32, gw: u32, out: Buf, h: ?Hook) !void {
        const P = gh * gw;
        if (gh == 0 or gw == 0 or gh % 2 != 0 or gw % 2 != 0 or P > t.max_patches or pixels.len != @as(usize, P) * patch_width) return error.BadLfm3ImageShape;
        const s = &t.s;
        if (t.split != null) t.t0 = try t.stamp();
        for (t.host[0..pixels.len], pixels) |*o, v| o.* = toBf(v);
        try t.r.upload(s.x0, std.mem.sliceAsBytes(t.host[0..pixels.len]));
        try t.gemm(s.x0, patch_width, t.patchifier, null, s.d, width, P, width, patch_width);
        try run(&t.k.rms, .{ P, 1, 1 }, .{ s.d, t.norm, s.h, width, eps });
        try t.hook(h, -1, s.h, P, width);
        try run(&t.k.rms, .{ P, 1, 1 }, .{ s.h, t.layers[0].operator_norm, s.xn, width, eps });
        for (&t.layers, 0..) |*ly, i| {
            switch (ly.mixer) {
                .conv => |c| {
                    try t.gemm(s.xn, width, c.in, null, s.proj, 3 * width, P, 3 * width, width);
                    try run(&t.k.conv, .{ width / 64, P, 1 }, .{ s.proj, c.spatial, s.att, gh, gw });
                    try t.gemm(s.att, width, c.out, null, s.d, width, P, width, width);
                },
                .attention => |a| {
                    const kvw = kv_heads * head_dim;
                    try t.gemm(s.xn, width, a.q, null, s.q, width, P, width, width);
                    try t.gemm(s.xn, width, a.k, null, s.k, kvw, P, kvw, width);
                    try t.gemm(s.xn, width, a.v, null, s.v, kvw, P, kvw, width);
                    try run(&t.k.rope, .{ P * heads / 4, 1, 1 }, .{ s.q, width, s.q, a.q_norm, t.inv_freq, heads, gw, eps });
                    try run(&t.k.rope, .{ P * kv_heads / 4, 1, 1 }, .{ s.k, kvw, s.k, a.k_norm, t.inv_freq, kv_heads, gw, eps });
                    try run(&t.k.attn, .{ (P + 31) / 32, heads, 1 }, .{ s.q, s.k, s.v, s.att, P, heads, kv_heads, @as(f32, 0.125) });
                    try t.gemm(s.att, width, a.out, null, s.d, width, P, width, width);
                },
            }
            try run(&t.k.add_rms, .{ P, 1, 1 }, .{ s.h, s.d, ly.ffn_norm, s.xn, width, eps });
            try t.gemm(s.xn, width, ly.w1, null, s.a, intermediate, P, intermediate, width);
            try t.gemm(s.xn, width, ly.w3, null, s.b, intermediate, P, intermediate, width);
            try run(&t.k.swiglu, .{ (P * intermediate + 63) / 64, 1, 1 }, .{ s.a, s.b, s.act, P * intermediate });
            try t.gemm(s.act, intermediate, ly.w2, null, s.d, width, P, width, intermediate);
            // the residual add, and the next layer's operator norm (after the last: discarded)
            const next = if (i + 1 < depth) t.layers[i + 1].operator_norm else t.layers[0].operator_norm;
            try run(&t.k.add_rms, .{ P, 1, 1 }, .{ s.h, s.d, next, s.xn, width, eps });
            try t.hook(h, @intCast(i), s.h, P, width);
        }
        const t_layers = if (t.split != null) try t.stamp() else 0;
        // projector: LayerNorm a patch, the 2 x 2 block's four rows as one 4096-wide row, linear_1, GELU, linear_2
        const R = P / 4;
        try run(&t.k.ln, .{ P, 1, 1 }, .{ s.h, t.pn_w, t.pn_b, s.xn, width, ln_eps });
        try t.gemm(s.xn, proj_hidden, t.l1, t.b1, s.p1, proj_hidden, R, proj_hidden, proj_hidden);
        try run(&t.k.gelu, .{ (R * proj_hidden + 63) / 64, 1, 1 }, .{ s.p1, s.g, R * proj_hidden });
        try t.gemm(s.g, proj_hidden, t.l2, t.b2, out, out_width, R, out_width, proj_hidden);
        try t.hook(h, depth, out, R, out_width);
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
