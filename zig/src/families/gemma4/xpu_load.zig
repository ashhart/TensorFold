//! Gemma 4 E4B weights onto the device: MLX affine 4/64 projections in the GEMM layout, norms, the two raw tables.
const std = @import("std");
const ze = @import("ze");
const dev = @import("xpu_dev.zig");
const cfg = @import("xpu_config.zig");
const Dev = dev.Dev;
const Checkpoint = dev.Checkpoint;
const alloc = dev.alloc;

/// An int4 projection in the GEMM layout (words [K/8, N], fp16 scales and biases [K/64, N]); `wide`: no K slices.
pub const Proj = struct { w: *anyopaque, s: *anyopaque, bg: *anyopaque, n: u32, k: u32, wide: bool, dense: bool = false, bits8: bool = false };

pub const Layer = struct {
    in_norm: *anyopaque,
    post_attn_norm: *anyopaque,
    pre_ffn_norm: *anyopaque,
    post_ffn_norm: *anyopaque,
    post_ple_norm: *anyopaque,
    scalar: *anyopaque, // bf16 [1]
    qkv: Proj, // q | k | v of the layers with their own K/V, else q alone
    o: Proj,
    gu: Proj, // gate | up
    down: Proj,
    ple_gate: Proj,
    ple_proj: Proj,
    qn: *anyopaque,
    kn: ?*anyopaque,
};

pub const Model = struct {
    embed_w: *anyopaque, // the tied table as stored (the gather reads it), then the repacked head
    embed_s: *anyopaque,
    embed_b: *anyopaque,
    ple_w: *anyopaque, // per-layer table as stored [V, 10752 / 8 words]
    ple_s: *anyopaque,
    ple_b: *anyopaque,
    ple_proj: Proj, // per_layer_model_projection
    ple_norm: *anyopaque, // per_layer_projection_norm
    norm: *anyopaque,
    head: Proj,
    layers: [cfg.layers]Layer,
    invf_s: *anyopaque, // fp32 [128]
    invf_g: *anyopaque, // fp32 [256], zeros after the 64 rotated pairs
};

fn bf16(d: *Dev, c: *const Checkpoint, name: []const u8) !*anyopaque {
    const t = try c.get(name);
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.UnexpectedDtype;
    return d.upload(t.bytes);
}

fn named(comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(alloc, fmt, args) catch @panic("out of memory");
}

/// 8-bit MLX parts stacked on N, kept as stored (uchar codes [N, K], bf16 scales and biases [N, K / 64]) for g4_w8.
fn projection8(d: *Dev, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, n_total: u32, k: u32) !Proj {
    if (n_total % 16 != 0 or k % 256 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k) * n_total), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = true, .bits8 = true };
    var at: usize = 0;
    var sat: usize = 0;
    for (parts) |p| {
        const w = try c.get(named("{s}{s}.weight", .{ prefix, p }));
        const s = try c.get(named("{s}{s}.scales", .{ prefix, p }));
        const b = try c.get(named("{s}{s}.biases", .{ prefix, p }));
        try d.copyIn(dev.off(out.w, at), w.bytes);
        try d.copyIn(dev.off(out.s, sat), s.bytes);
        try d.copyIn(dev.off(out.bg, sat), b.bytes);
        at += w.bytes.len;
        sat += s.bytes.len;
    }
    return out;
}

/// A fused projection from MLX parts stacked on N, repacked on the device through the staging area `stage`.
pub fn projection(d: *Dev, repack: ze.H, c: *const Checkpoint, prefix: []const u8, parts: []const []const u8, wide: bool, stage: *anyopaque) !Proj {
    var n_total: u32 = 0;
    var k: u32 = 0;
    if (parts.len == 1) { // a bf16 weight (not quantized): kept as stored, read by the dense kernel
        const w = try c.get(named("{s}{s}.weight", .{ prefix, parts[0] }));
        if (std.mem.eql(u8, w.dtype, "BF16")) {
            // bf16 to fp16 (round to nearest even; these weights are far inside fp16's range)
            const n = w.bytes.len / 2;
            const half = try alloc.alloc(f16, n);
            defer alloc.free(half);
            for (half, 0..) |*o, i| o.* = @floatCast(@as(f32, @bitCast(@as(u32, std.mem.readInt(u16, w.bytes[2 * i ..][0..2], .little)) << 16)));
            const p = try d.upload(std.mem.sliceAsBytes(half));
            return .{ .w = p, .s = p, .bg = p, .n = @intCast(w.shape[0]), .k = @intCast(w.shape[1]), .wide = true, .dense = true };
        }
    }
    var eight = false;
    for (parts) |p| {
        const w = try c.get(named("{s}{s}.weight", .{ prefix, p }));
        const sc = try c.get(named("{s}{s}.scales", .{ prefix, p }));
        n_total += @intCast(w.shape[0]);
        k = @intCast(sc.shape[1] * 64);
        eight = w.shape[1] * 32 == k * 8; // 8-bit codes: K bytes a row
    }
    if (eight) return projection8(d, c, prefix, parts, n_total, k);
    if (n_total % 256 != 0 or k % 64 != 0) return error.UnexpectedShape;
    const out: Proj = .{ .w = try d.alloc_(@as(usize, k / 8) * n_total * 4), .s = try d.alloc_(@as(usize, k / 64) * n_total * 2), .bg = try d.alloc_(@as(usize, k / 64) * n_total * 2), .n = n_total, .k = k, .wide = wide };
    var n0: u32 = 0;
    for (parts) |p| {
        const w = try c.get(named("{s}{s}.weight", .{ prefix, p }));
        const s = try c.get(named("{s}{s}.scales", .{ prefix, p }));
        const b = try c.get(named("{s}{s}.biases", .{ prefix, p }));
        const n: u32 = @intCast(w.shape[0]);
        const ss = dev.off(stage, w.bytes.len);
        const sb = dev.off(ss, s.bytes.len);
        try d.copyIn(stage, w.bytes);
        try d.copyIn(ss, s.bytes);
        try d.copyIn(sb, b.bytes);
        try d.launch(repack, .{ .x = (n + 15) / 16, .y = k / 8 }, .{ stage, ss, sb, out.w, out.s, out.bg, @as(i32, @intCast(n)), @as(i32, @intCast(k)), @as(i32, @intCast(n_total)), @as(i32, @intCast(n0)), @as(i32, 0) });
        try d.sync();
        n0 += n;
    }
    return out;
}

fn invFreq(comptime n: usize, theta: f64, dim: f64, rotated: usize) [n]f32 {
    var t: [n]f32 = @splat(0);
    for (0..@min(n, rotated)) |i| t[i] = @floatCast(1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * i)) / dim));
    return t;
}

pub fn load(d: *Dev, repack: ze.H, c: *const Checkpoint) !Model {
    var m: Model = undefined;
    const lm = "model.language_model.";
    const stage = try d.alloc_(512 << 20);
    const et = try c.get(lm ++ "embed_tokens.weight");
    m.embed_w = try d.upload(et.bytes);
    m.embed_s = try bf16(d, c, lm ++ "embed_tokens.scales");
    m.embed_b = try bf16(d, c, lm ++ "embed_tokens.biases");
    const pt = try c.get(lm ++ "embed_tokens_per_layer.weight");
    m.ple_w = try d.upload(pt.bytes);
    m.ple_s = try bf16(d, c, lm ++ "embed_tokens_per_layer.scales");
    m.ple_b = try bf16(d, c, lm ++ "embed_tokens_per_layer.biases");
    m.ple_proj = try projection(d, repack, c, lm, &.{"per_layer_model_projection"}, true, stage);
    m.ple_norm = try bf16(d, c, lm ++ "per_layer_projection_norm.weight");
    m.norm = try bf16(d, c, lm ++ "norm.weight");
    m.head = try projection(d, repack, c, lm ++ "embed_tokens", &.{""}, true, stage);
    m.invf_s = try d.upload(std.mem.sliceAsBytes(&invFreq(128, 10000.0, 256, 128)));
    m.invf_g = try d.upload(std.mem.sliceAsBytes(&invFreq(256, 1000000.0, 512, 64)));
    for (&m.layers, 0..) |*l, i| {
        const p = named(lm ++ "layers.{d}.", .{i});
        const sa = named("{s}self_attn.", .{p});
        const own = i < cfg.shared_from;
        const ml = named("{s}mlp.", .{p});
        l.* = .{
            .in_norm = try bf16(d, c, named("{s}input_layernorm.weight", .{p})),
            .post_attn_norm = try bf16(d, c, named("{s}post_attention_layernorm.weight", .{p})),
            .pre_ffn_norm = try bf16(d, c, named("{s}pre_feedforward_layernorm.weight", .{p})),
            .post_ffn_norm = try bf16(d, c, named("{s}post_feedforward_layernorm.weight", .{p})),
            .post_ple_norm = try bf16(d, c, named("{s}post_per_layer_input_norm.weight", .{p})),
            .scalar = try bf16(d, c, named("{s}layer_scalar", .{p})),
            .qkv = try projection(d, repack, c, sa, if (own) &.{ "q_proj", "k_proj", "v_proj" } else &.{"q_proj"}, true, stage),
            .o = try projection(d, repack, c, sa, &.{"o_proj"}, false, stage),
            .gu = try projection(d, repack, c, ml, &.{ "gate_proj", "up_proj" }, true, stage),
            .down = try projection(d, repack, c, ml, &.{"down_proj"}, false, stage),
            .ple_gate = try projection(d, repack, c, p, &.{"per_layer_input_gate"}, false, stage),
            .ple_proj = try projection(d, repack, c, p, &.{"per_layer_projection"}, false, stage),
            .qn = try bf16(d, c, named("{s}q_norm.weight", .{sa})),
            .kn = if (own) try bf16(d, c, named("{s}k_norm.weight", .{sa})) else null,
        };
    }
    try d.sync();
    d.g.free(stage);
    d.allocated -= 512 << 20;
    return m;
}
