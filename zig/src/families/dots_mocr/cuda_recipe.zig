//! dots.mocr as a cuda_dense recipe: its Qwen2 text decoder in HIP's rounding points, and its image tower.
const std = @import("std");
const cuda = @import("cuda");
const dense = @import("cuda_dense");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const kern = @import("cuda_kernels.zig");

const Recipe = @This();
const P = *const dense.Pass(Recipe);

pub const name = "dots.mocr";
pub const model_type = "dots_ocr";
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context: i64 = 32768;
/// Replies carry no reasoning block: the text is content.
pub const reasoning = false;

pub const Weights = wts.Weights;
pub const Kernels = kern.Kernels;
/// The image tower --vision opens (cuda_tower.zig).
pub const Tower = @import("cuda_tower.zig").Tower;

/// The engine's geometry of an admitted dots.mocr config.
pub fn geometryOf(c: cfg.Config) dense.Geometry {
    const h: usize = c.hidden;
    return .{ .hidden = h, .layers = c.layers, .heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.headDim(), .vocab = c.vocab, .eps = c.eps, .context = c.context, .acc = h + 2 * @as(usize, c.kvWidth()), .wide = 2 * @as(usize, c.intermediate), .mid = c.intermediate, .theta = c.theta, .inv_freq = if (c.theta == 1e6 and c.headDim() == 128) &hf_inv_freq else null };
}

/// config.json and generation_config.json admitted as dots.mocr's, before any tensor is read.
pub fn geometry(a: std.mem.Allocator, io: std.Io, dir: []const u8) !dense.Geometry {
    return geometryOf(try cfg.Config.read(a, io, dir));
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8) !Weights {
    return wts.load(gpa, io, d, s, dir);
}

/// The text decoder's bytes on the device, once every shard's header is checked.
pub fn weightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const c = try cfg.Config.read(a, io, dir);
    _ = try @import("weights.zig").inspect(a, io, dir, c);
    return wts.textBytes(a, c);
}

pub fn embed(p: P) !void {
    try p.k.embed.bf16(p.s, p.m.s.ids, p.w.embed, p.m.s.h, p.rows, p.g.hidden);
}

/// One Qwen2 layer: RMS, q|k|v sums, bias and RoPE in Torch's order, attention, o, RMS, SwiGLU.
pub fn layer(p: P, i: usize) !void {
    const x = p.m.s;
    const l = p.w.layers[i];
    try p.norm(if (i == 0) null else x.r, l.ln1);
    try p.projectF32(l.qkv, x.x, x.acc);
    try tapped(p, "qkv", i, x.x, p.g.hidden, x.acc, 2 * p.g.acc); // its fp32 sums: two bf16 columns a sum
    try p.own.qkvRope(p.s, x.acc, l.bias, p.m.inv, x.pos, x.q, x.kv, p.rows, p.g.heads, p.g.kv_heads);
    try p.attend(i, x.q, x.kv, x.y);
    try p.project(l.o, x.y, x.r);
    try tapped(p, "o", i, x.y, p.g.qWidth(), x.r, p.g.hidden);
    try p.norm(x.r, l.ln2);
    try p.project(l.gate_up, x.x, x.wide);
    try tapped(p, "gate_up", i, x.x, p.g.hidden, x.wide, p.g.wide);
    try p.siluMul(x.wide, x.y, p.g.mid);
    try p.project(l.down, x.y, x.r);
    try tapped(p, "down", i, x.y, p.g.mid, x.r, p.g.hidden);
}

/// A projection's input [rows, k] and output [rows, n] as `<class>.<layer>.x` and `.y`, when a check taps the pass.
fn tapped(p: P, class: []const u8, i: usize, in: u64, k: usize, y: u64, n: usize) !void {
    if (p.taps == null) return;
    var stage: [32]u8 = undefined;
    try p.tap(try std.fmt.bufPrint(&stage, "{s}.{d}.x", .{ class, i }), in, k);
    try p.tap(try std.fmt.bufPrint(&stage, "{s}.{d}.y", .{ class, i }), y, n);
}

pub fn final(p: P) !void {
    try p.norm(p.m.s.r, p.w.norm);
}

pub fn head(w: *const Weights) cuda.qmmf.Weight {
    return w.head;
}

pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.UnsupportedDotsConfig, error.BadDotsConfig, error.BadEos => "this checkpoint's config is not dots.mocr's (DotsOCRForCausalLM, Qwen2 1.5B text decoder in BF16), the one this engine serves",
        error.DotsDType, error.DotsShape, error.DotsExtent, error.DotsMissingTensor, error.DotsTensorCount, error.DotsIndexMismatch => "this dots.mocr checkpoint's tensors are not the released BF16 ones: download it again",
        else => null,
    };
}

/// Transformers' own float32 inv_freq for theta 1e6 at head dim 128 (fixtures/README.md): Torch's CPU pow, bit for bit.
const hf_inv_freq: [64]f32 = blk: {
    @setEvalBranchQuota(20_000);
    const text = @embedFile("fixtures/rope_inv_freq.txt");
    var out: [64]f32 = undefined;
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    for (&out) |*v| v.* = @bitCast(std.fmt.parseInt(u32, it.next().?, 0) catch unreachable);
    if (it.next() != null) @compileError("fixtures/rope_inv_freq.txt holds more than 64 values");
    break :blk out;
};

test "dots' rotary frequencies are Transformers' own: the correctly rounded pow differs at entry 37 only" {
    var mine: [64]f32 = undefined;
    @import("cuda_dense").rope.inverse(1e6, 128, &mine);
    for (hf_inv_freq, mine, 0..) |hf, own, i| {
        const one_ulp = @as(u32, @bitCast(hf)) -% @as(u32, @bitCast(own)) <= 1 or @as(u32, @bitCast(own)) -% @as(u32, @bitCast(hf)) <= 1;
        try std.testing.expect(if (i == 37) hf != own and one_ulp else hf == own);
    }
}
