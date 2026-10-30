//! A Liquid stack's description (LFM2 or LFM3 layers), shared by the CUDA and Metal stacks.
const std = @import("std");

pub const Kind = enum { conv, full, sliding };

pub const Rope = struct {
    theta: f32,
    /// Rotated share of the head (LFM3's zero-tail partial rotation keeps the full head's frequency progression).
    fraction: f32 = 1.0,
};

pub const Spec = struct {
    name: []const u8,
    /// Weight names: "{prefix}layers.{i}...." and "{prefix}embedding_norm.weight".
    prefix: []const u8,
    d: usize,
    ffn: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    conv_k: usize,
    kinds: []const Kind,
    /// Keys a sliding layer (and, in the speech encoder, every attention layer) sees, its own row included.
    window: usize = 0,
    /// The speech encoder masks every attention layer to `window`.
    window_all: bool = false,
    causal: bool = true,
    sandwich: bool,
    /// LFM3 attention: a sigmoid output gate per head and q_norm / k_norm / o_proj names; LFM2: q_layernorm / out_proj.
    lfm3: bool,
    rope_full: Rope,
    rope_sliding: Rope = .{ .theta = 10000 },
    /// 0: fp32 cos/sin; 1: bf16 cos/sin, fp32 products; 2: bf16 arithmetic.
    rope_mode: u8,
    eps: f32 = 1e-5,
    /// Rows one forward takes (scratch) and rows the caches hold.
    max_rows: usize,
    max_ctx: usize,
};

/// Inverse frequencies of `hd / 2` pairs, zero past the rotated share, as the reference computes them in fp32.
pub fn invFreq(out: []f32, r: Rope, hd: usize) void {
    const rot: usize = @as(usize, @intFromFloat(r.fraction * @as(f32, @floatFromInt(hd)))) / 2;
    for (out, 0..) |*x, i| x.* = if (i < rot) 1.0 / std.math.pow(f32, r.theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(hd))) else 0;
}
