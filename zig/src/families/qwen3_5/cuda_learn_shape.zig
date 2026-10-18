//! The shapes the CUDA engine is built for: the family's parsed config (config.zig), held to the 27B's.
const std = @import("std");
const owned = @import("cuda_geometry.zig");

pub const vocab = 248320;
pub const linear_dim = 128; // a DeltaNet key or value head's width
pub const conv_taps = 4;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000000;

/// Whether layer `index` is a DeltaNet layer: every fourth is attention.
pub fn linear(index: usize) bool {
    return index % 4 != 3;
}

/// The widths a forward sizes its buffers by.
pub const Geometry = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    linear_k_heads: usize, // DeltaNet key (and query) heads
    linear_v_heads: usize, // DeltaNet value heads, each key head serving a run of them
    query_heads: usize,
    kv_heads: usize,

    /// The DeltaNet input projection's row: queries, keys, then values.
    pub fn convDim(g: Geometry) usize {
        return (2 * g.linear_k_heads + g.linear_v_heads) * linear_dim;
    }
    pub fn kInner(g: Geometry) usize {
        return g.linear_k_heads * linear_dim;
    }
    pub fn vInner(g: Geometry) usize {
        return g.linear_v_heads * linear_dim;
    }
    pub fn qInner(g: Geometry) usize {
        return g.query_heads * head_dim;
    }
    pub fn kvInner(g: Geometry) usize {
        return g.kv_heads * head_dim;
    }
    /// The widest row an activation buffer holds: the recurrence's values or the attention's heads.
    pub fn inner(g: Geometry) usize {
        return @max(g.vInner(), g.qInner());
    }
    /// One layer's recurrent state (fp32) and its conv window (bf16).
    pub fn deltaBytes(g: Geometry) usize {
        return g.linear_v_heads * linear_dim * linear_dim * 4;
    }
    pub fn convBytes(g: Geometry) usize {
        return (conv_taps - 1) * g.convDim() * 2;
    }
};

/// Qwen3.8-27B, the one checkpoint shape the CUDA engine serves.
pub const the_27b: Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .linear_k_heads = 16, .linear_v_heads = 48, .query_heads = 24, .kv_heads = 4 };

/// This backward is qualified for the dense 27B geometry alone; serving keeps its wider registry.
pub fn from(g: owned.Geometry) !Geometry {
    const out: Geometry = .{ .hidden = g.hidden, .intermediate = g.intermediate, .layers = g.layers, .linear_k_heads = g.k_heads, .linear_v_heads = g.v_heads, .query_heads = g.q_heads, .kv_heads = g.kv_heads };
    if (!std.meta.eql(out, the_27b) or g.vocab != vocab or g.eps != eps) return error.UnsupportedLearningGeometry;
    for (g.linear[0..g.layers], 0..) |v, i| if (v != linear(i)) return error.UnsupportedLearningGeometry;
    if (g.moe() or g.bits.embed != 4 or g.bits.head != 4) return error.UnsupportedLearningFormat;
    for (g.bits.proj[0..g.layers]) |widths| for (widths) |bits| if (bits != 4) return error.UnsupportedLearningFormat;
    return out;
}

test "Dense learning: learning admits the dense affine4 layout and refuses mixed widths and routed experts" {
    var g: owned.Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = vocab, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = eps, .tied = false, .context = 4096, .linear = @splat(false) };
    for (0..g.layers) |i| g.linear[i] = linear(i);
    _ = try from(g);
    for ([_]owned.Slot{ .gate, .up, .down, .qkv, .o }) |slot| {
        g.bits.proj[1][@backingInt(slot)] = 6;
        try std.testing.expectError(error.UnsupportedLearningFormat, from(g));
        g.bits.proj[1][@backingInt(slot)] = 4;
    }
    g.bits.embed = 8;
    try std.testing.expectError(error.UnsupportedLearningFormat, from(g));
    g.bits.embed = 4;
    g.experts = 256;
    try std.testing.expectError(error.UnsupportedLearningFormat, from(g));
}
