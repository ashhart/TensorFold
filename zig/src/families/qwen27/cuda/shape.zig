//! The shapes the CUDA engine is built for: the family's parsed config (config.zig), held to the 27B's.
const std = @import("std");
const config = @import("../config.zig");

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

/// A checkpoint's config.json (and generation_config.json's end tokens), refused unless it is the 27B's.
pub const Config = struct {
    g: Geometry,
    eos: [config.max_eos]u32,
    eos_count: usize,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const text = try file(gpa, io, dir, "config.json") orelse return error.MissingConfig;
        defer gpa.free(text);
        const generation = try file(gpa, io, dir, "generation_config.json");
        defer if (generation) |x| gpa.free(x);
        return check(try config.parse(gpa, text, generation));
    }

    fn file(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?[]u8 {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => err,
        };
    }
};

/// The parsed config as the CUDA engine's Config, when every shape the kernels assume is the 27B's.
pub fn check(c: config.Config) !Config {
    const g: Geometry = .{ .hidden = c.hidden, .intermediate = c.intermediate, .layers = c.layers, .linear_k_heads = c.k_heads, .linear_v_heads = c.v_heads, .query_heads = c.heads, .kv_heads = c.kv_heads };
    if (!std.meta.eql(g, the_27b) or c.vocab != vocab or c.head_dim != head_dim or c.dk != linear_dim or c.dv != linear_dim) return error.UnsupportedQwenGeometry;
    if (c.conv_kernel != conv_taps or c.rope_dims != rotary_dim or c.eps != eps or c.rope_theta != theta) return error.UnsupportedQwenGeometry;
    for (c.kinds[0..c.layers], 0..) |kind, i| if ((kind == .linear) != linear(i)) return error.UnsupportedQwenGeometry;
    return .{ .g = g, .eos = c.eos, .eos_count = c.eos_count };
}

test "the 27B's shapes pass and another hidden size is refused" {
    var c: config.Config = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = vocab, .heads = 24, .kv_heads = 4, .head_dim = head_dim, .k_heads = 16, .v_heads = 48, .dk = linear_dim, .dv = linear_dim, .conv_kernel = conv_taps, .eps = eps, .rope_dims = rotary_dim, .rope_theta = theta };
    for (c.kinds[0..64], 0..) |*k, i| k.* = if (linear(i)) .linear else .attention;
    c.eos[0] = 248044;
    c.eos_count = 1;
    const ok = try check(c);
    try std.testing.expectEqual(@as(usize, 1), ok.eos_count);
    c.hidden = 2048;
    try std.testing.expectError(error.UnsupportedQwenGeometry, check(c));
}
