//! Flash Next on a HIP rank: the device weights the forward reads, and the dimensions this rank sees.
const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");

pub const Projection = hip.ops.Projection;
pub const Kind = hip.ops.Kind;
pub const Routed = hip.moe.Routed;

/// The dimensions of the checkpoint, and this rank's share of the heads tensor parallelism splits.
pub const Dims = struct {
    hidden: usize,
    streams: usize,
    low: usize,
    vocab: usize,
    layers: usize,
    eps: f32,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary: usize,
    theta: f32,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    index_heads: usize,
    index_dim: usize,
    index_ratio: usize,
    index_top: usize,
    ple_dim: usize,
    ple_taps: usize,
    ngram: usize,
    /// The rank's slice of the vocabulary rows of the head.
    vocab_rows: usize,
    /// The DeltaNet norm's output gate, as the checkpoint's `output_gate_type` names it.
    gate: hip.ops.Gate,

    pub fn wide(d: Dims) usize {
        return d.hidden * d.streams;
    }

    pub fn convWidth(d: Dims) usize {
        return 2 * d.key_heads * d.key_dim + d.value_heads * d.value_dim;
    }

    /// This rank's dimensions: query, value and key heads by rank, a kv head held by `world / kv_heads` ranks.
    pub fn of(c: *const config.Config, world: usize) error{UnevenSplit}!Dims {
        for ([_]usize{ c.heads, c.key_heads, c.value_heads, c.vocab }) |n| if (n % world != 0) return error.UnevenSplit;
        if (world > c.kv_heads and world % c.kv_heads != 0) return error.UnevenSplit;
        return .{
            .hidden = c.hidden,
            .streams = c.hc_count,
            .low = c.hc_lowrank,
            .vocab = c.vocab,
            .layers = c.layers,
            .eps = c.eps,
            .heads = c.heads / world,
            .kv_heads = @max(1, c.kv_heads / world),
            .head_dim = c.head_dim,
            .rotary = c.rotary_dim,
            .theta = @floatCast(c.rope_theta),
            .key_heads = c.key_heads / world,
            .value_heads = c.value_heads / world,
            .key_dim = c.key_dim,
            .value_dim = c.value_dim,
            .conv = c.conv,
            .index_heads = c.index_heads,
            .index_dim = c.index_dim,
            .index_ratio = c.index_ratio,
            .index_top = c.index_budget / c.index_ratio,
            .ple_dim = c.ple_dim,
            .ple_taps = c.ple_conv,
            .ngram = c.ngram,
            .vocab_rows = c.vocab / world,
            .gate = switch (c.gate) {
                .sigmoid => .sigmoid,
                .silu => .silu,
            },
        };
    }
};

/// A hyper-connection: its stream norm's fp32 scale (W), the down projection with the inject rows under it, the up.
pub const Hc = struct { scale: u64, down: Projection, up: Projection };

pub const Linear = struct {
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    out: Projection,
    /// fp32 (channels, taps), a_log and dt_bias fp32 (value heads), the gated norm's fp32 weight (value dim).
    conv: u64,
    a_log: u64,
    dt_bias: u64,
    norm: u64,
};

pub const Sparse = struct {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    index: Projection,
    /// fp32 scales: q and k norms (head dim), the indexer's q and k norms (index dim).
    q_norm: u64,
    k_norm: u64,
    iq_norm: u64,
    ik_norm: u64,
};

pub const Mixer = union(enum) { linear: Linear, sparse: Sparse };

pub const Layer = struct { ahc: Hc, mhc: Hc, mixer: Mixer, moe: Routed };

/// The n-gram layer: key and value projections stacked, three norms' fp32 scales (W), the conv's fp32 taps (W, taps).
pub const Ple = struct { layer: usize, kv: Projection, key_scale: u64, query_scale: u64, conv_scale: u64, conv: u64 };

/// The MTP head, whole on the drafting rank: input norms, projections, one sparse layer, its mixer, draft-vocab rows.
pub const Head = struct {
    e_norm: u64,
    h_norm: u64,
    fc_e: Projection,
    fc_h: Projection,
    layer: Layer,
    mix: Hc,
    dims: Dims,
    draft: Projection,
};

/// The DeltaNet's q and k normalization as RMS norms: constant fp32 weights (key dim wide) and their eps.
pub const Qk = struct { q: u64, k: u64, eps: f32 };

pub const Model = struct {
    dims: Dims,
    qk: Qk = undefined,
    act: Kind,
    /// The tensor-parallel communicator; null on one rank.
    tp: ?hip.collective.Comm = null,
    embed: Projection,
    layers: []const Layer,
    mix: Hc,
    /// The output head: this rank's rows of the vocabulary.
    head: Projection,
    ple: Ple,
    mtp: ?Head = null,
};

test "a rank's dimensions: heads, value heads and the vocabulary split, the kv heads held two ranks a head" {
    var c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const d = try Dims.of(&c, 4);
    try std.testing.expectEqual(@as(usize, 6), d.heads);
    try std.testing.expectEqual(@as(usize, 1), d.kv_heads);
    try std.testing.expectEqual(@as(usize, 12), d.value_heads);
    try std.testing.expectEqual(@as(usize, 4), d.key_heads);
    try std.testing.expectEqual(@as(usize, 512), d.index_top);
    try std.testing.expectEqual(@as(usize, 2 * 4 * 128 + 12 * 128), d.convWidth());
    try std.testing.expectError(error.UnevenSplit, Dims.of(&c, 3));
    try std.testing.expectEqual(hip.ops.Gate.sigmoid, d.gate);
}

test "the DeltaNet norm's gate is the one the checkpoint declares: a SiLU checkpoint runs SiLU" {
    const gpa = std.testing.allocator;
    const fixture = @embedFile("../fixtures/config.json");
    const sigmoid = "\"output_gate_type\": \"sigmoid\"";
    const text = try std.mem.replaceOwned(u8, gpa, fixture, sigmoid, "\"output_gate_type\": \"silu\"");
    defer gpa.free(text);
    try std.testing.expect(!std.mem.eql(u8, text, fixture));
    var c = try config.parse(gpa, text);
    defer c.deinit();
    try std.testing.expectEqual(hip.ops.Gate.silu, (try Dims.of(&c, 1)).gate);
    try std.testing.expectEqual(hip.ops.Gate.silu, (try Dims.of(&c, 4)).gate);
}
