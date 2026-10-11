//! Flash Next's bytes on one HIP rank: the weights as tensor parallelism splits them, a token's keys, a stream's state.
const std = @import("std");
const config = @import("../config.zig");
const Spec = @import("../spec.zig").Spec;
const Config = config.Config;

/// Where a checkpoint's text weights live on a rank of `world`.
pub const Weights = struct {
    /// On every rank: hyper-connections, routers, the embedding, indexers, the PLE projections, norms.
    replicated: u64 = 0,
    /// This rank's share: routed experts by rank, heads and value heads, the vocabulary rows of the head.
    split: u64 = 0,
    /// Rank 0 alone: the shared experts.
    shared: u64 = 0,
    /// The n-gram tables, read from host memory a few rows a token.
    host: u64 = 0,

    pub fn device(w: Weights, rank: u32) u64 {
        return w.replicated + w.split + if (rank == 0) w.shared else 0;
    }

    pub fn total(w: Weights, world: u32) u64 {
        return w.replicated + w.split * world + w.shared + w.host;
    }
};

/// Bytes of an MLX affine matrix: packed codes, then a bf16 scale and bias a group.
fn affine(s: Spec, n: u64, k: u64) u64 {
    return n * k * s.bits / 8 + 4 * n * (k / s.group);
}

const bf16 = 2;

fn even(n: u64, world: u32) !u64 {
    if (n % world != 0) return error.UnevenSplit;
    return n / world;
}

/// One decoder layer's bytes; `kind` names its mixer, as the config's layer types do.
fn layer(c: *const Config, s: Spec, kind: config.Kind, world: u32, w: *Weights) !void {
    const d: u64 = c.hidden;
    const wide: u64 = try c.wide();
    for (0..2) |_| w.replicated += 2 * affine(s, c.hc_lowrank, wide) + affine(s, c.hc_count, wide) + wide * bf16;
    w.replicated += c.experts * d * bf16; // the router is stored in bf16
    w.shared += 3 * affine(s, c.shared_width, d) + affine(s, 1, d);
    w.split += try even(c.experts, world) * 3 * affine(s, c.expert_width, d);
    if (kind == .linear_attention) {
        const keys: u64 = try c.keyWidth();
        const values: u64 = try c.valueWidth();
        const heads = try even(c.value_heads, world);
        _ = try even(c.key_heads, world);
        const conv = (2 * keys + values) / world;
        w.split += affine(s, conv, d) + affine(s, values / world, d) + 2 * affine(s, heads, d) + affine(s, d, values / world);
        w.split += conv * c.conv * bf16 + 2 * heads * bf16;
        w.replicated += c.value_dim * bf16;
    } else {
        const hd: u64 = c.head_dim;
        const heads = try even(c.heads, world);
        const kv: u64 = c.kv_heads / @min(world, c.kv_heads);
        w.split += affine(s, 2 * heads * hd, d) + affine(s, d, heads * hd) + 2 * affine(s, kv * hd, d);
        w.replicated += affine(s, (c.index_heads + c.index_kv_heads) * c.index_dim, d) + 2 * hd * bf16 + 2 * c.index_dim * bf16;
    }
}

/// The weights of `c` with every matrix at `s` on one rank of `world` (none of the vision tower).
pub fn weights(c: *const Config, s: Spec, world: u32) !Weights {
    var w: Weights = .{};
    const d: u64 = c.hidden;
    const wide: u64 = try c.wide();
    w.replicated += affine(s, c.vocab, d) + 2 * affine(s, c.hc_lowrank, wide) + wide * bf16;
    w.split += affine(s, try even(c.vocab, world), d);
    for (0..c.layers) |i| {
        try layer(c, s, try c.kind(i), world, &w);
        if (!c.ple[i]) continue;
        const heads: u64 = try c.ngramHeadCount();
        w.replicated += affine(s, wide, c.ple_dim) + affine(s, d, c.ple_dim) + 3 * wide * bf16 + wide * c.ple_conv * bf16 + (3 + 2 * heads) * 8;
        w.host += c.ngram_shards * affine(s, try tableRows(c), c.ple_dim / heads);
    }
    if (c.mtp.layers > 0) {
        w.replicated += 2 * affine(s, d, d) + (d + wide) * bf16 + 2 * affine(s, c.hc_lowrank, wide) + wide * bf16;
        for (c.mtp.kinds[0..c.mtp.layers]) |kind| try layer(c, s, kind, world, &w);
    }
    return w;
}

/// Rows of one n-gram table shard: the heads' prime table sizes, together, cut in `ngram_shards` equal parts.
pub fn tableRows(c: *const Config) !u64 {
    const ref = try config.pleRef(c);
    var rows: u64 = 0;
    for (ref.sizes) |n| rows += @intCast(n);
    return std.math.divCeil(u64, rows, c.ngram_shards);
}

/// A position's cache bytes on one rank: keys and values (kv_dtype bytes), the indexer's raw and pooled keys (bf16).
pub fn tokenBytes(c: *const Config, world: u32, kv_dtype: u64) !u64 {
    var layers: u64 = c.mtp.layers;
    for (0..c.layers) |i| layers += @intFromBool(try c.kind(i) == .sparse_attention);
    const kv: u64 = c.kv_heads / @min(world, c.kv_heads);
    const index: u64 = c.index_kv_heads * c.index_dim * bf16;
    return layers * (2 * kv * c.head_dim * kv_dtype + index + index / c.index_ratio);
}

/// A stream's state on one rank: the DeltaNet recurrences (fp32) and conv tails, the PLE conv tail and history.
pub fn streamBytes(c: *const Config, world: u32) !u64 {
    var linear: u64 = 0;
    for (0..c.layers) |i| linear += @intFromBool(try c.kind(i) == .linear_attention);
    const heads = try even(c.value_heads, world);
    const conv: u64 = (try c.convWidth()) / world;
    const ple: u64 = c.ple_count * c.ngram * (c.ple_conv - 1) * (try c.wide()) * bf16;
    return linear * (heads * c.key_dim * c.value_dim * 4 + (c.conv - 1) * conv * bf16) + ple;
}

/// What a rank needs at `window` positions for `streams` streams, past its weights and `scratch`, against `room`.
pub const Fit = struct {
    weights: u64,
    lanes: u64,
    room: u64,

    pub fn fits(f: Fit) bool {
        return f.weights + f.lanes <= f.room;
    }

    /// Positions one stream may hold in what the weights and `scratch` leave of `room`, or 0.
    pub fn window(room: u64, weights_bytes: u64, scratch: u64, token: u64, stream: u64) u64 {
        const fixed = weights_bytes + scratch + stream;
        return if (fixed >= room) 0 else (room - fixed) / token;
    }
};

pub fn fit(c: *const Config, s: Spec, world: u32, rank: u32, window: u64, streams: u64, scratch: u64, room: u64) !Fit {
    const w = try weights(c, s, world);
    const lanes = streams * (window * try tokenBytes(c, world, bf16) + try streamBytes(c, world)) + scratch;
    return .{ .weights = w.device(rank), .lanes = lanes, .room = room };
}

test "the plan counts every text byte of the 3-, 4- and 6-bit checkpoints, and splits them without loss" {
    const fixture = @embedFile("../fixtures/config.json");
    // the published checkpoints' safetensors headers, less the vision tower
    const totals = [_]struct { bits: []const u8, bytes: u64 }{ .{ .bits = "6", .bytes = 157182349016 }, .{ .bits = "4", .bytes = 112311293016 }, .{ .bits = "3", .bytes = 89875765016 } };
    for (totals) |t| {
        var buf: [16]u8 = undefined;
        const text = try std.mem.replaceOwned(u8, std.testing.allocator, fixture, "\"bits\": 6", try std.fmt.bufPrint(&buf, "\"bits\": {s}", .{t.bits}));
        defer std.testing.allocator.free(text);
        var c = try config.parse(std.testing.allocator, text);
        defer c.deinit();
        const one = try weights(&c, c.global_affine, 1);
        try std.testing.expectEqual(t.bytes, one.total(1));
        for ([_]u32{ 2, 4 }) |world| {
            const w = try weights(&c, c.global_affine, world);
            try std.testing.expectEqual(one.host, w.host);
            try std.testing.expectEqual(one.shared, w.shared);
            try std.testing.expect(w.total(world) >= one.total(1)); // replicated parts add, nothing is lost
        }
    }
}

test "a 4-bit checkpoint fits four 30 GiB cards at the model's window; 4-bit on two 45 GiB cards leaves a shorter one" {
    var c = try config.parse(std.testing.allocator, @embedFile("../fixtures/config.json"));
    defer c.deinit();
    const q4 = try Spec.init(4, 32);
    const q3 = try Spec.init(3, 32);
    const gib: u64 = 1 << 30;
    const v620: u64 = 28 * gib; // a V620's free memory less its reserve, near enough
    const w7800: u64 = 43 * gib;
    try std.testing.expect((try fit(&c, q4, 4, 0, 262144, 1, 2 * gib, v620)).fits());
    try std.testing.expect(!(try fit(&c, q4, 1, 0, 4096, 1, 2 * gib, v620)).fits());
    try std.testing.expect(!(try fit(&c, q4, 2, 0, 262144, 1, 2 * gib, w7800)).fits());
    try std.testing.expect((try fit(&c, q4, 2, 0, 65536, 1, 2 * gib, w7800)).fits());
    try std.testing.expect((try fit(&c, q3, 2, 0, 262144, 1, 2 * gib, w7800)).fits());
    try std.testing.expectError(error.UnevenSplit, weights(&c, q4, 3));
    // the KV and indexer bytes of a position: 13 attention layers (12 and the MTP head's), one kv head a rank past one
    try std.testing.expectEqual(@as(u64, 13 * (2 * 2 * 256 * 2 + 256 + 64)), try tokenBytes(&c, 1, 2));
    try std.testing.expectEqual(@as(u64, 13 * (2 * 1 * 256 * 2 + 256 + 64)), try tokenBytes(&c, 4, 2));
}
