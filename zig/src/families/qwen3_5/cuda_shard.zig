//! A TP rank's share of the dense Qwen3.5 hybrids on CUDA (the 27B): whole heads, widths in 128s (EXL3 tiles).

const std = @import("std");

/// Elements of an EXL3 tile: every n and k of a cut weight is a multiple of it (cuda_exl3_admit.zig).
pub const unit = 128;
/// qwen35.cu's fixed widths (cuda_geometry.zig): an attention head's rows, and a linear-attention head's rows.
pub const head_dim = 256;
pub const dk = 128;
/// The most ranks a group holds (fabric/mesh/shard.zig's max_world).
pub const max_world = 256;

/// UnevenSplit: a head count; KvHeads: the KV heads; NotOnTiles: a width or a half off the 128s, never rounded.
pub const Error = error{ BadRank, UnevenSplit, KvHeads, NotOnTiles, EmptyShare };

pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

/// The widths a plan cuts, as Geometry reads them from config.json.
pub const Shape = struct {
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    q_heads: usize, // attention query heads
    kv_heads: usize, // attention key/value heads
    k_heads: usize, // linear-attention key heads
    v_heads: usize, // linear-attention value heads
};

/// One rank's spans, each in its weight's own rows (a row-split weight's are columns); even shares, no rank larger.
pub const Share = struct {
    rank: usize,
    world: usize,
    q: Span, // q_proj rows: whole query heads, each head's q and gate rows together
    kv: Span, // k_proj and v_proj rows: whole KV heads; past kv_heads ranks one head is kept by its query ranks
    o: Span, // o_proj columns (k): the attention width of the same query heads
    qkv: [3]Span, // in_proj_qkv rows: its q, k and v segments each cut evenly, at their offsets
    z: Span, // in_proj_z rows: whole value heads
    heads: Span, // in_proj_a and in_proj_b rows, A_log and dt_bias: the same value heads
    out: Span, // out_proj columns (k): the value heads of z
    mlp: Span, // gate_proj and up_proj rows, and down_proj columns
    vocab: Span, // lm_head rows

    /// Rank `rank` of `world`; refused where a count does not split evenly or a cut would leave the 128-tiles.
    pub fn of(s: Shape, rank: usize, world: usize) Error!Share {
        if (world == 0 or world > max_world or rank >= world) return error.BadRank;
        if (s.hidden % unit != 0) return error.NotOnTiles; // the k of every column cut and the n of every row cut
        const q = try evenly(s.q_heads, rank, world);
        const kv = try kvHeads(s.kv_heads, rank, world);
        const k = try evenly(s.k_heads, rank, world);
        const v = try evenly(s.v_heads, rank, world);
        const mlp = try tiles(s.intermediate, rank, world);
        const vocab = try tiles(s.vocab, rank, world);
        const keys = s.k_heads * dk;
        return .{
            .rank = rank,
            .world = world,
            .q = scale(q, 2 * head_dim),
            .kv = scale(kv, head_dim),
            .o = scale(q, head_dim),
            .qkv = .{ scale(k, dk), shift(scale(k, dk), keys), shift(scale(v, dk), 2 * keys) },
            .z = scale(v, dk),
            .heads = v,
            .out = scale(v, dk),
            .mlp = mlp,
            .vocab = vocab,
        };
    }

    /// conv1d's channels are in_proj_qkv's rows, so it is cut by the same three spans.
    pub fn conv(s: Share) [3]Span {
        return s.qkv;
    }
};

/// Rank `rank`'s [lo, hi) of `total` units (heads, or 128-tiles), split evenly.
fn evenly(total: usize, rank: usize, world: usize) Error!Span {
    if (total % world != 0) return error.UnevenSplit;
    if (total == 0) return error.EmptyShare;
    const part = total / world;
    return .{ .lo = rank * part, .hi = (rank + 1) * part };
}

/// Rank `rank`'s KV heads: whole heads a rank, or, with fewer heads than ranks, one head kept by its query ranks.
fn kvHeads(total: usize, rank: usize, world: usize) Error!Span {
    if (total >= world) {
        if (total % world != 0) return error.KvHeads;
        return evenly(total, rank, world);
    }
    if (total == 0 or world % total != 0) return error.KvHeads;
    const head = rank / (world / total);
    return .{ .lo = head, .hi = head + 1 };
}

/// Rank `rank`'s [lo, hi) of `total` elements in whole 128-tiles; refused, not rounded, where a half would leave them.
fn tiles(total: usize, rank: usize, world: usize) Error!Span {
    if (total % unit != 0) return error.NotOnTiles;
    const part = evenly(total / unit, rank, world) catch |e| return if (e == error.UnevenSplit) error.NotOnTiles else e;
    return scale(part, unit);
}

fn scale(s: Span, by: usize) Span {
    return .{ .lo = s.lo * by, .hi = s.hi * by };
}

fn shift(s: Span, by: usize) Span {
    return .{ .lo = s.lo + by, .hi = s.hi + by };
}

/// The 27B's widths (cuda_geometry.zig's test of them): 24 query heads over 4 KV heads, 48 value heads over 16.
const qwen27: Shape = .{ .hidden = 5120, .intermediate = 17408, .vocab = 248320, .q_heads = 24, .kv_heads = 4, .k_heads = 16, .v_heads = 48 };

test "TP=2 halves every weight of the 27B: whole heads, 2,048-wide qkv segments cut at 1,024, the vocabulary at 124,160" {
    const r0 = try Share.of(qwen27, 0, 2);
    const r1 = try Share.of(qwen27, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 6144 }, r0.q); // 12 heads of 512 rows
    try std.testing.expectEqual(Span{ .lo = 6144, .hi = 12288 }, r1.q);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 512 }, r0.kv); // 2 of 4 KV heads
    try std.testing.expectEqual(Span{ .lo = 512, .hi = 1024 }, r1.kv);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 3072 }, r0.o);
    try std.testing.expectEqual(Span{ .lo = 3072, .hi = 6144 }, r1.o);
    try std.testing.expectEqual([3]Span{ .{ .lo = 0, .hi = 1024 }, .{ .lo = 2048, .hi = 3072 }, .{ .lo = 4096, .hi = 7168 } }, r0.qkv);
    try std.testing.expectEqual([3]Span{ .{ .lo = 1024, .hi = 2048 }, .{ .lo = 3072, .hi = 4096 }, .{ .lo = 7168, .hi = 10240 } }, r1.qkv);
    try std.testing.expectEqual(r1.qkv, r1.conv());
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 3072 }, r0.z);
    try std.testing.expectEqual(Span{ .lo = 3072, .hi = 6144 }, r1.z);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 24 }, r0.heads); // 24 of 48 value heads
    try std.testing.expectEqual(Span{ .lo = 24, .hi = 48 }, r1.heads);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 3072 }, r0.out);
    try std.testing.expectEqual(Span{ .lo = 3072, .hi = 6144 }, r1.out);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 8704 }, r0.mlp); // 68 of 136 tiles
    try std.testing.expectEqual(Span{ .lo = 8704, .hi = 17408 }, r1.mlp);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 124160 }, r0.vocab); // 970 of 1,940 tiles
    try std.testing.expectEqual(Span{ .lo = 124160, .hi = 248320 }, r1.vocab);
}

test "TP=1 keeps every weight whole" {
    const r = try Share.of(qwen27, 0, 1);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 12288 }, r.q);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 1024 }, r.kv);
    try std.testing.expectEqual([3]Span{ .{ .lo = 0, .hi = 2048 }, .{ .lo = 2048, .hi = 4096 }, .{ .lo = 4096, .hi = 10240 } }, r.qkv);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 248320 }, r.vocab);
}

test "TP=4 takes a KV head and 6 query heads a rank; TP=8 keeps each KV head on two ranks and refuses the head's tiles" {
    const r3 = try Share.of(qwen27, 3, 4);
    try std.testing.expectEqual(Span{ .lo = 9216, .hi = 12288 }, r3.q);
    try std.testing.expectEqual(Span{ .lo = 768, .hi = 1024 }, r3.kv);
    try std.testing.expectEqual(Span{ .lo = 13056, .hi = 17408 }, r3.mlp);
    try std.testing.expectEqual(Span{ .lo = 186240, .hi = 248320 }, r3.vocab);
    // 1,940 vocabulary tiles do not split by 8: refused on the head, the last width read
    try std.testing.expectError(error.NotOnTiles, Share.of(qwen27, 5, 8));
    var wide = qwen27;
    wide.vocab = 248832; // 1,944 tiles
    const r5 = try Share.of(wide, 5, 8);
    try std.testing.expectEqual(Span{ .lo = 7680, .hi = 9216 }, r5.q);
    try std.testing.expectEqual(Span{ .lo = 512, .hi = 768 }, r5.kv); // KV head 2, kept by ranks 4 and 5
    try std.testing.expectEqual(Span{ .lo = 1280, .hi = 1536 }, r5.qkv[0]); // key heads 10 and 11 of 16
}

test "a width or a half off the 128-tiles is refused by name, never rounded" {
    var s = qwen27;
    s.intermediate = 17408 + 64; // the whole width is off the tiles
    try std.testing.expectError(error.NotOnTiles, Share.of(s, 0, 2));
    s = qwen27;
    s.intermediate = 3 * 128; // whole tiles, but a half would be 192 wide
    try std.testing.expectError(error.NotOnTiles, Share.of(s, 1, 2));
    s = qwen27;
    s.hidden = 5120 + 64;
    try std.testing.expectError(error.NotOnTiles, Share.of(s, 0, 1));
    s = qwen27;
    s.vocab = 248320 + 1;
    try std.testing.expectError(error.NotOnTiles, Share.of(s, 0, 2));
}

test "heads that do not split, and ranks that are not there, are refused" {
    try std.testing.expectError(error.KvHeads, Share.of(qwen27, 0, 3)); // 4 KV heads over 3 ranks
    try std.testing.expectError(error.UnevenSplit, Share.of(qwen27, 0, 16)); // 24 query heads over 16
    try std.testing.expectError(error.BadRank, Share.of(qwen27, 2, 2));
    try std.testing.expectError(error.BadRank, Share.of(qwen27, 0, 0));
    try std.testing.expectError(error.BadRank, Share.of(qwen27, 0, max_world + 1));
    var s = qwen27;
    s.v_heads = 0;
    try std.testing.expectError(error.EmptyShare, Share.of(s, 0, 2));
}

test "a rank keeps the key head of each of its value heads and the KV head of each of its query heads" {
    var s = qwen27;
    s.vocab = 248832; // 1,944 tiles split by 8 too
    for ([_]usize{ 1, 2, 4, 8 }) |world| for (0..world) |rank| {
        const p = try Share.of(s, rank, world);
        const keys = s.k_heads * dk;
        const key_lo = (p.qkv[1].lo - keys) / dk;
        const key_hi = (p.qkv[1].hi - keys) / dk;
        for (p.heads.lo..p.heads.hi) |hv| {
            const hk = hv / (s.v_heads / s.k_heads);
            try std.testing.expect(hk >= key_lo and hk < key_hi);
        }
        try std.testing.expectEqual(p.heads.lo * dk, p.z.lo);
        try std.testing.expectEqual(p.heads.hi * dk, p.z.hi);
        for (p.q.lo / (2 * head_dim)..p.q.hi / (2 * head_dim)) |h| {
            const kv_head = h / (s.q_heads / s.kv_heads);
            try std.testing.expect(kv_head * head_dim >= p.kv.lo and (kv_head + 1) * head_dim <= p.kv.hi);
        }
    };
}
