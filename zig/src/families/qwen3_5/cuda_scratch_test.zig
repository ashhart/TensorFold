//! Host-only interval checks use the actual arena carver; no CUDA allocation or launch occurs.
const std = @import("std");
const E = @import("cuda_engine.zig");
const scratch = @import("cuda_scratch.zig");
const T = std.testing;
const R = E.max_rows;
const geometry = @import("cuda_geometry.zig");
const Span = struct { begin: u64, end: u64 };

fn span(address: u64, bytes: usize) Span {
    return .{ .begin = address, .end = address + bytes };
}
fn separate(a: Span, b: Span) bool {
    return a.end <= b.begin or b.end <= a.begin;
}
fn disjoint(spans: []const Span) bool {
    for (spans, 0..) |a, i| for (spans[0..i]) |b| {
        if (!separate(a, b)) return false;
    };
    return true;
}
fn dense() E.Geometry {
    return .{
        .hidden = 5120,
        .intermediate = 17408,
        .layers = 64,
        .vocab = 248320,
        .k_heads = 16,
        .v_heads = 48,
        .q_heads = 24,
        .kv_heads = 4,
        .eps = 1e-6,
        .tied = false,
        .context = 131072,
        .linear = @splat(false),
    };
}
fn engine(g: E.Geometry) E.Engine {
    var e: E.Engine = undefined;
    e.g = g;
    e.arena_at = 512;
    e.arena = .{ .d = undefined, .ptr = 0x100000000, .len = 1 << 40, .borrowed = true };
    return e;
}
fn common(e: *const E.Engine) [4]Span {
    return .{
        span(e.h, R * e.g.hidden * 2), span(e.x, R * e.g.hidden * 2),
        span(e.r, R * e.g.hidden * 2), span(e.xs, R * (@max(e.g.intermediate, wide(e.g)) / 64) * 4),
    };
}
fn wide(g: E.Geometry) usize {
    return @max(g.hidden, g.valueWidth(), g.attnWidth());
}
fn partialBytes(g: E.Geometry) usize {
    return E.most_verify * g.q_heads * E.max_chunks * 4 * (2 + geometry.head_dim);
}
fn deltaBytes(g: E.Geometry) usize {
    return 4 * R * wide(g) * 2 + 2 * R * g.k_heads * geometry.dk * 2 +
        R * g.convWidth() * 2 + 3 * R * g.v_heads * 2 + R * g.v_heads * 4;
}
fn attentionBytes(g: E.Geometry) usize {
    return 2 * R * wide(g) * 2 + R * 2 * g.attnWidth() * 2 +
        R * g.attnWidth() * 2 + 3 * R * g.kv_heads * geometry.head_dim * 2;
}
fn mlpBytes(g: E.Geometry) usize {
    return 3 * R * g.intermediate * 2;
}
fn expectedSaving(g: E.Geometry) usize {
    return @max(deltaBytes(g), attentionBytes(g), mlpBytes(g)) + partialBytes(g) -
        @max(deltaBytes(g), attentionBytes(g) + partialBytes(g), mlpBytes(g));
}
fn attention(e: *const E.Engine) [10]Span {
    const g = e.g;
    const p = E.most_verify * g.q_heads * E.max_chunks;
    return .{
        span(e.y, R * wide(g) * 2),                           span(e.mix, R * wide(g) * 2),
        span(e.qraw, R * 2 * g.attnWidth() * 2),              span(e.qn, R * g.attnWidth() * 2),
        span(e.kraw, R * g.kv_heads * geometry.head_dim * 2), span(e.vraw, R * g.kv_heads * geometry.head_dim * 2),
        span(e.kn, R * g.kv_heads * geometry.head_dim * 2),   span(e.pm, p * 4),
        span(e.pl, p * 4),                                    span(e.po, p * geometry.head_dim * 4),
    };
}
fn delta(e: *const E.Engine) [11]Span {
    const g = e.g;
    return .{
        span(e.y, R * wide(g) * 2),                 span(e.mix, R * wide(g) * 2),
        span(e.z, R * wide(g) * 2),                 span(e.v, R * wide(g) * 2),
        span(e.q, R * g.k_heads * geometry.dk * 2), span(e.k, R * g.k_heads * geometry.dk * 2),
        span(e.qkv, R * g.convWidth() * 2),         span(e.a, R * g.v_heads * 2),
        span(e.b, R * g.v_heads * 2),               span(e.beta, R * g.v_heads * 2),
        span(e.g_, R * g.v_heads * 4),
    };
}
fn mlp(e: *const E.Engine) [3]Span {
    return .{
        span(e.gate, R * e.g.intermediate * 2), span(e.up, R * e.g.intermediate * 2),
        span(e.act, R * e.g.intermediate * 2),
    };
}
fn tail(e: *const E.Engine) [3]Span {
    return .{
        span(e.logits, E.most_verify * e.g.vocab * 2),
        span(e.amax, E.most_verify * E.argmaxSlices(e.g.vocab) * 8),
        span(e.picks, E.most_verify * @import("cuda").sample.Draws.row_scratch),
    };
}
fn phasesSafe(e: *const E.Engine) !void {
    const shared = common(e);
    const after = tail(e);
    try T.expect(disjoint(&shared));
    try T.expect(disjoint(&after));
    inline for (.{ delta(e), attention(e), mlp(e) }) |phase| {
        try T.expect(disjoint(&phase));
        for (phase) |view| {
            try T.expect(view.begin % 256 == 0);
            try T.expect(view.begin >= e.arena.ptr and view.end <= e.arena.ptr + e.arena_at);
            for (shared) |live| try T.expect(separate(view, live));
            for (after) |live| try T.expect(separate(view, live));
        }
    }
}

test "scratch overlay: dense27 footprint removes only the dead inter-phase space" {
    var e = engine(dense());
    const before = e.arena_at;
    const bytes = try scratch.scratch(&e, .allocate);
    const fixed = 3 * R * e.g.hidden * 2 + R * (e.g.intermediate / 64) * 4;
    const tail_bytes = E.most_verify * e.g.vocab * 2 + E.most_verify * E.argmaxSlices(e.g.vocab) * 8 +
        E.most_verify * @import("cuda").sample.Draws.row_scratch;
    const legacy = fixed + @max(deltaBytes(e.g), attentionBytes(e.g), mlpBytes(e.g)) + partialBytes(e.g) + tail_bytes;
    try T.expectEqual(expectedSaving(e.g), legacy - bytes);
    try T.expectEqual(@as(usize, if (R == 8192) 202899456 else 150994944), expectedSaving(e.g));
    if (R == 8192) {
        try T.expectEqual(@as(usize, 202899456), legacy - bytes);
        try T.expectEqual(@as(usize, 1180890112), bytes);
    }
    try T.expectEqual(before + bytes, e.arena_at);
    try T.expect(e.pm >= e.qn + R * e.g.attnWidth() * 2);
    if (R == 8192) try T.expect(e.po + E.most_verify * e.g.q_heads * E.max_chunks * geometry.head_dim * 4 <= e.gate + mlpBytes(e.g));
    try phasesSafe(&e);
}

test "scratch overlay: measure matches allocation and refuses a one-byte-short arena" {
    var measured = engine(dense());
    const at = measured.arena_at;
    const required = try scratch.scratch(&measured, .measure);
    try T.expectEqual(at, measured.arena_at);
    var exact = engine(dense());
    exact.arena.len = at + required;
    try T.expectEqual(required, try scratch.scratch(&exact, .allocate));
    var short = engine(dense());
    short.arena.len = at + required - 1;
    try T.expectError(error.ArenaExhausted, scratch.scratch(&short, .allocate));
}

test "scratch overlay: other dense and MoE geometries retain every live phase" {
    var small = dense();
    small.hidden = 2048;
    small.intermediate = 2048;
    small.vocab = 32000;
    small.k_heads = 16;
    small.v_heads = 16;
    small.q_heads = 16;
    small.kv_heads = 2;
    var moe = small;
    moe.experts = 8;
    moe.top_k = 2;
    for ([_]E.Geometry{ small, moe }) |g| {
        var e = engine(g);
        const bytes = try scratch.scratch(&e, .allocate);
        var measured = engine(g);
        try T.expectEqual(bytes, try scratch.scratch(&measured, .measure));
        try phasesSafe(&e);
        if (g.moe()) {
            const pairs = R * g.slots();
            const count = g.experts + 1;
            const routed = [_]Span{
                span(e.routed.logits, R * count * 4),
                span(e.routed.picks, pairs * 4),
                span(e.routed.wts, pairs * 4),
                span(e.routed.members, pairs * 4),
                span(e.routed.items, 3 * @import("cuda").grouped.maxItems(pairs, count, 16) * 4),
                span(e.routed.counts, 8),
                span(e.routed.rank, pairs * 4),
                span(e.routed.hist, ((pairs + 1023) / 1024) * count * 4),
                span(e.routed.act, pairs * g.intermediate * 2),
                span(e.routed.y, pairs * g.hidden * 4),
            };
            try T.expect(disjoint(&routed));
            for (routed) |view| {
                try T.expect(view.begin >= e.logits + E.most_verify * g.vocab * 2);
                try T.expect(view.end <= e.amax);
                for (attention(&e)) |live| try T.expect(separate(view, live));
            }
        }
    }
}

test "scratch overlay: interval oracle detects live attention aliasing" {
    var e = engine(dense());
    _ = try scratch.scratch(&e, .allocate);
    try T.expect(disjoint(&attention(&e)));
    e.pm = e.qn;
    try T.expect(!disjoint(&attention(&e)));
}
