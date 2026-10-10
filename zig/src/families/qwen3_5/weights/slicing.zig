const std = @import("std");
const config = @import("config.zig");
const host = @import("host.zig");

const quant = @import("core").quant;
const Allocator = std.mem.Allocator;
const takeRows = quant.slice.takeRows;
const even = quant.slice.even;

pub const Error = error{ UnevenSplit, KvHeads } || Allocator.Error;

pub const Rank = quant.Rank;
const Span = quant.Span;

/// The spec as a rank sees it: its heads and value heads (the vocabulary stays whole; the head's rows are split).
pub fn localSpec(s: config.Spec, r: Rank) Error!config.Spec {
    var out = s;
    out.heads = try even(s.heads, r.world, "heads");
    out.key_heads = try even(s.key_heads, r.world, "key_heads");
    out.value_heads = try even(s.value_heads, r.world, "value_heads");
    _ = try even(s.vocab, r.world, "vocab");
    out.kv_heads = @max(1, s.kv_heads / r.world);
    return out;
}

/// `_rows`: output rows `spans` of a projection, the share of a column-split one.
fn sliceRows(a: Allocator, p: host.Projection, spans: []const Span) Error!host.Projection {
    return quant.sliceRows(a, p, spans);
}

/// `_cols`: one rank's whole input groups of a row-split projection, whose fp32 outputs the ranks sum.
fn cols(a: Allocator, p: host.Projection, r: Rank, what: []const u8) Error!host.Projection {
    return quant.sliceCols(a, p, r, what);
}

/// This rank's rows of each segment of a concatenated output (q | k | v), each segment split evenly.
fn segments(widths: []const usize, r: Rank, out: []Span, what: []const u8) Error![]const Span {
    var base: usize = 0;
    for (widths, out[0..widths.len]) |w, *span| {
        const part = try even(w, r.world, what);
        span.* = .{ .from = base + r.rank * part, .to = base + (r.rank + 1) * part };
        base += w;
    }
    return out[0..widths.len];
}

/// This rank's k or v rows; with fewer KV heads than ranks each head is kept by its query ranks.
fn kvSpan(width: usize, kv_heads: usize, r: Rank, out: []Span) Error![]const Span {
    if (kv_heads >= r.world) return segments(&.{width}, r, out, "kv rows");
    if (r.world % kv_heads != 0) return error.KvHeads;
    const head = r.rank / (r.world / kv_heads);
    const part = width / kv_heads;
    out[0] = .{ .from = head * part, .to = (head + 1) * part };
    return out[0..1];
}

/// One layer's tensors for rank `r`, cut in place (`l.arena` holds the copies). `s` is the whole model's spec.
pub fn layer(l: *host.Layer, s: config.Spec, r: Rank) Error!void {
    const a = l.arena.allocator();
    var buf: [3]Span = undefined;
    switch (l.body) {
        .full => |*f| {
            f.q = try sliceRows(a, f.q, try segments(&.{s.heads * s.head_dim * 2}, r, &buf, "q_proj"));
            const kv = s.kv_heads * s.head_dim;
            f.k = try sliceRows(a, f.k, try kvSpan(kv, s.kv_heads, r, &buf));
            f.v = try sliceRows(a, f.v, try kvSpan(kv, s.kv_heads, r, &buf));
            f.o = try cols(a, f.o, r, "o_proj groups");
            try mlp(a, &f.mlp, r);
        },
        .linear => |*x| {
            const qkv = try segments(&.{ s.keyWidth(), s.keyWidth(), s.valueWidth() }, r, &buf, "in_proj_qkv");
            x.qkv = try sliceRows(a, x.qkv, qkv);
            x.conv = try takeRows(a, x.conv, qkv);
            x.z = try sliceRows(a, x.z, try segments(&.{s.valueWidth()}, r, &buf, "in_proj_z"));
            const heads = try segments(&.{s.value_heads}, r, &buf, "value heads");
            x.a = try sliceRows(a, x.a, heads);
            x.b = try sliceRows(a, x.b, heads);
            x.a_log = try takeRows(a, x.a_log, heads);
            x.dt_bias = try takeRows(a, x.dt_bias, heads);
            x.out = try cols(a, x.out, r, "out_proj groups");
            try mlp(a, &x.mlp, r);
        },
    }
}

fn mlp(a: Allocator, m: *host.Mlp, r: Rank) Error!void {
    switch (m.*) {
        .dense => |*d| {
            var buf: [1]Span = undefined;
            const spans = try segments(&.{d.gate.rows()}, r, &buf, "mlp");
            d.gate = try sliceRows(a, d.gate, spans);
            d.up = try sliceRows(a, d.up, spans);
            d.down = try cols(a, d.down, r, "mlp.down_proj groups");
        },
        .routed => |*x| try routed(a, x, r),
    }
}

/// A rank's contiguous routed experts, the shared one on rank 0, and the remap to its own ids (-1: another rank's).
fn routed(a: Allocator, x: *host.Routed, r: Rank) Error!void {
    const total = x.experts.count - 1;
    const part = try even(total, r.world, "experts");
    x.experts.fused = try quant.sliceStack(a, x.experts.fused, r, part);
    x.experts.down = try quant.sliceStack(a, x.experts.down, r, part);
    x.experts.count = part + @intFromBool(r.rank == 0);
    const remap = try a.alloc(i32, total + 1);
    @memset(remap, -1);
    for (0..part) |i| remap[r.rank * part + i] = @intCast(i);
    if (r.rank == 0) remap[total] = @intCast(part);
    x.remap = .{ .dtype = .i32, .rank = 1, .shape = .{ total + 1, 1, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(remap) };
}

/// The output head's rows for rank `r`: its share of the vocabulary.
pub fn vocabRows(a: Allocator, p: host.Projection, vocab: usize, r: Rank) Error!host.Projection {
    var buf: [1]Span = undefined;
    return sliceRows(a, p, try segments(&.{vocab}, r, &buf, "vocab"));
}

test "a kv head is kept by its query ranks" {
    var buf: [1]Span = undefined;
    const spans = try kvSpan(256, 2, .{ .rank = 3, .world = 4 }, &buf);
    try std.testing.expectEqual(Span{ .from = 128, .to = 256 }, spans[0]);
    try std.testing.expectError(error.KvHeads, kvSpan(256, 3, .{ .rank = 0, .world = 4 }, &buf));
}
