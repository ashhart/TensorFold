//! Lexical state in the test backend: signed hash remainders, EOS cuts, dilated tails and grouped norms.
const std = @import("std");
const m = @import("math.zig");
const State = @import("state.zig").State;
const ng = @import("hybrid_ngram");
const Tensor = m.Tensor;
fn part(o: anytype, pre: []const u8, tail: []const u8) ![]const u8 {
    return std.fmt.allocPrint(o.a, "{s}.{s}", .{ pre, tail });
}
pub fn apply(o: anytype, layer: usize, state: *State, tokens: []const u32, h: Tensor) !Tensor {
    const c = o.c;
    const pre = try o.name(layer, "ple");
    const cache = &state.cache[layer];
    var ordinal: usize = 0;
    for (c.ple[0..layer]) |yes| if (yes) {
        ordinal += 1;
    };
    const heads = (c.ngram - 1) * c.ngram_heads;
    const policy = try ng.Policy.init(o.a, c.ngram, c.ngram_heads, c.vocab, c.ple_eos, c.ngram_vocab, c.seed, ordinal);
    defer policy.deinit();
    const rows = try ng.tableRows(c.ngram_vocab, heads, ordinal, c.ngram_divisor);
    const width = c.ple_dim / heads;
    const table = try m.source(o.a, try part(o, pre, "ple_embedding.ngram_embedding.weight"), rows * width, false, false);
    const embed = try m.tensor(o.a, h.rows, c.ple_dim);
    const ids = try o.a.alloc(usize, heads);
    for (tokens, 0..) |token, r| {
        try policy.step(cache.history, token, ids);
        for (ids, 0..) |id, head|
            @memcpy(embed.row(r)[head * width ..][0..width], table[id * width ..][0..width]);
    }
    const key = try m.norm(o.a, try m.project(o.a, embed, try part(o, pre, "key_proj"), c.wide()), try part(o, pre, "norm_key"), c.hidden, c.eps);
    const value = try m.project(o.a, embed, try part(o, pre, "value_proj"), c.hidden);
    const query = try m.norm(o.a, h, try part(o, pre, "norm_query"), c.hidden, c.eps);
    const gated = try m.tensor(o.a, h.rows, c.wide());
    for (0..h.rows) |r| for (0..c.streams) |s| {
        var sum: f32 = 0;
        for (0..c.hidden) |d| sum += m.round(m.f(key.row(r)[s * c.hidden + d]) * m.f(query.row(r)[s * c.hidden + d]));
        const dot = m.round(sum / @sqrt(@as(f32, @floatFromInt(c.hidden))));
        const root = if (dot == 0) @as(f32, 0) else std.math.copysign(@sqrt(@max(@abs(dot), 1e-6)), dot);
        const gate = m.round(m.sig(m.round(root)));
        for (0..c.hidden) |d| gated.row(r)[s * c.hidden + d] = m.bf(gate * m.f(value.row(r)[d]));
    };
    const normed = try m.norm(o.a, gated, try part(o, pre, "norm_conv"), c.hidden, c.eps);
    const conv = try m.source(o.a, try part(o, pre, "conv1d.weight"), c.wide() * c.ple_conv, false, false);
    const out = try m.tensor(o.a, h.rows, c.wide());
    for (0..h.rows) |r| {
        for (0..c.wide()) |d| {
            var sum: f32 = 0;
            for (0..c.ple_conv) |tap| {
                const from = tap * c.ngram;
                const v = if (from == (c.ple_conv - 1) * c.ngram) normed.row(r)[d] else cache.ple_conv[from * c.wide() + d];
                sum += m.f(v) * m.f(conv[d * c.ple_conv + tap]);
            }
            const branch = m.round(m.silu(m.round(sum)));
            out.row(r)[d] = m.bf(m.f(h.row(r)[d]) + m.round(m.f(gated.row(r)[d]) + branch));
        }
        std.mem.copyForwards(u16, cache.ple_conv[0 .. cache.ple_conv.len - c.wide()], cache.ple_conv[c.wide()..]);
        @memcpy(cache.ple_conv[cache.ple_conv.len - c.wide() ..], normed.row(r));
    }
    return out;
}
