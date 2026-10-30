//! The numeric instrument consumes the production lexical projection pack and snapshots every actual native row.
const std = @import("std");
const n = @import("numeric_gpu.zig");
const m = @import("math.zig");
pub fn apply(g: *n.G, w: n.Model.Ple, state: anytype, layer: usize, projections: n.Tensor, h: n.Tensor, gated: n.Tensor, normed: n.Tensor) !void {
    const c = g.c;
    const range = state.layout.layers[layer].ple;
    const tail = n.view(u16, try state.current(range));
    for (0..h.rows) |r| {
        const p = n.row(projections, r);
        const key = try n.norm(g, w.key_norm, p[0..c.wide()], c.hidden, c.eps);
        const query = try n.norm(g, w.query_norm, n.row(h, r), c.hidden, c.eps);
        const value = p[c.wide()..];
        for (0..c.streams) |s| {
            var sum: f32 = 0;
            for (0..c.hidden) |d| sum += m.round(m.f(key[s * c.hidden + d]) * m.f(query[s * c.hidden + d]));
            const dot = m.round(m.round(sum) / @sqrt(@as(f32, @floatFromInt(c.hidden))));
            const root = if (dot == 0) @as(f32, 0) else std.math.copysign(@sqrt(@max(@abs(dot), 1e-6)), dot);
            const gate = m.round(m.sig(m.round(root)));
            for (0..c.hidden) |d| n.row(gated, r)[s * c.hidden + d] = m.bf(gate * m.f(value[d]));
        }
        @memcpy(n.row(normed, r), try n.norm(g, w.conv_norm, n.row(gated, r), c.hidden, c.eps));
        const cw = n.vector(w.conv);
        for (n.row(h, r), 0..) |*v, d| {
            var sum: f32 = 0;
            for (0..c.ple_conv) |tap| {
                const from = tap * c.ngram;
                const x = if (from == (c.ple_conv - 1) * c.ngram) n.row(normed, r)[d] else tail[from * c.wide() + d];
                sum += m.f(x) * cw[d * c.ple_conv + tap];
            }
            const branch = m.round(m.silu(m.round(sum)));
            v.* = m.bf(m.f(v.*) + m.round(m.f(n.row(gated, r)[d]) + branch));
        }
        std.mem.copyForwards(u16, tail[0 .. tail.len - c.wide()], tail[c.wide()..]);
        @memcpy(tail[tail.len - c.wide() ..], n.row(normed, r));
        @memcpy(n.view(u16, try state.row(r, range)), tail);
    }
}
