//! What some rows of a Metal backbone pass take beside its layers: adapter shifts, steering deltas, masked heads.
const std = @import("std");
const cfg = @import("../config.zig");
const k = @import("kernels.zig");
const ek = @import("encoder_kernels.zig");
const adp = @import("adapter.zig");
const engine = @import("engine.zig");
const Engine = engine.Engine;
const Lane = engine.Lane;
const Turn = engine.Turn;
const Steer = engine.Steer;
const Probed = engine.Probed;

const D = cfg.dim;
const W = cfg.frame_width;

/// The span-end logits of row `row` of the last pass, whose head product is row `at` of `head_lo`.
pub fn endLogits(e: *Engine, row: usize, at: usize, out: *[cfg.codebooks]f32) !void {
    var xn: [D]f32 = undefined;
    var lo: [adp.max_rank]f32 = undefined;
    try e.g.get(e.xn + row * D * 4, std.mem.asBytes(&xn));
    try e.g.get(e.head_lo + at * adp.max_rank * 4, std.mem.asBytes(&lo));
    e.adapter.?.endLogits(&xn, &lo, out);
}

/// y [rows, n] = x W^T and, with factors `f`, the adapter's shift: (x down^T) up^T added in fp32 (accents.linear).
pub fn lin(e: *Engine, f: ?adp.Pair, lo: u64, w: u64, x: u64, kdim: usize, n: usize, y: u64, rows: usize, o: k.Linear) void {
    const g = e.g;
    const p = f orelse return k.linear(g, w, x, kdim, kdim, n, y, n, rows, o);
    const picked = o.picks != 0;
    k.linear(g, w, x, kdim, kdim, n, e.pair, 2 * n, rows, o);
    k.linear(g, p.down, x, kdim, kdim, p.rank, lo, adp.max_rank, rows, .{ .picks = o.picks, .pick_stride = if (picked) p.rank * kdim else 0, .xrep = o.xrep });
    k.linear(g, p.up, lo, adp.max_rank, p.rank, n, e.pair + n * 4, 2 * n, rows, .{ .picks = o.picks, .pick_stride = if (picked) n * p.rank else 0 });
    k.combine(g, e.pair, e.ones, n, y, rows);
}

/// A row with a span-end code, summed on the host as z3_embed sums: ten audio rows in order, then the text row.
pub fn hostRow(e: *Engine, r: [W]i32, out: *[D]f32) !void {
    var raw: [D * 2]u8 = undefined;
    @memset(out, 0);
    for (r, 0..) |id, col| {
        if (col < cfg.codebooks and id == cfg.span_end_id) {
            for (out, &e.adapter.?.end_rows[col]) |*o, x| o.* += x;
            continue;
        }
        const at = if (col < cfg.codebooks) col * cfg.audio_vocab + @as(usize, @intCast(id)) else cfg.codebooks * cfg.audio_vocab + @as(usize, @intCast(id));
        try e.g.get(e.w.tables + at * D * 2, &raw);
        for (out, 0..) |*o, i| o.* += @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16);
    }
}

/// ACI (lm._masked_heads): turn `b`'s head again over the keys it may see, closed up into one cache.
pub fn maskedHead(e: *Engine, l: *const Lane, li: usize, b: usize, m: @import("../laci_probe.zig").Mask, p: *const Probed) !void {
    const g = e.g;
    const row = cfg.kv_width * 4;
    const after = p.text_start + p.text_len;
    if (m.hi >= p.text_len or m.lo > m.hi or after > l.position) return error.BadProbeWindow;
    const window = m.hi - m.lo + 1;
    const tail = l.position + 1 - after; // the row's own key is in the cache already
    for (e.mask_kv, l.kv[li]) |dst, src| {
        if (p.text_start > 0) g.copy(dst, src, p.text_start * row);
        g.copy(dst + p.text_start * row, src + (p.text_start + m.lo) * row, window * row);
        g.copy(dst + (p.text_start + window) * row, src + after * row, tail * row);
    }
    const qb = e.q + b * cfg.attn_inner * 4;
    k.attention(g, qb, e.mask_kv[0], e.mask_kv[1], e.gate + b * cfg.heads * 4, p.text_start + window + tail - 1, e.mask_out, 1);
    const head = m.head * cfg.head_dim * 4;
    g.copy(e.y + b * cfg.attn_inner * 4 + head, e.mask_out + head, cfg.head_dim * 4);
}

/// A pass's steering deltas on the device before it runs: turn `b`'s j-th at staged row b * sources + j.
pub fn stageSteers(e: *Engine, many: ?[]const Turn, lone: []const Steer) !void {
    const n = if (many) |turns| turns.len else 1;
    for (0..n) |b| {
        const list = if (many) |turns| turns[b].steer else lone;
        if (list.len > cfg.sources) return error.TooManySteers;
        for (list, 0..) |st, j| try e.g.put(e.steer + (b * cfg.sources + j) * D * 4, std.mem.asBytes(st.delta));
    }
}

/// Bank source `source` of this pass takes its rows' deltas: source + delta, one fp32 add (lm.forward's source()).
pub fn steerSource(e: *Engine, source: usize, many: ?[]const Turn, lone: []const Steer) void {
    const n = if (many) |turns| turns.len else 1;
    for (0..n) |b| for (if (many) |turns| turns[b].steer else lone, 0..) |st, j| {
        if (st.source != source) continue;
        ek.add(e.g, e.slot(source) + b * D * 4, e.steer + (b * cfg.sources + j) * D * 4, D);
    };
}
