//! Flash Next's forward on a HIP rank: a stream's rows through every layer, a window's each with a serial step's bits.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const tables = @import("tables.zig");
const mixers = @import("mixers.zig");
const ngram = @import("ngram.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const Kind = hip.ops.Kind;

/// What a forward reads from the host: the pinned staging its uploads go through, and the n-gram tables.
pub const Host = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    staging: hip.HostBuffer,
    /// Bytes of `staging` this forward used: the stream is synchronized before a forward starts again.
    used: usize = 0,
    tables: *const tables.Tables,

    /// `bytes` copied through the staging into fresh arena memory, queued on the stream.
    pub fn put(h: *Host, o: Ops, bytes: []const u8) !u64 {
        const from = std.mem.alignForward(usize, h.used, 16);
        if (from + bytes.len > h.staging.bytes.len) return error.StagingTooSmall;
        const dst = try o.arena.take(bytes.len);
        @memcpy(h.staging.bytes[from..][0..bytes.len], bytes);
        h.used = from + bytes.len;
        if (o.stream != hip.counting and bytes.len > 0) try hip.runtime.check(h.d.api.hipMemcpyHtoDAsync(@ptrFromInt(dst), h.staging.bytes[from..].ptr, bytes.len, o.stream));
        return dst;
    }
};

/// A device buffer of `n` values of `kind` from the arena.
pub fn take(o: Ops, kind: Kind, n: usize) !Tensor {
    return .{ .ptr = try o.arena.take(n * kind.size()), .kind = kind };
}

/// `t` advanced by `n` values.
pub fn at(t: Tensor, n: usize) Tensor {
    return .{ .ptr = t.ptr + n * t.kind.size(), .kind = t.kind };
}

/// One call's rows: their tokens, the position of the first, and whether they are a window (else a prompt's).
pub const Call = struct { tokens: []const u32, pos: usize, window: bool };

/// One stream's rows in a forward: its caches, its tokens and the position of its first row.
pub const Seg = struct { c: *cache.Caches, tokens: []const u32, pos: usize };

/// The final mixer's rows (rows, hidden) for the head, and the residual streams (rows, wide) before it.
pub const Out = struct { mixed: Tensor, streams: Tensor };

/// A hyper-connection's read of the normed streams `x`: the block input, and the inject gates when `inject` is set.
pub fn hcPre(o: Ops, m: *const model.Model, hc: model.Hc, x: Tensor, rows: usize, inject: ?Tensor) !Tensor {
    const d = m.dims;
    const z = try o.project(x, hc.down, rows, true);
    const act = try take(o, m.act, rows * d.low);
    const kind = @backingInt(m.act);
    try o.l.fn_hc_gates(@ptrFromInt(z.ptr), @ptrFromInt(act.ptr), if (inject) |i| @ptrFromInt(i.ptr) else null, kind, @intCast(rows), @intCast(d.low), @intCast(d.streams), o.stream);
    const g = try o.project(act, hc.up, rows, true);
    const mixed = try take(o, m.act, rows * d.hidden);
    try o.l.fn_hc_mix(@ptrFromInt(g.ptr), @ptrFromInt(x.ptr), @ptrFromInt(mixed.ptr), kind, @intCast(rows), @intCast(d.streams), @intCast(d.hidden), o.stream);
    return mixed;
}

/// h += round(branch * inject) per stream when a branch is given, then x = the streams normed with `scale`.
pub fn hcNorm(o: Ops, m: *const model.Model, h: Tensor, branch: ?Tensor, inject: ?Tensor, x: Tensor, scale: u64, rows: usize) !void {
    const d = m.dims;
    try o.l.fn_hc_norm(@ptrFromInt(h.ptr), if (branch) |b| @ptrFromInt(b.ptr) else null, if (inject) |i| @ptrFromInt(i.ptr) else null, if (branch != null) @ptrFromInt(h.ptr) else null, @ptrFromInt(x.ptr), @ptrFromInt(scale), @backingInt(m.act), @intCast(rows), @intCast(d.streams), @intCast(d.hidden), d.eps, o.stream);
}

/// The n-gram layer over `rows` rows: h += gated + silu(conv); `cin` is the conv input (the tail, then the rows).
fn ple(o: Ops, m: *const model.Model, host: *Host, c: *cache.Caches, h: Tensor, call: Call, cin: u64) !void {
    const d = m.dims;
    const rows = call.tokens.len;
    const t = host.tables;
    const heads = d.ple_dim / t.dims;
    if (heads != ngram.heads) return error.UnsupportedNgramHeads;
    var buf: [64][ngram.heads]u32 = undefined;
    const mark = o.arena.mark();
    defer o.arena.release(mark);
    const emb = try take(o, m.act, rows * d.ple_dim);
    var row: usize = 0;
    var history = c.history;
    while (row < rows) : (row += buf.len) {
        const n = @min(buf.len, rows - row);
        t.hasher.ids(history, call.tokens[row..][0..n], buf[0..n]);
        history = ngram.after(history, call.tokens[row..][0..n]);
        const flat: []const u32 = @as([*]const u32, @ptrCast(&buf))[0 .. n * heads];
        const count = n * heads;
        const wb = t.wordBytes();
        const tb = t.tableBytes();
        const words = try o.arena.take(count * wb);
        const scales = try o.arena.take(count * tb);
        const biases = try o.arena.take(count * tb);
        const stage = host.staging.bytes;
        const at0 = std.mem.alignForward(usize, host.used, 16);
        if (at0 + count * (wb + 2 * tb) > stage.len) return error.StagingTooSmall;
        try t.gather(flat, stage[at0..][0 .. count * wb], stage[at0 + count * wb ..][0 .. count * tb], stage[at0 + count * (wb + tb) ..][0 .. count * tb]);
        host.used = at0 + count * (wb + 2 * tb);
        if (o.stream != hip.counting) {
            try hip.runtime.check(host.d.api.hipMemcpyHtoDAsync(@ptrFromInt(words), stage[at0..].ptr, count * wb, o.stream));
            try hip.runtime.check(host.d.api.hipMemcpyHtoDAsync(@ptrFromInt(scales), stage[at0 + count * wb ..].ptr, count * tb, o.stream));
            try hip.runtime.check(host.d.api.hipMemcpyHtoDAsync(@ptrFromInt(biases), stage[at0 + count * (wb + tb) ..].ptr, count * tb, o.stream));
        }
        var iota: [64 * 16]i32 = undefined;
        for (iota[0..count], 0..) |*x, i| x.* = @intCast(i);
        const local = try host.put(o, std.mem.sliceAsBytes(iota[0..count]));
        try o.l.tf_embed_rows(@ptrFromInt(words), @ptrFromInt(scales), @ptrFromInt(biases), @backingInt(Kind.bf16), @ptrFromInt(local), @intCast(count), t.bits, @intCast(t.group), @intCast(t.dims), @ptrFromInt(at(emb, row * d.ple_dim).ptr), @backingInt(m.act), o.stream);
    }
    const kv = try o.project(emb, m.ple.kv, rows, false);
    const gated = try take(o, m.act, rows * d.wide());
    const tail = c.layout.tail_rows * d.wide() * m.act.size();
    try hip.raw.copy(.{ .r = host.d, .ptr = @ptrFromInt(cin), .len = tail }, 0, c.tail(), tail, o.stream);
    const kind = @backingInt(m.act);
    try o.l.fn_ple_gate(@ptrFromInt(kv.ptr), @ptrFromInt(h.ptr), @ptrFromInt(m.ple.key_scale), @ptrFromInt(m.ple.query_scale), @ptrFromInt(m.ple.conv_scale), @ptrFromInt(gated.ptr), @ptrFromInt(cin + tail), kind, @intCast(rows), @intCast(d.streams), @intCast(d.hidden), d.eps, o.stream);
    try o.l.fn_ple_conv(@ptrFromInt(cin), @ptrFromInt(m.ple.conv), @ptrFromInt(gated.ptr), @ptrFromInt(h.ptr), @ptrFromInt(h.ptr), kind, @intCast(rows), @intCast(d.wide()), @intCast(d.ple_taps), @intCast(d.ngram), o.stream);
    // a prompt's rows are kept: the tail moves on now; a window's waits for its keep
    if (!call.window) {
        try hip.raw.copy(.{ .r = host.d, .ptr = @ptrFromInt(c.tail()), .len = tail }, 0, cin + rows * d.wide() * m.act.size(), tail, o.stream);
        c.history = history;
    }
}

/// Every segment's rows through every layer: one stream's prompt chunk, or several streams' windows together.
pub fn run(ops: Ops, m: *const model.Model, host: *Host, segs: []const Seg, window: bool) !Out {
    var o = ops;
    o.prefill = !window;
    o.window = window;
    const d = m.dims;
    var rows: usize = 0;
    for (segs) |g| rows += g.tokens.len;
    const ids4 = try host.gpa.alloc(i32, rows * d.streams);
    defer host.gpa.free(ids4);
    const pos = try host.gpa.alloc(i32, rows);
    defer host.gpa.free(pos);
    var row: usize = 0;
    for (segs) |g| {
        for (g.tokens, 0..) |t, r| {
            for (0..d.streams) |s| ids4[(row + r) * d.streams + s] = @intCast(t);
            pos[row + r] = @intCast(g.pos + r);
        }
        row += g.tokens.len;
    }
    const dev_ids = try host.put(o, std.mem.sliceAsBytes(ids4));
    const dev_pos = try host.put(o, std.mem.sliceAsBytes(pos));
    const h = try take(o, m.act, rows * d.wide());
    const x = try take(o, m.act, rows * d.wide());
    try o.embedRows(m.embed, dev_ids, rows * d.streams, h);
    var cins: [max_segs]u64 = undefined;
    if (segs.len > max_segs) return error.TooManyStreams;
    for (segs, cins[0..segs.len]) |g, *cin| cin.* = if (window) g.c.cin.base() else try o.arena.take((g.c.layout.tail_rows + g.tokens.len) * d.wide() * m.act.size());
    for (m.layers, 0..) |layer, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        if (i == m.ple.layer) {
            var off: usize = 0;
            for (segs, cins[0..segs.len]) |g, cin| {
                try ple(o, m, host, g.c, at(h, off * d.wide()), .{ .tokens = g.tokens, .pos = g.pos, .window = window }, cin);
                off += g.tokens.len;
            }
        }
        if (i == 0 or i == m.ple.layer) try hcNorm(o, m, h, null, null, x, layer.ahc.scale, rows);
        const inject = try take(o, m.act, rows * d.streams);
        const mixed = try hcPre(o, m, layer.ahc, x, rows, inject);
        const branch = switch (layer.mixer) {
            .linear => |l| try mixers.linear(o, m, l, segs, i, mixed, rows, window),
            .sparse => |s| try mixers.sparse(o, m, d, s, segs, i, mixed, rows, dev_pos, host),
        };
        try hcNorm(o, m, h, branch, inject, x, layer.mhc.scale, rows);
        const mixed2 = try hcPre(o, m, layer.mhc, x, rows, inject);
        const y = try hip.moe.run(o, m.act, layer.moe, mixed2, rows, !window);
        const next = if (i + 1 < m.layers.len) m.layers[i + 1].ahc.scale else m.mix.scale;
        try hcNorm(o, m, h, y, inject, x, next, rows);
    }
    for (segs) |g| g.c.pooled = (g.pos + g.tokens.len) / d.index_ratio;
    return .{ .mixed = try hcPre(o, m, m.mix, x, rows, null), .streams = h };
}

/// Streams a forward takes at once at most.
pub const max_segs = 64;
