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
        if (o.stream != hip.counting and bytes.len > 0) try hip.runtime.check(h.d.api.hipMemcpyHtoDAsync(
            @ptrFromInt(dst),
            h.staging.bytes[from..].ptr,
            bytes.len,
            o.stream,
        ));
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
    try o.l.fn_hc_gates(
        @ptrFromInt(z.ptr),
        @ptrFromInt(act.ptr),
        if (inject) |i| @ptrFromInt(i.ptr) else null,
        kind,
        @intCast(rows),
        @intCast(d.low),
        @intCast(d.streams),
        o.stream,
    );
    const g = try o.project(act, hc.up, rows, true);
    const mixed = try take(o, m.act, rows * d.hidden);
    try o.l.fn_hc_mix(
        @ptrFromInt(g.ptr),
        @ptrFromInt(x.ptr),
        @ptrFromInt(mixed.ptr),
        kind,
        @intCast(rows),
        @intCast(d.streams),
        @intCast(d.hidden),
        o.stream,
    );
    return mixed;
}

/// A block's output into the streams, then their norm: a tp rank's fp32 share is gathered and summed in the kernel.
pub fn hcBranch(o: Ops, m: *const model.Model, h: Tensor, branch: Tensor, inject: Tensor, x: Tensor, scale: u64, rows: usize) !void {
    if (branch.kind != .f32) return hcNorm(o, m, h, branch, inject, x, scale, rows);
    const d = m.dims;
    const c = m.tp.?;
    const n = rows * d.hidden;
    const parts = try o.arena.of(f32, c.world * n);
    try c.allGather(branch.ptr, parts, n, .f32, o.stream);
    try o.l.fn_hc_norm_shares(@ptrFromInt(h.ptr), @ptrFromInt(parts), @ptrFromInt(inject.ptr), @ptrFromInt(h.ptr), @ptrFromInt(x.ptr), @ptrFromInt(scale), @backingInt(m.act), @intCast(rows), @intCast(d.streams), @intCast(d.hidden), d.eps, @intCast(c.world), @intCast(n), o.stream);
}

/// h += round(branch * inject) per stream when a branch is given, then x = the streams normed with `scale`.
pub fn hcNorm(
    o: Ops,
    m: *const model.Model,
    h: Tensor,
    branch: ?Tensor,
    inject: ?Tensor,
    x: Tensor,
    scale: u64,
    rows: usize,
) !void {
    const d = m.dims;
    try o.l.fn_hc_norm(
        @ptrFromInt(h.ptr),
        if (branch) |b| @ptrFromInt(b.ptr) else null,
        if (inject) |i| @ptrFromInt(i.ptr) else null,
        if (branch != null) @ptrFromInt(h.ptr) else null,
        @ptrFromInt(x.ptr),
        @ptrFromInt(scale),
        @backingInt(m.act),
        @intCast(rows),
        @intCast(d.streams),
        @intCast(d.hidden),
        d.eps,
        o.stream,
    );
}

/// A forward's host inputs, laid out in one span of the staging (byte offsets into it) and sent up in one copy.
pub const Staged = struct {
    from: usize = 0,
    len: usize = 0,
    ids: usize = 0,
    pos: usize = 0,
    ends: usize = 0,
    complete: usize = 0,
    local: usize = 0,
    words: usize = 0,
    scales: usize = 0,
    biases: usize = 0,
    /// Blocks a window's scores walk: all its caches can hold, so the launch is the same at any position.
    bound: usize = 0,
};

/// The staged inputs on the device: token ids (one a stream), positions, ends, complete blocks, n-gram rows.
pub const Inputs = struct {
    ids: u64,
    pos: u64,
    ends: u64,
    complete: u64,
    local: u64,
    words: u64,
    scales: u64,
    biases: u64,
    bound: usize,
};

/// Every host input of a forward of `segs` written into the staging, n-gram rows hashed and gathered: no launch.
pub fn stage(m: *const model.Model, host: *Host, segs: []const Seg, window: bool) !Staged {
    const d = m.dims;
    const t = host.tables;
    if (d.ple_dim / t.dims != ngram.heads) return error.UnsupportedNgramHeads;
    var rows: usize = 0;
    for (segs) |g| rows += g.tokens.len;
    const n16 = rows * ngram.heads;
    var s: Staged = .{ .from = std.mem.alignForward(usize, host.used, 16) };
    var used: usize = 0;
    inline for (.{
        .{ "ids", rows * d.streams * 4 },
        .{ "pos", rows * 4 },
        .{ "ends", rows * 4 },
        .{ "complete", rows * 4 },
        .{ "local", n16 * 4 },
        .{ "words", n16 * t.wordBytes() },
        .{ "scales", n16 * t.tableBytes() },
        .{ "biases", n16 * t.tableBytes() },
    }) |f| {
        @field(s, f[0]) = used;
        used = std.mem.alignForward(usize, used + f[1], 16);
    }
    s.len = used;
    if (s.from + used > host.staging.bytes.len) return error.StagingTooSmall;
    const span = host.staging.bytes[s.from..][0..used];
    const ints = struct {
        fn of(b: []u8, off: usize, n: usize) []i32 {
            return @as([*]i32, @ptrCast(@alignCast(b[off..].ptr)))[0..n];
        }
    }.of;
    const ids = ints(span, s.ids, rows * d.streams);
    const pos = ints(span, s.pos, rows);
    const ends = ints(span, s.ends, rows);
    const complete = ints(span, s.complete, rows);
    for (ints(span, s.local, n16), 0..) |*x, i| x.* = @intCast(i);
    const hashed = try host.gpa.alloc([ngram.heads]u32, rows);
    defer host.gpa.free(hashed);
    var row: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        for (g.tokens, 0..) |tok, r| {
            for (0..d.streams) |k| ids[(row + r) * d.streams + k] = @intCast(tok);
            pos[row + r] = @intCast(g.pos + r);
            ends[row + r] = @intCast(g.pos + r + 1);
            complete[row + r] = @intCast((g.pos + r + 1) / d.index_ratio);
        }
        t.hasher.ids(g.c.history, g.tokens, hashed[row..][0..n]);
        row += n;
    }
    const flat: []const u32 = @as([*]const u32, @ptrCast(hashed.ptr))[0..n16];
    try t.gather(
        flat,
        span[s.words..][0 .. n16 * t.wordBytes()],
        span[s.scales..][0 .. n16 * t.tableBytes()],
        span[s.biases..][0 .. n16 * t.tableBytes()],
    );
    // a window's scores walk every block its caches hold: the launch is the same at any position
    if (window) for (segs) |g| {
        s.bound = @max(s.bound, g.c.layout.cap / d.index_ratio);
    };
    host.used = s.from + used;
    return s;
}

/// The staged span up to fresh arena memory in one copy, queued on the stream (inside a graph when one records).
pub fn send(o: Ops, host: *Host, s: Staged) !Inputs {
    const dst = try o.arena.take(s.len);
    if (o.stream != hip.counting) try hip.runtime.check(host.d.api.hipMemcpyHtoDAsync(
        @ptrFromInt(dst),
        host.staging.bytes[s.from..].ptr,
        s.len,
        o.stream,
    ));
    return .{
        .ids = dst + s.ids,
        .pos = dst + s.pos,
        .ends = dst + s.ends,
        .complete = dst + s.complete,
        .local = dst + s.local,
        .words = dst + s.words,
        .scales = dst + s.scales,
        .biases = dst + s.biases,
        .bound = s.bound,
    };
}

/// The n-gram layer over a segment's rows (from the forward's row `row0`): h += gated + silu(conv) over `cin`.
fn ple(
    o: Ops,
    m: *const model.Model,
    host: *Host,
    c: *cache.Caches,
    h: Tensor,
    call: Call,
    cin: u64,
    in: Inputs,
    row0: usize,
) !void {
    const d = m.dims;
    const rows = call.tokens.len;
    const t = host.tables;
    const mark = o.arena.mark();
    defer o.arena.release(mark);
    const emb = try take(o, m.act, rows * d.ple_dim);
    const first = row0 * ngram.heads;
    try o.l.tf_embed_rows(
        @ptrFromInt(in.words + first * t.wordBytes()),
        @ptrFromInt(in.scales + first * t.tableBytes()),
        @ptrFromInt(in.biases + first * t.tableBytes()),
        @backingInt(Kind.bf16),
        @ptrFromInt(in.local),
        @intCast(rows * ngram.heads),
        t.bits,
        @intCast(t.group),
        @intCast(t.dims),
        @ptrFromInt(emb.ptr),
        @backingInt(m.act),
        o.stream,
    );
    const kv = try o.project(emb, m.ple.kv, rows, false);
    const gated = try take(o, m.act, rows * d.wide());
    const tail = c.layout.tail_rows * d.wide() * m.act.size();
    try hip.raw.copy(.{ .r = host.d, .ptr = @ptrFromInt(cin), .len = tail }, 0, c.tail(), tail, o.stream);
    const kind = @backingInt(m.act);
    try o.l.fn_ple_gate(
        @ptrFromInt(kv.ptr),
        @ptrFromInt(h.ptr),
        @ptrFromInt(m.ple.key_scale),
        @ptrFromInt(m.ple.query_scale),
        @ptrFromInt(m.ple.conv_scale),
        @ptrFromInt(gated.ptr),
        @ptrFromInt(cin + tail),
        kind,
        @intCast(rows),
        @intCast(d.streams),
        @intCast(d.hidden),
        d.eps,
        o.stream,
    );
    try o.l.fn_ple_conv(
        @ptrFromInt(cin),
        @ptrFromInt(m.ple.conv),
        @ptrFromInt(gated.ptr),
        @ptrFromInt(h.ptr),
        @ptrFromInt(h.ptr),
        kind,
        @intCast(rows),
        @intCast(d.wide()),
        @intCast(d.ple_taps),
        @intCast(d.ngram),
        o.stream,
    );
    // a prompt's rows are kept: the tail moves on now; a window's waits for its keep
    if (!call.window) {
        try hip.raw.copy(
            .{ .r = host.d, .ptr = @ptrFromInt(c.tail()), .len = tail },
            0,
            cin + rows * d.wide() * m.act.size(),
            tail,
            o.stream,
        );
        c.history = ngram.after(c.history, call.tokens);
    }
}

/// Every segment's rows through every layer: one stream's prompt chunk, or several streams' windows together.
pub fn run(ops: Ops, m: *const model.Model, host: *Host, segs: []const Seg, window: bool, staged: Staged) !Out {
    var o = ops;
    o.prefill = !window;
    o.window = window;
    const d = m.dims;
    var rows: usize = 0;
    for (segs) |g| rows += g.tokens.len;
    const in = try send(o, host, staged);
    const dev_ids = in.ids;
    const dev_pos = in.pos;
    const h = try take(o, m.act, rows * d.wide());
    const x = try take(o, m.act, rows * d.wide());
    try o.embedRows(m.embed, dev_ids, rows * d.streams, h);
    var cins: [max_segs]u64 = undefined;
    if (segs.len > max_segs) return error.TooManyStreams;
    for (segs, cins[0..segs.len]) |g, *cin| cin.* = if (window)
        g.c.cin.base()
    else
        try o.arena.take((g.c.layout.tail_rows + g.tokens.len) * d.wide() * m.act.size());
    for (m.layers, 0..) |layer, i| {
        const mark = o.arena.mark();
        defer o.arena.release(mark);
        if (i == m.ple.layer) {
            var off: usize = 0;
            for (segs, cins[0..segs.len]) |g, cin| {
                try ple(
                    o,
                    m,
                    host,
                    g.c,
                    at(h, off * d.wide()),
                    .{ .tokens = g.tokens, .pos = g.pos, .window = window },
                    cin,
                    in,
                    off,
                );
                off += g.tokens.len;
            }
        }
        if (i == 0 or i == m.ple.layer) try hcNorm(o, m, h, null, null, x, layer.ahc.scale, rows);
        const inject = try take(o, m.act, rows * d.streams);
        const mixed = try hcPre(o, m, layer.ahc, x, rows, inject);
        const branch = switch (layer.mixer) {
            .linear => |l| try mixers.linear(o, m, l, segs, i, mixed, rows, window),
            .sparse => |s| try mixers.sparse(
                o,
                m,
                d,
                s,
                segs,
                i,
                mixed,
                rows,
                dev_pos,
                host,
                .{ .ends = in.ends, .complete = in.complete, .bound = in.bound },
            ),
        };
        try hcBranch(o, m, h, branch, inject, x, layer.mhc.scale, rows);
        const mixed2 = try hcPre(o, m, layer.mhc, x, rows, inject);
        const y = try hip.moe.run(o, m.act, layer.moe, mixed2, rows, !window);
        const next = if (i + 1 < m.layers.len) m.layers[i + 1].ahc.scale else m.mix.scale;
        try hcBranch(o, m, h, y, inject, x, next, rows);
    }
    // a window's count moves in `Engine.round`, which a replay runs too: the launches change nothing on the host
    if (!window) for (segs) |g| {
        g.c.pooled = (g.pos + g.tokens.len) / d.index_ratio;
    };
    return .{ .mixed = try hcPre(o, m, m.mix, x, rows, null), .streams = h };
}

/// Streams a forward takes at once at most.
pub const max_segs = 64;
