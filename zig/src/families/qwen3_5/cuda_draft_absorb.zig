//! DFlash2's absorbs of a round's kept rows; TF_Q35_DFLASH_ABSORB_BATCH=1 runs the live ones' projections once.
const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const dr = lanes.drafter;
const draft = @import("cuda_draft.zig");
const Ctx = draft.Ctx;

/// Rows one absorb pass runs (the drafter's absorb scratch).
const max_rows = engine.most_verify;

/// The switch; default off. The startup check refused every malformed value already, so a plain default stands here.
pub fn on() bool {
    return @import("sys").environment.read(bool, "TF_Q35_DFLASH_ABSORB_BATCH", false) catch false;
}

/// Rows [start, start + n) an absorb runs, the context's window opened for them (the ring holds no more behind).
pub const Span = struct { start: u64, n: usize };

pub fn open(d: anytype, ctx: *Ctx, start_: u64, n_: usize) Span {
    var start = start_;
    var n = n_;
    if (start < ctx.end) ctx.end = start;
    if (start > ctx.end) ctx.begin = start; // a gap (a prompt restored from a cache): the context starts here
    const keep = @min(d.c.window, d.e.tapRows() - engine.most_verify);
    if (n > keep) {
        start += n - keep;
        n = keep;
        ctx.begin = start;
    }
    ctx.end = start;
    return .{ .start = start, .n = n };
}

/// The context's window after its rows landed.
pub fn close(d: anytype, ctx: *Ctx) void {
    if (ctx.end > d.c.window and ctx.begin < ctx.end - d.c.window) ctx.begin = ctx.end - d.c.window;
}

/// Whether `items` absorb as one pass: the switch on, two or more streams, each once, live, device bf16, fitting.
pub fn batches(enabled: bool, items: []const dr.Absorb) bool {
    if (!enabled or items.len < 2 or items.len > max_rows) return false;
    for (items, 0..) |it, i| {
        if (it.prompt or it.features.space != .device or it.features.dtype != .bf16) return false;
        const n = if (it.rows) |rs| rs.len else it.features.rows;
        if (n > max_rows or (it.rows != null and n > engine.most_window)) return false;
        for (items[0..i]) |prior| if (prior.stream == it.stream) return false;
    }
    return true;
}

/// The round's absorbs: each stream's tree landing noted, then its kept rows into its context.
pub fn run(d: anytype, items: []const dr.Absorb) !void {
    const batched = batches(d.absorb_batch, items);
    var segs: [max_rows]Seg = undefined;
    var count: usize = 0;
    for (items) |it| {
        const ctx = d.streams.get(it.stream) orelse return error.UnknownStream;
        if (it.features.space != .device or it.features.dtype != .bf16) return error.UnsupportedFeatures;
        if (it.offered > 0 and ctx.grown > 0) {
            var prefix: [engine.most_window]u32 = undefined;
            for (prefix[0..@min(it.follow.len, prefix.len)], 0..) |*x, i| x.* = @intCast(i);
            const kept = it.rows orelse prefix[0..@min(it.follow.len, prefix.len)];
            d.landing.observeTree(it.stream.sampling != null, ctx.parents[0..ctx.grown], ctx.own[0..ctx.grown], it.offered, kept);
        }
        if (d.trace) |t| if (ctx.trace_active) {
            var prefix: [engine.most_window]u32 = undefined;
            for (prefix[0..@min(it.follow.len, prefix.len)], 0..) |*x, i| x.* = @intCast(i);
            const kept = it.rows orelse prefix[0..@min(it.follow.len, prefix.len)];
            t.line(.{
                .stream = it.stream.id,
                .position = ctx.trace_position,
                .pending = ctx.trace_pending,
                .offered = it.offered,
                .nodes = ctx.held[0..ctx.grown],
                .parents = ctx.parents[0..ctx.grown],
                .own = ctx.own[0..ctx.grown],
                .accepted_rows = kept,
                .accepted_tokens = it.follow,
                .candidates = d.traceCandidates(ctx),
            });
            ctx.trace_active = false;
        };
        ctx.grown = 0;
        // the target's features: its ring, `offset` the first row's position (cuda_native.zig's featuresFn)
        const n = if (it.rows) |rs| rs.len else it.features.rows;
        if (!batched) {
            try d.absorb(ctx, it.features.buffer, it.features.offset, n, it.rows, it.prompt);
            continue;
        }
        if (n == 0) continue;
        segs[count] = .{ .ctx = ctx, .ring = it.features.buffer, .start = it.features.offset, .n = n, .rows = it.rows };
        count += 1;
    }
    if (batched) try absorbMany(d, segs[0..count]);
}

const Seg = struct { ctx: *Ctx, ring: u64, start: u64, n: usize, rows: ?[]const u32, row0: usize = 0 };

/// Passes of at most max_rows: fc, norm and k/v once over all rows; each segment's gather, rope and ring put its own.
fn absorbMany(d: anytype, segs: []Seg) !void {
    const previous = d.projection_role;
    defer d.projection_role = previous;
    d.projection_role = .live;
    var first: usize = 0;
    var rows: usize = 0;
    for (segs, 0..) |*s, i| {
        const span = open(d, s.ctx, s.start, s.n);
        s.start = span.start;
        s.n = span.n;
        if (rows + s.n > max_rows) {
            try pass(d, segs[first..i], rows);
            first = i;
            rows = 0;
        }
        s.row0 = rows;
        rows += s.n;
    }
    if (rows > 0) try pass(d, segs[first..], rows);
    for (segs) |s| {
        s.ctx.end = s.start + s.n;
        close(d, s.ctx);
    }
}

fn pass(d: anytype, segs: []const Seg, rows: usize) !void {
    const c = d.c;
    const e = d.e;
    const tw = c.n_taps * c.hidden;
    const kw = c.kvWidth();
    for (segs) |s| {
        if (s.n == 0) continue;
        const to = d.taps_in + s.row0 * tw * 2;
        if (s.rows) |rs| {
            var tr: draft.TapRows = .{ .n = @intCast(s.n), .rows = @splat(0) };
            for (rs, 0..) |r, i| tr.rows[i] = @intCast(r);
            try e.run(d.fns.gather_rows, .{ s.n, 1, 1 }, .{ 256, 1, 1 }, .{ s.ring, @as(c_int, @intCast(e.tapRows())), @as(c_longlong, @intCast(s.start)), tr, @as(c_int, @intCast(tw)), to });
        } else try e.run(d.fns.gather_taps, .{ s.n, 1, 1 }, .{ 256, 1, 1 }, .{ s.ring, @as(c_int, @intCast(e.tapRows())), @as(c_longlong, @intCast(s.start)), @as(c_int, @intCast(tw)), to });
    }
    try d.proj(d.w.fc, d.taps_in, d.th, rows);
    try d.norm(d.th, d.w.hidden_norm, d.thn, rows);
    for (d.w.layers, 0..) |l, i| {
        if (e.affine.pdl) {
            try d.projManySums(&.{ l.k, l.v }, d.thn, &.{ d.kabs, d.vabs }, rows, false);
            for (segs) |s| try d.normRope(d.kabs + s.row0 * kw * 2, l.k_norm, d.kn + s.row0 * kw * 2, s.n, c.kv_heads, s.start);
        } else {
            try d.projAgain(l.k, d.thn, d.kabs, rows);
            for (segs) |s| try d.normRope(d.kabs + s.row0 * kw * 2, l.k_norm, d.kn + s.row0 * kw * 2, s.n, c.kv_heads, s.start);
            try d.projAgain(l.v, d.thn, d.vabs, rows);
        }
        for (segs) |s| {
            try e.run(d.fns.ring_put, .{ s.n, 1, 1 }, .{ 256, 1, 1 }, .{ d.kn + s.row0 * kw * 2, d.vabs + s.row0 * kw * 2, d.ringK(s.ctx, i), d.ringV(s.ctx, i), @as(c_int, @intCast(kw)), @as(c_longlong, @intCast(s.start)), @as(c_int, @intCast(c.ringRows())) });
        }
    }
}

test {
    _ = @import("cuda_draft_absorb_test.zig");
}
