//! A window's rows absorbed into a stream's drafter context before its acceptance is known, then settled at it.
const std = @import("std");
const op = @import("../qwen27/dflash/operators.zig");
const cfg = @import("../qwen27/dflash/config.zig");
const model_mod = @import("metal_dflash_model.zig");
const Model = model_mod.Model;
const Context = model_mod.Context;

/// A stream's window rows: taps at positions [first, first + rows) of its own ring, its caches the pass's batch.
pub const Ahead = struct { context: *Context, first: u64, rows: u32 };

/// The most rows a stream's window absorbs ahead: what its K/V ring holds past a full context.
pub const max_rows = 16;

/// Whether `x` takes its window's rows ahead: it has absorbed every row before the window and holds none past it.
pub fn ready(c: cfg.Config, x: *const Context, first: u64, rows: u32) bool {
    return x.session.ready and x.session.context.end == first and rows > 0 and rows <= max_rows and rows <= c.window;
}

/// The context's end after its window kept `kept` drafts: the window's first row and the kept ones are its rows.
pub fn settle(c: cfg.Config, x: *Context, first: u64, kept: u32) void {
    const end = first + kept + 1;
    const next: op.Context = .{ .cache = x.session.context.cache, .begin = @max(x.session.context.begin, end -| (@as(u64, c.window) - 1)), .end = end };
    x.session.context = next;
    x.context = next;
    x.committed_end = end;
}

/// Each stream's window rows into its caches at their positions; rows it rejects lie past its end until overwritten.
pub fn absorb(m: *Model, frame: u64, aheads: []const Ahead) !void {
    const c = m.graph.config;
    const g = &m.graph;
    const ops = m.backend.ops();
    const b = &m.backend;
    if (aheads.len == 0 or aheads.len > 8 or b.batch.len != aheads.len) return error.BadDraftContext;
    var total: u32 = 0;
    for (aheads, b.batch) |x, caches| {
        if (!ready(c, x.context, x.first, x.rows) or caches.ptr != x.context.caches.ptr) return error.BadDraftContext;
        total += x.rows;
    }
    var positions: [8 * max_rows]u64 = undefined;
    const taps = try b.frame.alloc(total, c.tapWidth(), .bf16);
    const to = try b.frame.ref(taps);
    const e = try b.encoder(frame);
    const stride = @as(usize, c.tapWidth()) * 2;
    var at: u32 = 0;
    for (aheads) |x| {
        var done: u32 = 0; // the ring wraps at most once inside a window's rows
        while (done < x.rows) {
            const slot: u32 = @intCast((x.first + done) % c.window);
            const n = @min(x.rows - done, c.window - slot);
            try b.copy(e, .{ .buf = x.context.pending, .off = @as(usize, slot) * stride }, .{ .buf = to.buf, .off = to.off + @as(usize, at + done) * stride }, n, c.tapWidth());
            done += n;
        }
        for (positions[at..][0..x.rows], 0..) |*p, i| p.* = x.first + i;
        at += x.rows;
    }
    const pos = positions[0..total];
    const mode = m.preparation.mode;
    if (mode != .bf16_reference) if (b.copies) |*cp| if (cp.opt) |*o| {
        const opt_mod = @import("metal_dflash_opt.zig");
        var sp: [8]opt_mod.StreamPos = undefined;
        var row0: u32 = 0;
        for (aheads, 0..) |x, k| {
            sp[k] = .{ .first = x.first, .row0 = row0, .rows = x.rows };
            row0 += x.rows;
        }
        return o.absorbFromTaps(b, ops, g, frame, taps, sp[0..aheads.len], b.batch, total);
    };
    const fusion = try ops.vtable.project(ops.ptr, frame, taps, g.fusion, mode);
    const hidden = try ops.vtable.norm(ops.ptr, frame, fusion, .{ .weight = g.hidden_norm, .eps = c.eps, .heads = 1, .head_dim = c.hidden });
    for (g.layers, 0..) |layer, i| {
        const raw = try ops.vtable.project(ops.ptr, frame, hidden, layer.attention.k, mode);
        const normed = try ops.vtable.norm(ops.ptr, frame, raw, .{ .weight = layer.attention.k_norm, .eps = c.eps, .heads = c.kv_heads, .head_dim = c.head_dim });
        const keys = try ops.vtable.rope(ops.ptr, frame, normed, .{ .positions = pos, .heads = c.kv_heads, .dim = c.head_dim, .theta = c.theta });
        const values = try ops.vtable.project(ops.ptr, frame, hidden, layer.attention.v, mode);
        var first: u32 = 0;
        for (aheads, 0..) |x, k| {
            const kr = try b.frame.ref(keys);
            const vr = try b.frame.ref(values);
            const ks = try b.frame.add(.{ .buf = kr.buf, .off = kr.off + @as(usize, first) * keys.stride * 2 }, x.rows, keys.width, keys.stride, .bf16);
            const vs = try b.frame.add(.{ .buf = vr.buf, .off = vr.off + @as(usize, first) * values.stride * 2 }, x.rows, values.width, values.stride, .bf16);
            try ops.vtable.stream_append(ops.ptr, frame, @intCast(k), @intCast(i), ks, vs, pos[first..][0..x.rows]);
            first += x.rows;
        }
    }
}

/// The streams' window rows absorbed ahead in a frame of their own, the frame waited for (the checks' entry).
pub fn absorbNow(m: *Model, aheads: []const Ahead) !void {
    var caches: [8][]@import("metal_dflash_backend.zig").Cache = undefined;
    if (aheads.len > caches.len) return error.BadDraftContext;
    for (aheads, caches[0..aheads.len]) |x, *c| c.* = x.context.caches;
    m.backend.batch = caches[0..aheads.len];
    defer m.backend.batch = &.{};
    const ops = m.backend.ops();
    const frame = try ops.vtable.begin(ops.ptr);
    defer ops.vtable.release(ops.ptr, frame);
    try absorb(m, frame, aheads);
    m.backend.frame.finish() catch |err| {
        m.backend.failed = true;
        return err;
    };
}

test "a settled window keeps its first row and its kept drafts, the context's start sliding with its end" {
    var c: cfg.Config = .{};
    c.window = 8;
    var x: Context = .{ .pending = undefined, .caches = &.{}, .session = try @import("../qwen27/dflash/session.zig").Session.init(c, 1, 0) };
    x.session.context = .{ .cache = 1, .begin = 0, .end = 4 };
    x.session.ready = true;
    try std.testing.expect(ready(c, &x, 4, 5));
    try std.testing.expect(!ready(c, &x, 3, 5)); // a row before the window is absorbed already
    try std.testing.expect(!ready(c, &x, 6, 5)); // a row before the window is missing
    try std.testing.expect(!ready(c, &x, 4, max_rows + 1));
    settle(c, &x, 4, 2);
    try std.testing.expectEqual(op.Context{ .cache = 1, .begin = 0, .end = 7 }, x.session.context);
    try std.testing.expectEqual(@as(u64, 7), x.committed_end);
    settle(c, &x, 7, 4);
    try std.testing.expectEqual(op.Context{ .cache = 1, .begin = 5, .end = 12 }, x.session.context);
    try std.testing.expectEqualDeep(x.session.context, x.context);
}
