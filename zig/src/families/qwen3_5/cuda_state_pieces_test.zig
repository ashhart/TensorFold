//! The 27B and 35B-A3B lane maps at 256-row pages: pieces in pages, copy on rows 1.., counts, wide rows refused.
const std = @import("std");
const kv = @import("core").kv_pool;
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const sp = @import("cuda_state_pieces.zig");
const States = @import("cuda_states.zig").States;
const Overlay = @import("core").recurrent_workspace.Overlay;
const t = std.testing;
const gpa = t.allocator;

/// The dense 27B's and the 35B-A3B's linear widths: every fourth layer attention.
fn shape(layers: usize, v_heads: usize) geometry.Geometry {
    var linear: [geometry.max_layers]bool = @splat(false);
    for (linear[0..layers], 0..) |*l, i| l.* = i % 4 != 3;
    return .{ .hidden = 256, .intermediate = 256, .layers = layers, .vocab = 8, .k_heads = 16, .v_heads = v_heads, .q_heads = 16, .kv_heads = 2, .eps = 1e-6, .tied = true, .context = 64, .linear = linear };
}

const Span = struct { lo: u64, hi: u64 };

fn apart(a: Span, b: Span) bool {
    return a.hi <= b.lo or b.hi <= a.lo;
}

/// The table of a lane on pages in reverse order (no layout of consecutive pages matches) at a pool base.
fn table(m: sp.Map, g: geometry.Geometry, page: u64, host: []u64) !void {
    const ids = try gpa.alloc(u32, m.own.pages);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(3 * (ids.len - i));
    m.fill(host, g, 0x7f00_0000_0000, ids);
    _ = page;
}

fn onePage(at: u64, len: u64, page: u64) bool {
    const off = (at - 0x7f00_0000_0000) % page;
    return off + len <= page;
}

test "Qwen lane pieces: every piece inside one page, the copy beside no state and no row 0, rows in order" {
    for ([_][2]usize{ .{ 64, 48 }, .{ 40, 32 } }, [_]u64{ 16 << 20, 5 << 20 }) |dims, page| {
        const g = shape(dims[0], dims[1]);
        const nl = g.deltas();
        var m = try sp.plan(gpa, g, page);
        defer m.deinit(gpa);
        try kv.pieces.check(gpa, m.own);
        const host = try gpa.alloc(u64, States.len(g));
        defer gpa.free(host);
        try table(m, g, page, host);
        const row = nl * g.logRowBytes();
        var state: [2 * geometry.max_layers]Span = undefined;
        for (0..nl) |l| {
            state[l] = .{ .lo = host[l], .hi = host[l] + g.recBytes() };
            state[nl + l] = .{ .lo = host[nl + l], .hi = host[nl + l] + g.convBytes() };
        }
        const row0: Span = .{ .lo = host[4 * nl], .hi = host[4 * nl] + row };
        for (0..2 * nl) |i| {
            const len = if (i < nl) g.recBytes() else g.convBytes();
            const c: Span = .{ .lo = host[2 * nl + i], .hi = host[2 * nl + i] + len };
            try t.expect(onePage(c.lo, len, page));
            try t.expect(apart(c, row0));
            for (state[0 .. 2 * nl]) |s| try t.expect(apart(c, s));
            for (0..2 * nl) |j| if (j != i) {
                const lj = if (j < nl) g.recBytes() else g.convBytes();
                try t.expect(apart(c, .{ .lo = host[2 * nl + j], .hi = host[2 * nl + j] + lj }));
            };
        }
        for (0..engine.most_window) |r| {
            const lo = host[4 * nl + r];
            try t.expect(onePage(lo, row, page));
            for (state[0 .. 2 * nl]) |s| try t.expect(apart(.{ .lo = lo, .hi = lo + row }, s));
            if (r > 0) try t.expect(apart(.{ .lo = lo, .hi = lo + row }, row0));
            // a group's rows follow each other, as one block's do (cuda_cache.zig pendRows copies them as one run)
            if (r > 1 and (r - 1) % m.group_rows != 0) try t.expectEqual(host[4 * nl + r - 1] + row, lo);
        }
    }
}

test "Qwen lane pieces at 256-row pages: pages a lane against today's run (the run as runBytes took it)" {
    // today's run: state, then the Overlay of log and copy (cuda_cache.zig runBytes without taps or ring)
    for ([_][2]usize{ .{ 64, 48 }, .{ 40, 32 } }, [_]u64{ 16 << 20, 5 << 20 }, [_]u32{ 26, 38 }) |dims, page, run| {
        const g = shape(dims[0], dims[1]);
        const over = Overlay.init(g.stateBytes(), g.logBytes(1), engine.most_window);
        const bytes = std.mem.alignForward(u64, g.stateBytes(), 256) + std.mem.alignForward(u64, over.bytes, 256);
        try t.expectEqual(run, @as(u32, @intCast((bytes + page - 1) / page)));
        var m = try sp.plan(gpa, g, page);
        defer m.deinit(gpa);
        try t.expect(m.copy != null); // the copy fits in the log rows 1.. at both sizes
        std.debug.print("qwen lane pieces {d} layers: {d} pieces on {d} pages (run {d}), {d} rows a group\n", .{ dims[0], m.summary().n, m.own.pages, run, m.group_rows });
    }
}

test "Qwen lane pieces: a log row wider than a page is refused by name; small pages give the copy its own pieces" {
    const g = shape(64, 48);
    try t.expectError(error.QwenLogRowOverPage, sp.plan(gpa, g, g.deltas() * g.logRowBytes() - 1));
    const g35 = shape(40, 32);
    // one row a page and a recurrence wider than a row: the copy cannot lie in the rows
    var m = try sp.plan(gpa, g35, 5 << 19);
    defer m.deinit(gpa);
    try t.expect(m.copy == null);
    try t.expectEqual(@as(usize, 1), m.group_rows);
    try kv.pieces.check(gpa, m.own);
}
