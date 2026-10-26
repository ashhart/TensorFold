//! Contiguous attention has its own entry point and graph key; mixed batches retain per-block page selection.
const std = @import("std");
const Round = @import("cuda_round.zig");
const geometry = @import("cuda_geometry.zig");
const sig = @import("cuda_signatures.zig");

pub fn hasPages(segs: anytype) bool {
    for (segs) |sg| if (sg.pt != 0) return true;
    return false;
}

pub fn partial(e: anytype, shape: anytype, layer: c_int) !void {
    return partialRows(e, shape, layer, 0, shape.total);
}

/// `partial` for the `rows` rows from `first`: the partials hold most_verify rows, a paged prompt chunk up to max_rows.
pub fn partialRows(e: anytype, shape: anytype, layer: c_int, first: usize, rows: usize) !void {
    const f = if (e.kv8) (if (shape.paged) e.fp8.attn_partial_paged else e.fp8.attn_partial_contiguous) else (if (shape.paged) e.fns.attn_partial_paged else e.fns.attn_partial_contiguous);
    const t = e.tables.ptr;
    const q = e.qn + first * e.g.q_heads * geometry.head_dim * 2;
    try e.run(f, .{ 64, e.g.kv_heads, rows }, .{ 32 * (e.g.q_heads / e.g.kv_heads), 1, 1 }, .{ e.geo, q, t + Round.segs_at, t + Round.rowseg_at + first * 4, t + Round.rowanc_at + first * @sizeOf(Round.Ancestor), layer, t + Round.params_at, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
}

/// A paged chunk's partial and merge, `slice` rows a launch, each on its rows' queries, tables and outputs.
pub fn pagedSlices(e: anytype, shape: anytype, layer: c_int, slice: usize) !void {
    const geo = e.geo;
    const t = e.tables.ptr;
    var r: usize = 0;
    while (r < shape.total) : (r += slice) {
        const n = @min(slice, shape.total - r);
        try partialRows(e, shape, layer, r, n);
        const y = e.y + r * e.g.q_heads * geometry.head_dim * 2;
        const rows = t + Round.rowseg_at + r * 4;
        const anc = t + Round.rowanc_at + r * @sizeOf(Round.Ancestor);
        try launch(e, e.fns.attn_merge, sig.attn_merge, .{ e.g.q_heads, n, 1 }, .{ 32, 1, 1 }, .{ geo, e.pm, e.pl, e.po, t + Round.segs_at, rows, anc, t + Round.params_at, y });
    }
}

fn launch(e: anytype, f: anytype, comptime s: sig.Sig, grid: [3]usize, block: [3]usize, args: anytype) !void {
    sig.check(s, @TypeOf(args));
    return e.run(f, grid, block, args);
}

test "Qwen attention path: unpaged, paged and mixed segment tables" {
    const Seg = struct { pt: u64 };
    try std.testing.expect(!hasPages(@as([]const Seg, &.{})));
    try std.testing.expect(!hasPages(@as([]const Seg, &.{ .{ .pt = 0 }, .{ .pt = 0 } })));
    try std.testing.expect(hasPages(@as([]const Seg, &.{ .{ .pt = 0x4000 }, .{ .pt = 0 } })));
    try std.testing.expect(hasPages(@as([]const Seg, &.{ .{ .pt = 0 }, .{ .pt = 0x4000 } })));
}

test "Qwen attention path: layout is part of every captured shape key" {
    const Shape = @import("cuda_shapes.zig").Shape;
    for (1..129) |rows| for (1..rows + 1) |segments| for ([_]u8{ 1, 2, 4, 8, 16 }) |slots| {
        var plain = Shape.init(rows, segments, true, false, slots);
        for ([_]bool{ false, true }) |hidden| {
            plain.hidden = hidden;
            var pages = plain;
            pages.paged = true;
            try std.testing.expectEqual(plain.key() | (1 << 28), pages.key());
            try std.testing.expect(plain.key() != pages.key());
        }
    };
}

const Fake = struct {
    const Fns = struct { attn_partial_paged: usize, attn_partial_contiguous: usize };
    kv8: bool,
    fns: Fns = .{ .attn_partial_paged = 1, .attn_partial_contiguous = 2 },
    fp8: Fns = .{ .attn_partial_paged = 3, .attn_partial_contiguous = 4 },
    g: struct { q_heads: usize = 24, kv_heads: usize = 4 } = .{},
    tables: struct { ptr: u64 = 0x10000 } = .{},
    geo: u32 = 24,
    qn: u64 = 0x20000,
    pm: u64 = 0x30000,
    pl: u64 = 0x40000,
    po: u64 = 0x50000,
    calls: usize = 0,
    got: usize = 0,
    rows: usize = 0,
    pub fn run(e: *@This(), f: usize, grid: [3]usize, threads: [3]usize, args: anytype) !void {
        e.calls += 1;
        e.got = f;
        try std.testing.expectEqualSlices(usize, &.{ 64, 4, e.rows }, &grid);
        try std.testing.expectEqualSlices(usize, &.{ 192, 1, 1 }, &threads);
        try std.testing.expectEqualDeep(.{ e.geo, e.qn, e.tables.ptr + Round.segs_at, e.tables.ptr + Round.rowseg_at, e.tables.ptr + Round.rowanc_at, @as(c_int, 2), e.tables.ptr + Round.params_at, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po }, args);
    }
};

test "Qwen attention path: production launch selects one partial, no per-chunk launch or synchronization" {
    for ([_]bool{ false, true }) |fp8| for ([_]bool{ false, true }) |paged| for ([_]usize{ 1, 16, 17, 64, 128 }) |rows| {
        var e = Fake{ .kv8 = fp8, .rows = rows };
        try partial(&e, .{ .paged = paged, .total = rows }, 2);
        try std.testing.expectEqual(@as(usize, 1), e.calls);
        try std.testing.expectEqual(@as(usize, if (fp8) (if (paged) 3 else 4) else (if (paged) 1 else 2)), e.got);
    };
}

/// Every partial and merge launch with the row offsets it was given.
const Recorder = struct {
    const Geo = extern struct { heads: u32 };
    const Launch = struct { f: usize, grid: [3]usize, q: u64 = 0, rows: u64, anc: u64, y: u64 = 0 };
    kv8: bool = false,
    fns: struct { attn_partial_paged: usize = 1, attn_partial_contiguous: usize = 2, attn_merge: usize = 5 } = .{},
    fp8: struct { attn_partial_paged: usize = 3, attn_partial_contiguous: usize = 4 } = .{},
    g: struct { q_heads: usize = 24, kv_heads: usize = 4 } = .{},
    tables: struct { ptr: u64 = 0x10000 } = .{},
    geo: Geo = .{ .heads = 24 },
    qn: u64 = 0x20000,
    pm: u64 = 0x30000,
    pl: u64 = 0x40000,
    po: u64 = 0x50000,
    y: u64 = 0x60000,
    got: [64]Launch = undefined,
    n: usize = 0,
    pub fn run(e: *@This(), f: usize, grid: [3]usize, threads: [3]usize, args: anytype) !void {
        _ = threads;
        // the merge takes 9 arguments, the partial 11
        e.got[e.n] = if (comptime args.len == 9)
            .{ .f = f, .grid = grid, .rows = args[5], .anc = args[6], .y = args[8] }
        else
            .{ .f = f, .grid = grid, .q = args[1], .rows = args[3], .anc = args[4] };
        e.n += 1;
    }
};

test "Qwen attention path: a 257-row paged chunk issues three slices, each shifted to its first row" {
    const row = 24 * geometry.head_dim * 2;
    for ([_]bool{ false, true }) |fp8| {
        var e = Recorder{ .kv8 = fp8 };
        try pagedSlices(&e, .{ .paged = true, .total = @as(usize, 257) }, 2, 128);
        try std.testing.expectEqual(@as(usize, 6), e.n);
        for ([_]usize{ 128, 128, 1 }, 0..) |rows, s| {
            const first = s * 128;
            const p = e.got[2 * s];
            const m = e.got[2 * s + 1];
            try std.testing.expectEqual(@as(usize, if (fp8) 3 else 1), p.f);
            try std.testing.expectEqualSlices(usize, &.{ 64, 4, rows }, &p.grid);
            try std.testing.expectEqual(e.qn + first * row, p.q);
            try std.testing.expectEqual(@as(usize, 5), m.f);
            try std.testing.expectEqualSlices(usize, &.{ 24, rows, 1 }, &m.grid);
            try std.testing.expectEqual(e.y + first * row, m.y);
            for ([_]Recorder.Launch{ p, m }) |l| {
                try std.testing.expectEqual(e.tables.ptr + Round.rowseg_at + first * 4, l.rows);
                try std.testing.expectEqual(e.tables.ptr + Round.rowanc_at + first * @sizeOf(Round.Ancestor), l.anc);
            }
        }
    }
    // around the boundary: one launch pair per started slice, every row merged once
    for ([_][2]usize{ .{ 1, 1 }, .{ 127, 1 }, .{ 128, 1 }, .{ 129, 2 }, .{ 256, 2 }, .{ 257, 3 }, .{ 4096, 32 } }) |c| {
        var e = Recorder{};
        try pagedSlices(&e, .{ .paged = true, .total = c[0] }, 0, 128);
        try std.testing.expectEqual(2 * c[1], e.n);
        var merged: usize = 0;
        for (0..c[1]) |s| merged += e.got[2 * s + 1].grid[1];
        try std.testing.expectEqual(c[0], merged);
    }
}
