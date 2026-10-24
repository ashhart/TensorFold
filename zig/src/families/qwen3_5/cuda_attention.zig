//! Contiguous attention has its own entry point and graph key; mixed batches retain per-block page selection.
const std = @import("std");
const Round = @import("cuda_round.zig");

pub fn hasPages(segs: anytype) bool {
    for (segs) |sg| if (sg.pt != 0) return true;
    return false;
}

pub fn partial(e: anytype, shape: anytype, layer: c_int) !void {
    const f = if (e.kv8) (if (shape.paged) e.fp8.attn_partial_paged else e.fp8.attn_partial_contiguous) else (if (shape.paged) e.fns.attn_partial_paged else e.fns.attn_partial_contiguous);
    const t = e.tables.ptr;
    try e.run(f, .{ 64, e.g.kv_heads, shape.total }, .{ 32 * (e.g.q_heads / e.g.kv_heads), 1, 1 }, .{ e.geo, e.qn, t + Round.segs_at, t + Round.rowseg_at, t + Round.rowanc_at, layer, t + Round.params_at, @as(f32, 1.0 / 16.0), e.pm, e.pl, e.po });
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
