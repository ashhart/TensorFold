//! Record production token-at-a-time attention without a driver or shader compiler.
const std = @import("std");
const d = @import("decode");

const Launch = struct {
    pipe: usize = 0,
    bufs: [12]d.Ref = @splat(.{ .buf = 0 }),
    dims: [5]i32 = @splat(0),
    page_bytes: u64 = 0,
    grid: [3]usize = @splat(0),
    group: [3]usize = @splat(0),
};
const Record = struct {
    next: Launch = .{},
    calls: [256]Launch = undefined,
    used: usize = 0,
    pub fn pipe(r: *@This(), p: usize) void {
        r.next = .{ .pipe = p };
    }
    pub fn buf(r: *@This(), b: usize, off: usize, i: usize) void {
        r.next.bufs[i] = .{ .buf = b, .off = off };
    }
    pub fn bytes(r: *@This(), value: anytype, _: usize) void {
        if (@TypeOf(value) == [5]i32) r.next.dims = value;
        if (@TypeOf(value) == u64) r.next.page_bytes = value;
    }
    pub fn run(r: *@This(), grid: [3]usize, group: [3]usize) void {
        r.next.grid = grid;
        r.next.group = group;
        r.calls[r.used] = r.next;
        r.used += 1;
    }
};
const pipes = d.Pipelines{ .q = 1, .partial = 2, .merge = 3, .out = 4 };
fn call(rows: usize, len: usize, first: usize) d.Call {
    return .{ .head_dim = 128, .heads = 32, .kv_heads = 2, .qkv_n = 4608, .capacity = 1 << 19, .len = len, .first = first, .rows = rows, .qkv = .{ .buf = 1, .off = 64 }, .qp = .{ .buf = 2, .off = 64 }, .k = .{ .buf = 3, .off = 64 }, .v = .{ .buf = 4, .off = 64 }, .po = .{ .buf = 5, .off = 64 }, .pm = .{ .buf = 6, .off = 64 }, .pl = .{ .buf = 7, .off = 64 }, .att = .{ .buf = 8, .off = 64 }, .out = .{ .buf = 9, .off = 64 } };
}

test "decode attention: only pre-M5 Lightning enables the verify policy" {
    try std.testing.expect(d.enabled(false, false));
    try std.testing.expect(!d.enabled(true, false));
    try std.testing.expect(!d.enabled(false, true));
    try std.testing.expect(!d.enabled(true, true));
}

test "decode attention: all 1..64 rows use one-row programs and extents at tile and chunk seams" {
    for (1..65) |rows| for ([_]usize{ 0, 63, 64, 511, 512, 2047, 22017, 262079 }) |len| {
        var e: Record = .{};
        const c = call(rows, len, 7);
        d.encode(&e, pipes, c);
        try std.testing.expectEqual(4 * rows, e.used);
        for (0..rows) |r| {
            const q = e.calls[4 * r];
            const p = e.calls[4 * r + 1];
            const m = e.calls[4 * r + 2];
            const o = e.calls[4 * r + 3];
            try std.testing.expectEqual(@as(usize, 1), q.pipe);
            try std.testing.expectEqual(@as(usize, 2), p.pipe);
            try std.testing.expectEqual(@as(usize, 3), m.pipe);
            try std.testing.expectEqual(@as(usize, 4), o.pipe);
            const keys = len + r + 1;
            const chunks = (keys + 511) / 512;
            try std.testing.expectEqual([5]i32{ @intCast(keys), @intCast(chunks), 1, 1, 1 }, p.dims);
            try std.testing.expectEqual(p.dims, m.dims);
            try std.testing.expectEqual([3]usize{ 64, chunks, 1 }, p.grid);
            try std.testing.expectEqual([3]usize{ 32, 1, 1 }, p.group);
            try std.testing.expectEqual([3]usize{ 64, 16, 1 }, m.grid);
            try std.testing.expectEqual([3]usize{ 128, 32, 1 }, o.grid);
            try std.testing.expectEqual(64 + (7 + r) * 4608 * 2, q.bufs[0].off);
            try std.testing.expectEqual(64 + (7 + r) * 4096 * 2, o.bufs[1].off);
            try std.testing.expectEqual(c.k, p.bufs[1]);
            try std.testing.expectEqual(c.v, p.bufs[3]);
        }
    };
}

test "decode attention: page offsets and table stay bound to the one-row paged twin" {
    var c = call(3, 511, 0);
    c.k.off = 4096;
    c.v.off = 8192;
    c.table = .{ .buf = 17, .off = 2048 };
    c.page_bytes = 131072;
    var e: Record = .{};
    d.encode(&e, pipes, c);
    for (0..3) |r| {
        const p = e.calls[4 * r + 1];
        try std.testing.expectEqual(c.k, p.bufs[1]);
        try std.testing.expectEqual(c.v, p.bufs[3]);
        try std.testing.expectEqual(c.table.?, p.bufs[10]);
        try std.testing.expectEqual(c.page_bytes, p.page_bytes);
    }
}

test "decode attention: only eligible paged verify segments serialize" {
    for ([_]bool{ false, true }) |verify| for ([_]bool{ false, true }) |paged| for ([_]bool{ false, true }) |lone| for ([_]bool{ false, true }) |tree| for ([_]bool{ false, true }) |gpu| {
        try std.testing.expectEqual(verify and paged and !lone and !tree and !gpu, d.serialForSegment(verify, paged, lone, tree, gpu));
    };
}
