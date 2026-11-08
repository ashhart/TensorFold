//! Groups sized by bytes: rows a page, state pages and padding, one group == today's layout, the load report.
const std = @import("std");
const kv = @import("kv_pool.zig");
const groups = @import("kv_groups.zig");
const t = std.testing;

test "a page holds whole rows of a group, and the tail past them is its padding" {
    try t.expectEqual(groups.Fit{ .rows = 3, .pages = 1, .pad = 100 }, groups.rows(300, 1000));
    try t.expectEqual(groups.Fit{ .rows = 4, .pages = 1, .pad = 0 }, groups.rows(250, 1000));
    try t.expectEqual(groups.Fit{ .rows = 0, .pages = 1, .pad = 1000 }, groups.rows(1001, 1000));
}

test "a state takes whole pages, the last one's tail padding" {
    try t.expectEqual(groups.Fit{ .rows = 0, .pages = 3, .pad = 500 }, groups.state(2500, 1000));
    try t.expectEqual(groups.Fit{ .rows = 0, .pages = 2, .pad = 0 }, groups.state(2000, 1000));
    try t.expectEqual(groups.Fit{ .rows = 0, .pages = 1, .pad = 999 }, groups.state(1, 1000));
    try t.expectEqual(groups.Fit{ .rows = 0, .pages = 0, .pad = 0 }, groups.state(0, 1000));
}

test "a group splits its bytes a row over its layers only when they divide evenly" {
    try t.expectEqual(groups.Group{ .kind = .window, .row = 16384, .layers = 50 }, groups.Group.split(.window, 50 * 16384, 50).?);
    try t.expectEqual(@as(?groups.Group, null), groups.Group.split(.window, 50 * 16384 + 1, 50));
    try t.expectEqual(@as(?groups.Group, null), groups.Group.split(.window, 0, 0));
}

/// Gemma 4's pages: 256 rows of each global layer; a sliding row `times` a global one.
fn gemma(times: u64) kv.Layout {
    const row = 2 * 4 * 512 * 2;
    var l = kv.Layout.ofRows(.{ .kind = .full, .row = row, .layers = 10 }, 256, 1 << 30, 0);
    l.window = .{ .kind = .window, .row = times * row, .layers = 50 };
    return l;
}

test "sliding rows 2x and 4x a global row fill a layer's page with half and a quarter the rows, no padding" {
    const s2 = gemma(2).shape();
    try t.expectEqual(groups.Fit{ .rows = 128, .pages = 1, .pad = 0 }, groups.rows(s2.window.?.row, s2.layerPage()));
    const s4 = gemma(4).shape();
    try t.expectEqual(groups.Fit{ .rows = 64, .pages = 1, .pad = 0 }, groups.rows(s4.window.?.row, s4.layerPage()));
    try t.expectEqual(groups.Fit{ .rows = 256, .pages = 1, .pad = 0 }, groups.rows(s2.full.row, s2.layerPage()));
}

/// Today's layouts written by hand, each family's form: token bytes over every layer, one group.
const today = [_]kv.Layout{
    .{ .fixed_bytes = 1 << 30, .token_bytes = 10 * 8192, .page_tokens = 256, .snap_bytes = 0 },
    .{ .fixed_bytes = 180 << 20, .token_bytes = 13_056, .page_tokens = 512, .snap_bytes = 0 },
    .{ .fixed_bytes = 64 << 20, .token_bytes = 20_480, .page_tokens = 256, .snap_bytes = 48 << 20, .run_bytes = 51 << 20 },
    .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 0, .run_bytes = 3 },
};

test "one group sized by bytes a row is today's layout: the same pages, runs, needs and plans" {
    const by_rows = kv.Layout.ofRows(.{ .kind = .full, .row = 8192, .layers = 10 }, 256, 1 << 30, 0);
    try t.expectEqual(today[0].pageBytes(), by_rows.pageBytes());
    try t.expectEqual(today[0].token_bytes, by_rows.group().bytes());
    for (today) |l| {
        try t.expectEqual(l.token_bytes, l.group().bytes());
        try t.expectEqual((l.run_bytes + l.pageBytes() - 1) / l.pageBytes(), l.runPages());
        for ([_]u64{ 0, 1, 255, 256, 8191, 70_000 }) |n| try t.expectEqual(l.pages(n) + l.runPages(), l.need(n, 0, 0));
    }
    for ([_]u64{ 0, 1, 4096, 4097, 1 << 20 }) |run| {
        var l = today[2];
        l.run_bytes = run;
        try t.expectEqual((run + l.pageBytes() - 1) / l.pageBytes(), l.runPages());
    }
    const b: kv.Budget = .{ .room = 40 << 30, .engine = 2 << 30, .lanes = 4, .cache = 4 << 30, .window = 8192 };
    try t.expectEqual(try kv.plan(today[0], b), try kv.plan(by_rows, b));
}

test "the load report shows a Qwen state run's pages and padding at 256-row pages" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const shown = try groups.line(arena.allocator(), today[2].shape());
    const want = "KV page groups: one page 5.00 MiB, 256 rows of the full group; full: layers not named, 20.00 KiB a token, " ++
        "256 rows a page, 0 B padding (0.0 %); state run, in the pool: 51.00 MiB on 11 pages, 4.00 MiB padding " ++
        "(7.3 %); snapshot, outside the pool: 48.00 MiB on 10 pages, 2.00 MiB padding (4.0 %)";
    try t.expectEqualStrings(want, shown);
}

test "a family that doesn't name its layers shows bytes a token, never one layer; one named layer shows as one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    for (today) |l| {
        try t.expect(!l.shape().named);
        const shown = try groups.line(arena.allocator(), l.shape());
        try t.expect(std.mem.indexOf(u8, shown, "layers not named") != null and std.mem.indexOf(u8, shown, " x ") == null);
    }
    const one = kv.Layout.ofRows(.{ .kind = .full, .row = 7168, .layers = 1 }, 512, 0, 0);
    const want = "KV page groups: one page 3.50 MiB, 512 rows of the full group; full: 1 layer x 7.00 KiB a row, " ++
        "512 rows a layer's page, 0 B padding (0.0 %)";
    try t.expectEqualStrings(want, try groups.line(arena.allocator(), one.shape()));
}

test "the load report shows a window outside the pool by its bytes a row against a full row" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const shown = try groups.line(arena.allocator(), gemma(2).shape());
    const want = "KV page groups: one page 20.00 MiB, 256 rows of the full group; full: 10 layers x 8.00 KiB a row, " ++
        "256 rows a layer's page, 0 B padding (0.0 %); window, outside the pool: 50 layers x 16.00 KiB a row " ++
        "(2.00x a full row), 128 rows a layer's page, 0 B padding (0.0 %)";
    try t.expectEqualStrings(want, shown);
}

test "an uneven window row leaves its padding in each layer's page" {
    var l = gemma(2);
    l.window = .{ .kind = .window, .row = 3000, .layers = 2 };
    const s = l.shape();
    try t.expectEqual(groups.Fit{ .rows = 699, .pages = 1, .pad = 2097152 - 699 * 3000 }, groups.rows(3000, s.layerPage()));
}
