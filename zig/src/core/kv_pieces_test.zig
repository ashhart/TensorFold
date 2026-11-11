//! The packer on Flash Next's sizes (256-row pages with its MTP head): page counts, checks and refusals.
const std = @import("std");
const pieces = @import("kv_pieces.zig");
const t = std.testing;
const gpa = t.allocator;

// Flash Next's state and draft pieces, whole layers: recurrence, conv tail, MTP tail and projection, a page head
const rec: u64 = 48 * 128 * 128 * 4;
const conv: u64 = 3 * 10240 * 2;
const tail: u64 = 9 * 4 * 2560 * 2;
const proj: u64 = 32 * 16480 * 2;
const page_head: u64 = 256 * (2 * 2 * 256 * 2 + 128 * 2) * 13 + 64 * 128 * 2 * 13;

fn flashNext(late: bool) !std.ArrayList(u64) {
    var s: std.ArrayList(u64) = .empty;
    errdefer s.deinit(gpa);
    try s.appendNTimes(gpa, rec, 36);
    try s.appendNTimes(gpa, conv, 36);
    try s.append(gpa, tail);
    try s.appendNTimes(gpa, rec, 36); // spares
    try s.appendNTimes(gpa, conv, 36); // saved conv windows
    try s.appendNTimes(gpa, proj, 36); // saved projection rows
    try s.appendSlice(gpa, &.{ tail, 32 * 4 * 2560 * 2, 4 * 2560 * 2, 32768 * 2, 32 * 16 + 32 * 64 * 6 });
    if (late) {
        try s.append(gpa, 256);
        try s.appendNTimes(gpa, 48 * (16 * 128 + 48 * 128 + 48 * 2) * 4, 36);
    }
    return s;
}

test "Flash Next at 256-row pages with its head: pages and pieces as counted by hand, every piece inside one page" {
    try t.expectEqual(@as(u64, 7_880_704), page_head);
    for ([_]bool{ false, true }, [_]u32{ 37, 45 }, [_]u32{ 186, 223 }) |late, pages, n| {
        var s = try flashNext(late);
        defer s.deinit(gpa);
        var m = try pieces.pack(gpa, s.items, page_head, 256);
        defer m.deinit(gpa);
        try pieces.check(gpa, m);
        const sum = m.summary();
        try t.expectEqual(n, sum.n);
        try t.expectEqual(pages, sum.pages);
        for (m.at) |p| try t.expectEqual(@as(u64, 0), p.offset % 256);
    }
    var s = try flashNext(false);
    defer s.deinit(gpa);
    var state = try pieces.pack(gpa, s.items[0..73], page_head, 256); // the stream's own state alone
    defer state.deinit(gpa);
    try t.expectEqual(@as(u32, 18), state.pages);
    var all = try pieces.pack(gpa, s.items, page_head, 256);
    defer all.deinit(gpa);
    try t.expectEqual(@as(u64, 21_577_216), all.summary().pad(page_head));
}

test "packing is deterministic: the same sizes give the same places, piece order kept" {
    var s = try flashNext(true);
    defer s.deinit(gpa);
    var a = try pieces.pack(gpa, s.items, page_head, 256);
    defer a.deinit(gpa);
    var b = try pieces.pack(gpa, s.items, page_head, 256);
    defer b.deinit(gpa);
    try t.expectEqualSlices(pieces.Place, a.at, b.at);
    try t.expectEqualSlices(u64, s.items, a.sizes);
    var ids: [45]u32 = undefined;
    for (&ids, 0..) |*id, k| id.* = @intCast(90 - 2 * k); // any pages, in any order
    for (a.at, 0..) |p, i| try t.expectEqual(0x1000 + @as(u64, ids[p.page]) * page_head + p.offset, a.addr(i, 0x1000, &ids));
}

test "a piece wider than a page is refused by name, and an empty list takes no page" {
    try t.expectError(error.PieceOverPage, pieces.pack(gpa, &.{ 10, 4097 }, 4096, 256));
    try t.expectError(error.PieceOverPage, pieces.pack(gpa, &.{4000}, 4096, 256 * 64)); // aligned past the page
    var m = try pieces.pack(gpa, &.{}, 4096, 256);
    defer m.deinit(gpa);
    try t.expectEqual(@as(u32, 0), m.pages);
    try pieces.check(gpa, m);
}

test "first fit decreasing: small pieces fill the larger ones' gaps, ties in the caller's order" {
    var m = try pieces.pack(gpa, &.{ 100, 3000, 3000, 700, 1000 }, 4096, 256);
    defer m.deinit(gpa);
    try t.expectEqual(@as(u32, 2), m.pages);
    try t.expectEqual(pieces.Place{ .page = 0, .offset = 0 }, m.at[1]);
    try t.expectEqual(pieces.Place{ .page = 1, .offset = 0 }, m.at[2]);
    try t.expectEqual(pieces.Place{ .page = 0, .offset = 3072 }, m.at[4]);
    try t.expectEqual(pieces.Place{ .page = 1, .offset = 3072 }, m.at[3]);
    try t.expectEqual(pieces.Place{ .page = 1, .offset = 3840 }, m.at[0]);
    try pieces.check(gpa, m);
    try t.expectEqual(@as(u64, 2 * 4096 - 7800), m.summary().pad(4096));
}

test "a wrong map is refused: two pieces overlapping, a piece past its page or past the map's pages" {
    var m = try pieces.pack(gpa, &.{ 1000, 1000, 1000 }, 4096, 256);
    defer m.deinit(gpa);
    try pieces.check(gpa, m);
    const keep = m.at[1];
    m.at[1].offset = m.at[0].offset + 512;
    try t.expectError(error.PiecesOverlap, pieces.check(gpa, m));
    m.at[1] = .{ .page = 0, .offset = 3584 };
    try t.expectError(error.PieceOverPage, pieces.check(gpa, m));
    m.at[1] = .{ .page = 1, .offset = 0 };
    try t.expectError(error.PieceOverPage, pieces.check(gpa, m));
    m.at[1] = keep;
    try pieces.check(gpa, m);
}

test "rows by block: sizes, runs that never cross a block, addresses through any pages" {
    const r: pieces.Rows = .{ .rows = 100, .block = 32, .row = 20 };
    try t.expectEqual(@as(u32, 4), r.blocks());
    var s: std.ArrayList(u64) = .empty;
    defer s.deinit(gpa);
    try r.sizes(gpa, &s);
    try t.expectEqualSlices(u64, &.{ 640, 640, 640, 80 }, s.items);
    const x = r.run(30, 10);
    try t.expectEqual(@as(u32, 0), x.block);
    try t.expectEqual(@as(u64, 600), x.offset);
    try t.expectEqual(@as(u32, 2), x.rows); // rows 30 and 31; row 32 starts the next block
    try t.expectEqual(@as(u32, 4), r.run(96, 32).rows); // the short last block
    try t.expectEqual(@as(u32, 5), r.run(64, 5).rows);
    const bases = [_]u64{ 9000, 1000, 5000, 3000 };
    try t.expectEqual(@as(u64, 9000 + 31 * 20), r.at(&bases, 31));
    try t.expectEqual(@as(u64, 1000), r.at(&bases, 32));
    try t.expectEqual(@as(u64, 3000 + 3 * 20), r.at(&bases, 99));
    var i: u32 = 0; // every row once, block by block
    var n: u32 = 0;
    while (i < r.rows) : (n += 1) i += r.run(i, r.rows).rows;
    try t.expectEqual(@as(u32, 4), n);
}

/// A walk over `n` caller rows against Rows.at over `ref` from row `from`: whether every address matches.
fn walkMatches(p: pieces.Placed, ref: []const u64, from: u32, n: u32) bool {
    var i: u32 = 0;
    while (i < n) {
        const x = p.next(i, n);
        for (0..x.rows) |k| if (x.addr + k * p.rows.row != p.rows.at(ref, from + i + @as(u32, @intCast(k)))) return false;
        i += x.rows;
    }
    return true;
}

test "placed rows: a walk from any row hits each row's own address; planted: swapped blocks or a lost offset are seen" {
    const r: pieces.Rows = .{ .rows = 100, .block = 32, .row = 20 };
    const bases = [_]u64{ 9000, 1000, 5000, 3000 };
    for ([_]u32{ 0, 1, 20, 31, 32, 63 }) |from| try t.expect(walkMatches(.{ .bases = &bases, .rows = r, .from = from }, &bases, from, 100 - from));
    const swapped = [_]u64{ 9000, 5000, 1000, 3000 };
    try t.expect(!walkMatches(.{ .bases = &swapped, .rows = r, .from = 20 }, &bases, 20, 50));
    try t.expect(!walkMatches(.{ .bases = &bases, .rows = r, .from = 0 }, &bases, 20, 50)); // the offset lost
    const p: pieces.Placed = .{ .bases = &bases, .rows = r, .from = 20 };
    try t.expectEqual(@as(u32, 12), p.next(0, 50).rows); // rows 20..31, then block 1
    try t.expectEqual(@as(u64, 1000), p.next(12, 50).addr);
}
