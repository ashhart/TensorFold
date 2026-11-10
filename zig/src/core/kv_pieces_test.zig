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
