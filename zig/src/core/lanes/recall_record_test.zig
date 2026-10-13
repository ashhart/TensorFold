//! The record against plain scans: longest matches, held runs, marks, distinct continuations, dropped chunks.
const std = @import("std");
const rr = @import("recall_record.zig");
const Record = rr.Record;
const gpa = std.testing.allocator;

/// A plain scan's longest match of `ctx`'s tail with an id after it, the newest of equal length (0: none).
fn scanLongest(ids: []const u32, ctx: []const u32) rr.Match {
    var best: rr.Match = .{};
    if (ctx.len < rr.key_len) return best;
    var p: usize = ids.len;
    while (p > 0) {
        p -= 1;
        if (p + 1 >= ids.len or ids[p + 1] == rr.mark) continue;
        var n: u32 = 0;
        while (n < rr.max_match and n < ctx.len and p >= n and ids[p - n] != rr.mark and ids[p - n] == ctx[ctx.len - 1 - n]) n += 1;
        if (n >= rr.key_len and n > best.len) best = .{ .len = n, .at = @intCast(p + 1) };
    }
    return best;
}

fn scanHeld(ids: []const u32, s: []const u32) usize {
    var best: usize = 0;
    if (s.len < rr.key_len) return 0;
    for (0..ids.len) |q| {
        var n: usize = 0;
        while (n < s.len and q + n < ids.len and ids[q + n] != rr.mark and ids[q + n] == s[n]) n += 1;
        if (n >= rr.key_len) best = @max(best, n);
    }
    return best;
}

test "longest and heldRun equal plain scans over segments of a small alphabet" {
    var rng = std.Random.DefaultPrng.init(11);
    const r = rng.random();
    for (0..60) |_| {
        var rec = try Record.init(gpa, 16);
        defer rec.deinit();
        var all: std.ArrayList(u32) = .empty;
        defer all.deinit(gpa);
        const symbols = r.intRangeAtMost(u32, 2, 6);
        for (0..r.intRangeAtMost(usize, 1, 12)) |_| {
            var seg: [40]u32 = undefined;
            const n = r.intRangeAtMost(usize, 1, seg.len);
            for (seg[0..n]) |*t| t.* = r.uintLessThan(u32, symbols);
            try rec.add(seg[0..n]);
            try all.appendSlice(gpa, seg[0..n]);
            try all.append(gpa, rr.mark);
        }
        for (0..20) |_| {
            var ctx: [30]u32 = undefined;
            const n = r.intRangeAtMost(usize, 1, ctx.len);
            for (ctx[0..n]) |*t| t.* = r.uintLessThan(u32, symbols);
            const want = scanLongest(all.items, ctx[0..n]);
            const got = rec.longest(ctx[0..n]);
            // a walk stops after max_steps candidates (colliding keys count too): compare where it saw them all
            const tail = if (n >= rr.key_len) countKeys(all.items, ctx[n - rr.key_len .. n]) else 0;
            if (tail <= rr.max_steps / 2) {
                try std.testing.expectEqual(want.len, got.len);
                if (want.len > 0) try std.testing.expectEqual(want.at, got.at);
            } else try std.testing.expect(got.len <= want.len);
            if (countKeys(all.items, ctx[0..@min(n, rr.key_len)]) <= rr.max_steps / 2) try std.testing.expectEqual(scanHeld(all.items, ctx[0..n]), rec.heldRun(ctx[0..n]).len);
        }
    }
}

/// Positions whose key (the key_len ids ending there, no mark) equals `key`.
fn countKeys(ids: []const u32, key: []const u32) usize {
    if (key.len < rr.key_len or ids.len < rr.key_len) return 0;
    var n: usize = 0;
    for (rr.key_len - 1..ids.len) |p| {
        if (std.mem.eql(u32, ids[p + 1 - rr.key_len .. p + 1], key)) n += 1;
    }
    return n;
}

test "no key spans a mark and a copy stops at one" {
    var rec = try Record.init(gpa, 12);
    defer rec.deinit();
    try rec.add(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    try rec.add(&.{ 11, 12, 13 });
    var out: [8]u32 = undefined;
    const m = rec.longest(&.{ 3, 4, 5, 6, 7, 8, 9, 10 });
    try std.testing.expectEqual(@as(u32, 0), m.len); // its continuation is the mark
    const hit = rec.longest(&.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 });
    try std.testing.expectEqual(@as(u32, 8), hit.len);
    try std.testing.expectEqualSlices(u32, &.{ 9, 10 }, rec.copy(hit.at, &out));
    try std.testing.expectEqual(@as(u32, 0), rec.longest(&.{ 6, 7, 8, 9, 10, 11, 12, 13 }).len); // across the join
}

test "continuations are distinct, newest first, and shorter matches than `min` are left out" {
    var rec = try Record.init(gpa, 12);
    defer rec.deinit();
    const key = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try rec.add(&(key ++ [_]u32{ 20, 21 }));
    try rec.add(&([_]u32{ 9, 9 } ++ key ++ [_]u32{ 30, 31 }));
    try rec.add(&(key ++ [_]u32{ 20, 21, 22 }));
    var scratch: [4]u32 = undefined;
    var out: [4]rr.Found = undefined;
    const n = rec.continuations(&([_]u32{ 9, 9 } ++ key), 8, 2, &scratch, &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    var got: [4]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 20, 21, 22 }, rec.copy(out[0].at, &got)); // the newest of the 20, 21 pair
    try std.testing.expectEqualSlices(u32, &.{ 30, 31 }, rec.copy(out[1].at, &got));
    try std.testing.expectEqual(@as(u32, 10), out[1].len);
    try std.testing.expectEqual(@as(usize, 1), rec.continuations(&([_]u32{ 9, 9 } ++ key), 10, 2, &scratch, &out));
}

test "a dropped chunk's matches are gone, later ones stay, and clear empties the record" {
    var rec = try Record.init(gpa, 16);
    defer rec.deinit();
    const early = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109 };
    try rec.add(&early);
    const filler = try gpa.alloc(u32, rr.chunk_tokens + 50);
    defer gpa.free(filler);
    for (filler, 0..) |*t, i| t.* = @intCast(1000 + i % 5000);
    try rec.add(filler);
    const late = [_]u32{ 200, 201, 202, 203, 204, 205, 206, 207, 208, 209 };
    try rec.add(&late);
    try std.testing.expect(rec.longest(early[0..9]).len > 0);
    try std.testing.expect(rec.dropOldest());
    try std.testing.expectEqual(@as(u32, 0), rec.longest(early[0..9]).len);
    try std.testing.expectEqual(@as(u32, 9), rec.longest(late[0..9]).len);
    try std.testing.expect(!rec.dropOldest()); // the newest chunk stays
    rec.clear();
    try std.testing.expectEqual(@as(u32, 0), rec.held());
    try std.testing.expectEqual(@as(u32, 0), rec.longest(late[0..9]).len);
    try rec.add(&late);
    try std.testing.expectEqual(@as(u32, 9), rec.longest(late[0..9]).len);
}
