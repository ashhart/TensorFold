//! Row mode's per-row metadata: a mixed-slot window padded to a bucket, its per-row values as one int64 device table.

const std = @import("std");

pub const buckets = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64 };
/// rows a slot's window holds at most (graphs.MAX_ROWS)
pub const seg_max = 16;
pub const Col = enum(u32) { ids, pos, rslot, wslot, prev, src };
pub const ncol = 6;

comptime {
    std.debug.assert(@backingInt(Col.src) + 1 == ncol);
}

/// One slot's window of a mix: its slot, first position and rows, in plan order.
pub const Seg = struct { slot: u32, start: u64, rows: u32 };

/// The smallest bucket of `cap` or fewer rows holding `n` (null: more than the largest).
pub fn bucketRows(n: u32, cap: u32) ?u32 {
    for (buckets) |b| if (b >= n and b <= cap) return b;
    return null;
}

/// Each window's rows [a, b) in the padded run, into `out` (len = mix.len).
pub fn bounds(mix: []const Seg, out: [][2]u32) void {
    var a: u32 = 0;
    for (mix, out) |m, *o| {
        o.* = .{ a, a + m.rows };
        a += m.rows;
    }
}

pub fn totalRows(mix: []const Seg) u32 {
    var n: u32 = 0;
    for (mix) |m| n += m.rows;
    return n;
}

pub const Error = error{ NoWindow, SlotTwice, SegmentRows, TooManyRows, PastLimit, SlotRange, IdCount };

/// Checks a mix: windows of 1..seg_max rows, no slot twice, within `limit` and the slots.
pub fn check(mix: []const Seg, slots: u32, limit: u64) Error!void {
    if (mix.len == 0) return error.NoWindow;
    var seen: u64 = 0;
    for (mix) |m| {
        if (m.slot >= slots or m.slot >= 64) return error.SlotRange;
        const bit = @as(u64, 1) << @intCast(m.slot);
        if (seen & bit != 0) return error.SlotTwice;
        seen |= bit;
        if (m.rows < 1 or m.rows > seg_max) return error.SegmentRows;
        if (m.start + m.rows > limit) return error.PastLimit;
    }
}

/// The last position any row of the mix holds (the context bucket's input).
pub fn lastPos(mix: []const Seg) u64 {
    var last: u64 = 0;
    for (mix) |m| last = @max(last, m.start + m.rows - 1);
    return last;
}

/// The table int64 [NCOL, rmax] of `mix` padded to `R` rows, then rslot as int32 [rmax]; `slots` carry rows first.
pub fn plan(mix: []const Seg, ids: []const u32, R: u32, rmax: u32, slots: u32, out: []i64, rslot32: []i32) Error!void {
    const n = totalRows(mix);
    if (ids.len != n) return error.IdCount;
    if (n < 1 or n > R or R > rmax) return error.TooManyRows;
    std.debug.assert(out.len >= ncol * rmax and rslot32.len >= rmax);
    const col = struct {
        fn of(o: []i64, rm: u32, c: Col) []i64 {
            return o[@backingInt(c) * rm ..][0..rm];
        }
    }.of;
    const id_c = col(out, rmax, .ids);
    const pos_c = col(out, rmax, .pos);
    const rs_c = col(out, rmax, .rslot);
    const ws_c = col(out, rmax, .wslot);
    const prev_c = col(out, rmax, .prev);
    const src_c = col(out, rmax, .src);
    for (ids, 0..) |t, i| id_c[i] = t;
    var a: u32 = 0;
    for (mix) |m| {
        if (m.slot >= slots) return error.SlotRange;
        const s: i64 = m.slot;
        for (a..a + m.rows, 0..) |r, k| {
            pos_c[r] = @intCast(m.start + k);
            rs_c[r] = s;
            ws_c[r] = s;
            // the slot's carry for the window's first row (position start - 1), else the row before
            prev_c[r] = if (k == 0) s else @as(i64, slots) + @as(i64, @intCast(r)) - 1;
            src_c[r] = @intCast(r);
        }
        a += m.rows;
    }
    // padding: the last real row, writing nothing
    for (n..R) |r| {
        id_c[r] = id_c[n - 1];
        pos_c[r] = pos_c[n - 1];
        rs_c[r] = rs_c[n - 1];
        ws_c[r] = -1;
        prev_c[r] = prev_c[n - 1];
        src_c[r] = src_c[n - 1];
    }
    for (rslot32[0..R], rs_c[0..R]) |*d, v| d.* = @intCast(v);
}

/// Bytes of the device table: the int64 columns, then the int32 rslot copy.
pub fn tableBytes(rmax: u32) usize {
    return 8 * ncol * @as(usize, rmax) + 4 * @as(usize, rmax);
}

/// Byte offset of column `c` (int64 [rmax]) in the table.
pub fn colOffset(c: Col, rmax: u32) usize {
    return 8 * @as(usize, @backingInt(c)) * rmax;
}

/// Byte offset of the int32 rslot copy.
pub fn rslot32Offset(rmax: u32) usize {
    return 8 * ncol * @as(usize, rmax);
}

/// Splits a mix into forwards of at most `cap` rows, segments whole and in order; `out` gets [first, end); the count.
pub fn pack(mix: []const Seg, cap: u32, out: [][2]usize) Error!usize {
    var k: usize = 0;
    var i: usize = 0;
    while (i < mix.len) {
        var rows: u32 = 0;
        var j = i;
        while (j < mix.len and rows + mix[j].rows <= cap) : (j += 1) rows += mix[j].rows;
        if (j == i) return error.SegmentRows; // one segment past the cap
        if (k == out.len) return error.TooManyRows;
        out[k] = .{ i, j };
        k += 1;
        i = j;
    }
    return k;
}

// -- host tests ---------------------------------------------------------------------------------------------------

const testing = std.testing;

test "rowtab: buckets and the largest bucket a cap allows" {
    try testing.expectEqual(@as(?u32, 1), bucketRows(1, 64));
    try testing.expectEqual(@as(?u32, 8), bucketRows(8, 64));
    try testing.expectEqual(@as(?u32, 12), bucketRows(9, 64));
    try testing.expectEqual(@as(?u32, 16), bucketRows(13, 64));
    try testing.expectEqual(@as(?u32, 64), bucketRows(49, 64));
    try testing.expectEqual(@as(?u32, null), bucketRows(65, 64));
    try testing.expectEqual(@as(?u32, null), bucketRows(17, 16));
}

test "rowtab: the plan of a mix as Python's plan_rows (padding mirrors the last real row and writes nothing)" {
    // reference: plan_rows([(2, 10, 3), (0, 40, 2)], ids, 8, carries=4)
    const mix = [_]Seg{ .{ .slot = 2, .start = 10, .rows = 3 }, .{ .slot = 0, .start = 40, .rows = 2 } };
    const ids = [_]u32{ 7, 8, 9, 100, 101 };
    const rmax = 16;
    var t: [ncol * rmax]i64 = undefined;
    var r32: [rmax]i32 = undefined;
    try plan(&mix, &ids, 8, rmax, 4, &t, &r32);
    const C = struct {
        fn c(x: []const i64, k: Col) []const i64 {
            return x[@backingInt(k) * rmax ..][0..8];
        }
    }.c;
    try testing.expectEqualSlices(i64, &.{ 7, 8, 9, 100, 101, 101, 101, 101 }, C(&t, .ids));
    try testing.expectEqualSlices(i64, &.{ 10, 11, 12, 40, 41, 41, 41, 41 }, C(&t, .pos));
    try testing.expectEqualSlices(i64, &.{ 2, 2, 2, 0, 0, 0, 0, 0 }, C(&t, .rslot));
    try testing.expectEqualSlices(i64, &.{ 2, 2, 2, 0, 0, -1, -1, -1 }, C(&t, .wslot));
    // prev: the slot's carry for a window's first row, else carries + the row before (carries = 4)
    try testing.expectEqualSlices(i64, &.{ 2, 4, 5, 0, 7, 7, 7, 7 }, C(&t, .prev));
    try testing.expectEqualSlices(i64, &.{ 0, 1, 2, 3, 4, 4, 4, 4 }, C(&t, .src));
    try testing.expectEqualSlices(i32, &.{ 2, 2, 2, 0, 0, 0, 0, 0 }, r32[0..8]);
    var b: [2][2]u32 = undefined;
    bounds(&mix, &b);
    try testing.expectEqual([2]u32{ 0, 3 }, b[0]);
    try testing.expectEqual([2]u32{ 3, 5 }, b[1]);
    try testing.expectEqual(@as(u64, 41), lastPos(&mix));
    try testing.expectError(error.TooManyRows, plan(&mix, &ids, 4, rmax, 4, &t, &r32));
    try testing.expectError(error.IdCount, plan(&mix, ids[0..4], 8, rmax, 4, &t, &r32));
}

test "rowtab: checks (a slot twice, rows a window, the limit, the slots)" {
    try check(&.{ .{ .slot = 0, .start = 0, .rows = 1 }, .{ .slot = 3, .start = 5, .rows = 16 } }, 4, 4096);
    try testing.expectError(error.SlotTwice, check(&.{ .{ .slot = 1, .start = 0, .rows = 1 }, .{ .slot = 1, .start = 5, .rows = 1 } }, 4, 4096));
    try testing.expectError(error.SegmentRows, check(&.{.{ .slot = 0, .start = 0, .rows = 17 }}, 4, 4096));
    try testing.expectError(error.PastLimit, check(&.{.{ .slot = 0, .start = 4095, .rows = 2 }}, 4, 4096));
    try testing.expectError(error.SlotRange, check(&.{.{ .slot = 4, .start = 0, .rows = 1 }}, 4, 4096));
    try testing.expectError(error.NoWindow, check(&.{}, 4, 4096));
}

test "rowtab: a mix packed into forwards under the row cap, each segment whole" {
    const mix = [_]Seg{ .{ .slot = 0, .start = 0, .rows = 9 }, .{ .slot = 1, .start = 0, .rows = 7 }, .{ .slot = 2, .start = 0, .rows = 4 }, .{ .slot = 3, .start = 0, .rows = 16 } };
    var out: [4][2]usize = undefined;
    try testing.expectEqual(@as(usize, 3), try pack(&mix, 16, &out));
    try testing.expectEqual([2]usize{ 0, 2 }, out[0]);
    try testing.expectEqual([2]usize{ 2, 3 }, out[1]);
    try testing.expectEqual([2]usize{ 3, 4 }, out[2]);
    try testing.expectEqual(@as(usize, 1), try pack(&mix, 64, &out));
    try testing.expectError(error.SegmentRows, pack(&mix, 8, &out));
}

test "rowtab: the device table's layout" {
    try testing.expectEqual(@as(usize, 8 * 6 * 64 + 4 * 64), tableBytes(64));
    try testing.expectEqual(@as(usize, 512), colOffset(.pos, 64));
    try testing.expectEqual(@as(usize, 3072), rslot32Offset(64));
}
