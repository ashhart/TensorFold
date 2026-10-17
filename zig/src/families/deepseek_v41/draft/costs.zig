//! The verify / DSpark cost table the depth prices with (Python depth.Costs, calib.py's table shape and cache file).
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const max_rows = 64; // the widest row-graph bucket (TF_DSV41_GRAPH_ROWS_MAX)
pub const slot_rows = 16; // a slot's window at most (MAX_ROWS)
pub const wide = [_]u32{ 24, 32, 48, 64 }; // the shared buckets calib.py times on several slots

/// `verify[r - 1]`: ms of a window of r rows in total; `draft`: one DSpark pass; `slot`: each further slot's ms.
pub const Costs = struct {
    verify: []f64,
    draft: f64,
    slot: f64 = 0.0,

    pub fn deinit(c: *Costs, gpa: Allocator) void {
        gpa.free(c.verify);
    }

    pub fn clone(c: Costs, gpa: Allocator) !Costs {
        return .{ .verify = try gpa.dupe(f64, c.verify), .draft = c.draft, .slot = c.slot };
    }

    /// A window of `rows` rows, clamped to the table (Python window_ms).
    pub fn windowMs(c: Costs, rows: i64) f64 {
        const n: i64 = @intCast(c.verify.len);
        return c.verify[@intCast(@max(0, @min(rows, n) - 1))];
    }

    /// A window of `rows` total rows; past the table on the line of its last two entries (Python rows_ms).
    pub fn rowsMs(c: Costs, rows: i64) f64 {
        const v = c.verify;
        const r = @max(1, rows);
        if (r <= v.len) return v[@intCast(r - 1)];
        const slope = if (v.len >= 2) @max(v[v.len - 1] - v[v.len - 2], 0.0) else 0.0;
        return v[v.len - 1] + slope * @as(f64, @floatFromInt(r - @as(i64, @intCast(v.len))));
    }

    /// Microsecond ints [draft, slot, verify...] (both ranks share these; Python encode, round half to even).
    pub fn encode(c: Costs, gpa: Allocator) ![]i64 {
        const out = try gpa.alloc(i64, 2 + c.verify.len);
        out[0] = micro(c.draft);
        out[1] = micro(c.slot);
        for (out[2..], c.verify) |*o, v| o.* = micro(v);
        return out;
    }

    pub fn decode(gpa: Allocator, ints: []const i64) !Costs {
        if (ints.len < 3) return error.BadCosts;
        const verify = try gpa.alloc(f64, ints.len - 2);
        for (verify, ints[2..]) |*v, x| v.* = @as(f64, @floatFromInt(x)) / 1000.0;
        return .{ .verify = verify, .draft = @as(f64, @floatFromInt(ints[0])) / 1000.0, .slot = @as(f64, @floatFromInt(ints[1])) / 1000.0 };
    }

    /// Python's cached calibration (`calib-<key>.json`: verify, draft, slot); at least 2 rows.
    pub fn parseJson(gpa: Allocator, bytes: []const u8) !Costs {
        const File = struct { verify: []f64, draft: f64, slot: f64 = 0.0 };
        const parsed = try std.json.parseFromSlice(File, gpa, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        if (parsed.value.verify.len < 2) return error.BadCosts;
        return .{ .verify = try gpa.dupe(f64, parsed.value.verify), .draft = parsed.value.draft, .slot = parsed.value.slot };
    }
};

fn micro(v: f64) i64 {
    return @max(1, @min(roundEven(v * 1000.0), 1_000_000_000));
}

/// Python's round(): half to even.
pub fn roundEven(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    var r = f;
    if (d > 0.5 or (d == 0.5 and @mod(f, 2.0) != 0.0)) r = f + 1.0;
    return @intFromFloat(r);
}

/// Section 1.3's round model before any measurement: ~37 ms + 5 ms a row, the pass 5.6 ms (Python default_costs).
pub fn defaults(gpa: Allocator, rows: usize) !Costs {
    const verify = try gpa.alloc(f64, rows);
    for (verify, 0..) |*v, r| v.* = 37.0 + 5.0 * @as(f64, @floatFromInt(r + 1));
    return .{ .verify = verify, .draft = 5.6 };
}

/// Each row's value made non-decreasing by pooling adjacent violators (Python calib.monotone), in place.
pub fn monotone(gpa: Allocator, times: []f64) !void {
    const Block = struct { sum: f64, n: f64 };
    var blocks: std.ArrayList(Block) = .empty;
    defer blocks.deinit(gpa);
    for (times) |x| {
        try blocks.append(gpa, .{ .sum = x, .n = 1.0 });
        while (blocks.items.len > 1) {
            const b = blocks.items[blocks.items.len - 1];
            const a = &blocks.items[blocks.items.len - 2];
            if (!(a.sum / a.n > b.sum / b.n)) break;
            a.sum += b.sum;
            a.n += b.n;
            blocks.items.len -= 1;
        }
    }
    var at: usize = 0;
    for (blocks.items) |b| {
        for (0..@intFromFloat(b.n)) |_| {
            times[at] = b.sum / b.n;
            at += 1;
        }
    }
}

pub const Point = struct { rows: u32, ms: f64 };

/// `table` (rows 1 .. len) continued to the widest point, linear between them, non-decreasing (Python calib.extend).
pub fn extend(gpa: Allocator, table: []const f64, points: []const Point) ![]f64 {
    var out: std.ArrayList(f64) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, table);
    if (table.len == 0) return out.toOwnedSlice(gpa);
    var x0: u32 = @intCast(table.len);
    var y0 = table[table.len - 1];
    for (points) |p| { // ascending rows
        if (p.rows <= x0) continue;
        var r = x0 + 1;
        while (r <= p.rows) : (r += 1) {
            try out.append(gpa, y0 + (p.ms - y0) * @as(f64, @floatFromInt(r - x0)) / @as(f64, @floatFromInt(p.rows - x0)));
        }
        x0 = p.rows;
        y0 = p.ms;
    }
    for (1..out.items.len) |i| out.items[i] = @max(out.items[i], out.items[i - 1]);
    return out.toOwnedSlice(gpa);
}

/// The table calib.py measures: rows 1..16 each its own (monotone), the wide buckets less the slot overhead, extended.
pub fn measured(gpa: Allocator, rows: []const f64, buckets: []const Point, draft: f64, slot: f64) !Costs {
    const table = try gpa.dupe(f64, rows);
    defer gpa.free(table);
    try monotone(gpa, table);
    const pts = try gpa.alloc(Point, buckets.len);
    defer gpa.free(pts);
    for (pts, buckets) |*p, b| {
        const slots: f64 = @floatFromInt((b.rows + slot_rows - 1) / slot_rows);
        p.* = .{ .rows = b.rows, .ms = b.ms - slot * (slots - 1.0) };
    }
    return .{ .verify = try extend(gpa, table, pts), .draft = draft, .slot = slot };
}

test "rows past the table follow its last line; encode rounds like Python" {
    const gpa = std.testing.allocator;
    var c = try defaults(gpa, 16);
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(f64, 42.0), c.windowMs(1));
    try std.testing.expectEqual(@as(f64, 117.0), c.windowMs(99));
    try std.testing.expectEqual(@as(f64, 127.0), c.rowsMs(18));
    try std.testing.expectEqual(@as(i64, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(i64, 4), roundEven(3.5));
    const ints = try c.encode(gpa);
    defer gpa.free(ints);
    var back = try Costs.decode(gpa, ints);
    defer back.deinit(gpa);
    try std.testing.expectEqualSlices(f64, c.verify, back.verify);
    try std.testing.expectEqual(@as(f64, 5.6), back.draft);
}

test "monotone pools a dip with the row before it" {
    const gpa = std.testing.allocator;
    var t = [_]f64{ 1, 3, 2, 4 };
    try monotone(gpa, &t);
    try std.testing.expectEqualSlices(f64, &.{ 1, 2.5, 2.5, 4 }, &t);
}
