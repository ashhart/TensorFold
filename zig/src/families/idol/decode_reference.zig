//! Gating a decode against idol-sdk's: worst rows, count and topology differences, activated fields, PLY columns.
const std = @import("std");
const gs = @import("gaussian.zig");
const ref = @import("reference.zig");

/// The row of `width` with the largest relative L2 error, and that error.
pub const Worst = struct { row: usize, err: f64 };

pub fn worst(ours: []const f32, theirs: []const f32, width: usize) Worst {
    std.debug.assert(ours.len == theirs.len and ours.len % width == 0);
    var w: Worst = .{ .row = 0, .err = 0 };
    for (0..ours.len / width) |r| {
        var d: f64 = 0;
        var b: f64 = 0;
        for (ours[r * width ..][0..width], theirs[r * width ..][0..width]) |x, y| {
            d += (@as(f64, x) - y) * (@as(f64, x) - y);
            b += @as(f64, y) * y;
        }
        const e = @sqrt(d / @max(b, 1e-30));
        if (!(e <= w.err)) w = .{ .row = r, .err = e };
    }
    return w;
}

/// The distance a gate allows twice of: the worst of the SDK's own, the fp16 floor's and the SDK's nudged reruns'.
pub fn reference(distances: []const ref.Match) ref.Match {
    var r = distances[0];
    for (distances[1..]) |d| r = .{ .max_abs = @max(r.max_abs, d.max_abs), .cosine = @min(r.cosine, d.cosine), .worst_row = @max(r.worst_row, d.worst_row) };
    return r;
}

/// Rows of `width` counts that differ, and the first.
pub const Counts = struct { rows: usize, first: ?usize };

pub fn counts(ours: []const u32, theirs: []const u32, width: usize) Counts {
    if (ours.len != theirs.len) return .{ .rows = @max(ours.len, theirs.len) / width, .first = 0 };
    var c: Counts = .{ .rows = 0, .first = null };
    for (0..ours.len / width) |r| {
        if (std.mem.eql(u32, ours[r * width ..][0..width], theirs[r * width ..][0..width])) continue;
        c.rows += 1;
        if (c.first == null) c.first = r;
    }
    return c;
}

/// Element-wise: largest absolute difference and where, and how many exceed |d| <= rel |theirs| + abs.
pub const Field = struct { max_abs: f64, at: usize, bad: usize, exact: usize };

pub fn field(ours: []const f32, theirs: []const f32, rel: f64, abs: f64) Field {
    std.debug.assert(ours.len == theirs.len);
    var f: Field = .{ .max_abs = 0, .at = 0, .bad = 0, .exact = 0 };
    for (ours, theirs, 0..) |x, y, i| {
        const d = @abs(@as(f64, x) - y);
        if (@as(u32, @bitCast(x)) == @as(u32, @bitCast(y))) f.exact += 1;
        if (!(d <= f.max_abs)) f = .{ .max_abs = d, .at = i, .bad = f.bad, .exact = f.exact };
        if (!(d <= rel * @abs(@as(f64, y)) + abs)) f.bad += 1;
    }
    return f;
}

/// A PLY file against the SDK's: headers, vertex counts, and each property column's largest difference.
pub const Ply = struct {
    header_equal: bool,
    rows_equal: bool,
    rows: usize,
    max_abs: [gs.properties.len]f64,
    worst_row: [gs.properties.len]usize,
    differing: [gs.properties.len]usize,
};

fn body(bytes: []const u8) ?[]const u8 {
    const end = "end_header\n";
    const at = std.mem.indexOf(u8, bytes, end) orelse return null;
    return bytes[at + end.len ..];
}

pub fn ply(ours: []const u8, theirs: []const u8) Ply {
    var p: Ply = .{ .header_equal = false, .rows_equal = false, .rows = 0, .max_abs = @splat(0), .worst_row = @splat(0), .differing = @splat(0) };
    const a = body(ours) orelse return p;
    const b = body(theirs) orelse return p;
    p.header_equal = std.mem.eql(u8, ours[0 .. ours.len - a.len], theirs[0 .. theirs.len - b.len]);
    p.rows_equal = a.len == b.len and a.len % gs.row_bytes == 0;
    if (!p.rows_equal) return p;
    p.rows = a.len / gs.row_bytes;
    for (0..p.rows) |r| for (0..gs.properties.len) |c| {
        const at = r * gs.row_bytes + c * 4;
        const x: f32 = @bitCast(std.mem.readInt(u32, a[at..][0..4], .little));
        const y: f32 = @bitCast(std.mem.readInt(u32, b[at..][0..4], .little));
        if (@as(u32, @bitCast(x)) == @as(u32, @bitCast(y))) continue;
        p.differing[c] += 1;
        const d = @abs(@as(f64, x) - y);
        if (!(d <= p.max_abs[c])) {
            p.max_abs[c] = d;
            p.worst_row[c] = r;
        }
    };
    return p;
}

/// The PLY gate: same header and rows; every column bit-equal but opacity and scales (device logs), within `tol`.
pub fn plyPass(p: Ply, tol: f64) bool {
    if (!p.header_equal or !p.rows_equal) return false;
    for (0..gs.properties.len) |c| {
        const transcendental = c >= 9 and c <= 12;
        if (transcendental) {
            if (!(p.max_abs[c] <= tol)) return false;
        } else if (p.differing[c] != 0) return false;
    }
    return true;
}

test "the reference takes the worst distance of each kind; a planted error past twice it still fails" {
    const lucky: ref.Match = .{ .max_abs = 1e-5, .cosine = 1 - 1e-9, .worst_row = 1e-4 };
    const nudged: ref.Match = .{ .max_abs = 2e-5, .cosine = 1 - 4e-9, .worst_row = 2e-4 };
    const floor: ref.Match = .{ .max_abs = 0, .cosine = 1 - 1e-10, .worst_row = 1e-5 };
    const r = reference(&.{ lucky, floor, nudged });
    try std.testing.expectEqual(nudged.worst_row, r.worst_row);
    try std.testing.expectEqual(nudged.cosine, r.cosine);
    try std.testing.expectEqual(lucky, reference(&.{lucky}));
    try std.testing.expect(ref.within(.{ .max_abs = 0, .cosine = 1 - 7e-9, .worst_row = 3.9e-4 }, r));
    try std.testing.expect(!ref.within(.{ .max_abs = 0, .cosine = 1 - 7e-9, .worst_row = 4.1e-4 }, r));
    try std.testing.expect(!ref.within(.{ .max_abs = 0, .cosine = 1 - 9e-9, .worst_row = 1e-4 }, r));
}

test "a planted count difference is found at its row" {
    const a = [_]u32{ 1, 0, 0, 0, 2, 0, 0, 0 };
    var b = a;
    try std.testing.expectEqual(Counts{ .rows = 0, .first = null }, counts(&a, &b, 4));
    b[5] = 1;
    b[4] = 1;
    try std.testing.expectEqual(Counts{ .rows = 1, .first = 1 }, counts(&a, &b, 4));
    try std.testing.expectEqual(@as(usize, 2), counts(&a, b[0..4], 4).rows);
}

test "the worst row and a field's planted error are found where they were planted" {
    var x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const y = x;
    x[4] = 5.5;
    try std.testing.expectEqual(@as(usize, 2), worst(&x, &y, 2).row);
    const f = field(&x, &y, 1e-5, 0);
    try std.testing.expectEqual(@as(usize, 4), f.at);
    try std.testing.expectEqual(@as(usize, 1), f.bad);
    try std.testing.expectEqual(@as(usize, 5), f.exact);
    x[1] = std.math.nan(f32);
    try std.testing.expectEqual(@as(usize, 2), field(&x, &y, 1e-5, 0).bad);
}

test "a PLY passes against itself; a planted header, row or quaternion bit fails" {
    const a = std.testing.allocator;
    var xyz = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    var f = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var s = [_]f32{ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06 };
    var q = [_]f32{ 1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5 };
    var o = [_]f32{ 0.2, 0.9 };
    const g: gs.Gaussians = .{ .xyz = &xyz, .features_dc = &f, .scaling = &s, .rotation = &q, .opacity = &o };
    const ours = try gs.ply(a, g);
    defer a.free(ours);
    try std.testing.expect(plyPass(ply(ours, ours), 1e-5));
    const theirs = try a.dupe(u8, ours);
    defer a.free(theirs);
    // the last row's rot_3, one bit off
    const at = theirs.len - 4;
    std.mem.writeInt(u32, theirs[at..][0..4], std.mem.readInt(u32, theirs[at..][0..4], .little) ^ 1, .little);
    const p = ply(ours, theirs);
    try std.testing.expect(!plyPass(p, 1e-5));
    try std.testing.expectEqual(@as(usize, 1), p.differing[16]);
    try std.testing.expectEqual(@as(usize, 1), p.worst_row[16]);
    try std.testing.expect(!plyPass(ply(ours, theirs[0 .. theirs.len - gs.row_bytes]), 1e-5));
    theirs[0] = 'P';
    try std.testing.expect(!plyPass(ply(ours, theirs), 1e-5));
}
