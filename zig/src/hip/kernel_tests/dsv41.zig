//! DeepSeek-V4.1's indexer top-512 on HIP: a stable rank of each row against a sort, at every width the indexer serves.
const std = @import("std");
const rig = @import("rig.zig");
const dsv41 = @import("../launches/dsv41.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

fn rank(v: f32) f32 {
    return if (std.math.isNan(v)) -std.math.inf(f32) else v;
}

/// A row's selection: the min(512, n) largest of its first n = clamp(vis, 0, width) by a stable rank, ascending, -1s.
fn expected(row: []const f32, vis: i32, out: *[dsv41.keep]i32) !void {
    const n: usize = @min(@as(usize, @intCast(@max(vis, 0))), row.len);
    const idx = try gpa.alloc(u32, n);
    defer gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(i);
    const Order = struct {
        fn before(r: []const f32, a: u32, b: u32) bool {
            const x = rank(r[a]);
            const y = rank(r[b]);
            return x > y or (x == y and a < b);
        }
    };
    std.mem.sort(u32, idx, row, Order.before);
    const kept = idx[0..@min(n, dsv41.keep)];
    std.mem.sort(u32, kept, {}, std.sort.asc(u32));
    @memset(out, -1);
    for (kept, 0..) |v, i| out[i] = @intCast(v);
}

/// What a row holds: 0 spread values, 1 seven values (ties everywhere), 2 a third -inf (masked), 3 NaN, +-0, +-inf.
fn fill(rng: *rig.Rng, kind: u32, row: []f32) void {
    for (row) |*v| v.* = switch (kind) {
        0 => rng.unit() * 8 - 4,
        1 => @as(f32, @floatFromInt(rng.next() % 7)) - 3,
        2 => if (rng.next() % 3 == 0) -std.math.inf(f32) else rng.unit() * 2 - 1,
        else => switch (rng.next() % 9) {
            0 => std.math.nan(f32),
            1 => -0.0,
            2 => 0.0,
            3 => std.math.inf(f32),
            4 => -std.math.inf(f32),
            else => @floatFromInt(rng.next() % 5),
        },
    };
}

/// Row r's visible count: whole, none, half and one, 512, 513, negative, past the width, then random.
fn visible(rng: *rig.Rng, r: usize, width: usize) i32 {
    const w: i32 = @intCast(width);
    return switch (r) {
        0 => w,
        1 => 0,
        2 => @divTrunc(w, 2) + 1,
        3 => @min(w, 512),
        4 => @min(w, 513),
        5 => -3,
        6 => w + 5,
        else => @intCast(rng.next() % (width + 1)),
    };
}

fn run(t: *Rig, scores: []const f32, vis: []const i32, width: usize, fault: c_int) ![]i32 {
    var ds = try t.upload(scores);
    defer ds.free();
    var dv = try t.upload(vis);
    defer dv.free();
    var dout = try t.alloc(vis.len * dsv41.keep * 4);
    defer dout.free();
    try dout.fill8(0x7f);
    try dsv41.topk512(&t.on, at(ds), at(dv), at(dout), vis.len, width, fault, t.stream.handle);
    try t.stream.synchronize();
    return rig.download(i32, dout);
}

test "top-512 keeps each row's largest by a stable rank, indices ascending then -1, at every width the indexer serves" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xD541 };
    const widths = [_]usize{ 1, 128, 511, 512, 513, 1024, 4096, 16384, 131072 };
    for (widths) |width| for (0..4) |kind| {
        const rows: usize = if (width >= 131072) 3 else 16;
        const scores = try gpa.alloc(f32, rows * width);
        defer gpa.free(scores);
        const vis = try gpa.alloc(i32, rows);
        defer gpa.free(vis);
        for (0..rows) |r| {
            fill(&rng, @intCast(kind), scores[r * width ..][0..width]);
            vis[r] = visible(&rng, r, width);
        }
        const got = try run(t, scores, vis, width, 0);
        defer gpa.free(got);
        for (0..rows) |r| {
            var want: [dsv41.keep]i32 = undefined;
            try expected(scores[r * width ..][0..width], vis[r], &want);
            const have = got[r * dsv41.keep ..][0..dsv41.keep];
            if (!std.mem.eql(i32, &want, have)) {
                const i = std.mem.indexOfDiff(i32, &want, have).?;
                std.debug.print("width {d} kind {d} row {d} (visible {d}): slot {d} is {d} where {d}\n", .{ width, kind, r, vis[r], i, have[i], want[i] });
                return error.TestUnexpectedResult;
            }
        }
    };
}

test "a row's top-512 is the same alone as among other rows, and the scores are left as they were" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x512 };
    const rows = 9;
    const width = 4096;
    const scores = try gpa.alloc(f32, rows * width);
    defer gpa.free(scores);
    const vis = try gpa.alloc(i32, rows);
    defer gpa.free(vis);
    for (0..rows) |r| {
        fill(&rng, @intCast(r % 4), scores[r * width ..][0..width]);
        vis[r] = visible(&rng, r, width);
    }
    var ds = try t.upload(scores);
    defer ds.free();
    var dv = try t.upload(vis);
    defer dv.free();
    var dall = try t.alloc(rows * dsv41.keep * 4);
    defer dall.free();
    try dsv41.topk512(&t.on, at(ds), at(dv), at(dall), rows, width, 0, t.stream.handle);
    try t.stream.synchronize();
    const all = try rig.download(i32, dall);
    defer gpa.free(all);
    const after = try rig.download(f32, ds);
    defer gpa.free(after);
    try std.testing.expectEqualSlices(u32, @ptrCast(scores), @ptrCast(after));
    for (0..rows) |r| {
        const got = try run(t, scores[r * width ..][0..width], vis[r .. r + 1], width, 0);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(i32, all[r * dsv41.keep ..][0..dsv41.keep], got);
    }
}
