//! Distributed raw head snapshots against an independently sorted, float64 probability reference.
const std = @import("std");
const tp = @import("tp");
const lp = @import("logprob_pipe.zig");
const sampling = @import("sampling_pipe.zig");
const partition = @import("partition.zig");
const Row = @import("lanes").logprob.Row;
const a = std.testing.allocator;

pub fn reference(values: []const f32, token: u32, k: u8) !Row {
    const ids = try a.alloc(u32, values.len);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i);
    std.sort.heap(u32, ids, values, struct {
        fn less(v: []const f32, x: u32, y: u32) bool {
            return v[x] > v[y] or (v[x] == v[y] and x < y);
        }
    }.less);
    const max: f64 = values[ids[0]];
    var total: f64 = 0;
    for (ids) |id| total += @exp(@as(f64, values[id]) - max);
    var out: Row = .{ .token = token, .logprob = @floatCast(@log(@exp(@as(f64, values[token]) - max) / total)), .count = @intCast(@min(values.len, k)) };
    for (values) |v| out.rank += @intFromBool(v >= values[token]);
    for (ids[0..out.count], out.ids[0..out.count], out.logprobs[0..out.count]) |id, *o, *p| {
        o.* = id;
        p.* = @floatCast(@log(@exp(@as(f64, values[id]) - max) / total));
    }
    return out;
}

pub fn equal(expected: Row, actual: Row) !void {
    try std.testing.expectEqual(expected.token, actual.token);
    try std.testing.expectEqual(expected.count, actual.count);
    try std.testing.expectEqual(expected.rank, actual.rank);
    if (std.math.isInf(expected.logprob)) try std.testing.expectEqual(expected.logprob, actual.logprob) else try std.testing.expectApproxEqAbs(expected.logprob, actual.logprob, 0.000002);
    try std.testing.expectEqualSlices(u32, expected.ids[0..expected.count], actual.ids[0..actual.count]);
    for (expected.logprobs[0..expected.count], actual.logprobs[0..actual.count]) |e, v| {
        if (std.math.isInf(e)) try std.testing.expectEqual(e, v) else try std.testing.expectApproxEqAbs(e, v, 0.000002);
    }
}

const Case = struct { values: []const f32, rows: usize, world: u32, planted: bool = false };

fn rankBody(comm: tp.collective.Collective, rank: u32, c: *Case) !void {
    const vocab: u32 = @intCast(c.values.len / c.rows);
    const part = try partition.share(vocab, rank, c.world);
    const own = try a.alloc(f32, c.rows * part.len);
    defer a.free(own);
    for (0..c.rows) |r| @memcpy(own[r * part.len ..][0..part.len], c.values[r * vocab + part.lo ..][0..part.len]);
    var host: sampling.HostDevice = .{ .gpa = a };
    defer host.deinit();
    const hd = host.device();
    const dev: lp.Device = .{ .ptr = hd.ptr, .vtable = &.{ .scratch = hd.vtable.scratch, .pad = hd.vtable.pad.?, .download = hd.vtable.download, .stream = hd.vtable.stream } };
    var capture: lp.Capture = .{};
    defer capture.deinit(a);
    try capture.gather(a, comm, dev, @intFromPtr(own.ptr), c.rows, vocab);
    if (rank != 0) {
        try std.testing.expectEqual(@as(usize, 0), capture.values.items.len);
        return;
    }
    try std.testing.expectEqualSlices(f32, c.values, capture.values.items);
    for (0..c.rows) |r| for ([_]u8{ 0, 1, 20 }) |k| for ([_]u32{ 0, part.len - 1, @min(part.len, vocab - 1), vocab - 1 }) |token| {
        const values = c.values[r * vocab ..][0..vocab];
        const expected = try reference(values, token, k);
        try equal(expected, try capture.row(r, token, k));
    };
    // A shard-local normalizer and a lost global ID must fail this independent check.
    if (c.planted) {
        const full = c.values[0..vocab];
        const expected = try reference(full, 0, 20);
        const wrong = try lp.score(full[0..part.len], 0, 20);
        try std.testing.expect(@abs(wrong.logprob - expected.logprob) > 0.1);
        const right = try capture.row(0, vocab - 1, 20);
        try std.testing.expect(right.ids[0] >= part.len);
    }
}

test "DeepSeek logprobs: gathered TP2 rows, uneven TP3/4 and top-k match the independent float64 distribution" {
    const vocab = 1280;
    const values = try a.alloc(f32, 4 * vocab);
    defer a.free(values);
    for (0..4) |r| {
        for (values[r * vocab ..][0..vocab], 0..) |*v, i|
            v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + r * 53) % 100)))) / 16 - 3;
    }
    values[0] = 10;
    values[vocab - 1] = 12;
    values[vocab + 639] = 20;
    values[vocab + 640] = 20;
    for (values[2 * vocab .. 3 * vocab]) |*v| v.* += 10000;
    @memset(values[3 * vocab ..][0 .. vocab / 2], -std.math.inf(f32));
    for ([_]u32{ 1, 2, 3, 4 }) |world| {
        var c: Case = .{ .values = values, .rows = 4, .world = world, .planted = world == 2 };
        try tp.host.run(world, &c, rankBody);
    }
}

test "DeepSeek logprobs: large offsets, minus infinity, ties and malformed head rows" {
    const v = [_]f32{ -std.math.inf(f32), 1e20, 1e20, 1e20 };
    const row = try lp.score(&v, 2, 20);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(-@log(@as(f64, 3)))), row.logprob, 0.000001);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 0 }, row.ids[0..row.count]);
    try std.testing.expectEqual(@as(u32, 3), row.rank);
    try std.testing.expectError(error.NonFiniteLogit, lp.score(&.{ std.math.nan(f32), 0 }, 0, 1));
    try std.testing.expectError(error.NonFiniteLogit, lp.score(&.{ std.math.inf(f32), 0 }, 0, 1));
    try std.testing.expectError(error.NoFiniteLogit, lp.score(&.{-std.math.inf(f32)}, 0, 1));
    try std.testing.expectError(error.LogprobRow, lp.score(&.{0}, 1, 1));
    try std.testing.expectError(error.LogprobRow, lp.score(&.{0}, 0, 21));
}
