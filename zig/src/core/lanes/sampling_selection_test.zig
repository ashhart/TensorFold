//! Compare selection with the original stable sort before testing the unchanged keyed draw.
const std = @import("std");
const sampling = @import("sampling.zig");
const selection = @import("sampling_selection.zig");
const reference = @import("sampling_reference.zig");
const t = std.testing;

fn sameCandidates(want: []const sampling.Candidate, got: []const sampling.Candidate) !void {
    try t.expectEqual(want.len, got.len);
    for (want, got) |a, b| {
        try t.expectEqual(a.id, b.id);
        try t.expectEqual(@as(u64, @bitCast(a.value)), @as(u64, @bitCast(b.value)));
    }
}

fn checkSelection(values: []const f64, ids: []const u64) !void {
    const full = try reference.ordered(t.allocator, values, ids);
    defer t.allocator.free(full);
    try t.expectEqual(full[0].id, selection.greedy(values, ids).?);
    for (1..values.len + 1) |k| {
        const selected = (try selection.topK(t.allocator, values, ids, k)).?;
        defer t.allocator.free(selected);
        try sameCandidates(full[0..k], selected);
    }
}

test "host sampling: top-k candidates equal every full-sort prefix bit for bit" {
    const values = [_]f64{ -0.0, 0.0, -0.0, 0.0, 9, 9, -5, 9, 9, 1e300, -1e300, 0x1p-1074, -0x1p-1074 };
    const ids = [_]u64{ 7, 7, 1, 1, 5, 5, 1, 0, 11, 80, 81, 82, 83 };
    try checkSelection(&values, &ids);
    var increasing: [65]f64 = undefined;
    var decreasing: [65]f64 = undefined;
    var equal: [65]f64 = @splat(0.0);
    var keys: [65]u64 = undefined;
    for (&increasing, &decreasing, &equal, &keys, 0..) |*up, *down, *flat, *id, i| {
        up.* = @floatFromInt(i);
        down.* = @floatFromInt(65 - i);
        flat.* = if (i % 2 == 0) -0.0 else 0.0;
        id.* = (i * 17) % 13;
    }
    try checkSelection(&increasing, &keys);
    try checkSelection(&decreasing, &keys);
    try checkSelection(&equal, &keys);
    try checkSelection(&.{-0.0}, &.{std.math.maxInt(u64)});
}

fn checkDraws(values: []const f64, ids: []const u64) !void {
    const temperatures = [_]f64{ -1, -0.0, 0.0, 1e-10, 0.7, 1, 2, std.math.inf(f64), std.math.nan(f64) };
    const filters = [_][2]f64{ .{ 0, 0 }, .{ 1, 0 }, .{ 0.95, 0 }, .{ 0.5, 0.1 }, .{ 0.01, 1 }, .{ 0.9, 2 } };
    const ks = [_]u32{ 0, 1, 2, 7, 20, @intCast(values.len / 2), @intCast(values.len), std.math.maxInt(u32) };
    for (temperatures) |temperature| for (filters) |filter| for (ks) |k| {
        for ([_]u64{ 0, 7, std.math.maxInt(u64) }) |seed| for ([_]u64{ 0, 65, std.math.maxInt(u64) }) |position| {
            const s: sampling.Sampling = .{
                .seed = seed,
                .temperature = temperature,
                .top_k = k,
                .top_p = filter[0],
                .min_p = filter[1],
            };
            try t.expectEqual(try reference.choose(t.allocator, values, ids, position, s), try sampling.choose(t.allocator, values, ids, position, s));
        };
    };
}

test "host sampling: keyed draws preserve filtering, seeds, positions and nonfinite settings" {
    var values: [65]f64 = undefined;
    var ids: [65]u64 = undefined;
    for (&values, &ids, 0..) |*v, *id, i| {
        v.* = @as(f64, @floatFromInt((i * 37) % 23)) / 8 - 2;
        id.* = (i * 61) % 65;
    }
    values[0] = -0.0;
    values[1] = 0.0;
    values[2] = -0x1p-1074;
    values[3] = 0x1p-1074;
    const unchanged_values = values;
    const unchanged_ids = ids;
    try checkDraws(&values, &ids);
    try t.expectEqualSlices(u8, std.mem.asBytes(&unchanged_values), std.mem.asBytes(&values));
    try t.expectEqualSlices(u64, &unchanged_ids, &ids);
    try checkDraws(&.{ 1e300, -1e300, 0.0, -0.0 }, &.{ 3, 2, 1, 1 });
}

test "host sampling: randomized heap boundaries preserve ordered candidates and drawn ids" {
    var values: [257]f64 = undefined;
    var ids: [257]u64 = undefined;
    var bits: u64 = 12345;
    for (0..120) |trial| {
        const n = 1 + trial * 71 % values.len;
        for (values[0..n], ids[0..n]) |*value, *id| {
            bits = sampling.mix(bits +% 1);
            value.* = @as(f64, @floatFromInt(bits % 131)) / 16 - 4;
            id.* = (bits >> 16) % 97;
        }
        const full = try reference.ordered(t.allocator, values[0..n], ids[0..n]);
        defer t.allocator.free(full);
        for ([_]usize{ 1, @max(1, n / 2), @max(1, n - 1), n }) |k| {
            const selected = (try selection.topK(t.allocator, values[0..n], ids[0..n], k)).?;
            defer t.allocator.free(selected);
            try sameCandidates(full[0..k], selected);
            const s: sampling.Sampling = .{ .seed = bits, .top_k = @intCast(k), .temperature = 0.65, .top_p = 0.87, .min_p = 0.15 };
            try t.expectEqual(try reference.choose(t.allocator, values[0..n], ids[0..n], trial, s), try sampling.choose(t.allocator, values[0..n], ids[0..n], trial, s));
        }
    }
}

test "host sampling: NaNs and infinities retain the original full-sort behavior" {
    const nan: f64 = @bitCast(@as(u64, 0x7ff8000000000042));
    const cases = [_][8]f64{
        .{ nan, 3, 2, 1, 0, -1, -2, -3 },
        .{ 3, 2, 1, nan, 0, -1, -2, -3 },
        .{ 3, 2, 1, 0, -1, -2, -3, nan },
        .{ std.math.inf(f64), 2, 1, 0, -1, -2, -3, -std.math.inf(f64) },
        @splat(-std.math.inf(f64)),
        @splat(nan),
    };
    const ids = [_]u64{ 8, 4, 7, 0, 6, 1, 5, 2 };
    for (cases) |values| {
        try t.expectEqual(@as(?u64, null), selection.greedy(&values, &ids));
        try t.expectEqual(@as(?[]selection.Candidate, null), try selection.topK(t.allocator, &values, &ids, 2));
        try checkDraws(&values, &ids);
    }
}

fn allocatingDraw(gpa: std.mem.Allocator, s: sampling.Sampling, nonfinite: bool) !void {
    const values = [_]f64{ 0, 9, 8, -2, 7, 6, 5, if (nonfinite) std.math.nan(f64) else 4 };
    const ids = [_]u64{ 8, 7, 6, 5, 4, 3, 2, 1 };
    _ = try sampling.choose(gpa, &values, &ids, 32, s);
}

test "host sampling: greedy needs no allocations and allocation failures never leak" {
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    try t.expectEqual(@as(u64, 2), try sampling.choose(failing.allocator(), &.{ 4, 4, 0 }, &.{ 9, 2, 1 }, 0, .{ .seed = 0, .temperature = 0 }));
    try t.expectEqual(@as(usize, 0), failing.allocations);
    try t.expect(!failing.has_induced_failure);
    try t.expectError(error.OutOfMemory, reference.choose(failing.allocator(), &.{ 4, 4, 0 }, &.{ 9, 2, 1 }, 0, .{ .seed = 0, .temperature = 0 }));
    try t.checkAllAllocationFailures(t.allocator, allocatingDraw, .{ sampling.Sampling{ .seed = 7, .top_k = 2 }, false });
    try t.checkAllAllocationFailures(t.allocator, allocatingDraw, .{ sampling.Sampling{ .seed = 7, .top_k = 0 }, false });
    try t.checkAllAllocationFailures(t.allocator, allocatingDraw, .{ sampling.Sampling{ .seed = 7, .top_k = 2 }, true });
}

test "host sampling: bounded top-k storage depends on k instead of vocabulary width" {
    var values: [4096]f64 = undefined;
    var ids: [4096]u64 = undefined;
    for (&values, &ids, 0..) |*value, *id, i| {
        value.* = @floatFromInt((i * 137) % 4096);
        id.* = i;
    }
    var tracked = std.testing.FailingAllocator.init(t.allocator, .{});
    _ = try sampling.choose(tracked.allocator(), &values, &ids, 0, .{ .seed = 1, .top_k = 20 });
    try t.expectEqual(@as(usize, 20 * (@sizeOf(usize) + @sizeOf(sampling.Candidate) + 2 * @sizeOf(f64))), tracked.allocated_bytes);
    try t.expectEqual(tracked.allocated_bytes, tracked.freed_bytes);
}

test "host sampling: empty and mismatched rows return explicit errors before allocation" {
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const gpa = failing.allocator();
    const s: sampling.Sampling = .{ .seed = 0 };
    try t.expectError(error.NoAllowedToken, sampling.choose(gpa, &.{}, &.{}, 0, s));
    try t.expectError(error.NoAllowedToken, sampling.chooseOrdered(gpa, &.{}, 0, s));
    try t.expectError(error.MismatchedSamplingInputs, sampling.choose(gpa, &.{1}, &.{}, 0, s));
    try t.expectError(error.MismatchedSamplingInputs, sampling.choose(gpa, &.{}, &.{1}, 0, s));
    try t.expect(!failing.has_induced_failure);
}
