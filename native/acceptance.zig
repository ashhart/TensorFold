//! Shared acceptance policy for chained MTP and branched DFlash verification.
//! Only target-selected tokens are emitted; accepted rows form an ancestor path.
const std = @import("std");
pub const Result = struct {
    tokens: [32]u32 = undefined,
    count: usize = 0,
    path: [32]i32 = undefined,
    kept: usize = 1,
    accepted: usize = 0,
    pending: i32 = 0,
    stop: bool = false,
};
/// Convert a queued chain to its logical prefix. Evaluated GPU rows after the
/// first proposed EOS must never enter acceptance, hashing or cache commits.
pub fn copyChain(destination: []i32, proposals: []const u32, comptime is_eos: fn (i32) bool) !usize {
    if (proposals.len > destination.len) return error.InvalidVerification;
    for (proposals, 0..) |token, i| {
        if (token > std.math.maxInt(i32)) return error.InvalidToken;
        destination[i] = @intCast(token);
        if (is_eos(destination[i])) return i + 1;
    }
    return proposals.len;
}
pub fn select(tokens: []const i32, parents: []const i32, samples: []const i32, remaining: usize, comptime is_eos: fn (i32) bool) !Result {
    if (tokens.len == 0 or tokens.len > 32 or samples.len != tokens.len or parents.len != tokens.len or remaining == 0) return error.InvalidVerification;
    _ = try @import("lanes.zig").Tree.init(parents);
    var out = Result{};
    out.path[0] = 0;
    var row: usize = 0;
    while (true) {
        const want = samples[row];
        if (want < 0) return error.InvalidToken;
        out.tokens[out.count] = @intCast(want);
        out.count += 1;
        out.pending = want;
        var child: ?usize = null;
        for (1..tokens.len) |j| if (parents[j] == row and tokens[j] == want) {
            child = j;
            break;
        };
        if (child) |j| {
            out.path[out.kept] = @intCast(j);
            out.kept += 1;
            out.accepted += 1;
            row = j;
        }
        out.stop = is_eos(want) or out.count == remaining;
        if (out.stop or child == null) return out;
    }
}
fn eos(id: i32) bool {
    return id == 2 or id == 11;
}
test "GPU chain suffix after EOS cannot affect logical acceptance" {
    var window: [16]i32 = undefined;
    window[0] = 100;
    var proposals: [15]u32 = undefined;
    var parents: [16]i32 = undefined;
    var samples: [16]i32 = undefined;
    for (0..16) |i| parents[i] = @as(i32, @intCast(i)) - 1;
    for ([_]u32{ 2, 11 }) |stop| for (0..15) |at| {
        for (&proposals, 0..) |*token, i| token.* = @intCast(101 + i);
        proposals[at] = stop;
        // An invalid unreachable suffix is deliberately ignored.
        if (at + 1 < proposals.len) proposals[at + 1] = std.math.maxInt(u32);
        const n = 1 + try copyChain(window[1..], &proposals, eos);
        try std.testing.expectEqual(at + 2, n);
        for (0..n - 1) |i| samples[i] = window[i + 1];
        samples[n - 1] = 999;
        const result = try select(window[0..n], parents[0..n], samples[0..n], 32, eos);
        try std.testing.expectEqual(at + 1, result.count);
        try std.testing.expectEqual(n, result.kept);
        try std.testing.expectEqual(stop, result.tokens[result.count - 1]);
        try std.testing.expect(result.stop);
        const bounded = try select(window[0..n], parents[0..n], samples[0..n], 1, eos);
        try std.testing.expectEqual(@as(usize, 1), bounded.count);
        try std.testing.expectEqual(@as(usize, 2), bounded.kept);
        try std.testing.expect(bounded.stop);
        samples[0] = 999;
        const rejected = try select(window[0..n], parents[0..n], samples[0..n], 32, eos);
        try std.testing.expectEqual(@as(usize, 1), rejected.kept);
        try std.testing.expectEqual(@as(i32, 999), rejected.pending);
        try std.testing.expect(!rejected.stop);
    };
    try std.testing.expectError(error.InvalidToken, copyChain(window[1..], &.{std.math.maxInt(u32)}, eos));
    try std.testing.expectError(error.InvalidVerification, copyChain(window[1..2], &.{ 101, 102 }, eos));
    try std.testing.expectEqual(@as(usize, 0), try copyChain(window[1..], &.{}, eos));
}
test "all acceptance lengths and output budgets including maximum chain" {
    var tokens: [32]i32 = undefined;
    var parents: [32]i32 = undefined;
    var samples: [32]i32 = undefined;
    for (0..32) |i| {
        tokens[i] = @intCast(100 + i);
        parents[i] = @as(i32, @intCast(i)) - 1;
    }
    for (0..32) |accept| for (1..34) |budget| {
        for (&samples, 0..) |*sample, i| sample.* = if (i < accept) tokens[i + 1] else 999;
        const out = try select(&tokens, &parents, &samples, budget, eos);
        try std.testing.expectEqual(@min(accept + 1, budget), out.count);
        try std.testing.expectEqual(@min(accept, budget), out.accepted);
        try std.testing.expectEqual(out.accepted + 1, out.kept);
        for (out.tokens[0..out.count], 0..) |token, i| try std.testing.expectEqual(@as(u32, @intCast(if (i < accept) tokens[i + 1] else 999)), token);
    };
}
test "accepted EOS, rejected EOS, bonus EOS and token budget terminate exactly" {
    const accepted = try select(&.{ 100, 101, 2, 103 }, &.{ -1, 0, 1, 2 }, &.{ 101, 2, 103, 999 }, 10, eos);
    try std.testing.expectEqualSlices(u32, &.{ 101, 2 }, accepted.tokens[0..accepted.count]);
    try std.testing.expectEqual(@as(usize, 3), accepted.kept);
    try std.testing.expect(accepted.stop);
    const rejected = try select(&.{ 100, 2, 103 }, &.{ -1, 0, 1 }, &.{ 999, 103, 104 }, 10, eos);
    try std.testing.expectEqualSlices(u32, &.{999}, rejected.tokens[0..rejected.count]);
    try std.testing.expect(!rejected.stop);
    const bonus = try select(&.{ 100, 101 }, &.{ -1, 0 }, &.{ 101, 11 }, 10, eos);
    try std.testing.expectEqualSlices(u32, &.{ 101, 11 }, bonus.tokens[0..bonus.count]);
    try std.testing.expect(bonus.stop);
    try std.testing.expectError(error.InvalidVerification, select(&.{100}, &.{-1}, &.{2}, 0, eos));
}
test "random trees only accept connected target-selected paths" {
    var rng = std.Random.DefaultPrng.init(0x53454544);
    const random = rng.random();
    var tokens: [32]i32 = undefined;
    var parents: [32]i32 = undefined;
    var samples: [32]i32 = undefined;
    for (0..1000) |_| {
        const count = random.intRangeAtMost(usize, 1, 32);
        for (0..count) |i| {
            tokens[i] = @intCast(100 + i);
            parents[i] = if (i == 0) -1 else @intCast(random.uintLessThan(usize, i));
            samples[i] = 999;
        }
        for (1..count) |i| if (random.boolean()) {
            samples[@intCast(parents[i])] = tokens[i];
        };
        const budget = random.intRangeAtMost(usize, 1, 32);
        const out = try select(tokens[0..count], parents[0..count], samples[0..count], budget, eos);
        try std.testing.expect(out.count <= budget);
        try std.testing.expectEqual(@as(i32, 0), out.path[0]);
        for (1..out.kept) |i| {
            const row: usize = @intCast(out.path[i]);
            try std.testing.expectEqual(out.path[i - 1], parents[row]);
            try std.testing.expectEqual(samples[@intCast(out.path[i - 1])], tokens[row]);
            try std.testing.expectEqual(@as(u32, @intCast(tokens[row])), out.tokens[i - 1]);
        }
    }
}
