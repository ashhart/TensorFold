const std = @import("std");
const memory = @import("../pack/exl3_startup_memory.zig");
const layout = @import("../dplan/exl3_dplan_layout.zig");

pub const Plan = struct {
    pool_bytes: u64,
    planner_bytes: u64,
};

pub fn apply(text: []const u8, uniform_lines: usize, decoder_pools: usize, line_bytes: []const usize, min_lines: usize, planner_rows: ?u64, fixed_bytes: u64, startup_reservation: u64, counts: []usize) !Plan {
    if (counts.len != line_bytes.len or decoder_pools > counts.len or min_lines == 0) return error.InvalidLineDeltaIndex;
    if (uniform_lines < min_lines) return error.PoolTooSmall;
    @memset(counts, 0);
    var added: u64 = 0;
    var removed: u64 = 0;
    var entries = std.mem.splitScalar(u8, text, ',');
    while (entries.next()) |entry| {
        const colon = std.mem.indexOfScalar(u8, entry, ':') orelse return error.InvalidLineDeltas;
        const index = try decimal(entry[0..colon]);
        if (index >= decoder_pools) return error.InvalidLineDeltaIndex;
        if (counts[index] != 0) return error.DuplicateLineDelta;
        const delta = entry[colon + 1 ..];
        if (delta.len < 2 or (delta[0] != '+' and delta[0] != '-')) return error.InvalidLineDeltas;
        const magnitude = try decimal(delta[1..]);
        const result = if (delta[0] == '+') result: {
            added = try std.math.add(u64, added, magnitude);
            break :result try std.math.add(usize, uniform_lines, magnitude);
        } else result: {
            removed = try std.math.add(u64, removed, magnitude);
            if (magnitude > uniform_lines) return error.PoolTooSmall;
            break :result uniform_lines - magnitude;
        };
        if (result < min_lines) return error.PoolTooSmall;
        counts[index] = result;
    }
    if (added != removed) return error.NonZeroLineDeltas;
    var before: u64 = 0;
    var after: u64 = 0;
    var expert_bytes: u64 = 0;
    for (counts, line_bytes) |*count, bytes| {
        if (count.* == 0) count.* = uniform_lines;
        _ = try layout.PoolState.init(count.*, 0);
        const cost = try std.math.add(u64, bytes, 8);
        before = try std.math.add(u64, before, try std.math.mul(u64, uniform_lines, cost));
        after = try std.math.add(u64, after, try std.math.mul(u64, count.*, cost));
        expert_bytes = try std.math.add(u64, expert_bytes, try std.math.mul(u64, count.*, bytes));
    }
    if (before != after) return error.LineDeltaByteBudgetChanged;
    const planner_bytes = if (planner_rows) |rows| try memory.plannerBytesForPools(counts, rows) else 0;
    const pool_bytes = try std.math.add(u64, try std.math.add(u64, expert_bytes, fixed_bytes), planner_bytes);
    if (pool_bytes > startup_reservation) return error.InsufficientStartupMemory;
    return .{ .pool_bytes = pool_bytes, .planner_bytes = planner_bytes };
}

fn decimal(text: []const u8) !usize {
    if (text.len == 0) return error.InvalidLineDeltas;
    for (text) |digit| if (digit < '0' or digit > '9') return error.InvalidLineDeltas;
    return std.fmt.parseInt(usize, text, 10);
}

test "exl3: line deltas parse signed decoder ordinals and leave MTP unchanged" {
    var counts: [22]usize = undefined;
    const bytes: [22]usize = @splat(256);
    const plan = try apply("20:+1,15:-1", 32, 21, &bytes, 10, 16, 100, std.math.maxInt(u64), &counts);
    for (counts, 0..) |count, index| try std.testing.expectEqual(@as(usize, if (index == 20) 33 else if (index == 15) 31 else 32), count);
    try std.testing.expectEqual(try memory.plannerBytes(22, 32, 16), plan.planner_bytes);
    try std.testing.expectEqual(22 * 32 * 256 + 100 + plan.planner_bytes, plan.pool_bytes);
    try std.testing.expectEqual(@as(usize, 32), counts[21]);
}

test "exl3: line deltas reject malformed empty duplicate and out of range maps" {
    var counts: [3]usize = undefined;
    const bytes = [_]usize{ 256, 256, 512 };
    for ([_][]const u8{ "", "0", "0:1", "0:+", ":+1", "0:++1", "0:+1,", ",0:+1", "0:+1,,1:-1", " 0:+1,1:-1", "0:+1, 1:-1", "0:+1 ,1:-1", "-1:+1", "+0:+1", "0x0:+1", "0:+1_0", "0:+1:0" }) |text| {
        try std.testing.expectError(error.InvalidLineDeltas, apply(text, 32, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    }
    for ([_][]const u8{ "0:+1,0:-1", "0:+0,0:-0", "00:+0,0:+0" }) |text| {
        try std.testing.expectError(error.DuplicateLineDelta, apply(text, 32, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    }
    for ([_][]const u8{ "2:+0", "3:+0", "99:+0" }) |text| {
        try std.testing.expectError(error.InvalidLineDeltaIndex, apply(text, 32, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    }
    try std.testing.expectError(error.InvalidLineDeltaIndex, apply("0:+0", 32, 0, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
}

test "exl3: line deltas conserve lines and actual bytes independently" {
    var counts: [4]usize = undefined;
    const bytes = [_]usize{ 256, 512, 256, 1024 };
    for ([_][]const u8{ "0:+1", "0:-1", "0:+2,2:-1" }) |text| {
        try std.testing.expectError(error.NonZeroLineDeltas, apply(text, 32, 3, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    }
    for ([_][]const u8{ "0:+1,1:-1", "0:-1,1:+1" }) |text| {
        try std.testing.expectError(error.LineDeltaByteBudgetChanged, apply(text, 32, 3, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    }
    const expected_counts = [_]usize{ 33, 32, 31, 32 };
    const planner = try memory.plannerBytes(4, 32, 16);
    const expected_bytes = 32 * (256 + 512 + 256 + 1024) + planner + 100;
    const plan = try apply("0:+1,2:-1", 32, 3, &bytes, 10, 16, 100, expected_bytes, &counts);
    try std.testing.expectEqualSlices(usize, &expected_counts, &counts);
    try std.testing.expectEqual(expected_bytes, plan.pool_bytes);
    try std.testing.expectEqual(planner, plan.planner_bytes);
    try std.testing.expectError(error.InsufficientStartupMemory, apply("0:+1,2:-1", 32, 3, &bytes, 10, 16, 100, expected_bytes - 1, &counts));
    const without_planner = try apply("0:+1,2:-1", 32, 3, &bytes, 10, null, 100, expected_bytes - planner, &counts);
    try std.testing.expectEqual(@as(u64, 0), without_planner.planner_bytes);
    try std.testing.expectEqual(expected_bytes - planner, without_planner.pool_bytes);
}

test "exl3: line deltas enforce floor before allocation and reject arithmetic overflow" {
    var counts: [2]usize = undefined;
    const bytes = [_]usize{ 256, 256 };
    _ = try apply("0:+22,1:-22", 32, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts);
    try std.testing.expectEqualSlices(usize, &.{ 54, 10 }, &counts);
    for ([_][]const u8{ "0:+23,1:-23", "0:+32,1:-32", "0:+33,1:-33" }) |text| {
        try std.testing.expectError(error.PoolTooSmall, apply(text, 32, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    }
    try std.testing.expectError(error.PoolTooSmall, apply("0:+0", 9, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    try std.testing.expectError(error.Overflow, apply("0:+18446744073709551615,1:-18446744073709551615", 32, 2, &bytes, 10, 16, 0, std.math.maxInt(u64), &counts));
    try std.testing.expectError(error.Overflow, apply("0:+1,1:-1", 32, 2, &.{ std.math.maxInt(usize), std.math.maxInt(usize) }, 10, 16, 0, std.math.maxInt(u64), &counts));
}

test "exl3: line deltas allow byte neutral multiway transfers within conservative startup sizing" {
    var counts: [4]usize = undefined;
    const bytes = [_]usize{ 256, 512, 768, 1024 };
    const fixed: u64 = 100;
    const reservation = try memory.poolBytes(32, 1024 + 8, 4, fixed + try memory.plannerBytes(4, 0, 16));
    const plan = try apply("0:+1,1:-2,2:+1", 32, 3, &bytes, 10, 16, fixed, reservation, &counts);
    try std.testing.expectEqualSlices(usize, &.{ 33, 30, 33, 32 }, &counts);
    try std.testing.expect(plan.pool_bytes <= reservation);
    try std.testing.expectEqual(@as(u64, 32 * (256 + 512 + 768 + 1024) + fixed) + try memory.plannerBytes(4, 32, 16), plan.pool_bytes);
    _ = try apply("0:+0", 32, 3, &bytes, 10, 16, fixed, reservation, &counts);
    try std.testing.expectEqualSlices(usize, &.{ 32, 32, 32, 32 }, &counts);
}
