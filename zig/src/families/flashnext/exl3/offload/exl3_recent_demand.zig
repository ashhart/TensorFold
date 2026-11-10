const std = @import("std");

pub const expert_count: usize = 512;
pub const Count = u32;
const Row = [expert_count / @bitSizeOf(u64)]u64;

pub const RecentDemand = struct {
    history: []Row,
    counts: [expert_count]Count = @splat(0),
    next_row: usize = 0,
    retained_rows: usize = 0,

    pub fn init(allocator: std.mem.Allocator, window_rows: usize) !RecentDemand {
        if (window_rows == 0 or window_rows > std.math.maxInt(Count)) return error.InvalidWindowSize;
        const history = try allocator.alloc(Row, window_rows);
        @memset(history, @splat(0));
        return .{ .history = history };
    }

    pub fn deinit(self: *RecentDemand, allocator: std.mem.Allocator) void {
        allocator.free(self.history);
        self.* = undefined;
    }

    pub fn reset(self: *RecentDemand) void {
        @memset(self.history, @splat(0));
        self.counts = @splat(0);
        self.next_row = 0;
        self.retained_rows = 0;
    }

    pub fn windowRows(self: *const RecentDemand) usize {
        return self.history.len;
    }

    pub fn rowCount(self: *const RecentDemand) usize {
        return self.retained_rows;
    }

    pub fn scores(self: *const RecentDemand) *const [expert_count]Count {
        return &self.counts;
    }

    pub fn score(self: *const RecentDemand, expert_id: i32) Count {
        if (expert_id < 0 or expert_id >= expert_count) return 0;
        return self.counts[@intCast(expert_id)];
    }

    pub fn observeRow(self: *RecentDemand, picks: []const i32) void {
        var incoming: Row = @splat(0);
        for (picks) |expert_id| {
            if (expert_id < 0 or expert_id >= expert_count) continue;
            const expert: usize = @intCast(expert_id);
            incoming[expert / @bitSizeOf(u64)] |= @as(u64, 1) << @as(u6, @intCast(expert % @bitSizeOf(u64)));
        }

        const outgoing = self.history[self.next_row];
        for (outgoing, incoming, 0..) |expired_word, incoming_word, word_index| {
            var removed = expired_word & ~incoming_word;
            while (removed != 0) {
                const expert = word_index * @bitSizeOf(u64) + @ctz(removed);
                self.counts[expert] -= 1;
                removed &= removed - 1;
            }
            var added = incoming_word & ~expired_word;
            while (added != 0) {
                const expert = word_index * @bitSizeOf(u64) + @ctz(added);
                self.counts[expert] += 1;
                added &= added - 1;
            }
        }
        self.history[self.next_row] = incoming;
        self.next_row += 1;
        if (self.next_row == self.history.len) self.next_row = 0;
        if (self.retained_rows < self.history.len) self.retained_rows += 1;
    }
};

test "demand expires after exactly the configured number of rows" {
    var demand = try RecentDemand.init(std.testing.allocator, 3);
    defer demand.deinit(std.testing.allocator);

    demand.observeRow(&.{0});
    demand.observeRow(&.{ 1, 511 });
    demand.observeRow(&.{0});
    try std.testing.expectEqual(@as(Count, 2), demand.score(0));
    try std.testing.expectEqual(@as(Count, 1), demand.score(511));
    try std.testing.expectEqual(@as(usize, 3), demand.rowCount());

    demand.observeRow(&.{});
    try std.testing.expectEqual(@as(Count, 1), demand.score(0));
    try std.testing.expectEqual(@as(Count, 1), demand.score(1));
    demand.observeRow(&.{ -1, 512 });
    try std.testing.expectEqual(@as(Count, 0), demand.score(1));
    try std.testing.expectEqual(@as(Count, 0), demand.score(511));
    demand.observeRow(&.{});
    try std.testing.expectEqual(@as([expert_count]Count, @splat(0)), demand.scores().*);
    try std.testing.expectEqual(@as(usize, 3), demand.rowCount());
}

test "duplicates contribute once and shared masked and invalid picks are ignored" {
    var demand = try RecentDemand.init(std.testing.allocator, 2);
    defer demand.deinit(std.testing.allocator);

    demand.observeRow(&.{ -1, 0, 0, 63, 64, 64, 511, 511, 512, 512, -2, 513, std.math.maxInt(i32), std.math.minInt(i32) });
    for (demand.scores(), 0..) |count, expert| {
        const expected: Count = if (expert == 0 or expert == 63 or expert == 64 or expert == 511) 1 else 0;
        try std.testing.expectEqual(expected, count);
    }
    try std.testing.expectEqual(@as(Count, 0), demand.score(-1));
    try std.testing.expectEqual(@as(Count, 0), demand.score(512));
    try std.testing.expectEqual(@as(Count, 0), demand.score(std.math.maxInt(i32)));
}

test "pick permutations produce identical state through wraparound" {
    var ordered = try RecentDemand.init(std.testing.allocator, 3);
    defer ordered.deinit(std.testing.allocator);
    var permuted = try RecentDemand.init(std.testing.allocator, 3);
    defer permuted.deinit(std.testing.allocator);

    for (0..10) |_| {
        ordered.observeRow(&.{ 0, 63, 64, 511 });
        permuted.observeRow(&.{ 511, 64, 0, 63, 0, 512, -1 });
        ordered.observeRow(&.{ 7, 8 });
        permuted.observeRow(&.{ 8, 7, 8 });
        ordered.observeRow(&.{});
        permuted.observeRow(&.{ -1, 512 });
        try std.testing.expectEqual(ordered.counts, permuted.counts);
        try std.testing.expectEqualSlices(Row, ordered.history, permuted.history);
        try std.testing.expectEqual(ordered.next_row, permuted.next_row);
    }
}

test "expiration follows row order" {
    var forward = try RecentDemand.init(std.testing.allocator, 2);
    defer forward.deinit(std.testing.allocator);
    var reverse = try RecentDemand.init(std.testing.allocator, 2);
    defer reverse.deinit(std.testing.allocator);

    forward.observeRow(&.{1});
    forward.observeRow(&.{2});
    reverse.observeRow(&.{2});
    reverse.observeRow(&.{1});
    try std.testing.expectEqual(forward.counts, reverse.counts);
    forward.observeRow(&.{});
    reverse.observeRow(&.{});
    try std.testing.expectEqual(@as(Count, 0), forward.score(1));
    try std.testing.expectEqual(@as(Count, 1), forward.score(2));
    try std.testing.expectEqual(@as(Count, 1), reverse.score(1));
    try std.testing.expectEqual(@as(Count, 0), reverse.score(2));
}

test "reset and replay reproduce every state transition" {
    const rows = [_][]const i32{ &.{ 1, 2, 2 }, &.{511}, &.{}, &.{ -1, 512 }, &.{ 2, 511 }, &.{7}, &.{ 1, 7 } };
    var replayed = try RecentDemand.init(std.testing.allocator, 3);
    defer replayed.deinit(std.testing.allocator);
    var fresh = try RecentDemand.init(std.testing.allocator, 3);
    defer fresh.deinit(std.testing.allocator);

    for (rows) |row| replayed.observeRow(row);
    replayed.reset();
    try std.testing.expectEqual(fresh.counts, replayed.counts);
    try std.testing.expectEqualSlices(Row, fresh.history, replayed.history);
    try std.testing.expectEqual(@as(usize, 0), replayed.rowCount());
    try std.testing.expectEqual(@as(usize, 0), replayed.next_row);
    try std.testing.expectEqual(@as(usize, 3), replayed.windowRows());

    for (rows) |row| {
        replayed.observeRow(row);
        fresh.observeRow(row);
        try std.testing.expectEqual(fresh.counts, replayed.counts);
        try std.testing.expectEqualSlices(Row, fresh.history, replayed.history);
        try std.testing.expectEqual(fresh.next_row, replayed.next_row);
        try std.testing.expectEqual(fresh.rowCount(), replayed.rowCount());
    }
    replayed.reset();
    replayed.reset();
    try std.testing.expectEqual(@as([expert_count]Count, @splat(0)), replayed.scores().*);
}

test "all expert scores stay exact at capacity over repeated replacements" {
    var demand = try RecentDemand.init(std.testing.allocator, 1);
    defer demand.deinit(std.testing.allocator);
    var picks: [expert_count]i32 = undefined;
    for (&picks, 0..) |*expert, index| expert.* = @intCast(index);
    for (0..1024) |_| {
        demand.observeRow(&picks);
        try std.testing.expectEqual(@as([expert_count]Count, @splat(1)), demand.scores().*);
    }
    demand.observeRow(&.{});
    try std.testing.expectEqual(@as([expert_count]Count, @splat(0)), demand.scores().*);

    var larger = try RecentDemand.init(std.testing.allocator, 4);
    defer larger.deinit(std.testing.allocator);
    for (0..16) |index| {
        larger.observeRow(&picks);
        const expected: Count = @intCast(@min(index + 1, larger.windowRows()));
        for (larger.scores()) |count| try std.testing.expectEqual(expected, count);
    }
    for (0..4) |index| {
        larger.observeRow(&.{});
        const expected: Count = @intCast(3 - index);
        for (larger.scores()) |count| try std.testing.expectEqual(expected, count);
    }
}

test "invalid capacities are rejected before allocation" {
    try std.testing.expectError(error.InvalidWindowSize, RecentDemand.init(std.testing.allocator, 0));
    if (@bitSizeOf(usize) > @bitSizeOf(Count)) {
        try std.testing.expectError(error.InvalidWindowSize, RecentDemand.init(std.testing.allocator, @as(usize, std.math.maxInt(Count)) + 1));
    }
}
