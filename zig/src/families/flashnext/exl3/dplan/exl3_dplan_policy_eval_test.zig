//! Host tests of the CPU policy counterfactuals: round hits and admissions, decay, LRU, Belady and bad input.

const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");
const policy = @import("exl3_dplan_policy_eval.zig");
const expert_count = policy.expert_count;
const State = policy.State;
const Metrics = policy.Metrics;
const Options = policy.Options;
const Event = policy.Event;
const evaluate = policy.evaluate;
const belady = policy.belady;
const routed = policy.routed;

const Synthetic = struct {
    words: []u32,
    max_rows: usize,

    fn init(picks: []const i32) !Synthetic {
        const rows = @max(1, (picks.len + 10) / 11);
        const shape = try trace.header(0, 0, 2, rows);
        const words = try std.testing.allocator.alloc(u32, shape[7]);
        @memset(words, 0);
        words[0] = @intCast(rows);
        const self: Synthetic = .{ .words = words, .max_rows = rows };
        const raw_offset = trace.record_header_words;
        @memset(words[raw_offset..][0 .. rows * 11], std.math.maxInt(u32));
        var counts: [expert_count]u32 = @splat(0);
        for (picks, 0..) |expert, index| {
            words[raw_offset + index] = @bitCast(expert);
            if (routed(@bitCast(expert))) |valid| {
                if (counts[valid] == 0) words[2] += 1;
                counts[valid] += 1;
                words[1] += 1;
            }
        }
        return self;
    }

    fn round(self: Synthetic) trace.Round {
        return .{ .words = self.words, .lines = 2, .max_rows = self.max_rows };
    }

    fn deinit(self: Synthetic) void {
        std.testing.allocator.free(self.words);
    }
};

test "exl3: policy evaluation pins round hits and admissions and separates mapped misses" {
    var fixture = try Synthetic.init(&.{ 0, 2, 2, 2, 3, 512, -1, 999, -9 });
    defer fixture.deinit();
    var state = try State.init(std.testing.allocator, .lifetime, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = @splat(0), .now = 2 },
    });
    defer state.deinit();
    const result = try state.observe(fixture.round());
    try std.testing.expectEqual(Metrics{ .rounds = 1, .hit_picks = 1, .miss_picks = 4, .hit_experts = 1, .miss_experts = 2, .admissions = 1, .copies = 1, .evictions = 1, .mapped_experts = 1, .mapped_picks = 1 }, result);
    try std.testing.expectEqualSlices(i32, &.{ 0, 2 }, state.line_ids);
}

test "exl3: lifetime admission saturation and victim ties match current planner" {
    var fixture = try Synthetic.init(&.{ 2, 2, 2, 2, 2, 3 });
    defer fixture.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 5;
    counts[1] = 3;
    counts[2] = std.math.maxInt(u32) - 2;
    var state = try State.init(std.testing.allocator, .lifetime, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = counts, .now = std.math.maxInt(u32) },
    });
    defer state.deinit();
    const result = try state.observe(fixture.round());
    try std.testing.expectEqual(@as(u64, 2), result.copies);
    try std.testing.expectEqualSlices(i32, &.{ 3, 2 }, state.line_ids);
    try std.testing.expectEqual(std.math.maxInt(u32), state.scores[2]);
    try std.testing.expectEqual(@as(u32, 0), state.now);

    var tie = try State.init(std.testing.allocator, .lifetime, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 7, 7 }, .counts = @splat(0), .now = 7 },
    });
    defer tie.deinit();
    var single = try Synthetic.init(&.{9});
    defer single.deinit();
    _ = try tie.observe(single.round());
    try std.testing.expectEqualSlices(i32, &.{ 0, 9 }, tie.line_ids);
}

test "exl3: LRU preserves raw last occurrence order when the round exceeds capacity" {
    var first = try Synthetic.init(&.{ 1, 2, 1 });
    defer first.deinit();
    var second = try Synthetic.init(&.{ 1, 1, 2 });
    defer second.deinit();
    var left = try State.init(std.testing.allocator, .lru, .{ .capacity = 1 });
    defer left.deinit();
    var right = try State.init(std.testing.allocator, .lru, .{ .capacity = 1 });
    defer right.deinit();
    _ = try left.observe(first.round());
    _ = try right.observe(second.round());
    try std.testing.expectEqualSlices(i32, &.{1}, left.line_ids);
    try std.testing.expectEqualSlices(i32, &.{2}, right.line_ids);
    _ = try left.observe(second.round());
    try std.testing.expectEqualSlices(i32, &.{1}, left.line_ids);
}

test "exl3: frequency decay halves scores at configured round boundaries before new demand" {
    var fixture = try Synthetic.init(&.{ 0, 1, 1, 1 });
    defer fixture.deinit();
    var empty = try Synthetic.init(&.{});
    defer empty.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 64;
    const options: Options = .{
        .capacity = 1,
        .seed = .{ .line_ids = &.{-1}, .stamps = &.{0}, .counts = counts, .now = 0 },
        .decay_interval_rounds = 1,
        .decay_shift = 6,
    };
    var lifetime = try State.init(std.testing.allocator, .lifetime, options);
    defer lifetime.deinit();
    var decayed = try State.init(std.testing.allocator, .decayed_frequency, options);
    defer decayed.deinit();
    _ = try lifetime.observe(fixture.round());
    _ = try decayed.observe(fixture.round());
    try std.testing.expectEqualSlices(i32, &.{0}, lifetime.line_ids);
    try std.testing.expectEqualSlices(i32, &.{1}, decayed.line_ids);

    var interval = try State.init(std.testing.allocator, .decayed_frequency, .{ .capacity = 0, .decay_interval_rounds = 2 });
    defer interval.deinit();
    _ = try interval.observe(fixture.round());
    try std.testing.expectEqual(@as(u32, 3), interval.scores[1]);
    _ = try interval.observe(empty.round());
    try std.testing.expectEqual(@as(u32, 1), interval.scores[1]);
}

test "exl3: LRU evicts by within-round recency and decayed eviction forgets old frequency" {
    var touching = try Synthetic.init(&.{ 0, 1, 0 });
    defer touching.deinit();
    var incoming = try Synthetic.init(&.{2});
    defer incoming.deinit();
    var empty = try Synthetic.init(&.{});
    defer empty.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 128;
    counts[1] = 4;
    const options: Options = .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = counts, .now = 2 },
        .decay_interval_rounds = 1,
        .decay_shift = 6,
    };
    var lru = try State.init(std.testing.allocator, .lru, options);
    defer lru.deinit();
    _ = try lru.observe(touching.round());
    _ = try lru.observe(incoming.round());
    try std.testing.expectEqualSlices(i32, &.{ 0, 2 }, lru.line_ids);

    var lifetime = try State.init(std.testing.allocator, .lifetime, options);
    defer lifetime.deinit();
    var decayed = try State.init(std.testing.allocator, .decayed_frequency, options);
    defer decayed.deinit();
    for (0..2) |_| {
        _ = try lifetime.observe(empty.round());
        _ = try decayed.observe(empty.round());
    }
    _ = try lifetime.observe(incoming.round());
    _ = try decayed.observe(incoming.round());
    try std.testing.expectEqualSlices(i32, &.{ 0, 2 }, lifetime.line_ids);
    try std.testing.expectEqualSlices(i32, &.{ 2, 1 }, decayed.line_ids);
}

test "exl3: frequency admission bypasses a colder newcomer and later admits it when hot" {
    var warm = try Synthetic.init(&.{ 1, 1 });
    defer warm.deinit();
    var hot = try Synthetic.init(&.{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
    defer hot.deinit();
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 10;
    var state = try State.init(std.testing.allocator, .decayed_admission, .{
        .capacity = 1,
        .seed = .{ .line_ids = &.{0}, .stamps = &.{1}, .counts = counts, .now = 1 },
    });
    defer state.deinit();
    const cold = try state.observe(warm.round());
    try std.testing.expectEqual(@as(u64, 0), cold.copies);
    try std.testing.expectEqual(@as(u64, 1), cold.mapped_experts);
    try std.testing.expectEqualSlices(i32, &.{0}, state.line_ids);
    const now_hot = try state.observe(hot.round());
    try std.testing.expectEqual(@as(u64, 1), now_hot.copies);
    try std.testing.expectEqual(@as(u64, 1), now_hot.evictions);
    try std.testing.expectEqualSlices(i32, &.{1}, state.line_ids);
}

test "exl3: comparison evolves independently from the initial seed and supports zero capacity" {
    var fixture = try Synthetic.init(&.{ 5, 5 });
    defer fixture.deinit();
    const rounds = [_]trace.Round{ fixture.round(), fixture.round() };
    const result = try evaluate(std.testing.allocator, &rounds, .{ .capacity = 1 });
    try std.testing.expectEqual(@as(u64, 2), result.lifetime.miss_picks);
    try std.testing.expectEqual(@as(u64, 2), result.lifetime.hit_picks);
    try std.testing.expectEqual(@as(u64, 1), result.lifetime.copies);
    try std.testing.expectEqual(result.lifetime, result.lru);
    try std.testing.expectEqual(result.lifetime, result.decayed_frequency);
    try std.testing.expectEqual(result.lifetime, result.decayed_admission);
    try std.testing.expectEqual(@as(u64, 1), result.belady_unique_misses_lower_bound);
    const zero = try evaluate(std.testing.allocator, &rounds, .{ .capacity = 0 });
    try std.testing.expectEqual(@as(u64, 4), zero.lifetime.mapped_picks);
    try std.testing.expectEqual(@as(u64, 0), zero.lifetime.copies);
    try std.testing.expectEqual(@as(u64, 4), zero.decayed_admission.mapped_picks);
    try std.testing.expectEqual(@as(u64, 0), zero.decayed_admission.copies);
    try std.testing.expectEqual(@as(u64, 2), zero.belady_unique_misses_lower_bound);
    const none = try evaluate(std.testing.allocator, &.{}, .{ .capacity = 2 });
    try std.testing.expectEqual(Metrics{}, none.lifetime);
    try std.testing.expectEqual(@as(u64, 0), none.belady_unique_misses_lower_bound);
}

test "exl3: Belady bypass retains a warm expert and bounds pinned round misses only" {
    var first = try Synthetic.init(&.{ 1, 1, 2 });
    defer first.deinit();
    var second = try Synthetic.init(&.{ 0, 0 });
    defer second.deinit();
    const options: Options = .{
        .capacity = 1,
        .seed = .{ .line_ids = &.{0}, .stamps = &.{0}, .counts = @splat(0), .now = 0 },
    };
    const result = try evaluate(std.testing.allocator, &.{ first.round(), second.round() }, options);
    try std.testing.expectEqual(@as(u64, 2), result.belady_unique_misses_lower_bound);
    try std.testing.expectEqual(@as(u64, 3), result.lifetime.miss_experts);
    try std.testing.expect(result.belady_unique_misses_lower_bound <= result.lru.miss_experts);
    try std.testing.expect(result.belady_unique_misses_lower_bound <= result.decayed_frequency.miss_experts);
}

fn exhaustiveMisses(events: []const Event, position: usize, mask: u8, capacity: usize) u64 {
    if (position == events.len) return 0;
    const bit = @as(u8, 1) << @as(u3, @intCast(events[position].expert));
    if (mask & bit != 0) return exhaustiveMisses(events, position + 1, mask, capacity);
    var best = exhaustiveMisses(events, position + 1, mask, capacity);
    if (@popCount(mask) < capacity) {
        best = @min(best, exhaustiveMisses(events, position + 1, mask | bit, capacity));
    } else {
        for (0..3) |expert| {
            const victim_bit = @as(u8, 1) << @as(u3, @intCast(expert));
            if (mask & victim_bit != 0) best = @min(best, exhaustiveMisses(events, position + 1, (mask & ~victim_bit) | bit, capacity));
        }
    }
    return 1 + best;
}

test "exl3: Belady equals exhaustive optional-admission optimum for short cold and warm streams" {
    for (0..243) |sequence| {
        var encoded = sequence;
        var events: [5]Event = undefined;
        for (&events) |*event| {
            event.* = .{ .expert = encoded % 3 };
            encoded /= 3;
        }
        for (0..3) |capacity| {
            try std.testing.expectEqual(exhaustiveMisses(&events, 0, 0, capacity), belady(&events, capacity, &.{}));
            if (capacity > 0) try std.testing.expectEqual(exhaustiveMisses(&events, 0, 1, capacity), belady(&events, capacity, &.{0}));
        }
    }
}

test "exl3: invalid input is rejected before state changes including planner unique overflow" {
    var state = try State.init(std.testing.allocator, .lifetime, .{ .capacity = 1 });
    defer state.deinit();
    var fixture = try Synthetic.init(&.{1});
    defer fixture.deinit();
    var truncated = fixture.round();
    truncated.words = truncated.words[0..3];
    try std.testing.expectError(error.InvalidRound, state.observe(truncated));
    fixture.words[1] = 2;
    try std.testing.expectError(error.InvalidRound, state.observe(fixture.round()));
    var picks: [257]i32 = undefined;
    for (&picks, 0..) |*expert, index| expert.* = @intCast(index);
    var overflow = try Synthetic.init(&picks);
    defer overflow.deinit();
    try std.testing.expectError(error.PlannerOverflow, state.observe(overflow.round()));
    try std.testing.expectEqual(Metrics{}, state.total);
    try std.testing.expectEqualSlices(i32, &.{-1}, state.line_ids);
    try std.testing.expectError(error.InvalidOptions, State.init(std.testing.allocator, .lru, .{ .capacity = 513 }));
    try std.testing.expectError(error.InvalidOptions, State.init(std.testing.allocator, .lru, .{ .capacity = 1, .decay_interval_rounds = 0 }));
    try std.testing.expectError(error.InvalidSeed, State.init(std.testing.allocator, .lru, .{
        .capacity = 2,
        .seed = .{ .line_ids = &.{ 1, 1 }, .stamps = &.{ 0, 0 }, .counts = @splat(0), .now = 0 },
    }));
}
