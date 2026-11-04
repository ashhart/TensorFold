const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");
const layout = @import("exl3_dplan_layout.zig");
const replay = @import("exl3_dplan_replay.zig");
const policies = @import("exl3_dplan_policy_eval.zig");
const fixture = @import("exl3_dplan_trace_test.zig");

fn capture(words: []u32, state: *replay.ExactState, picks: []const i32, recent: bool) !replay.Delta {
    const before = try std.testing.allocator.dupe(u32, state.words);
    defer std.testing.allocator.free(before);
    const record_words = try std.testing.allocator.alloc(u32, words[7]);
    defer std.testing.allocator.free(record_words);
    @memset(record_words, 0);
    record_words[0] = @intCast(picks.len / 11);
    record_words[3] = @intFromBool(recent);
    for (picks, 0..) |pick, index| record_words[trace.record_header_words + index] = @bitCast(pick);
    const delta = try state.replay(.{ .words = record_words, .lines = state.lines, .max_rows = words[5] });
    fixture.append(words, before, state.words, picks, recent, @intCast(delta.mapped_picks));
    return delta;
}

test "exl3: ownership checkpoints verify every segment including repeated and final transfers" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    var state = try replay.ExactState.init(allocator, 2, seed);
    defer state.deinit();
    fixture.checkpoint(words, state.words);
    _ = try capture(words, &state, &.{ 2, 2, 3, -1, -1, -1, -1, -1, -1, -1, 512 }, false);
    @memcpy(state.words, seed);
    state.words[0] = 77;
    state.words[4 + 77] = 0;
    state.words[4 + layout.experts_n + 77] = 90;
    state.words[layout.miscOffset(2)] = 500;
    fixture.checkpoint(words, state.words);
    state.words[layout.miscOffset(2)] += 10;
    fixture.checkpoint(words, state.words);
    _ = try capture(words, &state, &.{ 77, 77, 88, -1, -1, -1, -1, -1, -1, -1, 512 }, false);
    state.words[layout.miscOffset(2)] += 10;
    fixture.checkpoint(words, state.words);
    const view = try trace.View.init(words);
    const result = try replay.verifyTraceWithFinal(allocator, view, state.words);
    try std.testing.expectEqual(@as(usize, 2), result.matched_transitions);
    try std.testing.expectEqual(@as(u64, 2), result.delta.hit_picks);
    try std.testing.expectEqual(@as(u64, 4), result.delta.miss_picks);
    const comparison = try policies.evaluateTrace(allocator, view);
    try std.testing.expectEqual(result.delta.hit_picks, comparison.lifetime.hit_picks);
    try std.testing.expectEqual(result.delta.miss_picks, comparison.lifetime.miss_picks);
    try std.testing.expectEqual(result.delta.copies, comparison.lifetime.copies);
    try std.testing.expectEqual(result.delta.evictions, comparison.lifetime.evictions);
    try std.testing.expectEqual(result.delta.inplace, comparison.lifetime.mapped_experts);
    try std.testing.expectEqual(result.delta.mapped_picks, comparison.lifetime.mapped_picks);
    try std.testing.expectEqual(result.rounds, comparison.lifetime.rounds);
    try std.testing.expectEqual(@as(usize, 4), view.checkpoints());
    for ([_]usize{ 0, 1, 1, 2 }, 0..) |round, index| try std.testing.expectEqual(round, (try view.checkpoint(index)).roundIndex());
    const bytes = try view.encode(allocator);
    defer allocator.free(bytes);
    try std.testing.expectEqual(trace.pool_bytes, bytes.len);
    var snapshot = try trace.Snapshot.decode(allocator, bytes);
    defer snapshot.deinit();
    _ = try replay.verifyTraceWithFinal(allocator, try snapshot.view(), state.words);
    state.words[0] ^= 1;
    try std.testing.expectError(error.PlannerFinalStateMismatch, replay.verifyTraceWithFinal(allocator, view, state.words));
}

test "exl3: unexplained handoff and corrupted outgoing checkpoint cannot be reseeded away" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    var state = try replay.ExactState.init(allocator, 2, seed);
    defer state.deinit();
    const picks: [11]i32 = @splat(-1);
    _ = try capture(words, &state, &picks, false);
    state.words[layout.miscOffset(2)] += 100;
    _ = try capture(words, &state, &picks, false);
    try std.testing.expectError(error.TraceDiscontinuous, replay.verifyTrace(allocator, try trace.View.init(words)));
    words[12] = 0;
    try std.testing.expectError(error.PlannerCounterMismatch, replay.verifyTrace(allocator, try trace.View.init(words)));
    words[12] = 1;
    fixture.checkpoint(words, state.words);
    try std.testing.expectError(error.TraceDiscontinuous, replay.verifyTrace(allocator, try trace.View.init(words)));

    @memset(words, 0);
    @memcpy(words[0..trace.header_words], &try trace.header(0, 0, 2, 1));
    @memcpy(state.words, seed);
    _ = try capture(words, &state, &picks, false);
    state.words[layout.miscOffset(2)] += 100;
    fixture.checkpoint(words, state.words);
    _ = try capture(words, &state, &picks, false);
    const view = try trace.View.init(words);
    _ = try replay.verifyTrace(allocator, view);
    const offset = trace.pool_words - trace.checkpointWords(words[6]);
    for (0..words[6]) |index| {
        words[offset + 4 + index] ^= 1;
        try std.testing.expectError(error.PlannerCheckpointMismatch, replay.verifyTrace(allocator, try trace.View.init(words)));
        words[offset + 4 + index] ^= 1;
    }
    words[offset + 4 + words[6]] = 9999;
    try std.testing.expectError(error.InvalidSeed, replay.verifyTrace(allocator, try trace.View.init(words)));
}

test "exl3: dropped ownership checkpoints invalidate exact replay even without decode rounds" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 16);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    const capacity = (trace.pool_words - trace.header_words - 2 * words[6]) / trace.checkpointWords(words[6]);
    for (0..capacity + 1) |_| fixture.checkpoint(words, seed);
    try std.testing.expectError(error.TraceDroppedRecords, replay.verifyTrace(allocator, try trace.View.init(words)));
}

test "exl3: v4 exact warm replay matches known decisions counters and all terminal words" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    seed[0] = 0;
    seed[1] = 1;
    seed[2] = 10;
    seed[3] = 20;
    seed[4] = 0;
    seed[5] = 1;
    seed[4 + 512] = 5;
    seed[5 + 512] = 3;
    seed[layout.miscOffset(2)] = 20;
    var state = try replay.ExactState.init(allocator, 2, seed);
    defer state.deinit();
    const first = try capture(words, &state, &.{ 2, 2, 2, 2, 2, 3, -1, -1, -1, -1, 512 }, false);
    try std.testing.expectEqual(replay.Delta{ .miss_picks = 6, .copies = 2, .evictions = 2 }, first);
    try std.testing.expectEqualSlices(u32, &.{ 3, 2, 21, 21 }, state.words[0..4]);
    const second = try capture(words, &state, &.{ 2, 2, 4, -1, -1, -1, -1, -1, -1, -1, -1 }, false);
    try std.testing.expectEqual(replay.Delta{ .hit_picks = 2, .miss_picks = 1, .copies = 1, .evictions = 1 }, second);
    const view = try trace.View.init(words);
    const result = try replay.verifyTraceWithFinal(allocator, view, state.words);
    try std.testing.expectEqual(@as(usize, 2), result.matched_transitions);
    try std.testing.expectEqual(replay.Delta{ .hit_picks = 2, .miss_picks = 7, .copies = 3, .evictions = 3 }, result.delta);
    const comparison = try policies.evaluate(allocator, &.{ try view.round(0), try view.round(1) }, .{ .capacity = 2, .seed = view.seedState(policies.Seed) });
    try std.testing.expectEqual(result.delta.hit_picks, comparison.lifetime.hit_picks);
    try std.testing.expectEqual(result.delta.miss_picks, comparison.lifetime.miss_picks);
    try std.testing.expectEqual(result.delta.copies, comparison.lifetime.copies);
    try std.testing.expectEqual(result.delta.evictions, comparison.lifetime.evictions);
    state.words[state.words.len - 1] = 1;
    try std.testing.expectError(error.PlannerFinalStateMismatch, replay.verifyTraceWithFinal(allocator, view, state.words));
}

test "exl3: v4 1200 full rounds replay recent wrap saturated counts and modular counters" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 10, 16);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 10);
    defer allocator.free(seed);
    seed[20 + 512 + 7] = std.math.maxInt(u32);
    const misc = layout.miscOffset(10);
    @memset(seed[misc .. misc + 8], std.math.maxInt(u32) - 1);
    var state = try replay.ExactState.init(allocator, 10, seed);
    defer state.deinit();
    var total: replay.Delta = .{};
    for (0..1200) |round| {
        var picks: [16 * 11]i32 = undefined;
        for (&picks, 0..) |*pick, index| {
            pick.* = switch (index % 11) {
                0 => -1,
                1 => 512,
                2 => -99,
                3 => 800,
                else => @intCast((round * 7 + index * 3) % 512),
            };
        }
        const delta = try capture(words, &state, &picks, round % 3 != 0);
        inline for (comptime std.meta.fieldNames(replay.Delta)) |field| @field(total, field) += @field(delta, field);
    }
    const view = try trace.View.init(words);
    try std.testing.expectEqual(@as(usize, 1200), view.recorded());
    try std.testing.expectEqual(@as(u64, 0), view.dropped());
    const result = try replay.verifyTraceWithFinal(allocator, view, state.words);
    try std.testing.expectEqual(total, result.delta);
    try std.testing.expectEqual(std.math.maxInt(u32), state.words[20 + 512 + 7]);
    try std.testing.expectEqual(@as(u32, 1198), state.words[misc]);
    try std.testing.expectEqual(@as(u32, 256), state.words[misc + 9]);
    const bytes = try view.encode(allocator);
    defer allocator.free(bytes);
    try std.testing.expect(bytes.len <= trace.pool_bytes);
    var snapshot = try trace.Snapshot.decode(allocator, bytes);
    defer snapshot.deinit();
    _ = try replay.verifyTrace(allocator, try snapshot.view());
}

test "exl3: v4 starts from warm recent history and retains empty masked rounds" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    seed[layout.recentCountsOffset(2) + 7] = 2;
    seed[layout.recentOffset(2)] = 1 << 7;
    seed[layout.recentOffset(2) + 16] = 1 << 7;
    seed[layout.miscOffset(2) + 8] = 2;
    seed[layout.miscOffset(2) + 9] = 2;
    var state = try replay.ExactState.init(allocator, 2, seed);
    defer state.deinit();
    _ = try capture(words, &state, &.{ 7, 7, -1, 512, 999, -1, -1, -1, -1, -1, -1 }, true);
    try std.testing.expectEqual(@as(u32, 3), state.words[layout.recentCountsOffset(2) + 7]);
    try std.testing.expectEqual(@as(u32, 3), state.words[layout.miscOffset(2) + 8]);
    const masked = try capture(words, &state, &@as([11]i32, @splat(-1)), true);
    try std.testing.expectEqual(replay.Delta{}, masked);
    const result = try replay.verifyTraceWithFinal(allocator, try trace.View.init(words), state.words);
    try std.testing.expectEqual(@as(usize, 2), result.rounds);
}

test "exl3: v4 mapped experts count repeated reads and mask shared and invalid slots" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 1, 1);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 1);
    defer allocator.free(seed);
    seed[0] = 0;
    seed[2] = 0;
    var state = try replay.ExactState.init(allocator, 1, seed);
    defer state.deinit();
    const delta = try capture(words, &state, &.{ 0, 0, 1, 1, 1, 512, -1, 700, -9, 512, -1 }, false);
    try std.testing.expectEqual(replay.Delta{ .hit_picks = 2, .miss_picks = 3, .inplace = 1, .mapped_picks = 3 }, delta);
    const view = try trace.View.init(words);
    _ = try replay.verifyTrace(allocator, view);
    const comparison = try policies.evaluateTrace(allocator, view);
    try std.testing.expectEqual(delta.hit_picks, comparison.lifetime.hit_picks);
    try std.testing.expectEqual(delta.miss_picks, comparison.lifetime.miss_picks);
    try std.testing.expectEqual(delta.inplace, comparison.lifetime.mapped_experts);
    try std.testing.expectEqual(delta.mapped_picks, comparison.lifetime.mapped_picks);
    const offset = (try trace.View.init(words)).recordsOffset();
    words[offset + 14] = 2;
    try std.testing.expectError(error.PlannerCounterMismatch, replay.verifyTrace(allocator, try trace.View.init(words)));
}

test "exl3: v4 replay rejects altered production outcomes every terminal field and bad seeds" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 1);
    defer allocator.free(words);
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    var state = try replay.ExactState.init(allocator, 2, seed);
    defer state.deinit();
    _ = try capture(words, &state, &.{ 2, 2, 3, -1, -1, -1, -1, -1, -1, -1, 512 }, false);
    const view = try trace.View.init(words);
    for (0..layout.misc_words) |index| {
        const offset = view.recordsOffset() + 4 + index;
        words[offset] ^= 1;
        try std.testing.expectError(error.PlannerCounterMismatch, replay.verifyTrace(allocator, try trace.View.init(words)));
        words[offset] ^= 1;
    }
    const terminal_offset = trace.header_words + words[6];
    for (0..words[6]) |index| {
        words[terminal_offset + index] ^= 1;
        try std.testing.expectError(error.PlannerFinalStateMismatch, replay.verifyTrace(allocator, try trace.View.init(words)));
        words[terminal_offset + index] ^= 1;
    }
    for ([_]usize{ 0, 4, layout.recentCountsOffset(2), layout.miscOffset(2) + 8 }) |index| {
        const saved = words[trace.header_words + index];
        words[trace.header_words + index] = 9999;
        try std.testing.expectError(error.InvalidSeed, replay.verifyTrace(allocator, try trace.View.init(words)));
        words[trace.header_words + index] = saved;
    }
}

test "exl3: v4 overflow dropped empty invalid and discontinuous timelines cannot pass" {
    const allocator = std.testing.allocator;
    const words = try fixture.make(allocator, 2, 47);
    defer allocator.free(words);
    try std.testing.expectError(error.EmptyTrace, replay.verifyTrace(allocator, try trace.View.init(words)));
    const seed = try fixture.emptySeed(allocator, 2);
    defer allocator.free(seed);
    var state = try replay.ExactState.init(allocator, 2, seed);
    defer state.deinit();
    var picks: [47 * 11]i32 = @splat(512);
    for (0..512) |index| picks[index] = @intCast(index);
    const delta = try capture(words, &state, &picks, false);
    try std.testing.expectEqual(@as(u64, 512), delta.miss_picks);
    try std.testing.expectEqual(@as(u64, 0), delta.copies);
    try std.testing.expectEqual(@as(u64, 1), delta.overflow);
    try std.testing.expectError(error.PlannerOverflow, replay.verifyTrace(allocator, try trace.View.init(words)));
    words[12] = 1;
    try std.testing.expectError(error.TraceDiscontinuous, replay.verifyTrace(allocator, try trace.View.init(words)));
    words[12] = 0;
    words[13] = 1;
    try std.testing.expectError(error.TraceInvalidLaunch, replay.verifyTrace(allocator, try trace.View.init(words)));
    words[13] = 0;
    for (1..words[8] + 1) |_| _ = try capture(words, &state, &picks, false);
    try std.testing.expectError(error.TraceDroppedRecords, replay.verifyTrace(allocator, try trace.View.init(words)));
}
