const std = @import("std");
const trace = @import("exl3_dplan_trace.zig");
const layout = @import("exl3_dplan_layout.zig");

pub const expert_count: usize = 512;
pub const max_unique: usize = 256;

pub const Entry = struct { expert: u32, appearances: u32 };
pub const Round = struct {
    rows: u32,
    valid_picks: u32,
    entries: []const Entry,
    planner_overflow: bool = false,
};

pub const Seed = struct {
    line_ids: []const i32,
    stamps: []const u32,
    counts: [expert_count]u32,
    now: u32,
};

pub const Delta = struct {
    hit_picks: u64 = 0,
    miss_picks: u64 = 0,
    copies: u64 = 0,
    evictions: u64 = 0,
    inplace: u64 = 0,
    mapped_picks: u64 = 0,
    overflow: u64 = 0,
};

pub const Verification = struct { rounds: usize, matched_transitions: usize, delta: Delta };

pub const State = struct {
    allocator: std.mem.Allocator,
    line_ids: []i32,
    stamps: []u32,
    counts: [expert_count]u32,
    now: u32,

    pub fn init(allocator: std.mem.Allocator, seed: Seed) !State {
        if (seed.line_ids.len == 0 or seed.line_ids.len != seed.stamps.len) return error.InvalidSeed;
        var resident: [expert_count]bool = @splat(false);
        for (seed.line_ids) |id| {
            if (id < -1 or id >= @as(i32, expert_count)) return error.InvalidSeed;
            if (id >= 0) {
                const index: usize = @intCast(id);
                if (resident[index]) return error.InvalidSeed;
                resident[index] = true;
            }
        }
        const line_ids = try allocator.dupe(i32, seed.line_ids);
        errdefer allocator.free(line_ids);
        const stamps = try allocator.dupe(u32, seed.stamps);
        return .{ .allocator = allocator, .line_ids = line_ids, .stamps = stamps, .counts = seed.counts, .now = seed.now };
    }

    pub fn deinit(self: *State) void {
        self.allocator.free(self.line_ids);
        self.allocator.free(self.stamps);
        self.* = undefined;
    }

    pub fn replay(self: *State, round: Round) !Delta {
        if (round.planner_overflow or round.rows == 0 or round.rows > std.math.maxInt(u32) / 11 or round.valid_picks > round.rows * 11 or round.entries.len > max_unique) return error.InvalidRound;
        var seen: [expert_count]bool = @splat(false);
        var appearances: u64 = 0;
        for (round.entries) |entry| {
            if (entry.expert >= @as(u32, expert_count) or entry.appearances == 0 or seen[@intCast(entry.expert)]) return error.InvalidRound;
            seen[@intCast(entry.expert)] = true;
            appearances += entry.appearances;
        }
        if (appearances != round.valid_picks) return error.InvalidRound;

        self.now +%= 1;
        var delta: Delta = .{};
        for (round.entries) |entry| {
            const expert: usize = @intCast(entry.expert);
            self.counts[expert] = self.counts[expert] +| entry.appearances;
            if (self.lineOf(entry.expert) != null) {
                delta.hit_picks += entry.appearances;
            } else {
                delta.miss_picks += entry.appearances;
            }
        }

        for (round.entries) |entry| if (self.lineOf(entry.expert)) |line| {
            self.stamps[line] = self.now;
        };

        var misses: [max_unique]Entry = undefined;
        var miss_count: usize = 0;
        for (round.entries) |entry| if (self.lineOf(entry.expert) == null) {
            misses[miss_count] = entry;
            miss_count += 1;
        };
        var index: usize = 1;
        while (index < miss_count) : (index += 1) {
            const item = misses[index];
            var insert = index;
            while (insert > 0 and hotter(self, item, misses[insert - 1])) : (insert -= 1) {
                misses[insert] = misses[insert - 1];
            }
            misses[insert] = item;
        }

        for (misses[0..miss_count]) |entry| {
            if (self.victim(&seen)) |line| {
                if (self.line_ids[line] >= 0) delta.evictions += 1;
                self.line_ids[line] = @intCast(entry.expert);
                self.stamps[line] = self.now;
                delta.copies += 1;
            } else {
                delta.inplace += 1;
                delta.mapped_picks += entry.appearances;
            }
        }
        return delta;
    }

    fn lineOf(self: *const State, expert: u32) ?usize {
        const target: i32 = @intCast(expert);
        for (self.line_ids, 0..) |id, line| if (id == target) return line;
        return null;
    }

    fn victim(self: *const State, keep: *const [expert_count]bool) ?usize {
        var empty: ?usize = null;
        var coldest: ?usize = null;
        for (self.line_ids, 0..) |id, line| {
            if (id < 0) {
                if (empty == null) empty = line;
                continue;
            }
            if (keep[@intCast(id)]) continue;
            if (coldest == null or colder(self, line, coldest.?)) coldest = line;
        }
        return empty orelse coldest;
    }
};

pub const ExactState = struct {
    allocator: std.mem.Allocator,
    lines: usize,
    words: []u32,

    pub fn init(allocator: std.mem.Allocator, lines: usize, seed: []const u32) !ExactState {
        const entry = layout.PoolState.init(lines, 0) catch return error.InvalidSeed;
        if (seed.len != entry.words) return error.InvalidSeed;
        var reverse: [expert_count]u32 = @splat(std.math.maxInt(u32));
        for (seed[0..lines], 0..) |raw, line| {
            const id: i32 = @bitCast(raw);
            if (id < -1 or id >= expert_count) return error.InvalidSeed;
            if (id >= 0) {
                const expert: usize = @intCast(id);
                if (reverse[expert] != std.math.maxInt(u32)) return error.InvalidSeed;
                reverse[expert] = @intCast(line);
            }
        }
        if (!std.mem.eql(u32, &reverse, seed[2 * lines ..][0..expert_count])) return error.InvalidSeed;
        const misc = seed[layout.miscOffset(lines)..];
        if (misc[8] >= layout.recent_window or misc[9] > layout.recent_window) return error.InvalidSeed;
        var recent_counts: [expert_count]u32 = @splat(0);
        for (seed[layout.recentOffset(lines)..layout.miscOffset(lines)], 0..) |bits, index| {
            for (0..32) |bit| {
                if (bits & (@as(u32, 1) << @intCast(bit)) != 0) recent_counts[(index % 16) * 32 + bit] += 1;
            }
        }
        if (!std.mem.eql(u32, &recent_counts, seed[layout.recentCountsOffset(lines)..][0..expert_count])) return error.InvalidSeed;
        for (recent_counts) |count| if (count > misc[9]) return error.InvalidSeed;
        return .{ .allocator = allocator, .lines = lines, .words = try allocator.dupe(u32, seed) };
    }

    pub fn deinit(self: *ExactState) void {
        self.allocator.free(self.words);
        self.* = undefined;
    }

    pub fn replay(self: *ExactState, record: trace.Round) !Delta {
        const lines = self.lines;
        const ids = self.words[0..lines];
        const stamps = self.words[lines .. 2 * lines];
        const reverse = self.words[2 * lines ..][0..expert_count];
        const counts = self.words[2 * lines + expert_count ..][0..expert_count];
        const recent = self.words[layout.recentCountsOffset(lines)..][0..expert_count];
        const history = self.words[layout.recentOffset(lines)..layout.miscOffset(lines)];
        const misc = self.words[layout.miscOffset(lines)..];
        misc[0] +%= 1;
        misc[1] +%= 1;
        if (record.recentEnabled()) {
            for (0..record.rows()) |row| {
                const expired = history[misc[8] * 16 ..][0..16];
                for (expired, 0..) |bits, word| {
                    for (0..32) |bit| {
                        if (bits & (@as(u32, 1) << @intCast(bit)) != 0) recent[word * 32 + bit] -%= 1;
                    }
                }
                @memset(expired, 0);
                for (record.picks()[row * 11 ..][0..11]) |raw| {
                    const expert: i32 = @bitCast(raw);
                    if (expert < 0 or expert >= expert_count) continue;
                    const index: usize = @intCast(expert);
                    const bit = @as(u32, 1) << @intCast(index & 31);
                    if (expired[index >> 5] & bit == 0) {
                        expired[index >> 5] |= bit;
                        recent[index] += 1;
                    }
                }
                misc[8] = (misc[8] + 1) % @as(u32, layout.recent_window);
                misc[9] = @min(misc[9] + 1, @as(u32, layout.recent_window));
            }
        }
        var delta: Delta = .{};
        var ordered: [max_unique]usize = undefined;
        var appearances: [expert_count]u16 = @splat(0);
        var keep: [expert_count]bool = @splat(false);
        var unique: usize = 0;
        var overflow = false;
        for (record.picks()) |raw| {
            const expert: i32 = @bitCast(raw);
            if (expert < 0 or expert >= expert_count) continue;
            const index: usize = @intCast(expert);
            counts[index] +|= 1;
            if (reverse[index] != std.math.maxInt(u32)) delta.hit_picks += 1 else delta.miss_picks += 1;
            if (!keep[index]) {
                if (unique == max_unique) {
                    overflow = true;
                    continue;
                }
                keep[index] = true;
                ordered[unique] = index;
                unique += 1;
            }
            appearances[index] +%= 1;
        }
        var misses: [max_unique]usize = undefined;
        var missing: usize = 0;
        for (ordered[0..unique]) |expert| {
            if (reverse[expert] != std.math.maxInt(u32)) {
                stamps[reverse[expert]] = misc[0];
            } else {
                misses[missing] = expert;
                missing += 1;
            }
        }
        const scores = if (record.recentEnabled()) recent else counts;
        var cursor: usize = 1;
        while (cursor < missing) : (cursor += 1) {
            const item = misses[cursor];
            var insert = cursor;
            while (insert > 0) : (insert -= 1) {
                const previous = misses[insert - 1];
                const hotter_item = if (scores[item] != scores[previous]) scores[item] > scores[previous] else if (appearances[item] != appearances[previous]) appearances[item] > appearances[previous] else item < previous;
                if (!hotter_item) break;
                misses[insert] = previous;
            }
            misses[insert] = item;
        }
        if (overflow) {
            delta.overflow = 1;
        } else for (misses[0..missing]) |expert| {
            var victim: ?usize = null;
            for (ids, 0..) |raw, line| {
                if (raw == std.math.maxInt(u32)) {
                    victim = line;
                    break;
                }
                if (keep[raw]) continue;
                if (victim) |previous| {
                    const old = ids[previous];
                    if (scores[raw] > scores[old] or (scores[raw] == scores[old] and (stamps[line] > stamps[previous] or (stamps[line] == stamps[previous] and line < previous)))) continue;
                }
                victim = line;
            }
            if (victim) |line| {
                if (ids[line] != std.math.maxInt(u32)) {
                    reverse[ids[line]] = std.math.maxInt(u32);
                    delta.evictions += 1;
                }
                ids[line] = @intCast(expert);
                reverse[expert] = @intCast(line);
                stamps[line] = misc[0];
                delta.copies += 1;
            } else {
                delta.inplace += 1;
                for (record.picks()) |raw| {
                    if (raw == expert) delta.mapped_picks += 1;
                }
            }
        }
        inline for (.{ .{ 2, "hit_picks" }, .{ 3, "miss_picks" }, .{ 4, "copies" }, .{ 5, "evictions" }, .{ 6, "overflow" }, .{ 7, "inplace" } }) |field| {
            misc[field[0]] +%= @truncate(@field(delta, field[1]));
        }
        return delta;
    }
};

pub fn verifyTrace(allocator: std.mem.Allocator, view: trace.View) !Verification {
    return verifyTraceWithFinal(allocator, view, null);
}

pub fn verifyTraceWithFinal(allocator: std.mem.Allocator, view: trace.View, final_words: ?[]const u32) !Verification {
    if (view.dropped() != 0 or view.droppedCheckpoints() != 0) return error.TraceDroppedRecords;
    if (view.words[12] != 0) return error.TraceDiscontinuous;
    if (view.words[13] != 0) return error.TraceInvalidLaunch;
    if (view.recorded() == 0) return error.EmptyTrace;
    var state = try ExactState.init(allocator, view.words[4], view.initialSeed());
    defer state.deinit();
    var total: Delta = .{};
    var checkpoint_index: usize = 0;
    for (0..view.recorded() + 1) |index| {
        while (checkpoint_index < view.checkpoints()) {
            const checkpoint = try view.checkpoint(checkpoint_index);
            if (checkpoint.roundIndex() != index) break;
            if (!std.mem.eql(u32, state.words, checkpoint.before())) return error.PlannerCheckpointMismatch;
            var incoming = try ExactState.init(allocator, state.lines, checkpoint.after());
            std.mem.swap(ExactState, &state, &incoming);
            incoming.deinit();
            checkpoint_index += 1;
        }
        if (index == view.recorded()) break;
        const record = try view.round(index);
        const delta = try state.replay(record);
        if (!std.mem.eql(u32, state.words[layout.miscOffset(state.lines)..], record.misc()) or delta.mapped_picks != record.mappedPicks()) return error.PlannerCounterMismatch;
        try add(&total, delta);
    }
    if (!std.mem.eql(u32, state.words, view.terminalSeed())) return error.PlannerFinalStateMismatch;
    if (final_words) |expected| {
        if (!std.mem.eql(u32, state.words, expected)) return error.PlannerFinalStateMismatch;
    }
    if (total.overflow != 0) return error.PlannerOverflow;
    return .{ .rounds = view.recorded(), .matched_transitions = view.recorded(), .delta = total };
}

fn add(total: *Delta, delta: Delta) !void {
    inline for (.{ "hit_picks", "miss_picks", "copies", "evictions", "inplace", "mapped_picks", "overflow" }) |field| {
        @field(total, field) = try std.math.add(u64, @field(total, field), @field(delta, field));
    }
}

fn hotter(state: *const State, a: Entry, b: Entry) bool {
    const a_count = state.counts[@intCast(a.expert)];
    const b_count = state.counts[@intCast(b.expert)];
    if (a_count != b_count) return a_count > b_count;
    if (a.appearances != b.appearances) return a.appearances > b.appearances;
    if (a.expert != b.expert) return a.expert < b.expert;
    return false;
}

fn colder(state: *const State, a: usize, b: usize) bool {
    const a_id: usize = @intCast(state.line_ids[a]);
    const b_id: usize = @intCast(state.line_ids[b]);
    if (state.counts[a_id] != state.counts[b_id]) return state.counts[a_id] < state.counts[b_id];
    if (state.stamps[a] != state.stamps[b]) return state.stamps[a] < state.stamps[b];
    return a > b;
}

test "planner replay reproduces current lifetime-count admission and eviction" {
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 5;
    counts[1] = 3;
    var state = try State.init(std.testing.allocator, .{ .line_ids = &.{ 0, 1 }, .stamps = &.{ 1, 2 }, .counts = counts, .now = 2 });
    defer state.deinit();

    const delta = try state.replay(.{ .rows = 1, .valid_picks = 6, .entries = &.{ .{ .expert = 2, .appearances = 5 }, .{ .expert = 3, .appearances = 1 } } });
    try std.testing.expectEqual(Delta{ .miss_picks = 6, .copies = 2, .evictions = 2 }, delta);
    try std.testing.expectEqualSlices(i32, &.{ 3, 2 }, state.line_ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3 }, state.stamps);
    try std.testing.expectEqual(@as(u32, 5), state.counts[2]);
}

test "planner replay counts repeated slot misses and preserves held lines under overflow pressure" {
    var counts: [expert_count]u32 = @splat(0);
    counts[0] = 9;
    var state = try State.init(std.testing.allocator, .{ .line_ids = &.{0}, .stamps = &.{5}, .counts = counts, .now = 5 });
    defer state.deinit();

    const delta = try state.replay(.{ .rows = 1, .valid_picks = 4, .entries = &.{ .{ .expert = 0, .appearances = 3 }, .{ .expert = 1, .appearances = 1 } } });
    try std.testing.expectEqual(Delta{ .hit_picks = 3, .miss_picks = 1, .inplace = 1, .mapped_picks = 1 }, delta);
    try std.testing.expectEqualSlices(i32, &.{0}, state.line_ids);
    try std.testing.expectEqual(@as(u32, 6), state.stamps[0]);
}

test "planner replay consumes empty lines first and rejects incomplete evidence" {
    var state = try State.init(std.testing.allocator, .{ .line_ids = &.{ -1, -1 }, .stamps = &.{ 90, 91 }, .counts = @splat(0), .now = 0 });
    defer state.deinit();
    const delta = try state.replay(.{ .rows = 1, .valid_picks = 2, .entries = &.{ .{ .expert = 10, .appearances = 1 }, .{ .expert = 11, .appearances = 1 } } });
    try std.testing.expectEqual(Delta{ .miss_picks = 2, .copies = 2 }, delta);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11 }, state.line_ids);
    try std.testing.expectError(error.InvalidRound, state.replay(.{ .rows = 1, .valid_picks = 1, .entries = &.{.{ .expert = 12, .appearances = 1 }}, .planner_overflow = true }));
}
