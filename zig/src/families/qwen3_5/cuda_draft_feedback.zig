//! Earlier target draws from counterfactual branches only propose tokens; target verification stays authoritative.
const std = @import("std");
const tree = @import("cuda_draft_tree.zig");
const most = @import("cuda_engine.zig").most_window;
const capacity = 4 * most;
const Entry = struct { start: u64, position: u64, previous: u32, token: u32 };
const Hints = struct { ids: [4]u32 = undefined, count: [4]u32 = @splat(0), len: usize = 0 };

/// A bounded stream-local proposal record, cleared whenever its context rewinds or a pooled context is reset.
pub const State = struct {
    entries: [capacity]Entry = undefined,
    len: usize = 0,
    next: usize = 0,
    last: u64 = 0,

    pub fn record(s: *State, pending: u32, tokens: []const u32, positions: []const u64, draws: []const u32) !void {
        if (positions.len == 0 or positions.len > most or positions.len != draws.len or tokens.len + 1 != positions.len)
            return error.BadDraftFeedback;
        const start = positions[0];
        if (start == 0) return error.BadDraftFeedback;
        if (s.len > 0 and start <= s.last) s.* = .{};
        s.last = start;
        for (positions, draws, 0..) |position, token, row| {
            s.entries[s.next] = .{ .start = start, .position = position, .previous = if (row == 0) pending else tokens[row - 1], .token = token };
            s.next = (s.next + 1) % capacity;
            s.len = @min(s.len + 1, capacity);
        }
    }

    /// The strongest four first-order hints within two positions, from earlier rounds no more than 16 tokens old.
    fn hints(s: *const State, anchor: u64, position: u64, previous: u32) Hints {
        var ids: [capacity]u32 = undefined;
        var counts: [capacity]u32 = @splat(0);
        var count: usize = 0;
        for (s.entries[0..s.len]) |e| {
            if (e.start >= anchor or anchor - e.start > 16 or e.previous != previous) continue;
            const distance = if (e.position > position) e.position - position else position - e.position;
            if (distance > 2) continue;
            const at = std.mem.indexOfScalar(u32, ids[0..count], e.token) orelse blk: {
                ids[count] = e.token;
                count += 1;
                break :blk count - 1;
            };
            counts[at] += 1;
        }
        var result: Hints = .{};
        while (result.len < @min(count, result.ids.len)) {
            var best: usize = 0;
            for (1..count) |i| if (counts[i] > counts[best] or (counts[i] == counts[best] and ids[i] < ids[best])) {
                best = i;
            };
            result.ids[result.len] = ids[best];
            result.count[result.len] = counts[best];
            result.len += 1;
            counts[best] = 0;
        }
        return result;
    }
};

pub fn weight(raw: ?[]const u8) !f64 {
    const value = raw orelse return 0;
    const result = std.fmt.parseFloat(f64, value) catch return error.BadDraftFeedback;
    if (!std.math.isFinite(result) or result < 0 or result > 1) return error.BadDraftFeedback;
    return result;
}

pub fn configure(d: anytype) !void {
    d.feedback_weight = try weight(try @import("sys").environment.given("TF_DRAFT_FEEDBACK_WEIGHT"));
    if (d.feedback_weight > 0 and (d.cooperative_lanes != 1 or d.refine)) return error.FeedbackNeedsSingleDraftView;
}

/// These are the target's already-read draws; no extra target, draft, head projection or GPU transfer is issued.
pub fn observe(draft: anytype, window: anytype, result: anytype) !void {
    const d = draft orelse return;
    if (d.feedback_weight == 0) return;
    const ctx = d.streams.get(window.stream) orelse return;
    try ctx.feedback.record(window.pending, window.tokens, window.positions, result.sampled);
}

const Expert = struct {
    c: tree.Candidates,
    state: *const State,
    position: u64,
    alpha: f64,

    pub fn choices(e: Expert, row: usize, previous: u32) tree.Choices {
        var result = tree.baseChoices(e.c, row, previous);
        const h = e.state.hints(e.position, e.position + row, previous);
        if (h.len == 0) return result;
        var total: u32 = 0;
        for (h.count[0..h.len]) |n| total += n;
        for (result.logp[0..result.len]) |*p| p.* = (1 - e.alpha) * @exp(p.*);
        for (h.ids[0..h.len], h.count[0..h.len]) |id, n| {
            const at = std.mem.indexOfScalar(u32, result.tokens[0..result.len], id) orelse blk: {
                result.tokens[result.len] = id;
                result.logp[result.len] = 0;
                result.len += 1;
                break :blk result.len - 1;
            };
            result.logp[at] += e.alpha * @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(total));
        }
        for (result.logp[0..result.len]) |*p| p.* = @log(p.*);
        return result;
    }
};

pub fn grow(c: tree.Candidates, state: *const State, position: u64, alpha: f64, pending: u32, rows: usize, budget: usize, tokens: []u32, parents: []i32, own: []f64) usize {
    if (alpha == 0 or state.len == 0) return tree.grow(c, pending, rows, budget, tokens, parents, own);
    return tree.growWith(Expert{ .c = c, .state = state, .position = position, .alpha = alpha }, pending, rows, budget, tokens, parents, own);
}

test "feedback weights are explicit, finite probabilities" {
    try std.testing.expectEqual(@as(f64, 0), try weight(null));
    try std.testing.expectEqual(@as(f64, 0.2), try weight("0.2"));
    for ([_][]const u8{ "", "nan", "inf", "-0.1", "1.1", "true" }) |raw|
        try std.testing.expectError(error.BadDraftFeedback, weight(raw));
}

test "feedback retains earlier counterfactual draws only within its time and position limits" {
    var state: State = .{};
    try state.record(10, &.{ 20, 21 }, &.{ 100, 101, 102 }, &.{ 30, 31, 32 });
    const h = state.hints(101, 103, 20);
    try std.testing.expectEqual(@as(usize, 1), h.len);
    try std.testing.expectEqual(@as(u32, 31), h.ids[0]);
    try std.testing.expectEqual(@as(usize, 0), state.hints(100, 101, 20).len);
    try std.testing.expectEqual(@as(usize, 0), state.hints(101, 104, 20).len);
    try std.testing.expectEqual(@as(usize, 0), state.hints(117, 101, 20).len);
    try std.testing.expectError(error.BadDraftFeedback, state.record(1, &.{}, &.{}, &.{}));
    try state.record(1, &.{}, &.{80}, &.{2});
    try std.testing.expectEqual(@as(usize, 1), state.len);
    try std.testing.expectEqual(@as(usize, 0), state.hints(101, 101, 20).len);
}

test "feedback storage is bounded and replaces its oldest predictions" {
    var state: State = .{};
    for (0..capacity + 5) |i| try state.record(1, &.{}, &.{100 + i}, &.{@intCast(i)});
    try std.testing.expectEqual(@as(usize, capacity), state.len);
    const h = state.hints(100 + capacity + 5, 100 + capacity + 4, 1);
    try std.testing.expect(h.len > 0 and h.len <= 4);
    try std.testing.expect(std.mem.indexOfScalar(u32, h.ids[0..h.len], @intCast(capacity + 4)) != null);
}

test "feedback adds a rejected target token to the draft pool without committing it" {
    var words: [43 * 4]u8 = @splat(0);
    const w = std.mem.bytesAsSlice(u32, &words);
    w[2] = 0;
    w[3] = 1;
    w[22] = @bitCast(@as(f32, 4));
    w[23] = @bitCast(@as(f32, 0));
    const zeros: [8]u16 = @splat(0);
    const c: tree.Candidates = .{ .words = &words, .projected = std.mem.sliceAsBytes(zeros[0..1]), .topk = 2, .rank = 1, .predecessor = &zeros, .successor = &zeros };
    var state: State = .{};
    var tokens: [2]u32 = undefined;
    var parents: [2]i32 = undefined;
    var own: [2]f64 = undefined;
    var bt: [2]u32 = undefined;
    var bp: [2]i32 = undefined;
    var bo: [2]f64 = undefined;
    const count = tree.grow(c, 0, 1, 2, &bt, &bp, &bo);
    try std.testing.expectEqual(count, grow(c, &state, 10, 0.5, 0, 1, 2, &tokens, &parents, &own));
    try std.testing.expectEqualSlices(u32, &bt, &tokens);
    try std.testing.expectEqualSlices(i32, &bp, &parents);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&bo), std.mem.sliceAsBytes(&own));
    try state.record(4, &.{0}, &.{ 9, 10 }, &.{ 5, 2 });
    const expert: Expert = .{ .c = c, .state = &state, .position = 10, .alpha = 0.5 };
    const choices = expert.choices(0, 0);
    var sum: f64 = 0;
    for (choices.logp[0..choices.len]) |p| sum += @exp(p);
    try std.testing.expectApproxEqAbs(@as(f64, 1), sum, 1e-12);
    try std.testing.expectEqual(@as(usize, 2), grow(c, &state, 10, 0.5, 0, 1, 2, &tokens, &parents, &own));
    try std.testing.expectEqual(@as(u32, 2), tokens[0]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1 }, &parents);
    try std.testing.expectEqual(count, grow(c, &state, 10, 0, 0, 1, 2, &tokens, &parents, &own));
    try std.testing.expectEqualSlices(u32, &bt, &tokens);
}
