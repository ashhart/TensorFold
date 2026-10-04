//! One stream (Python LaneStream) and the round loop's state for it (Python's dicts keyed by stream id).
const std = @import("std");
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;
const State = @import("depth.zig").State;

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};

pub const Mode = enum { pipe, drain, verify, exit };

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes their tail).
pub const StopCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque, emitted: []const u32) bool,
};

/// Drafts the backend holds for the stream's next round; a tree keeps its tokens and parents on the host.
pub const Held = struct {
    count: u32,
    tokens: ?[]u32 = null,
    parents: ?[]i32 = null,
};

pub const Spec = struct {
    id: []const u8,
    prompt: []const u32,
    max_new: u32,
    eos: []const u32 = &.{},
    sampling: ?Sampling = null,
    drafts: bool = true,
    proposer: ?Proposer = null,
    stop_check: ?StopCheck = null,
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: i64 = -1,
    think_open: ?bool = null, // null: open when a budget is set (the server's rule)
    chunks: []const u32 = &.{}, // where prefill chunks start after 0 (Python's PrefillPlan); empty: the backend's step
};

pub const Stream = struct {
    id: []const u8,
    prompt_len: usize,
    max_new: u32,
    eos: []const u32,
    sampling: ?Sampling,
    drafts: bool,
    proposer: ?Proposer,
    stop_check: ?StopCheck,
    think_budget: u32,
    think_close: []const u32,
    think_end: i64,
    think_open: bool,
    chunks: []const u32,
    context: std.ArrayList(u32) = .empty,
    pending: ?u32 = null,
    force: std.ArrayList(u32) = .empty,
    cache_len: u64 = 0,
    finished: bool = false,
    reason: Reason = .none,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    branch_rows: u64 = 0, // suffix-match branch rows verified (lanes/fill.zig)
    branch_accepted: u64 = 0, // of those, rows on the accepted path
    graft_hit: f64 = 1, // recent share of grafted rows kept; a read of the chain that grafted none counts as a miss
    odds: ?shape.Odds = null, // how often the target took the head's rank-r lane at each depth (head trees)
    lanes: ?shape.Shape = null, // the shape of the held tree drafts
    landed: [shape.max_depth][shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes verified and taken
    picks: [plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks: each tree size, then chain + grafts
    graft_room: ?u32 = null, // grafted rows the planner left this round's window (null: no planner)
    held_levels: u32 = 0, // head levels drafted for the held window
    round_over: ?f64 = null, // the stream's round time beyond its window and head levels (ms, recent): chains
    tree_over: ?f64 = null, // and trees
    measured: [plan_lanes.sizes.len + 1]plan_lanes.Measured = @splat(.{}), // each pick's round times past its table price
    priced: ?struct { choice: usize, ms: f64 } = null, // the pick that shaped the held window, at its table price
    min_rows: u32 = 0,
    paused: bool = false,
    depth: ?State = null,
    mode: ?Mode = null,
    next: ?Held = null,
    inflight: ?u64 = null,
    served: ?u64 = null,
    granted: ?u32 = null,
    copy_width: ?u32 = null,

    pub fn init(gpa: Allocator, spec: Spec) !Stream {
        var s: Stream = .{
            .id = spec.id,
            .prompt_len = spec.prompt.len,
            .max_new = spec.max_new,
            .eos = spec.eos,
            .sampling = spec.sampling,
            .drafts = spec.drafts,
            .proposer = if (spec.drafts) spec.proposer else null,
            .stop_check = spec.stop_check,
            .think_budget = spec.think_budget,
            .think_close = spec.think_close,
            .think_end = spec.think_end,
            .think_open = spec.think_open orelse (spec.think_budget > 0),
            .chunks = spec.chunks,
        };
        try s.context.appendSlice(gpa, spec.prompt);
        return s;
    }

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        s.context.deinit(gpa);
        s.force.deinit(gpa);
        s.dropRoundState(gpa);
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    pub fn dropRoundState(s: *Stream, gpa: Allocator) void {
        if (s.depth) |*d| d.deinit(gpa);
        s.dropHeld(gpa);
        s.depth = null;
        s.mode = null;
        s.inflight = null;
        s.served = null;
        s.granted = null;
        s.copy_width = null;
    }

    pub fn dropHeld(s: *Stream, gpa: Allocator) void {
        if (s.next) |h| {
            if (h.tokens) |t| gpa.free(t);
            if (h.parents) |p| gpa.free(p);
        }
        s.next = null;
        if (s.lanes) |l| l.deinit(gpa);
        s.lanes = null;
    }

    pub fn prompt(s: *const Stream) []const u32 {
        return s.context.items[0..s.prompt_len];
    }

    pub fn emitted(s: *const Stream) []const u32 {
        return s.context.items[s.prompt_len..];
    }

    pub fn budgetLeft(s: *const Stream) i64 {
        return @as(i64, s.max_new) - @as(i64, @intCast(s.emitted().len));
    }

    fn budgetActive(s: *const Stream) bool {
        return s.think_open and s.think_budget > 0 and s.think_close.len > 0;
    }

    /// The index the thinking budget replaces with `think_close[0]`, or null if the block closed first.
    pub fn thinkCut(s: *const Stream, tokens: anytype) ?usize {
        if (!s.budgetActive()) return null;
        const done = s.emitted().len;
        for (tokens, 0..) |t, i| {
            if (done + i + 1 >= s.think_budget) return i;
            if (@as(i64, t) == s.think_end) return null;
        }
        return null;
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    pub fn cutsNext(s: *const Stream) bool {
        return s.budgetActive() and s.emitted().len + 1 >= s.think_budget;
    }

    /// Begin the thinking budget's close: its first token now, the rest forced after it.
    pub fn startClose(s: *Stream, gpa: Allocator) !u32 {
        s.think_open = false;
        s.force.clearRetainingCapacity();
        try s.force.appendSlice(gpa, s.think_close[1..]);
        return s.think_close[0];
    }

    /// Tokens a round may commit before the length limit or the thinking budget's cut.
    pub fn draftRoom(s: *const Stream) i64 {
        var room = s.budgetLeft();
        if (s.budgetActive()) room = @min(room, @as(i64, s.think_budget) - @as(i64, @intCast(s.emitted().len)));
        return room;
    }

    pub fn isEos(s: *const Stream, token: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, token) != null;
    }

    /// Append tokens until the stream finishes; how many landed (a prefix of `tokens`).
    pub fn commit(s: *Stream, gpa: Allocator, tokens: []const u32) !usize {
        var landed: usize = 0;
        for (tokens) |t| {
            if (s.finished) break;
            try s.context.append(gpa, t);
            landed += 1;
            if (@as(i64, t) == s.think_end) s.think_open = false;
            if (s.isEos(t) or s.stopped()) {
                s.finished = true;
                s.reason = .stop;
            } else if (s.emitted().len >= s.max_new) {
                s.finished = true;
                s.reason = .length;
            }
        }
        return landed;
    }

    fn stopped(s: *const Stream) bool {
        const c = s.stop_check orelse return false;
        return c.check(c.ptr, s.emitted());
    }

    pub fn popForce(s: *Stream) ?u32 {
        if (s.force.items.len == 0) return null;
        return s.force.orderedRemove(0);
    }
};

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}
