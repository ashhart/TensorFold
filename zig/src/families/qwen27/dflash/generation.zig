//! Host-controlled draft rounds keep only consumed rows and leave the final emitted bonus token pending.
const std = @import("std");
const mtl = @import("metal");
const TargetSession = @import("../session.zig");
const Runner = @import("../decode_round.zig").Runner;
const Round = @import("../round_plan.zig").Round;
const Draft = @import("runtime_model.zig").Model;
pub const Proposer = struct { ptr: *anyopaque, call: *const fn (*anyopaque, u32, u32) anyerror!@import("selector.zig").Tree };
pub const Result = struct { tokens: []u32, rounds: usize, verified: usize, matched: usize, min_rows: u32 = 0 };
pub const Generation = struct {
    allocator: std.mem.Allocator,
    runner: *Runner,
    draft: *Draft,
    kept: mtl.Buffer,
    taps: mtl.Buffer,
    proposer: ?Proposer = null,
    matcher: @import("round_gpu.zig").Matcher,
    commit: @import("core").tree_commit_gpu.Ops,
    pub fn init(a: std.mem.Allocator, runner: *Runner, draft: *Draft) !Generation {
        if (runner.model != draft.backend.target or runner.model.config.hidden != draft.graph.config.hidden or runner.model.config.layers != draft.graph.config.target_layers) return error.TargetBinding;
        runner.capture_taps = true;
        for (draft.graph.config.taps, &runner.taps.ids) |id, *to| to.* = id;
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const kept = try runner.model.device.buffer(128 * 4, options);
        errdefer kept.deinit();
        const taps = try runner.model.device.buffer(128 * @as(usize, draft.graph.config.tapWidth()) * 2, options);
        errdefer taps.deinit();
        const matcher = try @import("round_gpu.zig").Matcher.init(runner.model.device);
        errdefer matcher.deinit();
        const commit = try @import("core").tree_commit_gpu.Ops.init(runner.model.device);
        return .{ .allocator = a, .runner = runner, .draft = draft, .kept = kept, .taps = taps, .matcher = matcher, .commit = commit };
    }
    pub fn deinit(g: Generation) void {
        g.commit.deinit();
        g.matcher.deinit();
        g.kept.deinit();
        g.taps.deinit();
    }
    fn absorb(g: *Generation, path: []const u32) !void {
        @memcpy(g.kept.slice(u32, path.len), path);
        const cb = g.runner.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try g.runner.taps.accepted(g.runner.model.glue, e, path, .{ .buffer = g.kept }, .{ .buffer = g.taps });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.DraftTapGpuFailure;
        try g.draft.absorb(.{ .buf = g.taps }, @intCast(path.len));
        if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
    }
    pub fn advance(g: *Generation, token: u32) !void {
        try (TargetSession.Session{ .runner = g.runner }).step(token);
        try g.absorb(&.{0});
    }
    pub fn promptChunk(g: *Generation, ids: []const u32, last: bool) !void {
        if (g.runner.failed or g.draft.backend.failed or g.draft.committed_end != g.runner.offsets[0] or ids.len == 0 or ids.len > 128) return error.BadDraftContext;
        try (TargetSession.Session{ .runner = g.runner }).promptChunk(ids, last);
        var path: [128]u32 = undefined;
        for (path[0..ids.len], 0..) |*index, i| index.* = @intCast(i);
        try g.absorb(path[0..ids.len]);
    }
    /// `promptChunk` with decoded rows' arithmetic (an earlier reply's tokens).
    pub fn decodeChunk(g: *Generation, ids: []const u32, last: bool) !void {
        if (g.runner.failed or g.draft.backend.failed or g.draft.committed_end != g.runner.offsets[0] or ids.len == 0 or ids.len > 128) return error.BadDraftContext;
        try (TargetSession.Session{ .runner = g.runner }).decodeChunk(ids, last);
        var path: [128]u32 = undefined;
        for (path[0..ids.len], 0..) |*index, i| index.* = @intCast(i);
        try g.absorb(path[0..ids.len]);
    }
    pub fn prefill(g: *Generation, ids: []const u32, chunk: usize) !void {
        if (g.runner.failed or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != 0 or g.runner.offsets[0] != 0 or chunk == 0 or chunk > 128) return error.BadDraftContext;
        var first: usize = 0;
        while (first < ids.len) {
            const count = @min(ids.len - first, chunk);
            try g.promptChunk(ids[first..][0..count], first + count == ids.len);
            first += count;
        }
    }
    fn accept(g: *Generation, round: *const Round, ids: []const u32, parents: []const i32, budget: u32, eos: []const u32) !@import("core").tree_round.Decoded {
        errdefer g.runner.failed = true;
        try g.runner.taps.complete();
        const cb = g.runner.model.queue.commandBuffer();
        const e = cb.compute(.concurrent); // the matcher and keep fence their own dependencies; taps and head read only the matcher's result
        var ended = false;
        errdefer if (!ended) e.end();
        const logits = g.runner.model.frame.get(.logits);
        const width: u32 = @intCast(g.runner.model.config.vocab);
        const shape = try g.matcher.encode(e, .{ .buf = logits.buffer, .off = logits.offset }, ids, parents, width, budget, eos);
        const result = @import("core").tree_commit_gpu.Ref{ .buf = g.matcher.buffers[4] };
        try g.runner.keepDevice(round, e, g.commit, result);
        try g.commit.taps(e, result, .{ .buf = g.runner.taps.buffer }, .{ .buf = g.taps }, .{ .rows = shape.rows, .width = g.runner.taps.width, .capacity = g.runner.taps.capacity, .planes = 5 });
        try g.commit.head(e, result, .{ .buf = logits.buffer, .off = logits.offset }, .{ .rows = shape.rows, .width = width, .stride = width });
        e.end();
        ended = true;
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.DraftKeepGpuFailure;
        const accepted = try g.matcher.read(shape);
        try g.runner.completeDeviceKeep(round, accepted.path);
        try g.draft.absorb(.{ .buf = g.taps }, @intCast(accepted.path.len));
        if (g.draft.committed_end != g.runner.offsets[0]) return error.BadDraftPosition;
        return accepted;
    }
    pub fn run(g: *Generation, count: usize, force_length: bool, nodes: usize) !Result {
        return g.runImpl(count, force_length, nodes, null, false);
    }
    pub fn runBatch(g: *Generation, count: usize, nodes: usize, eos: []const u32, already_delivered: bool) !Result {
        if (count == 0 or count > 16 or eos.len > 16) return error.BadDraftOptions;
        return g.runImpl(count, false, nodes, eos, already_delivered);
    }
    pub fn runRound(g: *Generation, remaining: u32, nodes: usize, eos: []const u32, already_delivered: bool) !Result {
        if (g.runner.failed or g.runner.active != null or g.runner.offsets[0] == 0 or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != g.runner.offsets[0] or g.draft.committed_end == 0 or nodes == 0 or nodes > 15 or eos.len > 16) return error.BadDraftOptions;
        for (eos) |token| if (token >= g.runner.model.config.vocab) return error.BadDraftOptions;
        if (remaining == 0) return .{ .tokens = try g.allocator.alloc(u32, 0), .rounds = 0, .verified = 0, .matched = 0 };
        const pending = try (TargetSession.Session{ .runner = g.runner }).greedy();
        if (!already_delivered) {
            const tokens = try g.allocator.alloc(u32, 1);
            tokens[0] = pending;
            return .{ .tokens = tokens, .rounds = 0, .verified = 0, .matched = 0 };
        }
        if (std.mem.indexOfScalar(u32, eos, pending) != null) return .{ .tokens = try g.allocator.alloc(u32, 0), .rounds = 0, .verified = 0, .matched = 0 };
        const available = g.runner.capacity - g.runner.offsets[0];
        if (available == 0) return error.ContextFull;
        const offered: u32 = @intCast(@min(nodes, available - 1));
        var tree = if (g.proposer) |proposer| try proposer.call(proposer.ptr, pending, offered) else try g.draft.propose(pending, offered);
        defer tree.deinit();
        const parents = try tree.verifyParents(g.allocator);
        defer g.allocator.free(parents);
        const ids = try g.allocator.alloc(u32, tree.tokens.len + 1);
        defer g.allocator.free(ids);
        ids[0] = pending;
        @memcpy(ids[1..], tree.tokens);
        var round = try Round.init(g.allocator, &.{.{ .slot = 0, .start = g.runner.offsets[0], .capacity = g.runner.capacity, .ids = ids, .parents = parents }}, @intCast(g.runner.model.config.conv_kernel), g.runner.slots, @intCast(g.runner.model.config.vocab));
        defer round.deinit();
        try g.runner.verifyHead(&round, .all);
        const accepted = try g.accept(&round, ids, parents, remaining, eos);
        if (accepted.path.len == 0 or accepted.tokens.len == 0 or accepted.tokens.len > @min(16, remaining)) return error.EmptyAcceptedDraft;
        return .{ .tokens = try g.allocator.dupe(u32, accepted.tokens), .rounds = 1, .verified = ids.len, .matched = accepted.matched, .min_rows = @intCast(ids.len) };
    }
    fn runImpl(g: *Generation, count: usize, force_length: bool, nodes: usize, eos_override: ?[]const u32, already_delivered: bool) !Result {
        if (g.runner.failed or g.runner.active != null or g.runner.offsets[0] == 0 or g.draft.backend.failed or g.draft.backend.frame.active or g.draft.backend.transaction != null or g.draft.committed_end != g.runner.offsets[0] or g.draft.committed_end == 0 or nodes == 0 or nodes > 15 or count > 65536) return error.BadDraftOptions;
        const eos: []const u32 = if (force_length) &.{} else eos_override orelse g.runner.model.config.eos[0..g.runner.model.config.eos_count];
        for (eos) |token| if (token >= g.runner.model.config.vocab) return error.BadDraftOptions;
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(g.allocator);
        var rounds: usize = 0;
        var verified: usize = 0;
        var matched: usize = 0;
        var min_rows: u32 = 0;
        if (count == 0) return .{ .tokens = try tokens.toOwnedSlice(g.allocator), .rounds = 0, .verified = 0, .matched = 0 };
        const session = TargetSession.Session{ .runner = g.runner };
        var pending = try session.greedy();
        if (!already_delivered) try tokens.append(g.allocator, pending);
        while (tokens.items.len < count and std.mem.indexOfScalar(u32, eos, pending) == null) {
            const available = g.runner.capacity - g.runner.offsets[0];
            if (available == 0) return error.ContextFull;
            const offered: u32 = @intCast(@min(nodes, available - 1));
            var tree = if (g.proposer) |proposer| try proposer.call(proposer.ptr, pending, offered) else try g.draft.propose(pending, offered);
            defer tree.deinit();
            const parents = try tree.verifyParents(g.allocator);
            defer g.allocator.free(parents);
            const ids = try g.allocator.alloc(u32, tree.tokens.len + 1);
            defer g.allocator.free(ids);
            ids[0] = pending;
            @memcpy(ids[1..], tree.tokens);
            var round = try Round.init(g.allocator, &.{.{ .slot = 0, .start = g.runner.offsets[0], .capacity = g.runner.capacity, .ids = ids, .parents = parents }}, @intCast(g.runner.model.config.conv_kernel), g.runner.slots, @intCast(g.runner.model.config.vocab));
            defer round.deinit();
            try g.runner.verifyHead(&round, .all);
            rounds += 1;
            verified += ids.len;
            min_rows = if (min_rows == 0) @intCast(ids.len) else @min(min_rows, @as(u32, @intCast(ids.len)));
            const accepted = try g.accept(&round, ids, parents, @intCast(count - tokens.items.len), eos);
            try tokens.appendSlice(g.allocator, accepted.tokens);
            matched += accepted.matched;
            if (accepted.path.len == 0) return error.EmptyAcceptedDraft;
            if (accepted.pending) |next| pending = next else break;
        }
        return .{ .tokens = try tokens.toOwnedSlice(g.allocator), .rounds = rounds, .verified = verified, .matched = matched, .min_rows = min_rows };
    }
};

test "true request budget keeps a verified non-prefix continuation across delivery boundaries" {
    const r = @import("core").tree_round;
    const ids = [_]u32{ 1, 9, 2, 9, 3, 9, 4 };
    const parents = [_]i32{ -1, 0, 0, 2, 2, 4, 4 };
    const picks = [_]r.Pick{ .{ .token = 2 }, .{ .token = 0 }, .{ .token = 3 }, .{ .token = 0 }, .{ .token = 4 }, .{ .token = 0 }, .{ .token = 5 } };
    var full = try r.oracle(.{ .rows = ids.len, .vocab = 10, .budget = 33 }, &ids, &parents, &picks, &.{});
    const accepted = try r.decode(&full, .{ .rows = ids.len, .vocab = 10, .budget = 33 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4, 6 }, accepted.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4, 5 }, accepted.tokens);
    try std.testing.expectEqual(@as(?u32, 5), accepted.pending);
    var clipped = try r.oracle(.{ .rows = ids.len, .vocab = 10, .budget = 3 }, &ids, &parents, &picks, &.{});
    const terminal = try r.decode(&clipped, .{ .rows = ids.len, .vocab = 10, .budget = 3 });
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4 }, terminal.path);
    try std.testing.expectEqualSlices(u32, &.{ 2, 3, 4 }, terminal.tokens);
    try std.testing.expect(terminal.pending == null);
}
