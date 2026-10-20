//! Failed token publication preserves logical stream state and paired rows, including thinking boundaries.
const std = @import("std");
const Stream = @import("stream.zig").Stream;
const Row = @import("logprob.zig").Row;
const grammar = @import("grammar.zig");
const t = std.testing;
const a = t.allocator;
const close: [64]u32 = @splat(3);
const Rules = struct {
    accepted: usize = 0,
    inject: bool = false,
    fn rules(r: *Rules) grammar.Rules {
        return .{ .ptr = r, .vtable = &.{ .accept = accept, .rollback = rollback, .terminated = terminated, .fill = fill, .drafts = drafts, .free = free } };
    }
    fn self(ptr: *anyopaque) *Rules {
        return @ptrCast(@alignCast(ptr));
    }
    fn accept(ptr: *anyopaque, token: u32) !bool {
        const r = self(ptr);
        if (r.inject) return error.GrammarInjected;
        if (token != 7) return false;
        r.accepted += 1;
        return true;
    }
    fn rollback(ptr: *anyopaque, count: usize) !void {
        self(ptr).accepted -= count;
    }
    fn terminated(_: *anyopaque) bool {
        return false;
    }
    fn fill(_: *anyopaque, words: []u32) !void {
        @memset(words, 1 << 7);
    }
    fn drafts(_: *anyopaque, _: []u32) !usize {
        return 0;
    }
    fn free(_: *anyopaque) void {}
};
fn reservation(which: enum { context, rows, force }) !void {
    var rules: Rules = .{};
    var constraint = grammar.Constraint.init(rules.rules(), 1, null);
    var s = try Stream.init(a, .{ .id = "reserve", .prompt = &.{ 1, 2 }, .max_new = 128, .logprobs = 0, .grammar = &constraint, .think_end = 7, .think_open = true, .think_close = &close });
    defer s.deinit(a);
    s.loop_guard = true;
    try s.force.append(a, 9);
    const tokens: [64]u32 = @splat(7);
    const rows: [64]Row = @splat(.{ .token = 7, .logprob = -1 });
    if (which != .context) try s.context.ensureUnusedCapacity(a, tokens.len);
    if (which == .force) try s.rows.ensureUnusedCapacity(a, rows.len);
    const reason = s.reason;
    var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    try t.expectError(error.OutOfMemory, s.commit(fail.allocator(), &tokens, &rows));
    try t.expect(fail.has_induced_failure);
    try t.expectEqualSlices(u32, &.{ 1, 2 }, s.context.items);
    try t.expectEqual(@as(usize, 0), s.rows.items.len);
    try t.expectEqualSlices(u32, &.{9}, s.force.items);
    try t.expect(rules.accepted == 0 and s.think_open and s.answer_from == null);
    try t.expect(!s.finished and s.problem.len == 0 and s.loop_period == null);
    try t.expectEqual(reason, s.reason);
    try t.expectEqual(@as(usize, 64), try s.commit(a, &tokens, &rows));
    try t.expectEqualSlices(u32, &tokens, s.emitted());
    try t.expectEqual(@as(usize, 64), s.rows.items.len);
    try t.expectEqual(@as(usize, 64), rules.accepted);
}
test "commit storage: context reservation failure publishes nothing and retries" {
    try reservation(.context);
}
test "commit storage: row reservation failure publishes nothing and retries" {
    try reservation(.rows);
}
test "commit storage: loop-close reservation failure preserves pending force and retries" {
    try reservation(.force);
}
test "commit storage: grammar rejection publishes no token row or thinking end" {
    var rules: Rules = .{};
    var constraint = grammar.Constraint.init(rules.rules(), 1, null);
    var s = try Stream.init(a, .{ .id = "reject", .prompt = &.{ 1, 2 }, .max_new = 4, .logprobs = 0, .grammar = &constraint, .think_end = 8, .think_open = true });
    defer s.deinit(a);
    const row = [_]Row{.{ .token = 8, .logprob = -1 }};
    try t.expectEqual(@as(usize, 0), try s.commit(a, &.{8}, &row));
    try t.expectEqualSlices(u32, &.{ 1, 2 }, s.context.items);
    try t.expectEqual(@as(usize, 0), s.rows.items.len);
    try t.expect(rules.accepted == 0 and s.think_open and s.answer_from == null);
    try t.expect(s.finished and s.reason == .@"error" and s.problem.len > 0);
}
test "commit storage: a rejected suffix retains exactly the previously accepted prefix" {
    var rules: Rules = .{};
    var constraint = grammar.Constraint.init(rules.rules(), 1, null);
    var s = try Stream.init(a, .{ .id = "prefix", .prompt = &.{ 1, 2 }, .max_new = 4, .logprobs = 0, .grammar = &constraint, .think_end = 8, .think_open = true });
    defer s.deinit(a);
    const rows = [_]Row{ .{ .token = 7, .logprob = -1 }, .{ .token = 8, .logprob = -2 } };
    try t.expectEqual(@as(usize, 1), try s.commit(a, &.{ 7, 8 }, &rows));
    try t.expectEqualSlices(u32, &.{7}, s.emitted());
    try t.expectEqual(@as(usize, 1), s.rows.items.len);
    try t.expectEqual(@as(u32, 7), s.rows.items[0].token);
    try t.expect(rules.accepted == 1 and s.think_open and s.answer_from == null);
    try t.expect(s.finished and s.reason == .@"error");
}
test "commit storage: a grammar error leaves publication untouched for retry" {
    var rules: Rules = .{ .inject = true };
    var constraint = grammar.Constraint.init(rules.rules(), 1, null);
    var s = try Stream.init(a, .{ .id = "grammar-error", .prompt = &.{ 1, 2 }, .max_new = 4, .logprobs = 0, .grammar = &constraint, .think_end = 7, .think_open = true });
    defer s.deinit(a);
    const row = [_]Row{.{ .token = 7, .logprob = -1 }};
    try t.expectError(error.GrammarInjected, s.commit(a, &.{7}, &row));
    try t.expectEqualSlices(u32, &.{ 1, 2 }, s.context.items);
    try t.expectEqual(@as(usize, 0), s.rows.items.len);
    try t.expect(rules.accepted == 0 and s.think_open and s.answer_from == null and !s.finished);
    rules.inject = false;
    try t.expectEqual(@as(usize, 1), try s.commit(a, &.{7}, &row));
    try t.expectEqual(@as(?usize, 1), s.answer_from);
}
test "commit storage: failed explicit thinking close retains its forced suffix" {
    var s = try Stream.init(a, .{ .id = "close", .prompt = &.{ 1, 2 }, .max_new = 128, .think_close = &close, .think_open = true });
    defer s.deinit(a);
    try s.force.append(a, 9);
    var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    try t.expectError(error.OutOfMemory, s.startClose(fail.allocator()));
    try t.expect(s.think_open and s.answer_from == null);
    try t.expectEqualSlices(u32, &.{9}, s.force.items);
    try t.expectEqual(@as(u32, 3), try s.startClose(a));
    try t.expect(!s.think_open);
    try t.expectEqualSlices(u32, close[1..], s.force.items);
}
test "commit storage: empty and finished branches allocate and advance no grammar" {
    var rules: Rules = .{};
    var constraint = grammar.Constraint.init(rules.rules(), 1, null);
    var s = try Stream.init(a, .{ .id = "empty", .prompt = &.{ 1, 2 }, .max_new = 1, .grammar = &constraint, .think_close = &close, .think_open = true });
    defer s.deinit(a);
    s.loop_guard = true;
    var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    try t.expectEqual(@as(usize, 0), try s.commit(fail.allocator(), &.{}, &.{}));
    try t.expect(rules.accepted == 0 and !fail.has_induced_failure);
    try t.expectEqual(@as(usize, 1), try s.commit(a, &.{7}, &.{}));
    try t.expect(s.finished);
    try t.expectEqual(@as(usize, 0), try s.commit(fail.allocator(), &.{ 7, 7 }, &.{}));
    try t.expect(rules.accepted == 1 and !fail.has_induced_failure);
    try t.expectEqualSlices(u32, &.{7}, s.emitted());
}

fn indexReservation(map: bool) !void {
    var rules: Rules = .{};
    var constraint = grammar.Constraint.init(rules.rules(), 1, null);
    var s = try Stream.init(a, .{ .id = "index reserve", .prompt = &.{ 1, 2 }, .max_new = 100, .logprobs = 0, .grammar = &constraint, .think_end = 8, .think_open = true, .think_close = &close, .loop_guard = true });
    defer s.deinit(a);
    const tokens: [33]u32 = @splat(7);
    const rows: [33]Row = @splat(.{ .token = 7, .logprob = -1 });
    try s.context.ensureUnusedCapacity(a, tokens.len);
    try s.rows.ensureUnusedCapacity(a, rows.len);
    try s.force.append(a, 9);
    try s.force.ensureTotalCapacity(a, close.len);
    if (map) try s.long_loop.prev.ensureTotalCapacity(a, tokens.len);
    var fail = t.FailingAllocator.init(a, .{ .fail_index = 0, .resize_fail_index = 0 });
    try t.expectError(error.OutOfMemory, s.commit(fail.allocator(), &tokens, &rows));
    try t.expect(fail.has_induced_failure);
    try t.expectEqualSlices(u32, &.{ 1, 2 }, s.context.items);
    try t.expectEqual(@as(usize, 0), s.rows.items.len);
    try t.expectEqualSlices(u32, &.{9}, s.force.items);
    try t.expect(rules.accepted == 0 and s.think_open and s.answer_from == null);
    try t.expect(!s.finished and s.reason == .none and s.problem.len == 0 and s.loop_period == null);
    try t.expect(s.long_loop.upto == 0 and s.long_loop.prev.items.len == 0 and s.long_loop.last.count() == 0);
    try t.expectEqual(@as(usize, 33), try s.commit(a, &tokens, &rows));
    try t.expectEqualSlices(u32, &tokens, s.emitted());
    try t.expectEqual(@as(usize, 33), rules.accepted);
    try t.expectEqual(@as(usize, 33), s.rows.items.len);
    try t.expectEqual(@as(usize, 33), s.long_loop.upto);
}
test "commit storage: index rows reserve before grammar and publication, then retry" {
    try indexReservation(false);
}
test "commit storage: index map reserves before grammar and publication, then retries" {
    try indexReservation(true);
}
