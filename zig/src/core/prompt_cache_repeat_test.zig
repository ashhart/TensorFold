//! Raw completion repeats add one useful kept mark; chat endpoints, budgets and full keys retain priority.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fake = @import("prompt_cache_fake.zig");
test "repeat cache: first computes, second keeps, third resumes with identical state" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = a };
    var store = pc.Store.init(a, f.snapshots(), .{ .repeat_min = 16, .min_prompt = 0, .lookahead = 1 }, 1024);
    defer store.deinit();
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    const first = try store.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(usize, 0), first.marks.len);
    const want = f.pass(&store, &prompt, first);
    store.repeat_log.note(a, store.rules.repeat_min, &prompt);
    const second = try store.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{19}, second.marks);
    try std.testing.expectEqual(want, f.pass(&store, &prompt, second));
    const third = try store.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 19), third.from);
    try std.testing.expectEqual(want, f.pass(&store, &prompt, third));
    var changed = prompt;
    changed[19] += 1;
    const miss = try store.begin(arena.allocator(), &changed, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), miss.from);
    store.rules.repeat_min = 0;
    const disabled = try store.marks(arena.allocator(), &prompt, 0, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 0), disabled.len);
    store.rules.repeat_min = 16;
    store.budget = 1;
    const small = try store.lookup(arena.allocator(), &prompt, 0, &.{}, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 0), small.marks.len);
    const chat = try store.marks(arena.allocator(), &prompt, 0, 12, &.{}, &.{}, &.{});
    try std.testing.expectEqualSlices(u32, &.{12}, chat);
}
