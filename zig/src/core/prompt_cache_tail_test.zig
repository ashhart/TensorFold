//! A planned store's tail: a prompt that names no history keeps its last chunk start; a chat keeps its history's alone.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const fake = @import("prompt_cache_fake.zig");
test "tail cache: a raw prompt keeps its last chunk start, and its repeat and a longer prompt resume there" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var f: fake.Fake = .{ .gpa = a };
    var store = pc.Store.init(a, f.snapshots(), .{ .planned = true, .grid = 4, .tail = true, .min_prompt = 0, .min_gap = 2, .lookahead = 1 }, 1024);
    defer store.deinit();
    var prompt: [14]u32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
    const raw = prompt[0..10];
    const first = try store.begin(arena.allocator(), raw, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{8}, first.marks);
    const want = f.pass(&store, raw, first);
    const again = try store.begin(arena.allocator(), raw, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 8), again.from);
    try std.testing.expectEqual(@as(usize, 0), again.marks.len);
    try std.testing.expectEqual(want, f.pass(&store, raw, again));
    const longer = try store.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 8), longer.from);
    try std.testing.expectEqualSlices(u32, &.{12}, longer.marks);
    // a chat names its history: that mark alone, where a tail would also keep the chunk start past it
    try std.testing.expectEqualSlices(u32, &.{8}, try store.marks(arena.allocator(), &prompt, 0, 9, &.{}, &.{}, &.{}));
    // a prompt shorter than one chunk keeps nothing, and without the rule a raw prompt keeps nothing either
    try std.testing.expectEqual(@as(usize, 0), (try store.marks(arena.allocator(), prompt[0..3], 0, 0, &.{}, &.{}, &.{})).len);
    store.rules.tail = false;
    try std.testing.expectEqual(@as(usize, 0), (try store.marks(arena.allocator(), raw, 0, 0, &.{}, &.{}, &.{})).len);
}
