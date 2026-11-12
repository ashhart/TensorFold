//! The status copy of the prompt cache: counts, capacity, time stamps and the newest retained entries.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const status = @import("prompt_cache_status.zig");

const vtable: pc.Snapshots.VTable = .{
    .bytes = bytes,
    .save = save,
    .restore = restore,
    .drop = drop,
};
fn bytes(_: *anyopaque, at: u32) u64 {
    return @as(u64, at) << 10;
}
fn save(_: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
    return @ptrFromInt(at);
}
fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
fn drop(_: *anyopaque, _: pc.Saved) void {}

test "the snapshot copies counts and the most recent touch" {
    const gpa = std.testing.allocator;
    var s = pc.Store{ .gpa = gpa, .family = .{ .ptr = @ptrFromInt(1), .vtable = &vtable }, .rules = .{}, .budget = 1 << 20, .counts = .{ .hits = 7, .misses = 3, .kept = 9, .evicted = 2, .refused = 1, .failed = 4 }, .held = 500_000, .hold = true };
    defer s.entries.deinit(gpa);
    const e1 = try gpa.create(pc.Entry);
    defer gpa.destroy(e1);
    const e2 = try gpa.create(pc.Entry);
    defer gpa.destroy(e2);
    var tok = [_]u32{ 1, 2 };
    e1.* = .{ .tokens = tok[0..], .at = 256, .saved = @ptrFromInt(1), .bytes = 1024, .born = 0, .used = 1, .last = &.{}, .seen = .{ .created_ms = 10, .last_ms = 300, .hits = 2 } };
    e2.* = .{ .tokens = tok[0..], .at = 512, .saved = @ptrFromInt(1), .bytes = 2048, .born = 0, .used = 2, .last = &.{}, .seen = .{ .created_ms = 20, .last_ms = 200, .hits = 5 } };
    try s.entries.append(gpa, e1);
    try s.entries.append(gpa, e2);
    const view = status.PromptCache(&s);
    try std.testing.expectEqual(@as(u64, 7), view.hits);
    try std.testing.expectEqual(@as(u64, 3), view.misses);
    try std.testing.expectEqual(@as(u64, 9), view.kept);
    try std.testing.expectEqual(@as(u64, 2), view.evicted);
    try std.testing.expectEqual(@as(u64, 1), view.refused);
    try std.testing.expectEqual(@as(u64, 4), view.failed);
    try std.testing.expectEqual(@as(u64, 1 << 20), view.budget_bytes);
    try std.testing.expect(view.hold);
    try std.testing.expectEqual(@as(u64, 2), view.entry_count);
    try std.testing.expectEqual(@as(i64, 300), view.last_access_ms);
    try std.testing.expectEqual(@as(u32, 256), view.entries[0].at); // the most recently used first, not the newest kept
    try std.testing.expectEqual(@as(i64, 10), view.entries[0].created_at_ms);
    try std.testing.expectEqual(@as(u64, 2), view.entries[0].hits);
    try std.testing.expectEqual(@as(u32, 512), view.entries[1].at);
}

test "the snapshot keeps the most recently used entries when more are kept than it holds" {
    const gpa = std.testing.allocator;
    var s = pc.Store{ .gpa = gpa, .family = .{ .ptr = @ptrFromInt(1), .vtable = &vtable }, .rules = .{}, .budget = 1 << 20 };
    defer s.entries.deinit(gpa);
    var made: [100]pc.Entry = undefined;
    for (&made, 0..) |*e, i| { // last uses out of order: 0, 37, 74, 11, ...
        e.* = .{ .tokens = &.{}, .at = @intCast(i), .saved = @ptrFromInt(1), .bytes = 1, .born = 0, .used = 0, .last = &.{}, .seen = .{ .last_ms = @intCast(i * 37 % 100) } };
        try s.entries.append(gpa, e);
    }
    const view = status.PromptCache(&s);
    try std.testing.expectEqual(@as(u64, 100), view.entry_count);
    for (view.entries, 0..) |e, i| try std.testing.expectEqual(@as(i64, @intCast(99 - i)), e.last_access_ms);
}

test "the snapshot hides a disk tier the store lacks" {
    const gpa = std.testing.allocator;
    var s = pc.Store{ .gpa = gpa, .family = .{ .ptr = @ptrFromInt(1), .vtable = &vtable }, .rules = .{}, .budget = 1 << 20 };
    try std.testing.expect(status.Disk(&s) == null);
}

test "an entry's id is its state name's stem, as the disk tier names the file" {
    const gpa = std.testing.allocator;
    var s = pc.Store{ .gpa = gpa, .family = .{ .ptr = @ptrFromInt(1), .vtable = &vtable }, .rules = .{}, .budget = 1 << 20 };
    const tokens = [_]u32{ 5, 6, 7, 8 };
    try std.testing.expectEqual(@as(@FieldType(status.Seen, "id"), @splat(0)), status.Seen.of(&s, &tokens, &.{}, &.{}, 3).id); // dashboard off
    status.named.store(true, .monotonic);
    defer status.named.store(false, .monotonic);
    const seen = status.Seen.of(&s, &tokens, &.{}, &.{}, 3);
    const chain = @import("rows_chain.zig");
    try std.testing.expectEqual(chain.stem(chain.state(.{ .identity = 0, .block = 0 }, &tokens, &.{}, &.{}, 3)), seen.id);
    const other = [_]u32{ 5, 6, 7, 9 };
    try std.testing.expect(!std.mem.eql(u8, &seen.id, &status.Seen.of(&s, &other, &.{}, &.{}, 3).id)); // another prompt
    try std.testing.expect(seen.created_ms > 0 and seen.created_ms == seen.last_ms);
}
