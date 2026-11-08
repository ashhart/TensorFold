//! The prompt-cache interior test that lives beside `qwen27_served` keeps the file under the line cap.
const std = @import("std");
const served = @import("qwen27_served.zig");
const pc = @import("engine_api").prompt_cache;

var fake_state: served.FakeStates = .{};

test "Qwen27 interior state uses a guaranteed cold request cut" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try a.alloc(u32, 128 * 1024 + 64);
    @memset(tokens, 7);
    var store = pc.Store.init(gpa, .{ .ptr = &fake_state, .vtable = &.{ .bytes = served.FakeStates.bytes, .save = served.FakeStates.save, .restore = served.FakeStates.restore, .drop = served.FakeStates.drop } }, served.cacheRules(2048), 32 << 30);
    defer store.deinit();
    var starts: std.ArrayList(u32) = .empty;
    var at: u32 = 2048;
    while (at < tokens.len) : (at += 2048) try starts.append(a, at);
    const p = try store.beginUser(a, tokens, @intCast(tokens.len - 8), 16, @intCast(tokens.len - 8), &.{}, starts.items, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 129024, @intCast(tokens.len - 8) }, p.marks);
    for (p.marks) |m| try std.testing.expect(store.keep(tokens, m, null, starts.items, &.{}));
    tokens[tokens.len - 512] += 1;
    const followup = try store.beginUser(a, tokens, @intCast(tokens.len - 8), 16, @intCast(tokens.len - 8), &.{}, starts.items, null, &.{});
    try std.testing.expectEqual(@as(u32, 129024), followup.from);
    try std.testing.expect(store.held <= store.budget);
}
