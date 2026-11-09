//! The prompt-cache interior test that lives beside `qwen27_served` keeps the file under the line cap.
const std = @import("std");
const served = @import("qwen27_served.zig");
const pc = @import("engine_api").prompt_cache;
const tf = @import("tensorfold");

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

test "--prompt-cache-gib 40: a later 192k-token slot takes its room back from kept states, never the one it restores" {
    const gpa = std.testing.allocator;
    const g: u64 = 1 << 30;
    var fake: served.FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = served.FakeStates.bytes, .save = served.FakeStates.save, .restore = served.FakeStates.restore, .drop = served.FakeStates.drop } }, .{}, 40 * g);
    defer store.deinit();
    const tokens = try gpa.alloc(u32, 200_000);
    defer gpa.free(tokens);
    for ([_]u32{ 100_000, 150_000, 190_000 }, 0..) |at, k| { // three conversations' histories: 6.1, 9.2 and 11.6 GiB
        for (tokens, 0..) |*t, i| t.* = @intCast(k * 1_000_000 + i);
        try std.testing.expect(store.keep(tokens[0 .. at + 1], at, null, &.{}, &.{}));
    }
    const restoring = store.entries.items[2].saved; // the newest conversation's next turn resumes its state
    const slot: u64 = 12_900_000_000; // a --context 196608 slot's K/V rows, taps ring and scratch
    try std.testing.expect(!served.takeRoom(null, 4 * g, slot, null, tokens)); // without a cache to give back: LaneMemory
    try std.testing.expect(served.takeRoom(&store, 4 * g, slot, restoring, tokens)); // 4 GiB free + 6.1 + 9.2 freed
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expect(store.entries.items[0].saved == restoring);
    try std.testing.expect(!served.takeRoom(&store, 0, 2 * slot, restoring, tokens)); // its own state never goes
    try std.testing.expectEqual(@as(usize, 1), fake.live);
}

test "a reply's end is born with its request's prompt: the history mark that request kept outlives the reply's keep" {
    const gpa = std.testing.allocator;
    const at: u32 = 5000; // a history mark at 5000 tokens, then a 7-token generation prompt and a 100-token reply
    var fake: served.FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = served.FakeStates.bytes, .save = served.FakeStates.save, .restore = served.FakeStates.restore, .drop = served.FakeStates.drop } }, .{ .warm = .on }, (3 * at + 100) << 16);
    defer store.deinit();
    var other: [at + 7]u32 = undefined; // another conversation's prompt, its history kept first
    for (&other, 0..) |*t, i| t.* = @intCast(i);
    try std.testing.expect(store.keep(&other, at, null, &.{}, &.{}));
    var tokens: [at + 107]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(1_000_000 + i);
    const prompt = tokens[0 .. at + 7];
    try std.testing.expect(store.keep(prompt, at, null, &.{}, &.{}));
    var s = try tf.lanes.Stream.init(gpa, .{ .id = "reply", .prompt = prompt, .max_new = 100 });
    defer s.deinit(gpa);
    try s.context.appendSlice(gpa, tokens[at + 7 ..]);
    served.keepEnd(&store, gpa, &s, tokens.len); // three states pass the budget: the least recently used goes
    try std.testing.expect(store.find(&other, &.{}, &.{}) == null);
    const history = store.find(prompt, &.{}, &.{}) orelse return error.HistoryEvicted; // a regenerate resumes it
    try std.testing.expectEqual(at, history.at);
    try std.testing.expectEqual(@as(u32, at + 7), store.entries.items[1].born);
    try std.testing.expectEqual(@as(usize, 2), fake.live);
}

test {
    _ = @import("qwen27_share_fit.zig");
    _ = @import("qwen27_pair_tests.zig");
}

test "qwen27 startup: draft stream count, singular text and optional suffixes" {
    const limit = try lb.draftSlots();
    const limited = platform.c.getenv("TF_Q27_DRAFT_SLOTS") != null;
    try std.testing.expectEqual(@as(u32, if (limited) 1 else 8), limit);
    const cases = [_]struct { u32, []const u8, []const u8 }{
        .{ 1, "1 stream", "1 stream" },
        .{ 2, "2 streams", "2 streams" },
        .{ 9, "9 streams", "8 streams" },
        .{ 16, "16 streams", "8 streams" },
    };
    var mismatches: usize = 0;
    for (cases) |case| {
        for ([_]bool{ false, true }) |drafter| {
            for ([_]bool{ false, true }) |with_images| {
                var draft: [64]u8 = undefined;
                const suffix = if (drafter) try std.fmt.bufPrint(&draft, ", DFlash2 drafts for {s}", .{if (limited) "1 stream" else case[2]}) else "";
                var expected: [192]u8 = undefined;
                const want = try std.fmt.bufPrint(&expected, "Qwen3.8-27B: {s} of 4096 tokens{s}{s}", .{ case[1], suffix, if (with_images) ", images read" else "" });
                var actual: [192]u8 = undefined;
                const got = try std.fmt.bufPrint(&actual, "{f}", .{StartupLine{ .streams = case[0], .context = 4096, .draft_slots = if (drafter) limit else 0, .images = with_images }});
                std.testing.expectEqualStrings(want, got) catch {
                    mismatches += 1;
                };
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}
