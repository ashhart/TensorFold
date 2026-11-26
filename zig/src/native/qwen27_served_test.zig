//! Prompt-cache, startup wording, and reply-cache unit tests extracted from qwen27_served.zig.
const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const pc = api.prompt_cache;
const lb = @import("qwen27_lanes.zig");
const served = @import("qwen27_served.zig");
const reply = @import("qwen27_reply.zig");
const startup = @import("qwen27_startup.zig");
const slot_state = @import("qwen27_slots.zig");

const FakeStates = struct {
    live: usize = 0,
    fn bytes(_: *anyopaque, at: u32) u64 {
        return @as(u64, at) << 16;
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *FakeStates = @ptrCast(@alignCast(ptr));
        f.live += 1;
        return @ptrFromInt(@as(usize, at) << 4);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
    fn drop(ptr: *anyopaque, _: pc.Saved) void {
        const f: *FakeStates = @ptrCast(@alignCast(ptr));
        f.live -= 1;
    }
};

test "--prompt-cache-gib 40: a later 192k-token slot takes its room back from kept states, never the one it restores" {
    const gpa = std.testing.allocator;
    const g: u64 = 1 << 30;
    var fake: FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } }, .{}, 40 * g);
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

test "a reply's end keeps its prompt's born: newest conversations precede redundant history under pressure" {
    var fake: FakeStates = .{};
    const snapshots: pc.Snapshots = .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } };
    try @import("qwen27_reply_cache_test.zig").replyBorn(reply.keepEnd, snapshots, &fake.live);
}

test "qwen27 startup: draft stream count, singular text and optional suffixes" {
    const limit = try lb.draftSlots();
    const limited = platform.c.getenv("TF_Q27_DRAFT_SLOTS") != null;
    try std.testing.expectEqual(@as(u32, if (limited) 1 else 8), limit);
    // the wording: the smaller of the stream count and the draft-slot setting
    const wording = [_]struct { streams: u32, slots: u32, want: []const u8 }{
        .{ .streams = 1, .slots = 0, .want = "" },
        .{ .streams = 1, .slots = 1, .want = "DFlash2 drafts for the only stream" },
        .{ .streams = 1, .slots = 8, .want = "DFlash2 drafts for the only stream" },
        .{ .streams = 1, .slots = 99, .want = "DFlash2 drafts for the only stream" },
        .{ .streams = 2, .slots = 0, .want = "" },
        .{ .streams = 2, .slots = 1, .want = "DFlash2 drafts for 1 of 2 streams" },
        .{ .streams = 2, .slots = 8, .want = "DFlash2 drafts for both streams" },
        .{ .streams = 2, .slots = 99, .want = "DFlash2 drafts for both streams" },
        .{ .streams = 4, .slots = 1, .want = "DFlash2 drafts for 1 of 4 streams" },
        .{ .streams = 4, .slots = 8, .want = "DFlash2 drafts for all 4 streams" },
        .{ .streams = 4, .slots = 99, .want = "DFlash2 drafts for all 4 streams" },
        .{ .streams = 16, .slots = 1, .want = "DFlash2 drafts for 1 of 16 streams" },
        .{ .streams = 16, .slots = 8, .want = "DFlash2 drafts for 8 of 16 streams" },
        .{ .streams = 16, .slots = 99, .want = "DFlash2 drafts for all 16 streams" },
    };
    var buf: [64]u8 = undefined;
    for (wording) |case| try std.testing.expectEqualStrings(case.want, try startup.draftWording(&buf, case.streams, case.slots));
    // the whole line: the wording, a drafter's absence and the optional suffixes
    const lines = [_]struct { streams: u32, slots: u32, images: bool, want: []const u8 }{
        .{ .streams = 1, .slots = 8, .images = false, .want = "Qwen3.8-27B: 1 stream of 4096 tokens, DFlash2 drafts for the only stream" },
        .{ .streams = 2, .slots = 8, .images = true, .want = "Qwen3.8-27B: 2 streams of 4096 tokens, DFlash2 drafts for both streams, images read" },
        .{ .streams = 4, .slots = 8, .images = false, .want = "Qwen3.8-27B: 4 streams of 4096 tokens, DFlash2 drafts for all 4 streams" },
        .{ .streams = 16, .slots = 8, .images = false, .want = "Qwen3.8-27B: 16 streams of 4096 tokens, DFlash2 drafts for 8 of 16 streams" },
        .{ .streams = 2, .slots = 0, .images = false, .want = "Qwen3.8-27B: 2 streams of 4096 tokens" },
    };
    var actual: [192]u8 = undefined;
    for (lines) |case| {
        const got = try std.fmt.bufPrint(&actual, "{f}", .{startup.StartupLine{ .streams = case.streams, .context = 4096, .draft_slots = case.slots, .images = case.images }});
        try std.testing.expectEqualStrings(case.want, got);
    }
}

test {
    _ = @import("qwen27_share_fit.zig");
}

test "Qwen27 interior state uses a guaranteed cold request cut" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const tokens = try a.alloc(u32, 128 * 1024 + 64);
    @memset(tokens, 7);
    var fake: FakeStates = .{};
    var store = pc.Store.init(gpa, .{ .ptr = &fake, .vtable = &.{ .bytes = FakeStates.bytes, .save = FakeStates.save, .restore = FakeStates.restore, .drop = FakeStates.drop } }, served.cacheRules(2048, 0), 32 << 30);
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

test "qwen27 reply cache: warm work preserves both client histories within rounded budgets" {
    try @import("qwen27_reply_cache_test.zig").exercise(reply.keepEnd, slot_state.opening);
}

test "qwen27 reply cache: three reasoning-free turns resume with a full 16 GiB cache" {
    try @import("qwen27_reply_cache_test.zig").fullCache(reply.keepEnd, slot_state.opening);
}
