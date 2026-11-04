//! The bounded ladder of intermediate checkpoints: planned, admitted beside the endpoint, resumed when compacted.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const im = @import("prompt_intermediate.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;

const ladder: pc.Rules = .{ .chunk = 2048, .intermediate = 8192, .intermediate_most = 4 };

test "ladder takes the largest powers-of-two positions under the history" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { rules: pc.Rules, prompt: usize, history: u32, from: u32, marks: []const u32, want: []const u32 }{
        .{ .rules = ladder, .prompt = 160 * 1024, .history = 160 * 1024 - 8, .from = 0, .marks = &.{}, .want = &.{ 32768, 65536, 131072 } },
        .{ .rules = ladder, .prompt = 40 * 1024, .history = 40 * 1024, .from = 0, .marks = &.{}, .want = &.{32768} },
        // An existing mark and the resume point drop their positions; a short prompt or the default rules plan none.
        .{ .rules = ladder, .prompt = 40 * 1024, .history = 40 * 1024, .from = 0, .marks = &.{32768}, .want = &.{} },
        .{ .rules = ladder, .prompt = 40 * 1024, .history = 40 * 1024, .from = 32768, .marks = &.{}, .want = &.{} },
        .{ .rules = ladder, .prompt = 40 * 1024, .history = 40 * 1024, .from = 0, .marks = &.{40000}, .want = &.{32768} },
        .{ .rules = ladder, .prompt = 8192, .history = 8192, .from = 0, .marks = &.{}, .want = &.{} },
        .{ .rules = .{ .chunk = 2048 }, .prompt = 160 * 1024, .history = 160 * 1024, .from = 0, .marks = &.{}, .want = &.{} },
    };
    for (cases) |c| {
        const got = try im.cuts(gpa, c.rules, c.prompt, c.history, c.from, c.marks);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, c.want, got);
    }
}

test "a compacted long history resumes its longest ladder checkpoint exactly" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 160 * 1024);
    for (prompt, 0..) |*t, i| t.* = @intCast(i % 65521);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), ladder, 1 << 20);
    defer s.deinit();
    const history: u32 = @intCast(prompt.len - 8);
    const p = try s.beginUser(a, prompt, history, 0, 0, &.{4096}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 4096, 32768, 65536, 131072, history }, p.marks);
    try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    const compacted = try a.dupe(u32, prompt[0 .. 120 * 1024]);
    for (compacted[100 * 1024 ..]) |*t| t.* = @intCast((t.* + 1) % 65521);
    const q = try s.beginUser(a, compacted, @intCast(compacted.len - 8), 0, 0, &.{4096}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 65536), q.from);
    try std.testing.expectEqual(fresh(compacted), f.pass(&s, compacted, q));
    try std.testing.expectEqual(@as(u64, 1), s.counts.hits);
}

test "without the ladder a compacted long history resumes only its shared cut" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 160 * 1024);
    for (prompt, 0..) |*t, i| t.* = @intCast(i % 65521);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .chunk = 2048 }, 1 << 20);
    defer s.deinit();
    const history: u32 = @intCast(prompt.len - 8);
    const p = try s.beginUser(a, prompt, history, 0, 0, &.{4096}, &.{}, null, &.{});
    try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    const compacted = try a.dupe(u32, prompt[0 .. 120 * 1024]);
    for (compacted[100 * 1024 ..]) |*t| t.* = @intCast((t.* + 1) % 65521);
    const q = try s.beginUser(a, compacted, @intCast(compacted.len - 8), 0, 0, &.{4096}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4096), q.from);
}

test "ladder checkpoints stay across an append turn and are evictable" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 160 * 1024);
    for (prompt, 0..) |*t, i| t.* = @intCast(i % 65521);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), ladder, 1 << 20);
    defer s.deinit();
    const p = try s.beginUser(a, prompt, @intCast(prompt.len - 8), 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    const rung = for (s.entries.items) |e| {
        if (e.at == 65536) break e;
    } else null;
    try std.testing.expect(rung != null and rung.?.intermediate);
    // The next turn extends the prompt; the checkpoint is not superseded by the later state.
    const grown = try a.alloc(u32, 200 * 1024);
    @memcpy(grown[0..prompt.len], prompt);
    for (grown[prompt.len..], 0..) |*t, i| t.* = @intCast(i % 7);
    const q = try s.beginUser(a, grown, @intCast(grown.len - 8), 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, @intCast(prompt.len - 8)), q.from);
    try std.testing.expectEqual(fresh(grown), f.pass(&s, grown, q));
    const compacted = try a.dupe(u32, grown[0 .. 120 * 1024]);
    for (compacted[100 * 1024 ..]) |*t| t.* = @intCast((t.* + 1) % 7);
    const r = try s.beginUser(a, compacted, @intCast(compacted.len - 8), 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 65536), r.from);
    try std.testing.expectEqual(fresh(compacted), f.pass(&s, compacted, r));
    // Under pressure the same checkpoint is no immortal state: evicting picks it once better victims are gone.
    var picked = false;
    while (s.entries.items.len > 0) {
        const victim = s.victimBut(null, grown, @intCast(grown.len)) orelse break;
        picked = picked or s.entries.items[victim].intermediate;
        s.remove(victim);
    }
    try std.testing.expect(picked);
}

test "ladder admission keeps the endpoint ahead of the checkpoints" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 160 * 1024);
    @memset(prompt, 5);
    const history: u32 = @intCast(prompt.len - 8);
    const endpoint: u64 = 100 + history;
    for ([_]u64{ 0, endpoint - 1, endpoint, endpoint + 100 + 32768, 1 << 20 }) |budget| for ([_]usize{ 1, 2, 3, 6 }) |slots| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), ladder, budget);
        defer s.deinit();
        s.most = slots;
        const p = try s.beginUser(a, prompt, history, 0, 0, &.{}, &.{}, null, &.{});
        const rungs = for (p.marks) |m| {
            if (im.is(ladder, prompt.len, m)) break true;
        } else false;
        try std.testing.expectEqual(budget >= endpoint and slots >= 1, std.mem.indexOfScalar(u32, p.marks, history) != null);
        try std.testing.expectEqual(budget >= endpoint + 100 + 32768 and slots >= 2, rungs);
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
        try std.testing.expect(s.held + s.spare() <= budget);
        try std.testing.expect(s.entries.items.len <= slots);
    };
}

test "a short prompt or a cache-off store plans no ladder" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 8000);
    @memset(prompt, 7);
    for ([_]pc.Rules{ ladder, .{ .chunk = 2048 } }) |rules| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 20);
        defer s.deinit();
        const p = try s.beginUser(a, prompt, 7992, 16, 7992, &.{}, &.{}, null, &.{});
        for (p.marks) |m| try std.testing.expect(!im.is(rules, prompt.len, m));
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    }
}
