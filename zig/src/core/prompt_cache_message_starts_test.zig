//! Message-start states: kept without displacing other states, so an edit of message N resumes from N's start.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const ms = @import("prompt_message_starts.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;

const metal: pc.Rules = .{ .warm = .on, .chunk = 2048, .chunk_marks = false, .message_starts = 4 };
const starts_32k = [_]u32{ 1000, 5000, 9000, 13000, 17000, 21000, 25000, 29000 };
const len_32k = 32768;
const history_32k = 32700;

fn tokens(a: std.mem.Allocator, n: usize, salt: u32) ![]u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*t, i| t.* = @intCast((i * 7 + salt) % 65521);
    return out;
}

fn has(marks: []const u32, at: u32) bool {
    return std.mem.indexOfScalar(u32, marks, at) != null;
}

fn planned(marks: []const u32) usize {
    var n: usize = 0;
    for (starts_32k) |at| n += @intFromBool(has(marks, at));
    return n;
}

test "cuts picks the latest starts under the history, apart from the resume point, the marks and each other" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { rules: pc.Rules, len: usize, history: u32, from: u32, messages: []const u32, marks: []const u32, want: []const u32 }{
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 0, .messages = &.{ 1000, 3000, 5000, 7000, 9000, 11000 }, .marks = &.{}, .want = &.{ 3000, 5000, 7000, 9000 } },
        // A start within min_gap of the history, or at or past it, is no cut.
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 0, .messages = &.{ 2000, 4000, 6000, 8000, 9900, 10000 }, .marks = &.{}, .want = &.{ 2000, 4000, 6000, 8000 } },
        // Starts at or within min_gap after the resume point are dropped, and none before it is reached.
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 2800, .messages = &.{ 1000, 2900, 5000 }, .marks = &.{}, .want = &.{5000} },
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 3000, .messages = &.{ 3000, 5000 }, .marks = &.{}, .want = &.{5000} },
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 5000, .messages = &.{ 1000, 3000, 5000 }, .marks = &.{}, .want = &.{} },
        // A start near a mark or a later start gives way to it.
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 0, .messages = &.{ 5000, 7000, 9000 }, .marks = &.{7100}, .want = &.{ 5000, 9000 } },
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 0, .messages = &.{ 5000, 5100, 9000 }, .marks = &.{}, .want = &.{ 5100, 9000 } },
        // The count limits the latest starts; zero or no starts plan none.
        .{ .rules = .{ .message_starts = 2 }, .len = 20000, .history = 10000, .from = 0, .messages = &.{ 1000, 3000, 5000, 7000 }, .marks = &.{}, .want = &.{ 5000, 7000 } },
        .{ .rules = .{ .message_starts = 0 }, .len = 20000, .history = 10000, .from = 0, .messages = &.{ 1000, 3000, 5000, 7000 }, .marks = &.{}, .want = &.{} },
        .{ .rules = metal, .len = 20000, .history = 10000, .from = 0, .messages = &.{}, .marks = &.{}, .want = &.{} },
    };
    for (cases) |c| {
        const got = try ms.cuts(gpa, c.rules, c.len, c.history, c.from, c.messages, c.marks);
        defer gpa.free(got);
        try std.testing.expectEqualSlices(u32, c.want, got);
    }
}

fn edited(a: std.mem.Allocator, prompt: []const u32, at: usize) ![]u32 {
    const out = try a.alloc(u32, history_32k + 100);
    @memcpy(out[0..history_32k], prompt[0..history_32k]);
    out[at] +%= 1;
    for (out[history_32k..], 0..) |*t, i| t.* = @intCast(60000 + i);
    return out;
}

test "an edit inside the last tool message resumes from that message's start and equals a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, len_32k, 3);
    const change = try edited(a, prompt, 27000);
    var from: [2]u32 = undefined;
    for ([_]u8{ 4, 0 }, &from) |count, *got| {
        var rules = metal;
        rules.message_starts = count;
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 22);
        defer s.deinit();
        s.message_starts = if (count > 0) &starts_32k else &.{};
        const p = try s.beginUser(a, prompt, history_32k, 0, 0, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(usize, count), planned(p.marks));
        try std.testing.expect(has(p.marks, history_32k));
        if (count > 0) for ([_]u32{ 17000, 21000, 25000, 29000 }) |at| try std.testing.expect(has(p.marks, at));
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
        const q = try s.beginUser(a, change, @intCast(change.len), 0, 0, &.{}, &.{}, null, &.{});
        got.* = q.from;
        try std.testing.expectEqual(fresh(change), f.pass(&s, change, q));
    }
    try std.testing.expectEqual(@as(u32, 25000), from[0]);
    try std.testing.expect(from[1] < 25000);
}

test "message starts fit only the budget the endpoint and fewer states leave" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, len_32k, 5);
    const endpoint: u64 = 100 + history_32k;
    const cases = [_]struct { budget: u64, most: usize, want: usize }{
        .{ .budget = endpoint, .most = 99, .want = 0 },
        .{ .budget = endpoint + 99, .most = 99, .want = 0 },
        .{ .budget = endpoint + 29099, .most = 99, .want = 1 },
        .{ .budget = endpoint + 29100, .most = 99, .want = 1 },
        .{ .budget = endpoint + 29100 + 25100, .most = 99, .want = 2 },
        .{ .budget = 1 << 22, .most = 99, .want = 4 },
        .{ .budget = 1 << 22, .most = 3, .want = 2 },
    };
    for (cases) |c| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), metal, c.budget);
        defer s.deinit();
        s.most = c.most;
        s.message_starts = &starts_32k;
        const p = try s.beginUser(a, prompt, history_32k, 0, 0, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(c.want, planned(p.marks));
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
        try std.testing.expect(s.held + s.spare() <= c.budget);
    }
}

test "message starts never displace another conversation's state" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const other = try tokens(a, len_32k, 11);
    const own = try tokens(a, len_32k, 13);
    const endpoint: u64 = 100 + history_32k;
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), metal, 2 * endpoint + 100);
    defer s.deinit();
    const o = try s.beginUser(a, other, history_32k, 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(fresh(other), f.pass(&s, other, o));
    s.message_starts = &starts_32k;
    const p = try s.beginUser(a, own, history_32k, 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(usize, 0), planned(p.marks));
    try std.testing.expectEqual(fresh(own), f.pass(&s, own, p));
    try std.testing.expect(s.held <= s.budget);
    s.message_starts = &.{};
    const again = try s.beginUser(a, other, history_32k, 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, history_32k), again.from);
    try std.testing.expectEqual(fresh(other), f.pass(&s, other, again));
}

test "message starts take the room a superseded state of the same conversation leaves" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, len_32k, 17);
    // Turns end at 8000 and 20000; the third resumes the second, so the first state is superseded and its room free.
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), metal, (100 + history_32k) + (100 + 20000) + 29100);
    defer s.deinit();
    for ([_]u32{ 8000, 20000 }) |history| {
        const p = try s.beginUser(a, prompt[0 .. history + 100], history, 0, 0, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(fresh(prompt[0 .. history + 100]), f.pass(&s, prompt[0 .. history + 100], p));
    }
    s.message_starts = &starts_32k;
    const q = try s.beginUser(a, prompt, history_32k, 0, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 20000), q.from);
    try std.testing.expectEqualSlices(u32, &.{ 29000, history_32k }, q.marks);
    try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, q));
    try std.testing.expect(s.held + s.spare() <= s.budget);
}

test "raw prompts without message starts plan nothing extra" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, len_32k, 19);
    var plain: ?[]const u32 = null;
    for ([_][]const u32{ &.{}, &starts_32k }) |starts| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), metal, 1 << 22);
        defer s.deinit();
        s.message_starts = starts;
        const p = try s.beginUser(a, prompt, history_32k, 0, 0, &.{}, &.{}, null, &.{});
        if (starts.len == 0) {
            try std.testing.expectEqual(@as(usize, 0), planned(p.marks));
            plain = try a.dupe(u32, p.marks);
        } else {
            try std.testing.expectEqual(@as(usize, 4), planned(p.marks));
            try std.testing.expectEqual(plain.?.len + 4, p.marks.len);
        }
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    }
}

test "rewind moves to the message start only when one lies within min_gap at or before it" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .gpa = gpa };
    const cases = [_]struct { rules: pc.Rules, rewind: u32, want: u32 }{
        .{ .rules = .{}, .rewind = 9004, .want = 9004 },
        .{ .rules = metal, .rewind = 9004, .want = 9000 },
        .{ .rules = metal, .rewind = 9000, .want = 9000 },
        .{ .rules = metal, .rewind = 9255, .want = 9000 },
        .{ .rules = metal, .rewind = 9256, .want = 9256 },
        .{ .rules = metal, .rewind = 900, .want = 900 },
        .{ .rules = metal, .rewind = 0, .want = 0 },
    };
    for (cases) |c| {
        var s = pc.Store.init(gpa, f.snapshots(), c.rules, 1 << 20);
        defer s.deinit();
        s.message_starts = &.{ 1000, 5000, 9000, 20000 };
        try std.testing.expectEqual(c.want, ms.rewind(&s, c.rewind));
    }
}

test "a short last user message resumes at its start when a later request replaces what follows it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const base = try tokens(a, 32000, 23);
    const question: u32 = 31900;
    const tool3: u32 = 28000;
    const first_starts = [_]u32{ 5000, 12000, 20000, tool3, question };
    const second = try a.alloc(u32, 36000);
    @memcpy(second[0..question], base[0..question]);
    for (second[question..], 0..) |*t, i| t.* = @intCast(40000 + i);
    const second_starts = [_]u32{ 5000, 12000, 20000, tool3, question, 32200, 35900 };
    var from: [2]u32 = undefined;
    for ([_]u8{ 4, 0 }, &from) |count, *got| {
        var rules = metal;
        rules.message_starts = count;
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 22);
        defer s.deinit();
        s.message_starts = &first_starts;
        const p = try s.beginUser(a, base, 31992, question + 4, 0, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(fresh(base), f.pass(&s, base, p));
        s.message_starts = &second_starts;
        const q = try s.beginUser(a, second, 35992, 35904, 0, &.{}, &.{}, null, &.{});
        got.* = q.from;
        try std.testing.expectEqual(fresh(second), f.pass(&s, second, q));
    }
    try std.testing.expectEqual(@as(u32, question), from[0]);
    try std.testing.expect(from[1] < tool3 + 1);
}

fn noneNeeded(_: *anyopaque, _: *const pc.Entry) bool {
    return false;
}

/// The client asked before: its rewind prefix is among the recent lookups'.
fn askedBefore(s: *pc.Store, prompt: []const u32, at: u32) void {
    s.recent_rewinds[0] = pc.Store.sharedKey(prompt[0 .. at + s.rules.lookahead]);
}

test "a full guarded store makes room for the last user start beside the endpoint, so a new question resumes there" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // A voice assistant's turn: long fixed tools and system prompt, then a short user message (start 4907).
    const prompt = try tokens(a, 4944, 3);
    const starts = [_]u32{ 0, 4907 };
    const history: u32 = 4939;
    const rewind_len: u32 = 4910;
    for ([_]u8{ 4, 0 }) |count| {
        var rules = metal;
        rules.message_starts = count;
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, 3 * (100 + 4939) + 50);
        defer s.deinit();
        // Other conversations' one-off states fill the store.
        for (0..3) |i| {
            const other = try tokens(a, 4939, @intCast(1000 + i));
            f.at = 4939;
            f.sum = fresh(other);
            try std.testing.expect(s.keep(other, 4939, null, &.{}, &.{}));
        }
        try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
        s.message_starts = if (count > 0) &starts else &.{};
        askedBefore(&s, prompt, 4907);
        s.reservation_guard = .{ .ctx = &f, .needed = noneNeeded };
        defer s.reservation_guard = null;
        const l = try s.lookupUser(a, prompt, @intCast(prompt.len), history, rewind_len, history, &.{}, &.{}, &.{});
        // With message starts the user message's start is kept in a full store; without them the plan is main's.
        try std.testing.expect(has(l.marks, history));
        try std.testing.expectEqual(count > 0, has(l.marks, 4907));
        try std.testing.expectEqual(@as(u64, if (count > 0) 2 else 1), s.counts.evicted);
    }
}

fn firstTwoNeeded(_: *anyopaque, e: *const pc.Entry) bool {
    return e.tokens[0] == 1000 or e.tokens[0] == 1001;
}

test "the rewind state never evicts a state a queued job needs; it is trimmed instead" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, 4944, 3);
    const starts = [_]u32{ 0, 4907 };
    const history: u32 = 4939;
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), metal, 3 * (100 + 4939) + 50);
    defer s.deinit();
    for (0..3) |i| {
        const other = try tokens(a, 4939, @intCast(1000 + i));
        f.at = 4939;
        f.sum = fresh(other);
        try std.testing.expect(s.keep(other, 4939, null, &.{}, &.{}));
    }
    s.message_starts = &starts;
    askedBefore(&s, prompt, 4907);
    s.reservation_guard = .{ .ctx = &f, .needed = firstTwoNeeded };
    defer s.reservation_guard = null;
    const l = try s.lookupUser(a, prompt, @intCast(prompt.len), history, 4910, history, &.{}, &.{}, &.{});
    // Only the third state may go: it makes room for the endpoint, and the rewind state is trimmed.
    try std.testing.expect(has(l.marks, history));
    try std.testing.expect(!has(l.marks, 4907));
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    for (s.entries.items) |e| try std.testing.expect(e.tokens[0] == 1000 or e.tokens[0] == 1001);
}

test "the rewind state never evicts another conversation's rewind state, so clients asking in turn keep theirs" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, 4944, 3);
    const starts = [_]u32{ 0, 4907 };
    const history: u32 = 4939;
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), metal, 3 * (100 + 4939) + 50);
    defer s.deinit();
    for (0..3) |i| {
        const other = try tokens(a, 4939, @intCast(1000 + i));
        f.at = 4939;
        f.sum = fresh(other);
        try std.testing.expect(s.keep(other, 4939, null, &.{}, &.{}));
    }
    // The two newer states are other clients' rewind states.
    for (s.entries.items) |e| if (e.tokens[0] != 1000) {
        e.rewind = true;
    };
    s.message_starts = &starts;
    askedBefore(&s, prompt, 4907);
    s.reservation_guard = .{ .ctx = &f, .needed = noneNeeded };
    defer s.reservation_guard = null;
    const l = try s.lookupUser(a, prompt, @intCast(prompt.len), history, 4910, history, &.{}, &.{}, &.{});
    try std.testing.expect(has(l.marks, history));
    try std.testing.expect(!has(l.marks, 4907));
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    for (s.entries.items) |e| try std.testing.expect(e.rewind);
}

test "a first-time prompt's rewind state takes only free room, so one-off conversations keep main's plan" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try tokens(a, 4944, 3);
    const starts = [_]u32{ 0, 4907 };
    const history: u32 = 4939;
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), metal, 3 * (100 + 4939) + 50);
    defer s.deinit();
    for (0..3) |i| {
        const other = try tokens(a, 4939, @intCast(1000 + i));
        f.at = 4939;
        f.sum = fresh(other);
        try std.testing.expect(s.keep(other, 4939, null, &.{}, &.{}));
    }
    s.message_starts = &starts;
    s.reservation_guard = .{ .ctx = &f, .needed = noneNeeded };
    defer s.reservation_guard = null;
    const first = try s.lookupUser(a, prompt, @intCast(prompt.len), history, 4910, history, &.{}, &.{}, &.{});
    try std.testing.expect(has(first.marks, history));
    try std.testing.expect(!has(first.marks, 4907));
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    // Asked again: now a repeating client, so its rewind state gets room beside the endpoint.
    const again = try s.lookupUser(a, prompt, @intCast(prompt.len), history, 4910, history, &.{}, &.{}, &.{});
    try std.testing.expect(has(again.marks, 4907));
}

test "a client that keeps asking stays a repeat while fewer than 16 other prompts come between" {
    const gpa = std.testing.allocator;
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), metal, 1 << 30);
    defer s.deinit();
    for (0..40) |round| {
        s.active_rewind_key = 7;
        s.noteRewind();
        try std.testing.expectEqual(round > 0, s.active_rewind_repeat);
        for (0..15) |i| { // many rounds of one-off prompts: the asking client moves up each time
            s.active_rewind_key = 1000 + round * 100 + i;
            s.noteRewind();
            try std.testing.expect(!s.active_rewind_repeat);
        }
    }
    for (0..16) |i| {
        s.active_rewind_key = 9000 + i;
        s.noteRewind();
    }
    s.active_rewind_key = 7;
    s.noteRewind();
    try std.testing.expect(!s.active_rewind_repeat); // 16 others pushed it out
    s.active_rewind_key = 0;
    s.noteRewind();
    try std.testing.expect(!s.active_rewind_repeat);
}
