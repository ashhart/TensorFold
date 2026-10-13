//! The prompt cache's tests: a fake family whose state is a running sum, so a resumed pass equals a fresh one exactly when it should.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Snapshots = pc.Snapshots;
const Saved = pc.Saved;
const Store = pc.Store;
const Plan = pc.Plan;
const Allocator = std.mem.Allocator;

const Peer = @import("prompt_cache_fake.zig").Peer;
pub const Fake = @import("prompt_cache_fake.zig").Fake;

pub fn fresh(prompt: []const u32) u64 {
    var sum: u64 = 0;
    for (prompt) |t| sum = sum *% 31 +% t;
    return sum;
}

test "a growing conversation resumes each turn where the last one's history ended, and equals a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 }; // history 5, then a two-token generation prompt
    var p = try s.begin(a, &t1, 5, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqualSlices(u32, &.{5}, p.marks);
    try std.testing.expectEqual(fresh(&t1), f.pass(&s, &t1, p));
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 9, 7, 7, 6, 6, 9, 9 }; // the reply and a tool result, history 10
    p = try s.begin(a, &t2, 10, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 5), p.from);
    try std.testing.expectEqualSlices(u32, &.{10}, p.marks);
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
    const edited = [_]u32{ 1, 2, 3, 8, 5, 9, 7, 7, 6, 6, 9, 9 }; // an earlier turn edited: nothing resumes past it
    p = try s.begin(a, &edited, 10, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(fresh(&edited), f.pass(&s, &edited, p));
    try std.testing.expectEqual(@as(u64, 1), s.counts.hits);
    try std.testing.expectEqual(@as(u64, 2), s.counts.misses);
}

test "an entry keys its lookahead tokens: a prompt that differs right after the state does not resume it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 9 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{})); // keeps [1 2 3 4] + 5
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 6, 9 }, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 4), s.find(&.{ 1, 2, 3, 4, 5, 6 }, &.{}, &.{}).?.at);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4 }, &.{}, &.{}) == null); // the lookahead token must be in the prompt
}

test "planned families resume and keep only at the request's chunk starts" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const p = try s.begin(a, &t1, 7, &.{}, &.{ 4, 6 }, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{6}, p.marks); // history 7 floored to the start at 6
    _ = f.pass(&s, &t1, p);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5, 6, 7, 9 }, &.{4}, &.{}) == null); // 6 is not one of this prompt's starts
    try std.testing.expectEqual(@as(u32, 6), s.find(&.{ 1, 2, 3, 4, 5, 6, 7, 9 }, &.{ 4, 6 }, &.{}).?.at);
}

test "a planned family's own grid gives the starts when a request names none, and a resumed pass equals a fresh one" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const p1 = try s.begin(a, &t1, 7, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{4}, p1.marks); // history 7 floored to the grid's 4
    _ = f.pass(&s, &t1, p1);
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14 };
    try std.testing.expect(s.find(&t2, &.{6}, &.{}) == null); // a request's own starts win over the grid
    const p2 = try s.begin(a, &t2, 9, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), p2.from);
    try std.testing.expectEqualSlices(u32, &.{8}, p2.marks);
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p2));
}

test "a planned family keeps marks on a short prompt; an unplanned one waits for min_prompt" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var planned = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4 }, 1 << 20);
    defer planned.deinit();
    const p1 = try planned.begin(a, &t1, 7, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{4}, p1.marks); // 10 tokens, under the default min_prompt
    var serial = Store.init(gpa, f.snapshots(), .{ .lookahead = 1 }, 1 << 20);
    defer serial.deinit();
    const p2 = try serial.begin(a, &t1, 7, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(usize, 0), p2.marks.len);
}

test "eviction frees a conversation's superseded state first, then the oldest; a state past the budget is refused" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 330); // fake bytes: 100 + at
    defer s.deinit();
    const other = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &other, try s.begin(a, &other, 3, &.{}, &.{}, null, &.{})); // 103 bytes
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{})); // 104: 207 held
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 6, &.{}, &.{}, null, &.{})); // 106 more: 313
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    const t3 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    _ = f.pass(&s, &t3, try s.begin(a, &t3, 8, &.{}, &.{}, null, &.{})); // 108: t1's state (extended by later turns) goes first
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.find(&.{ 5, 5, 5, 5 }, &.{}, &.{}) != null);
    try std.testing.expectEqual(@as(u32, 8), s.find(&t3, &.{}, &.{}).?.at);
    const big: [300]u32 = @splat(7);
    _ = f.pass(&s, &big, try s.begin(a, &big, 299, &.{}, &.{}, null, &.{})); // 399 > 330: refused, nothing evicted
    try std.testing.expectEqual(@as(u64, 1), s.counts.refused);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.held <= s.budget);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a failed copy keeps nothing and a failed restore prefills from the start" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa, .fail_save = true };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    f.fail_save = false;
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    f.fail_restore = true;
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    const p = try s.begin(a, &t2, 6, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len); // the entry that failed is gone
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, p));
}

test "marks: the stable prefix with the last prompt and shared blocks, past the resume point, before the end" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_gap = 2, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const prev = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 0, 0 };
    try std.testing.expectEqualSlices(u32, &.{ 3, 7, 10 }, try s.marks(a, &prompt, 0, 10, &.{ 3, 12, 0 }, &.{}, &prev));
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 7, 10, &.{3}, &.{}, &prev));
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, &prompt, 0, 0, &.{}, &.{}, &.{})); // a raw prompt keeps nothing
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 0, 10, &.{9}, &.{}, &.{})); // a block next to the history
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 9, 10, &.{}, &.{}, &.{})); // resumed next to the history: kept
    s.rules.warm = true; // replies prefilled in the background: the resumed state serves the next turn
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, &prompt, 9, 10, &.{}, &.{}, &.{}));
    try std.testing.expectEqualSlices(u32, &.{1}, try s.marks(a, &prompt, 0, 1, &.{}, &.{}, &.{})); // from the start the history's mark stays
    // a warm pass (history the whole prompt) keeps where its state can be keyed, however near it resumed
    try std.testing.expectEqualSlices(u32, &.{11}, try s.marks(a, &prompt, 10, 12, &.{}, &.{}, &.{}));
}

test "prompts under min_prompt keep nothing (their extra prompt call would cost more than reuse saves), longer ones keep as before" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 8 }, 1 << 20);
    defer s.deinit();
    const short = [_]u32{ 1, 2, 3, 4, 5, 9, 9 }; // 7 tokens, history 5
    const p = try s.begin(a, &short, 5, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{}, p.marks);
    try std.testing.expectEqual(fresh(&short), f.pass(&s, &short, p));
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    const long = [_]u32{ 1, 2, 3, 4, 5, 9, 7, 7, 9, 9 }; // 10 tokens: the next turn keeps its history, resuming nothing
    const q = try s.begin(a, &long, 8, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), q.from);
    try std.testing.expectEqualSlices(u32, &.{8}, q.marks);
    try std.testing.expectEqual(fresh(&long), f.pass(&s, &long, q));
    try std.testing.expectEqual(@as(u32, 8), s.find(&.{ 1, 2, 3, 4, 5, 9, 7, 7, 9, 9, 4 }, &.{}, &.{}).?.at);
}

test "kept states charge their real storage, a save takes spare storage first, and spare stays inside the budget" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.pooled(), .{ .min_prompt = 0 }, 330); // new storage 100 + at, a kept state 90 + at
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u64, 94), s.held);
    f.spare_bytes = 120; // a buffer readied while idle
    try std.testing.expectEqual(@as(u64, 116), s.room());
    const other = [_]u32{ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
    _ = f.pass(&s, &other, try s.begin(a, &other, 10, &.{}, &.{}, null, &.{})); // takes the readied buffer: nothing evicted
    try std.testing.expectEqual(@as(u64, 194), s.held);
    try std.testing.expectEqual(@as(u64, 20), s.spare());
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    var t2: [102]u32 = undefined;
    for (&t2, 1..) |*t, i| t.* = @intCast(i);
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 101, &.{}, &.{}, null, &.{})); // resumes t1's; 201 new: the spare shrinks, then `other` goes
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 94 + 191), s.held);
    try std.testing.expect(s.held + s.spare() <= s.budget);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a shared system cut outlives its own conversation's turns: a second conversation resumes it after cold prompts fill the budget" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 600); // bytes 100 + at
    defer s.deinit();
    var c1: [30]u32 = @splat(9);
    var c2: [30]u32 = @splat(8);
    for ([_][]u32{ &c1, &c2 }) |c| { // cold prompts first
        _ = f.pass(&s, c, try s.begin(a, c, 29, &.{}, &.{}, null, &.{}));
        try std.testing.expect(s.held <= s.budget);
    }
    var conv: [45]u32 = undefined;
    for (&conv, 1..) |*t, i| t.* = @intCast(i); // a 20-token system block, then the turns
    for ([_]u32{ 25, 35, 45 }, [_]u32{ 24, 34, 44 }) |len, history| {
        _ = f.pass(&s, conv[0..len], try s.begin(a, conv[0..len], history, &.{20}, &.{}, null, &.{}));
        try std.testing.expect(s.held <= s.budget);
    }
    var c3: [60]u32 = @splat(7);
    _ = f.pass(&s, &c3, try s.begin(a, &c3, 59, &.{}, &.{}, null, &.{}));
    try std.testing.expect(s.held <= s.budget);
    var other: [25]u32 = undefined;
    @memcpy(other[0..20], conv[0..20]);
    for (other[20..], 0..) |*t, i| t.* = @intCast(90 + i);
    const plan = try s.begin(a, &other, 24, &.{20}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 20), plan.from); // the cut survived the first conversation's turns
    try std.testing.expectEqual(fresh(&other), f.pass(&s, &other, plan));
    try std.testing.expect(s.held <= s.budget);
}

test "a pass makes room for every state it keeps before it starts, so a peer told of the evictions with the request stays inside the budget" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var peer: Peer = .{};
    var f: Fake = .{ .gpa = gpa, .peer = &peer };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 600); // bytes 100 + at
    defer s.deinit();
    var conv: [200]u32 = undefined;
    for (&conv, 1..) |*t, i| t.* = @intCast(i);
    var len: u32 = 30;
    while (len <= 200) : (len += 17) { // a conversation's turns: each keeps its history and the stable prefix, two states a pass
        peer.request(); // the drops the last pass made reach the peer with this request
        const plan = try s.begin(a, conv[0..len], len - 3, &.{20}, &.{}, null, &.{}); // its own evictions too, before the peer keeps anything
        _ = f.pass(&s, conv[0..len], plan);
        try std.testing.expect(s.held <= s.budget);
    }
    try std.testing.expect(s.counts.evicted > 0);
    try std.testing.expect(peer.max <= s.budget); // evicting at keep time instead, the peer holds the old states until the next request
}

test "a state a peer cannot resume is forgotten: the next prompt misses instead of asking again" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{}));
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const plan = try s.begin(a, &t2, 6, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), plan.from);
    s.forget(&t2, plan.from, &.{}); // the peer's answer: it lacks that state
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try std.testing.expectEqual(@as(?*pc.Entry, null), s.find(&t2, &.{}, &.{}));
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "a kept state grows only into what the budget leaves, and evicts nothing to do it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 330); // fake bytes: 100 + at
    defer s.deinit();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    _ = f.pass(&s, &t1, try s.begin(a, &t1, 4, &.{}, &.{}, null, &.{})); // 104
    const t2 = [_]u32{ 9, 9, 9, 9, 9, 9 };
    _ = f.pass(&s, &t2, try s.begin(a, &t2, 5, &.{}, &.{}, null, &.{})); // 105: 209 held, 121 left
    const e1 = s.find(&t1, &.{}, &.{}).?;
    try std.testing.expect(s.grow(e1.saved, 100)); // 309 held
    try std.testing.expectEqual(@as(u64, 204), e1.bytes);
    try std.testing.expectEqual(@as(u64, 309), s.held);
    try std.testing.expect(!s.grow(e1.saved, 22)); // 331 would pass the budget: refused, and both states stay
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(s.grow(e1.saved, 21)); // exactly the budget
    try std.testing.expectEqual(s.budget, s.held);
    var stranger: u8 = 0;
    try std.testing.expect(!s.grow(&stranger, 0)); // a state the store doesn't hold
}
