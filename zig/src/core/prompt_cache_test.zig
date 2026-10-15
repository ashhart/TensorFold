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

/// Prompt calls of a pass from `from`: `step` rows on from each cut, sooner at any of `cuts` (as qwen27_slots.prefill).
fn calls(len: u32, from: u32, step: u32, cuts: []const []const u32) usize {
    var n: usize = 0;
    var at = from;
    while (at < len) : (n += 1) {
        var end = @min(len, at + step);
        for (cuts) |list| for (list) |c| if (c > at and c < end) {
            end = c;
        };
        at = end;
    }
    return n;
}

test "a short prompt: a planned family keeps its marks, an unplanned one only its history's mark, at a chunk end its pass cuts anyway" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var planned = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4 }, 1 << 20);
    defer planned.deinit();
    try std.testing.expectEqualSlices(u32, &.{4}, (try planned.begin(a, &t1, 7, &.{}, &.{}, null, &.{})).marks); // 12 tokens, under the default min_prompt
    var serial = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .chunk = 3 }, 1 << 20);
    defer serial.deinit();
    const p = try serial.begin(a, &t1, 7, &.{2}, &.{}, null, &.{}); // the shared block at 2 waits for min_prompt
    try std.testing.expectEqualSlices(u32, &.{6}, p.marks); // history 7 floored to the pass's cut at 6 (3, 6, 9, 12)
    try std.testing.expectEqual(calls(12, 0, 3, &.{}), calls(12, 0, 3, &.{p.marks})); // no extra prompt call
    try std.testing.expect(calls(12, 0, 3, &.{&.{7}}) > calls(12, 0, 3, &.{})); // the history's own position would add one
    try std.testing.expectEqual(fresh(&t1), f.pass(&serial, &t1, p));
    try std.testing.expectEqual(@as(u32, 6), serial.find(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0 }, &.{}, &.{}).?.at); // the same history, a new tail
    var taps = Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .chunk = 1 }, 1 << 20); // a pass that keeps at any row (Flash Next's mark taps)
    defer taps.deinit();
    try std.testing.expectEqualSlices(u32, &.{7}, (try taps.begin(a, &t1, 7, &.{2}, &.{}, null, &.{})).marks);
    var none = Store.init(gpa, f.snapshots(), .{ .lookahead = 1 }, 1 << 20); // naming no chunk rows: nothing, as before
    defer none.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try none.begin(a, &t1, 7, &.{}, &.{}, null, &.{})).marks.len);
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
    s.rules.warm = .on; // replies prefilled in the background: the resumed state serves the next turn
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, &prompt, 9, 10, &.{}, &.{}, &.{}));
    try std.testing.expectEqualSlices(u32, &.{1}, try s.marks(a, &prompt, 0, 1, &.{}, &.{}, &.{})); // from the start the history's mark stays
    // a warm pass (history the whole prompt) keeps where its state can be keyed, however near it resumed
    try std.testing.expectEqualSlices(u32, &.{11}, try s.marks(a, &prompt, 10, 12, &.{}, &.{}, &.{}));
    s.rules.warm = .waits; // those passes wait for a quiet engine and may never run: the history's mark stays
    try std.testing.expectEqualSlices(u32, &.{10}, try s.marks(a, &prompt, 9, 10, &.{}, &.{}, &.{}));
    try std.testing.expectEqualSlices(u32, &.{11}, try s.marks(a, &prompt, 10, 12, &.{}, &.{}, &.{}));
}

test "under min_prompt the history's mark floors to its pass's last cut from the resume point, so it adds no prompt call, and the next turn resumes there exactly" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .warm = .on, .min_prompt = 64, .min_gap = 8, .chunk = 4 }, 1 << 20); // the 27B's rules, scaled down
    defer s.deinit();
    var t2: [34]u32 = undefined;
    for (&t2, 1..) |*t, i| t.* = @intCast(i);
    const t1 = t2[0..23]; // the first turn: history 18, the request's starts at 4 and 11 (a system block's cut)
    const starts1 = [_]u32{ 4, 11 };
    const p = try s.begin(a, t1, 18, &.{}, &starts1, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{15}, p.marks); // its cuts 4, 8, 11, 15, 19, 23: a step on from 11, not 16 on a grid from 0
    try std.testing.expectEqual(calls(23, 0, 4, &.{&starts1}), calls(23, 0, 4, &.{ &starts1, p.marks }));
    try std.testing.expect(calls(23, 0, 4, &.{ &starts1, &.{16} }) > calls(23, 0, 4, &.{&starts1}));
    try std.testing.expectEqual(fresh(t1), f.pass(&s, t1, p));
    const starts2 = [_]u32{ 4, 11, 25 }; // the next turn: history 30, the reply decoded in rows 19 to 27
    const spans2 = [_][2]u32{.{ 19, 27 }};
    const q = try s.begin(a, &t2, 30, &.{}, &starts2, null, &spans2);
    try std.testing.expectEqual(@as(u32, 15), q.from);
    try std.testing.expectEqualSlices(u32, &.{27}, q.marks); // from 15 its cuts are 19, 23, 25, 27, 31, 34: the span's end, not 29
    try std.testing.expectEqual(calls(34, 15, 4, &.{ &starts2, &.{ 19, 27 } }), calls(34, 15, 4, &.{ &starts2, &.{ 19, 27 }, q.marks }));
    try std.testing.expectEqual(fresh(&t2), f.pass(&s, &t2, q));
    try std.testing.expectEqualSlices(u32, &.{29}, try s.marks(a, &t2, 17, 30, &.{}, &starts1, &.{})); // resumed at a reply's end: 21, 25, 29, not 27 from 11
    try std.testing.expectEqual(calls(34, 17, 4, &.{&starts1}), calls(34, 17, 4, &.{ &starts1, &.{29} }));
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, &t2, 0, 0, &.{}, &.{}, &.{})); // a raw prompt keeps nothing
    try std.testing.expectEqualSlices(u32, &.{}, try s.marks(a, t2[0..30], 15, 30, &.{}, &starts2, &.{})); // a warm pass's end is its last cut: the host keeps it
    var long: [70]u32 = undefined; // past min_prompt: the history exactly and the shared block, as before
    for (&long, 0..) |*t, i| t.* = @intCast(200 + i);
    try std.testing.expectEqualSlices(u32, &.{ 40, 66 }, (try s.begin(a, &long, 66, &.{40}, &.{}, null, &.{})).marks);
}

test "a short prompt's mark inside its shared block is the block's state: the reply's own keep evicts another state, and the next turn resumes it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .warm = .on, .min_prompt = 64, .chunk = 4 }, 350); // fake bytes 100 + at: three states pass it
    defer s.deinit();
    var other: [13]u32 = undefined; // an older conversation's state, the least recently used
    for (&other, 0..) |*t, i| t.* = @intCast(90 + i);
    _ = f.pass(&s, &other, try s.begin(a, &other, 12, &.{}, &.{}, null, &.{}));
    var ctx: [30]u32 = undefined; // a 20-token system block, the user's turn to 22, the generation prompt to 25, then the reply
    for (&ctx, 1..) |*t, i| t.* = @intCast(i);
    const starts = [_]u32{ 4, 8, 12, 16, 19 }; // the request's starts, the last one before the conversation's own text
    const p = try s.begin(a, ctx[0..25], 22, &.{20}, &starts, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{19}, p.marks);
    _ = f.pass(&s, ctx[0..25], p);
    _ = f.pass(&s, &ctx, .{ .from = 25, .marks = &.{29} }); // the reply's end kept as the 27B keeps it: room is made first
    try std.testing.expect(s.find(&other, &.{}, &.{}) == null);
    var next = ctx; // the next turn drops the reply's reasoning: it shares only the history
    for (next[22..]) |*t| t.* = 77;
    const q = try s.begin(a, &next, 28, &.{20}, &starts, null, &.{});
    try std.testing.expectEqual(@as(u32, 19), q.from);
    try std.testing.expectEqual(fresh(&next), f.pass(&s, &next, q));
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

test "an entry pinned for a restore still to come outlives the lookups set up after it; unpinned, it goes again" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 220); // fake bytes: 100 + at
    defer s.deinit();
    const x = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &x, try s.begin(a, &x, 3, &.{}, &.{}, null, &.{})); // 103 bytes, the oldest
    const y = [_]u32{ 6, 6, 6, 6 };
    _ = f.pass(&s, &y, try s.begin(a, &y, 3, &.{}, &.{}, null, &.{})); // 103: 206 held
    const xa = [_]u32{ 5, 5, 5, 5, 9 };
    const l = try s.lookup(a, &xa, 3, &.{}, &.{}, &.{}); // a job set up to resume x
    const e = l.entry.?;
    _ = s.pin(e);
    const z = [_]u32{ 7, 7, 7, 7, 7, 7, 7, 7 };
    _ = try s.lookup(a, &z, 7, &.{}, &.{}, &.{}); // the next job's room: y goes, not x
    try std.testing.expect(s.find(&xa, &.{}, &.{}) == e);
    try std.testing.expect(s.find(&.{ 6, 6, 6, 6 }, &.{}, &.{}) == null);
    const w: [130]u32 = @splat(8);
    _ = try s.lookup(a, &w, 120, &.{}, &.{}, &.{}); // 220 bytes, and nothing but x to evict: it stays
    try std.testing.expect(!s.keep(&w, 120, null, &.{}, &.{})); // and a keep with no room is refused
    try std.testing.expect(s.find(&xa, &.{}, &.{}) == e);
    s.unpin(e);
    _ = try s.lookup(a, &w, 120, &.{}, &.{}, &.{});
    try std.testing.expect(s.find(&xa, &.{}, &.{}) == null);
    try std.testing.expectEqual(s.entries.items.len, f.live);
}

test "two jobs pin one entry and the first restore fails: lookups miss it, and it lives until the second job unpins" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const x = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &x, try s.begin(a, &x, 3, &.{}, &.{}, null, &.{}));
    const xa = [_]u32{ 5, 5, 5, 5, 9 };
    const xb = [_]u32{ 5, 5, 5, 5, 8 };
    const e = s.pin((try s.lookup(a, &xa, 3, &.{}, &.{}, &.{})).entry.?);
    try std.testing.expect(s.pin((try s.lookup(a, &xb, 3, &.{}, &.{}, &.{})).entry.?) == e);
    const held = s.held;
    s.resumed(e, &xa, false); // the first job's restore failed: it prefilled from 0
    s.unpin(e);
    try std.testing.expect(s.find(&xb, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), f.live); // the second job's restore still reads it
    try std.testing.expectEqual(held, s.held);
    try std.testing.expectEqual(@as(u32, 1), e.pins);
    s.resumed(e, &xb, true); // no longer kept: its report changes nothing
    s.unpin(e);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(u64, 0), s.held);
}

test "forget leaves a pinned entry to its restore" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const x = [_]u32{ 5, 5, 5, 5 };
    _ = f.pass(&s, &x, try s.begin(a, &x, 3, &.{}, &.{}, null, &.{}));
    const xa = [_]u32{ 5, 5, 5, 5, 9 };
    const e = s.pin((try s.lookup(a, &xa, 3, &.{}, &.{}, &.{})).entry.?);
    s.forget(&xa, 3, &.{});
    try std.testing.expect(s.find(&xa, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    s.unpin(e);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}
