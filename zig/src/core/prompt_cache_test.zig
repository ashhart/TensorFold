//! The prompt cache's tests: a fake family whose state is a running sum, so a resumed pass equals a fresh one exactly when it should.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const rmTree = @import("prompt_imprint_test.zig").rmTree;
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

test "a learned harness state outlives its store: a fresh session on a new one resumes it from disk, equal to a fresh pass" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-test-{d}", .{std.c.getpid()});
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const harness = [_]u32{ 7, 7, 7, 7, 7, 7 }; // shared by every session: a state at 5 reads it all (lookahead 1)
    const key = imprint.Imprint.keyOf(&harness);
    defer rmTree(root);
    {
        var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
        defer im.deinit();
        {
            var f: Fake = .{ .gpa = gpa };
            var s = Store.init(gpa, f.learned(), rules, 1 << 20);
            defer s.deinit();
            s.imprint = &im;
            const first = harness ++ [_]u32{ 1, 2, 9 };
            try std.testing.expectEqual(fresh(&first), f.pass(&s, &first, try s.begin(a, &first, 8, &.{5}, &.{}, null, &.{})));
            try std.testing.expect(im.has(key));
        }
        var again = try imprint.Imprint.open(gpa, root, 3, 1 << 30); // a new server reads the index back
        defer again.deinit();
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &again;
        const second = harness ++ [_]u32{ 3, 4, 9 };
        const p = try s.begin(a, &second, 8, &.{5}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 5), p.from);
        try std.testing.expectEqual(fresh(&second), f.pass(&s, &second, p));
    }
}

test "past the learned-state cap the least recently used state is forgotten, and a file that no longer reads is too" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-cap-{d}", .{std.c.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const one = [_]u32{ 7, 7, 7, 7, 7, 7 };
    const two = [_]u32{ 8, 8, 8, 8, 8, 8 };
    var im = try imprint.Imprint.open(gpa, root, 3, 440); // One state plus conservative index/header allowance fits.
    defer im.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const p1 = one ++ [_]u32{ 1, 9 };
    try std.testing.expectEqual(fresh(&p1), f.pass(&s, &p1, try s.begin(a, &p1, 7, &.{5}, &.{}, null, &.{})));
    const p2 = two ++ [_]u32{ 1, 9 };
    try std.testing.expectEqual(fresh(&p2), f.pass(&s, &p2, try s.begin(a, &p2, 7, &.{5}, &.{}, null, &.{})));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&two)) and !im.has(imprint.Imprint.keyOf(&one)));
    var path: [512]u8 = undefined;
    try std.testing.expect(std.c.unlink(try Fake.file(&path, im.dir, imprint.Imprint.keyOf(&one))) != 0); // its file went too
    _ = std.c.unlink(try Fake.file(&path, im.dir, imprint.Imprint.keyOf(&two))); // a file lost behind the store's back
    var g: Fake = .{ .gpa = gpa };
    var t = Store.init(gpa, g.learned(), rules, 1 << 20);
    defer t.deinit();
    t.imprint = &im;
    const p3 = two ++ [_]u32{ 2, 9 };
    const plan = try t.begin(a, &p3, 7, &.{5}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), plan.from); // the read failed: a fresh pass, and the state is forgotten
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(&two)));
    try std.testing.expectEqual(fresh(&p3), g.pass(&t, &p3, plan));
    try std.testing.expect(im.has(imprint.Imprint.keyOf(&two))); // learned again by that pass
}

test "learned states are prompt arithmetic: a request that decodes rows below one prefills instead, and a decoded state is never learned" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root_buf: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&root_buf, "/tmp/tf-learn-spans-{d}", .{std.c.getpid()});
    defer rmTree(root);
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 };
    const harness = [_]u32{ 7, 7, 7, 7, 7, 7 };
    var im = try imprint.Imprint.open(gpa, root, 3, 1 << 30);
    defer im.deinit();
    {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        const first = harness ++ [_]u32{ 1, 2, 9 };
        _ = f.pass(&s, &first, try s.begin(a, &first, 8, &.{5}, &.{}, null, &.{}));
        try std.testing.expect(im.has(imprint.Imprint.keyOf(&harness)));
    }
    const second = harness ++ [_]u32{ 3, 4, 9 };
    for ([_]struct { spans: []const [2]u32, from: u32 }{ .{ .spans = &.{.{ 2, 4 }}, .from = 0 }, .{ .spans = &.{.{ 6, 8 }}, .from = 5 } }) |c| {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.learned(), rules, 1 << 20);
        defer s.deinit();
        s.imprint = &im;
        const p = try s.begin(a, &second, 8, &.{5}, &.{}, null, c.spans);
        try std.testing.expectEqual(c.from, p.from);
    }
    const other = [_]u32{ 8, 8, 8, 8, 8, 8, 1, 9 };
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.learned(), rules, 1 << 20);
    defer s.deinit();
    s.imprint = &im;
    const p = try s.begin(a, &other, 7, &.{5}, &.{}, null, &.{.{ 1, 3 }});
    try std.testing.expectEqual(@as(u32, 0), p.from);
    f.at = 5;
    try std.testing.expect(s.keep(&other, 5, null, &.{}, &.{.{ 1, 3 }}));
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(other[0..6])));
}

test "Store learning preserves the disk floor, backs off writes across keys, and recovers without retained failures" {
    const Disk = struct {
        var space: ?u64 = 0;
        var tick: u64 = 100;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return tick;
        }
    };
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-learn-backoff-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 11, 1 << 20);
    defer im.deinit();
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    var fake: Fake = .{ .gpa = a };
    var store = Store.init(a, fake.learned(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    for (0..6) |i| {
        const token: u32 = @intCast(i + 1);
        const prompt = [_]u32{ token, token, token, token, token, token, 9, 10 };
        if (i == 2) {
            Disk.space = 10000;
            fake.fail_write = true;
        }
        if (i == 5) {
            Disk.tick += 100;
            fake.fail_write = false;
        }
        const plan = try store.begin(arena.allocator(), &prompt, 7, &.{5}, &.{}, null, &.{});
        try std.testing.expectEqual(fresh(&prompt), fake.pass(&store, &prompt, plan));
        try std.testing.expectEqual(@as(usize, if (i < 2) 0 else if (i < 5) 1 else 2), fake.writes);
    }
    try std.testing.expectEqual(@as(usize, 1), im.metas.items.len);
    try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
}

test "learning reserves the existing index rewrite above the disk floor and resumes an unchanged kept key" {
    const Disk = struct {
        var space: u64 = 485;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return 100;
        }
    };
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-learn-index-floor-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 12, 1 << 20);
    defer im.deinit();
    try im.add(imprint.Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 1);
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    var fake: Fake = .{ .gpa = a };
    var store = Store.init(a, fake.learned(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    const plan = try store.begin(arena.allocator(), &prompt, 7, &.{5}, &.{}, null, &.{});
    _ = fake.pass(&store, &prompt, plan);
    try std.testing.expectEqual(@as(usize, 0), fake.writes);
    Disk.space += try im.indexScratchBytes();
    fake.at = 5;
    try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), fake.writes);
    try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
}

test "actual Store learn writes neither half on peer refusal, backs off, and removes both halves after a peer fault" {
    const Disk = struct {
        var tick: u64 = 100;
        fn free(_: [:0]const u8) ?u64 {
            return 10000;
        }
        fn clock() ?u64 {
            return tick;
        }
    };
    const a = std.testing.allocator;
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-pair-refuse-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 51, 1 << 20);
    defer im.deinit();
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    var fake: Fake = .{ .gpa = a, .at = 5, .disk_need_bytes = 1 };
    var store = Store.init(a, fake.paired(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    _ = try store.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    for (0..32) |_| try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 0), fake.writes);
    try std.testing.expectEqual(@as(usize, 0), fake.disk_peer_writes);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_queries);
    Disk.tick += 100;
    fake.disk_need_bytes = 0;
    fake.disk_fail_peer = true;
    try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), fake.writes);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_peer_writes);
    try std.testing.expect(!fake.disk_reserved);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_finishes);
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(prompt[0..6])));
    try std.testing.expectEqual(@as(u64, 0), im.admission.reserved);
    var file_path: [512]u8 = undefined;
    const file = try Fake.file(&file_path, im.dir, imprint.Imprint.keyOf(prompt[0..6]));
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd >= 0) {
        _ = std.c.close(fd);
        return error.OrphanedHalf;
    }
}
test "under-cap disk pressure selects the same LRU victim before either half is persisted" {
    const Disk = struct {
        var space: u64 = 0;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return 100;
        }
        fn release() void {
            space += 16;
        }
    };
    const a = std.testing.allocator;
    var path: [128]u8 = undefined;
    const root = try std.fmt.bufPrint(&path, "/tmp/tf-pair-victim-{d}", .{std.c.getpid()});
    defer rmTree(root);
    var im = try imprint.Imprint.open(a, root, 52, 1 << 20);
    defer im.deinit();
    try im.add(imprint.Imprint.keyOf(&.{ 1, 2 }), 1, &.{ 1, 2 }, &.{}, 16);
    var victim_path: [512]u8 = undefined;
    const victim_fd = std.c.open(try Fake.file(&victim_path, im.dir, imprint.Imprint.keyOf(&.{ 1, 2 })), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (victim_fd < 0) return error.Create;
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(victim_fd, 16));
    _ = std.c.close(victim_fd);
    im.admission = .{ .floor = 100, .free = Disk.free, .clock = Disk.clock };
    const prompt = [_]u32{ 7, 7, 7, 7, 7, 7, 9, 10 };
    Disk.space = 100 + 105 + 256 + (6 * 4) + (try im.indexScratchBytes()) - 8;
    var fake: Fake = .{ .gpa = a, .at = 5, .disk_need_bytes = 8, .disk_release = Disk.release };
    var store = Store.init(a, fake.paired(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    store.imprint = &im;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    _ = try store.lookup(arena.allocator(), &prompt, 7, &.{5}, &.{}, &.{});
    try std.testing.expect(store.keep(&prompt, 5, null, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), fake.disk_forgets);
    try std.testing.expect(!im.has(imprint.Imprint.keyOf(&.{ 1, 2 })));
    try std.testing.expectEqual(@as(usize, 1), fake.writes);
    try std.testing.expectEqual(@as(usize, 1), fake.disk_peer_writes);
    try std.testing.expect(!fake.disk_reserved);
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
