//! Long documents and a new question in the same last user message reuse an interior cold cut.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;

test "same document and a new question resume an interior cold cut exactly" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 128 * 1024 + 64);
    for (prompt, 0..) |*t, i| t.* = @intCast(i % 65521);
    const question = prompt.len - 512;
    for ([_]pc.Rules{ .{ .chunk = 2048, .warm = .on }, .{ .grid = 2048, .planned = true, .lookahead = 1 } }) |rules| {
        var f: Fake = .{ .gpa = gpa };
        var store = pc.Store.init(gpa, f.snapshots(), rules, 1 << 24);
        defer store.deinit();
        const p = try store.beginUser(a, prompt, @intCast(prompt.len - 8), 16, @intCast(prompt.len - 8), &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(fresh(prompt), f.pass(&store, prompt, p));
        const followup = try a.dupe(u32, prompt);
        followup[question] += 1;
        const q = try store.beginUser(a, followup, @intCast(followup.len - 8), 16, @intCast(followup.len - 8), &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(@as(u32, 126 * 1024 + if (rules.planned) @as(u32, 0) else 16), q.from);
        try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
        try std.testing.expectEqual(fresh(followup), f.pass(&store, followup, q));
        try std.testing.expect(store.held + store.spare() <= store.budget);
    }
}

const interior = @import("prompt_interior.zig");

test "interior cut is latest cold cut after existing marks, starts and decoded edges" {
    const end: u32 = 13000;
    const limit = end - interior.QUESTION_TOKENS;
    const r: pc.Rules = .{ .chunk = 2048 };
    try std.testing.expectEqual(@as(u32, 10240), interior.cut(r, end + 8, 0, end, 16, &.{}, &.{}, &.{}));
    // A turn boundary is already a cold cut, even off the zero-anchored grid.
    try std.testing.expectEqual(limit, interior.cut(r, end + 8, 0, end, 16, &.{limit}, &.{}, &.{}));
    // Existing marks restart chunks, as do canonical decoded span edges.
    try std.testing.expectEqual(@as(u32, 10900), interior.cut(r, end + 8, 0, end, 16, &.{}, &.{.{ 8500, 10900 }}, &.{}));
    try std.testing.expectEqual(@as(u32, 10924), interior.cut(r, end + 8, 0, end, 16, &.{}, &.{}, &.{ 16, 8876, end }));
    // CUDA's pass owns its cuts and ignores request starts.
    try std.testing.expectEqual(@as(u32, 10240), interior.cut(.{ .chunk = 2048, .chunk_starts = false }, end + 8, 0, end, 16, &.{limit}, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 10240), interior.cut(.{ .planned = true, .grid = 2048 }, end + 8, 0, end, 16, &.{}, &.{}, &.{}));
    try std.testing.expectEqual(limit - 1, interior.cut(.{ .planned = true, .lookahead = 1 }, end + 8, 0, end, 16, &.{ limit - 1, limit, limit + 1 }, &.{}, &.{}));
}

test "interior cut refuses short, unknown, exhausted and unavailable cold geometry" {
    const r: pc.Rules = .{ .chunk = 2048 };
    try std.testing.expectEqual(@as(u32, 0), interior.cut(r, 13008, 0, 13000, 0, &.{}, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 0), interior.cut(r, 13008, 0, 13000, 5000, &.{}, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 0), interior.cut(r, 13008, 0, 13009, 16, &.{}, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 0), interior.cut(r, 13008, 12000, 13000, 16, &.{}, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 0), interior.cut(.{}, 13008, 0, 13000, 16, &.{}, &.{}, &.{}));
    try std.testing.expectEqual(@as(u32, 0), interior.cut(.{ .planned = true }, 13008, 0, 13000, 16, &.{}, &.{}, &.{}));
}

test "interior copy is skipped unless endpoint, rewind and every required mark fit together" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 13008);
    @memset(prompt, 7);
    const r: pc.Rules = .{ .chunk = 2048, .min_prompt = 0 };
    const at: u32 = 10256; // ordinary rewind at 16 restarts the cold chunks
    const all: u64 = 100 + 13000 + 100 + 16 + 100 + at;
    for ([_]u64{ 0, 13100, all - 1, all, all + 1 }) |budget| for ([_]usize{ 0, 1, 2, 3, 4 }) |slots| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), r, budget);
        defer s.deinit();
        s.most = slots;
        const p = try s.beginUser(a, prompt, 13000, 16, 13000, &.{}, &.{}, null, &.{});
        const extra = std.mem.indexOfScalar(u32, p.marks, at) != null;
        try std.testing.expectEqual(budget >= all and slots >= 3, extra);
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
        try std.testing.expect(s.held + s.spare() <= budget);
        try std.testing.expect(s.entries.items.len <= slots);
        if (extra) try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    };
}

test "short messages retain only the ordinary endpoint and rewind states" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const prompt = try arena.allocator().alloc(u32, interior.MIN_MESSAGE_TOKENS + 15);
    @memset(prompt, 3);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .chunk = 2048, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const p = try s.beginUser(arena.allocator(), prompt, @intCast(prompt.len - 1), 16, @intCast(prompt.len - 1), &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, @intCast(prompt.len - 1) }, p.marks);
}

// A chunk-sensitive integer backend: splitting even one cold chunk changes its observable state.
fn chunkPass(f: *Fake, store: ?*pc.Store, prompt: []const u32, plan: pc.Plan, step: u32) !usize {
    if (plan.from == 0) {
        f.at = 0;
        f.sum = 0;
    }
    var calls: usize = 0;
    while (f.at < prompt.len) {
        const begin = f.at;
        var stop = @min(@as(u32, @intCast(prompt.len)), begin + step);
        for (plan.marks) |m| if (m > begin and m < stop) {
            stop = m;
        };
        for (prompt[begin..stop]) |t| f.sum = f.sum *% 31 +% t;
        f.sum ^= @as(u64, stop - begin) *% 0x9e3779b97f4a7c15;
        f.at = stop;
        calls += 1;
        if (store) |s| if (std.mem.indexOfScalar(u32, plan.marks, stop) != null) {
            try std.testing.expect(s.keep(prompt, stop, null, &.{}, &.{}));
        };
    }
    return calls;
}

test "interior keeps add no cold chunk and resume a chunk-sensitive backend bit for bit" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 128 * 1024 + 64);
    for (prompt, 0..) |*t, i| t.* = @intCast(i % 101);
    for ([_]pc.Rules{ .{ .chunk = 1536, .chunk_starts = false, .lookahead = 1 }, .{ .chunk = 2048 }, .{ .planned = true, .grid = 2048, .lookahead = 1 } }) |rules| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), rules, 1 << 24);
        defer s.deinit();
        const end: u32 = @intCast(prompt.len - 8);
        const p = try s.beginUser(a, prompt, end, 16, end, &.{}, &.{}, null, &.{});
        const at = interior.cut(rules, prompt.len, 0, end, 16, &.{}, &.{}, p.marks);
        var ordinary: std.ArrayList(u32) = .empty;
        defer ordinary.deinit(a);
        for (p.marks) |m| if (m != at) {
            try ordinary.append(a, m);
        };
        var cold: Fake = .{ .gpa = gpa };
        const step = if (rules.planned) rules.grid else rules.chunk;
        const cold_calls = try chunkPass(&cold, null, prompt, .{ .marks = ordinary.items }, step);
        const kept_calls = try chunkPass(&f, &s, prompt, p, step);
        try std.testing.expectEqual(cold_calls, kept_calls);
        try std.testing.expectEqual(cold.sum, f.sum);
        const edited = try a.dupe(u32, prompt);
        edited[edited.len - 512] += 1;
        const q = try s.beginUser(a, edited, end, 16, end, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(at, q.from);
        _ = try chunkPass(&f, &s, edited, q, step);
        // The existing stable-prefix policy may add a later mark on this lookup. Match that ordinary policy.
        try ordinary.appendSlice(a, q.marks);
        _ = try chunkPass(&cold, null, edited, .{ .marks = ordinary.items }, step);
        try std.testing.expectEqual(cold.sum, f.sum);
    }
}

test "interior cut for a tap bank uses only guaranteed request chunk starts" {
    const r: pc.Rules = .{ .chunk = 2048, .chunk_marks = false };
    const starts = [_]u32{ 2048, 4096, 6144, 8192, 10240, 12288 };
    try std.testing.expectEqual(@as(u32, 10240), interior.cut(r, 13008, 0, 13000, 16, &starts, &.{}, &.{ 16, 9000, 13000 }));
    // No metadata: guessing whether the bank will allocate could introduce a new cold split.
    try std.testing.expectEqual(@as(u32, 0), interior.cut(r, 13008, 0, 13000, 16, &.{}, &.{}, &.{ 16, 9000, 13000 }));
}

test "interior budget includes outstanding foreign borrows, but ordinary eviction still frees old states" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 13008);
    @memset(prompt, 7);
    const other = try a.alloc(u32, 1008);
    @memset(other, 9);
    for ([_]bool{ false, true }) |borrowed| {
        var f: Fake = .{ .gpa = gpa };
        var s = pc.Store.init(gpa, f.snapshots(), .{ .chunk = 2048, .min_prompt = 0 }, 23572);
        defer s.deinit();
        f.at = 1000;
        try std.testing.expect(s.keep(other, 1000, null, &.{}, &.{}));
        const held = s.entries.items[0];
        if (borrowed) held.borrowers = 1;
        const p = try s.beginUser(a, prompt, 13000, 16, 13000, &.{}, &.{}, null, &.{});
        try std.testing.expectEqual(!borrowed, std.mem.indexOfScalar(u32, p.marks, 10256) != null);
        try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
        try std.testing.expect(s.held + s.spare() <= s.budget);
        if (borrowed) {
            try std.testing.expectEqual(@as(u32, 1), held.borrowers);
            s.release(held);
        } else try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    }
}

test "interior copies share the existing spare-buffer budget without exceeding it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const prompt = try arena.allocator().alloc(u32, 13008);
    @memset(prompt, 7);
    var f: Fake = .{ .gpa = gpa, .spare_bytes = 23572 };
    var s = pc.Store.init(gpa, f.pooled(), .{ .chunk = 2048, .min_prompt = 0 }, 23572);
    defer s.deinit();
    const p = try s.beginUser(arena.allocator(), prompt, 13000, 16, 13000, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 10256, 13000 }, p.marks);
    try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.held + s.spare() <= s.budget);
}

test "interior snapshot keys retain lookahead and canonical decoded arithmetic" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 13008);
    @memset(prompt, 7);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .chunk = 2048, .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const p = try s.beginUser(a, prompt, 13000, 16, 13000, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(fresh(prompt), f.pass(&s, prompt, p));
    const at = p.marks[1];
    const edited = try a.dupe(u32, prompt);
    edited[at] += 1; // outside consumed rows, inside this family's MTP key
    const q = try s.lookupUser(a, edited, @intCast(edited.len), 13000, 16, 13000, &.{}, &.{}, &.{});
    defer if (q.entry) |e| s.release(e);
    try std.testing.expect(q.entry == null or q.entry.?.at < at);
    const different = try s.lookupUser(a, prompt, @intCast(prompt.len), 13000, 16, 13000, &.{}, &.{}, &.{.{ 100, 200 }});
    defer if (different.entry) |e| s.release(e);
    try std.testing.expect(different.entry == null or different.entry.?.at < at);
}

test "interior state requires a known long user span, not a long tool or reply tail" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const prompt = try a.alloc(u32, 13008);
    @memset(prompt, 7);
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .chunk = 2048, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    for ([_]u32{ 0, 1000, 13001 }) |end| {
        const p = try s.beginUser(a, prompt, 13000, 16, end, &.{}, &.{}, null, &.{});
        try std.testing.expectEqualSlices(u32, &.{ 16, 13000 }, p.marks);
    }
    // Old direct callers and background warm passes supply no user end and retain their previous policy.
    const legacy = try s.beginRewind(a, prompt, 13000, 16, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 13000 }, legacy.marks);
}
