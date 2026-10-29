//! The production snapshot index against the prior scan, lifetime boundaries and allocation faults.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const oracle = @import("prompt_cache_index_oracle.zig");
const index = @import("prompt_cache_index.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const a = std.testing.allocator;

pub fn keep(s: *pc.Store, f: *Fake, prompt: []const u32, at: u32, spans: []const modes.Span) !void {
    f.at = at;
    f.sum = @import("prompt_cache_test.zig").fresh(prompt[0..at]);
    try std.testing.expect(s.keep(prompt, at, null, &.{}, spans));
}

fn duplicate(s: *pc.Store, f: *Fake, from: *const pc.Entry) !*pc.Entry {
    const e = try a.create(pc.Entry);
    errdefer a.destroy(e);
    e.* = from.*;
    e.ext = .empty; // its own extension links and token copy, not the original's
    e.own = &.{};
    e.tokens = try a.dupe(u32, from.tokens);
    errdefer a.free(e.tokens);
    e.last = try a.dupe(u32, from.last);
    errdefer a.free(e.last);
    e.decode_spans = try a.dupe(modes.Span, from.decode_spans);
    errdefer a.free(e.decode_spans);
    f.at = from.at;
    const saved: *const Fake.State = @ptrCast(@alignCast(from.saved));
    f.sum = saved.sum;
    e.saved = try s.family.vtable.save(s.family.ptr, null, from.at);
    errdefer s.family.vtable.drop(s.family.ptr, e.saved);
    try index.add(s, e);
    @import("prompt_cache_policy.zig").link(s, e);
    s.held += e.bytes;
    return e;
}

test "snapshot index: longest-first selection keeps owner order and equal-position winner" {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    for ([_]u32{ 8, 4, 12, 6 }) |at| try keep(&s, &f, &prompt, at, &.{});
    const first = s.entries.items[0];
    const later = try duplicate(&s, &f, first);
    try oracle.check(&s);
    for ([_]u32{ 8, 4, 12, 6, 8 }, s.entries.items) |at, e| try std.testing.expectEqual(at, e.at);
    try std.testing.expect(s.find(prompt[0..9], &.{}, &.{}) == first);
    s.remove(2);
    try std.testing.expect(s.find(&prompt, &.{}, &.{}) == first);
    s.remove(0);
    try std.testing.expect(s.find(&prompt, &.{}, &.{}) == later);
    try oracle.check(&s);
    s.remove(s.entries.items.len - 1);
    try std.testing.expectEqual(@as(u32, 6), s.find(&prompt, &.{}, &.{}).?.at);
    try oracle.check(&s);
}

fn next(seed: *u64) u32 {
    seed.* ^= seed.* << 13;
    seed.* ^= seed.* >> 7;
    seed.* ^= seed.* << 17;
    return @truncate(seed.*);
}

fn spanSet(which: u32) []const modes.Span {
    return switch (which % 4) {
        0 => &.{},
        1 => &.{.{ 2, 7 }},
        2 => &.{.{ 12, 24 }},
        else => &.{ .{ 2, 7 }, .{ 12, 24 } },
    };
}

test "snapshot index: randomized mutation and query oracle covers modes grids ties and misses" {
    var seed: u64 = 0xc0ffee8a02;
    var queries: [4][257]u32 = undefined;
    for (&queries, 0..) |*q, branch| for (q, 0..) |*token, i| {
        token.* = @intCast(1 + i * 3 + if (i >= branch * 40 and branch > 0) branch * 1000 else 0);
    };
    const explicit = [_]u32{ 0, 4, 9, 13, 18, 31, 45, 80, 96, 128, 160, 192, 224, 256 };
    for (0..6) |mode| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{
            .lookahead = @intCast(mode % 2),
            .planned = mode >= 2,
            .grid = if (mode < 4) 8 else 0,
            .min_prompt = 0,
        }, 1 << 25);
        defer s.deinit();
        const starts: []const u32 = if (mode >= 4) &explicit else &.{};
        for (0..400) |_| {
            const op = next(&seed) % 10;
            if (op < 7) {
                const branch = next(&seed) % queries.len;
                const at = 1 + next(&seed) % 240;
                try keep(&s, &f, &queries[branch], at, spanSet(next(&seed)));
            } else if (s.entries.items.len > 0) {
                const at = next(&seed) % s.entries.items.len;
                if (op == 7) s.expired(s.entries.items[at].saved) else if (op == 8) s.remove(at) else s.collectExpired();
            }
            try oracle.check(&s);
            try std.testing.expectEqual(s.entries.items.len, f.live);
            for (0..8) |_| {
                const query = &queries[next(&seed) % queries.len];
                const n = next(&seed) % (query.len + 1);
                const requested = spanSet(next(&seed));
                try std.testing.expect(oracle.find(&s, query[0..n], starts, requested) == s.find(query[0..n], starts, requested));
            }
        }
    }
}

test "snapshot index: explicit lookahead spans and planned starts retain old winner" {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .planned = true, .grid = 4 }, 1 << 20);
    defer s.deinit();
    try keep(&s, &f, &prompt, 4, &.{});
    try keep(&s, &f, &prompt, 8, &.{.{ 5, 7 }});
    try keep(&s, &f, &prompt, 12, &.{});
    try std.testing.expectEqual(@as(u32, 12), s.find(&prompt, &.{}, &.{}).?.at);
    try std.testing.expectEqual(@as(u32, 8), s.find(&prompt, &.{ 0, 4, 8 }, &.{.{ 5, 7 }}).?.at);
    try std.testing.expectEqual(@as(u32, 4), s.find(&prompt, &.{ 0, 4, 8 }, &.{}).?.at);
    var changed = prompt;
    changed[8] = 99;
    try std.testing.expectEqual(@as(u32, 4), s.find(&changed, &.{}, &.{.{ 5, 7 }}).?.at);
    try std.testing.expect(s.find(prompt[0..4], &.{}, &.{}) == null);
    try std.testing.expect(s.find(&prompt, &.{ 0, 3, 7 }, &.{}) == null);
    try oracle.check(&s);
}

test "snapshot index: two borrowers survive invalidation and a later insertion" {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    try keep(&s, &f, &prompt, 4, &.{});
    try keep(&s, &f, &prompt, 8, &.{});
    const one = (try s.lookup(arena.allocator(), &prompt, 0, &.{}, &.{}, &.{})).entry.?;
    const two = (try s.lookup(arena.allocator(), &prompt, 0, &.{}, &.{}, &.{})).entry.?;
    try std.testing.expect(one == two);
    s.expired(one.saved);
    s.collectExpired();
    try std.testing.expectEqual(@as(u32, 4), s.find(&prompt, &.{}, &.{}).?.at);
    try keep(&s, &f, &prompt, 12, &.{});
    s.release(one);
    try std.testing.expectEqual(@as(usize, 3), f.live);
    try oracle.check(&s);
    s.release(two);
    try std.testing.expectEqual(@as(usize, 2), f.live);
    try oracle.check(&s);
    const borrowed = (try s.lookup(arena.allocator(), &prompt, 0, &.{}, &.{}, &.{})).entry.?;
    s.clear();
    try std.testing.expect(s.find(&prompt, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    try oracle.check(&s);
    s.release(borrowed);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try oracle.check(&s);
}

test "snapshot index: failed restore removes only its indexed state and retries a shorter one" {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var f: Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    try keep(&s, &f, &prompt, 4, &.{});
    try keep(&s, &f, &prompt, 8, &.{});
    f.fail_restore = true;
    const failed = try s.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), failed.from);
    try std.testing.expectEqual(@as(u64, 0), s.counts.hits);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try oracle.check(&s);
    f.fail_restore = false;
    const resumed = try s.begin(arena.allocator(), &prompt, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 4), resumed.from);
    try std.testing.expectEqual(@import("prompt_cache_test.zig").fresh(&prompt), f.pass(&s, &prompt, resumed));
    try oracle.check(&s);
}

fn full(s: *pc.Store, f: *Fake, prompt: []const u32) !void {
    try keep(s, f, prompt, 1, &.{});
    while (s.entries.items.len < s.entries.capacity) try keep(s, f, prompt, @intCast(s.entries.items.len + 1), &.{});
    try std.testing.expectEqual(s.entries.capacity, s.by_position.capacity);
}

test "snapshot index: either publication allocation can fail without a partial pointer" {
    var prompt: [128]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    for (0..2) |fail_at| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{}, 1 << 20);
        defer s.deinit();
        try full(&s, &f, &prompt);
        const old_len = s.entries.items.len;
        var unpublished = s.entries.items[0].*;
        unpublished.at = 100;
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_at, .resize_fail_index = 0 });
        s.gpa = failing.allocator();
        const result = index.add(&s, &unpublished);
        s.gpa = a;
        try std.testing.expectError(error.OutOfMemory, result);
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expectEqual(old_len, s.entries.items.len);
        try oracle.check(&s);
        try keep(&s, &f, &prompt, 100, &.{});
        try std.testing.expectEqual(old_len + 1, s.entries.items.len);
        try oracle.check(&s);
    }
}

test "snapshot index: RAM keep unwinds every allocation failure and failed family save" {
    var prompt: [128]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    var failures: usize = 0;
    var succeeded = false;
    for (0..16) |fail_at| {
        var f: Fake = .{ .gpa = a };
        var s = pc.Store.init(a, f.snapshots(), .{}, 1 << 20);
        defer s.deinit();
        try full(&s, &f, &prompt);
        const old_len = s.entries.items.len;
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_at, .resize_fail_index = 0 });
        s.gpa = failing.allocator();
        f.gpa = failing.allocator();
        f.at = 100;
        const kept = s.keep(&prompt, 100, null, &.{}, &.{.{ 2, 7 }});
        s.gpa = a;
        f.gpa = a;
        try oracle.check(&s);
        try std.testing.expectEqual(s.entries.items.len, f.live);
        if (failing.has_induced_failure) {
            failures += 1;
            try std.testing.expect(!kept);
            try std.testing.expectEqual(old_len, s.entries.items.len);
        } else {
            succeeded = true;
            try std.testing.expect(kept);
            try std.testing.expectEqual(old_len + 1, s.entries.items.len);
            f.at = 110;
            f.fail_save = true;
            try std.testing.expect(!s.keep(&prompt, 110, null, &.{}, &.{}));
            try std.testing.expectEqual(old_len + 1, s.entries.items.len);
            try oracle.check(&s);
            break;
        }
    }
    try std.testing.expect(succeeded); // spans, Entry, the one prompt copy, the save, then both publication reserves
    try std.testing.expectEqual(@as(usize, 6), failures);
}

/// A conversation tree: each prompt a kept one's prefix and new tokens, so states share long prefixes and fork.
const Talks = struct {
    all: std.ArrayList([]u32) = .empty,

    fn deinit(t: *Talks) void {
        for (t.all.items) |x| a.free(x);
        t.all.deinit(a);
    }

    fn next(t: *Talks, r: std.Random) ![]const u32 {
        const from: []const u32 = if (t.all.items.len == 0 or r.uintLessThan(u8, 8) == 0) &.{} else t.pick(r);
        const out = try a.alloc(u32, from.len + 8 + r.uintLessThan(usize, 120));
        @memcpy(out[0..from.len], from);
        for (out[from.len..]) |*token| token.* = r.uintLessThan(u32, 3); // three tokens: forks often agree by chance
        try t.all.append(a, out);
        return out;
    }

    /// A kept prompt, whole or cut short by up to 39 tokens (a regenerate or an edit).
    fn pick(t: *const Talks, r: std.Random) []const u32 {
        const x = t.all.items[r.uintLessThan(usize, t.all.items.len)];
        return x[0 .. x.len - r.uintLessThan(usize, @min(x.len, 40))];
    }
};

/// The four kinds of family the store serves: any row or a chunk grid, with and without a lookahead token.
const kinds = [_]pc.Rules{
    .{ .lookahead = 1, .min_prompt = 0, .min_gap = 4 },
    .{ .lookahead = 0, .min_prompt = 0, .min_gap = 16 },
    .{ .lookahead = 1, .min_prompt = 0, .min_gap = 4, .planned = true, .grid = 8 },
    .{ .lookahead = 0, .min_prompt = 0, .min_gap = 4, .planned = true },
};

/// A request's chunk starts: every fifth row for the planned family without a grid, else none.
fn startsFor(arena: std.mem.Allocator, s: *const pc.Store, len: usize) ![]const u32 {
    if (!s.rules.planned or s.rules.grid > 0) return &.{};
    const out = try arena.alloc(u32, len / 5 + 1);
    for (out, 0..) |*x, i| x.* = @intCast(i * 5);
    return out;
}

fn spansFor(r: std.Random) []const modes.Span {
    return if (r.uintLessThan(u8, 4) == 0) spanSet(r.int(u32)) else &.{};
}

/// Bytes the store may hold past which giveBack frees kept states.
const Left = struct {
    store: *const pc.Store,
    most: u64,
    pub fn fits(l: Left) bool {
        return l.store.held <= l.most;
    }
};

/// Random lookups against the scan and the index's ownership after one store operation.
fn agree(s: *pc.Store, f: *const Fake, r: std.Random, talks: *const Talks, arena: std.mem.Allocator) !void {
    try oracle.check(s);
    try std.testing.expectEqual(s.entries.items.len, f.live);
    for (0..4) |_| {
        const q = talks.pick(r);
        const starts = try startsFor(arena, s, q.len);
        const spans = spansFor(r);
        try std.testing.expect(oracle.find(s, q, starts, spans) == s.find(q, starts, spans));
    }
}

/// One store through forked conversations: passes, held lookups, keeps, drops, forgets, give-backs, evictions.
fn converse(seed: u64, hits: *u64, evicted: *u64) !void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var arena_state: std.heap.ArenaAllocator = .init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var f: Fake = .{ .gpa = a };
    const family = if (seed % 2 == 0) f.snapshots() else f.pooled(); // the pooled one keeps spare storage
    var s = pc.Store.init(a, family, kinds[seed % 4], if (seed % 5 < 2) 60_000 else 1 << 40);
    s.most = ([_]usize{ 8, 24, 200 })[seed % 3];
    defer s.deinit();
    var talks: Talks = .{};
    defer talks.deinit();
    var held: std.ArrayList(*pc.Entry) = .empty;
    defer held.deinit(a);
    for (0..300) |_| {
        _ = arena_state.reset(.retain_capacity);
        const op = r.uintLessThan(u8, 12);
        if (op < 6 or s.entries.items.len == 0) { // a request: a pass, or a lookup the caller holds and reports later
            const prompt = try talks.next(r);
            const starts = try startsFor(arena, &s, prompt.len);
            const spans = spansFor(r);
            const history = r.uintLessThan(u32, @intCast(prompt.len));
            const rewind = r.uintLessThan(u32, @intCast(prompt.len));
            const want = oracle.find(&s, prompt, starts, spans);
            if (op < 4) {
                f.fail_restore = r.uintLessThan(u8, 8) == 0;
                const want_at = if (want) |w| w.at else 0; // read first: a resumed state may go to make room
                const plan = try s.beginRewind(arena, prompt, history, rewind, &.{}, starts, null, spans);
                try std.testing.expectEqual(if (f.fail_restore) 0 else want_at, plan.from);
                hits.* += @intFromBool(plan.from > 0);
                f.fail_restore = false;
                try std.testing.expectEqual(@import("prompt_cache_test.zig").fresh(prompt), f.pass(&s, prompt, plan));
            } else {
                const l = try s.lookupRewind(arena, prompt, @intCast(prompt.len), history, rewind, &.{}, starts, spans);
                try std.testing.expect(l.entry == want);
                if (l.entry) |e| try held.append(a, e);
            }
        } else switch (op) {
            6 => if (held.items.len > 0) { // a held lookup resumed, failed or released
                const e = held.swapRemove(r.uintLessThan(usize, held.items.len));
                if (r.boolean()) s.resumed(e, talks.pick(r), r.uintLessThan(u8, 4) != 0) else s.release(e);
            },
            7 => { // a keep at any row, some born as a shorter request (a reply's end)
                const prompt = talks.pick(r);
                if (prompt.len > 1 + s.rules.lookahead) {
                    const at: u32 = 1 + r.uintLessThan(u32, @intCast(prompt.len - 1 - s.rules.lookahead));
                    f.at, f.sum = .{ at, @import("prompt_cache_test.zig").fresh(prompt[0..at]) };
                    _ = s.keepBorn(prompt, if (r.boolean()) @intCast(prompt.len) else at + 1, at, null, &.{}, spansFor(r));
                }
            },
            8 => { // the family drops a state
                s.expired(s.entries.items[r.uintLessThan(usize, s.entries.items.len)].saved);
                if (r.boolean()) s.collectExpired();
            },
            9 => { // a peer's failed state forgotten
                const e = s.entries.items[r.uintLessThan(usize, s.entries.items.len)];
                s.forget(try arena.dupe(u32, e.tokens), e.at, try arena.dupe(modes.Span, e.decode_spans));
            },
            10 => { // storage given back, sparing a held state
                const spare: ?pc.Saved = if (held.items.len > 0) held.items[0].saved else null;
                _ = s.giveBack(spare, talks.pick(r), Left{ .store = &s, .most = s.held / 2 });
            },
            else => { // a weight change clears the store, else the oldest unborrowed state goes
                if (r.uintLessThan(u8, 16) == 0) s.clear();
                for (s.entries.items, 0..) |e, i| if (e.borrowers == 0) {
                    s.remove(i);
                    break;
                };
            },
        }
        try agree(&s, &f, r, &talks, arena);
    }
    for (held.items) |e| s.release(e);
    try agree(&s, &f, r, &talks, arena);
    evicted.* += s.counts.evicted;
}

test "snapshot index: forked conversations on full stores resume and evict as the whole-store scan did" {
    var hits: u64 = 0;
    var evicted: u64 = 0;
    for (0..24) |seed| try converse(seed, &hits, &evicted);
    try std.testing.expect(hits > 500 and evicted > 1000); // resumes and evictions were exercised
}

test {
    _ = @import("prompt_cache_index_persistence_test.zig");
}
