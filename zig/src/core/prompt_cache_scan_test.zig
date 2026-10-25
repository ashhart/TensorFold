//! Eviction choices with the extension links against the whole-store scans they replace, on random long-prefix stores.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const policy = @import("prompt_cache_policy.zig");
const Fake = @import("prompt_cache_fake.zig").Fake;
const gpa = std.testing.allocator;

/// The victim as the scans chose it: every entry's prefix compared again in each pass, and in each pairwise order.
fn victimScan(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32, born: u32) ?usize {
    var best: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e.borrowers > 0 or e.shared or e.branched or e == skip or rewound(s, e, prompt)) continue;
        const now = born > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
        if ((now or policy.extendedScan(s, e)) and (best == null or e.used < s.entries.items[best.?].used)) best = i;
    }
    if (best) |i| return i;
    var oldest: ?usize = null;
    for (s.entries.items, 0..) |e, i| if (e.borrowers == 0 and e != skip and (oldest == null or before(s, e, s.entries.items[oldest.?], prompt))) {
        oldest = i;
    };
    return oldest;
}

fn before(s: *const pc.Store, e: *const pc.Entry, o: *const pc.Entry, prompt: []const u32) bool {
    const e_rewind = rewound(s, e, prompt);
    const o_rewind = rewound(s, o, prompt);
    return if (e_rewind != o_rewind) o_rewind else e.used < o.used;
}

fn rewound(s: *const pc.Store, e: *const pc.Entry, prompt: []const u32) bool {
    return !e.expired and e.at == s.active_rewind_at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
}

/// A conversation tree: each prompt a stored one's prefix plus new tokens, so entries share long prefixes.
const Prompts = struct {
    all: std.ArrayList([]u32) = .empty,
    fn deinit(p: *Prompts) void {
        for (p.all.items) |x| gpa.free(x);
        p.all.deinit(gpa);
    }
    fn next(p: *Prompts, r: std.Random) ![]const u32 {
        const from: []const u32 = if (p.all.items.len == 0 or r.uintLessThan(u8, 8) == 0) &.{} else blk: {
            const x = p.all.items[r.uintLessThan(usize, p.all.items.len)];
            break :blk x[0 .. x.len - r.uintLessThan(usize, @min(x.len, 40))];
        };
        const out = try gpa.alloc(u32, from.len + 8 + r.uintLessThan(usize, 120));
        @memcpy(out[0..from.len], from);
        for (out[from.len..]) |*t| t.* = r.uintLessThan(u32, 3); // a small vocabulary: prompts often agree by chance
        try p.all.append(gpa, out);
        return out;
    }
};

fn agree(s: *pc.Store, r: std.Random, prompts: *Prompts) !void {
    for (0..@min(8, s.entries.items.len)) |_| {
        const e = s.entries.items[r.uintLessThan(usize, s.entries.items.len)];
        try std.testing.expectEqual(policy.extendedScan(s, e), policy.extended(s, e));
    }
    for (0..4) |_| {
        const prompt = prompts.all.items[r.uintLessThan(usize, prompts.all.items.len)];
        const skip: ?*pc.Entry = if (s.entries.items.len > 0 and r.boolean()) s.entries.items[r.uintLessThan(usize, s.entries.items.len)] else null;
        const born: u32 = r.uintLessThan(u32, @intCast(prompt.len + 2));
        const saved = s.active_rewind_at;
        if (s.entries.items.len > 0 and r.boolean()) s.active_rewind_at = s.entries.items[r.uintLessThan(usize, s.entries.items.len)].at;
        try std.testing.expectEqual(victimScan(s, skip, prompt, born), s.victimBut(skip, prompt, born));
        s.active_rewind_at = saved;
    }
}

fn run(seed: u64, most: usize) !u64 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0, .min_gap = 4 }, 1 << 40);
    s.most = most;
    defer s.deinit();
    var prompts: Prompts = .{};
    defer prompts.deinit();
    for (0..400) |_| {
        const prompt = try prompts.next(r);
        switch (r.uintLessThan(u8, 6)) {
            0, 1, 2 => { // a pass keeps a few states, some born as a shorter request (a reply's end)
                for (0..1 + r.uintLessThan(usize, 3)) |_| {
                    const at: u32 = 1 + r.uintLessThan(u32, @intCast(prompt.len - 1));
                    f.at = at;
                    _ = s.keepBorn(prompt, if (r.boolean()) @intCast(prompt.len) else at + 1, at, null, &.{}, &.{});
                }
            },
            3, 4 => { // a lookup, resumed or failed
                const history: u32 = r.uintLessThan(u32, @intCast(prompt.len));
                const l = try s.lookupRewind(arena.allocator(), prompt, @intCast(prompt.len), history, r.uintLessThan(u32, @intCast(prompt.len)), &.{}, &.{}, &.{});
                if (l.entry) |e| s.resumed(e, prompt, r.uintLessThan(u8, 5) != 0);
            },
            else => if (s.entries.items.len > 0) { // a state the family drops
                s.expired(s.entries.items[r.uintLessThan(usize, s.entries.items.len)].saved);
                if (r.boolean()) s.collectExpired();
            },
        }
        try agree(&s, r, &prompts);
    }
    return s.counts.evicted;
}

test "eviction with extension links picks every victim the whole-store scans picked" {
    var evicted: u64 = 0;
    for (0..24) |seed| evicted += try run(seed, if (seed % 4 == 3) 200 else 24);
    try std.testing.expect(evicted > 1000); // the stores were full: the choices were exercised
}

test "a state's tokens are its kept prompt's prefix, held once" {
    var f: Fake = .{ .gpa = gpa };
    var s = pc.Store.init(gpa, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };
    f.at = 3;
    try std.testing.expect(s.keep(&prompt, 3, null, &.{}, &.{}));
    const e = s.entries.items[0];
    try std.testing.expectEqualSlices(u32, prompt[0..4], e.tokens);
    try std.testing.expectEqualSlices(u32, &prompt, e.last);
    try std.testing.expect(e.tokens.ptr == e.last.ptr);
    s.resumed(e, &.{ 1, 2, 3, 4, 9 }, true); // a later prompt takes `last`'s place; the tokens keep their copy
    try std.testing.expectEqualSlices(u32, prompt[0..4], e.tokens);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 9 }, e.last);
}
