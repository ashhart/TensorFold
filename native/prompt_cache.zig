const std = @import("std");

pub const Boundary = struct {
    step: usize = 1,
    starts: ?[]const usize = null,
    pub fn allows(b: Boundary, count: usize) bool {
        if (b.starts) |starts| return std.mem.indexOfScalar(usize, starts, count) != null;
        return b.step != 0 and count % b.step == 0;
    }
};

pub fn commonPrefix(lhs: []const i32, rhs: []const i32) usize {
    var n: usize = 0;
    while (n < @min(lhs.len, rhs.len) and lhs[n] == rhs[n]) : (n += 1) {}
    return n;
}

const Checkpoints = struct { values: [2]usize = .{ 0, 0 }, count: usize = 0 };

pub fn checkpoints(history: usize, cached: usize, previous: ?[]const i32, prompt: []const i32) Checkpoints {
    var out = Checkpoints{};
    if (previous) |prior| {
        const stable = commonPrefix(prior, prompt);
        if (stable > cached and stable < history and stable >= history / 2 and stable < prompt.len) {
            out.values[out.count] = stable;
            out.count += 1;
        }
    }
    if (history > cached and history < prompt.len) {
        out.values[out.count] = history;
        out.count += 1;
    }
    return out;
}

/// Payloads own their arrays and provide clone, deinit and nbytes.
pub fn Store(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Entry = struct {
            tokens: []i32,
            cache: T,
            last_prompt: []i32,
            nbytes: u64,
            pinned: bool,
            fn deinit(e: *Entry, a: std.mem.Allocator) void {
                a.free(e.tokens);
                a.free(e.last_prompt);
                e.cache.deinit();
            }
        };
        pub const Hit = struct {
            count: usize,
            cache: T,
            last_prompt: []i32,
            pub fn deinit(hit: *Hit, a: std.mem.Allocator) void {
                hit.cache.deinit();
                a.free(hit.last_prompt);
            }
        };
        a: std.mem.Allocator,
        slots: usize,
        pinned_slots: usize = 3,
        budget_bytes: ?u64,
        admit_oversize: bool = false,
        entries: std.ArrayList(Entry) = .empty,
        hits: u64 = 0,
        misses: u64 = 0,
        evictions: u64 = 0,

        pub fn init(a: std.mem.Allocator, slots: usize, budget: ?u64) !Self {
            if (slots == 0 or budget == 0) return error.InvalidPromptCacheBudget;
            return .{ .a = a, .slots = slots, .budget_bytes = budget };
        }
        pub fn deinit(s: *Self) void {
            for (s.entries.items) |*entry| entry.deinit(s.a);
            s.entries.deinit(s.a);
        }
        pub fn nbytes(s: *const Self) u64 {
            var total: u64 = 0;
            for (s.entries.items) |entry| total +|= entry.nbytes;
            return total;
        }
        pub fn best(s: *const Self, prompt: []const i32, boundary: Boundary) ?usize {
            var result: ?usize = null;
            var length: usize = 0;
            for (s.entries.items, 0..) |entry, i| {
                const n = entry.tokens.len;
                if (n <= length or n >= prompt.len or !boundary.allows(n)) continue;
                if (std.mem.eql(i32, entry.tokens, prompt[0..n])) {
                    result = i;
                    length = n;
                }
            }
            return result;
        }
        pub fn longest(s: *const Self, prompt: []const i32, boundary: Boundary) usize {
            return if (s.best(prompt, boundary)) |i| s.entries.items[i].tokens.len else 0;
        }
        pub fn match(s: *Self, prompt: []const i32, boundary: Boundary, take: bool) !?Hit {
            const index = s.best(prompt, boundary) orelse {
                s.misses +|= 1;
                return null;
            };
            const original = &s.entries.items[index];
            const previous = try s.a.dupe(i32, original.last_prompt);
            errdefer s.a.free(previous);
            const count = original.tokens.len;
            if (take) {
                const entry = s.entries.orderedRemove(index);
                s.a.free(entry.tokens);
                s.a.free(entry.last_prompt);
                s.hits +|= 1;
                return .{ .count = count, .cache = entry.cache, .last_prompt = previous };
            }
            const next_prompt = try s.a.dupe(i32, prompt);
            errdefer s.a.free(next_prompt);
            const copy = try original.cache.clone();
            var entry = s.entries.orderedRemove(index);
            s.a.free(entry.last_prompt);
            entry.last_prompt = next_prompt;
            s.entries.insertAssumeCapacity(0, entry);
            s.hits +|= 1;
            return .{ .count = count, .cache = copy, .last_prompt = previous };
        }
        /// Takes ownership even when rejected or an allocation fails.
        pub fn insertOwned(s: *Self, tokens: []const i32, payload: T, last_prompt: []const i32, pinned: bool) !void {
            var cache = payload;
            var adopted = false;
            defer if (!adopted) cache.deinit();
            const size = cache.nbytes();
            const oversize = if (s.budget_bytes) |budget| size > budget else false;
            if (tokens.len == 0 or (oversize and !s.admit_oversize)) return;
            const key = try s.a.dupe(i32, tokens);
            errdefer s.a.free(key);
            const previous = try s.a.dupe(i32, last_prompt);
            errdefer s.a.free(previous);
            try s.entries.ensureUnusedCapacity(s.a, 1);
            var keep_pinned = pinned;
            for (s.entries.items, 0..) |entry, i| if (std.mem.eql(i32, entry.tokens, tokens)) {
                var replaced = s.entries.orderedRemove(i);
                keep_pinned = keep_pinned or replaced.pinned;
                replaced.deinit(s.a);
                break;
            };
            s.entries.insertAssumeCapacity(0, .{ .tokens = key, .cache = cache, .last_prompt = previous, .nbytes = size, .pinned = keep_pinned });
            adopted = true;
            var pinned_count: usize = 0;
            for (s.entries.items) |*entry| if (entry.pinned) {
                pinned_count += 1;
                if (pinned_count > s.pinned_slots) entry.pinned = false;
            };
            var limit = s.budget_bytes;
            if (oversize) {
                limit = size;
                for (s.entries.items[1..]) |entry| if (entry.pinned) {
                    limit.? +|= entry.nbytes;
                };
            }
            while (true) {
                var ordinary: usize = 0;
                var oldest: ?usize = null;
                for (s.entries.items, 0..) |entry, i| if (!entry.pinned) {
                    ordinary += 1;
                    if (i > 0) oldest = i;
                };
                const over_budget = if (limit) |budget| s.entries.items.len > 1 and s.nbytes() > budget else false;
                if (ordinary <= s.slots and !over_budget) break;
                const remove = oldest orelse if (over_budget) s.entries.items.len - 1 else break;
                var gone = s.entries.orderedRemove(remove);
                gone.deinit(s.a);
                s.evictions +|= 1;
            }
        }
        pub fn evictOne(s: *Self, keep: ?[]const i32) bool {
            var oldest: ?usize = null;
            var ordinary: ?usize = null;
            for (s.entries.items, 0..) |entry, i| {
                if (keep) |tokens| if (std.mem.eql(i32, tokens, entry.tokens)) continue;
                oldest = i;
                if (!entry.pinned) ordinary = i;
            }
            const index = ordinary orelse oldest orelse return false;
            var gone = s.entries.orderedRemove(index);
            gone.deinit(s.a);
            s.evictions +|= 1;
            return true;
        }
    };
}

const FixturePayload = struct {
    id: u64,
    size: u64,
    pub fn clone(p: *const FixturePayload) !FixturePayload {
        return p.*;
    }
    pub fn deinit(_: *FixturePayload) void {}
    pub fn nbytes(p: *const FixturePayload) u64 {
        return p.size;
    }
};

pub fn check(io: std.Io, path: []const u8) !void {
    const a = std.heap.page_allocator;
    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 * 1024 * 1024));
    defer a.free(source);
    const Entry = struct { tokens: []const i32, payload: FixturePayload, previous: []const i32, pinned: bool };
    const Hit = struct { count: usize, payload: FixturePayload, previous: []const i32 };
    const Operation = struct {
        kind: enum { insert, match, longest, evict },
        prompt: []const i32,
        boundary: Boundary,
        payload: ?FixturePayload = null,
        previous: []const i32 = &.{},
        pinned: bool = false,
        oversize: bool = false,
        take: bool = false,
        hit: ?Hit = null,
        longest: usize = 0,
        keep: ?[]const i32 = null,
        evicted: bool = false,
        entries: []const Entry,
        nbytes: u64,
        hits: u64,
        misses: u64,
        evictions: u64,
    };
    const Fixture = struct {
        stores: []const struct { slots: usize, budget: ?u64, pinned_slots: usize, operations: []const Operation },
        checkpoints: []const struct { prompt: []const i32, previous: ?[]const i32, history: usize, cached: usize, expected: []const usize },
    };
    const parsed = try std.json.parseFromSlice(Fixture, a, source, .{});
    defer parsed.deinit();
    var count: usize = 0;
    for (parsed.value.stores) |case| {
        var store = try Store(FixturePayload).init(a, case.slots, case.budget);
        defer store.deinit();
        store.pinned_slots = case.pinned_slots;
        for (case.operations) |op| {
            switch (op.kind) {
                .insert => {
                    store.admit_oversize = op.oversize;
                    try store.insertOwned(op.prompt, op.payload.?, op.previous, op.pinned);
                },
                .match => {
                    var hit = try store.match(op.prompt, op.boundary, op.take);
                    defer if (hit) |*value| value.deinit(a);
                    try std.testing.expectEqual(op.hit != null, hit != null);
                    if (hit) |value| {
                        try std.testing.expectEqual(op.hit.?.count, value.count);
                        try std.testing.expectEqualDeep(op.hit.?.payload, value.cache);
                        try std.testing.expectEqualSlices(i32, op.hit.?.previous, value.last_prompt);
                    }
                },
                .longest => try std.testing.expectEqual(op.longest, store.longest(op.prompt, op.boundary)),
                .evict => try std.testing.expectEqual(op.evicted, store.evictOne(op.keep)),
            }
            try std.testing.expectEqual(op.nbytes, store.nbytes());
            try std.testing.expectEqual(op.hits, store.hits);
            try std.testing.expectEqual(op.misses, store.misses);
            try std.testing.expectEqual(op.evictions, store.evictions);
            try std.testing.expectEqual(op.entries.len, store.entries.items.len);
            for (op.entries, store.entries.items) |expected, actual| {
                try std.testing.expectEqualSlices(i32, expected.tokens, actual.tokens);
                try std.testing.expectEqualSlices(i32, expected.previous, actual.last_prompt);
                try std.testing.expectEqualDeep(expected.payload, actual.cache);
                try std.testing.expectEqual(expected.pinned, actual.pinned);
            }
            count += 1;
        }
    }
    for (parsed.value.checkpoints) |case| {
        const actual = checkpoints(case.history, case.cached, case.previous, case.prompt);
        try std.testing.expectEqualSlices(usize, case.expected, actual.values[0..actual.count]);
    }
    std.debug.print("PASS: {d} upstream prompt-cache operations and {d} checkpoint selections\n", .{ count, parsed.value.checkpoints.len });
}

test "prefixes are strict and checkpoints exclude cached or terminal positions" {
    try std.testing.expectEqual(@as(usize, 2), commonPrefix(&.{ 1, 2, 3 }, &.{ 1, 2, 4 }));
    const result = checkpoints(4, 0, &.{ 1, 2, 3, 9 }, &.{ 1, 2, 3, 4, 5 });
    try std.testing.expectEqualSlices(usize, &.{ 3, 4 }, result.values[0..result.count]);
    try std.testing.expectEqual(@as(usize, 0), checkpoints(4, 4, null, &.{ 1, 2, 3, 4 }).count);
}

const OwnedPayload = struct {
    a: std.mem.Allocator,
    value: *u64,
    fn init(a: std.mem.Allocator) !OwnedPayload {
        const value = try a.create(u64);
        value.* = 19;
        return .{ .a = a, .value = value };
    }
    pub fn clone(p: *const OwnedPayload) !OwnedPayload {
        const copy = try init(p.a);
        copy.value.* = p.value.*;
        return copy;
    }
    pub fn deinit(p: *OwnedPayload) void {
        p.a.destroy(p.value);
    }
    pub fn nbytes(_: *const OwnedPayload) u64 {
        return @sizeOf(u64);
    }
};

fn allocationFailures(a: std.mem.Allocator) !void {
    var store = try Store(OwnedPayload).init(a, 1, 16);
    defer store.deinit();
    try store.insertOwned(&.{}, try OwnedPayload.init(a), &.{}, false);
    try store.insertOwned(&.{1}, try OwnedPayload.init(a), &.{ 1, 2 }, true);
    try store.insertOwned(&.{1}, try OwnedPayload.init(a), &.{ 1, 3 }, false);
    var hit = (try store.match(&.{ 1, 4 }, .{}, false)).?;
    defer hit.deinit(a);
    hit.cache.value.* = 42;
    try std.testing.expectEqual(@as(u64, 19), store.entries.items[0].cache.value.*);
    var taken = (try store.match(&.{ 1, 5 }, .{}, true)).?;
    defer taken.deinit(a);
    try std.testing.expectEqual(@as(u64, 19), taken.cache.value.*);
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
    try store.insertOwned(&.{2}, try OwnedPayload.init(a), &.{ 2, 3 }, false);
    try store.insertOwned(&.{3}, try OwnedPayload.init(a), &.{ 3, 4 }, false);
    try std.testing.expectEqual(@as(u64, 1), store.evictions);
    try std.testing.expect(store.evictOne(null));
    store.budget_bytes = 1;
    try store.insertOwned(&.{4}, try OwnedPayload.init(a), &.{ 4, 5 }, false);
    try std.testing.expectEqual(@as(usize, 0), store.entries.items.len);
}

test "owned prefixes survive cloning, replacement, transfer, eviction and allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailures, .{});
}
