//! Long reasoning loops: a passage repeated verbatim, found from where each `gram`-token run of the reasoning ended.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Tokens a run is hashed over; the shortest period looked for (shorter ones are stream.zig's short cycles).
pub const gram = 32;
pub const min_period = 9;
/// The repeat that fires: a whole period, and at least this many tokens, each equal to the one a period before.
pub const min_repeat = 1024;

/// Each indexed run's end (an emitted() position) chained to the previous end with the same hash.
pub const Index = struct {
    last: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    prev: std.ArrayList(u32) = .empty, // prev[n]: the previous end with run n's hash (0: none)
    upto: usize = 0, // positions indexed so far: run ends in [gram, upto]

    pub fn deinit(x: *Index, gpa: Allocator) void {
        x.last.deinit(gpa);
        x.prev.deinit(gpa);
    }

    /// Reserve every possible new run before publishing tokens or changing the index's logical state.
    pub fn reserve(x: *Index, gpa: Allocator, upto: usize) !void {
        if (upto <= x.upto) return;
        try x.prev.ensureTotalCapacity(gpa, upto);
        const first = @max(x.upto, gram - 1);
        if (upto > first) try x.last.ensureUnusedCapacity(gpa, @intCast(upto - first));
    }

    /// Index every run end up to `tokens.len` (tokens: emitted(), committed only).
    pub fn add(x: *Index, gpa: Allocator, tokens: []const u32) !void {
        try x.reserve(gpa, tokens.len);
        x.addAssumeCapacity(tokens);
    }

    /// Commit runs after reserve succeeded, without an allocation that could fail after token publication.
    pub fn addAssumeCapacity(x: *Index, tokens: []const u32) void {
        while (x.upto < tokens.len) {
            x.upto += 1;
            x.prev.appendAssumeCapacity(0);
            if (x.upto < gram) continue;
            const got = x.last.getOrPutAssumeCapacity(hash(tokens, &.{}, x.upto));
            x.prev.items[x.upto - 1] = if (got.found_existing) got.value_ptr.* else 0;
            got.value_ptr.* = @intCast(x.upto);
        }
    }

    /// The period back to the run's latest end `min_period`+ back, if max(period, min_repeat) tokens repeat it.
    pub fn cycleAt(x: *const Index, head: []const u32, tail: []const u32, n: usize) ?u32 {
        if (n < gram + min_period) return null;
        var q: u32 = x.last.get(hash(head, tail, n)) orelse return null;
        while (q > 0 and q + min_period > n) q = x.prev.items[q - 1];
        if (q == 0) return null;
        const p = n - q;
        const need = @max(p, min_repeat);
        if (n < need + p) return null;
        for (1..need + 1) |k| if (at(head, tail, n - k) != at(head, tail, n - k - p)) return null;
        return @intCast(p);
    }
};

fn at(head: []const u32, tail: []const u32, i: usize) u32 {
    return if (i < head.len) head[i] else tail[i - head.len];
}

/// The run of `gram` tokens ending at position n.
fn hash(head: []const u32, tail: []const u32, n: usize) u64 {
    var h = std.hash.Wyhash.init(0x6c6f6f70);
    for (n - gram..n) |i| h.update(std.mem.asBytes(&at(head, tail, i)));
    return h.final();
}

/// Where the cycle fires in `prefix` unrelated tokens then a `period`-token passage repeated, `step` tokens a round.
fn fireAt(gpa: Allocator, prefix: usize, period: usize, total: usize, step: usize) !?usize {
    var prng = std.Random.DefaultPrng.init(period);
    const r = prng.random();
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..prefix) |_| try tokens.append(gpa, r.int(u32));
    var passage: std.ArrayList(u32) = .empty;
    defer passage.deinit(gpa);
    for (0..period) |_| try passage.append(gpa, r.int(u32));
    while (tokens.items.len < total) try tokens.append(gpa, passage.items[(tokens.items.len - prefix) % period]);
    var x: Index = .{};
    defer x.deinit(gpa);
    var done: usize = 0;
    while (done < total) {
        const w = @min(step, total - done);
        const head = tokens.items[0..done];
        for (0..w) |i| if (x.cycleAt(head, tokens.items[done..][0..w], done + i + 1)) |p| {
            try std.testing.expectEqual(@as(u32, @intCast(period)), p);
            return done + i + 1;
        };
        done += w;
        try x.add(gpa, tokens.items[0..done]);
    }
    return null;
}

test "a repeated passage fires once a whole period and 1,024 tokens repeat, at the same token in windows" {
    const gpa = std.testing.allocator;
    // 4,500-token passage: fires when its second copy completes
    try std.testing.expectEqual(@as(?usize, 3000 + 2 * 4500), try fireAt(gpa, 3000, 4500, 20000, 1));
    try std.testing.expectEqual(@as(?usize, 3000 + 2 * 4500), try fireAt(gpa, 3000, 4500, 20000, 6));
    // 100-token passage: 1,024 repeated tokens past its first copy
    try std.testing.expectEqual(@as(?usize, 500 + 100 + 1024), try fireAt(gpa, 500, 100, 5000, 1));
    try std.testing.expectEqual(@as(?usize, 500 + 100 + 1024), try fireAt(gpa, 500, 100, 5000, 6));
}

test "no long cycle in text that does not repeat, or that repeats less than the rule asks" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqual(@as(?usize, null), try fireAt(gpa, 20000, 1, 20000, 1));
    // a 3,000-token passage written twice but cut 1 token short of its second copy
    try std.testing.expectEqual(@as(?usize, null), try fireAt(gpa, 100, 3000, 100 + 2 * 3000 - 1, 6));
}
