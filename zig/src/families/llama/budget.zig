//! What a Llama load on Metal may hold: weights, scratch and every stream's caches, below the panic line.
const std = @import("std");
const cfg = @import("config.zig");
const st = @import("state.zig");

/// The default share of RAM: past about 80% wired, macOS stalls until its watchdog restarts the Mac.
pub const default_share = 0.75;
/// Kept free inside the limit for the command buffers, the keepalive and the host.
pub const margin: u64 = 2 << 30;

/// Bytes for caches: `ram` times the share (or LLAMA_LOAD_LIMIT_GB's GB, at most the GPU's working set), less the rest.
pub fn caches(ram: u64, working_set: u64, text: ?[]const u8, resident: u64) error{BadLoadLimit}!u64 {
    var limit: u64 = @intFromFloat(@as(f64, @floatFromInt(ram)) * default_share);
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len > 0) {
        const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
        if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
        limit = @min(@as(u64, @intFromFloat(gb * 1e9)), @min(ram, working_set));
    }
    return if (limit > resident + margin) limit - resident - margin else 0;
}

/// One budget for every cache a load holds: each stream's, the drafter's slots (taken or kept) and their growth copies.
pub const Ledger = struct {
    limit: u64,
    held: u64 = 0,

    pub fn fits(l: *const Ledger, bytes: u64) bool {
        return l.held + bytes <= l.limit;
    }

    pub fn take(l: *Ledger, bytes: u64) void {
        std.debug.assert(l.fits(bytes));
        l.held += bytes;
    }

    pub fn give(l: *Ledger, bytes: u64) void {
        l.held -= bytes;
    }

    pub fn left(l: *const Ledger) u64 {
        return l.limit - l.held;
    }
};

/// A drafter's caches beside a stream's: `streams` of them, each the window and a decode window of rows more.
pub const Drafted = struct { c: *const cfg.Config, streams: u32 };

/// The longest window one stream's caches fit in `bytes`, at most `most`: a stream holds its window and one slot more.
pub fn longest(c: *const cfg.Config, bytes: u64, most: usize) usize {
    return longestWith(c, null, bytes, most);
}

/// As `longest`, with the drafter's caches for every stream taken from the same bytes.
pub fn longestWith(c: *const cfg.Config, drafted: ?Drafted, bytes: u64, most: usize) usize {
    if (needed(c, drafted, 1) > bytes) return 0;
    var lo: usize = 1;
    var hi: usize = most;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (needed(c, drafted, mid) <= bytes) lo = mid else hi = mid - 1;
    }
    return lo;
}

/// Bytes a window of `n` tokens takes: one stream's caches a slot longer, and the drafter's caches when it drafts.
pub fn needed(c: *const cfg.Config, drafted: ?Drafted, n: usize) u64 {
    var total: u64 = st.cacheBytes(c, n + 1);
    if (drafted) |d| total += @as(u64, @max(d.streams, 1)) * st.cacheBytes(d.c, n + st.window_rows);
    return total;
}

test "llama budget: two drafted requests that each fit alone are refused together, and pass in turn" {
    const gpa = std.testing.allocator;
    const gib: u64 = 1 << 30;
    const c = try cfg.parse(gpa, @embedFile("fixtures/config.json"));
    const dc = try cfg.parse(gpa, @embedFile("fixtures/drafter_config.json"));
    // a 100 GiB budget, a 262,144-token context and two streams: startup admits the window
    const drafted: Drafted = .{ .c = &dc, .streams = 2 };
    try std.testing.expect(longestWith(&c, drafted, 100 * gib, 262144) == 262144);
    // a 240,000-token request: its caches and its drafter slot's at the same capacity (backend.prefillFn's need)
    const capacity = 240_000 + 1;
    const one = st.cacheBytes(&c, capacity) + st.cacheBytes(&dc, capacity + st.window_rows);
    var l: Ledger = .{ .limit = 100 * gib };
    try std.testing.expect(l.fits(one));
    l.take(one);
    try std.testing.expect(!l.fits(one)); // the second waits: together they would pass the budget by 24.5 GiB
    try std.testing.expect(2 * one - l.limit > 24 * gib);
    // the first ends: its stream's caches are given back, its drafter slot kept (and still counted)
    l.give(st.cacheBytes(&c, capacity));
    try std.testing.expect(l.fits(one));
}

test "llama budget: a share of RAM less the weights, a set limit capped at the working set, bad text refused" {
    const gib: u64 = 1 << 30;
    try std.testing.expectEqual(@as(u64, 192 * gib - 62 * gib - margin), try caches(256 * gib, 240 * gib, null, 62 * gib));
    try std.testing.expectEqual(@as(u64, 0), try caches(256 * gib, 240 * gib, null, 191 * gib));
    try std.testing.expectEqual(@as(u64, 128 * gib - 62 * gib - margin), try caches(256 * gib, 240 * gib, "137.438953472", 62 * gib));
    try std.testing.expectError(error.BadLoadLimit, caches(256 * gib, 240 * gib, "lots", 62 * gib));
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/config.json"));
    try std.testing.expectEqual(@as(usize, 4095), longest(&c, st.cacheBytes(&c, 4096), 1 << 20)); // 4,096 + 1 needs a block more
    try std.testing.expectEqual(@as(usize, 4159), longest(&c, st.cacheBytes(&c, 4097), 1 << 20));
    try std.testing.expectEqual(@as(usize, 4096), longest(&c, st.cacheBytes(&c, 4097), 4096));
    try std.testing.expectEqual(@as(usize, 0), longest(&c, 1000, 1 << 20));
    // a drafter of the same geometry for two streams: its caches come out of the same bytes
    const two: Drafted = .{ .c = &c, .streams = 2 };
    try std.testing.expectEqual(st.cacheBytes(&c, 4081) + 2 * st.cacheBytes(&c, 4096), needed(&c, two, 4080));
    try std.testing.expectEqual(@as(usize, 4080), longestWith(&c, two, needed(&c, two, 4080), 1 << 20)); // 4,081 needs a block more
    try std.testing.expect(longestWith(&c, two, st.cacheBytes(&c, 4097), 1 << 20) < 1500);
    try std.testing.expectEqual(@as(usize, 0), longestWith(&c, two, 1000, 1 << 20));
}
