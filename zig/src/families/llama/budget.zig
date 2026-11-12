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

/// A stream's admission inside the limit: peak checked, kept states (and an unfitting restore) trimmed, then made.
pub fn admitStream(l: *Ledger, need: u64, h: anytype) !void {
    var peak = h.growth();
    if (!l.fits(need + peak)) {
        h.shed(need + peak);
        peak = h.growth();
    }
    if (!l.fits(need + peak)) return error.LaneMemory;
    const room = l.left() - need - peak;
    if (h.restoreBytes() > room) h.dropRestore();
    h.trim(room);
    l.take(need);
    errdefer l.give(need);
    try h.prepare();
}

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

/// A fake prompt store, drafter slot and allocation recorder: the most the ledger and the kept states held at once.
const Fake = struct {
    const gib: u64 = 1 << 30;
    l: *Ledger,
    kept: u64, // kept prompt states but the restore entry
    restore: u64, // the entry this stream resumes from (0: none)
    old: u64, // the drafter slot's cache before it grows
    grown: u64, // its cache after
    copy: u64, // the old state's copy while it moves
    fail: bool = false,
    peak: u64 = 0,

    fn note(f: *Fake) void {
        f.peak = @max(f.peak, f.l.held + f.kept + f.restore);
    }
    fn growth(f: *Fake) u64 {
        return f.grown + f.copy;
    }
    fn shed(_: *Fake, _: u64) void {}
    fn restoreBytes(f: *Fake) u64 {
        return f.restore;
    }
    fn dropRestore(f: *Fake) void {
        f.restore = 0;
    }
    fn trim(f: *Fake, bytes: u64) void {
        f.kept = @min(f.kept, bytes -| f.restore);
        f.note();
    }
    fn prepare(f: *Fake) !void {
        f.l.take(f.grown);
        f.note();
        if (f.fail) {
            f.l.give(f.grown);
            return error.OutOfMemory;
        }
        f.l.take(f.copy);
        f.note();
        f.l.give(f.copy);
        f.l.give(f.old);
    }
};

test "llama budget: admission trims kept prompt states before it allocates, so its peak stays inside the budget" {
    const g = Fake.gib;
    // 8 GiB, slot 1 and kept states 3 held; allocating the stream's 3 + 2.5 before the trim would reach 9.5 GiB
    var l: Ledger = .{ .limit = 8 * g };
    l.take(g);
    var f: Fake = .{ .l = &l, .kept = 2 * g, .restore = g, .old = g, .grown = 2 * g, .copy = g / 2 };
    try admitStream(&l, 3 * g, &f);
    try std.testing.expect(f.peak <= l.limit);
    try std.testing.expectEqual(g, f.restore); // the protected entry stayed
    try std.testing.expectEqual(5 * g, l.held); // the stream's 3 and the grown slot's 2
    // a restore entry that cannot stay beside the stream's peak is dropped, not kept past the budget
    var l2: Ledger = .{ .limit = 8 * g };
    l2.take(g);
    var f2: Fake = .{ .l = &l2, .kept = g, .restore = 2 * g, .old = g, .grown = 2 * g, .copy = g / 2 };
    try admitStream(&l2, 3 * g, &f2);
    try std.testing.expect(f2.peak <= l2.limit);
    try std.testing.expectEqual(@as(u64, 0), f2.restore);
}

test "llama budget: a failed drafter growth gives every byte back, and its peak stays inside the budget" {
    const g = Fake.gib;
    var l: Ledger = .{ .limit = 8 * g };
    l.take(g);
    var f: Fake = .{ .l = &l, .kept = 2 * g, .restore = g, .old = g, .grown = 2 * g, .copy = g / 2, .fail = true };
    try std.testing.expectError(error.OutOfMemory, admitStream(&l, 3 * g, &f));
    try std.testing.expect(f.peak <= l.limit);
    try std.testing.expectEqual(g, l.held); // the stream's caches and the growth given back; the old slot still held
    // a peak that cannot fit at all waits before anything is trimmed or made
    var f2: Fake = .{ .l = &l, .kept = 2 * g, .restore = g, .old = g, .grown = 6 * g, .copy = g };
    try std.testing.expectError(error.LaneMemory, admitStream(&l, 3 * g, &f2));
    try std.testing.expectEqual(@as(u64, 2 * g), f2.kept);
    try std.testing.expectEqual(@as(u64, 0), f2.peak);
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
