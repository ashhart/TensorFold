//! Host pages a row gather copies from: read in once each, so a cold gather's faults are in flight together.
const std = @import("std");
const builtin = @import("builtin");

/// MADV_POPULATE_READ (Linux 5.14+): map the pages read now; Zig's table stops at COLD.
pub const populate_read: u32 = 22;

/// Pages the set tracks at most: 2 MiB of bits, a 64 GiB table at 4 KiB pages (a larger one gathers untracked).
pub const max_pages: usize = 1 << 24;

/// One mapped array a gather reads, and the bit its first page holds in the set.
pub const Region = struct { base: usize, end: usize, first: u64 };

/// Reads a gather's pages in before it copies them; a page asked for once stays seen, so warm gathers make no call.
pub const Warmer = struct {
    arena: std.mem.Allocator = undefined,
    /// The arrays the gather reads, in the order it warms them.
    regions: []const Region = &.{},
    /// One bit a page of every region together: set once the page was asked for.
    bits: []std.atomic.Value(usize) = &.{},
    /// A kernel without MADV_POPULATE_READ (before 5.14) takes MADV_WILLNEED from then on.
    populate: std.atomic.Value(bool) = std.atomic.Value(bool).init(true),
    /// The spans hold more than `max_pages` pages: nothing is tracked, and every gather asks for its pages.
    over: bool = false,

    /// Room for every page of `spans` (the arrays a gather reads, a slice each), none of them seen yet.
    pub fn init(a: std.mem.Allocator, spans: []const []const u8) !Warmer {
        return initWith(a, spans, max_pages);
    }

    /// `init` over a table of up to `cap` pages: one past it keeps no bits and asks for every page each gather.
    pub fn initWith(a: std.mem.Allocator, spans: []const []const u8, cap: usize) !Warmer {
        var w: Warmer = .{ .arena = a };
        errdefer w.deinit(a);
        var pages: u64 = 0;
        const regions = try a.alloc(Region, spans.len);
        for (spans, regions) |s, *r| {
            const base = std.mem.alignBackward(usize, @intFromPtr(s.ptr), std.heap.pageSize());
            const end = @intFromPtr(s.ptr) + s.len;
            r.* = .{ .base = base, .end = end, .first = pages };
            pages += @intCast(std.mem.alignForward(usize, end, std.heap.pageSize()) / std.heap.pageSize() - base / std.heap.pageSize());
        }
        w.regions = regions;
        // a table too large to track is still warmed: its rows are read through the sizes, not the set
        if (pages > cap) {
            w.over = true;
            return w;
        }
        const words = (pages + @bitSizeOf(usize) - 1) / @bitSizeOf(usize);
        w.bits = try a.alloc(std.atomic.Value(usize), @intCast(words));
        @memset(w.bits, std.atomic.Value(usize).init(0));
        return w;
    }

    pub fn deinit(w: *Warmer, a: std.mem.Allocator) void {
        if (w.regions.len > 0) a.free(w.regions);
        if (w.bits.len > 0) a.free(w.bits);
        w.* = .{};
    }

    /// `bytes`, part of region `i`, read in; a set too large to track asks for every page again.
    pub fn warm(w: *Warmer, i: usize, bytes: []const u8) void {
        if (bytes.len == 0 or builtin.os.tag != .linux) return;
        const r = w.regions[i];
        const at = @intFromPtr(bytes.ptr);
        if (at < r.base or at + bytes.len > r.end) return;
        const lo: usize = @intCast(r.first + (at - r.base) / std.heap.pageSize());
        const hi: usize = @intCast(r.first + (at - r.base + bytes.len + std.heap.pageSize() - 1) / std.heap.pageSize());
        if (w.over) {
            w.ask(r, lo, hi);
            return;
        }
        // the pages this range holds: one call each run of them not yet asked for
        var run: ?usize = null;
        var n = lo;
        while (n < hi) : (n += 1) {
            const fresh = !w.seen(n);
            if (!fresh) {
                if (run) |from| w.ask(r, from, n);
                run = null;
                continue;
            }
            if (run == null) run = n;
        }
        if (run) |from| w.ask(r, from, hi);
    }

    /// Whether page `n` of the set was asked for before (a read only: the pages of a warm gather cost no write).
    pub fn seen(w: *Warmer, n: usize) bool {
        if (w.over) return false;
        return (w.bits[n / @bitSizeOf(usize)].load(.monotonic) >> @intCast(n % @bitSizeOf(usize))) & 1 != 0;
    }

    /// Whether `bytes`, part of region `i`, holds a page no gather asked for yet.
    pub fn cold(w: *Warmer, i: usize, bytes: []const u8) bool {
        if (bytes.len == 0 or builtin.os.tag != .linux) return false;
        if (w.over) return true;
        const r = w.regions[i];
        const at = @intFromPtr(bytes.ptr);
        if (at < r.base or at + bytes.len > r.end) return false;
        const lo: usize = @intCast(r.first + (at - r.base) / std.heap.pageSize());
        const hi: usize = @intCast(r.first + (at - r.base + bytes.len + std.heap.pageSize() - 1) / std.heap.pageSize());
        for (lo..hi) |n| if (!w.seen(n)) return true;
        return false;
    }

    /// Pages [lo, hi) of `r`, as one call over their bytes.
    fn ask(w: *Warmer, r: Region, lo: usize, hi: usize) void {
        const linux = std.os.linux;
        const page = std.heap.pageSize();
        if (!w.over) for (lo..hi) |n| {
            const bit = @as(usize, 1) << @intCast(n % @bitSizeOf(usize));
            _ = w.bits[n / @bitSizeOf(usize)].fetchOr(bit, .acq_rel);
        };
        const at = r.base + (lo - @as(usize, @intCast(r.first))) * page;
        const len = (hi - lo) * page;
        if (w.populate.load(.acquire)) {
            if (linux.madvise(@ptrFromInt(at), len, populate_read) == 0) return;
            w.populate.store(false, .release);
        }
        _ = linux.madvise(@ptrFromInt(at), len, linux.MADV.WILLNEED);
    }
};

test "a region's pages are asked for once: a second warm makes no call and the bytes are unchanged" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const bytes = try a.alignedAlloc(u8, .fromByteUnits(std.heap.pageSize()), 4 * std.heap.pageSize());
    defer a.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i * 11 + 5);
    var w = try Warmer.init(a, &.{ bytes, bytes[0..0] });
    defer w.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), w.bits.len);
    const hot = bytes[1 .. 2 * std.heap.pageSize() - 7];
    w.warm(0, hot);
    for (w.bits) |word| try std.testing.expectEqual(@as(usize, 3), word.load(.acquire));
    w.warm(0, hot);
    w.warm(0, hot[3..]);
    for (w.bits) |word| try std.testing.expectEqual(@as(usize, 3), word.load(.acquire));
    w.warm(1, "not this region");
    w.warm(0, &.{});
    for (w.bits) |word| try std.testing.expectEqual(@as(usize, 3), word.load(.acquire));
    try std.testing.expectEqualSlices(u8, bytes[0..256], bytes[0..256]);
}

test "a range off its region's bytes is left alone" {
    const a = std.testing.allocator;
    var other: [64]u8 = @splat(7);
    var w = try Warmer.init(a, &.{other[0..16]});
    defer w.deinit(a);
    w.warm(0, other[0..64]);
    for (w.bits) |word| try std.testing.expectEqual(@as(usize, 0), word.load(.acquire));
}

test "a set past its page cap keeps no bits: every range is warm-gathering and its bytes survive a warm" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    const bytes = try a.alignedAlloc(u8, .fromByteUnits(std.heap.pageSize()), 4 * std.heap.pageSize());
    defer a.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i * 13 + 3);
    const want = try a.dupe(u8, bytes[0..64]);
    defer a.free(want);
    var w = try Warmer.initWith(a, &.{bytes}, 3);
    defer w.deinit(a);
    try std.testing.expect(w.over);
    try std.testing.expectEqual(@as(usize, 0), w.bits.len);
    try std.testing.expect(w.cold(0, bytes[0..16]));
    w.warm(0, bytes[0..16]);
    try std.testing.expect(w.cold(0, bytes[0..16]));
    try std.testing.expectEqualSlices(u8, want, bytes[0..64]);
}
