//! A set of free page indices, lowest first in O(log64 n): a bit a page, then a bit a non-empty word a level up.

const std = @import("std");

const max_levels = 6; // 64^6 pages: more than any pool

pub const FreeSet = struct {
    /// levels[0] holds a bit a page, levels[k] a bit a non-empty word of levels[k - 1]
    levels: [max_levels][]u64 = @splat(&.{}),
    depth: u8 = 0,
    n: u32 = 0,
    free: u32 = 0,

    /// All of 0 .. n - 1 free.
    pub fn init(gpa: std.mem.Allocator, n: u32) !FreeSet {
        var s: FreeSet = .{ .n = n };
        errdefer s.deinit(gpa);
        var bits: usize = @max(n, 1);
        while (true) {
            const words = (bits + 63) / 64;
            s.levels[s.depth] = try gpa.alloc(u64, words);
            @memset(s.levels[s.depth], 0);
            s.depth += 1;
            if (words == 1) break;
            bits = words;
        }
        for (0..n) |i| s.setBit(@intCast(i));
        s.free = n;
        return s;
    }

    pub fn deinit(s: *FreeSet, gpa: std.mem.Allocator) void {
        for (s.levels[0..s.depth]) |l| gpa.free(l);
        s.* = .{};
    }

    pub fn count(s: *const FreeSet) u32 {
        return s.free;
    }

    pub fn contains(s: *const FreeSet, i: u32) bool {
        return s.levels[0][i / 64] & (@as(u64, 1) << @intCast(i % 64)) != 0;
    }

    /// The lowest free index, taken; null when none is free.
    pub fn takeLowest(s: *FreeSet) ?u32 {
        if (s.free == 0) return null;
        var idx: usize = 0;
        var lv: usize = s.depth;
        while (lv > 0) {
            lv -= 1;
            const w = s.levels[lv][idx];
            idx = idx * 64 + @ctz(w);
        }
        const i: u32 = @intCast(idx);
        s.clearBit(i);
        s.free -= 1;
        return i;
    }

    /// Takes a given free index (error when it is not free).
    pub fn take(s: *FreeSet, i: u32) error{NotFree}!void {
        if (i >= s.n or !s.contains(i)) return error.NotFree;
        s.clearBit(i);
        s.free -= 1;
    }

    /// Gives an index back (error when it is already free: a double free).
    pub fn give(s: *FreeSet, i: u32) error{DoubleFree}!void {
        if (i >= s.n or s.contains(i)) return error.DoubleFree;
        s.setBit(i);
        s.free += 1;
    }

    fn setBit(s: *FreeSet, i: u32) void {
        var idx: usize = i;
        for (s.levels[0..s.depth]) |l| {
            const was = l[idx / 64];
            l[idx / 64] = was | (@as(u64, 1) << @intCast(idx % 64));
            if (was != 0) return; // the word was non-empty: the levels above already say so
            idx /= 64;
        }
    }

    fn clearBit(s: *FreeSet, i: u32) void {
        var idx: usize = i;
        for (s.levels[0..s.depth]) |l| {
            l[idx / 64] &= ~(@as(u64, 1) << @intCast(idx % 64));
            if (l[idx / 64] != 0) return; // the word still has a free bit: the levels above stay set
            idx /= 64;
        }
    }
};

test "lowest first, give back, double free refused, across levels" {
    const gpa = std.testing.allocator;
    var s = try FreeSet.init(gpa, 70_000); // three levels
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(u8, 3), s.depth);
    for (0..5000) |i| try std.testing.expectEqual(@as(?u32, @intCast(i)), s.takeLowest());
    try s.give(4097);
    try s.give(17);
    try std.testing.expectError(error.DoubleFree, s.give(17));
    try std.testing.expectEqual(@as(?u32, 17), s.takeLowest());
    try std.testing.expectEqual(@as(?u32, 4097), s.takeLowest());
    try std.testing.expectEqual(@as(?u32, 5000), s.takeLowest());
    try s.take(69_999);
    try std.testing.expectError(error.NotFree, s.take(69_999));
    try std.testing.expectEqual(@as(u32, 70_000 - 5001 - 1), s.count());
    while (s.takeLowest()) |_| {}
    try std.testing.expectEqual(@as(u32, 0), s.count());
    try s.give(69_999);
    try std.testing.expectEqual(@as(?u32, 69_999), s.takeLowest());
}
