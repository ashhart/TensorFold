//! Byte trie over added-token contents for leftmost-longest matching, like tokenizers' Aho-Corasick split.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const MaskWork = struct { spans: usize = 0, matcher: usize = 0, masked: usize = 0 };

pub const Trie = struct {
    first: [256]u32 = @splat(0),
    first_byte: ?u8 = null,
    first_count: u16 = 0,
    edges: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    ends: std.ArrayList(?u32) = .empty,

    pub const Match = struct { start: usize, end: usize, token: u32 };

    pub fn insert(t: *Trie, a: Allocator, s: []const u8, token: u32) Allocator.Error!void {
        if (s.len == 0) return;
        if (t.ends.items.len == 0) try t.ends.append(a, null);
        var node: u32 = 0;
        for (s, 0..) |b, i| {
            const slot = if (i == 0) &t.first[b] else (try t.edges.getOrPutValue(a, @as(u64, node) << 8 | b, 0)).value_ptr;
            if (slot.* == 0) {
                try t.ends.append(a, null);
                slot.* = @intCast(t.ends.items.len - 1);
                if (i == 0) {
                    t.first_count += 1;
                    t.first_byte = b;
                }
            }
            node = slot.*;
        }
        if (t.ends.items[node] == null) t.ends.items[node] = token;
    }

    pub fn next(t: *const Trie, s: []const u8, from: usize) ?Match {
        return t.nextCounted(s, from, null);
    }

    fn matchAt(t: *const Trie, s: []const u8, i: usize, work: ?*MaskWork) ?Match {
        var node = t.first[s[i]];
        if (node == 0) return null;
        var best: ?Match = null;
        var j = i + 1;
        while (true) {
            if (t.ends.items[node]) |token| best = .{ .start = i, .end = j, .token = token };
            if (j == s.len) break;
            if (work) |w| w.matcher += 1;
            node = t.edges.get(@as(u64, node) << 8 | s[j]) orelse break;
            j += 1;
        }
        return best;
    }

    pub fn nextCounted(t: *const Trie, s: []const u8, from: usize, work: ?*MaskWork) ?Match {
        if (t.ends.items.len == 0) return null;
        if (work == null and t.first_count == 1) {
            const fb = t.first_byte.?;
            var at = from;
            while (std.mem.indexOfScalarPos(u8, s, at, fb)) |i| : (at = i + 1) {
                if (t.matchAt(s, i, null)) |b| return b;
            }
            return null;
        }
        for (from..s.len) |i| {
            if (work) |w| w.matcher += 1;
            if (t.matchAt(s, i, work)) |b| return b;
        }
        return null;
    }
};
