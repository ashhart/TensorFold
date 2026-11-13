//! A stream's state as pieces on whole pool pages: each piece inside one page, any pages, first-fit decreasing.
const std = @import("std");
const Allocator = std.mem.Allocator;
const groups = @import("kv_groups.zig");

/// Where a piece lies: its page among the stream's state pages (Lease.state), and its byte offset in that page.
pub const Place = struct { page: u32, offset: u64 };

/// Pieces placed on pages, in the caller's order; the same sizes and page always give the same map.
pub const Map = struct {
    at: []Place,
    sizes: []u64,
    page: u64,
    pages: u32,

    pub fn deinit(m: *Map, gpa: Allocator) void {
        gpa.free(m.at);
        gpa.free(m.sizes);
        m.* = undefined;
    }

    /// The pieces' count, pages and bytes, as the layout and the load report read them.
    pub fn summary(m: Map) groups.Pieces {
        var bytes: u64 = 0;
        for (m.sizes) |n| bytes += n;
        return .{ .n = @intCast(m.sizes.len), .pages = m.pages, .bytes = bytes };
    }

    /// Piece `i`'s device address: `ids` are the stream's state pages, `base` the pool's first page.
    pub fn addr(m: Map, i: usize, base: u64, ids: []const u32) u64 {
        return base + @as(u64, ids[m.at[i].page]) * m.page + m.at[i].offset;
    }
};

/// `sizes` packed first-fit decreasing on pages of `page` bytes, each piece starting at a multiple of `alignment`.
pub fn pack(gpa: Allocator, sizes: []const u64, page: u64, alignment: u64) error{ PieceOverPage, OutOfMemory }!Map {
    const order = try gpa.alloc(u32, sizes.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const By = struct {
        sizes: []const u64,
        al: u64,
        fn more(c: @This(), a: u32, b: u32) bool {
            const x = std.mem.alignForward(u64, c.sizes[a], c.al);
            const y = std.mem.alignForward(u64, c.sizes[b], c.al);
            return x > y or (x == y and a < b);
        }
    };
    std.sort.pdq(u32, order, By{ .sizes = sizes, .al = alignment }, By.more);
    const at = try gpa.alloc(Place, sizes.len);
    errdefer gpa.free(at);
    var used: std.ArrayList(u64) = .empty; // bytes taken on each page so far
    defer used.deinit(gpa);
    for (order) |i| {
        const n = std.mem.alignForward(u64, sizes[i], alignment);
        if (n > page) return error.PieceOverPage;
        const k = for (used.items, 0..) |u, k| {
            if (page - u >= n) break k;
        } else blk: {
            try used.append(gpa, 0);
            break :blk used.items.len - 1;
        };
        at[i] = .{ .page = @intCast(k), .offset = used.items[k] };
        used.items[k] += n;
    }
    const own = try gpa.dupe(u64, sizes);
    return .{ .at = at, .sizes = own, .page = page, .pages = @intCast(used.items.len) };
}

/// Every piece inside one page of the map's and no two overlapping; a wrong map is refused by name.
pub fn check(gpa: Allocator, m: Map) error{ PieceOverPage, PiecesOverlap, OutOfMemory }!void {
    const order = try gpa.alloc(u32, m.at.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| {
        o.* = @intCast(i);
        if (m.at[i].page >= m.pages or m.at[i].offset + m.sizes[i] > m.page) return error.PieceOverPage;
    }
    const By = struct {
        at: []const Place,
        fn less(c: @This(), a: u32, b: u32) bool {
            const x = c.at[a];
            const y = c.at[b];
            return x.page < y.page or (x.page == y.page and x.offset < y.offset);
        }
    };
    std.sort.pdq(u32, order, By{ .at = m.at }, By.less);
    for (order[0..order.len -| 1], order[@min(1, order.len)..]) |a, b| {
        const x = m.at[a];
        const y = m.at[b];
        if (x.page == y.page and x.offset + m.sizes[a] > y.offset) return error.PiecesOverlap;
    }
}

test {
    _ = @import("kv_pieces_test.zig");
}

/// The n-th highest of distinct `ids` (n <= ids.len), so the n highest are those at or above it; maxInt for n = 0.
pub fn nthHighest(ids: []const u32, n: usize) u32 {
    var cut: u32 = std.math.maxInt(u32);
    for (0..n) |_| {
        var best: u32 = 0;
        for (ids) |id| best = if (id < cut and id >= best) id else best;
        cut = best;
    }
    return cut;
}
