//! TF_GLM_KV_SPLIT's owner-only latent rows: a rank holds its own class's latent pages and one trash page, no others.
const std = @import("std");

/// A rank's latent region: its class's pages, page id at id / classes, then the trash page every other id maps to.
pub const Region = struct {
    rank: u32,
    classes: u32,
    cap: usize, // pages a class holds (Pool.classCap); the trash page's index

    /// Whether this rank holds page `id`'s latent rows (its class is this rank's).
    pub fn holds(r: Region, id: u32) bool {
        return id % r.classes == r.rank;
    }

    /// The latent table's entry for page `id`: its place in this rank's region, else the trash page.
    pub fn entry(r: Region, id: u32) u32 {
        return if (r.holds(id)) id / r.classes else @intCast(r.cap);
    }

    /// Pages the region holds: the class's, then the trash page.
    pub fn pages(r: Region) usize {
        return r.cap + 1;
    }
};

/// Bytes a page costs one rank: every pooled key, and a share of the latent rows (those of one class in `classes`).
pub fn pageCost(pooled: u64, latent: u64, classes: u32) u64 {
    return pooled + (std.math.divCeil(u64, latent, classes) catch unreachable);
}

/// A layout's bytes a token: the page's cost to one rank over the rows a page holds, rounded up.
pub fn tokenBytes(pooled: u64, latent: u64, classes: u32, page_rows: u64) u64 {
    return std.math.divCeil(u64, pageCost(pooled, latent, classes), page_rows) catch unreachable;
}

/// The arena: every page's pooled keys, then this rank's region (its class's latent pages and the trash page).
pub fn arenaBytes(total: usize, pooled: u64, latent: u64, classes: u32) u64 {
    return total * pooled + (total / classes + 1) * latent;
}

/// Pages a plan of `pages` at `page` bytes each makes: a multiple of `classes` whose arena and trash page fit in them.
pub fn funded(pages: usize, page: u64, pooled: u64, latent: u64, classes: u32) usize {
    const budget = pages * page;
    if (budget < latent) return 0;
    const unit = classes * pooled + latent; // one page of each class on this rank
    return @min(pages - pages % classes, (budget - latent) / unit * classes);
}

const t = std.testing;

test "GLM KV split: owner-only rows cost a rank its pooled keys and a 1/W share of the latent rows" {
    // GLM-5.3-Flash: 12 caches of 512 latent rows (1,024 B) and 128 pooled keys (256 B) a page, 13,056 B a token
    const latent: u64 = 12 * 512 * 1024;
    const pooled: u64 = 12 * 128 * 256;
    try t.expectEqual(@as(u64, 13_056), tokenBytes(pooled, latent, 1, 512)); // one class: the whole page
    try t.expectEqual(@as(u64, 6_912), tokenBytes(pooled, latent, 2, 512)); // TP=2: 768 + 6,144, 1.89x the tokens
    try t.expectEqual(@as(u64, 4_864), tokenBytes(pooled, latent, 3, 512)); // TP=3: 768 + 4,096
    try t.expectEqual(@as(u64, 102), tokenBytes(100, 3, 2, 1)); // a share that does not divide rounds up
}

test "GLM KV split: a region holds its class's pages at id / W and maps every other id to the trash page" {
    const r: Region = .{ .rank = 1, .classes = 3, .cap = 4 };
    try t.expectEqual(@as(usize, 5), r.pages());
    for ([_]u32{ 1, 4, 7, 10 }, 0..) |id, i| {
        try t.expect(r.holds(id));
        try t.expectEqual(@as(u32, @intCast(i)), r.entry(id));
    }
    for ([_]u32{ 0, 2, 3, 5, 6, 8, 9, 11 }) |id| {
        try t.expect(!r.holds(id));
        try t.expectEqual(@as(u32, 4), r.entry(id));
    }
}

test "GLM KV split: the funded pages leave room for the trash page inside the plan's bytes, a multiple of W" {
    const latent: u64 = 12 * 512 * 1024;
    const pooled: u64 = 12 * 128 * 256;
    for ([_]u32{ 2, 3 }) |w| {
        const page = tokenBytes(pooled, latent, w, 512) * 512;
        for ([_]usize{ w, 2 * w, 64 * w, 4098, 4100, 8196, 12_345 }) |pages| {
            const n = funded(pages, page, pooled, latent, w);
            try t.expect(n % w == 0 and n <= pages);
            try t.expect(arenaBytes(n, pooled, latent, w) <= pages * page);
            try t.expect(n + 2 * w >= pages); // at most a page of each class gives way to the trash page
            if (n + w <= pages) try t.expect(arenaBytes(n + w, pooled, latent, w) > pages * page); // and no more
        }
    }
    try t.expectEqual(@as(usize, 0), funded(1, 10, 2, 100, 2)); // less than the trash page itself
}
