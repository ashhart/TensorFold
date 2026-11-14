//! A stream's sliding-window pages (P4, #2444): bands of sliding layers in pool pages, retired once no kernel reads it.
const std = @import("std");
const kv = @import("kv_pool.zig");
const Allocator = std.mem.Allocator;

/// How a family's sliding layers sit in pool pages: `rows` a page (its full pages'), `per_band` layers a page.
pub const Geometry = struct {
    /// The window: a row at `p` reads keys from `p - w + 1`.
    w: u32,
    /// The widest first-key rounding of the family's sliding kernels (1: none).
    a: u32,
    rows: u32,
    bands: u32,
    per_band: u32,
    /// Bytes a page holds no row in.
    padding: u64,
};

/// The geometry of `layers` sliding layers of `row_bytes` a row (K and V) in pages of `page_bytes` and `rows` rows.
pub fn geometry(w: u32, a: u32, rows: u32, layers: u32, row_bytes: u64, page_bytes: u64) error{WindowPageTooSmall}!Geometry {
    if (w == 0 or a == 0 or rows == 0 or row_bytes == 0 or layers == 0) return error.WindowPageTooSmall;
    const band_rows = @as(u64, rows) * row_bytes;
    const per: u32 = @intCast(@min(page_bytes / band_rows, layers));
    if (per == 0) return error.WindowPageTooSmall;
    return .{ .w = w, .a = a, .rows = rows, .bands = (layers + per - 1) / per, .per_band = per, .padding = page_bytes - per * band_rows };
}

/// The first row any sliding kernel loads for a forward whose first query sits at `c`.
pub fn first(g: Geometry, c: u32) u32 {
    const lo = if (c + 1 > g.w) c + 1 - g.w else 0;
    return lo - lo % g.a;
}

/// Pages a stream's window holds at most, for chunks of `chunk` rows and `drafts` rows past them.
pub fn promise(g: Geometry, chunk: u32, drafts: u32) usize {
    const live = g.w - 1 + g.a - 1 + chunk + drafts;
    return g.bands * ((live + g.rows - 1) / g.rows + 1);
}

/// Where a band's ids reach the device: logical pages from `from` of band `band` (`kv.retirement.none`: never read).
pub const Tables = struct {
    ctx: *anyopaque,
    write: *const fn (ctx: *anyopaque, band: u32, ids: []const u32, from: usize) anyerror!void,
};

/// A band's pages a keep retains: logical pages from `base`, the last partial when the keep is inside it.
pub const Band = struct { base: u32, ids: []u32 };

/// One stream's window pages: a lease whose ids are page-major (`ids[j * bands + b]`), retired from the front.
pub const Window = struct {
    lease: kv.Lease = .{},
    g: Geometry,
    /// Logical pages below this one are retired (or were never held).
    low: u32 = 0,

    fn pages(w: *const Window) u32 {
        return @intCast(w.lease.ids.items.len / w.g.bands);
    }

    /// Before a forward to row `end - 1` from committed `k`: pages below `first(k)` retire and settle; cover `end`.
    pub fn step(w: *Window, pool: *kv.Pool, k: u32, end: u32, token: u64, t: Tables) !void {
        const low = @min(first(w.g, k) / w.g.rows, w.pages());
        errdefer kv.retirement.settle(pool, token) catch {}; // retired pages never stay pending: one stream, their readers are queued
        if (low > w.low) { // a retry passes retired entries by and writes their tables again
            for (w.low..low) |j| for (0..w.g.bands) |b| {
                const i = j * w.g.bands + b;
                if (w.lease.ids.items[i] != kv.retirement.none) try kv.retirement.retire(pool, &w.lease, i, token);
            };
            try w.reach(t, w.low, low);
            w.low = low;
        }
        try kv.retirement.settle(pool, token);
        const need = (end + w.g.rows - 1) / w.g.rows;
        const had = w.pages();
        if (need <= had) return;
        const old = w.lease.ids.items.len;
        errdefer pool.trim(&w.lease, old); // all or nothing: a retry takes and maps them afresh
        for (had..need) |_| for (0..w.g.bands) |_| {
            _ = try pool.take(&w.lease);
        };
        try w.reach(t, had, need);
    }

    /// Band `b`'s ids of logical pages [from, to) to its table, one band at a time.
    fn reach(w: *const Window, t: Tables, from: usize, to: usize) !void {
        if (to <= from) return;
        var buf: [64]u32 = undefined;
        for (0..w.g.bands) |b| {
            var j = from;
            while (j < to) {
                const n = @min(to - j, buf.len);
                for (buf[0..n], j..) |*o, jj| o.* = w.lease.ids.items[jj * w.g.bands + b];
                try t.write(t.ctx, @intCast(b), buf[0..n], j);
                j += n;
            }
        }
    }

    /// A keep at `at`: each band's pages for rows [first(at), at) into `out` (ids in `gpa`, the caller frees them).
    pub fn parts(w: *const Window, at: u32, gpa: Allocator, out: []Band) !void {
        std.debug.assert(out.len == w.g.bands);
        const base = first(w.g, at) / w.g.rows;
        const top = (at + w.g.rows - 1) / w.g.rows;
        if (base < w.low or top > w.pages()) return error.WindowPagesGone;
        var made: usize = 0;
        errdefer for (out[0..made]) |o| gpa.free(o.ids);
        for (out, 0..) |*o, b| {
            const ids = try gpa.alloc(u32, top - base);
            for (ids, base..) |*id, j| id.* = w.lease.ids.items[j * w.g.bands + b];
            o.* = .{ .base = base, .ids = ids };
            made += 1;
            if (std.mem.indexOfScalar(u32, ids, kv.retirement.none) != null) return error.WindowPagesGone; // a retire that failed midway
        }
    }

    /// A resume at `from` into an empty window: entries below `first(from)` skipped, then pages for the store's copies.
    pub fn resumeAt(w: *Window, pool: *kv.Pool, from: u32, per_band_pages: u32, t: Tables) !void {
        std.debug.assert(w.lease.ids.items.len == 0);
        const base = first(w.g, from) / w.g.rows;
        try kv.retirement.skip(pool, &w.lease, base * w.g.bands);
        w.low = base;
        for (0..per_band_pages) |_| for (0..w.g.bands) |_| {
            _ = try pool.take(&w.lease);
        };
        try w.reach(t, base, base + per_band_pages);
    }

    /// The stream ended: its window pages and promise go back.
    pub fn finish(w: *Window, pool: *kv.Pool) void {
        pool.finish(&w.lease);
        w.low = 0;
    }
};

test {
    _ = @import("kv_window_test.zig");
}
