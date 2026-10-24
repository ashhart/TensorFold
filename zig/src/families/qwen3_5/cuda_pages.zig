//! Qwen3.5's attention caches from the core pool: the streams beside the first take pages of one device arena.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const engine = @import("cuda_engine.zig");
const caches = @import("cuda_cache.zig");

const Engine = engine.Engine;
const Cache = caches.Cache;

/// Rows past a request's prompt and reply its pages hold: one more forward's (a verify window or a prompt chunk).
pub const window_rows: u32 = engine.max_rows;

/// TF_QWEN_KV_PAGES=n: streams beside the first take n-position pages from one pool (unset or 0: a window each).
pub fn paging() !usize {
    const v = std.c.getenv("TF_QWEN_KV_PAGES") orelse return 0;
    return parsePaging(std.mem.span(v));
}

pub fn parsePaging(v: []const u8) !usize {
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadQwenKvPages;
    if (n != 0 and !caches.validPageTokens(n)) return error.BadQwenKvPages;
    return n;
}

/// TF_QWEN_KV_PAGES_ALL=1: every stream on pages, the first stream's reserved window left idle (checks).
pub fn pagedOnly() bool {
    const v = std.c.getenv("TF_QWEN_KV_PAGES_ALL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// Pages a stream maps at most: a full window and one more forward.
fn mostPages(capacity: usize, tokens: usize) usize {
    return (capacity + window_rows + tokens - 1) / tokens;
}

/// A lane's table and `fixed` bytes; its keys and values a token; a run of its state, log, taps and a `ring`-byte ring.
pub fn layout(e: *const Engine, capacity: usize, tokens: usize, ring: usize, fixed: usize) kv.Layout {
    return .{ .fixed_bytes = mostPages(capacity, tokens) * 4 + fixed, .token_bytes = e.g.kvBytes(1, e.kv8), .page_tokens = @intCast(tokens), .snap_bytes = e.g.stateBytes(), .run_bytes = Cache.runBytes(e, ring) };
}

/// One pool lane: a cache whose table maps its stream's pages, that stream's lease and its run's first page.
const Lane = struct { cache: *Cache, lease: kv.Lease = .{}, used: bool = false, run: u32 = 0 };

pub const Pages = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    layout: kv.Layout,
    capacity: usize,
    ring: usize,
    pool: kv.Pool,
    arena: cuda.DeviceBuffer,
    lanes: []Lane,

    /// `streams` lanes and `pages` pages of layout `l`, made and touched now; each run holds a `ring`-byte ring.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, l: kv.Layout, capacity: usize, ring: usize, streams: usize, pages: usize) !*Pages {
        const tokens: usize = l.page_tokens;
        if (!caches.validPageTokens(tokens) or capacity > std.math.maxInt(c_int) - window_rows) return error.BadQwenKvPages;
        if (pages > @as(usize, std.math.maxInt(c_int)) + 1) return error.BadQwenKvPages;
        const page_bytes = try std.math.mul(usize, @intCast(l.token_bytes), tokens);
        if (page_bytes == 0) return error.BadQwenKvPages;
        _ = try std.math.add(u64, l.run_bytes, page_bytes - 1);
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        var arena = try cuda.DeviceBuffer.alloc(e.d, try std.math.mul(usize, pages, page_bytes));
        errdefer arena.free();
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .capacity = capacity, .ring = ring, .pool = try kv.Pool.init(gpa, pages, page_bytes), .arena = arena, .lanes = &.{} };
        errdefer p.pool.deinit();
        p.lanes = try gpa.alloc(Lane, streams);
        errdefer gpa.free(p.lanes);
        var made: usize = 0;
        errdefer for (p.lanes[0..made]) |x| {
            x.cache.deinit(gpa);
            gpa.destroy(x.cache);
        };
        for (p.lanes) |*x| {
            const c = try gpa.create(Cache);
            errdefer gpa.destroy(c);
            c.* = try Cache.initPaged(gpa, e, mostPages(capacity, tokens), tokens, arena.ptr);
            x.* = .{ .cache = c };
            made += 1;
        }
        errdefer cuda.completion.drain(e.d, e.stream, "qwen3_5: a failed pool initialization did not drain");
        try arena.fill8(0, e.stream.handle);
        for (p.lanes) |x| try x.cache.paged.?.table.fill8(0, e.stream.handle);
        try e.stream.synchronize();
        return p;
    }

    pub fn deinit(p: *Pages) void {
        cuda.completion.drain(p.e.d, p.e.stream, "qwen3_5: the pool did not drain at close");
        for (p.lanes) |*x| {
            if (x.used) {
                p.pool.releaseRun(x.run, p.layout.runPages());
                p.pool.finish(&x.lease);
            }
            x.cache.deinit(p.gpa);
            p.gpa.destroy(x.cache);
        }
        p.gpa.free(p.lanes);
        p.arena.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    fn need(p: *const Pages, prompt: usize, max_tokens: usize) usize {
        return p.layout.need(prompt, max_tokens, window_rows);
    }

    /// Whether a lane is free, the pool can promise a request's pages now and a run of them is free.
    pub fn fits(p: *const Pages, prompt: usize, max_tokens: usize) bool {
        if (prompt > p.capacity or max_tokens > p.capacity - prompt) return false;
        for (p.lanes) |x| if (!x.used) return p.pool.fits(p.need(prompt, max_tokens)) and p.pool.findRun(p.layout.runPages()) != null;
        return false;
    }

    /// A free lane's cache, reset, holding the pages of `prompt` + `max_tokens` rows and a window; LaneMemory: not now.
    pub fn open(p: *Pages, prompt: usize, max_tokens: usize) !*Cache {
        if (prompt > p.capacity or max_tokens > p.capacity - prompt) return error.InvalidQwenPrompt;
        const x = for (p.lanes) |*x| {
            if (!x.used) break x;
        } else return error.LaneMemory;
        const k = p.layout.runPages();
        const n = p.need(prompt, max_tokens) - k; // keys and values
        if (n > mostPages(p.capacity, p.layout.page_tokens) or n * p.layout.page_tokens > std.math.maxInt(c_int)) return error.InvalidQwenPrompt;
        try p.pool.promise(&x.lease, n + k);
        errdefer p.pool.finish(&x.lease);
        // the run first: KV pages fill from the bottom, runs from the top; a fragmented pool waits (LaneMemory)
        const run = p.pool.claimRun(&x.lease, k) orelse return error.LaneMemory;
        errdefer p.pool.releaseRun(run, k);
        errdefer cuda.completion.drain(p.e.d, p.e.stream, "qwen3_5: a failed lane open did not drain");
        var up: Upload = .{ .p = p, .cache = x.cache };
        try p.pool.cover(&x.lease, n, .{ .ctx = &up, .write = Upload.write });
        try x.cache.mapRun(p.e, p.arena.ptr + run * p.layout.pageBytes(), k * p.layout.pageBytes(), p.ring);
        try x.cache.reset(p.e);
        x.cache.paged.?.ids = x.lease.ids.items;
        x.cache.cap = n * p.layout.page_tokens;
        x.run = run;
        x.used = true;
        return x.cache;
    }

    /// The stream on `cache` ended: its pages go back once the forwards that read them are done. False: not a lane's.
    pub fn close(p: *Pages, cache: *Cache) bool {
        const x = for (p.lanes) |*x| {
            if (x.cache == cache) break x;
        } else return false;
        if (!x.used) return true;
        cuda.completion.drain(p.e.d, p.e.stream, "qwen3_5: a closed stream did not drain");
        p.pool.releaseRun(x.run, p.layout.runPages());
        p.pool.finish(&x.lease);
        cache.unmapRun(p.e);
        cache.paged.?.ids = &.{};
        cache.cap = 0;
        x.used = false;
        return true;
    }

    /// Whether `cache` is one of the pool's lanes.
    pub fn owns(p: *const Pages, cache: *const Cache) bool {
        for (p.lanes) |x| if (x.cache == cache) return true;
        return false;
    }

    /// A lane's new table entries uploaded to its device table.
    const Upload = struct {
        p: *Pages,
        cache: *Cache,

        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(ctx));
            const t = u.cache.paged.?.table;
            errdefer cuda.completion.drain(u.p.e.d, u.p.e.stream, "qwen3_5: a failed table upload did not drain");
            try t.uploadAsync(from * 4, std.mem.sliceAsBytes(ids[from..]), u.p.e.stream.handle);
            try u.p.e.stream.synchronize(); // the ids slice may move as the lease grows
        }
    };
};

test "a stream's pages: its prompt, reply and one more forward, no more than a full window's" {
    const l: kv.Layout = .{ .fixed_bytes = 1, .token_bytes = 64 << 10, .page_tokens = 256, .snap_bytes = 1 };
    try std.testing.expectEqual(@as(usize, 17), l.need(1, 0, window_rows));
    try std.testing.expectEqual(@as(usize, 17), l.need(100, 29, window_rows));
    try std.testing.expectEqual(@as(usize, 144), mostPages(32768, 256));
    try std.testing.expect(l.need(32768 - 1, 1, window_rows) <= mostPages(32768, 256));
}

test "KV page dimensions: pool rows fit the signed kernel argument and zero disables paging" {
    try std.testing.expectEqual(@as(usize, 0), try parsePaging("0"));
    try std.testing.expectEqual(@as(usize, 48), try parsePaging("48"));
    try std.testing.expectError(error.BadQwenKvPages, parsePaging("2147483648"));
    try std.testing.expectError(error.BadQwenKvPages, parsePaging("18446744073709551615"));
}
