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
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch error.BadQwenKvPages;
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

/// A stream's fixed part (state, log, tap ring, table), its keys and values a token, a kept state's recurrences.
pub fn layout(e: *const Engine, capacity: usize, tokens: usize) kv.Layout {
    return .{ .fixed_bytes = e.streamBytes(0) + mostPages(capacity, tokens) * 4, .token_bytes = e.g.kvBytes(1, e.kv8), .page_tokens = @intCast(tokens), .snap_bytes = e.g.stateBytes() };
}

/// One pool lane: a cache whose table maps its stream's pages, and that stream's lease.
const Lane = struct { cache: *Cache, lease: kv.Lease = .{}, used: bool = false };

pub const Pages = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    layout: kv.Layout,
    capacity: usize,
    pool: kv.Pool,
    arena: cuda.DeviceBuffer,
    lanes: []Lane,

    /// `streams` lanes and `pages` pages, made and touched now.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, capacity: usize, tokens: usize, streams: usize, pages: usize) !*Pages {
        const l = layout(e, capacity, tokens);
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        var arena = try cuda.DeviceBuffer.alloc(e.d, pages * l.pageBytes());
        errdefer arena.free();
        try arena.fill8(0, null); // committed now (the driver commits on first touch)
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .capacity = capacity, .pool = try kv.Pool.init(gpa, pages, l.pageBytes()), .arena = arena, .lanes = &.{} };
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
        return p;
    }

    pub fn deinit(p: *Pages) void {
        p.e.stream.synchronize() catch std.log.warn("qwen3_5: the pool's stream did not drain at close", .{});
        for (p.lanes) |*x| {
            if (x.used) p.pool.finish(&x.lease);
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

    /// Whether a lane is free and the pool can promise a request's pages now.
    pub fn fits(p: *const Pages, prompt: usize, max_tokens: usize) bool {
        for (p.lanes) |x| if (!x.used) return p.pool.fits(p.need(prompt, max_tokens));
        return false;
    }

    /// A free lane's cache, reset, holding the pages of `prompt` + `max_tokens` rows and a window; LaneMemory: not now.
    pub fn open(p: *Pages, prompt: usize, max_tokens: usize) !*Cache {
        const x = for (p.lanes) |*x| {
            if (!x.used) break x;
        } else return error.LaneMemory;
        const n = p.need(prompt, max_tokens);
        if (n > mostPages(p.capacity, p.layout.page_tokens)) return error.InvalidQwenPrompt;
        try p.pool.promise(&x.lease, n);
        errdefer p.pool.finish(&x.lease);
        var up: Upload = .{ .p = p, .cache = x.cache };
        try p.pool.cover(&x.lease, n, .{ .ctx = &up, .write = Upload.write });
        try x.cache.reset(p.e);
        x.cache.paged.?.ids = x.lease.ids.items;
        x.cache.cap = n * p.layout.page_tokens;
        x.used = true;
        return x.cache;
    }

    /// The stream on `cache` ended: its pages go back once the forwards that read them are done. False: not a lane's.
    pub fn close(p: *Pages, cache: *Cache) bool {
        const x = for (p.lanes) |*x| {
            if (x.cache == cache) break x;
        } else return false;
        if (!x.used) return true;
        // a page another stream takes next may still be read by queued work on the engine's stream
        p.e.stream.synchronize() catch std.log.warn("qwen3_5: a closed stream's forwards did not drain; its pages go back", .{});
        p.pool.finish(&x.lease);
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
            try t.uploadAsync(from * 4, std.mem.sliceAsBytes(ids[from..]), u.p.e.stream.handle);
            try u.p.e.stream.synchronize(); // the ids slice may move as the lease grows
        }
    };
};

test "a stream's pages: its prompt, reply and one more forward, no more than a full window's" {
    const l: kv.Layout = .{ .fixed_bytes = 1, .token_bytes = 64 << 10, .page_tokens = 256, .snap_bytes = 1 };
    try std.testing.expectEqual(@as(usize, 1), l.need(1, 0, window_rows));
    try std.testing.expectEqual(@as(usize, 2), l.need(100, 29, window_rows));
    try std.testing.expectEqual(@as(usize, 129), mostPages(32768, 256));
    try std.testing.expect(l.need(32768 - 1, 1, window_rows) <= mostPages(32768, 256));
}
