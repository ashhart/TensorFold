//! The core KV pool on Metal: every page in one arena buffer, each stream's page table in one shared buffer.
const std = @import("std");
const mtl = @import("metal");
const kv = @import("kv_pool.zig");
const Allocator = std.mem.Allocator;

/// Positions a page holds: a multiple of every Metal attention key tile (4 to 256), so no tile spans two pages.
pub const page_tokens = 256;

/// A position's byte offset in the arena through its stream's table (kernels/metal/core/kv_pages.h).
pub const address_msl = @import("kernel_sources").kv_pages;

comptime {
    const define = std.fmt.comptimePrint("#define TF_PAGE_TOKENS {d}\n", .{page_tokens});
    if (std.mem.indexOf(u8, address_msl, define) == null) @compileError("kv_pages.h's TF_PAGE_TOKENS differs from page_tokens");
}

/// Every page in one buffer, page `id` at id x page_bytes: one binding and one residency entry, as CUDA's arena.
pub const Arena = struct {
    buf: ?mtl.Buffer, // null: no pages, or a host test without a device
    pages: usize,
    page_bytes: u64,
    owned: bool = true, // false: the family's buffer (Pages.initIn), which it releases

    /// `pages` pages of `page_bytes` in one buffer; ArenaTooLarge past the device's longest buffer.
    pub fn init(device: ?mtl.Device, pages: usize, page_bytes: u64, options: usize) !Arena {
        const d = device orelse return .{ .buf = null, .pages = pages, .page_bytes = page_bytes };
        if (pages == 0) return .{ .buf = null, .pages = 0, .page_bytes = page_bytes };
        const bytes = try std.math.mul(u64, pages, page_bytes);
        if (bytes > d.maxBufferLength()) return error.ArenaTooLarge;
        return .{ .buf = try d.buffer(@intCast(bytes), options), .pages = pages, .page_bytes = page_bytes };
    }

    pub fn deinit(a: *Arena) void {
        if (a.owned) if (a.buf) |b| b.deinit();
        a.* = undefined;
    }

    /// Page `id`'s first byte in the arena.
    pub fn offset(a: Arena, id: u32) u64 {
        return @as(u64, id) * a.page_bytes;
    }
};

/// Every stream's page table, `most` entries a stream in one buffer the host writes and the kernels read.
pub const Tables = struct {
    buf: ?mtl.Buffer, // shared storage, so a table write is a store; null: host memory for a host test
    host: []u32,
    at: []At,
    most: usize,
    owned: bool = true, // false: the family's buffer (Pages.initIn), which it releases

    /// A stream's table, the context kv_pool.Table writes through.
    const At = struct { entries: []u32 };

    pub fn init(gpa: Allocator, device: ?mtl.Device, lanes: usize, most: usize) !Tables {
        const n = try std.math.mul(usize, lanes, most);
        const buf: ?mtl.Buffer = if (device) |d| try d.buffer(@max(n, 1) * 4, mtl.ResourceOptions.shared) else null;
        errdefer if (buf) |b| b.deinit();
        const host: []u32 = if (buf) |b| b.slice(u32, n) else try gpa.alloc(u32, n);
        errdefer if (buf == null) gpa.free(host);
        return on(gpa, buf, host, lanes, most, true);
    }

    /// The tables over `host` (`buf`'s contents, or host memory when it is null), zeroed.
    fn on(gpa: Allocator, buf: ?mtl.Buffer, host: []u32, lanes: usize, most: usize, owned: bool) !Tables {
        @memset(host, 0);
        const at = try gpa.alloc(At, lanes);
        for (at, 0..) |*a, i| a.* = .{ .entries = host[i * most ..][0..most] };
        return .{ .buf = buf, .host = host, .at = at, .most = most, .owned = owned };
    }

    pub fn deinit(t: *Tables, gpa: Allocator) void {
        if (!t.owned) {} else if (t.buf) |b| b.deinit() else gpa.free(t.host);
        gpa.free(t.at);
        t.* = undefined;
    }

    /// Lane `lane`'s table as the pool writes it: entries from `from` on, those in use by queued work left alone.
    pub fn table(t: *Tables, lane: usize) kv.Table {
        return .{ .ctx = &t.at[lane], .write = write };
    }

    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const a: *At = @ptrCast(@alignCast(ctx));
        if (ids.len > a.entries.len) return error.PastTable;
        @memcpy(a.entries[from..ids.len], ids[from..]);
    }

    /// Entry `i` of lane `lane` again, after Pool.share swapped its page for the store's.
    pub fn set(t: *Tables, lane: usize, i: usize, id: u32) void {
        t.at[lane].entries[i] = id;
    }

    /// Lane `lane`'s table, `n` entries.
    pub fn entries(t: *const Tables, lane: usize, n: usize) []const u32 {
        return t.at[lane].entries[0..n];
    }

    /// Where lane `lane`'s table starts in the buffer, for a kernel's binding.
    pub fn offset(t: *const Tables, lane: usize) usize {
        return lane * t.most * 4;
    }
};

/// The pool, its arena and tables, and a lease a lane: what a Metal family's paged streams hold.
pub const Pages = struct {
    gpa: Allocator,
    layout: kv.Layout,
    pool: kv.Pool,
    arena: Arena,
    tables: Tables,
    leases: []kv.Lease,

    /// `lanes` streams over `pages` pages, each at most `most_tokens` rows; `device` null: host memory, no arena.
    pub fn init(gpa: Allocator, device: ?mtl.Device, l: kv.Layout, lanes: usize, pages: usize, most_tokens: u64, options: usize) !*Pages {
        if (l.page_tokens != page_tokens) return error.PageTokens; // the kernels' tables index 256 rows a page
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        var pool = try kv.Pool.init(gpa, pages, l.pageBytes());
        errdefer pool.deinit();
        var arena = try Arena.init(device, pages, l.pageBytes(), options);
        errdefer arena.deinit();
        var tables = try Tables.init(gpa, device, lanes, l.pages(most_tokens));
        errdefer tables.deinit(gpa);
        const leases = try gpa.alloc(kv.Lease, lanes);
        @memset(leases, .{});
        p.* = .{ .gpa = gpa, .layout = l, .pool = pool, .arena = arena, .tables = tables, .leases = leases };
        return p;
    }

    /// The bytes initIn's two buffers take: the arena's pages and every lane's table.
    pub fn bytes(l: kv.Layout, lanes: usize, pages: usize, most_tokens: u64) struct { arena: u64, tables: u64 } {
        return .{ .arena = pages * l.pageBytes(), .tables = @max(lanes * l.pages(most_tokens), 1) * 4 };
    }

    /// As init, over a family's own buffers (sized by `bytes`, in its residency set): it keeps and releases them.
    pub fn initIn(gpa: Allocator, l: kv.Layout, lanes: usize, pages: usize, most_tokens: u64, arena: ?mtl.Buffer, tables: mtl.Buffer) !*Pages {
        if (l.page_tokens != page_tokens) return error.PageTokens;
        const sized = bytes(l, lanes, pages, most_tokens);
        if (pages > 0 and (arena == null or arena.?.length() < sized.arena)) return error.ArenaTooSmall;
        if (tables.length() < sized.tables) return error.TablesTooSmall;
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        var pool = try kv.Pool.init(gpa, pages, l.pageBytes());
        errdefer pool.deinit();
        const most = l.pages(most_tokens);
        var t = try Tables.on(gpa, tables, tables.slice(u32, lanes * most), lanes, most, false);
        errdefer t.deinit(gpa);
        const leases = try gpa.alloc(kv.Lease, lanes);
        @memset(leases, .{});
        p.* = .{ .gpa = gpa, .layout = l, .pool = pool, .arena = .{ .buf = arena, .pages = pages, .page_bytes = l.pageBytes(), .owned = false }, .tables = t, .leases = leases };
        return p;
    }

    pub fn deinit(p: *Pages) void {
        for (p.leases) |*l| p.pool.finish(l);
        p.gpa.free(p.leases);
        p.tables.deinit(p.gpa);
        p.arena.deinit();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    /// Lane `lane` reserves `need` pages (Layout.need); LaneMemory when they do not fit now.
    pub fn admit(p: *Pages, lane: usize, need: usize) !void {
        if (need > p.tables.most) return error.PastTable;
        try p.pool.promise(&p.leases[lane], need);
    }

    /// Lane `lane` holds the pages of `tokens` rows, the new ones written to its table; a failure takes them back.
    pub fn cover(p: *Pages, lane: usize, tokens: u64) !void {
        try p.pool.cover(&p.leases[lane], p.layout.pages(tokens), p.tables.table(lane));
    }

    /// Lane `lane`'s page `i` becomes the store's page `id` holding the same rows, in its table too.
    pub fn share(p: *Pages, lane: usize, i: usize, id: u32) void {
        p.pool.share(&p.leases[lane], i, id);
        p.tables.set(lane, i, id);
    }

    /// Lane `lane`'s stream ended: its promise and pages go back.
    pub fn finish(p: *Pages, lane: usize) void {
        p.pool.finish(&p.leases[lane]);
    }

    /// Lane `lane`'s page ids in token order.
    pub fn ids(p: *const Pages, lane: usize) []const u32 {
        return p.leases[lane].ids.items;
    }

    /// No lane holds or is promised a page.
    pub fn idle(p: *const Pages) bool {
        for (p.leases) |l| if (l.promised > 0 or l.ids.items.len > 0) return false;
        return p.pool.idle();
    }
};

fn testLayout() kv.Layout {
    return .{ .fixed_bytes = 0, .token_bytes = 64, .page_tokens = page_tokens, .snap_bytes = 0 };
}

test "a lease's pages reach its table and go back when the stream ends" {
    const p = try Pages.init(std.testing.allocator, null, testLayout(), 2, 8, 4 * page_tokens, 0);
    defer p.deinit();
    try p.admit(0, 3);
    try p.cover(0, 3 * page_tokens);
    try std.testing.expectEqualSlices(u32, p.ids(0), p.tables.entries(0, 3));
    try std.testing.expectEqual(@as(usize, 5), p.pool.available());
    p.finish(0);
    try std.testing.expect(p.idle());
    try std.testing.expectEqual(@as(usize, 8), p.pool.available());
}

test "a full pool refuses a lane until another lane's pages go back" {
    const p = try Pages.init(std.testing.allocator, null, testLayout(), 2, 4, 4 * page_tokens, 0);
    defer p.deinit();
    try p.admit(0, 3);
    try std.testing.expectError(error.LaneMemory, p.admit(1, 2)); // a promise counts before its pages are taken
    try p.cover(0, 2 * page_tokens);
    try std.testing.expectError(error.LaneMemory, p.admit(1, 2));
    p.finish(0);
    try p.admit(1, 2);
    try p.cover(1, 2 * page_tokens);
    p.finish(1);
    try std.testing.expect(p.idle());
}

test "a partial page is a whole one, growth writes only the new entries, a table's end refuses" {
    const p = try Pages.init(std.testing.allocator, null, testLayout(), 2, 8, 3 * page_tokens, 0);
    defer p.deinit();
    try std.testing.expectError(error.PastTable, p.admit(0, 4)); // past a lane's table
    try p.admit(0, 3);
    try p.cover(0, 1);
    try std.testing.expectEqual(@as(usize, 1), p.ids(0).len);
    p.tables.set(0, 0, 77); // a sentinel: growth leaves entries already written alone
    try p.cover(0, page_tokens + 1);
    try std.testing.expectEqual(@as(u32, 77), p.tables.entries(0, 2)[0]);
    try std.testing.expectEqual(p.ids(0)[1], p.tables.entries(0, 2)[1]);
    try std.testing.expectError(error.PastReservation, p.cover(0, 4 * page_tokens)); // past its promise: none kept
    try std.testing.expectEqual(@as(usize, 2), p.ids(0).len);
    p.finish(0);
    try std.testing.expect(p.idle());
}

test "an idle pool holds no page: interleaved lanes, a run between them, every page back" {
    const p = try Pages.init(std.testing.allocator, null, testLayout(), 3, 6, 6 * page_tokens, 0);
    defer p.deinit();
    try p.admit(0, 3);
    try p.admit(1, 3);
    for (1..4) |n| {
        try p.cover(0, n * page_tokens);
        try p.cover(1, n * page_tokens);
    }
    try std.testing.expectEqual(@as(?u32, null), p.pool.findRun(1)); // every page held
    p.finish(0); // its pages are every other id: no run of two is free
    try std.testing.expectEqual(@as(?u32, null), p.pool.findRun(2));
    try p.admit(2, 2);
    try p.cover(2, 2 * page_tokens);
    try std.testing.expect(!p.idle());
    p.finish(1);
    p.finish(2);
    try std.testing.expect(p.idle());
    const c = p.pool.census();
    try std.testing.expectEqual(c.total, c.free);
    try std.testing.expectEqual(@as(usize, 0), c.leased);
}

test "a page shared with the store's copy is written to the table" {
    const p = try Pages.init(std.testing.allocator, null, testLayout(), 2, 4, 2 * page_tokens, 0);
    defer p.deinit();
    try p.admit(0, 2);
    try p.cover(0, 2 * page_tokens);
    try p.admit(1, 1);
    try p.cover(1, page_tokens);
    const kept = p.ids(1)[0];
    p.pool.retain(kept); // the store's hold, as a keep takes
    p.share(0, 1, kept);
    try std.testing.expectEqual(kept, p.tables.entries(0, 2)[1]);
    p.finish(0);
    p.finish(1);
    p.pool.release(kept);
    try std.testing.expect(p.idle());
}

test "the kernels' address helper pages by TF_PAGE_TOKENS" {
    try std.testing.expect(std.mem.indexOf(u8, address_msl, "pos / TF_PAGE_TOKENS") != null);
    try std.testing.expect(std.mem.indexOf(u8, address_msl, "pos % TF_PAGE_TOKENS") != null);
}

test {
    _ = @import("metal_kv_pages_test.zig");
}
