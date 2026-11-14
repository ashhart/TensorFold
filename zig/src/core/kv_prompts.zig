//! TF_KV_SHARE: what a family on the core pool calls of one prompt store over its pages (native/cuda_prompt_pages.zig).
const std = @import("std");
const kv = @import("kv_pool.zig");

/// Where a pass resumes and the marks it keeps states at (`marks` the store's allocation, freed by `free`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// What a pass asks the store about (prompt_radix.Query as a family on the core pool gives it).
pub const Query = struct {
    /// The words the store compares: the request's cache key, one word a model row; for text, the prompt.
    key: []const u32,
    /// The prompt's length; a key of another length reuses nothing.
    prompt_len: u32,

    /// A query of `key` alone, as long as its prompt.
    pub fn of(key: []const u32) Query {
        return .{ .key = key, .prompt_len = @intCast(key.len) };
    }
};

/// The store as a family sees it: a prompt pass begins, keeps at its marks, and reports once its prompt is in.
pub const Prompts = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The deepest state `prompt` resumes: its pages into `adopt` (held once more), the rest restored into `owner`.
        begin: *const fn (ctx: *anyopaque, q: Query, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) anyerror!Plan,
        /// At `at` the store takes the pages of `mine` it lacks and the state; `path`'s first n replace the stream's.
        keep: *const fn (ctx: *anyopaque, q: Query, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize,
        /// One log line after a prompt pass: where it resumed and how many states it kept.
        report: *const fn (ctx: *anyopaque, prompt: usize, from: u32, kept: u64) void,
        free: *const fn (ctx: *anyopaque, marks: []const u32) void,
    };

    pub fn begin(p: Prompts, q: Query, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) !Plan {
        return p.vtable.begin(p.ctx, q, history_len, shared, owner, adopt);
    }

    pub fn keep(p: Prompts, q: Query, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize {
        return p.vtable.keep(p.ctx, q, at, owner, mine, path);
    }

    pub fn report(p: Prompts, prompt: usize, from: u32, kept: u64) void {
        p.vtable.report(p.ctx, prompt, from, kept);
    }

    pub fn free(p: Prompts, marks: []const u32) void {
        p.vtable.free(p.ctx, marks);
    }

    /// `begin`, its marks in `gpa` for a lane to own; a failure gives `adopt`'s pages back.
    pub fn plan(p: Prompts, gpa: std.mem.Allocator, pool: *kv.Pool, q: Query, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) !Plan {
        const got = try p.begin(q, history_len, shared, owner, adopt);
        defer p.free(got.marks);
        const marks = gpa.dupe(u32, got.marks) catch |e| {
            for (adopt.items) |id| pool.release(id);
            return e;
        };
        return .{ .from = got.from, .marks = marks };
    }

    /// At mark `at` the store takes the stream's pages and state; pages it held replace `l`'s, the whole table written.
    pub fn keepPages(p: Prompts, gpa: std.mem.Allocator, pool: *kv.Pool, l: *kv.Lease, q: Query, at: usize, owner: *anyopaque, page_tokens: usize, table: kv.Table) !bool {
        const n = at / page_tokens;
        const path = try gpa.alloc(u32, n);
        defer gpa.free(path);
        const mine = try gpa.dupe(u32, l.ids.items);
        defer gpa.free(mine);
        const shared = p.keep(q, @intCast(at), owner, mine[0..n], path);
        if (std.mem.eql(u32, mine[0..shared], path[0..shared])) return shared == n;
        @memcpy(mine[0..shared], path[0..shared]); // the same tokens' rows: a resume from any row is exact
        try table.write(table.ctx, mine, 0);
        for (path[0..shared], 0..) |id, i| pool.share(l, i, id);
        return shared == n;
    }
};

/// `n` free pool pages the store holds for a state's copied rows; NoCachePages: none, even reclaiming.
pub fn holdBlob(gpa: std.mem.Allocator, pool: *kv.Pool, n: usize) ![]u32 {
    if (n == 0) return &.{};
    const ids = try gpa.alloc(u32, n);
    if (!pool.hold(ids)) {
        gpa.free(ids);
        return error.NoCachePages;
    }
    return ids;
}

/// A state's held pages back to the pool; queued copies out of them must be done.
pub fn dropBlob(gpa: std.mem.Allocator, pool: *kv.Pool, blob: *[]u32) void {
    for (blob.*) |id| pool.release(id);
    gpa.free(blob.*);
    blob.* = &.{};
}

/// The idle server's line: nothing promised, and every page free unless a store holds some.
pub fn idleLine(pool: *const kv.Pool, store: bool) void {
    if (@import("builtin").is_test) return;
    const c = pool.census();
    const ok = pool.idle() and (store or c.free == c.total);
    std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total });
}

/// A kept state's copied rows in store-held pages: byte o in chunk c = o / chunk, region c % regions, page c / regions.
pub const Blob = struct {
    ids: []const u32,
    chunk: u64, // the bytes one page holds in one region (a layer's K or V)
    regions: usize,

    /// Pages that hold `bytes` this way.
    pub fn pagesFor(bytes: u64, chunk: u64, regions: usize) usize {
        const page = chunk * regions;
        return @intCast((bytes + page - 1) / page);
    }

    /// Bytes [off, off + len) as pieces inside one chunk: f(ctx, region, page id, in-chunk offset, from `off`, len).
    pub fn each(b: Blob, off: u64, len: u64, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, u32, u64, u64, u64) anyerror!void) !void {
        var o = off;
        while (o < off + len) {
            const c = o / b.chunk;
            const within = o % b.chunk;
            const n = @min(off + len - o, b.chunk - within);
            const k: usize = @intCast(c / b.regions);
            if (k >= b.ids.len) return error.PastBlob;
            try f(ctx, @intCast(c % b.regions), b.ids[k], within, o - off, n);
            o += n;
        }
    }
};

/// TF_KV_SHARE=n: keep prompts as pool pages and at most n states past them (0, empty or unset: copies, as before).
pub fn sharing() error{BadKvShare}!usize {
    const v = @import("profile").given(.TF_KV_SHARE) orelse return 0;
    return std.fmt.parseInt(usize, v, 10) catch error.BadKvShare;
}

const testing = std.testing;

test "TF_KV_SHARE: unset, empty or whitespace-only text keeps copies, a count keeps its meaning, other text is refused" {
    if (comptime @import("builtin").os.tag == .windows) return error.SkipZigTest;
    const env = @import("sys").env;
    const name = "TF_KV_SHARE";
    const before = if (std.c.getenv(name)) |v| try testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (before) |v| {
            _ = env.setenv(name, v, 1);
            testing.allocator.free(v);
        } else _ = env.unsetenv(name);
    }
    _ = env.unsetenv(name);
    try testing.expectEqual(@as(usize, 0), try sharing());
    for ([_][:0]const u8{ "", " ", "\t", " \t\r\n" }) |v| {
        try testing.expectEqual(@as(c_int, 0), env.setenv(name, v, 1));
        try testing.expectEqual(@as(usize, 0), try sharing());
    }
    for ([_][:0]const u8{ "0", "16", "007" }, [_]usize{ 0, 16, 7 }) |v, n| {
        try testing.expectEqual(@as(c_int, 0), env.setenv(name, v, 1));
        try testing.expectEqual(n, try sharing());
    }
    for ([_][:0]const u8{ "on", " 16", "16 ", "-1" }) |v| {
        try testing.expectEqual(@as(c_int, 0), env.setenv(name, v, 1));
        try testing.expectError(error.BadKvShare, sharing());
    }
}

const Pieces = struct {
    got: *std.ArrayList([5]u64),
    fn add(p: Pieces, region: usize, id: u32, within: u64, from: u64, n: u64) anyerror!void {
        try p.got.append(testing.allocator, .{ region, id, within, from, n });
    }
};

const Stub = struct {
    held: [2]u32,
    fn of(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }
    fn begin(_: *anyopaque, _: Query, _: u32, _: []const u32, _: *anyopaque, _: *std.ArrayList(u32)) anyerror!Plan {
        return .{ .from = 256, .marks = try testing.allocator.dupe(u32, &.{512}) };
    }
    fn keep(ctx: *anyopaque, _: Query, _: u32, _: *anyopaque, mine: []const u32, path: []u32) usize {
        @memcpy(path, Stub.of(ctx).held[0..mine.len]);
        return mine.len;
    }
    fn report(_: *anyopaque, _: usize, _: u32, _: u64) void {}
    fn free(_: *anyopaque, marks: []const u32) void {
        testing.allocator.free(marks);
    }
    const vtable: Prompts.VTable = .{ .begin = begin, .keep = keep, .report = report, .free = free };
};

const Written = struct {
    ids: [4]u32 = @splat(0),
    from: usize = 99,
    cap: usize = 0, // pages a forward may reach: a write ends the table, as mapPages sets caps
    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const w: *Written = @ptrCast(@alignCast(ctx));
        w.from = from;
        w.cap = ids.len;
        @memcpy(w.ids[from..][0 .. ids.len - from], ids[from..]);
    }
};

test "keeping swaps the store's pages in and writes the whole table: a table of the prefix alone hides the rest" {
    var pool = try kv.Pool.init(testing.allocator, 8, 1);
    defer pool.deinit();
    var held: [2]u32 = undefined;
    try testing.expect(pool.hold(&held));
    var l: kv.Lease = .{};
    defer pool.finish(&l);
    try pool.promise(&l, 4);
    try pool.cover(&l, 4, null);
    const rest = [2]u32{ l.ids.items[2], l.ids.items[3] };
    var stub: Stub = .{ .held = held };
    const p: Prompts = .{ .ctx = &stub, .vtable = &Stub.vtable };
    var w: Written = .{};
    var owner: u8 = 0;
    const whole = try p.keepPages(testing.allocator, &pool, &l, .of(&.{ 1, 2, 3 }), 512, &owner, 256, .{ .ctx = &w, .write = Written.write });
    try testing.expect(whole);
    try testing.expectEqual(@as(usize, 0), w.from);
    try testing.expectEqual(@as(usize, 4), w.cap);
    try testing.expectEqualSlices(u32, &.{ held[0], held[1], rest[0], rest[1] }, &w.ids);
    try testing.expectEqualSlices(u32, &w.ids, l.ids.items);
    try testing.expectEqual(@as(u32, 2), pool.holders(held[0]));
    for (held) |id| pool.release(id);
}

test "a plan's marks are the lane's own; a blob's pages come from the pool and go back" {
    var pool = try kv.Pool.init(testing.allocator, 4, 1);
    defer pool.deinit();
    var stub: Stub = .{ .held = .{ 0, 0 } };
    const p: Prompts = .{ .ctx = &stub, .vtable = &Stub.vtable };
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    var owner: u8 = 0;
    const got = try p.plan(testing.allocator, &pool, .of(&.{1}), 1, &.{}, &owner, &adopt);
    defer testing.allocator.free(got.marks);
    try testing.expectEqual(@as(u32, 256), got.from);
    try testing.expectEqualSlices(u32, &.{512}, got.marks);
    var blob = try holdBlob(testing.allocator, &pool, 3);
    try testing.expectEqual(@as(usize, 1), pool.available());
    try testing.expectError(error.NoCachePages, holdBlob(testing.allocator, &pool, 2));
    dropBlob(testing.allocator, &pool, &blob);
    try testing.expectEqual(@as(usize, 4), pool.available());
    try testing.expectEqual(@as(usize, 0), blob.len);
    try testing.expectEqual(@as(usize, 0), (try holdBlob(testing.allocator, &pool, 0)).len);
}

test "a blob's bytes walk its regions a chunk at a time, then its next page; a run splits at chunk edges" {
    const b: Blob = .{ .ids = &.{ 7, 3 }, .chunk = 10, .regions = 2 };
    try testing.expectEqual(@as(usize, 2), Blob.pagesFor(21, 10, 2));
    var got: std.ArrayList([5]u64) = .empty;
    defer got.deinit(testing.allocator);
    try b.each(5, 30, Pieces{ .got = &got }, Pieces.add); // bytes 5..35: chunks 0 (page 7, K), 1 (page 7, V), 2 (page 3, K), 3 (page 3, V)
    try testing.expectEqualSlices([5]u64, &.{ .{ 0, 7, 5, 0, 5 }, .{ 1, 7, 0, 5, 10 }, .{ 0, 3, 0, 15, 10 }, .{ 1, 3, 0, 25, 5 } }, got.items);
    try testing.expectError(error.PastBlob, b.each(39, 2, Pieces{ .got = &got }, Pieces.add));
}
