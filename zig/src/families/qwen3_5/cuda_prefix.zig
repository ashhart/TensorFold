//! TF_QWEN_KV_SHARE: a prompt cache over the pool's pages (prompt_radix.zig); a resume adopts them, restores the state.
const std = @import("std");
const cuda = @import("cuda");
pub const radix = @import("engine_api").prompt_radix;
const kv = @import("core").kv_pool;
const engine = @import("cuda_engine.zig");
const pages_mod = @import("cuda_pages.zig");
const caches = @import("cuda_cache.zig");

const Engine = engine.Engine;
const Cache = caches.Cache;

/// TF_QWEN_KV_SHARE=n: keep prompts as pool pages and at most n recurrent-state snapshots (0 or unset: off).
pub fn sharing() !usize {
    const v = std.c.getenv("TF_QWEN_KV_SHARE") orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch error.BadQwenKvShare;
}

/// A stream the store keeps from or restores into: its lane's cache, or (before the lane opens) the state to restore.
pub const Owner = struct { cache: ?*Cache = null, pending: ?radix.Saved = null };

/// A kept recurrent state: the cache's state block (recurrences and convolution tails) in pinned host memory.
const Snap = struct { host: cuda.HostBuffer };

pub const Prefix = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    pages: *pages_mod.Pages,
    store: radix.Store,

    pub fn init(gpa: std.mem.Allocator, e: *Engine, p: *pages_mod.Pages, snaps: usize) !*Prefix {
        const x = try gpa.create(Prefix);
        errdefer gpa.destroy(x);
        const rules: radix.Rules = .{ .page = p.layout.page_tokens, .min_gap = 256, .min_prompt = 0, .tail = true };
        const pool_pages = radix.pagesOf(kv.Pool, &p.pool, p.layout.pageBytes());
        x.* = .{ .gpa = gpa, .e = e, .pages = p, .store = try radix.Store.init(gpa, .{ .ptr = x, .vtable = &family }, pool_pages, rules) };
        x.store.limit(.{ .pages = p.pool.total(), .snaps = snaps });
        p.pool.reclaim = .{ .ctx = x, .call = reclaim }; // a promise evicts the store's idle pages before it fails
        return x;
    }

    pub fn deinit(x: *Prefix) void {
        x.pages.pool.reclaim = null;
        x.store.deinit(); // drops every snapshot and releases the tree's pages
        x.gpa.destroy(x);
    }

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const x: *Prefix = @ptrCast(@alignCast(ctx));
        return x.store.reclaim(want);
    }

    /// Before a pass: the deepest kept boundary (pages in `adopt`, state in `owner.pending`) and the marks to keep.
    pub fn begin(x: *Prefix, prompt: []const u32, history_len: u32, shared: []const u32, owner: *Owner, adopt: *std.ArrayList(u32)) !radix.Plan {
        return x.store.begin(x.gpa, prompt, history_len, shared, owner, adopt);
    }

    /// The pending state copied into the stream's cache (its lane open and its pages adopted).
    pub fn put(x: *Prefix, owner: *Owner) !void {
        const saved = owner.pending orelse return;
        const s: *Snap = @ptrCast(@alignCast(saved));
        const cache = owner.cache.?;
        const e = x.e;
        try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(cache.state.ptr, s.host.bytes.ptr, e.g.stateBytes(), e.stream.handle), "cuMemcpyHtoDAsync");
        try e.stream.synchronize();
    }

    /// At page boundary `at` the tree takes the pages it lacks and the state; ones it held replace the stream's own.
    pub fn keep(x: *Prefix, cache: *Cache, prompt: []const u32, at: usize) !void {
        if (!x.store.on()) return;
        try x.e.stream.synchronize(); // the rows below `at` are written before the tree holds them
        const n = at / x.pages.layout.page_tokens;
        const path = try x.gpa.alloc(u32, n);
        defer x.gpa.free(path);
        const mine = try x.gpa.dupe(u32, cache.paged.?.ids[0..n]);
        defer x.gpa.free(mine);
        var owner: Owner = .{ .cache = cache };
        const kept = x.store.keep(prompt, @intCast(at), &owner, mine, path);
        try x.pages.swap(cache, path[0..kept.shared]);
    }

    /// The deepest kept boundary of the prompt `ids` and a pool lane holding its pages; the lane's state put back.
    pub fn open(x: *Prefix, ids: []const u32, max_new: usize, history_len: u32, shared: []const u32) !struct { cache: *Cache, plan: radix.Plan } {
        var owner: Owner = .{};
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(x.gpa);
        var plan = try x.store.begin(x.gpa, ids, history_len, shared, &owner, &adopt);
        plan.marks = try x.onGrid(plan);
        errdefer x.gpa.free(plan.marks);
        // released if the lane cannot open: the request waits and asks again
        owner.cache = try x.pages.openWith(ids.len, max_new, adopt.items);
        errdefer _ = x.pages.close(owner.cache.?);
        try x.put(&owner);
        return .{ .cache = owner.cache.?, .plan = plan };
    }

    /// The plan's marks down to the cold pass's chunk grid, where a pass cuts and so can keep; the plan's freed.
    fn onGrid(x: *Prefix, plan: radix.Plan) ![]const u32 {
        defer x.gpa.free(plan.marks);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(x.gpa);
        for (plan.marks) |m| {
            const g = m - m % engine.max_rows;
            if (g > plan.from and (out.items.len == 0 or out.items[out.items.len - 1] != g)) try out.append(x.gpa, g);
        }
        return out.toOwnedSlice(x.gpa);
    }

    fn of(ptr: *anyopaque) *Prefix {
        return @ptrCast(@alignCast(ptr));
    }

    fn snapBytes(ptr: *anyopaque, _: u32) u64 {
        return of(ptr).e.g.stateBytes();
    }

    fn save(ptr: *anyopaque, owner: ?*anyopaque, _: u32) anyerror!radix.Saved {
        const x = of(ptr);
        const o: *Owner = @ptrCast(@alignCast(owner orelse return error.NoQwenOwner));
        const cache = o.cache orelse return error.NoQwenOwner;
        const e = x.e;
        const s = try x.gpa.create(Snap);
        errdefer x.gpa.destroy(s);
        s.* = .{ .host = try cuda.HostBuffer.alloc(e.d, e.g.stateBytes()) };
        errdefer s.host.free();
        try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(s.host.bytes.ptr, cache.state.ptr, e.g.stateBytes(), e.stream.handle), "cuMemcpyDtoHAsync");
        try e.stream.synchronize();
        return @ptrCast(s);
    }

    fn restore(_: *anyopaque, owner: ?*anyopaque, saved: radix.Saved) anyerror!void {
        const o: *Owner = @ptrCast(@alignCast(owner orelse return error.NoQwenOwner));
        o.pending = saved; // copied once the stream's lane is open (put)
    }

    fn drop(ptr: *anyopaque, saved: radix.Saved) void {
        const x = of(ptr);
        const s: *Snap = @ptrCast(@alignCast(saved));
        s.host.free();
        x.gpa.destroy(s);
    }

    const family: radix.Snapshots.VTable = .{ .bytes = snapBytes, .save = save, .restore = restore, .drop = drop };
};
