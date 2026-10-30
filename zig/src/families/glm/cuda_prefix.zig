//! TF_GLM_KV_SHARE: a prompt cache over the pool's KV pages (prompt_radix.zig); a resume adopts them and the rest.
const std = @import("std");
pub const radix = @import("engine_api").prompt_radix;
const kv = @import("core").kv_pool;
const engine = @import("cuda_engine.zig");
const pages_mod = @import("cuda_pages.zig");
const page_rows = @import("cuda_state.zig").page_rows;

const Engine = engine.Engine;

/// TF_GLM_KV_SHARE=n: keep prompts as pool pages and at most n states past them (0 or unset: off).
pub fn sharing() !usize {
    const v = std.c.getenv("TF_GLM_KV_SHARE") orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch error.BadGlmKvShare;
}

/// Below 4 states a stream, live streams evict each other's before a resume reaches them.
pub fn fewStates(states: usize, parallel: usize) bool {
    return states > 0 and states < 4 * parallel;
}

test "TF_GLM_KV_SHARE below 4 states a stream is too few; off is not" {
    try std.testing.expect(fewStates(4, 4));
    try std.testing.expect(fewStates(15, 4));
    try std.testing.expect(!fewStates(16, 4));
    try std.testing.expect(!fewStates(0, 4));
}

/// The store's rules: 512-row pages, states on the 2,048-row prompt grid, the MTP head's row at-1 keyed by token at.
pub fn rules(grid: u32) radix.Rules {
    return .{ .page = page_rows, .planned = true, .grid = grid, .lookahead = 1, .min_prompt = 0, .tail = true };
}

/// A resume's state: restored by the prompt pass's start on every rank (engine.Resume), once its pages are mapped.
pub const Owner = struct { pending: ?*engine.Kept = null };

/// A pass's marks (the engine's Marks.ctx): the mark it stands at, or the last it completed when cancelled, keeps.
pub const Keeps = struct {
    x: *Prefix,
    slot: u32,
    prompt: []const u32,

    pub fn at(ctx: *anyopaque, mark: u32) void {
        const k: *Keeps = @ptrCast(@alignCast(ctx));
        k.x.keep(k.slot, k.prompt, mark);
    }
};

pub const Prefix = struct {
    gpa: std.mem.Allocator,
    pages: *pages_mod.Pages,
    store: radix.Store,

    /// The store over `p`'s pool, keeping `e`'s states past the pages it holds (`e.share_pages`).
    pub fn init(gpa: std.mem.Allocator, e: *Engine, p: *pages_mod.Pages, grid: u32, snaps: usize) !*Prefix {
        return initWith(gpa, p, grid, snaps, .{ .ptr = e, .vtable = &family });
    }

    pub fn initWith(gpa: std.mem.Allocator, p: *pages_mod.Pages, grid: u32, snaps: usize, states: radix.Snapshots) !*Prefix {
        const x = try gpa.create(Prefix);
        errdefer gpa.destroy(x);
        const pool_pages = radix.pagesOf(kv.Pool, &p.pool, p.layout.pageBytes());
        x.* = .{ .gpa = gpa, .pages = p, .store = try radix.Store.init(gpa, states, pool_pages, rules(grid)) };
        x.store.limit(.{ .pages = p.pool.total(), .snaps = snaps });
        p.pool.reclaim = .{ .ctx = x, .call = reclaim }; // a promise evicts the store's idle pages before it fails
        return x;
    }

    pub fn deinit(x: *Prefix) void {
        x.pages.pool.reclaim = null;
        x.store.deinit(); // drops every state and releases the tree's pages, before the pool goes
        x.gpa.destroy(x);
    }

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const x: *Prefix = @ptrCast(@alignCast(ctx));
        return x.store.reclaim(want);
    }

    /// Before a prompt pass: the deepest kept state's pages in `adopt` (held for the stream), the state in `owner`.
    pub fn begin(x: *Prefix, prompt: []const u32, history_len: u32, shared: []const u32, owner: *Owner, adopt: *std.ArrayList(u32)) !radix.Plan {
        return x.store.begin(x.gpa, prompt, history_len, shared, owner, adopt);
    }

    /// Slot `slot`'s pass at mark `at`: the tree takes the pages below it it lacks, and the state (no table swaps).
    pub fn keep(x: *Prefix, slot: u32, prompt: []const u32, at: u32) void {
        if (!x.store.on() or !x.pages.paged(slot)) return;
        const n = at / page_rows;
        const ids = x.pages.leases[slot].ids.items;
        if (n > ids.len) return;
        const path = x.gpa.alloc(u32, n) catch return;
        defer x.gpa.free(path);
        var owner: Owner = .{};
        _ = x.store.keep(prompt, at, &owner, ids[0..n], path);
    }

    fn of(ptr: *anyopaque) *Engine {
        return @ptrCast(@alignCast(ptr));
    }

    fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return of(ptr).stateBytes(at);
    }

    /// The bound slot's state at `at` (a mark at a chunk start), its pages below `at` left to the tree.
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!radix.Saved {
        const e = of(ptr);
        if (e.pos() != at) return error.NotAtMark;
        const k = try e.gpa.create(engine.Kept);
        errdefer e.gpa.destroy(k);
        k.* = .{ .id = try e.keepState(), .at = at };
        return @ptrCast(k);
    }

    /// The store's restore: the state waits in `owner` for the pass's start (Resume), its pages for the stream's lease.
    pub fn restore(_: *anyopaque, owner: ?*anyopaque, saved: radix.Saved) anyerror!void {
        const o: *Owner = @ptrCast(@alignCast(owner orelse return error.NoGlmOwner));
        o.pending = @ptrCast(@alignCast(saved));
    }

    fn drop(ptr: *anyopaque, saved: radix.Saved) void {
        const e = of(ptr);
        const k: *engine.Kept = @ptrCast(@alignCast(saved));
        e.dropState(k.id);
        e.gpa.destroy(k);
    }

    const family: radix.Snapshots.VTable = .{ .bytes = snapBytes, .save = save, .restore = restore, .drop = drop };
};
