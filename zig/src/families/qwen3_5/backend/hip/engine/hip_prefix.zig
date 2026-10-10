//! The prompt cache's HIP side: pages and snapshots for core/prompt_radix.zig; rank 0 decides, other ranks apply.

const std = @import("std");
const hip = @import("hip");
const radix = @import("engine_api").prompt_radix;
const memory = @import("memory.zig");
const worker = @import("worker.zig");
const state = @import("../forward/state.zig");
const pages = @import("../forward/pages.zig");
const Engine = @import("engine.zig").Engine;
const Allocator = std.mem.Allocator;

/// What a stream gives the prompt cache: its caches and the id the other ranks know it by.
pub const Owner = struct { caches: *state.Caches, id: u32 };

pub const Prefix = struct {
    gpa: Allocator,
    e: *Engine,
    store: radix.Store,
    snaps: state.Snapshots,
    /// Snapshot slots in use.
    used: std.ArrayList(bool) = .empty,
    /// The other ranks, when this is rank 0 of a group.
    link: ?*const hip.link.Link = null,
    msg: std.ArrayList(u32) = .empty,
    /// The slot the last restore copied from (reset as a pass begins): the message that begins the pass names it.
    restored: u32 = worker.no_snapshot,

    pub fn init(gpa: Allocator, e: *Engine) !*Prefix {
        const p = try gpa.create(Prefix);
        errdefer gpa.destroy(p);
        const rules: radix.Rules = .{ .page = pages.tokens, .min_gap = @import("prefix.zig").min_gap, .min_prompt = 0, .tail = true };
        p.* = .{ .gpa = gpa, .e = e, .store = try radix.Store.init(gpa, .{ .ptr = p, .vtable = &family }, .{ .ptr = &e.pool, .vtable = &pool_pages, .bytes = e.pool.pageBytes() }, rules), .snaps = state.Snapshots.init(gpa, &e.driver) };
        return p;
    }

    pub fn deinit(p: *Prefix) void {
        p.store.deinit();
        p.snaps.deinit();
        p.used.deinit(p.gpa);
        p.msg.deinit(p.gpa);
        p.gpa.destroy(p);
    }

    /// Keep up to `slots` snapshots and the pages the rest of `budget` bytes buys, a budget the memory plan granted.
    pub fn keepPrompts(p: *Prefix, slots: usize, budget: usize) void {
        const m = p.e.model();
        const room = memory.cache(m.spec, m.act.size(), budget, slots);
        p.keepPages(@min(room.pages, p.e.pool.count), room.snaps);
    }

    /// The tree may hold `max_pages` pages and `snaps` snapshots.
    pub fn keepPages(p: *Prefix, max_pages: usize, snaps: usize) void {
        p.store.limit(.{ .pages = max_pages, .snaps = snaps });
        if (p.store.tree.snaps == 0) p.snaps.clear();
    }

    pub fn send(p: *Prefix, words: []const u32) !void {
        if (p.link) |l| try l.send(words);
    }

    /// Tells the other ranks that the stream's pages from page `first` on are `ids`.
    pub fn sendPages(p: *Prefix, id: u32, first: usize, ids: []const u32) !void {
        if (p.link == null) return;
        p.msg.clearRetainingCapacity();
        try p.msg.appendSlice(p.gpa, &.{ @backingInt(worker.Op.pages), id, @intCast(first), @intCast(ids.len) });
        try p.msg.appendSlice(p.gpa, ids);
        try p.send(p.msg.items);
    }

    /// Before a prompt pass: the longest resumable state restored into the stream, its pages in `adopt`, the marks.
    pub fn begin(p: *Prefix, prompt: []const u32, history_len: u32, shared: []const u32, owner: *Owner, adopt: *std.ArrayList(u32)) !radix.Plan {
        return p.store.begin(p.gpa, prompt, history_len, shared, owner, adopt);
    }

    /// At `at` the tree takes the pages it lacks and the linear state; the stream swaps in the pages the tree had.
    pub fn keep(p: *Prefix, owner: *Owner, prompt: []const u32, at: usize) !void {
        if (!p.store.on()) return;
        const n = at / pages.tokens;
        const path = try p.gpa.alloc(u32, n);
        defer p.gpa.free(path);
        const caches = owner.caches;
        const kept = p.store.keep(prompt, @intCast(at), owner, caches.table.items[0..n], path);
        const own = path[0..kept.shared];
        if (!std.mem.eql(u32, own, caches.table.items[0..kept.shared])) {
            for (own) |id| p.e.pool.retain(id);
            try caches.set(p.gpa, 0, own);
            try p.sendPages(owner.id, 0, own);
        }
    }

    fn slotOf(saved: radix.Saved) u32 {
        return @intCast(@intFromPtr(saved) - 1);
    }

    fn of(ptr: *anyopaque) *Prefix {
        return @ptrCast(@alignCast(ptr));
    }

    fn snapBytes(ptr: *anyopaque, _: u32) u64 {
        return memory.linearBytes(of(ptr).e.model().spec);
    }

    fn save(ptr: *anyopaque, owner: ?*anyopaque, _: u32) anyerror!radix.Saved {
        const p = of(ptr);
        const o: *Owner = @ptrCast(@alignCast(owner.?));
        const slot: u32 = for (p.used.items, 0..) |u, i| {
            if (!u) break @intCast(i);
        } else blk: {
            try p.used.append(p.gpa, false);
            break :blk @intCast(p.used.items.len - 1);
        };
        try p.snaps.take(slot, o.caches, p.e.stream.handle);
        p.used.items[slot] = true;
        try p.send(&.{ @backingInt(worker.Op.snap), o.id, slot });
        return @ptrFromInt(@as(usize, slot) + 1);
    }

    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: radix.Saved) anyerror!void {
        const p = of(ptr);
        const o: *Owner = @ptrCast(@alignCast(owner.?));
        try p.snaps.put(slotOf(saved), o.caches, p.e.stream.handle);
        p.restored = slotOf(saved);
    }

    fn dropSaved(ptr: *anyopaque, saved: radix.Saved) void {
        of(ptr).used.items[slotOf(saved)] = false;
    }

    const family: radix.Snapshots.VTable = .{ .bytes = snapBytes, .save = save, .restore = restore, .drop = dropSaved };

    fn pool(ptr: *anyopaque) *pages.Pool {
        return @ptrCast(@alignCast(ptr));
    }

    fn retain(ptr: *anyopaque, id: u32) void {
        pool(ptr).retain(id);
    }

    fn release(ptr: *anyopaque, id: u32) void {
        pool(ptr).release(id);
    }

    fn holders(ptr: *anyopaque, id: u32) u32 {
        return pool(ptr).ids.refs[id];
    }

    fn available(ptr: *anyopaque) usize {
        return pool(ptr).ids.available();
    }

    const pool_pages: radix.Pages.VTable = .{ .retain = retain, .release = release, .holders = holders, .available = available };
};
