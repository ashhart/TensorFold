//! GLM-5.3-Flash's KV pages on CUDA: the core pool's pages in one device arena, and the paged sequences that map them.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const fw = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const types = @import("cuda_engine_types.zig");
const Engine = @import("cuda_engine.zig").Engine;

/// Table entries a sequence of `capacity` rows takes.
pub fn tableLen(capacity: usize) usize {
    return (capacity + state.page_rows - 1) / state.page_rows;
}

/// What a stream of engine `e` holds: its fixed part and table (and `extra`, its other buffers), 13,056 bytes a token.
pub fn layout(gpa: std.mem.Allocator, e: *const Engine, extra: u64) !kv.Layout {
    const fixed = try state.State.pagedBytes(gpa, &e.w, e.capacity, tableLen(e.capacity));
    const page = state.pageBytes(&e.w);
    return .{ .fixed_bytes = fixed + extra, .token_bytes = page / state.page_rows, .page_tokens = state.page_rows, .snap_bytes = 0 };
}

/// A request's need: its prompt, its reply and one last verify window.
pub fn need(l: kv.Layout, prompt: usize, max_tokens: usize) usize {
    return l.need(prompt, max_tokens, fw.window_rows);
}

pub const Pages = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    layout: kv.Layout,
    pool: kv.Pool, // rank 0 takes and gives back pages; the other ranks map the ids it sends
    arena: ?cuda.DeviceBuffer,
    leases: []kv.Lease, // paged slot 1 + i's: rank 0's lease, or the ids another rank mapped
    used: []bool,
    leader: bool,
    poison: bool, // TF_KV_POISON=1: a stream's pages start as NaN, so a row read before its write shows in the reply

    /// `streams` paged sequences (slots 1..streams, beside the engine's own slot 0) and `pages` pages, all made now.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, l: kv.Layout, streams: usize, pages: usize, leader: bool) !*Pages {
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        const page = l.pageBytes();
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .pool = try kv.Pool.init(gpa, pages, page), .arena = null, .leases = &.{}, .used = &.{}, .leader = leader, .poison = poisoned() };
        errdefer p.pool.deinit();
        if (pages > 0) p.arena = try cuda.DeviceBuffer.alloc(e.d, pages * page);
        errdefer if (p.arena) |*a| a.free();
        p.leases = try gpa.alloc(kv.Lease, streams);
        errdefer gpa.free(p.leases);
        @memset(p.leases, .{});
        p.used = try gpa.alloc(bool, streams);
        errdefer gpa.free(p.used);
        @memset(p.used, false);
        if (e.seqs.items.len < streams + 1) try e.seqs.appendNTimes(e.gpa, null, streams + 1 - e.seqs.items.len);
        const seqs = e.seqs.items[1..][0..streams];
        var made: usize = 0;
        errdefer for (seqs[0..made]) |*q| {
            q.*.?.st.deinit();
            e.gpa.destroy(q.*.?);
            q.* = null;
        };
        const at: state.Paged = .{ .kv = if (p.arena) |a| a.ptr else 0, .pages = tableLen(e.capacity), .page_bytes = page };
        for (seqs) |*q| {
            if (q.* != null) return error.SlotTaken; // attach comes before any stream
            const s = try e.gpa.create(types.Seq);
            errdefer e.gpa.destroy(s);
            s.* = .{ .st = try fw.State.initPaged(e.gpa, e.d, &e.w, e.capacity, at) };
            q.* = s;
            made += 1;
        }
        try e.stream.synchronize();
        return p;
    }

    /// Another rank: the pool rank 0's `pool` step names; a page of other bytes than rank 0's is error.RankMismatch.
    pub fn follow(gpa: std.mem.Allocator, e: *Engine, streams: usize, pages: usize, page_bytes: u64) !*Pages {
        const l = try layout(gpa, e, 0);
        if (l.pageBytes() != page_bytes) {
            if (!@import("builtin").is_test) std.log.err("rank {d}: a KV page is {d} bytes here and {d} on rank 0", .{ e.share.rank, l.pageBytes(), page_bytes });
            return error.RankMismatch;
        }
        return init(gpa, e, l, streams, pages, false);
    }

    fn poisoned() bool {
        const v = std.c.getenv("TF_KV_POISON") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// The pool and the pages' arena; the paged sequences are the engine's (it frees every slot).
    pub fn deinit(p: *Pages) void {
        p.e.stream.synchronize() catch {};
        for (p.leases, p.used) |*l, u| {
            if (p.leader and u) p.pool.finish(l) else l.ids.deinit(p.gpa);
        }
        p.gpa.free(p.leases);
        p.gpa.free(p.used);
        if (p.arena) |*a| a.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    /// Whether `slot` is one of the paged sequences.
    pub fn paged(p: *const Pages, slot: u32) bool {
        return slot >= 1 and slot <= p.used.len;
    }

    /// Whether a request's pages fit now and a paged sequence is free for it.
    pub fn fits(p: *const Pages, prompt: usize, max_tokens: usize) bool {
        for (p.used) |u| if (!u) return p.pool.fits(need(p.layout, prompt, max_tokens));
        return false;
    }

    /// Rank 0: paged slot `slot` takes the pages of a prompt, its reply and a last window; LaneMemory: not yet.
    pub fn open(p: *Pages, slot: u32, prompt: usize, max_tokens: usize) ![]const u32 {
        if (!p.paged(slot) or p.used[slot - 1]) return error.SlotTaken;
        const n = need(p.layout, prompt, max_tokens);
        if (n > tableLen(p.e.capacity)) return error.PromptTooLong;
        const l = &p.leases[slot - 1];
        try p.pool.promise(l, n);
        errdefer p.pool.finish(l);
        var up: Upload = .{ .p = p, .slot = slot };
        try p.pool.cover(l, n, .{ .ctx = &up, .write = Upload.write });
        try p.mapped(slot, l.ids.items);
        return l.ids.items;
    }

    /// Another rank: paged slot `slot` maps rank 0's page ids (sent with the stream's begin step).
    pub fn map(p: *Pages, slot: u32, ids: []const u32) !void {
        if (!p.paged(slot) or ids.len == 0 or ids.len > tableLen(p.e.capacity)) return error.BadStep;
        for (ids) |id| if (id >= p.pool.total()) return error.BadStep;
        const l = &p.leases[slot - 1];
        l.ids.clearRetainingCapacity();
        try l.ids.appendSlice(p.gpa, ids);
        var up: Upload = .{ .p = p, .slot = slot };
        try Upload.write(&up, l.ids.items, 0);
        try p.mapped(slot, l.ids.items);
    }

    /// The slot's sequence reaches its pages' rows (NaN first under TF_KV_POISON).
    fn mapped(p: *Pages, slot: u32, ids: []const u32) !void {
        const st = &p.e.seqs.items[slot].?.st;
        const page = p.layout.pageBytes();
        if (p.poison) for (ids) |id| try p.e.d.check(p.e.d.api.cuMemsetD32Async(p.arena.?.ptr + id * page, 0xFFFF_FFFF, page / 4, p.e.stream.handle), "cuMemsetD32Async");
        st.pages = ids;
        st.capacity = @min(ids.len * state.page_rows, p.e.capacity);
        p.used[slot - 1] = true;
    }

    /// The stream on paged slot `slot` ended: its sequence reaches no rows, and rank 0's lease gives its pages back.
    pub fn close(p: *Pages, slot: u32) void {
        if (!p.paged(slot) or !p.used[slot - 1]) return;
        const st = &p.e.seqs.items[slot].?.st;
        st.pages = &.{};
        st.capacity = 0;
        p.used[slot - 1] = false;
        if (!p.leader) return p.leases[slot - 1].ids.clearRetainingCapacity();
        p.pool.finish(&p.leases[slot - 1]);
        for (p.used) |u| if (u) return;
        p.report();
    }

    /// The idle server's line: nothing promised, every page free (the copy store holds none).
    fn report(p: *const Pages) void {
        if (@import("builtin").is_test) return;
        const c = p.pool.census();
        const ok = p.pool.idle() and c.free == c.total;
        std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total });
    }

    /// A stream's table entries written to its sequence's device table (blocking: the stream waits for the copy).
    const Upload = struct {
        p: *Pages,
        slot: u32,

        fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
            const u: *Upload = @ptrCast(@alignCast(ctx));
            const e = u.p.e;
            const st = &e.seqs.items[u.slot].?.st;
            const bytes = std.mem.sliceAsBytes(ids[from..]);
            try e.d.check(e.d.api.cuMemcpyHtoD_v2(st.table + from * 4, bytes.ptr, bytes.len), "cuMemcpyHtoD");
        }
    };
};
