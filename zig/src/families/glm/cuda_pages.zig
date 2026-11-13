//! GLM-5.3-Flash's KV pages on CUDA: the core pool's pages in one device arena, and the paged sequences that map them.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const kp = @import("core").kv_prompts;
const fw = @import("cuda_forward.zig");
const state = @import("cuda_state.zig");
const types = @import("cuda_engine_types.zig");
const Engine = @import("cuda_engine.zig").Engine;
const kv_host = @import("cuda_kv_host.zig");

/// Table entries a sequence of `capacity` rows takes.
pub fn tableLen(capacity: usize) usize {
    return (capacity + state.page_rows - 1) / state.page_rows;
}

/// What a stream of engine `e` holds: its fixed part and table (and `extra`, its other buffers), 13,056 bytes a token.
pub fn layout(gpa: std.mem.Allocator, e: *const Engine, extra: u64) !kv.Layout {
    const fixed = try state.State.pagedBytes(gpa, &e.w, e.capacity, tableLen(e.capacity), e.settings.kv_split);
    // TF_GLM_KV_HOST: a page costs the device only its pooled keys (the pages themselves are in host memory)
    const page = if (kv_host.onHost()) state.pkPageBytes(&e.w) else state.pageBytes(&e.w);
    const classes: u32 = if (e.settings.kv_split) @intCast(e.share.world) else 1; // TF_GLM_KV_SPLIT: an owner class a rank
    return .{ .fixed_bytes = fixed + extra, .token_bytes = page / state.page_rows, .page_tokens = state.page_rows, .snap_bytes = 0, .classes = classes };
}

/// TF_KV_SHARE: a kept state's slot (`e.share_pages`: its prefix's pages left to the store), at the window's last mark.
pub fn shareSlot(e: *const Engine) u64 {
    return e.stateBytes(e.capacity - e.capacity % @import("cuda_native.zig").snap_grid);
}

/// A request's need: its prompt, its reply and one last verify window.
pub fn need(l: kv.Layout, prompt: usize, max_tokens: usize) usize {
    return l.need(prompt, max_tokens, fw.window_rows);
}

/// The engine's own sequence (slot 0) when it pages, bound: its fixed part and table, no rows until its pool places it.
pub fn ownSequence(e: *Engine) !void {
    if (e.seqs.items.len == 0) try e.seqs.append(e.gpa, null);
    if (e.seqs.items[0] != null) return error.SlotTaken;
    const q = try e.gpa.create(types.Seq);
    errdefer e.gpa.destroy(q);
    q.* = .{ .st = try fw.State.initPaged(e.gpa, e.d, &e.w, e.capacity, on(e, 0, 0)) };
    e.seqs.items[0] = q;
    e.cur = q;
    e.slot = 0;
}

/// A sequence's place on `n` pages at `base` (TF_GLM_KV_SPLIT: every pooled key first, then every latent row).
fn on(e: *const Engine, base: u64, n: usize) state.Paged {
    const at: state.Paged = .{ .kv = base, .pages = tableLen(e.capacity), .page_bytes = state.pageBytes(&e.w) };
    if (!e.settings.kv_split) return at;
    const pooled = state.pkPageBytes(&e.w);
    return .{ .kv = base + n * pooled, .pages = at.pages, .page_bytes = state.latentPageBytes(&e.w), .pk = base, .pk_page = pooled, .split = true };
}

pub const Pages = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    layout: kv.Layout,
    page: u64, // a page's bytes in its arena (TF_GLM_KV_HOST: the whole page, though the layout counts its keys)
    pool: kv.Pool, // rank 0 takes and gives back pages; the other ranks map the ids it sends
    arena: ?cuda.DeviceBuffer,
    host: ?cuda.HostBuffer = null, // TF_GLM_KV_HOST: the pages in pinned host memory instead of `arena`
    shared: ?kv_host.SharedPages = null, // TF_GLM_KV_HOST_SHARED: the same, one copy for the ranks on this host
    host_ptr: u64 = 0,
    pk_arena: ?cuda.DeviceBuffer = null, // TF_GLM_KV_HOST: the pooled keys' device pages
    leases: []kv.Lease, // slot i's: rank 0's lease, or the ids another rank mapped
    used: []bool,
    leader: bool,
    poison: bool, // TF_KV_POISON=1: a stream's pages start as NaN, so a row read before its write shows in the reply
    prompts: ?kp.Prompts = null, // TF_KV_SHARE: the shared prompt store over this pool (rank 0's), from the server

    /// `streams` sequences from slot 0 on `pages` pages: the engine's own (`ownSequence`) placed, the others made now.
    pub fn init(gpa: std.mem.Allocator, e: *Engine, l: kv.Layout, streams: usize, pages: usize, leader: bool) !*Pages {
        const own = if (e.seqs.items.len > 0) e.seqs.items[0] else null;
        if (own == null or own.?.st.table == 0) return error.OwnNotPaged; // slot 0 holds a window, or none was made
        if (l.classes > 1 and pages / l.classes < tableLen(e.capacity)) { // a stream's pages all come from one class
            if (!@import("builtin").is_test) std.log.err("the pool's {d} pages a class hold less than one stream's {d}", .{ pages / l.classes, tableLen(e.capacity) });
            return error.ClassTooSmall;
        }
        const p = try gpa.create(Pages);
        errdefer gpa.destroy(p);
        const page = if (kv_host.onHost()) state.pageBytes(&e.w) else l.pageBytes();
        p.* = .{ .gpa = gpa, .e = e, .layout = l, .page = page, .pool = try kv.Pool.initClasses(gpa, pages, page, l.classes), .arena = null, .leases = &.{}, .used = &.{}, .leader = leader, .poison = poisoned() };
        errdefer p.pool.deinit();
        const host = kv_host.onHost() and pages > 0;
        if (pages > 0 and !host) p.arena = try cuda.DeviceBuffer.alloc(e.d, pages * page);
        errdefer if (p.arena) |*a| a.free();
        if (host) {
            if (kv_host.shared()) {
                p.shared = try kv_host.SharedPages.open(e.d, pages * page, leader, e.share.world);
                p.host_ptr = p.shared.?.dev;
            } else {
                p.host = try cuda.HostBuffer.allocMapped(e.d, pages * page);
                p.host_ptr = try p.host.?.device();
            }
            p.pk_arena = try cuda.DeviceBuffer.alloc(e.d, pages * state.pkPageBytes(&e.w));
            try kv_host.prepare(e.d, page, e.capacity, e.w.config.kv_lora * 2); // made before the room was read
            std.log.info("kv host: {d} pages ({d} MiB) in pinned host memory, pooled keys {d} MiB on the device", .{ pages, pages * page >> 20, pages * state.pkPageBytes(&e.w) >> 20 });
        }
        errdefer if (p.host) |*h| h.free();
        errdefer if (p.shared) |*h| h.close();
        errdefer if (p.pk_arena) |*a| a.free();
        p.leases = try gpa.alloc(kv.Lease, streams);
        errdefer gpa.free(p.leases);
        @memset(p.leases, .{});
        p.used = try gpa.alloc(bool, streams);
        errdefer gpa.free(p.used);
        @memset(p.used, false);
        if (e.seqs.items.len < streams) try e.seqs.appendNTimes(e.gpa, null, streams - e.seqs.items.len);
        const seqs = e.seqs.items[1..@max(streams, 1)];
        var made: usize = 0;
        errdefer for (seqs[0..made]) |*q| {
            q.*.?.st.deinit();
            e.gpa.destroy(q.*.?);
            q.* = null;
        };
        errdefer cuda.completion.drain(e.d, e.stream, "GLM page construction did not drain");
        var at = on(e, p.base(), pages);
        if (p.pk_arena) |a| {
            at.pk = a.ptr;
            at.pk_page = state.pkPageBytes(&e.w);
        }
        for (seqs) |*q| {
            if (q.* != null) return error.SlotTaken; // attach comes before any stream
            const s = try e.gpa.create(types.Seq);
            errdefer e.gpa.destroy(s);
            s.* = .{ .st = try fw.State.initPaged(e.gpa, e.d, &e.w, e.capacity, at) };
            q.* = s;
            made += 1;
        }
        try e.stream.synchronize();
        own.?.st.place(&e.w, at); // last: a failure before it leaves the engine's own as it was
        if (l.classes > 1 and !@import("builtin").is_test) std.log.info("kv split: {d} owner classes of {d} pages; a stream takes at most one class ({d} pages at the window); every rank still holds every latent row, so this saves no memory yet", .{ l.classes, pages / l.classes, tableLen(e.capacity) });
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

    /// `follow` with rank 0's TF_GLM_KV_SPLIT (its `pool` step's flag): a rank that books otherwise is RankMismatch.
    pub fn followFlagged(gpa: std.mem.Allocator, e: *Engine, streams: usize, pages: usize, page_bytes: u64, split: bool) !*Pages {
        if (e.settings.kv_split != split) {
            if (!@import("builtin").is_test) std.log.err("rank {d}: TF_GLM_KV_SPLIT is {} here and {} on rank 0", .{ e.share.rank, e.settings.kv_split, split });
            return error.RankMismatch;
        }
        return follow(gpa, e, streams, pages, page_bytes);
    }

    /// The pages' first byte as kernels address it (device arena or mapped host pages).
    fn base(p: *const Pages) u64 {
        return if (p.arena) |a| a.ptr else p.host_ptr;
    }

    fn poisoned() bool {
        const v = std.c.getenv("TF_KV_POISON") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    /// A page's parts, each at its region's base and stride: the whole page, or its pooled keys and latent rows.
    fn parts(p: *const Pages) [2]struct { at: u64, bytes: u64 } {
        if (!p.e.settings.kv_split) return .{ .{ .at = p.base(), .bytes = p.page }, .{ .at = 0, .bytes = 0 } };
        const pooled = state.pkPageBytes(&p.e.w);
        return .{ .{ .at = p.base(), .bytes = pooled }, .{ .at = p.base() + p.pool.total() * pooled, .bytes = p.page - pooled } };
    }

    /// The pool and the pages' arena; the paged sequences are the engine's (it frees every slot).
    pub fn deinit(p: *Pages) void {
        cuda.completion.drain(p.e.d, p.e.stream, "GLM page ownership did not drain");
        for (p.leases, p.used) |*l, u| {
            if (p.leader and u) p.pool.finish(l) else l.ids.deinit(p.gpa);
        }
        p.gpa.free(p.leases);
        p.gpa.free(p.used);
        if (p.arena) |*a| a.free();
        if (p.host) |*h| h.free();
        if (p.shared) |*h| h.close();
        if (p.pk_arena) |*a| a.free();
        p.pool.deinit();
        p.gpa.destroy(p);
    }

    /// Whether `slot` is one of the paged sequences.
    pub fn paged(p: *const Pages, slot: u32) bool {
        return slot < p.used.len;
    }

    /// Whether a request's pages fit now and a paged sequence is free for it.
    pub fn fits(p: *const Pages, prompt: usize, max_tokens: usize) bool {
        const n = need(p.layout, prompt, max_tokens);
        // past a class, `open` refuses it by name: the lane host drops it, not waits for room that never comes
        for (p.used) |u| if (!u) return p.pool.fitsAny(n) or (p.pool.classes > 1 and n > p.pool.classCap());
        return false;
    }

    /// Rank 0: paged slot `slot` takes the pages of a prompt, its reply and a last window; LaneMemory: not yet.
    pub fn open(p: *Pages, slot: u32, prompt: usize, max_tokens: usize) ![]const u32 {
        return p.openWith(slot, prompt, max_tokens, &.{});
    }

    /// The prompt cache's pages a resume held, given back when no lease took them.
    pub fn give(p: *Pages, adopted: []const u32) void {
        for (adopted) |id| p.pool.release(id);
    }

    /// `open` with the prompt cache's `adopted` pages (held for this call) first: their rows stay as the cache kept.
    pub fn openWith(p: *Pages, slot: u32, prompt: usize, max_tokens: usize, adopted: []const u32) ![]const u32 {
        return p.openOwned(slot, prompt, max_tokens, adopted, p.pool.roomiest());
    }

    /// `open` on owner class `owner` (the harness pins it; serving takes the roomiest).
    pub fn openOn(p: *Pages, slot: u32, prompt: usize, max_tokens: usize, owner: u32) ![]const u32 {
        return p.openOwned(slot, prompt, max_tokens, &.{}, owner);
    }

    fn openOwned(p: *Pages, slot: u32, prompt: usize, max_tokens: usize, adopted: []const u32, owner: u32) ![]const u32 {
        var held = adopted.len > 0;
        errdefer if (held) for (adopted) |id| p.pool.release(id);
        if (!p.paged(slot) or p.used[slot]) return error.SlotTaken;
        const n = need(p.layout, prompt, max_tokens);
        if (n > tableLen(p.e.capacity) or adopted.len > n) return error.PromptTooLong;
        if (p.pool.classes > 1 and n > p.pool.classCap()) return error.ClassTooSmall; // not LaneMemory: never requeued
        if (owner >= p.pool.classes) return error.BadStep;
        const l = &p.leases[slot];
        l.owner = owner; // every id it takes is of this class
        try p.pool.promise(l, n - adopted.len);
        errdefer p.pool.finish(l);
        if (adopted.len > 0) {
            held = false;
            try p.pool.adopt(l, adopted); // the lease holds them now (its finish releases them)
        }
        var up: Upload = .{ .p = p, .slot = slot };
        if (adopted.len > 0) try Upload.write(&up, l.ids.items, 0);
        try p.pool.cover(l, n, .{ .ctx = &up, .write = Upload.write });
        try p.mapped(slot, l.ids.items, adopted.len);
        return l.ids.items;
    }

    /// Another rank: paged slot `slot` maps rank 0's page ids (its begin step's), the prompt cache's `kept` first.
    pub fn map(p: *Pages, slot: u32, ids: []const u32, kept: usize) !void {
        return p.mapOwned(slot, ids, kept, 0);
    }

    /// `map` of a stream of owner class `owner` (its begin step's): every id of that class, or ResidueMismatch.
    pub fn mapOwned(p: *Pages, slot: u32, ids: []const u32, kept: usize, owner: u32) !void {
        if (!p.paged(slot) or ids.len == 0 or ids.len > tableLen(p.e.capacity) or kept > ids.len) return error.BadStep;
        if (p.used[slot] or owner >= p.pool.classes) return error.BadStep;
        for (ids) |id| if (id >= p.pool.total()) return error.BadStep;
        for (ids) |id| if (id % p.pool.classes != owner) return error.ResidueMismatch;
        const l = &p.leases[slot];
        l.owner = owner;
        l.ids.clearRetainingCapacity();
        try l.ids.appendSlice(p.gpa, ids);
        errdefer l.ids.clearRetainingCapacity();
        var up: Upload = .{ .p = p, .slot = slot };
        try Upload.write(&up, l.ids.items, 0);
        try p.mapped(slot, l.ids.items, kept);
    }

    /// The slot's sequence reaches its pages' rows (past the `kept` first, NaN first under TF_KV_POISON).
    fn mapped(p: *Pages, slot: u32, ids: []const u32, kept: usize) !void {
        const st = &p.e.seqs.items[slot].?.st;
        if (p.poison) for (ids[kept..]) |id| for (p.parts()) |r| if (r.bytes > 0) try p.e.d.check(p.e.d.api.cuMemsetD32Async(r.at + id * r.bytes, 0xFFFF_FFFF, r.bytes / 4, p.e.stream.handle), "cuMemsetD32Async");
        st.pages = ids;
        st.capacity = @min(ids.len * state.page_rows, p.e.capacity);
        p.used[slot] = true;
        if (kv_host.active) |*g| try g.invalidate(p.e.stream); // the row cache may hold these pages' old rows
    }

    /// The stream on paged slot `slot` ended: its sequence reaches no rows, and rank 0's lease gives its pages back.
    pub fn close(p: *Pages, slot: u32) void {
        if (!p.paged(slot) or !p.used[slot]) return;
        const st = &p.e.seqs.items[slot].?.st;
        st.pages = &.{};
        st.capacity = 0;
        p.used[slot] = false;
        if (kv_host.active) |*g| g.invalidate(p.e.stream) catch {};
        if (!p.leader) {
            p.leases[slot].owner = 0;
            return p.leases[slot].ids.clearRetainingCapacity();
        }
        p.pool.finish(&p.leases[slot]);
        for (p.used) |u| if (u) return;
        p.report();
    }

    /// The idle server's line: nothing promised, every page free or, under TF_KV_SHARE, the store's.
    fn report(p: *const Pages) void {
        if (@import("builtin").is_test) return;
        const c = p.pool.census();
        const ok = p.pool.idle() and (p.e.share_pages or c.free == c.total);
        var buf: [kv.max_classes * 48]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        const shown: usize = if (p.pool.classes > 1) p.pool.classes else 0; // TF_GLM_KV_SPLIT: each owner class
        for (0..shown) |k| {
            const x: u32 = @intCast(k);
            w.print("{s}{d} + {d} of {d}", .{ if (k == 0) " (classes: free " else ", ", p.pool.classFree(x), p.pool.classPromised(x), p.pool.classCap() }) catch break;
        }
        if (p.pool.classes > 1) w.writeAll(")") catch {};
        std.log.info("kv pool idle{s}: promised {d}, free {d} + store {d} == total {d}{s}", .{ if (ok) "" else " (UNEVEN)", c.promised, c.free, c.kept, c.total, w.buffered() });
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
            // TF_GLM_KV_SPLIT: the latent table holds the same ids while every rank holds every row
            if (st.ltable != 0) try e.d.check(e.d.api.cuMemcpyHtoD_v2(st.ltable + from * 4, bytes.ptr, bytes.len), "cuMemcpyHtoD");
        }
    };
};
