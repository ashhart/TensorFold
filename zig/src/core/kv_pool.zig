//! One pool of KV pages for every stream: a family's layout, page ids and their holders, and the streams' leases.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A family's stream memory: a fixed part at any length, paged caches a token, and a kept snapshot's slot.
pub const Layout = struct {
    /// Bytes a stream holds whatever its length: recurrent states, window scratch, its page table.
    fixed_bytes: u64,
    /// Bytes one token adds across every paged cache and layer.
    token_bytes: u64,
    /// Tokens a page holds.
    page_tokens: u32,
    /// Bytes of one kept snapshot's slot, outside the pages (0: no slots, the family's kept states are copies alone).
    snap_bytes: u64,
    /// Bytes a running stream takes from the pages beside its keys and values, as one run of whole pages (0: none).
    run_bytes: u64 = 0,

    pub fn pageBytes(l: Layout) u64 {
        return l.token_bytes * l.page_tokens;
    }

    /// Pages `tokens` rows take.
    pub fn pages(l: Layout, tokens: u64) usize {
        return @intCast((tokens + l.page_tokens - 1) / l.page_tokens);
    }

    /// Pages a request reserves at admission: its prompt, its reply and one last verify window, and its run.
    pub fn need(l: Layout, prompt: u64, max_tokens: u64, window: u64) usize {
        return l.pages(prompt + max_tokens + window) + l.runPages();
    }

    /// Consecutive pages a running stream's `run_bytes` take.
    pub fn runPages(l: Layout) usize {
        return @intCast((l.run_bytes + l.pageBytes() - 1) / l.pageBytes());
    }
};

/// Slots a plan sets aside for snapshots when the caller names no count.
pub const most_snaps = 64;

/// What the room at load pays for, in order: the engine's own sequence, the other streams, the prompt cache.
pub const Budget = struct {
    /// What the device grants past the weights.
    room: u64,
    /// The engine's own sequence, counted once: its full window, or its fixed part when its rows are pages.
    engine: u64,
    /// Streams asked for, the engine's own among them.
    lanes: u32,
    /// --parallel named a number: refuse rather than serve fewer.
    fixed: bool = false,
    /// The prompt cache's budget: its snapshot slots and its copied KV.
    cache: u64 = 0,
    /// Snapshot slots (null: half the cache's bytes in slots); most_snaps at most.
    snaps: ?usize = null,
    /// Rows one stream holds at most (0: no bound): pages past what the paged streams hold at once stay unmade.
    window: u64 = 0,
    /// The engine's own rows are pages too: a window of them first, then the cache after the windows the room holds.
    engine_paged: bool = false,
};

/// The streams served, the cache's slots and copied-KV bytes, and the pages left for the streams.
pub const Plan = struct { lanes: u32, snaps: usize, copied: u64, pages: usize };

/// The prompt cache's split: slots for its snapshots, the rest for their copied KV rows (no slot: no cache).
pub fn split(cache: u64, snap_bytes: u64, named: ?usize) struct { snaps: usize, copied: u64 } {
    if (snap_bytes == 0) return .{ .snaps = 0, .copied = cache }; // a family without slots: its states are all copies
    const snaps = @min(most_snaps, named orelse cache / 2 / snap_bytes, cache / snap_bytes);
    return .{ .snaps = @intCast(snaps), .copied = if (snaps > 0) cache - snaps * snap_bytes else 0 };
}

/// The pool as room - engine - (lanes - 1) x fixed - snaps x snap_bytes - copied, a page a stream beside it at least.
pub fn plan(l: Layout, b: Budget) error{ NoStream, TooMany }!Plan {
    const own = ownPages(l, b);
    if (b.engine + own > b.room or b.lanes == 0) return error.NoStream;
    var left = b.room - b.engine - own;
    const extra: u64 = b.lanes - 1;
    const each = l.fixed_bytes + (1 + l.runPages()) * l.pageBytes();
    const fit = @min(extra, left / each);
    if (fit < extra and b.fixed) return error.TooMany;
    left -= fit * l.fixed_bytes;
    const c = split(@min(b.cache, left -| held(l, b, fit)), l.snap_bytes, b.snaps);
    left -= c.snaps * l.snap_bytes + c.copied;
    const most: u64 = if (b.window > 0) (fit + @intFromBool(b.engine_paged)) * l.pages(b.window) + fit * l.runPages() else std.math.maxInt(u64);
    return .{ .lanes = @intCast(1 + fit), .snaps = c.snaps, .copied = c.copied, .pages = @intCast(@min((left + own) / l.pageBytes(), most)) };
}

/// Whether `b.cache` fits whole in what `plan` leaves it: the room past the engine and a fixed part and page a stream.
pub fn holds(l: Layout, b: Budget) bool {
    const own = ownPages(l, b);
    if (b.engine + own > b.room or b.lanes == 0) return false;
    const left = b.room - b.engine - own;
    const each = l.fixed_bytes + (1 + l.runPages()) * l.pageBytes();
    const fit = @min(@as(u64, b.lanes - 1), left / each);
    return left -| (fit * l.fixed_bytes + held(l, b, fit)) >= b.cache;
}

/// The pages the cache leaves the other streams: a page and a run each, or the windows the room would hold if more.
fn held(l: Layout, b: Budget, fit: u64) u64 {
    return @max(fit * (1 + l.runPages()), (windows(l, b) - 1) * l.pages(b.window)) * l.pageBytes();
}

/// Streams a window each would serve in this room, the engine's own among them (1 unless its rows are pages).
fn windows(l: Layout, b: Budget) u64 {
    if (!b.engine_paged or b.window == 0) return 1;
    const own = ownPages(l, b);
    return @min(b.lanes, 1 + (b.room - own) / (own + l.fixed_bytes));
}

/// The bytes of the engine's own window as pages, set aside first when its rows are pages (0 when it has a window).
fn ownPages(l: Layout, b: Budget) u64 {
    return if (b.engine_paged) l.pages(b.window) * l.pageBytes() else 0;
}

/// A stream's reservation: pages promised but not yet taken, and the pages it holds in token order (its table).
pub const Lease = struct {
    promised: usize = 0,
    ids: std.ArrayList(u32) = .empty,
};

/// Where a stream's page table goes on the device: `ids[from..]` written at entry `from`.
pub const Table = struct {
    ctx: *anyopaque,
    write: *const fn (ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void,
};

/// The store's eviction a promise asks for before it fails: free pages until `want` are available.
pub const Reclaim = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, want: usize) bool,
};

/// What an idle pool must show: nothing promised, every page free or the store's alone.
pub const Census = struct { total: usize, free: usize, promised: usize, kept: usize, leased: usize };

/// The core owner of page ids: holders a page, the free list and the promised count (prompt_radix.pagesOf: its Pages).
pub const Pool = struct {
    gpa: Allocator,
    refs: []u32, // holders a page: each lease that maps it once, the store once
    leased: []u32, // of those, the leases'
    free: []u32, // a stack of free ids, `nfree` deep
    nfree: usize,
    promised: usize = 0,
    nreclaimable: usize = 0, // one non-lease holder, so eviction can free the page
    nkept: usize = 0, // pages with a holder beyond their leases; may also be leased
    nleased: usize = 0, // pages mapped by at least one lease, counted once
    page_bytes: u64,
    reclaim: ?Reclaim = null,

    pub fn init(gpa: Allocator, count: usize, page_bytes: u64) !Pool {
        const refs = try gpa.alloc(u32, count);
        errdefer gpa.free(refs);
        const leased = try gpa.alloc(u32, count);
        errdefer gpa.free(leased);
        const free = try gpa.alloc(u32, count);
        @memset(refs, 0);
        @memset(leased, 0);
        for (free, 0..) |*f, i| f.* = @intCast(count - 1 - i); // page 0 taken first
        return .{ .gpa = gpa, .refs = refs, .leased = leased, .free = free, .nfree = count, .page_bytes = page_bytes };
    }

    pub fn deinit(p: *Pool) void {
        p.gpa.free(p.refs);
        p.gpa.free(p.leased);
        p.gpa.free(p.free);
        p.* = undefined;
    }

    pub fn total(p: *const Pool) usize {
        return p.refs.len;
    }

    /// Pages free and not promised to a stream.
    pub fn available(p: *const Pool) usize {
        return p.nfree - p.promised;
    }

    /// Pages the store alone holds: evicting its idle leaves frees them.
    pub fn reclaimable(p: *const Pool) usize {
        return p.nreclaimable;
    }

    /// Whether `need` pages could be promised now, the store's idle pages counted.
    pub fn fits(p: *const Pool, need: usize) bool {
        return p.available() + p.reclaimable() >= need;
    }

    /// The counts an idle server logs: promised 0, free + kept == total, no page leased.
    pub fn census(p: *const Pool) Census {
        return .{ .total = p.total(), .free = p.nfree, .promised = p.promised, .kept = p.nkept, .leased = p.nleased };
    }

    /// Ownership changes update page classes once; holders and leases remain the authoritative per-page state.
    fn ownership(p: *Pool, id: usize, refs: u32, leased: u32) void {
        const r = p.refs[id];
        const l = p.leased[id];
        p.nreclaimable = p.nreclaimable - @intFromBool(r == 1 and l == 0) + @intFromBool(refs == 1 and leased == 0);
        p.nkept = p.nkept - @intFromBool(r > l) + @intFromBool(refs > leased);
        p.nleased = p.nleased - @intFromBool(l > 0) + @intFromBool(leased > 0);
        p.refs[id] = refs;
        p.leased[id] = leased;
    }

    /// A lease gives back its reference; its last reference returns the page to the same free stack.
    fn unlease(p: *Pool, id: u32) void {
        if (p.leased[id] == 0 or p.refs[id] < p.leased[id]) std.debug.panic("kv pool: page {d} released with no holder", .{id});
        p.ownership(id, p.refs[id] - 1, p.leased[id] - 1);
        if (p.refs[id] > 0) return;
        p.free[p.nfree] = id;
        p.nfree += 1;
    }

    pub fn idle(p: *const Pool) bool {
        const c = p.census();
        return c.promised == 0 and c.leased == 0 and c.free + c.kept == c.total;
    }

    /// How many hold the page: each lease that maps it, and the store.
    pub fn holders(p: *const Pool, id: u32) u32 {
        return p.refs[id];
    }

    /// One more holder of a held page; a free page is never retained, only taken.
    pub fn retain(p: *Pool, id: u32) void {
        if (p.refs[id] == 0) std.debug.panic("kv pool: page {d} retained while free", .{id});
        p.ownership(id, p.refs[id] + 1, p.leased[id]);
    }

    /// One holder fewer; the last one's release frees the page. Releasing a page nobody holds panics.
    pub fn release(p: *Pool, id: u32) void {
        p.drop(id) catch std.debug.panic("kv pool: page {d} released with no holder", .{id});
    }

    /// `release`, as an error when no holder but the leases' holds the page (the store's release of a page it lacks).
    pub fn drop(p: *Pool, id: u32) error{NotHeld}!void {
        if (p.refs[id] <= p.leased[id]) return error.NotHeld; // no holder past the leases' (callers drop theirs first)
        p.ownership(id, p.refs[id] - 1, p.leased[id]);
        if (p.refs[id] > 0) return;
        p.free[p.nfree] = id;
        p.nfree += 1;
    }

    /// Promise `n` more pages to `l`, the store reclaiming first if it must; LaneMemory when they do not fit.
    pub fn promise(p: *Pool, l: *Lease, n: usize) error{LaneMemory}!void {
        if (p.available() < n) if (p.reclaim) |r| {
            _ = r.call(r.ctx, n);
        };
        if (p.available() < n) return error.LaneMemory;
        p.promised += n;
        l.promised += n;
    }

    /// The part of `l`'s promise it has not taken goes back.
    pub fn unpromise(p: *Pool, l: *Lease) void {
        p.promised -= l.promised;
        l.promised = 0;
    }

    /// A free page from `l`'s promise, appended to its table.
    pub fn take(p: *Pool, l: *Lease) error{ PastReservation, OutOfMemory }!u32 {
        if (l.promised == 0) return error.PastReservation;
        try l.ids.ensureUnusedCapacity(p.gpa, 1);
        p.nfree -= 1;
        const id = p.free[p.nfree];
        p.ownership(id, 1, 1);
        l.promised -= 1;
        p.promised -= 1;
        l.ids.appendAssumeCapacity(id);
        return id;
    }

    /// Pages a store's resume retained for this stream (its matched path), at the head of its table.
    pub fn adopt(p: *Pool, l: *Lease, ids: []const u32) error{OutOfMemory}!void {
        l.ids.appendSlice(p.gpa, ids) catch |e| {
            for (ids) |id| p.release(id);
            return e;
        };
        for (ids) |id| p.ownership(id, p.refs[id], p.leased[id] + 1);
    }

    /// The stream's page `i` swapped for the store's copy `id` of the same tokens (a keep's Kept.shared).
    pub fn share(p: *Pool, l: *Lease, i: usize, id: u32) void {
        const old = l.ids.items[i];
        if (old == id) return;
        p.retain(id);
        p.ownership(id, p.refs[id], p.leased[id] + 1);
        l.ids.items[i] = id;
        p.unlease(old);
    }

    /// `l` covers `n` pages, taken from its promise, and the new entries reach `table`; a failure takes them back.
    pub fn cover(p: *Pool, l: *Lease, n: usize, table: ?Table) !void {
        const old = l.ids.items.len;
        errdefer p.shrink(l, old);
        while (l.ids.items.len < n) _ = try p.take(l);
        if (table) |t| if (l.ids.items.len > old) try t.write(t.ctx, l.ids.items, old);
    }

    /// `cover` for an empty table: n consecutive ids from a multiple of `step` when they are free, else as `cover`.
    pub fn coverRun(p: *Pool, l: *Lease, n: usize, step: usize, table: ?Table) !void {
        if (l.ids.items.len > 0 or n == 0 or l.promised < n) return p.cover(l, n, table);
        try l.ids.ensureTotalCapacity(p.gpa, n);
        const start = p.findAligned(n, step) orelse return p.cover(l, n, table);
        p.claimAt(l, start, n);
        for (0..n) |i| l.ids.appendAssumeCapacity(@intCast(start + i)); // held as `take` holds a page: `finish` frees them
        errdefer p.shrink(l, 0);
        if (table) |t| try t.write(t.ctx, l.ids.items, 0);
    }

    /// Pages past `n` in `l`'s table go back to its promise, as they were before it took them.
    fn shrink(p: *Pool, l: *Lease, n: usize) void {
        while (l.ids.items.len > n) {
            const id = l.ids.pop().?;
            p.unlease(id);
            l.promised += 1;
            p.promised += 1;
        }
    }

    /// The first id of `k` consecutive free pages, highest first (the free stack hands out low ids), or null.
    pub fn findRun(p: *const Pool, k: usize) ?u32 {
        if (k == 0 or k > p.refs.len) return null;
        var end = p.refs.len;
        while (end >= k) {
            const start = end - k;
            const busy = for (p.refs[start..end], start..) |r, i| {
                if (r != 0) break i;
            } else null;
            if (busy) |i| end = i else return @intCast(start);
        }
        return null;
    }

    /// The highest first id of `k` free consecutive pages at a multiple of `step` (> 0), or null.
    pub fn findAligned(p: *const Pool, k: usize, step: usize) ?u32 {
        if (k == 0 or k > p.refs.len) return null;
        var start = (p.refs.len - k) / step * step;
        while (true) : (start -= step) {
            if (std.mem.allEqual(u32, p.refs[start..][0..k], 0)) return @intCast(start);
            if (start < step) return null;
        }
    }

    /// `k` consecutive pages from `l`'s promise, held as one run until releaseRun; null: no such run is free now.
    pub fn claimRun(p: *Pool, l: *Lease, k: usize) ?u32 {
        if (l.promised < k) return null;
        const start = p.findRun(k) orelse return null;
        p.claimAt(l, start, k);
        return start;
    }

    /// Free pages [start, start + k) held by `l`, from its promise.
    fn claimAt(p: *Pool, l: *Lease, start: u32, k: usize) void {
        var n: usize = 0;
        for (p.free[0..p.nfree]) |id| if (id < start or id >= start + k) {
            p.free[n] = id;
            n += 1;
        };
        std.debug.assert(n + k == p.nfree);
        p.nfree = n;
        @memset(p.refs[start..][0..k], 1);
        @memset(p.leased[start..][0..k], 1); // held as a lease holds its pages: never the store's to reclaim
        p.nleased += k; // every page in this free run gained its first lease and no other holder
        l.promised -= k;
        p.promised -= k;
    }

    /// A run claimed at `start` goes back to the free pages.
    pub fn releaseRun(p: *Pool, start: u32, k: usize) void {
        for (start..start + k) |id| {
            if (p.refs[id] != 1 or p.leased[id] != 1) std.debug.panic("kv pool: run page {d} released while not a run's", .{id});
            p.ownership(id, 0, 0);
            p.free[p.nfree] = @intCast(id);
            p.nfree += 1;
        }
    }

    /// The stream ended (finished, cancelled or failed): its promise and every page it holds go back.
    pub fn finish(p: *Pool, l: *Lease) void {
        p.unpromise(l);
        for (l.ids.items) |id| p.unlease(id);
        l.ids.deinit(p.gpa);
        l.* = .{};
    }
};

test {
    _ = @import("kv_pool_test.zig");
    _ = @import("kv_pool_counts_test.zig");
    _ = @import("kv_pool_walk_test.zig");
}
