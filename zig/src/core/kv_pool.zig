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
    /// Bytes of one kept snapshot's slot, outside the pages.
    snap_bytes: u64,

    pub fn pageBytes(l: Layout) u64 {
        return l.token_bytes * l.page_tokens;
    }

    /// Pages `tokens` rows take.
    pub fn pages(l: Layout, tokens: u64) usize {
        return @intCast((tokens + l.page_tokens - 1) / l.page_tokens);
    }

    /// Pages a request reserves at admission: its prompt, its reply and one last verify window.
    pub fn need(l: Layout, prompt: u64, max_tokens: u64, window: u64) usize {
        return l.pages(prompt + max_tokens + window);
    }
};

/// Slots a plan sets aside for snapshots when the caller names no count.
pub const most_snaps = 64;

/// What the room at load pays for, in order: the engine's own sequence, the other streams, the prompt cache.
pub const Budget = struct {
    /// What the device grants past the weights.
    room: u64,
    /// The engine's own full-window sequence, counted once.
    engine: u64,
    /// Streams asked for, the engine's own among them.
    lanes: u32,
    /// --parallel named a number: refuse rather than serve fewer.
    fixed: bool = false,
    /// The prompt cache's budget: its snapshot slots and its copied KV.
    cache: u64 = 0,
    /// Snapshot slots (null: half the cache's bytes in slots); most_snaps at most.
    snaps: ?usize = null,
    /// Rows one stream holds at most (0: no bound): pages past what every other stream holds at once stay unmade.
    window: u64 = 0,
};

/// The streams served, the cache's slots and copied-KV bytes, and the pages left for the streams.
pub const Plan = struct { lanes: u32, snaps: usize, copied: u64, pages: usize };

/// The prompt cache's split: slots for its snapshots, the rest for their copied KV rows (no slot: no cache).
pub fn split(cache: u64, snap_bytes: u64, named: ?usize) struct { snaps: usize, copied: u64 } {
    const snaps = @min(most_snaps, named orelse cache / 2 / snap_bytes, cache / snap_bytes);
    return .{ .snaps = @intCast(snaps), .copied = if (snaps > 0) cache - snaps * snap_bytes else 0 };
}

/// The pool as room - engine - (lanes - 1) x fixed - snaps x snap_bytes - copied, a page a stream beside it at least.
pub fn plan(l: Layout, b: Budget) error{ NoStream, TooMany }!Plan {
    if (b.engine > b.room or b.lanes == 0) return error.NoStream;
    var left = b.room - b.engine;
    const extra: u64 = b.lanes - 1;
    const each = l.fixed_bytes + l.pageBytes();
    const fit = @min(extra, left / each);
    if (fit < extra and b.fixed) return error.TooMany;
    left -= fit * l.fixed_bytes;
    const c = split(@min(b.cache, left - fit * l.pageBytes()), l.snap_bytes, b.snaps);
    left -= c.snaps * l.snap_bytes + c.copied;
    const most: u64 = if (b.window > 0) fit * l.pages(b.window) else std.math.maxInt(u64);
    return .{ .lanes = @intCast(1 + fit), .snaps = c.snaps, .copied = c.copied, .pages = @intCast(@min(left / l.pageBytes(), most)) };
}

/// Whether `b.cache` fits whole in what `plan` leaves it: the room past the engine and a fixed part and page a stream.
pub fn holds(l: Layout, b: Budget) bool {
    if (b.engine > b.room or b.lanes == 0) return false;
    const left = b.room - b.engine;
    const each = l.fixed_bytes + l.pageBytes();
    return left - @min(@as(u64, b.lanes - 1), left / each) * each >= b.cache;
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
        var n: usize = 0;
        for (p.refs, p.leased) |r, l| n += @intFromBool(r == 1 and l == 0);
        return n;
    }

    /// Whether `need` pages could be promised now, the store's idle pages counted.
    pub fn fits(p: *const Pool, need: usize) bool {
        return p.available() + p.reclaimable() >= need;
    }

    /// The counts an idle server logs: promised 0, free + kept == total, no page leased.
    pub fn census(p: *const Pool) Census {
        var kept: usize = 0;
        var leased: usize = 0;
        for (p.refs, p.leased) |r, l| {
            kept += @intFromBool(r > l);
            leased += @intFromBool(l > 0);
        }
        return .{ .total = p.total(), .free = p.nfree, .promised = p.promised, .kept = kept, .leased = leased };
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
        p.refs[id] += 1;
    }

    /// One holder fewer; the last one's release frees the page. Releasing a page nobody holds panics.
    pub fn release(p: *Pool, id: u32) void {
        p.drop(id) catch std.debug.panic("kv pool: page {d} released with no holder", .{id});
    }

    /// `release`, as an error when no holder but the leases' holds the page (the store's release of a page it lacks).
    pub fn drop(p: *Pool, id: u32) error{NotHeld}!void {
        if (p.refs[id] <= p.leased[id]) return error.NotHeld; // no holder past the leases' (callers drop theirs first)
        p.refs[id] -= 1;
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
        p.refs[id] = 1;
        p.leased[id] = 1;
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
        for (ids) |id| p.leased[id] += 1;
    }

    /// The stream's page `i` swapped for the store's copy `id` of the same tokens (a keep's Kept.shared).
    pub fn share(p: *Pool, l: *Lease, i: usize, id: u32) void {
        const old = l.ids.items[i];
        if (old == id) return;
        p.retain(id);
        p.leased[id] += 1;
        l.ids.items[i] = id;
        p.leased[old] -= 1;
        p.release(old);
    }

    /// `l` covers `n` pages, taken from its promise, and the new entries reach `table`; a failure takes them back.
    pub fn cover(p: *Pool, l: *Lease, n: usize, table: ?Table) !void {
        const old = l.ids.items.len;
        errdefer p.shrink(l, old);
        while (l.ids.items.len < n) _ = try p.take(l);
        if (table) |t| if (l.ids.items.len > old) try t.write(t.ctx, l.ids.items, old);
    }

    /// Pages past `n` in `l`'s table go back to its promise, as they were before it took them.
    fn shrink(p: *Pool, l: *Lease, n: usize) void {
        while (l.ids.items.len > n) {
            const id = l.ids.pop().?;
            p.leased[id] -= 1;
            p.release(id);
            l.promised += 1;
            p.promised += 1;
        }
    }

    /// The stream ended (finished, cancelled or failed): its promise and every page it holds go back.
    pub fn finish(p: *Pool, l: *Lease) void {
        p.unpromise(l);
        for (l.ids.items) |id| {
            p.leased[id] -= 1;
            p.release(id);
        }
        l.ids.deinit(p.gpa);
        l.* = .{};
    }
};

test {
    _ = @import("kv_pool_test.zig");
}
