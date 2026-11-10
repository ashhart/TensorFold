//! CUDA binds the shared recurrent lease owner to this engine's ordered stream and existing completion waits.
const std = @import("std");
const cuda = @import("cuda");
const common = @import("core").recurrent_workspace;
const Engine = @import("cuda_engine.zig").Engine;
const Cache = @import("cuda_cache.zig").Cache;
pub const Pool = common.Workspace(cuda.DeviceBuffer);
pub const Lease = Pool.Lease;
pub const budget = @import("cuda_recurrent_budget.zig");
pub const Budget = struct { streams: usize = 0, limit: usize = 0 };

pub fn enabled(e: *const Engine) bool {
    return e.g.hidden == 5120 and e.g.layers == 64 and e.g.experts == 0 and
        e.g.v_heads == 48 and e.g.k_heads == 16 and e.g.deltas() == 48;
}

pub fn needsFullKeep(e: *const Engine) bool {
    if (!enabled(e)) return false;
    return if (@hasField(Engine, "rec_budget")) e.rec_budget.streams > 1 else false;
}

pub fn sharedBytes(e: *const Engine) usize {
    if (!enabled(e)) return 0;
    return if (@hasField(Engine, "rec_budget")) e.rec_budget.limit else 0;
}

/// Configure once before caches exist; already released warm-up scratch is drained and discarded.
pub fn configure(e: *Engine, streams: usize) !void {
    if (streams <= 1 or !enabled(e)) return;
    if (@import("cuda_cache.zig").keepLate()) return; // pending buffers are not part of the shared admission reserve
    if (streams > @import("cuda_engine.zig").max_segs) return error.InvalidRecurrentBudget;
    if (e.rec_scratch.cap != 0) return error.RecurrentBudgetBusy;
    const limit = try budget.frame(0, e.g.deltas() * e.g.logRowBytes(), streams, @import("cuda_engine.zig").most_verify, @import("cuda_engine.zig").most_window);
    idle(e);
    e.rec_budget = .{ .streams = streams, .limit = limit };
    e.rec_scratch.cap = limit;
}

/// Global mode reserves sharedBytes once; legacy paged mode keeps its fixed before/log run parts.
pub fn pagedBytes(_: *const Engine) usize {
    return 0;
}

pub fn attachPaged(e: *Engine, c: *Cache) !void {
    if (!needsFullKeep(e)) return;
    if (c.owner != null) return error.DuplicateRecurrentOwner;
    const bytes = pagedBytes(e);
    const cap = try std.math.add(usize, e.rec_scratch.cap, bytes);
    c.owner = &e.rec_scratch;
    c.pool_run = true;
    c.scratch = bytes;
    c.before = null;
    c.log = .{ .d = e.d, .ptr = 0, .len = 0, .borrowed = true };
    e.rec_scratch.cap = cap;
}

pub const Provider = struct {
    e: *const Engine,
    // Forward produces every byte read by keep; scratch never initializes the persistent recurrence.
    pub fn allocate(p: Provider, bytes: usize) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(p.e.d, bytes);
    }
    pub fn dispose(_: Provider, b: *cuda.DeviceBuffer) void {
        b.free();
    }
};

/// Paged admission reserves cap explicitly; a new size class cannot transiently allocate beyond it.
pub const PagedProvider = struct {
    e: *Engine,
    pub fn allocate(p: PagedProvider, bytes: usize) !cuda.DeviceBuffer {
        const pool = &p.e.rec_scratch;
        const cap = if (needsFullKeep(p.e)) sharedBytes(p.e) else pool.cap;
        if (try std.math.add(usize, pool.residentBytes(), bytes) > cap) {
            cuda.completion.drain(p.e.d, p.e.stream, "paged recurrent scratch did not drain before resizing");
            completed(p.e);
            try pool.collect(@intFromPtr(p.e), Provider{ .e = p.e });
        }
        if (try std.math.add(usize, pool.residentBytes(), bytes) > cap) return error.RecurrentScratchBudget;
        return cuda.DeviceBuffer.alloc(p.e.d, bytes);
    }
    pub fn dispose(_: PagedProvider, b: *cuda.DeviceBuffer) void {
        b.free();
    }
};

fn clear(c: *Cache, before: bool) void {
    const pool = c.owner orelse return;
    const lease = if (before) &c.before_lease else &c.log_lease;
    if (lease.*) |l| pool.release(pool.domain, l) catch unreachable;
    lease.* = null;
    if (before) c.before = null else c.log = .{ .d = c.state.d, .ptr = 0, .len = 0, .borrowed = true };
}

pub fn release(c: *Cache) void {
    clear(c, true);
    clear(c, false);
    c.before_skipped = false;
}

/// All decisions for this window are committed; queued reads remain protected by the completion watermark.
pub fn committed(e: *Engine, c: *Cache) void {
    if (c.owner == null or !needsFullKeep(e)) return;
    std.debug.assert(c.last == null and c.owner.? == &e.rec_scratch);
    release(c);
}

fn ensure(e: *Engine, c: *Cache, before: bool, bytes: usize, reuse_pending: bool) !void {
    const pool = c.owner orelse return;
    const lease = if (before) &c.before_lease else &c.log_lease;
    if (lease.*) |l| {
        const b = try pool.view(@intFromPtr(e), l);
        if (b.len == bytes) return;
        if (reuse_pending and !before and bytes < b.len and !try pool.reclaimable(@intFromPtr(e), l)) return;
        clear(c, before);
    }
    if (bytes == 0) return;
    const next = if (c.pool_run or needsFullKeep(e)) try pool.acquire(@intFromPtr(e), bytes, PagedProvider{ .e = e }) else try pool.acquire(@intFromPtr(e), bytes, Provider{ .e = e });
    var b = try pool.view(@intFromPtr(e), next);
    b.borrowed = true;
    lease.* = next;
    if (before) c.before = b else c.log = b;
}

/// A private or calibration caller retains rollback; the native serving hook supplies its root promise explicitly.
pub fn prepare(e: *Engine, c: *Cache, rows: usize, record: bool) !void {
    return prepareRoot(e, c, rows, record, false);
}

/// Reject a global frame before settling state when its unpromised roots cannot fit the declared reserve.
pub fn admitFrame(e: *const Engine, parts: anytype, record: bool) !void {
    if (!record or !needsFullKeep(e)) return;
    if (parts.len == 0 or parts.len > e.rec_budget.streams) return error.InvalidRecurrentWindow;
    var rows: usize = 0;
    var bytes: usize = 0;
    for (parts) |p| {
        rows = try std.math.add(usize, rows, p.rows);
        const committed_root = if (@hasField(@TypeOf(p), "root_committed")) p.root_committed else false;
        const state = if (committed_root) 0 else e.g.stateBytes();
        bytes = try std.math.add(usize, bytes, try budget.perWindow(state, e.g.deltas() * e.g.logRowBytes(), p.rows, @import("cuda_engine.zig").most_window));
    }
    if (rows > @import("cuda_engine.zig").most_verify) return error.InvalidRecurrentWindow;
    if (bytes > sharedBytes(e)) return error.RecurrentScratchBudget;
}

pub fn prepareRoot(e: *Engine, c: *Cache, rows: usize, record: bool, root_committed: bool) !void {
    const pool = c.owner orelse {
        c.before_skipped = false;
        return;
    };
    if (pool != &e.rec_scratch) return error.ForeignRecurrentQueue;
    if (!record) return release(c); // its scratch stays for another window until idle or the cap frees it
    const n = try common.rowClass(rows, @import("cuda_engine.zig").most_window);
    const skip_before = rows == 1 and root_committed and needsFullKeep(e);
    if (rows > 1 or skip_before) clear(c, true);
    try ensure(e, c, false, e.g.logBytes(n), rows == 1);
    if (rows == 1 and !skip_before) try ensure(e, c, true, e.g.stateBytes(), false);
    c.before_skipped = skip_before;
}

pub fn mark(c: *Cache, before: bool) !void {
    const pool = c.owner orelse return;
    const lease = if (before) c.before_lease else c.log_lease;
    if (lease) |l| try pool.markUsed(pool.domain, l);
}

/// The queue's work so far is done; idle scratch stays for the next window (idle or the cap frees it).
pub fn completed(e: *Engine) void {
    e.rec_scratch.complete(@intFromPtr(e)) catch unreachable;
}

pub fn idle(e: *Engine) void {
    if (e.rec_scratch.issued != e.rec_scratch.completed)
        cuda.completion.drain(e.d, e.stream, "recurrent scratch keep work did not drain");
    completed(e);
    e.rec_scratch.collect(@intFromPtr(e), Provider{ .e = e }) catch unreachable;
}

/// What one scratch-owning cache may hold at once: a rollback copy and a widest window's log (1.1.1's own two).
pub fn bound(e: *const Engine) usize {
    return e.g.stateBytes() + e.g.logBytes(@import("cuda_engine.zig").most_window);
}

pub fn shutdown(e: *Engine) void {
    idle(e);
    e.rec_scratch.deinit(@intFromPtr(e), Provider{ .e = e }) catch unreachable;
}
