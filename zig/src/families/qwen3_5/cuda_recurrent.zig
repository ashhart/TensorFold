//! CUDA binds the shared recurrent lease owner to this engine's ordered stream and existing completion waits.
const cuda = @import("cuda");
const common = @import("core").recurrent_workspace;
const Engine = @import("cuda_engine.zig").Engine;
const Cache = @import("cuda_cache.zig").Cache;
pub const Pool = common.Workspace(cuda.DeviceBuffer);
pub const Lease = Pool.Lease;

pub fn enabled(e: *const Engine) bool {
    return e.g.hidden == 5120 and e.g.layers == 64 and e.g.experts == 0 and
        e.g.v_heads == 48 and e.g.k_heads == 16 and e.g.deltas() == 48;
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
    const next = try pool.acquire(@intFromPtr(e), bytes, Provider{ .e = e });
    var b = try pool.view(@intFromPtr(e), next);
    b.borrowed = true;
    lease.* = next;
    if (before) c.before = b else c.log = b;
}

pub fn prepare(e: *Engine, c: *Cache, rows: usize, record: bool) !void {
    const pool = c.owner orelse return;
    if (pool != &e.rec_scratch) return error.ForeignRecurrentQueue;
    if (!record) return release(c); // its scratch stays for another window until idle or the cap frees it
    const n = try common.rowClass(rows, @import("cuda_engine.zig").most_window);
    if (rows > 1) clear(c, true);
    try ensure(e, c, false, e.g.logBytes(n), rows == 1);
    if (rows == 1) try ensure(e, c, true, e.g.stateBytes(), false);
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
