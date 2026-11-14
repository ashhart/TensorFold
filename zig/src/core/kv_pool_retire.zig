//! A lease's page leaving mid-stream (C5, #2444 P4): a logical free now, the page reused only after its readers.
const std = @import("std");
const kv = @import("kv_pool.zig");
const Pool = kv.Pool;
const Lease = kv.Lease;

/// A lease entry that names no page: retired, or below a resume's window; every loop over a table skips it.
pub const none: u32 = std.math.maxInt(u32);

/// A retired page's promise owed to its lease, granted when the forwards up to `token` are done.
pub const Credit = struct { token: u64, lease: *Lease };

/// A pool's retired pages: each one's last reader, the readers done, pages waiting on them, promises owed.
pub const Retired = struct {
    /// A page's last reader's token, 0 for none; allocated at the pool's first retire.
    tokens: []u64 = &.{},
    /// Pages whose token is not 0.
    ntokens: usize = 0,
    /// Every reader up to this token is done.
    settled: u64 = 0,
    /// Pages nobody holds whose readers are not done (room for every page, from the first retire).
    pending: std.ArrayList(u32) = .empty,
    credits: std.ArrayList(Credit) = .empty,

    pub fn deinit(r: *Retired, gpa: std.mem.Allocator) void {
        gpa.free(r.tokens);
        r.pending.deinit(gpa);
        r.credits.deinit(gpa);
    }
};

/// Lease entry `i` leaves; its page stays unused until `settle` passes `token` (the last forward that may read it).
pub fn retire(p: *Pool, l: *Lease, i: usize, token: u64) error{OutOfMemory}!void {
    const r = &p.retired;
    const id = l.ids.items[i];
    if (id == none) std.debug.panic("kv pool: lease entry {d} retired twice", .{i});
    if (r.tokens.len == 0) { // the pool's first retire: room for a token and a pending slot a page
        try r.pending.ensureTotalCapacity(p.gpa, p.total());
        r.tokens = try p.gpa.alloc(u64, p.total());
        @memset(r.tokens, 0);
    }
    try r.credits.ensureUnusedCapacity(p.gpa, 1);
    p.enter();
    defer p.leave();
    if (r.tokens[id] == 0) r.ntokens += 1;
    r.tokens[id] = @max(r.tokens[id], token);
    l.ids.items[i] = none;
    p.unlease(id);
    r.credits.appendAssumeCapacity(.{ .token = token, .lease = l });
}

/// Whether page `id` carries a retired page's token: not free to take, even when nobody holds it.
pub fn held(p: *const Pool, id: usize) bool {
    return p.retired.ntokens > 0 and p.retired.tokens[id] != 0;
}

/// Whether any page of [start, start + k) carries a token.
pub fn anyHeld(p: *const Pool, start: usize, k: usize) bool {
    if (p.retired.ntokens == 0) return false;
    for (start..start + k) |i| if (p.retired.tokens[i] != 0) return true;
    return false;
}

/// No page waits on its readers and no promise is owed.
pub fn idle(p: *const Pool) bool {
    return p.retired.pending.items.len == 0 and p.retired.credits.items.len == 0;
}

/// A page nobody holds: free when its readers are done, else pending until `settle` passes its token.
pub fn freeOrPend(p: *Pool, id: u32) void {
    const r = &p.retired;
    if (r.ntokens > 0 and r.tokens[id] > r.settled) return r.pending.appendAssumeCapacity(id); // room: one a page
    clear(p, id);
    p.free[p.nfree] = id;
    p.nfree += 1;
}

/// The forwards up to `token` are done: pages they read become free, then each retired page's promise goes back.
pub fn settle(p: *Pool, token: u64) error{LaneMemory}!void {
    const r = &p.retired;
    p.enter();
    defer p.leave();
    r.settled = @max(r.settled, token);
    var keep: usize = 0;
    for (r.pending.items) |id| {
        if (r.tokens[id] > r.settled) {
            r.pending.items[keep] = id;
            keep += 1;
        } else {
            clear(p, id);
            p.free[p.nfree] = id;
            p.nfree += 1;
        }
    }
    r.pending.shrinkRetainingCapacity(keep);
    while (firstDue(p)) |c| { // a granted promise may reclaim, which may free more pages: walk afresh
        try p.promise(r.credits.items[c].lease, 1);
        _ = r.credits.orderedRemove(c);
    }
}

fn clear(p: *Pool, id: u32) void {
    const r = &p.retired;
    if (r.ntokens == 0 or r.tokens[id] == 0) return;
    r.ntokens -= 1;
    r.tokens[id] = 0;
}

/// Pages the store alone holds whose readers are not done: reclaiming them frees nothing until `settle`.
pub fn waiting(p: *const Pool) usize {
    const r = &p.retired;
    if (r.ntokens == 0) return 0;
    var n: usize = 0;
    for (r.tokens, p.refs, p.leased) |t, refs, l| n += @intFromBool(t > r.settled and refs == 1 and l == 0);
    return n;
}

fn firstDue(p: *const Pool) ?usize {
    for (p.retired.credits.items, 0..) |c, i| if (c.token <= p.retired.settled) return i;
    return null;
}

/// `n` entries naming no page at the end of `l`'s table (a resume's rows below its window).
pub fn skip(p: *Pool, l: *Lease, n: usize) error{OutOfMemory}!void {
    try l.ids.appendNTimes(p.gpa, none, n);
}

/// `l` ended: the promises still owed to it are dropped.
pub fn forget(p: *Pool, l: *const Lease) void {
    const r = &p.retired;
    var keep: usize = 0;
    for (r.credits.items) |c| if (c.lease != l) {
        r.credits.items[keep] = c;
        keep += 1;
    };
    r.credits.shrinkRetainingCapacity(keep);
}

test {
    _ = @import("kv_pool_window_test.zig");
    _ = @import("kv_window.zig");
}
