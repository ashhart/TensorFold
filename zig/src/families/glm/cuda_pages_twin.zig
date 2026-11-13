//! TF_GLM_KV_SPLIT's prompt twins: a fresh prompt pass borrows each other class's pages so every rank attends its own.
const std = @import("std");
const kv = @import("core").kv_pool;
const Pages = @import("cuda_pages.zig").Pages;
const tableLen = @import("cuda_pages.zig").tableLen;

/// Rank 0's twins: slot i's lease of class c at i * classes + c (its owner's unused), and its begin step's ids.
pub const Twins = struct {
    leases: []kv.Lease = &.{},
    wire: []u32 = &.{}, // slot i's from i * classes * table: its ids, then each twin's in class order, as long as the ids
    on: []bool = &.{}, // slot i holds twins now

    pub fn init(gpa: std.mem.Allocator, streams: usize, classes: u32, table: usize) !Twins {
        const leases = try gpa.alloc(kv.Lease, streams * classes);
        errdefer gpa.free(leases);
        @memset(leases, .{});
        const ids = try gpa.alloc(u32, streams * classes * table);
        errdefer gpa.free(ids);
        const on = try gpa.alloc(bool, streams);
        @memset(on, false);
        return .{ .leases = leases, .wire = ids, .on = on };
    }

    pub fn deinit(t: *Twins, gpa: std.mem.Allocator) void {
        gpa.free(t.leases);
        gpa.free(t.wire);
        gpa.free(t.on);
    }
};

/// Rank 0, a fresh pass of `slot` on `n` pages: a twin of its first `m` (the prompt's) in every other class, or none.
pub fn borrow(p: *Pages, slot: u32, n: usize, m: usize) !bool {
    const c = p.pool.classes;
    if (c == 1 or p.twin.on.len == 0 or m == 0 or m > n) return false;
    const owner = p.leases[slot].owner;
    for (0..c) |k| if (k != owner and p.pool.availableIn(@intCast(k)) < m) return false;
    errdefer drop(p, slot);
    p.twin.on[slot] = true;
    for (0..c) |k| if (k != owner) {
        const l = &p.twin.leases[slot * c + k];
        l.owner = @intCast(k);
        try p.pool.promise(l, m);
        try p.pool.cover(l, m, null);
    };
    const w = p.twin.wire[slot * c * tableLen(p.e.capacity) ..][0 .. c * n];
    const ids = p.leases[slot].ids.items; // a twin's wire: its m pages, then the stream's ids
    @memcpy(w[0..n], ids);
    var at = n;
    for (0..c) |k| if (k != owner) {
        @memcpy(w[at..][0..m], p.twin.leases[slot * c + k].ids.items);
        @memcpy(w[at + m ..][0 .. n - m], ids[m..]);
        at += n;
    };
    return true;
}

/// A twin's own pages in its wire: the leading ids of class `rank` (the stream's ids after them are its owner's).
pub fn own(ids: []const u32, rank: u32, classes: u32) usize {
    var m: usize = 0;
    while (m < ids.len and ids[m] % classes == rank) m += 1;
    return m;
}

/// The begin step's ids for `slot`'s `ids`: the ids alone, or (twinned) the ids then each twin's in class order.
pub fn wire(p: *const Pages, slot: u32, ids: []const u32) []const u32 {
    if (p.twin.on.len == 0 or !p.twin.on[slot]) return ids;
    return p.twin.wire[slot * p.pool.classes * tableLen(p.e.capacity) ..][0 .. p.pool.classes * ids.len];
}

/// Whether rank 0 sent `slot` twins (its begin step's flag).
pub fn twinned(p: *const Pages, slot: u32) bool {
    return p.twin.on.len > 0 and p.twin.on[slot];
}

/// Every rank, a twinned begin's `wire` (the owner's ids, then each other class's): this rank's twin, if not owner.
pub fn mine(p: *const Pages, wire_ids: []const u32, owner: u32) ![]const u32 {
    const c = p.pool.classes;
    if (c < 2 or wire_ids.len % c != 0) return error.BadStep;
    const n = wire_ids.len / c;
    const rank: u32 = @intCast(p.e.share.rank);
    if (rank == owner) return &.{};
    const k = if (rank < owner) rank else rank - 1; // the twins come in class order, the owner's left out
    const ids = wire_ids[n * (1 + k) ..][0..n];
    const m = own(ids, rank, c);
    if (m == 0) return error.ResidueMismatch; // a twin holds the prompt's pages: at least one, of this rank's class
    for (ids[m..], wire_ids[m..n]) |id, its| if (id != its) return error.ResidueMismatch; // past them: the stream's ids
    return ids;
}

/// Every rank: `slot`'s stream is twinned for its prompt pass; this rank's latent table points into its twin `ids`.
pub fn map(p: *Pages, slot: u32, ids: []const u32) !void {
    const st = &p.e.seqs.items[slot].?.st;
    if (ids.len > 0) {
        try p.poisonLatent(ids[0..own(ids, @intCast(p.e.share.rank), p.pool.classes)]); // a row read before its write shows
        try upload(p, slot, ids);
    }
    st.twin = true; // every rank attends its own heads (no exchange) until the twin goes back
}

/// Every rank, before `slot`'s first decode round: its latent table back to its owner's places, rank 0's twins freed.
pub fn giveBack(p: *Pages, slot: u32) !void {
    if (!p.paged(slot)) return;
    const st = &p.e.seqs.items[slot].?.st;
    if (st.twin) {
        if (st.owner == null or st.owner.? != p.e.share.rank) try upload(p, slot, p.leases[slot].ids.items); // trash again
        st.twin = false;
    }
    if (p.leader) drop(p, slot);
}

/// Rank 0: `slot`'s twins back to their classes.
pub fn drop(p: *Pages, slot: u32) void {
    if (!twinned(p, slot)) return;
    const c = p.pool.classes;
    for (p.twin.leases[slot * c ..][0..c]) |*l| if (l.ids.items.len > 0 or l.promised > 0) p.pool.finish(l);
    p.twin.on[slot] = false;
}

/// `slot`'s latent table: each of `ids`' place in this rank's region (the trash page where it holds none).
fn upload(p: *Pages, slot: u32, ids: []const u32) !void {
    const e = p.e;
    const st = &e.seqs.items[slot].?.st;
    const t = tableLen(e.capacity);
    if (st.ltable == 0 or ids.len > t) return error.BadStep;
    const local = p.lat_ids[slot * t ..][0..ids.len];
    for (local, ids) |*x, id| x.* = p.region().entry(id);
    const b = std.mem.sliceAsBytes(local);
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(st.ltable, b.ptr, b.len), "cuMemcpyHtoD");
}
