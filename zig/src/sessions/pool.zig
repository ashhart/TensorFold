//! The paged KV pool's host book: lowest-free pages per residue mod W, holder counts, slot page tables, quotas.

const std = @import("std");
const FreeSet = @import("freeset.zig").FreeSet;

pub const Error = error{ PoolExhausted, PastCapacity, SlotNotEmpty, DoubleFree, BadPool, ResidueMismatch } || std.mem.Allocator.Error;

/// A cache family: rows of `row_bytes` every `ratio` tokens; `split` families shard by logical page over the ranks.
pub const Family = struct {
    name: []const u8,
    ratio: u32,
    row_bytes: u32,
    split: bool = false,

    pub fn pageRows(f: Family, page: u32) u32 {
        return page / f.ratio;
    }

    /// Bytes of one page of this family (a whole page's rows).
    pub fn pageBytes(f: Family, page: u32) u64 {
        return @as(u64, f.pageRows(page)) * f.row_bytes;
    }
};

/// The pool's families and page size: what a token costs and how pages are laid out.
pub const Layout = struct {
    families: []const Family,
    /// tokens a page (a power of two; 256 for every family we serve)
    page: u32 = 256,

    /// A rank's bytes a token: the split families count 1 / W.
    pub fn bytesPerToken(l: Layout, world: u32) f64 {
        var b: f64 = 0;
        for (l.families) |f| b += @as(f64, @floatFromInt(f.row_bytes)) / @as(f64, @floatFromInt(f.ratio)) / @as(f64, @floatFromInt(if (f.split and world > 1) world else 1));
        return b;
    }

    /// A rank's average bytes a page (split families 1 / W): what the residency budget counts.
    pub fn pageBytes(l: Layout, world: u32) u64 {
        var b: u64 = 0;
        for (l.families) |f| b += f.pageBytes(l.page) / (if (f.split and world > 1) world else 1);
        return b;
    }
};

pub fn pagesFor(tokens: u64, page: u32) u32 {
    return @intCast((tokens + page - 1) / page);
}

/// Logical pages [k0, k1) counted by residue mod `w` (O(w)).
pub fn span(k0: u32, k1: u32, w: u32, out: []u32) void {
    for (out[0..w], 0..) |*o, r| {
        const rr: u32 = @intCast(r);
        const first = k0 + (rr + w - k0 % w) % w;
        o.* = if (first >= k1) 0 else (k1 - 1 - first) / w + 1;
    }
}

pub const max_world = 8;

/// A file whose logical pages [0, pages) match a slot's or entry's rows: a later park writes only past it.
pub const Origin = struct {
    key: [16]u8,
    pages: u32,
};

/// One slot's page table: logical page i -> physical page; the device tables follow through the dirty range.
pub const Slot = struct {
    pool: *Pool,
    id: u32,
    capacity: u64,
    max_pages: u32,
    /// physical page of each mapped logical page; len is the mapped count
    pages: []u32,
    len: u32 = 0,
    /// pages the admission reserved for this slot's request (null: no admission control)
    quota: ?u32 = null,
    /// pages taken past the quota, from pages nobody had reserved
    over: u32 = 0,
    /// table entries changed since the device side last synced them: [lo, hi)
    dirty_lo: u32 = std.math.maxInt(u32),
    dirty_hi: u32 = 0,
    /// the file the leading pages came from (truncation shortens it, release clears it)
    origin: ?Origin = null,

    pub fn mapped(s: *const Slot) []const u32 {
        return s.pages[0..s.len];
    }

    pub fn tokens(s: *const Slot) u64 {
        return @as(u64, s.len) * s.pool.page;
    }

    fn mark(s: *Slot, lo: u32, hi: u32) void {
        s.dirty_lo = @min(s.dirty_lo, lo);
        s.dirty_hi = @max(s.dirty_hi, hi);
    }

    /// The table range to upload since the last call (empty when nothing changed): entries past len are the null page.
    pub fn takeDirty(s: *Slot) ?struct { lo: u32, hi: u32 } {
        if (s.dirty_lo >= s.dirty_hi) return null;
        defer {
            s.dirty_lo = std.math.maxInt(u32);
            s.dirty_hi = 0;
        }
        return .{ .lo = s.dirty_lo, .hi = s.dirty_hi };
    }

    /// The device table's value at logical page i (the page, or the null page past len).
    pub fn tableAt(s: *const Slot, i: u32) u32 {
        return if (i < s.len) s.pages[i] else s.pool.null_page;
    }

    /// The split families' device table at logical page i: the local page this rank stores, else the discard page.
    pub fn localTableAt(s: *const Slot, i: u32) u32 {
        return s.pool.localPage(s.tableAt(i));
    }

    fn reservedSpan(s: *const Slot, out: []u32) void {
        const q = s.quota orelse 0;
        span(s.len, @max(s.len, q), s.pool.world, out);
    }

    /// Maps every page holding positions < end (a no-op when they are: the common case).
    pub fn ensure(s: *Slot, end: u64) Error!void {
        const p = s.pool;
        const need = pagesFor(end, p.page);
        if (need <= s.len) return;
        if (need > s.max_pages) return error.PastCapacity;
        const w = p.world;
        var before: [max_world]u32 = undefined;
        s.reservedSpan(&before);
        if (s.quota) |q| if (need > q) {
            // past the reservation: only pages nobody else reserved, residue by residue
            var extra: [max_world]u32 = undefined;
            span(@max(q, s.len), need, w, &extra);
            for (0..w) |r| if (extra[r] > p.free[r].count() -| (p.reserved[r] - before[r])) return error.PoolExhausted;
            s.over += need - @max(q, s.len);
        };
        var want: [max_world]u32 = undefined;
        span(s.len, need, w, &want);
        for (0..w) |r| if (want[r] > p.free[r].count()) return error.PoolExhausted;
        const lo = s.len;
        for (lo..need) |k| {
            const pg = p.takeAt(@intCast(k)) catch unreachable; // counted above
            p.hold(pg, 0, 1); // the taker's hold becomes the slot's
            s.pages[k] = pg;
        }
        s.len = need;
        p.unreserve(&before);
        var after: [max_world]u32 = undefined;
        s.reservedSpan(&after);
        p.addReserved(&after);
        s.mark(lo, need);
    }

    /// Maps shared pages as the first pages of an empty slot (a RAM resume: read only, the slot writes past them).
    pub fn adopt(s: *Slot, shared: []const u32) Error!void {
        if (s.len != 0) return error.SlotNotEmpty;
        if (shared.len > s.max_pages) return error.PastCapacity;
        const p = s.pool;
        var before: [max_world]u32 = undefined;
        s.reservedSpan(&before);
        for (shared, 0..) |pg, k| if (pg >= p.npages or pg % p.world != k % p.world) return error.ResidueMismatch;
        for (shared, 0..) |pg, k| {
            p.hold(pg, 1, 1);
            s.pages[k] = pg;
        }
        s.len = @intCast(shared.len);
        p.unreserve(&before);
        var after: [max_world]u32 = undefined;
        s.reservedSpan(&after);
        p.addReserved(&after);
        s.mark(0, s.len);
    }

    /// Appends one page taken for logical page len (the copy-on-write page of a RAM resume).
    pub fn push(s: *Slot, pg: u32) Error!void {
        if (s.len >= s.max_pages) return error.PastCapacity;
        const p = s.pool;
        if (p.world > 1 and pg % p.world != s.len % p.world) return error.ResidueMismatch;
        var before: [max_world]u32 = undefined;
        s.reservedSpan(&before);
        p.hold(pg, 0, 1); // the taker's hold becomes the slot's
        s.pages[s.len] = pg;
        s.len += 1;
        p.unreserve(&before);
        var after: [max_world]u32 = undefined;
        s.reservedSpan(&after);
        p.addReserved(&after);
        s.mark(s.len - 1, s.len);
    }

    /// Lets go of the pages past position keep (rows nothing reads again); returns how many.
    pub fn truncate(s: *Slot, keep: u64) Error!u32 {
        const p = s.pool;
        // positions >= keep get written again: a page holding one no longer equals the origin file's
        if (s.origin) |*o| o.pages = @min(o.pages, @as(u32, @intCast(keep / p.page)));
        const k = pagesFor(keep, p.page);
        if (k >= s.len) return 0;
        var before: [max_world]u32 = undefined;
        s.reservedSpan(&before);
        const n = s.len - k;
        for (s.pages[k..s.len]) |pg| try p.release(pg, 1);
        const old = s.len;
        s.len = k;
        p.unreserve(&before);
        var after: [max_world]u32 = undefined;
        s.reservedSpan(&after);
        p.addReserved(&after);
        s.mark(k, old);
        return n;
    }

    /// Sets the admission's reservation (null: none).
    pub fn reserve(s: *Slot, quota: ?u32) void {
        var before: [max_world]u32 = undefined;
        s.reservedSpan(&before);
        s.quota = quota;
        s.pool.unreserve(&before);
        var after: [max_world]u32 = undefined;
        s.reservedSpan(&after);
        s.pool.addReserved(&after);
    }

    /// Frees the slot's pages and its reservation.
    pub fn releaseAll(s: *Slot) Error!u32 {
        s.reserve(null);
        s.over = 0;
        s.origin = null;
        return s.truncate(0);
    }
};

/// The pool's book: free sets by residue, holders a page, the slots and their reservations.
pub const Pool = struct {
    gpa: std.mem.Allocator,
    layout: Layout,
    npages: u32,
    page: u32,
    /// log2 of page tokens
    shift: u5,
    /// ranks the split families are sharded over (1: every rank stores every row)
    world: u32,
    rank: u32,
    /// the page unmapped table entries point at (all zeros: a stray read reads zeros)
    null_page: u32,
    /// a split family's local page count (npages / W), its local null page and its discard page (other ranks' rows)
    local_pages: u32,
    local_null: u32,
    discard: u32,
    free: [max_world]FreeSet = @splat(.{}),
    /// pages each residue's slots reserved and have not mapped yet
    reserved: [max_world]u32 = @splat(0),
    /// holders a page (slots + entries + in-flight jobs) and how many are slots
    refs: []u16,
    slot_refs: []u16,
    /// pages held by something and by no slot (session entries alone): what parking can free
    entry_only: u32 = 0,
    slots: std.ArrayList(*Slot) = .empty,

    pub fn init(gpa: std.mem.Allocator, layout: Layout, tokens: u64, world: u32, rank: u32) Error!Pool {
        if (layout.page == 0 or layout.page & (layout.page - 1) != 0) return error.BadPool;
        if (tokens % layout.page != 0 or tokens == 0) return error.BadPool;
        if (world == 0 or world > max_world or rank >= world) return error.BadPool;
        const npages: u32 = @intCast(tokens / layout.page);
        if (npages % world != 0) return error.BadPool;
        for (layout.families) |f| if (f.ratio == 0 or layout.page % f.ratio != 0) return error.BadPool;
        var p: Pool = .{
            .gpa = gpa,
            .layout = layout,
            .npages = npages,
            .page = layout.page,
            .shift = @intCast(std.math.log2_int(u32, layout.page)),
            .world = world,
            .rank = rank,
            .null_page = npages,
            .local_pages = npages / world,
            .local_null = npages / world,
            .discard = npages / world + 1,
            .refs = &.{},
            .slot_refs = &.{},
        };
        errdefer p.deinit();
        for (0..world) |r| p.free[r] = try FreeSet.init(gpa, npages / world);
        p.refs = try gpa.alloc(u16, npages);
        @memset(p.refs, 0);
        p.slot_refs = try gpa.alloc(u16, npages);
        @memset(p.slot_refs, 0);
        return p;
    }

    pub fn deinit(p: *Pool) void {
        for (p.slots.items) |s| {
            p.gpa.free(s.pages);
            p.gpa.destroy(s);
        }
        p.slots.deinit(p.gpa);
        for (0..p.world) |r| p.free[r].deinit(p.gpa);
        p.gpa.free(p.refs);
        p.gpa.free(p.slot_refs);
        p.* = undefined;
    }

    pub fn newSlot(p: *Pool, capacity: u64) Error!*Slot {
        const s = try p.gpa.create(Slot);
        errdefer p.gpa.destroy(s);
        const mp = pagesFor(capacity, p.page);
        s.* = .{ .pool = p, .id = @intCast(p.slots.items.len), .capacity = capacity, .max_pages = mp, .pages = try p.gpa.alloc(u32, mp) };
        errdefer p.gpa.free(s.pages);
        try p.slots.append(p.gpa, s);
        return s;
    }

    /// A free physical page for logical page k (its residue when split), held once by the caller.
    pub fn takeAt(p: *Pool, k: u32) Error!u32 {
        const r = k % p.world;
        const local = p.free[r].takeLowest() orelse return error.PoolExhausted;
        const pg = local * p.world + r;
        p.hold(pg, 1, 0); // the caller's hold (an entry-like holder until a slot maps it)
        return pg;
    }

    /// Holders change on page pg: dref total, dslot of them slots; keeps entry_only and gives the page back at zero.
    fn hold(p: *Pool, pg: u32, dref: i32, dslot: i32) void {
        const r0 = p.refs[pg];
        const s0 = p.slot_refs[pg];
        const r1: u16 = @intCast(@as(i32, r0) + dref);
        const s1: u16 = @intCast(@as(i32, s0) + dslot);
        p.refs[pg] = r1;
        p.slot_refs[pg] = s1;
        const was = r0 > 0 and s0 == 0;
        const now = r1 > 0 and s1 == 0;
        if (was and !now) p.entry_only -= 1;
        if (!was and now) p.entry_only += 1;
    }

    /// One holder lets go (slot: from a slot's table); the page returns to its free set when nobody holds it.
    fn release(p: *Pool, pg: u32, slot: u1) Error!void {
        if (p.refs[pg] == 0) return error.DoubleFree;
        p.hold(pg, -1, -@as(i32, slot));
        if (p.refs[pg] == 0) p.free[pg % p.world].give(pg / p.world) catch return error.DoubleFree;
    }

    /// One more holder (a session entry or an in-flight job) of each page.
    pub fn share(p: *Pool, pages: []const u32) void {
        for (pages) |pg| p.hold(pg, 1, 0);
    }

    /// Those holders let go.
    pub fn drop(p: *Pool, pages: []const u32) Error!void {
        for (pages) |pg| try p.release(pg, 0);
    }

    pub fn holders(p: *const Pool, pg: u32) u16 {
        return p.refs[pg];
    }

    fn unreserve(p: *Pool, by: []const u32) void {
        for (0..p.world) |r| p.reserved[r] -= by[r];
    }

    fn addReserved(p: *Pool, by: []const u32) void {
        for (0..p.world) |r| p.reserved[r] += by[r];
    }

    /// Free pages no admission reserved (split: W x the scarcer residue, so any logical run fits).
    pub fn available(p: *const Pool) u32 {
        var low: u32 = std.math.maxInt(u32);
        for (0..p.world) |r| low = @min(low, p.free[r].count() -| p.reserved[r]);
        return low * p.world;
    }

    pub fn freePages(p: *const Pool) u32 {
        var n: u32 = 0;
        for (0..p.world) |r| n += p.free[r].count();
        return n;
    }

    /// Pages an admission needs for a request (prompt + reply + slack, at most the slot's capacity).
    pub fn needPages(p: *const Pool, prompt: u64, max_tokens: u64, capacity: u64, slack: u64) u32 {
        return pagesFor(@min(prompt + max_tokens + slack, capacity), p.page);
    }

    /// This rank stores the split families' rows of page pg.
    pub fn owns(p: *const Pool, pg: u32) bool {
        return pg != p.null_page and pg % p.world == p.rank;
    }

    /// A split family's local page of pg: its own when owned, the local null for the null page, else the discard page.
    pub fn localPage(p: *const Pool, pg: u32) u32 {
        if (pg == p.null_page) return p.local_null;
        return if (pg % p.world == p.rank) pg / p.world else p.discard;
    }

    /// The page a family's tensor indexes for physical page pg (split families: localPage).
    pub fn familyPage(p: *const Pool, f: Family, pg: u32) u32 {
        return if (f.split and p.world > 1) p.localPage(pg) else pg;
    }

    /// Pages a family's device tensor holds: npages + null, or (split) local pages + local null + discard page.
    pub fn familyTensorPages(p: *const Pool, f: Family) u32 {
        return if (f.split and p.world > 1) p.local_pages + 2 else p.npages + 1;
    }

    /// Bytes this rank stores of family f for logical pages [first, first + n) of a slot or entry.
    pub fn familyBytes(p: *const Pool, f: Family, first: u32, n: u32) u64 {
        if (!(f.split and p.world > 1)) return @as(u64, n) * f.pageBytes(p.page);
        var by: [max_world]u32 = undefined;
        span(first, first + n, p.world, &by);
        return @as(u64, by[p.rank]) * f.pageBytes(p.page);
    }

    /// Bytes of every family's device tensor on this rank.
    pub fn tensorBytes(p: *const Pool) u64 {
        var b: u64 = 0;
        for (p.layout.families) |f| b += @as(u64, p.familyTensorPages(f)) * f.pageBytes(p.page);
        return b;
    }

    /// Pages held by session entries alone and their bytes on this rank (the residency budget's measure): O(1).
    pub fn entryOnlyBytes(p: *const Pool) u64 {
        return @as(u64, p.entry_only) * p.layout.pageBytes(p.world);
    }
};

test "residues, holders, quotas and the device tables" {
    const gpa = std.testing.allocator;
    const fams = [_]Family{ .{ .name = "comp", .ratio = 2, .row_bytes = 584, .split = true }, .{ .name = "index_k", .ratio = 2, .row_bytes = 132 } };
    var p = try Pool.init(gpa, .{ .families = &fams }, 16 * 256, 2, 1);
    defer p.deinit();
    try std.testing.expectEqual(@as(u32, 8 + 2), p.familyTensorPages(fams[0]));
    try std.testing.expectEqual(@as(u32, 16 + 1), p.familyTensorPages(fams[1]));
    try std.testing.expectApproxEqAbs(@as(f64, 146 + 66), p.layout.bytesPerToken(2), 1e-9);
    const a = try p.newSlot(8 * 256);
    const b = try p.newSlot(8 * 256);
    try a.ensure(3 * 256 + 1); // 4 pages: residues 0,1,0,1
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, a.mapped());
    try b.ensure(256);
    try std.testing.expectEqualSlices(u32, &.{4}, b.mapped());
    try std.testing.expectEqual(p.discard, a.localTableAt(0)); // page 0 is rank 0's
    try std.testing.expectEqual(@as(u32, 0), a.localTableAt(1)); // page 1 -> rank 1's local page 0
    try std.testing.expectEqual(p.local_null, a.localTableAt(7));
    const d = a.takeDirty().?;
    try std.testing.expectEqual(@as(u32, 0), d.lo);
    try std.testing.expectEqual(@as(u32, 4), d.hi);
    try std.testing.expect(a.takeDirty() == null);
    // an entry shares a's pages; a lets go; the pages stay held by the entry alone
    p.share(a.mapped()[0..3]);
    try std.testing.expectEqual(@as(u32, 0), p.entry_only);
    _ = try a.releaseAll();
    try std.testing.expectEqual(@as(u32, 3), p.entry_only);
    try std.testing.expectEqual(@as(u32, 16 - 1 - 3), p.freePages());
    try p.drop(&.{ 0, 1, 2 });
    try std.testing.expectEqual(@as(u32, 0), p.entry_only);
    try std.testing.expectError(error.DoubleFree, p.drop(&.{0}));
    // quotas: b reserves 6 pages; available counts W x the scarcer residue
    b.reserve(6);
    var by: [2]u32 = undefined;
    span(1, 6, 2, &by);
    try std.testing.expectEqual(@as(u32, 2), by[0]);
    try std.testing.expectEqual(@as(u32, 3), by[1]);
    try std.testing.expectEqual(@as(u32, 2 * @min(7 - 2, 8 - 3)), p.available());
    try b.ensure(6 * 256);
    try std.testing.expectEqual(@as(u32, 0), p.reserved[0] + p.reserved[1]);
    for (b.mapped(), 0..) |pg, k| try std.testing.expectEqual(k % 2, pg % 2);
    // RAM resume: adopt full pages of the right residues, refuse the wrong ones
    const c = try p.newSlot(8 * 256);
    try std.testing.expectError(error.ResidueMismatch, c.adopt(&.{b.pages[1]}));
    p.share(b.mapped()[0..2]);
    try c.adopt(b.mapped()[0..2]);
    try c.push(try p.takeAt(2));
    try std.testing.expectEqual(@as(u16, 3), p.holders(b.pages[0]));
    try std.testing.expectEqual(@as(u32, 0), c.pages[2] % 2);
}
