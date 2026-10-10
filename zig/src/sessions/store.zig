//! The session store: RAM entries holding pool pages and bounded state, a residency budget, and the NVMe tier.

const std = @import("std");
const Io = std.Io;
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Slot = pool_mod.Slot;
const PageStore = @import("pagestore.zig").PageStore;
const prefix = @import("prefix.zig");
const Digest = prefix.Digest;
const disk_mod = @import("disk.zig");
const Disk = disk_mod.Disk;
const jobs = @import("jobs.zig");
const Job = jobs.Job;

pub const none = prefix.none;

/// A slot's bounded state (rings, carries, lookback, taps) as the family snapshots it: opaque to the store.
pub const Bounded = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        nbytes: *const fn (ptr: *anyopaque) u64,
        /// host copies of its pieces for a park (owned by the caller: name and bytes allocated with gpa)
        blobs: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator) anyerror![]disk_mod.Blob,
        release: *const fn (ptr: *anyopaque) void,
    };
};

pub const Policy = struct {
    /// bounded state the RAM tier keeps (least recently used entries go past it)
    ram_bytes: u64 = 512 << 20,
    /// idle chats resident in the pool (0: no limit) and pool bytes held by entries alone (0: no limit)
    chats: u32 = 0,
    resident_bytes: u64 = 0,
    /// park an evicted chat's older turns too (default: drop them, the leaf's file covers them)
    park_prefixes: bool = false,
    /// drop, never write, an entry a newer entry of its chain covers
    skip_covered: bool = true,
    /// write a newer turn as a delta on the file its leading pages came from
    delta: bool = true,
    /// parks and restores on the tier thread (settled by the plan)
    async_io: bool = false,
};

pub const Extra = struct {
    /// RAM: the pool pages it holds, logical [0, npages)
    pages: []u32 = &.{},
    bounded: ?Bounded = null,
    /// a file with this key is registered on the NVMe tier
    file: bool = false,
    used: u64 = 0,
    origin: ?pool_mod.Origin = null,
    /// a newer entry of its chain was parked from RAM: writing this one again is redundant
    covered_by: bool = false,
    kind: u32 = 0,
    /// an unsettled park of it
    job: ?*Job = null,
    older: u32 = none,
    newer: u32 = none,
};

const Index = prefix.Index(Extra);

pub const Hit = struct { id: u32, ram: bool, pos: u64 };

pub const Stats = struct {
    saves: u64 = 0,
    dups: u64 = 0,
    hits_ram: u64 = 0,
    hits_disk: u64 = 0,
    parks: u64 = 0,
    delta_parks: u64 = 0,
    drops: u64 = 0,
    covered_drops: u64 = 0,
    chat_evictions: u64 = 0,
    park_ns: u64 = 0,
    restore_ns: u64 = 0,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: Io,
    pool: *Pool,
    pages: PageStore,
    index: Index,
    disk: ?*Disk,
    /// heap-held: the tier thread keeps its address
    runner: ?*jobs.Runner = null,
    policy: Policy,
    clock: u64 = 0,
    bounded_total: u64 = 0,
    /// RAM entries, least recently used first
    oldest: u32 = none,
    newest: u32 = none,
    ram_count: u32 = 0,
    pending: std.ArrayList(*Job) = .empty,
    scratch: std.ArrayList(Digest) = .empty,
    stats: Stats = .{},

    pub fn init(gpa: std.mem.Allocator, io: Io, pool: *Pool, pages: PageStore, d: ?*Disk, policy: Policy) !Store {
        var s: Store = .{ .gpa = gpa, .io = io, .pool = pool, .pages = pages, .index = Index.init(gpa, pool.page), .disk = d, .policy = policy };
        if (d) |dd| {
            const r = try gpa.create(jobs.Runner);
            r.* = jobs.Runner.init(gpa, io, dd);
            s.runner = r;
            if (policy.async_io) r.startThread() catch |e| {
                gpa.destroy(r);
                return e;
            };
        }
        return s;
    }

    pub fn deinit(s: *Store) void {
        if (s.runner) |r| {
            for (s.pending.items) |j| r.wait(j);
            r.deinit();
            s.gpa.destroy(r);
        }
        for (s.pending.items) |j| j.deinit(s.gpa);
        s.pending.deinit(s.gpa);
        for (s.index.entries.items) |*e| if (e.live) {
            if (e.extra.bounded) |b| b.vtable.release(b.ptr);
            s.pool.drop(e.extra.pages) catch {};
            s.gpa.free(e.extra.pages);
        };
        s.index.deinit();
        s.scratch.deinit(s.gpa);
        s.* = undefined;
    }

    fn digests(s: *Store, tag: u32, ids: []const i32) ![]const Digest {
        const n = ids.len / s.pool.page;
        try s.scratch.resize(s.gpa, n);
        prefix.chain(tag, ids, s.pool.page, s.scratch.items);
        return s.scratch.items;
    }

    pub fn entry(s: *Store, id: u32) *Index.Entry {
        return s.index.get(id);
    }

    /// The longest stored entry (RAM or NVMe) whose ids strictly prefix prompt, with the same tag.
    pub fn find(s: *Store, tag: u32, prompt: []const i32) !?Hit {
        const d = try s.digests(tag, prompt);
        const id = s.index.find(tag, prompt, d, .any) orelse return null;
        const e = s.index.get(id);
        return .{ .id = id, .ram = e.ram, .pos = e.pos };
    }

    fn link(s: *Store, id: u32) void {
        const x = &s.index.get(id).extra;
        x.older = s.newest;
        x.newer = none;
        if (s.newest != none) s.index.get(s.newest).extra.newer = id else s.oldest = id;
        s.newest = id;
    }

    fn unlink(s: *Store, id: u32) void {
        const x = &s.index.get(id).extra;
        if (x.older != none) s.index.get(x.older).extra.newer = x.newer else s.oldest = x.newer;
        if (x.newer != none) s.index.get(x.newer).extra.older = x.older else s.newest = x.older;
    }

    fn touch(s: *Store, id: u32) void {
        s.clock += 1;
        const e = s.index.get(id);
        e.extra.used = s.clock;
        if (e.ram) {
            s.unlink(id);
            s.link(id);
        }
    }

    /// An entry for a slot's state at len(ids), sharing its pages; null for a duplicate (bounded is released then).
    pub fn save(s: *Store, slot: *Slot, bounded: Bounded, ids: []const i32, tag: u32, kind: u32) !?u32 {
        if (ids.len == 0) return error.EmptyEntry;
        const need = pool_mod.pagesFor(ids.len, s.pool.page);
        if (slot.len < need) return error.SlotTooShort;
        const d = try s.digests(tag, ids);
        const ins = try s.index.insert(tag, ids, d, false, .{ .kind = kind });
        errdefer if (ins.new) s.index.remove(ins.id);
        const e = s.index.get(ins.id);
        if (e.ram) {
            bounded.vtable.release(bounded.ptr);
            s.touch(ins.id);
            s.stats.dups += 1;
            return null;
        }
        const pages = try s.gpa.dupe(u32, slot.mapped()[0..need]);
        errdefer s.gpa.free(pages);
        try s.index.setRam(ins.id, true, ids);
        s.pool.share(pages);
        const x = &s.index.get(ins.id).extra;
        x.pages = pages;
        x.bounded = bounded;
        x.kind = kind;
        if (slot.origin) |o| x.origin = .{ .key = o.key, .pages = @min(o.pages, @as(u32, @intCast(ids.len / s.pool.page))) };
        s.bounded_total += bounded.vtable.nbytes(bounded.ptr);
        s.ram_count += 1;
        s.link(ins.id);
        s.touch(ins.id);
        s.stats.saves += 1;
        try s.trimRam();
        _ = try s.enforce();
        return ins.id;
    }

    /// A RAM entry into an empty slot: full pages shared, the partial last page copied; returns its bounded state.
    pub fn restoreRam(s: *Store, id: u32, slot: *Slot) !Bounded {
        const e = s.index.get(id);
        if (!e.ram) return error.NotInRam;
        const full: u32 = @intCast(e.pos / s.pool.page);
        try slot.adopt(e.extra.pages[0..full]);
        errdefer _ = slot.releaseAll() catch {};
        if (e.pos % s.pool.page != 0) {
            const src = e.extra.pages[full];
            const dst = try s.pool.takeAt(full);
            for (s.pool.layout.families, 0..) |f, fi| {
                if (f.split and !s.pool.owns(src)) continue;
                try s.pages.copyPage(@intCast(fi), s.pool.familyPage(f, src), s.pool.familyPage(f, dst));
            }
            try slot.push(dst);
        }
        if (e.extra.origin) |o| slot.origin = .{ .key = o.key, .pages = @min(o.pages, full) };
        s.touch(id);
        s.stats.hits_ram += 1;
        return e.extra.bounded.?;
    }

    /// Starts an NVMe resume into an empty slot (pages mapped now, filled by the job); settle it before the slot runs.
    pub fn beginRestore(s: *Store, id: u32, slot: *Slot) !*Job {
        const d = s.disk orelse return error.NoDisk;
        const e = s.index.get(id);
        if (!e.extra.file) return error.NotOnDisk;
        if (slot.len != 0) return error.SlotNotEmpty;
        try slot.ensure(e.pos);
        const j = try s.gpa.create(Job);
        j.* = .{ .kind = .restore, .entry = id, .slot = slot.id };
        errdefer j.deinit(s.gpa);
        var ext: std.ArrayList(disk_mod.Extent) = .empty;
        errdefer ext.deinit(s.gpa);
        try d.extents(e.key, &ext);
        j.extents = try ext.toOwnedSlice(s.gpa);
        const np = pool_mod.pagesFor(e.pos, s.pool.page);
        j.lists = try s.familyLists(slot.mapped()[0..np], 0, false);
        j.sink = .{ .store = s.pages, .families = s.pool.layout.families, .page = s.pool.page, .world = s.pool.world, .rank = s.pool.rank, .pages = j.lists };
        try s.pending.append(s.gpa, j);
        try s.runner.?.submit(j);
        return j;
    }

    /// Per family: family-tensor pages of logical [first, len(pages)) (owned_only: this rank's of split families).
    fn familyLists(s: *Store, pages: []const u32, first: u32, owned_only: bool) ![][]u32 {
        const fams = s.pool.layout.families;
        const out = try s.gpa.alloc([]u32, fams.len);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |l| s.gpa.free(l);
            s.gpa.free(out);
        }
        for (fams, 0..) |f, fi| {
            const w = s.pool.world;
            const sharded = f.split and w > 1 and owned_only;
            var n: usize = pages.len - first;
            if (sharded) {
                var by: [pool_mod.max_world]u32 = undefined;
                pool_mod.span(first, @intCast(pages.len), w, &by);
                n = by[s.pool.rank];
            }
            const l = try s.gpa.alloc(u32, n);
            var i: usize = 0;
            var k: usize = first;
            if (sharded) k = first + (s.pool.rank + w - first % w) % w;
            while (k < pages.len) : (k += if (sharded) w else 1) {
                l[i] = s.pool.familyPage(f, pages[k]);
                i += 1;
            }
            out[fi] = l;
            made += 1;
        }
        return out;
    }

    /// Starts a park of a RAM entry (null: nothing to write, the entry is already settled out of RAM).
    pub fn beginPark(s: *Store, id: u32) !?*Job {
        const d = s.disk orelse return error.NoDisk;
        const e = s.index.get(id);
        if (!e.ram or e.extra.job != null) return error.NotParkable;
        if (e.extra.file and d.get(e.key) != null) {
            try s.leaveRam(id);
            return null;
        }
        const page = s.pool.page;
        const npages = pool_mod.pagesFor(e.pos, page);
        var base: ?Digest = null;
        var first: u32 = 0;
        if (s.policy.delta) if (e.extra.origin) |o| if (o.pages > 0) {
            var r = d.get(o.key);
            while (r) |rec| : (r = if (rec.base == disk_mod.none) null else &d.recs.items[rec.base]) {
                if (rec.first_page < o.pages) break;
            }
            if (r) |rec| if (rec.depth < d.opts.max_chain) {
                base = rec.key;
                first = o.pages;
            };
        };
        const j = try s.gpa.create(Job);
        j.* = .{ .kind = .park, .entry = id };
        errdefer j.deinit(s.gpa);
        j.ids = try s.gpa.alloc(i32, e.pos);
        try s.index.idsOf(id, j.ids);
        j.lists = try s.familyLists(e.extra.pages, first, true);
        j.blobs = try e.extra.bounded.?.vtable.blobs(e.extra.bounded.?.ptr, s.gpa);
        j.src = .{ .key = e.key, .base = base, .tag = e.tag, .kind = e.extra.kind, .page = page, .pos = e.pos, .first_page = first, .npages = npages, .ids = j.ids[first * page ..], .store = s.pages, .fence = try s.pages.fence(), .families = s.pool.layout.families, .pages = j.lists, .blobs = j.blobs };
        if (base) |b| d.get(b).?.refs += 1; // held while the job runs: a trim cannot take the base
        e.extra.job = j;
        try s.pending.append(s.gpa, j);
        try s.runner.?.submit(j);
        return j;
    }

    /// Settles a finished job at the plan's point: a park registers its file, a restore returns the bounded state.
    pub fn settle(s: *Store, j: *Job) !?disk_mod.Restored {
        s.runner.?.wait(j);
        defer {
            for (s.pending.items, 0..) |p, i| if (p == j) {
                _ = s.pending.orderedRemove(i);
                break;
            };
            j.deinit(s.gpa);
        }
        const d = s.disk.?;
        switch (j.kind) {
            .park => {
                if (j.src.base) |b| d.get(b).?.refs -= 1;
                const e = s.index.get(j.entry);
                e.extra.job = null;
                if (j.err) |err| return err; // the entry stays in RAM; the .tmp is gone
                _ = try d.register(&j.src, j.size);
                e.extra.file = true;
                s.stats.parks += 1;
                if (j.src.base != null) s.stats.delta_parks += 1;
                s.stats.park_ns += j.elapsed_ns;
                if (s.policy.skip_covered) {
                    var m: std.ArrayList(u32) = .empty;
                    defer m.deinit(s.gpa);
                    _ = try s.index.members(j.entry, &m, usedOf);
                    for (m.items) |o| s.index.get(o).extra.covered_by = true;
                }
                try s.leaveRam(j.entry);
                try s.trimDisk();
                return null;
            },
            .restore => {
                if (j.err) |err| {
                    // the slot as before the restore: empty, its reservation kept
                    _ = try s.pool.slots.items[j.slot].truncate(0);
                    return err;
                }
                const e = s.index.get(j.entry);
                d.touch(e.key);
                s.touch(j.entry);
                s.pool.slots.items[j.slot].origin = .{ .key = e.key, .pages = @intCast(e.pos / s.pool.page) };
                s.stats.hits_disk += 1;
                s.stats.restore_ns += j.elapsed_ns;
                const r = j.restored;
                j.restored = null;
                return r;
            },
        }
    }

    /// Settles every pending job in submission order (one rank, or the plan's "settle all").
    pub fn settleAll(s: *Store) !void {
        while (s.pending.items.len > 0) {
            if (try s.settle(s.pending.items[0])) |r| {
                var rr = r;
                rr.deinit(s.gpa);
            }
        }
    }

    fn usedOf(x: *const Extra) u64 {
        return x.used;
    }

    /// A RAM entry lets go of its pages and bounded state; it stays indexed if a file holds it.
    fn leaveRam(s: *Store, id: u32) !void {
        const e = s.index.get(id);
        if (!e.ram) return;
        try s.pool.drop(e.extra.pages);
        s.gpa.free(e.extra.pages);
        e.extra.pages = &.{};
        if (e.extra.bounded) |b| {
            s.bounded_total -= b.vtable.nbytes(b.ptr);
            b.vtable.release(b.ptr);
        }
        e.extra.bounded = null;
        s.unlink(id);
        s.ram_count -= 1;
        try s.index.setRam(id, false, null);
        if (!e.extra.file) s.index.remove(id);
    }

    /// An entry whose file could not be read back: the file deleted, the entry gone unless RAM holds it (every rank).
    pub fn forget(s: *Store, id: u32) !void {
        const e = s.index.get(id);
        if (e.extra.job != null) return error.Busy;
        if (e.extra.file) if (s.disk) |d| {
            d.stats.bad += 1;
            // a delta's base stays while a newer file stacks on it (an opt-in delta mode)
            d.remove(e.key) catch |err| return if (err == error.HasDependents) {} else err;
            e.extra.file = false;
        };
        if (!e.ram) s.index.remove(id);
    }

    /// Drops an entry from RAM without writing it.
    pub fn drop(s: *Store, id: u32) !void {
        try s.leaveRam(id);
        s.stats.drops += 1;
    }

    /// Lets go of a RAM entry: dropped when covered, parked if the tier takes it, else dropped; returns a started park.
    pub fn evict(s: *Store, id: u32) !?*Job {
        const e = s.index.get(id);
        if (e.extra.job != null) return null;
        if (s.policy.skip_covered and (e.extra.covered_by or s.index.covered(id, .any))) {
            try s.drop(id);
            s.stats.covered_drops += 1;
            return null;
        }
        if (s.disk) |d| if (d.accepts(e.pos)) {
            const j = try s.beginPark(id) orelse return null;
            if (!s.policy.async_io) _ = try s.settle(j);
            return if (s.policy.async_io) j else null;
        };
        try s.drop(id);
        return null;
    }

    /// Evicts least recently used RAM entries past the bounded-state budget.
    fn trimRam(s: *Store) !void {
        var id = s.oldest;
        while (s.bounded_total > s.policy.ram_bytes and s.ram_count > 1 and id != none) {
            const next = s.index.get(id).extra.newer;
            if (s.index.get(id).extra.job == null) _ = try s.evict(id);
            id = next;
        }
    }

    /// Files the NVMe budget deleted leave the index (a RAM entry with that key stays, unwritten again).
    fn trimDisk(s: *Store) !void {
        const d = s.disk orelse return;
        var gone: std.ArrayList(Digest) = .empty;
        defer gone.deinit(s.gpa);
        try d.trim(&gone);
        for (gone.items) |k| if (s.index.lookup(k)) |id| {
            s.index.get(id).extra.file = false;
            if (!s.index.get(id).ram) s.index.remove(id);
        };
    }

    const Chat = struct { leaf: u32, recency: u64 };

    fn chatLess(_: void, a: Chat, b: Chat) bool {
        return a.recency < b.recency or (a.recency == b.recency and a.leaf < b.leaf);
    }

    /// A chat is live while a slot maps its leaf's first page.
    fn idleLeaf(s: *Store, id: u32) bool {
        const e = s.index.get(id);
        return e.extra.job == null and s.index.isLeaf(id) and (e.extra.pages.len == 0 or s.pool.slot_refs[e.extra.pages[0]] == 0);
    }

    /// Evicts LRU idle chats past the residency budget (leaf parked, unshared older turns dropped); returns how many.
    pub fn enforce(s: *Store) !u32 {
        const pol = s.policy;
        if (pol.chats == 0 and pol.resident_bytes == 0) return 0;
        var chats: std.ArrayList(Chat) = .empty;
        defer chats.deinit(s.gpa);
        var members: std.ArrayList(u32) = .empty;
        defer members.deinit(s.gpa);
        var id = s.oldest;
        while (id != none) : (id = s.index.get(id).extra.newer) {
            if (!s.idleLeaf(id)) continue;
            members.clearRetainingCapacity();
            try chats.append(s.gpa, .{ .leaf = id, .recency = try s.index.members(id, &members, usedOf) });
        }
        std.mem.sort(Chat, chats.items, {}, chatLess);
        var idle: u32 = @intCast(chats.items.len);
        var gone: u32 = 0;
        for (chats.items) |c| {
            const over_n = pol.chats > 0 and idle > pol.chats;
            const over_b = pol.resident_bytes > 0 and s.pool.entryOnlyBytes() > pol.resident_bytes;
            if (!over_n and !over_b) break;
            if (!s.index.get(c.leaf).ram or !s.idleLeaf(c.leaf)) continue;
            members.clearRetainingCapacity();
            _ = try s.index.members(c.leaf, &members, usedOf);
            _ = try s.evict(c.leaf);
            // deepest first: a member more RAM entries extend than this chat's leaf is a branch point, as is all above
            const own: u32 = if (s.index.get(c.leaf).ram) 1 else 0;
            for (members.items) |m| {
                if (s.index.get(m).extra.job != null) continue;
                if (s.index.extensions(m) > own) break;
                if (pol.park_prefixes) _ = try s.evict(m) else try s.drop(m);
            }
            idle -= 1;
            gone += 1;
            s.stats.chat_evictions += 1;
        }
        return gone;
    }

    /// RAM entries to let go of (coldest first, never keep) so need pages are free: [] if they are, null if impossible.
    pub fn planSpill(s: *Store, need: u32, keep: []const u32, out: *std.ArrayList(u32)) !bool {
        out.clearRetainingCapacity();
        var avail = s.pool.available();
        if (avail >= need) return true;
        var left: std.AutoHashMapUnmanaged(u32, u16) = .empty;
        defer left.deinit(s.gpa);
        var id = s.oldest;
        while (id != none and avail < need) : (id = s.index.get(id).extra.newer) {
            if (std.mem.indexOfScalar(u32, keep, id) != null or s.index.get(id).extra.job != null) continue;
            try out.append(s.gpa, id);
            for (s.index.get(id).extra.pages) |pg| {
                const g = try left.getOrPut(s.gpa, pg);
                if (!g.found_existing) g.value_ptr.* = s.pool.refs[pg];
                g.value_ptr.* -= 1;
                if (g.value_ptr.* == 0) avail += 1; // split: a page of the scarcer residue may count for less; the check below decides
            }
        }
        return avail >= need;
    }

    /// Rebuilds NVMe entries on restart (ranks then intersect them), deltas on their bases' paths; returns how many.
    pub fn reconcile(s: *Store) !u32 {
        const d = s.disk orelse return 0;
        var found: std.ArrayList(Disk.Found) = .empty;
        defer {
            for (found.items) |f| s.gpa.free(f.ids);
            found.deinit(s.gpa);
        }
        try d.scan(&found);
        for (found.items) |f| {
            var anchor: u32 = none;
            if (f.head.hasBase()) {
                const b = s.index.lookup(f.head.base) orelse return error.NoBase;
                anchor = s.index.nodeAt(b, f.head.first_page) orelse return error.BadChain;
            }
            const r = try s.index.insertFrom(f.head.tag, anchor, f.ids, .{ .file = true, .kind = f.head.kind });
            s.clock += 1;
            s.index.get(r.id).extra.used = s.clock;
        }
        return @intCast(found.items.len);
    }

    /// Idle resident chats and the pool bytes held by entries alone.
    pub fn resident(s: *Store) struct { chats: u32, bytes: u64 } {
        var n: u32 = 0;
        var id = s.oldest;
        while (id != none) : (id = s.index.get(id).extra.newer) if (s.idleLeaf(id)) {
            n += 1;
        };
        return .{ .chats = n, .bytes = s.pool.entryOnlyBytes() };
    }
};
