//! RAM publication and partial-page restoration ownership for Store.
const pool_mod = @import("pool.zig");
const Slot = pool_mod.Slot;
const sm = @import("store.zig");
const Store = sm.Store;
const Bounded = sm.Bounded;

// Success transfers bounded; failure rolls back this publication, leaves ownership and may retain older evictions.
pub fn save(s: *Store, slot: *Slot, bounded: Bounded, ids: []const i32, tag: u32, kind: u32) !?u32 {
    if (ids.len == 0) return error.EmptyEntry;
    const need = pool_mod.pagesFor(ids.len, s.pool.page);
    if (slot.len < need) return error.SlotTooShort;
    const d = try s.digests(tag, ids);
    const ins = try s.index.insert(tag, ids, d, false, .{ .kind = kind });
    var published = false;
    errdefer if (ins.new and !published) s.index.remove(ins.id);
    const e = s.index.get(ins.id);
    if (e.ram) {
        bounded.vtable.release(bounded.ptr);
        s.touch(ins.id);
        s.stats.dups += 1;
        return null;
    }
    const previous = e.extra;
    const pages = try s.gpa.dupe(u32, slot.mapped()[0..need]);
    errdefer if (!published) s.gpa.free(pages);
    try s.index.setRam(ins.id, true, ids);
    s.pool.share(pages);
    const x = &s.index.get(ins.id).extra;
    x.pages = pages;
    x.bounded = bounded;
    x.kind = kind;
    if (slot.origin) |o| x.origin = .{ .key = o.key, .pages = @min(o.pages, @as(u32, @intCast(ids.len / s.pool.page))) };
    const nbytes = bounded.vtable.nbytes(bounded.ptr);
    s.bounded_total += nbytes;
    s.ram_count += 1;
    s.link(ins.id);
    s.touch(ins.id);
    s.stats.saves += 1;
    // Pre-publication errdefers stop here. On error the caller still owns bounded.
    published = true;
    errdefer {
        const extra = &s.index.get(ins.id).extra;
        extra.bounded = null;
        s.bounded_total -= nbytes;
        // Keep an existing non-RAM entry indexed even if it has no file flag.
        if (!ins.new) extra.file = true;
        s.leaveRam(ins.id) catch unreachable;
        if (!ins.new) {
            var prior = previous;
            // Earlier disk policy commits survive this RAM publication's rollback.
            if (s.disk) |dsk| prior.file = dsk.by_key.contains(s.index.get(ins.id).key);
            s.index.get(ins.id).extra = prior;
            if (previous.file and !prior.file) s.index.remove(ins.id);
        }
        s.stats.saves -= 1;
    }
    try s.trimRam(ins.id);
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
        errdefer s.pool.drop(&.{dst}) catch unreachable;
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
