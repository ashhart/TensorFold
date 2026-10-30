//! The host tier: two currencies, eviction order, pins, generations, canonical rows, lockstep ranks, mutants.
const std = @import("std");
const kv = @import("kv_pool.zig");
const kt = @import("kv_tier.zig");
const Tier = kt.Tier;
const Lease = kt.Lease;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const gpa = std.testing.allocator;

pub const page_bytes = 8;

/// A tier of `pages` logical pages and `frames` frames over a host store and a fake device.
pub const Rig = struct {
    host: []u8,
    dev: []u8,
    tier: Tier,

    pub fn open(r: *Rig, pages: usize, frames: usize) !void {
        r.host = try gpa.alloc(u8, pages * page_bytes);
        errdefer gpa.free(r.host);
        @memset(r.host, 0);
        r.dev = try gpa.alloc(u8, frames * page_bytes);
        errdefer gpa.free(r.dev);
        @memset(r.dev, 0);
        try r.tier.init(gpa, r.host, frames, page_bytes);
    }

    pub fn close(r: *Rig) void {
        r.tier.deinit();
        gpa.free(r.host);
        gpa.free(r.dev);
    }

    pub fn frame(r: *Rig, f: u32) []u8 {
        return r.dev[f * page_bytes ..][0..page_bytes];
    }

    /// A stream with `pages` pages covered and `frames` frames promised.
    pub fn lease(r: *Rig, pages: usize, frames: usize) !Lease {
        return r.leaseLent(pages, frames, 0);
    }

    /// A stream that also has `lent` of its frames lent for its life.
    pub fn leaseLent(r: *Rig, pages: usize, frames: usize, lent: usize) !Lease {
        var l: Lease = .{};
        errdefer r.tier.finish(&l);
        try r.tier.reserve(&l, pages, frames);
        try r.tier.cover(&l, pages, null);
        try r.tier.lend(&l, lent);
        return l;
    }

    /// Pin `id`, copying its canonical rows in on a miss.
    pub fn pin(r: *Rig, l: *Lease, id: u32) !kt.Fill {
        const f = try r.tier.pin(l, id);
        if (f.miss) @memcpy(r.frame(f.frame), r.tier.host_page(id));
        return f;
    }

    /// Write `b` into every row byte of the pinned page and flush it to the host store.
    pub fn write(r: *Rig, l: *const Lease, id: u32, b: u8) void {
        const tk = r.tier.resident(id).?;
        @memset(r.frame(tk.frame), b);
        r.tier.markDirty(l, id) catch unreachable;
    }

    pub fn flush(r: *Rig, id: u32) void {
        const tk = r.tier.resident(id).?;
        @memcpy(r.tier.host_page(id), r.frame(tk.frame));
        r.tier.sync(id);
    }
};

test "two currencies: a promise that fits one and not the other takes nothing, and finishing returns both" {
    var r: Rig = undefined;
    try r.open(8, 3);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(4, 2);
    try std.testing.expectError(error.FrameMemory, t.reserve(&a, 1, 2)); // frames: 2 + 2 > 3
    try expectEqual(@as(usize, 0), a.pages.promised);
    try expectEqual(@as(usize, 2), t.promised);
    var b: Lease = .{};
    try std.testing.expectError(error.LaneMemory, t.reserve(&b, 5, 1)); // pages: 4 + 5 > 8
    try expectEqual(@as(usize, 2), t.promised); // the frame promise was not kept
    try expectEqual(@as(usize, 0), t.pool.promised);
    try expect(!t.fits(5, 1) and t.fits(4, 1) and !t.fits(4, 2));
    var c = try r.lease(4, 1);
    try expect(!t.fits(1, 0)); // every page is promised or taken
    t.finish(&a);
    t.finish(&c);
    t.finish(&b);
    try expect(t.idle());
    try t.audit();
}

test "a lease cannot pin more frames than it promised, and a failed batch pins nothing" {
    var r: Rig = undefined;
    try r.open(8, 4);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(4, 2);
    var out: [3]kt.Fill = undefined;
    const ids = a.pages.ids.items;
    try std.testing.expectError(error.FrameMemory, t.fill(&a, ids[0..3], &out));
    try expectEqual(@as(usize, 0), t.census().pinned);
    try expectEqual(@as(usize, 0), a.pins.items.len);
    try expect(t.resident(ids[0]) == null and t.resident(ids[1]) == null); // the misses of the failed batch were unmapped
    try t.fill(&a, ids[0..2], out[0..2]);
    try expect(out[0].miss and out[1].miss);
    try expectEqual(@as(usize, 2), t.census().pinned);
    try std.testing.expectError(error.FrameMemory, t.pin(&a, ids[2]));
    t.finish(&a);
    try expect(t.idle());
}

test "eviction takes the least recently used unpinned frame, and a touch moves a page to the back" {
    var r: Rig = undefined;
    try r.open(8, 3);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(6, 1);
    const ids = a.pages.ids.items;
    for (ids[0..3]) |id| { // one frame promised: pin, read, unpin, three times over
        _ = try r.pin(&a, id);
        try t.unpin(&a, id);
    }
    try expectEqual(@as(usize, 3), t.census().resident);
    t.touch(ids[0]); // order now 1, 2, 0
    const f = try r.pin(&a, ids[3]); // evicts page 1
    try expect(f.miss);
    try expect(t.resident(ids[1]) == null and t.resident(ids[0]) != null and t.resident(ids[2]) != null);
    try t.unpin(&a, ids[3]);
    try expectEqual(@as(usize, 1), t.evict(1)); // page 2 is next
    try expect(t.resident(ids[2]) == null and t.resident(ids[0]) != null);
    try expectEqual(@as(usize, 2), t.evict(5)); // 0 and 3: all that is left
    try expectEqual(@as(usize, 3), t.census().free);
    try t.audit();
    t.finish(&a);
}

test "pinned frames are never evicted, and a dirty page stays until its rows reach the host store" {
    var r: Rig = undefined;
    try r.open(8, 2);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(2, 1);
    var b = try r.lease(2, 1);
    const pa = a.pages.ids.items[0];
    const pb = b.pages.ids.items;
    _ = try r.pin(&a, pa);
    r.write(&a, pa, 7);
    _ = try r.pin(&b, pb[0]);
    try t.unpin(&b, pb[0]);
    _ = try r.pin(&b, pb[1]); // the one frame that can go is pb[0]'s
    try expect(t.resident(pa) != null and t.resident(pb[0]) == null);
    try std.testing.expectError(error.Dirty, t.unpin(&a, pa));
    try expectEqual(@as(usize, 0), t.evict(2)); // pa pinned and dirty, pb[1] pinned
    try t.unpin(&b, pb[1]);
    try expectEqual(@as(usize, 1), t.evict(2)); // only pb[1]
    try expect(t.resident(pa) != null);
    try std.testing.expectError(error.Unsynced, t.canonical(pa));
    r.flush(pa);
    try t.unpin(&a, pa);
    try expectEqual(@as(u8, 7), (try t.canonical(pa))[3]);
    try expectEqual(@as(usize, 1), t.evict(2));
    t.finish(&a);
    t.finish(&b);
    try expect(t.idle());
}

/// Whether a table entry written before a frame was dropped and filled again with the same page is taken for current.
fn staleAccepted(fault: kt.Fault) !bool {
    var r: Rig = undefined;
    try r.open(4, 1);
    defer r.close();
    r.tier.fault = fault;
    var a = try r.lease(2, 1);
    defer r.tier.finish(&a);
    const p = a.pages.ids.items;
    const f0 = try r.pin(&a, p[0]);
    const old: kt.Ticket = .{ .frame = f0.frame, .gen = f0.gen };
    try r.tier.unpin(&a, p[0]);
    _ = try r.pin(&a, p[1]); // the only frame is recycled
    try r.tier.unpin(&a, p[1]);
    _ = try r.pin(&a, p[0]); // and returns to the first page: same frame, new contents
    try r.tier.unpin(&a, p[0]);
    return r.tier.valid(p[0], old);
}

test "generations: a recycled frame is not read through a table written before" {
    try expect(!try staleAccepted(.none));
    try expect(try staleAccepted(.keep_generation)); // mutation: without the generation the stale table passes
}

/// Whether a frame a call pins survives another stream's miss.
fn pinnedSurvives(fault: kt.Fault) !bool {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    r.tier.fault = fault;
    var a = try r.lease(1, 1);
    defer r.tier.finish(&a);
    var b = try r.lease(2, 1);
    defer r.tier.finish(&b);
    const pa = a.pages.ids.items[0];
    _ = try r.pin(&a, pa); // the oldest stamp, and pinned
    _ = try r.pin(&b, b.pages.ids.items[0]);
    try r.tier.unpin(&b, b.pages.ids.items[0]);
    _ = try r.pin(&b, b.pages.ids.items[1]);
    return r.tier.resident(pa) != null;
}

test "pin rule: the frame a call holds survives a miss; mutation: ignoring pins evicts it" {
    try expect(try pinnedSurvives(.none));
    try expect(!try pinnedSurvives(.ignore_pins));
}

test "freeing a page (its last holder lets go) frees its frame; a page the store still holds keeps it" {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(2, 2);
    const p = try gpa.dupe(u32, a.pages.ids.items); // ids outlive the lease
    defer gpa.free(p);
    _ = try r.pin(&a, p[0]);
    r.write(&a, p[0], 5);
    r.flush(p[0]);
    _ = try r.pin(&a, p[1]);
    t.pool.retain(p[0]); // the prompt store keeps the first page
    t.finish(&a);
    try expect(t.resident(p[0]) != null and t.resident(p[1]) == null);
    try expectEqual(@as(usize, 1), t.census().free);
    try expectEqual(@as(u8, 5), (try t.canonical(p[0]))[0]);
    try t.pool.drop(p[0]);
    try expect(t.resident(p[0]) == null and t.idle());
    try t.audit();
}

/// A fake serving loop: streams open, pin, write, flush, unpin, finish; every row read equals the page's last write.
const Sim = struct {
    r: Rig,
    rng: std.Random.DefaultPrng,
    slots: [3]?Lease = .{ null, null, null },
    model: [16]u8 = @splat(0), // the byte last written to a page (0: none)
    next: u8 = 1,
    misses: usize = 0,
    reads: usize = 0,

    fn open(s: *Sim, seed: u64, frames: usize) !void {
        s.* = .{ .r = undefined, .rng = .init(seed) };
        try s.r.open(16, frames);
    }

    fn close(s: *Sim) void {
        for (&s.slots) |*l| if (l.*) |*x| s.r.tier.finish(x);
        s.r.close();
    }

    fn check(s: *Sim, id: u32, tk: kt.Ticket) !void {
        if (s.model[id] == 0) return;
        for (s.r.frame(tk.frame)) |x| try expectEqual(s.model[id], x);
        s.reads += 1;
    }

    fn step(s: *Sim) !void {
        const t = &s.r.tier;
        const rnd = s.rng.random();
        const op = rnd.uintLessThan(u32, 100);
        const k = rnd.uintLessThan(usize, s.slots.len);
        const l: *Lease = if (s.slots[k]) |*x| x else {
            if (op < 60) s.slots[k] = s.r.leaseLent(4, 3, 1) catch null;
            return;
        };
        const ids = l.pages.ids.items;
        const id = ids[rnd.uintLessThan(usize, ids.len)];
        switch (op) {
            0...34 => {
                const f = s.r.pin(l, id) catch |e| {
                    if (e == error.FrameMemory and l.pins.items.len + l.lent.items.len >= l.frames) return;
                    return e;
                };
                s.misses += @intFromBool(f.miss);
                try s.check(id, .{ .frame = f.frame, .gen = f.gen });
            },
            35...59 => if (l.pins.items.len > 0) {
                const p = l.pins.items[rnd.uintLessThan(usize, l.pins.items.len)];
                s.model[p] = s.next;
                s.next = if (s.next == 255) 1 else s.next + 1;
                s.r.write(l, p, s.model[p]);
            },
            60...79 => if (l.pins.items.len > 0) {
                const p = l.pins.items[rnd.uintLessThan(usize, l.pins.items.len)];
                if (t.dirty[p]) s.r.flush(p);
                try t.unpin(l, p);
            },
            80...89 => _ = t.evict(1),
            else => {
                for (l.pins.items) |p| if (t.dirty[p]) s.r.flush(p);
                for (ids) |p| s.model[p] = 0;
                t.finish(l);
                s.slots[k] = null;
            },
        }
        try t.audit();
        for (t.page_of, 0..) |p, f| if (p != kt.none and !t.dirty[p]) try s.check(p, .{ .frame = @intCast(f), .gen = t.gen[f] });
    }
};

test "a random serving loop with fewer frames than pages: the audit holds, every read equals the last write, and it ends idle" {
    inline for (.{ 1, 2, 3 }) |seed| {
        var s: Sim = undefined;
        try s.open(seed, 6);
        defer s.close();
        for (0..4000) |_| try s.step();
        try expect(s.misses > 100 and s.reads > 500);
        for (&s.slots) |*l| if (l.*) |*x| {
            for (x.pins.items) |p| if (s.r.tier.dirty[p]) s.r.flush(p);
            s.r.tier.finish(x);
            l.* = null;
        };
        _ = s.r.tier.evict(100);
        try expect(s.r.tier.idle());
    }
}

test "two ranks in lockstep: the same openings and pins give the same frames, generations and order, step after step" {
    var a: Sim = undefined;
    var b: Sim = undefined;
    try a.open(9, 6);
    defer a.close();
    try b.open(9, 6);
    defer b.close();
    for (0..3000) |_| {
        try a.step();
        try b.step();
        try expectEqual(a.r.tier.digest(), b.r.tier.digest());
    }
}

test "two ranks: a step only one rank takes is caught by the digest at once" {
    var a: Sim = undefined;
    var b: Sim = undefined;
    try a.open(9, 6);
    defer a.close();
    try b.open(9, 6);
    defer b.close();
    var caught: ?usize = null;
    for (0..3000) |i| {
        try a.step();
        if (i == 1500) _ = b.r.tier.evict(1) else try b.step(); // rank 1 evicts instead of the step rank 0 took
        if (a.r.tier.digest() != b.r.tier.digest()) {
            caught = i;
            break;
        }
    }
    try expect(caught != null);
}

test "ranks agree on the smaller shape before they open" {
    const a: kt.Shape = .{ .host_pages = 100, .frames = 40 };
    const b: kt.Shape = .{ .host_pages = 90, .frames = 48 };
    try expectEqual(kt.Shape{ .host_pages = 90, .frames = 40 }, a.agree(b));
    try expectEqual(a.agree(b), b.agree(a));
}

test "lent frames: counted in the frame promise, never victims or mapped, back at finish with a new generation" {
    var r: Rig = undefined;
    try r.open(8, 4);
    defer r.close();
    const t = &r.tier;
    var a = try r.leaseLent(4, 3, 2); // two lent, one left to pin
    try expectEqual(@as(usize, 2), t.census().lent);
    try expectEqual(@as(usize, 2), a.lent.items.len);
    try std.testing.expectError(error.FrameMemory, t.lend(&a, 2));
    const gen0 = t.gen[a.lent.items[0]];
    var b = try r.lease(4, 1);
    const ids = a.pages.ids.items;
    _ = try r.pin(&a, ids[0]);
    try std.testing.expectError(error.FrameMemory, t.pin(&a, ids[1])); // the lent frames used the promise
    try t.unpin(&a, ids[0]);
    for (b.pages.ids.items) |id| { // misses cycle through the two frames left; the lent ones are never taken
        _ = try r.pin(&b, id);
        try t.unpin(&b, id);
        _ = try r.pin(&a, ids[1]);
        try t.unpin(&a, ids[1]);
        for (a.lent.items) |f| try expect(t.page_of[f] == kt.none);
    }
    try expect(t.evict(10) <= 2); // only the mapped frames can go, the lent ones are not among them
    try t.audit();
    const lent0 = a.lent.items[0];
    t.finish(&a);
    try expect(t.gen[lent0] != gen0);
    try expectEqual(@as(usize, 0), t.census().lent);
    t.finish(&b);
    try expect(t.idle());
    try t.audit();
}

test "ranks lending a different count are told apart by the digest" {
    var a: Rig = undefined;
    var b: Rig = undefined;
    try a.open(8, 4);
    defer a.close();
    try b.open(8, 4);
    defer b.close();
    var la = try a.leaseLent(4, 3, 2);
    defer a.tier.finish(&la);
    var lb = try b.leaseLent(4, 3, 2);
    defer b.tier.finish(&lb);
    try expectEqual(a.tier.digest(), b.tier.digest());
    try b.tier.lend(&lb, 0);
    var lc = try b.leaseLent(0, 1, 1);
    defer b.tier.finish(&lc);
    try expect(a.tier.digest() != b.tier.digest());
}

test "a lent run is consecutive frames, leaves the free frames, and comes back at finish" {
    var r: Rig = undefined;
    try r.open(8, 6);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(2, 4);
    _ = try r.pin(&a, a.pages.ids.items[0]); // a mapped frame in the way
    const first = try t.lendRun(&a, 3);
    try expectEqual(@as(usize, 3), t.census().lent);
    for (a.lent.items, 0..) |f, i| try expectEqual(first + i, f);
    try t.audit();
    var b = try r.lease(2, 2);
    try std.testing.expectError(error.FrameMemory, t.lendRun(&b, 3)); // past b's promise
    _ = try r.pin(&b, b.pages.ids.items[0]);
    _ = try r.pin(&b, b.pages.ids.items[1]);
    try std.testing.expectEqual(@as(usize, 0), t.nfree); // 1 mapped (a) + 3 lent + 2 mapped (b)
    t.finish(&a);
    t.finish(&b);
    try expect(t.idle());
    try t.audit();
    var c = try r.lease(1, 6);
    defer t.finish(&c);
    try expectEqual(@as(u32, 0), try t.lendRun(&c, 6));
}

/// A failed batch, then the same pages pinned again through the rig: the frames must equal the host pages.
fn failedFillReads(fault: kt.Fault) !bool {
    var r: Rig = undefined;
    try r.open(8, 4);
    defer r.close();
    r.tier.fault = fault;
    for (r.host, 0..) |*b, i| b.* = @intCast(1 + i / page_bytes); // page k holds k + 1
    @memset(r.dev, 0xee); // the frames hold something else
    var a = try r.lease(4, 2);
    defer r.tier.finish(&a);
    const ids = a.pages.ids.items;
    var out: [3]kt.Fill = undefined;
    try std.testing.expectError(error.FrameMemory, r.tier.fill(&a, ids[0..3], &out));
    try r.tier.audit();
    for (ids[0..2]) |id| {
        const f = try r.pin(&a, id);
        if (!std.mem.eql(u8, r.frame(f.frame), r.tier.host_page(id))) return false;
    }
    return true;
}

test "a failed batch leaves no miss mapped: the pages pinned again read the host rows" {
    try expect(try failedFillReads(.none));
}

test "a failed batch moves the generation of the frames it unmapped" {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    var a = try r.lease(3, 2);
    defer r.tier.finish(&a);
    const ids = a.pages.ids.items;
    var out: [3]kt.Fill = undefined;
    const before = [2]u32{ r.tier.gen[0], r.tier.gen[1] };
    try std.testing.expectError(error.FrameMemory, r.tier.fill(&a, ids[0..3], &out));
    try expect(r.tier.gen[0] != before[0] and r.tier.gen[1] != before[1]);
    try expectEqual(@as(usize, 2), r.tier.census().free);
}

test "dirty needs a pin by the lease: an unpinned frame cannot be marked; mutation: a dirty frame nobody pins starves a lease inside its promise" {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    var a = try r.lease(1, 1);
    defer r.tier.finish(&a);
    var b = try r.lease(1, 1);
    defer r.tier.finish(&b);
    const pa = a.pages.ids.items[0];
    _ = try r.pin(&a, pa);
    try r.tier.unpin(&a, pa);
    try std.testing.expectError(error.NotPinned, r.tier.markDirty(&a, pa));
    try std.testing.expectError(error.NotPinned, r.tier.markDirty(&b, pa)); // pinned by nobody of b
    try r.tier.audit();
    // the mutant lets the mark through on a frame no lease pins, and the only frame is then stuck
    var m: Rig = undefined;
    try m.open(4, 2);
    defer m.close();
    m.tier.fault = .unpinned_dirty;
    var c = try m.lease(2, 1);
    defer m.tier.finish(&c);
    var d = try m.lease(1, 1);
    defer m.tier.finish(&d);
    for (c.pages.ids.items) |pc| { // both frames hold c's pages, unpinned
        _ = try m.pin(&c, pc);
        try m.tier.unpin(&c, pc);
    }
    for (c.pages.ids.items) |pc| try m.tier.markDirty(&c, pc);
    try std.testing.expectError(error.FrameMemory, m.pin(&d, d.pages.ids.items[0]));
}

/// A lease that pins a dirty page it does not own, after the owner let go of its pin: whether it may end.
fn endsWithForeignDirty(fault: kt.Fault) !bool {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    r.tier.fault = fault;
    var a = try r.lease(1, 1);
    var b = try r.lease(1, 1);
    const pa = a.pages.ids.items[0];
    _ = try r.pin(&a, pa);
    _ = try r.pin(&b, pa);
    r.write(&a, pa, 9);
    try r.tier.unpin(&a, pa); // b still pins it
    const ok = r.tier.canFinish(&b);
    if (ok) {
        r.tier.finish(&b);
        r.flush(pa); // too late for a real stream; the test only wants the tier consistent to close
    } else {
        r.flush(pa);
        r.tier.finish(&b);
    }
    r.tier.finish(&a);
    return ok;
}

test "a stream cannot end while it holds the last pin of a dirty page, its own table or not; mutation: ignoring dirty lets it" {
    try expect(!try endsWithForeignDirty(.none));
    try expect(try endsWithForeignDirty(.ignore_dirty));
}

test "a stream ends cleanly once the dirty pages it pins are flushed, and one that shares a dirty page of its own cannot" {
    var r: Rig = undefined;
    try r.open(4, 2);
    defer r.close();
    const t = &r.tier;
    var a = try r.lease(2, 2);
    const p = a.pages.ids.items;
    _ = try r.pin(&a, p[0]);
    r.write(&a, p[0], 3);
    try expect(t.canFinish(&a)); // its own page and its only holder: the page goes with it
    t.pool.retain(p[0]);
    try expect(!t.canFinish(&a)); // now another holder would read rows the host lacks
    r.flush(p[0]);
    try expect(t.canFinish(&a));
    const keep = p[0];
    t.finish(&a);
    try t.pool.drop(keep);
    try expect(t.idle());
}
