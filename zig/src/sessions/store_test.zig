//! The session store on a host pool and a real NVMe tier: chats, evictions, resumes bit for bit, deltas, two ranks.

const std = @import("std");
const testing = std.testing;
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Slot = pool_mod.Slot;
const Family = pool_mod.Family;
const HostPool = @import("pagestore.zig").HostPool;
const disk_mod = @import("disk.zig");
const store_mod = @import("store.zig");
const Store = store_mod.Store;
const Bounded = store_mod.Bounded;

const page = 64;
const fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584, .split = true },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584, .split = true },
};

/// Bounded state in host memory: one blob naming its chat and length.
const HostBounded = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,

    fn make(gpa: std.mem.Allocator, chat: u32, pos: u64) !Bounded {
        const h = try gpa.create(HostBounded);
        h.* = .{ .gpa = gpa, .bytes = try gpa.alloc(u8, 3000) };
        for (h.bytes, 0..) |*b, i| b.* = @truncate(i +% chat *% 17 +% pos);
        return .{ .ptr = h, .vtable = &vt };
    }
    const vt: Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };
    fn nbytes(p: *anyopaque) u64 {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        return h.bytes.len;
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]disk_mod.Blob {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        const out = try gpa.alloc(disk_mod.Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "rings"), .bytes = try gpa.dupe(u8, h.bytes) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        h.gpa.free(h.bytes);
        h.gpa.destroy(h);
    }
};

/// One rank: its pool, host pages and store, plus what each chat's slot wrote (the reference rows by logical page).
const Rank = struct {
    pool: Pool,
    host: HostPool,
    disk: *disk_mod.Disk,
    store: Store,

    fn init(r: *Rank, gpa: std.mem.Allocator, root: []const u8, world: u32, rank: u32, policy: store_mod.Policy) !void {
        r.pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 96 * page, world, rank);
        r.host = try HostPool.init(gpa, &r.pool);
        r.disk = try disk_mod.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(7), .rank = rank, .world = world, .chunk = 64 << 10, .staging = 4, .lanes = 2, .min_tokens = 1 });
        r.store = try Store.init(gpa, testing.io, &r.pool, r.host.store(), r.disk, policy);
    }

    fn deinit(r: *Rank) void {
        r.store.deinit();
        r.disk.close();
        r.host.deinit();
        r.pool.deinit();
    }

    /// The rows a chat's slot writes at a logical page: a function of (chat, logical page, family) alone.
    fn write(r: *Rank, s: *const Slot, chat: u32, from_page: u32) void {
        for (s.mapped()[from_page..], from_page..) |pg, k| for (fams, 0..) |f, fi| {
            if (f.split and !r.pool.owns(pg)) continue;
            const v = r.host.pageOf(@intCast(fi), r.pool.familyPage(f, pg));
            for (v, 0..) |*b, i| b.* = @truncate(i *% 13 +% k *% 101 +% chat *% 7 +% fi);
        };
    }

    /// Every stored row of the slot's pages below pos equals what write() put there.
    fn check(r: *Rank, s: *const Slot, chat: u32, pos: u64) !void {
        const n = pool_mod.pagesFor(pos, page);
        for (s.mapped()[0..n], 0..) |pg, k| for (fams, 0..) |f, fi| {
            if (f.split and !r.pool.owns(pg)) continue;
            const v = r.host.pageOf(@intCast(fi), r.pool.familyPage(f, pg));
            const rows: usize = if (k + 1 == n and pos % page != 0) (pos % page) / f.ratio * f.row_bytes else v.len;
            for (v[0..rows], 0..) |b, i| try testing.expectEqual(@as(u8, @truncate(i *% 13 +% k *% 101 +% chat *% 7 +% fi)), b);
        };
    }
};

fn chatIds(gpa: std.mem.Allocator, chat: u32, n: usize) ![]i32 {
    const v = try gpa.alloc(i32, n);
    for (v, 0..) |*t, i| t.* = @intCast(chat * 100_000 + i);
    return v;
}

/// Prefill or resume a chat to len(ids) in slot s, save it, free the slot, enforce residency; returns the resume point.
fn turn(gpa: std.mem.Allocator, r: *Rank, s: *Slot, chat: u32, ids: []const i32) ![]const u8 {
    var tier: []const u8 = "fresh";
    var from: u32 = 0;
    if (try r.store.find(1, ids)) |hit| {
        if (hit.ram) {
            const b = try r.store.restoreRam(hit.id, s);
            try testing.expectEqual(@as(u64, 3000), b.vtable.nbytes(b.ptr));
            tier = "ram";
        } else {
            const j = try r.store.beginRestore(hit.id, s);
            var got = (try r.store.settle(j)).?;
            defer got.deinit(gpa);
            try testing.expectEqual(@as(u8, @truncate(chat *% 17 +% hit.pos)), got.blobs[0].bytes[0]);
            tier = "disk";
        }
        try r.check(s, chat, hit.pos);
        from = @intCast(hit.pos / page);
    }
    try s.ensure(ids.len);
    r.write(s, chat, from);
    _ = try r.store.save(s, try HostBounded.make(gpa, chat, ids.len), ids, 1, 0);
    _ = try s.releaseAll();
    _ = try r.store.enforce();
    try r.store.settleAll(); // the plan's settle point (async parks finish here)
    return tier;
}

/// The scenario on one rank; returns a trace of its decisions (stats and page ids) to compare ranks.
fn scenario(gpa: std.mem.Allocator, root: []const u8, world: u32, rank: u32, async_io: bool, trace: *std.ArrayList(u64)) !void {
    var r: Rank = undefined;
    try r.init(gpa, root, world, rank, .{ .chats = 1, .async_io = async_io });
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    const b = try chatIds(gpa, 2, 20 * page);
    defer gpa.free(b);
    const st = &r.store.stats;
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]));
    try testing.expectEqualStrings("ram", try turn(gpa, &r, s, 1, a[0 .. 8 * page + 3]));
    try testing.expectEqual(@as(u64, 0), st.parks);
    // a second idle chat: chat 1 (least recent) parked whole, its older turn dropped (not written)
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 2, b[0 .. 6 * page]));
    try testing.expectEqual(@as(u64, 1), st.parks);
    try testing.expectEqual(@as(u64, 1), st.drops);
    try testing.expectEqual(@as(u32, 1), r.store.resident().chats);
    const whole = r.disk.used;
    // chat 1 resumes from NVMe bit for bit; chat 2 goes
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 1, a[0 .. 12 * page + 40]));
    try testing.expectEqual(@as(u64, 2), st.parks);
    const before = r.disk.used;
    // chat 2 resumes; chat 1 goes again, written as a delta on its first file (pages 8..12 of 13)
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 2, b[0 .. 9 * page]));
    try testing.expectEqual(@as(u64, 3), st.parks);
    try testing.expectEqual(@as(u64, 1), st.delta_parks);
    // the delta holds 5 of the entry's 13 pages: less than chat 1's first whole file of 9 pages
    try testing.expect(r.disk.used - before < whole);
    // chat 1's newest turn restores through its two files, bit for bit
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 1, a[0 .. 15 * page]));
    try testing.expectEqual(@as(u64, 3), st.hits_disk);
    try trace.appendSlice(gpa, &.{ st.parks, st.delta_parks, st.drops, st.covered_drops, st.hits_ram, st.hits_disk, r.pool.freePages(), r.pool.entry_only });
    var id = r.store.oldest;
    while (id != store_mod.none) : (id = r.store.entry(id).extra.newer) for (r.store.entry(id).extra.pages) |pg| try trace.append(gpa, pg);
}

/// After the scenario: a new process on the same tier finds every file and resumes chat 1 through its delta chain.
fn restart(gpa: std.mem.Allocator, root: []const u8, world: u32, rank: u32) !void {
    var r: Rank = undefined;
    try r.init(gpa, root, world, rank, .{ .chats = 1 });
    defer r.deinit();
    try testing.expectEqual(@as(u32, 4), try r.store.reconcile());
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 1, a[0 .. 16 * page]));
    try testing.expectEqual(@as(u64, 1), r.store.stats.hits_disk);
}

fn tmpRoot(gpa: std.mem.Allocator, tmp: *std.testing.TmpDir, sub: []const u8) ![:0]u8 {
    try tmp.dir.createDirPath(testing.io, sub);
    return tmp.dir.realPathFileAlloc(testing.io, sub, gpa);
}

test "chats over turns: residency to NVMe, bit-identical resumes, deltas, older turns dropped (sync and async)" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var t0: std.ArrayList(u64) = .empty;
    defer t0.deinit(gpa);
    var t1: std.ArrayList(u64) = .empty;
    defer t1.deinit(gpa);
    const r0 = try tmpRoot(gpa, &tmp, "sync");
    defer gpa.free(r0);
    try scenario(gpa, r0, 1, 0, false, &t0);
    try restart(gpa, r0, 1, 0);
    const r1 = try tmpRoot(gpa, &tmp, "async");
    defer gpa.free(r1);
    try scenario(gpa, r1, 1, 0, true, &t1);
    try testing.expectEqualSlices(u64, t0.items, t1.items); // the tier thread changes when bytes move, not what is decided
}

test "two split ranks make the same decisions on the same calls and each restores its own rows" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var t0: std.ArrayList(u64) = .empty;
    defer t0.deinit(gpa);
    var t1: std.ArrayList(u64) = .empty;
    defer t1.deinit(gpa);
    const root = try tmpRoot(gpa, &tmp, "tier");
    defer gpa.free(root);
    try scenario(gpa, root, 2, 0, false, &t0);
    try scenario(gpa, root, 2, 1, true, &t1);
    try restart(gpa, root, 2, 0);
    try restart(gpa, root, 2, 1);
    try testing.expectEqualSlices(u64, t0.items, t1.items);
}
