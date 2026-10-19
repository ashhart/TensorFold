//! Sessions over 4 live slots against session_slots_golden.json: each step decides alike, each resume checked by row.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const sess = @import("sess.zig");

const Pool = sessions.Pool;
const Slot = sessions.Slot;
const Family = sessions.Family;
const Store = sessions.Store;
const Bounded = sessions.store.Bounded;
const Blob = sessions.disk.Blob;

const golden_json = @embedFile("session_slots_golden.json");
/// The split store and pool (2 ranks: their decisions alike) on 30 pages (golden.py --split 2 --pages 30)
const split_golden_json = @embedFile("session_slots_split_golden.json");

const Saved = struct { at: u64, nbytes: u64, ds_rows: u64, dup: bool };
const Step = struct {
    op: []const u8,
    label: []const u8 = "",
    slot: u32 = 0,
    chat: u32 = 0,
    branch: u32 = 0,
    n: u64 = 0,
    max_new: u64 = 0,
    reply: u64 = 0,
    tier: []const u8 = "",
    cached: u64 = 0,
    need: u32 = 0,
    want: u32 = 0,
    spills: []const []const u8 = &.{},
    saved: ?Saved = null,
    wait: ?bool = null,
    ram: []const []const u8,
    disk: []const []const u8,
    available: u32,
    free: u32,
};
const MeasureIn = struct { layers: u64, decoder_start: u64, window: u64, carries: u64, head_dim: u64, hidden: u64, hc: u64, ds_blocks: u64, keep: u64, row_bytes: u64, lookback: u64 };
const Golden = struct {
    tag: u32,
    page: u32,
    pages: u32,
    slots: u32,
    capacity: u64,
    slack: u64,
    ram_bytes: u64,
    disk_min: u64,
    measure: MeasureIn,
    steps: []const Step,
};

/// The pool's families as the test stores them (any sizes: the decisions are in pages; the rows are checked)
const fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584 },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584 },
};
/// TF_DSV41_KV_SPLIT=1: the comp rows of a page live on the rank that owns it, the indexer's on every rank
const split_fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584, .split = true },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584, .split = true },
};

/// A snapshot's bounded state in host memory: the drafter rows and stash start at `pos`, with the golden's RAM measure.
const HostBounded = struct {
    gpa: std.mem.Allocator,
    pos: u64,
    ds_rows: u64,
    stash_from: u64,
    measure: u64,

    const vt: Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };

    fn make(gpa: std.mem.Allocator, b: HostBounded) !Bounded {
        const h = try gpa.create(HostBounded);
        h.* = b;
        h.gpa = gpa;
        return .{ .ptr = h, .vtable = &vt };
    }
    fn nbytes(p: *anyopaque) u64 {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        return h.measure;
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        var b: [24]u8 = undefined;
        std.mem.writeInt(u64, b[0..8], h.pos, .little);
        std.mem.writeInt(u64, b[8..16], h.ds_rows, .little);
        std.mem.writeInt(u64, b[16..24], h.stash_from, .little);
        const out = try gpa.alloc(Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "slot"), .bytes = try gpa.dupe(u8, &b) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        h.gpa.destroy(h);
    }
};

/// A chat's ids (golden.py chat_ids): its own stream; a branch rewrites its last 100 tokens.
fn chatIds(out: []i32, chat: u32, branch: u32) void {
    const n = out.len;
    for (out, 0..) |*t, i| t.* = @intCast(chat * 100_000 + i);
    if (branch > 0) {
        const cut = n - 100;
        for (out[cut..], 0..) |*t, i| t.* = @intCast(chat * 100_000 + 50_000 + branch * 1000 + i);
    }
}

/// A row's bytes: a function of the id at its first position (what a fresh prefill of the same prefix writes there).
fn rowByte(id: i32, pos: u64, fi: usize, i: usize) u8 {
    return @truncate(@as(u64, @bitCast(@as(i64, id))) *% 31 +% pos *% 3 +% i *% 7 +% fi);
}

const World = struct {
    gpa: std.mem.Allocator,
    g: Golden,
    fams: []const Family,
    pool: Pool,
    host: sessions.HostPool,
    disk: *sessions.Disk,
    store: Store,
    slots: [4]*Slot = undefined,
    /// per slot: drafter.valid and the stash's first position (sessions_gpu.zig keeps the same)
    valid: [4]u64 = @splat(0),
    stash_from: [4]u64 = @splat(0),
    names: std.AutoHashMapUnmanaged(u32, []const u8) = .empty,
    arena: std.heap.ArenaAllocator,

    fn measure(w: *const World) sess.Measure {
        const m = w.g.measure;
        return .{ .rings = m.decoder_start, .window = m.window, .carries = m.carries, .head_dim = m.head_dim, .hidden = m.hidden, .hc = m.hc, .ds_blocks = m.ds_blocks, .replay = true };
    }

    /// Writes the rows of positions [from, to) of the slot's pages (each family's rows that start there).
    fn write(w: *World, s: *const Slot, ids: []const i32, from: u64, to: u64) void {
        const page = w.pool.page;
        for (w.fams, 0..) |f, fi| {
            var p = from - from % f.ratio + (if (from % f.ratio != 0) f.ratio else 0);
            while (p < to) : (p += f.ratio) {
                const pg = s.mapped()[@intCast(p / page)];
                if (f.split and !w.pool.owns(pg)) continue;
                const v = w.host.pageOf(@intCast(fi), w.pool.familyPage(f, pg));
                const at: usize = @intCast((p % page) / f.ratio * f.row_bytes);
                for (v[at..][0..f.row_bytes], 0..) |*b, i| b.* = rowByte(ids[@intCast(p)], p, fi, i);
            }
        }
    }

    /// Every row below `pos` of the slot's pages holds what a fresh prefill of `ids` writes there.
    fn check(w: *World, s: *const Slot, ids: []const i32, pos: u64) !void {
        const page = w.pool.page;
        for (w.fams, 0..) |f, fi| {
            var p: u64 = 0;
            while (p + f.ratio <= pos) : (p += f.ratio) {
                const pg = s.mapped()[@intCast(p / page)];
                if (f.split and !w.pool.owns(pg)) continue;
                const v = w.host.pageOf(@intCast(fi), w.pool.familyPage(f, pg));
                const at: usize = @intCast((p % page) / f.ratio * f.row_bytes);
                for (v[at..][0..f.row_bytes], 0..) |b, i| if (b != rowByte(ids[@intCast(p)], p, fi, i)) {
                    std.debug.print("row at position {d} (family {s}) is not the prompt's\n", .{ p, f.name });
                    return error.WrongRow;
                };
            }
        }
    }

    fn nameOf(w: *World, id: u32) []const u8 {
        return w.names.get(id) orelse "?";
    }

    fn sortNames(list: [][]const u8) void {
        std.mem.sort([]const u8, list, {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.less);
    }

    /// The RAM tier's entries, the NVMe tier's files, the pool's pages == the golden's after this step.
    fn expectState(w: *World, st: *const Step) !void {
        const a = w.arena.allocator();
        var ram: std.ArrayList([]const u8) = .empty;
        var disk: std.ArrayList([]const u8) = .empty;
        for (w.store.index.entries.items, 0..) |e, id| if (e.live) {
            if (e.ram) try ram.append(a, w.nameOf(@intCast(id)));
            if (e.extra.file) try disk.append(a, w.nameOf(@intCast(id)));
        };
        sortNames(ram.items);
        sortNames(disk.items);
        const j = struct {
            fn of(x: std.mem.Allocator, l: []const []const u8) []const u8 {
                return std.mem.join(x, " ", l) catch "?";
            }
        }.of;
        errdefer std.debug.print("step '{s}' {s} slot {d}: RAM {s} (reference {s}), NVMe {s} (reference {s}), available {d} (reference {d}), free {d} (reference {d})\n", .{ st.op, st.label, st.slot, j(a, ram.items), j(a, st.ram), j(a, disk.items), j(a, st.disk), w.pool.available(), st.available, w.pool.freePages(), st.free });
        try testing.expectEqual(st.ram.len, ram.items.len);
        for (st.ram, ram.items) |x, y| try testing.expectEqualStrings(x, y);
        try testing.expectEqual(st.disk.len, disk.items.len);
        for (st.disk, disk.items) |x, y| try testing.expectEqualStrings(x, y);
        try testing.expectEqual(st.available, w.pool.available());
        try testing.expectEqual(st.free, w.pool.freePages());
    }

    fn admit(w: *World, st: *const Step) !void {
        const a = w.arena.allocator();
        const ids = try a.alloc(i32, @intCast(st.n));
        chatIds(ids, st.chat, st.branch);
        const hit = try w.store.find(w.g.tag, ids);
        var spills: std.ArrayList(u32) = .empty;
        defer spills.deinit(w.gpa);
        const keep: []const u32 = if (hit) |h| &.{h.id} else &.{};
        const plan = try sess.admit(&w.store, hit, st.n, st.max_new, w.g.capacity, keep, 0, &spills);
        if (std.mem.eql(u8, st.op, "probe")) {
            try testing.expectEqual(st.wait.?, plan.wait);
            try testing.expectEqual(st.need, plan.need);
            try testing.expectEqual(st.want, plan.want);
            return;
        }
        errdefer std.debug.print("admit '{s}': tier {s} cached {d} need {d} want {d} spills {d}\n", .{ st.label, st.tier, st.cached, plan.need, plan.want, spills.items.len });
        try testing.expect(!plan.wait);
        try testing.expectEqual(st.need, plan.need);
        try testing.expectEqual(st.want, plan.want);
        try testing.expectEqual(st.spills.len, spills.items.len);
        var names: std.ArrayList([]const u8) = .empty;
        for (spills.items) |id| try names.append(a, w.nameOf(id));
        sortNames(names.items);
        for (st.spills, names.items) |x, y| try testing.expectEqualStrings(x, y);
        // the leader's spills in its order, then the slot's reservation (reserve, reset, restore)
        for (spills.items) |id| _ = try w.store.evict(id);
        try w.store.settleAll();
        const s = w.slots[st.slot];
        try testing.expectEqual(@as(u32, 0), s.len);
        s.reserve(plan.need);
        w.valid[st.slot] = 0;
        w.stash_from[st.slot] = 0;
        var cached: u64 = 0;
        var tier: []const u8 = "none";
        if (hit) |h| {
            if (h.ram) {
                const b = try w.store.restoreRam(h.id, s);
                const hb: *HostBounded = @ptrCast(@alignCast(b.ptr));
                try testing.expectEqual(h.pos, hb.pos);
                w.valid[st.slot] = hb.pos - hb.ds_rows;
                w.stash_from[st.slot] = hb.stash_from;
                tier = "ram";
            } else {
                const j = try w.store.beginRestore(h.id, s);
                var got = (try w.store.settle(j)).?;
                defer got.deinit(w.gpa);
                const b = got.blobs[0].bytes;
                try testing.expectEqual(h.pos, std.mem.readInt(u64, b[0..8], .little));
                w.valid[st.slot] = h.pos - std.mem.readInt(u64, b[8..16], .little);
                w.stash_from[st.slot] = std.mem.readInt(u64, b[16..24], .little);
                tier = "disk";
            }
            cached = h.pos;
            try w.check(s, ids, cached);
        }
        try testing.expectEqualStrings(st.tier, tier);
        try testing.expectEqual(st.cached, cached);
        // the prompt to its replay point, the snapshot there, the rest of it, the reply
        var at = cached;
        if (sess.savePoint(st.n, cached)) |save_at| {
            try s.ensure(save_at);
            w.write(s, ids, at, save_at);
            at = save_at;
            const ds = sess.dsRows(save_at, w.valid[st.slot], w.g.measure.window);
            const stash = sess.stashRows(save_at, w.stash_from[st.slot]);
            const m = w.measure().bytes(save_at, stash, ds);
            const want = st.saved orelse return error.ReferenceSavedNothing;
            try testing.expectEqual(want.at, save_at);
            try testing.expectEqual(want.ds_rows, ds);
            try testing.expectEqual(want.nbytes, m);
            const b = try HostBounded.make(w.gpa, .{ .gpa = w.gpa, .pos = save_at, .ds_rows = ds, .stash_from = save_at - stash, .measure = m });
            const id = try w.store.save(s, b, ids[0..@intCast(save_at)], w.g.tag, 1);
            try testing.expectEqual(want.dup, id == null);
            if (id) |x| try w.names.put(w.gpa, x, try std.fmt.allocPrint(a, "c{d}b{d}@{d}", .{ st.chat, st.branch, save_at }));
            try w.store.settleAll();
        } else try testing.expect(st.saved == null);
        try s.ensure(st.n - 1);
        w.write(s, ids, at, st.n - 1);
        try s.ensure(st.n + st.reply);
    }
};

/// The golden's steps on one rank of `world`; every step must decide as the golden did (replicated or split alike).
fn replay(gpa: std.mem.Allocator, g: Golden, families: []const Family, world: u32, rank: u32) !void {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);

    var w: World = .{ .gpa = gpa, .g = g, .fams = families, .pool = undefined, .host = undefined, .disk = undefined, .store = undefined, .arena = .init(gpa) };
    defer w.arena.deinit();
    defer w.names.deinit(gpa);
    w.pool = try Pool.init(gpa, .{ .families = families, .page = g.page }, @as(u64, g.pages) * g.page, world, rank);
    defer w.pool.deinit();
    w.host = try sessions.HostPool.init(gpa, &w.pool);
    defer w.host.deinit();
    w.disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .rank = rank, .world = world, .min_tokens = g.disk_min, .chunk = 1 << 20, .staging = 4, .lanes = 2, .direct = false });
    defer w.disk.close();
    // the golden's store: its RAM budget, no residency budget, no covered-entry or delta rules
    w.store = try Store.init(gpa, testing.io, &w.pool, w.host.store(), w.disk, .{ .ram_bytes = g.ram_bytes, .skip_covered = false, .delta = false });
    defer w.store.deinit();
    for (&w.slots) |*s| s.* = try w.pool.newSlot(g.capacity);
    try testing.expectEqual(@as(usize, g.slots), w.slots.len);
    var spills: u32 = 0;
    var hits: [2]u32 = .{ 0, 0 };
    for (g.steps) |*st| {
        if (std.mem.eql(u8, st.op, "admit") or std.mem.eql(u8, st.op, "probe")) {
            try w.admit(st);
            spills += @intCast(st.spills.len);
            if (std.mem.eql(u8, st.tier, "ram")) hits[0] += 1;
            if (std.mem.eql(u8, st.tier, "disk")) hits[1] += 1;
        } else if (std.mem.eql(u8, st.op, "finish")) {
            _ = try w.slots[st.slot].releaseAll();
        } else return error.UnknownStep;
        try w.expectState(st);
    }
    // the golden exercises what it claims: spills, both tiers' resumes, parks and drops
    try testing.expect(spills >= 2 and hits[0] >= 2 and hits[1] >= 3);
    try testing.expect(w.store.stats.parks >= 3 and w.store.stats.drops >= 2);
}

fn golden(gpa: std.mem.Allocator, text: []const u8) !std.json.Parsed(Golden) {
    const parsed = try std.json.parseFromSlice(Golden, gpa, text, .{ .ignore_unknown_fields = true });
    errdefer parsed.deinit();
    const g = parsed.value;
    try testing.expectEqual(@as(u64, sess.slack), g.slack);
    try testing.expectEqual(sess.row_bytes, g.measure.row_bytes);
    try testing.expectEqual(sess.keep, g.measure.keep);
    try testing.expectEqual(sess.lookback, g.measure.lookback);
    return parsed;
}

test "4 slots, interleaved resumes, spills, RAM trims and NVMe parks decide as the reference's store, pool and tier" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const parsed = try golden(testing.allocator, golden_json);
    defer parsed.deinit();
    try replay(testing.allocator, parsed.value, &fams, 1, 0);
}

test "split KV over 2 ranks with 4 slots: each rank decides as kvsplit's store and pool, restoring only its own rows" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const parsed = try golden(testing.allocator, split_golden_json);
    defer parsed.deinit();
    try replay(testing.allocator, parsed.value, &split_fams, 2, 0);
    try replay(testing.allocator, parsed.value, &split_fams, 2, 1);
}
