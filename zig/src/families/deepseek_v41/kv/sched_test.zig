//! The prefill scheduler against sched_golden.json: the Zig planner, store, pool and tier decide every round alike.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const sess = @import("sess.zig");
const sched = @import("sched.zig");

const Pool = sessions.Pool;
const Slot = sessions.Slot;
const Family = sessions.Family;
const Store = sessions.Store;
const Bounded = sessions.store.Bounded;
const Blob = sessions.disk.Blob;

const golden_json = @embedFile("sched_golden.json");

const SettingsIn = struct { rows: u64, short: u64, share: f64, long_prompt: u64, concurrent: u64, session_min: u64, adapt_gib: f64, adapt_up_gib: f64, adapt_min: u64, floor_gib: f64, hard_gib: f64, refuse_s: f64, cache_keep_gib: f64, index_budget_mib: u64 };
const JobIn = struct { id: u32, chat: u32, n: u64, max_tokens: u64, background: bool, submitted: f64 };
const AdmitIn = struct { id: u32, tier: []const u8, cached: u64, need: u32, damaged: bool };
const RoundIn = struct {
    round: u32,
    clock: f64,
    free_gib: f64,
    arrive: []const u32,
    cancel: []const u32,
    damage: bool,
    admits: []const AdmitIn,
    pieces: []const [3]u64,
    ran: []const [3]u64,
    skipped: []const u32,
    saves: []const u32,
    finals: []const u32,
    windows: []const u32,
    ended: []const u32,
    rows: ?u64,
    piece_s: f64,
    window_s: f64,
    counts: std.json.ArrayHashMap(u64),
    queue: []const u32,
    ram: []const u64,
    disk: []const u64,
    available: u32,
    debt: f64,
    fair_rounds: u64,
};
const Golden = struct {
    settings: SettingsIn,
    tag: u32,
    page: u32,
    pages: u32,
    slots: u32,
    capacity: u64,
    slack: u64,
    ram_bytes: u64,
    disk_min: u64,
    bounded_bytes: u64,
    window_s: f64,
    jobs: []const JobIn,
    rounds: []const RoundIn,
    counts: std.json.ArrayHashMap(u64),
};

const fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584 },
};

/// A snapshot's bounded state: a fixed RAM measure (golden.py's), one blob.
const Fixed = struct {
    gpa: std.mem.Allocator,
    pos: u64,
    bytes: u64,

    const vt: Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };

    fn make(gpa: std.mem.Allocator, pos: u64, bytes: u64) !Bounded {
        const h = try gpa.create(Fixed);
        h.* = .{ .gpa = gpa, .pos = pos, .bytes = bytes };
        return .{ .ptr = h, .vtable = &vt };
    }
    fn nbytes(p: *anyopaque) u64 {
        const h: *Fixed = @ptrCast(@alignCast(p));
        return h.bytes;
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
        const h: *Fixed = @ptrCast(@alignCast(p));
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, h.pos, .little);
        const out = try gpa.alloc(Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "slot"), .bytes = try gpa.dupe(u8, &b) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const h: *Fixed = @ptrCast(@alignCast(p));
        h.gpa.destroy(h);
    }
};

fn chatIds(out: []i32, chat: u32) void {
    for (out, 0..) |*t, i| t.* = @intCast(chat * 100_000 + i);
}

const World = struct {
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    g: Golden,
    pool: Pool,
    host: sessions.HostPool,
    disk: *sessions.Disk,
    store: Store,
    root: []const u8,
    slots: [4]*Slot = undefined,
    /// each slot's request (job id) and a release the next round runs
    owner: [4]?u32 = @splat(null),
    releasing: [4]bool = @splat(false),
    /// each request's decode position and reply tokens
    pos: std.AutoHashMapUnmanaged(u32, u64) = .empty,
    out: std.AutoHashMapUnmanaged(u32, u64) = .empty,

    fn job(w: *const World, id: u32) JobIn {
        for (w.g.jobs) |j| if (j.id == id) return j;
        unreachable;
    }

    fn ids(w: *World, id: u32) ![]i32 {
        const j = w.job(id);
        const x = try w.a.alloc(i32, @intCast(j.n));
        chatIds(x, j.chat);
        return x;
    }

    fn slotOf(w: *const World, id: u32) ?usize {
        for (w.owner, 0..) |o, i| if (o == id) return i;
        return null;
    }

    fn findFn(ctx: *anyopaque, key: usize) anyerror!?sched.Store.Hit {
        const w: *World = @ptrCast(@alignCast(ctx));
        const id: u32 = @intCast(key);
        const h = (try w.store.find(w.g.tag, try w.ids(id))) orelse return null;
        if (h.pos >= w.job(id).n) return null;
        return .{ .id = h.id, .pos = h.pos, .ram = h.ram };
    }

    fn poolFn(ctx: *anyopaque, key: usize, hit: ?sched.Store.Hit, keep: []const u32, extra: u32, spills: *std.ArrayList(u32)) anyerror!sess.Plan {
        const w: *World = @ptrCast(@alignCast(ctx));
        const j = w.job(@intCast(key));
        const h: ?sessions.store.Hit = if (hit) |x| .{ .id = x.id, .pos = x.pos, .ram = x.ram } else null;
        return sess.admit(&w.store, h, j.n, j.max_tokens, w.g.capacity, keep, extra, spills);
    }

    fn storeOf(w: *World) sched.Store {
        return .{ .ctx = w, .vtable = &.{ .find = findFn, .pool = poolFn } };
    }

    /// Every NVMe file damaged (a byte run in the middle of the file: its pages' data).
    fn damage(w: *World) !void {
        // the tier's files: <root>/<compat's first 8 bytes in hex>/rank<r> (disk.zig `open`)
        const compat: [8]u8 = @splat(7);
        const sub = try std.fmt.allocPrint(w.a, "{s}/{x}/rank0", .{ w.root, &compat });
        var dir = try std.Io.Dir.cwd().openDir(testing.io, sub, .{ .iterate = true });
        defer dir.close(testing.io);
        var it = dir.iterate();
        while (try it.next(testing.io)) |e| {
            if (!std.mem.endsWith(u8, e.name, ".tfs")) continue;
            var f = try dir.openFile(testing.io, e.name, .{ .mode = .read_write });
            defer f.close(testing.io);
            const size = (try f.stat(testing.io)).size;
            var junk: [32]u8 = @splat(0xff);
            try f.writePositionalAll(testing.io, &junk, size / 2);
        }
    }

    fn lengths(w: *World, disk: bool) ![]u64 {
        var l: std.ArrayList(u64) = .empty;
        for (w.store.index.entries.items) |e| if (e.live) {
            if (if (disk) e.extra.file else e.ram) try l.append(w.a, e.pos);
        };
        std.mem.sort(u64, l.items, {}, std.sort.asc(u64));
        return l.items;
    }
};

fn expectIds(label: []const u8, round: u32, want: []const u32, got: []const u32) !void {
    errdefer std.debug.print("round {d} {s}: reference {any}, ours {any}\n", .{ round, label, want, got });
    try testing.expectEqualSlices(u32, want, got);
}

fn sortedIds(a: std.mem.Allocator, l: []const u32) ![]u32 {
    const x = try a.dupe(u32, l);
    std.mem.sort(u32, x, {}, std.sort.asc(u32));
    return x;
}

test "the round planner decides as the reference batcher: admissions, waits, pieces, snapshots, finals, damaged entries, rows" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(Golden, gpa, golden_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const g = parsed.value;
    try testing.expectEqual(@as(u64, sess.slack), g.slack);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);

    var w: World = .{ .gpa = gpa, .a = arena.allocator(), .g = g, .pool = undefined, .host = undefined, .disk = undefined, .store = undefined, .root = root };
    defer w.pos.deinit(gpa);
    defer w.out.deinit(gpa);
    w.pool = try Pool.init(gpa, .{ .families = &fams, .page = g.page }, @as(u64, g.pages) * g.page, 1, 0);
    defer w.pool.deinit();
    w.host = try sessions.HostPool.init(gpa, &w.pool);
    defer w.host.deinit();
    w.disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(7), .min_tokens = g.disk_min, .chunk = 1 << 20, .staging = 4, .lanes = 2, .direct = false });
    defer w.disk.close();
    w.store = try Store.init(gpa, testing.io, &w.pool, w.host.store(), w.disk, .{ .ram_bytes = g.ram_bytes, .skip_covered = false, .delta = false });
    defer w.store.deinit();
    for (&w.slots) |*s| s.* = try w.pool.newSlot(g.capacity);

    const S = g.settings;
    var p = sched.Planner.init(gpa, .{
        .rows = S.rows,
        .short = S.short,
        .share = S.share,
        .long_prompt = S.long_prompt,
        .concurrent = S.concurrent,
        .session_min = S.session_min,
        .sessions = true,
        .adapt = .{ .gib = S.adapt_gib, .up_gib = S.adapt_up_gib, .min_rows = S.adapt_min },
        .floor = .{ .target_gib = S.floor_gib, .hard_gib = S.hard_gib, .refuse_s = S.refuse_s, .keep_gib = S.cache_keep_gib },
        .price = .{ .budget_mib = S.index_budget_mib },
    });
    defer p.deinit();
    var queue: std.ArrayList(u32) = .empty;
    defer queue.deinit(gpa);
    var pieces: std.ArrayList(sched.Piece) = .empty;
    defer pieces.deinit(gpa);
    var finals: std.ArrayList(usize) = .empty;
    defer finals.deinit(gpa);
    var seen = struct { damaged: u32 = 0, pool_waits: u64 = 0, mem_waits: u64 = 0, long_waits: u64 = 0, rows_down: u64 = 0, snapshots: u32 = 0 }{};

    for (g.rounds) |rd| {
        const a = w.a;
        const r = rd.round;
        errdefer std.debug.print("round {d} failed\n", .{r});
        const mi: sched.Meminfo = .{ .free = @intFromFloat(rd.free_gib * sched.GiB), .available = @intFromFloat((rd.free_gib + 3) * sched.GiB), .mapped = 1 << 30 };
        // arrivals: foreground before background, each in arrival order (lane_host.zig's queue)
        for (rd.arrive) |id| {
            var at = queue.items.len;
            if (!w.job(id).background) while (at > 0 and w.job(queue.items[at - 1]).background) {
                at -= 1;
            };
            try queue.insert(gpa, at, id);
        }
        if (rd.damage) try w.damage();
        // cancellations: a queued request leaves the queue; a running one ends (its slot's release runs this round)
        for (rd.cancel) |id| {
            if (std.mem.indexOfScalar(u32, queue.items, id)) |i| {
                _ = queue.orderedRemove(i);
                p.dropped(id);
            } else if (w.slotOf(id)) |s| {
                p.ended(id);
                w.owner[s] = null;
                w.releasing[s] = true;
            }
        }
        // the admission pass, decided whole before any of it runs
        const Adm = struct { id: u32, slot: usize, adm: sched.Admission };
        var adms: std.ArrayList(Adm) = .empty;
        var round = p.beginAdmit();
        defer round.deinit(gpa);
        var free: usize = 0;
        for (w.owner) |o| free += @intFromBool(o == null);
        var i: usize = 0;
        while (i < queue.items.len and free > 0) {
            const id = queue.items[i];
            const j = w.job(id);
            const v = try p.admit(&round, .{ .key = id, .n = j.n, .max_new = j.max_tokens, .submitted = j.submitted }, w.storeOf(), rd.clock, mi);
            switch (v) {
                .admit => |adm| {
                    const s = for (w.owner, 0..) |o, k| {
                        if (o == null) break k;
                    } else unreachable;
                    w.owner[s] = id;
                    free -= 1;
                    try adms.append(a, .{ .id = id, .slot = s, .adm = adm });
                    _ = queue.orderedRemove(i);
                },
                .wait => break,
                .skip => i += 1,
                .refuse => {
                    _ = queue.orderedRemove(i);
                    p.dropped(id);
                },
            }
        }
        // the round runs: releases, the round's spills, then each admission (reserve, restore; a damaged entry: none)
        for (&w.releasing, 0..) |*x, s| if (x.*) {
            _ = try w.slots[s].releaseAll();
            x.* = false;
        };
        for (round.spills.items) |id| _ = try w.store.evict(id);
        try w.store.settleAll();
        try testing.expectEqual(rd.admits.len, adms.items.len);
        for (adms.items, rd.admits) |x, want| {
            const s = w.slots[x.slot];
            try testing.expectEqual(@as(u32, 0), s.len);
            s.reserve(x.adm.need);
            var tier: []const u8 = "none";
            var bad = false;
            if (x.adm.hit) |h| {
                tier = if (h.ram) "ram" else "disk";
                if (h.ram) {
                    _ = try w.store.restoreRam(h.id, s);
                } else {
                    const ok = blk: {
                        const job_ = w.store.beginRestore(h.id, s) catch break :blk false;
                        var got = (w.store.settle(job_) catch break :blk false) orelse break :blk false;
                        got.deinit(gpa);
                        break :blk true;
                    };
                    if (!ok) {
                        _ = try s.truncate(0);
                        try w.store.forget(h.id);
                        p.damaged(x.id);
                        bad = true;
                        seen.damaged += 1;
                    }
                }
            }
            errdefer std.debug.print("round {d} admit {d}: tier {s} cached {d} need {d} damaged {}; reference {d} {s} {d} {d} {}\n", .{ r, x.id, tier, if (x.adm.hit) |h| h.pos else 0, x.adm.need, bad, want.id, want.tier, want.cached, want.need, want.damaged });
            try testing.expectEqual(want.id, x.id);
            try testing.expectEqualStrings(want.tier, tier);
            try testing.expectEqual(want.cached, if (x.adm.hit) |h| h.pos else 0);
            try testing.expectEqual(want.need, x.adm.need);
            try testing.expectEqual(want.damaged, bad);
            try testing.expectEqual(x.adm.need, s.quota.?);
        }
        // the pieces, run unless their entry was damaged, each snapshot right after its piece
        try p.plan(mi, &pieces, &finals);
        var planned: std.ArrayList([3]u64) = .empty;
        var ran: std.ArrayList([3]u64) = .empty;
        var saves: std.ArrayList(u32) = .empty;
        var rows: u64 = 0;
        for (pieces.items) |pc| {
            try planned.append(a, .{ pc.key, pc.start, pc.end });
            if (!pc.run) continue;
            try ran.append(a, .{ pc.key, pc.start, pc.end });
            rows += pc.end - pc.start;
            const id: u32 = @intCast(pc.key);
            const s = w.slots[w.slotOf(id).?];
            try s.ensure(pc.end);
            if (pc.save and pc.end >= S.session_min) {
                const x = try w.ids(id);
                _ = try w.store.save(s, try Fixed.make(gpa, pc.end, g.bounded_bytes), x[0..@intCast(pc.end)], g.tag, 1);
                try w.store.settleAll();
                try saves.append(a, id);
                seen.snapshots += 1;
            }
        }
        {
            errdefer std.debug.print("round {d} pieces: reference {any} (ran {any}), ours {any} (ran {any})\n", .{ r, rd.pieces, rd.ran, planned.items, ran.items });
            try testing.expectEqual(rd.pieces.len, planned.items.len);
            for (rd.pieces, planned.items) |x, y| try testing.expectEqualSlices(u64, &x, &y);
            try testing.expectEqual(rd.ran.len, ran.items.len);
            for (rd.ran, ran.items) |x, y| try testing.expectEqualSlices(u64, &x, &y);
        }
        try expectIds("saves", r, rd.saves, saves.items);
        var fin: std.ArrayList(u32) = .empty;
        for (finals.items) |k| {
            try fin.append(a, @intCast(k));
            try w.pos.put(gpa, @intCast(k), w.job(@intCast(k)).n - 1);
            try w.out.put(gpa, @intCast(k), 0);
        }
        try expectIds("finals", r, try sortedIds(a, rd.finals), try sortedIds(a, fin.items));
        // the windows: every decoding request a token (no drafts), its pages for the row
        var wins: std.ArrayList(u32) = .empty;
        for (p.seqs.items) |x| if (x.decoding and !x.damaged) try wins.append(a, @intCast(x.key));
        const wsorted = try sortedIds(a, wins.items);
        try expectIds("windows", r, rd.windows, wsorted);
        var ended: std.ArrayList(u32) = .empty;
        for (wins.items) |id| {
            const s = w.slots[w.slotOf(id).?];
            const at = w.pos.getPtr(id).?;
            try s.ensure(at.* + 1);
            at.* += 1;
            const o = w.out.getPtr(id).?;
            o.* += 1;
            if (o.* >= w.job(id).max_tokens) try ended.append(a, id);
        }
        try expectIds("ended", r, try sortedIds(a, rd.ended), try sortedIds(a, ended.items));
        for (ended.items) |id| {
            const s = w.slotOf(id).?;
            p.ended(id);
            w.owner[s] = null;
            w.releasing[s] = true; // released at the next round's start
        }
        const piece_s = @as(f64, @floatFromInt(rows)) / 8192.0;
        try testing.expectEqual(rd.piece_s, piece_s);
        p.after(piece_s, if (wins.items.len > 0) g.window_s else 0.0);
        {
            const ram_l = try w.lengths(false);
            const disk_l = try w.lengths(true);
            errdefer std.debug.print("round {d}: rows {?d} (reference {?d}), debt {d} (reference {d}), fair rounds {d} (reference {d}), queue {any} (reference {any}), available {d} (reference {d}), RAM {any} (reference {any}), NVMe {any} (reference {any})\n", .{ r, p.rows_now, rd.rows, p.fair.debt, rd.debt, p.fair.rounds, rd.fair_rounds, queue.items, rd.queue, w.pool.available(), rd.available, ram_l, rd.ram, disk_l, rd.disk });
            try testing.expectEqual(rd.rows, p.rows_now);
            try testing.expectEqual(rd.debt, p.fair.debt);
            try testing.expectEqual(rd.fair_rounds, p.fair.rounds);
            try testing.expectEqualSlices(u32, rd.queue, queue.items);
            try testing.expectEqual(rd.available, w.pool.available());
            try testing.expectEqualSlices(u64, rd.ram, ram_l);
            try testing.expectEqualSlices(u64, rd.disk, disk_l);
        }
    }
    // the golden's counters over the whole run
    const c = p.counts;
    const want = g.counts.map;
    try testing.expectEqual(want.get("long_waits") orelse 0, c.long_waits);
    try testing.expectEqual(want.get("mem_waits") orelse 0, c.mem_waits);
    try testing.expectEqual(want.get("pool_waits") orelse 0, c.pool_waits);
    try testing.expectEqual(want.get("mem_refused") orelse 0, c.mem_refused);
    try testing.expectEqual(want.get("rows_down") orelse 0, c.rows_down);
    try testing.expectEqual(want.get("rows_up") orelse 0, c.rows_up);
    for (c.rows) |x| if (x[0] != 0) {
        var nb: [32]u8 = undefined;
        try testing.expectEqual(want.get(try std.fmt.bufPrint(&nb, "rows_{d}", .{x[0]})) orelse 0, x[1]);
    };
    // the golden exercises what it claims
    seen.pool_waits = c.pool_waits;
    seen.mem_waits = c.mem_waits;
    seen.long_waits = c.long_waits;
    seen.rows_down = c.rows_down;
    try testing.expect(seen.damaged == 1 and seen.pool_waits > 0 and seen.mem_waits > 0 and seen.long_waits > 0 and seen.rows_down > 0 and seen.snapshots >= 8);
}
