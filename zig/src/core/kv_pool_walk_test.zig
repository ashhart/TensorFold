//! Seeded walks over every pool transition: after each one the kept counts equal main's scan of every page.
const std = @import("std");
const kv = @import("kv_pool.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

const most = 40; // pages a walk's pool holds at most

/// Census and reclaimable as main computed them before the counts: a scan of every page's holders and leases.
fn scanned(p: *const kv.Pool) struct { census: kv.Census, reclaimable: usize } {
    var kept: usize = 0;
    var leased: usize = 0;
    var reclaimable: usize = 0;
    for (p.refs, p.leased) |r, l| {
        kept += @intFromBool(r > l);
        leased += @intFromBool(l > 0);
        reclaimable += @intFromBool(r == 1 and l == 0);
    }
    return .{ .census = .{ .total = p.total(), .free = p.nfree, .promised = p.promised, .kept = kept, .leased = leased }, .reclaimable = reclaimable };
}

/// The counts, every fits answer and idle are the scan's.
fn agrees(p: *const kv.Pool) !void {
    const s = scanned(p);
    try equal(s.census, p.census());
    try equal(s.reclaimable, p.reclaimable());
    for (0..p.total() + 2) |need| try equal(p.nfree - p.promised + s.reclaimable >= need, p.fits(need));
    const c = s.census;
    try equal(c.promised == 0 and c.leased == 0 and c.free + c.kept == c.total, p.idle());
}

/// An allocator that refuses to grow while `fail` is set, so a lease's table growth fails where the walk says.
const Flaky = struct {
    fail: bool = false,

    fn allocator(f: *Flaky) std.mem.Allocator {
        return .{ .ptr = f, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const f: *Flaky = @ptrCast(@alignCast(ctx));
        return if (f.fail) null else std.testing.allocator.rawAlloc(len, a, ra);
    }
    fn resize(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const f: *Flaky = @ptrCast(@alignCast(ctx));
        return !(f.fail and n > m.len) and std.testing.allocator.rawResize(m, a, n, ra);
    }
    fn remap(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const f: *Flaky = @ptrCast(@alignCast(ctx));
        return if (f.fail and n > m.len) null else std.testing.allocator.rawRemap(m, a, n, ra);
    }
    fn free(_: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        std.testing.allocator.rawFree(m, a, ra);
    }
};

/// What the walks met, so a quiet generator cannot pass for coverage.
const Seen = struct {
    refused: usize = 0,
    oom: usize = 0,
    uploads: usize = 0,
    past: usize = 0,
    not_held: usize = 0,
    aligned: usize = 0,
    runs: usize = 0,
    reclaims: usize = 0,
    shares: usize = 0,
    adopts: usize = 0,
    rollbacks: usize = 0,
};

const Run = struct { start: u32, k: usize };

/// A pool, its leases and runs, and the store's holds beside them, as the walk's model has them.
const Walk = struct {
    pool: kv.Pool,
    flaky: *Flaky,
    rnd: std.Random,
    leases: [5]kv.Lease = @splat(.{}),
    runs: [6]Run = undefined,
    nruns: usize = 0,
    store: [most]u32 = @splat(0), // holds past the leases', as the paged store's
    in_run: [most]bool = @splat(false),
    upload_fails: bool = false,
    seen: Seen = .{},

    fn write(ctx: *anyopaque, _: []const u32, _: usize) anyerror!void {
        const w: *Walk = @ptrCast(@alignCast(ctx));
        if (w.upload_fails) return error.UploadFailed;
    }

    /// No table, or one whose write fails while `upload_fails` is set.
    fn table(w: *Walk) ?kv.Table {
        return if (w.rnd.boolean()) .{ .ctx = w, .write = write } else null;
    }

    /// The store's eviction as a promise asks for it: holds go, leased pages among them, until `want` are available.
    fn reclaim(ctx: *anyopaque, want: usize) bool {
        const w: *Walk = @ptrCast(@alignCast(ctx));
        w.seen.reclaims += 1;
        for (&w.store, 0..) |*n, id| while (n.* > 0 and w.pool.available() < want) : (n.* -= 1) w.pool.release(@intCast(id));
        return w.pool.available() >= want;
    }

    fn failed(w: *Walk, err: anyerror) !void {
        switch (err) {
            error.OutOfMemory => w.seen.oom += 1,
            error.UploadFailed => w.seen.uploads += 1,
            error.PastReservation => w.seen.past += 1,
            else => return err,
        }
    }

    fn claim(w: *Walk, l: *kv.Lease) void {
        if (w.nruns == w.runs.len) return;
        const k = 1 + w.rnd.uintLessThan(usize, 3);
        const start = w.pool.claimRun(l, k) orelse return;
        for (start..start + k) |id| w.in_run[id] = true;
        w.runs[w.nruns] = .{ .start = start, .k = k };
        w.nruns += 1;
        w.seen.runs += 1;
    }

    fn unclaim(w: *Walk, i: usize) void {
        const r = w.runs[i];
        w.pool.releaseRun(r.start, r.k);
        for (r.start..r.start + r.k) |id| w.in_run[id] = false;
        w.nruns -= 1;
        w.runs[i] = w.runs[w.nruns];
    }

    /// A page to act on: half the time one some lease maps (held, never a run's), else any page.
    fn page(w: *Walk) u32 {
        const l = &w.leases[w.rnd.uintLessThan(usize, w.leases.len)];
        if (l.ids.items.len > 0 and w.rnd.boolean()) return l.ids.items[w.rnd.uintLessThan(usize, l.ids.items.len)];
        return w.rnd.uintLessThan(u32, @intCast(w.pool.total()));
    }

    /// One transition at random, with allocations and table writes failing now and then.
    fn step(w: *Walk) !void {
        const p = &w.pool;
        const l = &w.leases[w.rnd.uintLessThan(usize, w.leases.len)];
        const id = w.page();
        const shared = p.refs[id] > 0 and !w.in_run[id]; // a run's pages are never the store's or another lease's
        w.upload_fails = w.rnd.uintLessThan(u8, 5) == 0;
        w.flaky.fail = w.rnd.uintLessThan(u8, 6) == 0;
        defer w.flaky.fail = false;
        switch (w.rnd.uintLessThan(u8, 13)) {
            0 => p.promise(l, w.rnd.uintLessThan(usize, 5)) catch {
                w.seen.refused += 1;
            },
            1 => p.unpromise(l),
            2 => _ = p.take(l) catch |e| try w.failed(e),
            3 => p.cover(l, l.ids.items.len + w.rnd.uintLessThan(usize, 4), w.table()) catch |e| try w.failed(e),
            4 => {
                const n = w.rnd.uintLessThan(usize, 5);
                const step_ = 1 + w.rnd.uintLessThan(usize, 4);
                const run = l.ids.items.len == 0 and n > 1 and l.promised >= n and p.findAligned(n, step_) != null;
                if (p.coverRun(l, n, step_, w.table())) {
                    w.seen.aligned += @intFromBool(run);
                } else |e| try w.failed(e);
            },
            5 => w.claim(l),
            6 => if (w.nruns > 0) w.unclaim(w.rnd.uintLessThan(usize, w.nruns)),
            7 => if (shared) {
                p.retain(id);
                w.store[id] += 1;
            },
            8 => if (p.drop(id)) {
                w.store[id] -= 1;
            } else |_| {
                try equal(@as(u32, 0), w.store[id]);
                w.seen.not_held += 1;
            },
            9 => if (shared) { // a resume: the store holds the page once more for the stream, which adopts it
                p.retain(id);
                p.adopt(l, &.{id}) catch {
                    w.seen.rollbacks += 1;
                };
                w.seen.adopts += 1;
            },
            10 => if (shared and l.ids.items.len > 0) {
                p.share(l, w.rnd.uintLessThan(usize, l.ids.items.len), id);
                w.seen.shares += 1;
            },
            11 => if (w.store[id] > 0) {
                p.release(id);
                w.store[id] -= 1;
            },
            12 => p.finish(l),
            else => unreachable,
        }
    }

    /// Holders and leases a page are the store's holds, the leases mapping it and its run's.
    fn model(w: *const Walk) !void {
        var mapped: [most]u32 = @splat(0);
        var promised: usize = 0;
        for (w.leases) |l| {
            promised += l.promised;
            for (l.ids.items) |id| mapped[id] += 1;
        }
        try equal(promised, w.pool.promised);
        var listed: [most]u32 = @splat(0);
        for (w.pool.free[0..w.pool.nfree]) |id| listed[id] += 1;
        for (0..w.pool.total()) |id| {
            const run: u32 = @intFromBool(w.in_run[id]);
            try equal(mapped[id] + run, w.pool.leased[id]);
            try equal(w.store[id] + mapped[id] + run, w.pool.refs[id]);
            try equal(@as(u32, @intFromBool(w.pool.refs[id] == 0)), listed[id]);
        }
    }

    /// Every lease, run and hold given back: the pool is idle and every page free.
    fn drain(w: *Walk) !void {
        for (&w.leases) |*l| w.pool.finish(l);
        while (w.nruns > 0) w.unclaim(0);
        for (&w.store, 0..) |*n, id| while (n.* > 0) : (n.* -= 1) w.pool.release(@intCast(id));
        try agrees(&w.pool);
        try expect(w.pool.idle());
        try equal(w.pool.total(), w.pool.nfree);
    }
};

test "page accounting: seeded walks over every transition keep the counts equal to main's full scan" {
    var flaky: Flaky = .{};
    var all: Seen = .{};
    for (0..96) |seed| {
        var prng = std.Random.DefaultPrng.init(0x7f4a_0000 + seed);
        var w: Walk = .{ .pool = try kv.Pool.init(flaky.allocator(), 8 + seed % (most - 7), 4096), .flaky = &flaky, .rnd = prng.random() };
        defer w.pool.deinit();
        if (seed % 3 != 0) w.pool.reclaim = .{ .ctx = &w, .call = Walk.reclaim };
        for (0..500) |_| {
            try w.step();
            try agrees(&w.pool);
            try w.model();
        }
        try w.drain();
        inline for (@typeInfo(Seen).@"struct".field_names) |f| @field(all, f) += @field(w.seen, f);
    }
    inline for (@typeInfo(Seen).@"struct".field_names) |f| try expect(@field(all, f) > 0);
}
