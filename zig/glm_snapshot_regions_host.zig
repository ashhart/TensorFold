//! TF_GLM_KV_SPLIT's kept-state plan: a page's pooled keys and latent rows in two regions, each through its own stride.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const Plan = glm.snapshots.layout.Plan;
const Paged = @typeInfo(@TypeOf(glm.forward.State.initPaged)).@"fn".param_types[4].?;

test {
    _ = @import("glm_kv_carry_host.zig");
}

const ids = [_]u32{ 5, 2, 9, 0, 7, 1, 3, 8, 4, 6 }; // every page of the pool, out of order
const at = ids.len * 512 - 8; // its latent rows and its pooled keys (at / 4 + 2 rows) reach every leased page

/// The fixture's page geometry: two caches (the DSA layer's and the MTP layer's), each a latent and a pooled part.
const Geometry = struct {
    lat_w: usize, // a latent row's bytes
    pk_w: usize, // a pooled key's bytes

    fn of(c: glm.config.Config) Geometry {
        return .{ .lat_w = c.kv_lora * 2, .pk_w = c.i_dim * 2 };
    }
    fn latCache(g: Geometry) usize {
        return 512 * g.lat_w;
    }
    fn pkCache(g: Geometry) usize {
        return 128 * g.pk_w;
    }
    fn page(g: Geometry) usize {
        return 2 * (g.latCache() + g.pkCache());
    }
    /// Latent row `r` of cache `j`: split, in the latent region past every page's pooled keys; else in its page.
    fn latRow(g: Geometry, kv: u64, split: bool, j: usize, r: usize) u64 {
        const id: u64 = ids[r / 512];
        const in = (r % 512) * g.lat_w;
        if (split) return kv + ids.len * 2 * g.pkCache() + id * 2 * g.latCache() + j * g.latCache() + in;
        return kv + id * g.page() + j * (g.latCache() + g.pkCache()) + in;
    }
    /// Pooled key `p` of cache `j`: split, in the pooled region first; else after its cache's latent rows.
    fn pkRow(g: Geometry, kv: u64, split: bool, j: usize, p: usize) u64 {
        const id: u64 = ids[p / 128];
        const in = (p % 128) * g.pk_w;
        if (split) return kv + id * 2 * g.pkCache() + j * g.pkCache() + in;
        return kv + id * g.page() + j * (g.latCache() + g.pkCache()) + g.latCache() + in;
    }
};

/// A paged sequence of the fixture's shape on the pages at `kv`, in two regions (`split`) or interleaved.
fn pagedSeq(d: *const cuda.Driver, f: *const Fixture, kv: u64, split: bool) !glm.forward.State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    const g = Geometry.of(f.e.c);
    const pages = glm.pages.tableLen(16448);
    const on: Paged = if (split)
        .{ .kv = kv + ids.len * 2 * g.pkCache(), .pages = pages, .page_bytes = 2 * g.latCache(), .pk = kv, .pk_page = 2 * g.pkCache(), .split = true }
    else
        .{ .kv = kv, .pages = pages, .page_bytes = g.page() };
    var st = try glm.forward.State.initPaged(t.allocator, d, &w, 16448, on);
    st.pages = &ids;
    if (split) st.lpages = &ids; // every row on every rank: the latent region holds every page at its id
    st.capacity = ids.len * 512;
    return st;
}

/// Each cache's `n` latent rows and pooled keys (n / 4 + 2), as the plan reads them; `f(addr, len, j, row, pooled)`.
fn eachRow(g: Geometry, kv: u64, split: bool, head: ?usize, ctx: anytype, comptime f: anytype) !void {
    for (0..2) |j| {
        const n = if (j == 0) at else head orelse break;
        for (0..n) |r| try f(ctx, g.latRow(kv, split, j, r), g.lat_w, j, r, false);
        for (0..n / 4 + 2) |p| try f(ctx, g.pkRow(kv, split, j, p), g.pk_w, j, p, true);
    }
}

/// Every byte a plan's paged parts read, counted on a map of the pages at `kv`; its other parts lie off the pages.
const Cover = struct {
    kv: u64,
    hits: []u8,

    fn count(c: *Cover, plan: *const Plan) !void {
        for (plan.parts[0..plan.count]) |part| {
            if (!plan.holds(part)) continue;
            var done: usize = 0;
            while (done < part.len) {
                const x = plan.piece(part, done);
                done += x.len;
                if (part.span == 0) {
                    try t.expect(x.live + x.len <= c.kv or x.live >= c.kv + c.hits.len);
                    continue;
                }
                try t.expect(x.live >= c.kv and x.live + x.len <= c.kv + c.hits.len);
                for (c.hits[x.live - c.kv ..][0..x.len]) |*h| h.* += 1;
            }
        }
    }
    fn expect(c: *Cover, addr: u64, len: usize, _: usize, _: usize, _: bool) !void {
        for (c.hits[addr - c.kv ..][0..len]) |*h| {
            try t.expectEqual(@as(u8, 1), h.*); // read once: no row missed and none read twice
            h.* = 0;
        }
    }
};

/// A row's bytes by what it is, so either layout holds the same row in the same bytes.
fn fill(_: void, addr: u64, len: usize, j: usize, row: usize, pooled: bool) !void {
    for (rec.bytes(addr, len), 0..) |*b, k| b.* = @truncate(j * 131 + row * 17 + k * 3 + @as(usize, @intFromBool(pooled)) * 101);
}

/// The plan's bytes in its order, as a kept state copies them.
fn gather(plan: *const Plan, out: *std.ArrayList(u8)) !void {
    for (plan.parts[0..plan.count]) |part| {
        if (!plan.holds(part)) continue;
        var done: usize = 0;
        while (done < part.len) {
            const x = plan.piece(part, done);
            try out.appendSlice(t.allocator, rec.bytes(x.live, x.len));
            done += x.len;
        }
    }
}

test "GLM snapshots: with TF_GLM_KV_SPLIT a plan reads each page's pooled keys and latent rows once, every page" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    const g = Geometry.of(f.e.c);
    var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * g.page());
    defer kv.free();
    var st = try pagedSeq(&d, &f, kv.ptr, true);
    defer st.deinit();
    const hits = try t.allocator.alloc(u8, kv.len);
    defer t.allocator.free(hits);
    for ([_]?usize{ null, at }) |head| {
        @memset(hits, 0);
        var c: Cover = .{ .kv = kv.ptr, .hits = hits };
        const plan = try Plan.init(&st, &f.e.c, at, head, 0);
        try c.count(&plan);
        // every leased page is read in both regions
        const pooled = ids.len * 2 * g.pkCache();
        for (ids) |id| {
            try t.expect(std.mem.indexOfNone(u8, hits[id * 2 * g.pkCache() ..][0 .. 2 * g.pkCache()], &.{0}) != null);
            try t.expect(std.mem.indexOfNone(u8, hits[pooled + id * 2 * g.latCache() ..][0 .. 2 * g.latCache()], &.{0}) != null);
        }
        try eachRow(g, kv.ptr, true, head, &c, Cover.expect);
        try t.expect(std.mem.indexOfNone(u8, hits, &.{0}) == null); // nothing read but the rows
    }
}

test "GLM snapshots: with TF_GLM_KV_SPLIT and every row on every rank, a plan's bytes are the interleaved pages' plan's" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    const g = Geometry.of(f.e.c);
    var states: [2]glm.forward.State = undefined;
    var kvs: [2]cuda.DeviceBuffer = undefined;
    for (0..2) |i| {
        kvs[i] = try cuda.DeviceBuffer.alloc(&d, ids.len * g.page());
        states[i] = try pagedSeq(&d, &f, kvs[i].ptr, i == 1);
        const a = states[i].arena.buf;
        for (rec.bytes(a.ptr, a.len), 0..) |*b, k| b.* = @truncate(k * 7 + 3); // the same fields, the same bytes
        @memset(rec.bytes(kvs[i].ptr, kvs[i].len), if (i == 1) 0xA5 else 0x3C); // between rows: not equal
        try eachRow(g, kvs[i].ptr, i == 1, at, {}, fill);
    }
    defer for (&states, &kvs) |*st, *kv| {
        st.deinit();
        kv.free();
    };
    for ([_]?usize{ null, at }) |head| {
        var bytes: [2]std.ArrayList(u8) = .{ .empty, .empty };
        defer for (&bytes) |*b| b.deinit(t.allocator);
        var lens: [2]usize = undefined;
        for (0..2) |i| {
            const plan = try Plan.init(&states[i], &f.e.c, at, head, 0);
            lens[i] = plan.len;
            try gather(&plan, &bytes[i]);
        }
        try t.expectEqual(lens[0], lens[1]);
        try t.expectEqual(lens[0], bytes[0].items.len);
        try t.expectEqualSlices(u8, bytes[0].items, bytes[1].items);
    }
}

/// Owner-only rows: pages of class 1 of 2 (odd ids), latent rows at place id / 2 of `cap` and a trash page.
const odd = [_]u32{ 11, 3, 19, 1, 15, 5, 7, 17, 9, 13 };
const cap = 10;

fn ownerSeq(d: *const cuda.Driver, f: *const Fixture, kv: u64, lat: []const u32) !glm.forward.State {
    var w: glm.weights.Weights = undefined;
    w.config = f.e.c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    const g = Geometry.of(f.e.c);
    const on: Paged = .{ .kv = kv + 2 * cap * 2 * g.pkCache(), .pages = glm.pages.tableLen(16448), .page_bytes = 2 * g.latCache(), .pk = kv, .pk_page = 2 * g.pkCache(), .split = true };
    var st = try glm.forward.State.initPaged(t.allocator, d, &w, 16448, on);
    st.pages = &odd;
    st.lpages = lat;
    st.capacity = odd.len * 512;
    return st;
}

test "GLM snapshots: with TF_GLM_KV_SPLIT the owner's plan reads latent rows at their region places, another's none" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.init(&d, 0, 2, true);
    defer f.deinit();
    const g = Geometry.of(f.e.c);
    const pooled = 2 * cap * 2 * g.pkCache(); // the pooled region: 20 pages' keys
    var kv = try cuda.DeviceBuffer.alloc(&d, pooled + (cap + 1) * 2 * g.latCache());
    defer kv.free();
    var places: [odd.len]u32 = undefined;
    for (&places, odd) |*x, id| x.* = id / 2;
    var own = try ownerSeq(&d, &f, kv.ptr, &places);
    defer own.deinit();
    var other = try ownerSeq(&d, &f, kv.ptr, &.{});
    defer other.deinit();
    const hits = try t.allocator.alloc(u8, kv.len);
    defer t.allocator.free(hits);
    for ([_]?usize{ null, at }) |head| {
        @memset(hits, 0);
        var c: Cover = .{ .kv = kv.ptr, .hits = hits };
        const mine = try Plan.init(&own, &f.e.c, at, head, 0);
        try c.count(&mine);
        for (0..2) |j| {
            const n = if (j == 0) at else head orelse break;
            for (0..n) |r| { // latent row r of cache j: page odd[r / 512] at place odd / 2, never the trash page
                const a = pooled + @as(usize, odd[r / 512] / 2) * 2 * g.latCache() + j * g.latCache() + (r % 512) * g.lat_w;
                for (hits[a..][0..g.lat_w]) |*h| {
                    try t.expectEqual(@as(u8, 1), h.*);
                    h.* = 0;
                }
            }
            for (0..n / 4 + 2) |q| try c.expect(kv.ptr + @as(u64, odd[q / 128]) * 2 * g.pkCache() + j * g.pkCache() + (q % 128) * g.pk_w, g.pk_w, j, q, true);
        }
        try t.expect(std.mem.indexOfNone(u8, hits, &.{0}) == null); // nothing else, the trash page included
        // another rank's copy: the same parts in the same order, the latent rows' neither read nor counted
        const theirs = try Plan.init(&other, &f.e.c, at, head, 0);
        var latent: usize = 0;
        for (mine.parts[0..mine.count]) |part| latent += if (part.alt) part.len else 0;
        try t.expect(latent > 0);
        try t.expectEqual(mine.len - latent, theirs.len);
        try t.expectEqual(mine.count, theirs.count);
        for (mine.parts[0..mine.count], theirs.parts[0..theirs.count]) |x, y| {
            try t.expect(x.alt == y.alt and x.len == y.len and mine.holds(x));
            try t.expectEqual(!y.alt, theirs.holds(y));
        }
        // sizing counts the latent rows on every rank, so every rank's lease is the owner's
        try t.expectEqual(try Plan.size(&own, &f.e.c, 16448, at, head, 0), try Plan.size(&other, &f.e.c, 16448, at, head, 0));
    }
}

test "GLM snapshots: with TF_GLM_KV_SPLIT a kept state records its home, every rank leases its size, restores at home" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.initFormat(&d, 1, 2, true, true); // rank 1, the owner; 512-wide latent rows: past a snapshot page
    defer f.deinit();
    const g = Geometry.of(f.e.c);
    const pooled = 2 * cap * 2 * g.pkCache();
    var kv = try cuda.DeviceBuffer.alloc(&d, pooled + (cap + 1) * 2 * g.latCache());
    defer kv.free();
    var places: [odd.len]u32 = undefined;
    for (&places, odd) |*x, id| x.* = id / 2;
    var own = f.seq;
    own.st = try ownerSeq(&d, &f, kv.ptr, &places);
    defer own.st.deinit();
    var other = f.seq;
    other.st = try ownerSeq(&d, &f, kv.ptr, &.{});
    defer other.st.deinit();
    const window = f.e.cur;
    defer f.e.cur = window;
    try f.e.prepareSnapshots(4 * f.e.stateBytes(at));
    for ([_]*@TypeOf(own){ &own, &other }, 0..) |s, k| {
        for (rec.bytes(s.st.arena.buf.ptr, s.st.arena.buf.len), 0..) |*b, i| b.* = @truncate(i * 7 + k);
        s.st.owner = 1; // class 1's stream: `own` is its owner's sequence, `other` another rank's
        try s.st.restoreHost(f.e.stream, at, at, &.{1}, .{ 0, 0 });
    }
    for (rec.bytes(kv.ptr, kv.len), 0..) |*b, i| b.* = @truncate(i * 13 + 5);
    var before: std.ArrayList(u8) = .empty;
    defer before.deinit(t.allocator);
    try gather(&try Plan.init(&own.st, &f.e.c, at, at, 0), &before);
    f.e.cur = &own;
    const mine = try f.e.keepState();
    f.e.cur = &other;
    const theirs = try f.e.keepState();
    const a = f.e.snaps.get(mine).?;
    const b = f.e.snaps.get(theirs).?;
    try t.expectEqual(@as(?u32, 1), a.home);
    try t.expectEqual(@as(?u32, 1), b.home);
    try t.expectEqual(a.lease.ids.items.len, b.lease.ids.items.len); // the home's size on every rank
    const sp = glm.snapshots.pages;
    try t.expectEqual(try sp.rounded(f.e.stateBytes(at)) / sp.page, b.lease.ids.items.len); // as the store charges it
    try t.expect(b.len < a.len); // but only the home's copy holds the latent rows
    // at home: the rows come back as kept
    @memset(rec.bytes(kv.ptr, kv.len), 0xE9);
    @memset(rec.bytes(own.st.arena.buf.ptr, own.st.arena.buf.len), 0xE9);
    f.e.cur = &own;
    try f.e.restoreState(mine);
    var after: std.ArrayList(u8) = .empty;
    defer after.deinit(t.allocator);
    try gather(&try Plan.init(&own.st, &f.e.c, at, at, 0), &after);
    try t.expectEqualSlices(u8, before.items, after.items);
    own.st.owner = null; // unsplit: a split state is not this sequence's
    f.e.cur = &own;
    try t.expectError(error.KvSplitAway, f.e.restoreState(mine));
}
