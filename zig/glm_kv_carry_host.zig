//! TF_GLM_KV_SPLIT's carry on two ranks of a recording driver: a kept state's latent rows from its home to the owner.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const t = std.testing;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const Plan = glm.snapshots.layout.Plan;
const carry = glm.kv_carry;
const Paged = @typeInfo(@TypeOf(glm.forward.State.initPaged)).@"fn".param_types[4].?;
const Seq = @typeInfo(@FieldType(glm.engine.Engine, "cur")).pointer.child;

const cap = 10; // pages a class: 20 pages, two classes
const at = 10 * 512 - 8; // the state's rows reach every one of its ten pages
const odd = [_]u32{ 11, 3, 19, 1, 15, 5, 7, 17, 9, 13 }; // the kept stream's, class 1
const even = [_]u32{ 4, 10, 0, 16, 2, 8, 14, 6, 18, 12 }; // the resumed stream's, class 0

/// A rank's pages: every page's pooled keys, then its region (`cap` places and the trash page).
const Pages = struct {
    kv: cuda.DeviceBuffer,
    lat: usize, // a page's latent rows (both caches)
    pk: usize, // a page's pooled keys

    fn init(d: *const cuda.Driver, c: glm.config.Config) !Pages {
        const lat = 2 * 512 * c.kv_lora * 2;
        const pk = 2 * 128 * c.i_dim * 2;
        return .{ .kv = try cuda.DeviceBuffer.alloc(d, 2 * cap * pk + (cap + 1) * lat), .lat = lat, .pk = pk };
    }

    /// A paged sequence on `ids` whose owner is `owner`; `held`: this rank is the owner, its places id / 2.
    fn seq(p: *const Pages, d: *const cuda.Driver, f: *Fixture, ids: []const u32, owner: u32, places: []const u32) !Seq {
        var w: glm.weights.Weights = undefined;
        w.config = f.e.c;
        w.share = f.e.share;
        var layers: [2]glm.weights.Layer = undefined;
        layers[0].mixer = .{ .mla = undefined };
        layers[1].mixer = .{ .kda = undefined };
        w.layers = &layers;
        const on: Paged = .{ .kv = p.kv.ptr + 2 * cap * p.pk, .pages = glm.pages.tableLen(16448), .page_bytes = p.lat, .pk = p.kv.ptr, .pk_page = p.pk, .split = true };
        var s = f.seq;
        s.st = try glm.forward.State.initPaged(t.allocator, d, &w, 16448, on);
        s.st.pages = ids;
        s.st.lpages = places;
        s.st.owner = owner;
        s.st.capacity = ids.len * 512;
        return s;
    }
};

/// Each part's bytes as the plan reads them (parts this rank does not hold: empty).
fn parts(plan: *const Plan) ![]std.ArrayList(u8) {
    const out = try t.allocator.alloc(std.ArrayList(u8), plan.count);
    for (out, plan.parts[0..plan.count]) |*o, part| {
        o.* = .empty;
        if (!plan.holds(part)) continue;
        var done: usize = 0;
        while (done < part.len) {
            const x = plan.piece(part, done);
            try o.appendSlice(t.allocator, rec.bytes(x.live, x.len));
            done += x.len;
        }
    }
    return out;
}

fn free(x: []std.ArrayList(u8)) void {
    for (x) |*o| o.deinit(t.allocator);
    t.allocator.free(x);
}

/// NCCL between the two ranks on the host: each send kept in order, each receive the next of the same size.
const Wire = struct {
    var sent: std.ArrayList([]u8) = .empty;
    var taken: usize = 0;
    var sends: usize = 0;
    var recvs: usize = 0;

    fn reset() void {
        for (sent.items) |x| t.allocator.free(x);
        sent.clearAndFree(t.allocator);
        taken = 0;
        sends = 0;
        recvs = 0;
    }
    fn ncclSend(ptr: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        sent.append(t.allocator, t.allocator.dupe(u8, rec.bytes(ptr, n)) catch return 1) catch return 1;
        sends += 1;
        return 0;
    }
    fn ncclRecv(ptr: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        if (taken == sent.items.len or sent.items[taken].len != n) return 1; // every receive meets a send of its size
        @memcpy(rec.bytes(ptr, n), sent.items[taken]);
        taken += 1;
        recvs += 1;
        return 0;
    }
    // carry.run's driver over the same wire
    d: *const cuda.Driver,
    pub fn copy(w: Wire, dst: u64, src: u64, n: usize) !void {
        _ = w.d;
        @memcpy(rec.bytes(dst, n), rec.bytes(src, n));
    }
    pub fn send(_: Wire, ptr: u64, n: usize, _: u32) !void {
        if (ncclSend(ptr, n, .u8, 0, null, null) != 0) return error.Wire;
    }
    pub fn recv(_: Wire, ptr: u64, n: usize, _: u32) !void {
        if (ncclRecv(ptr, n, .u8, 0, null, null) != 0) return error.Wire;
    }
};

fn fill(buf: cuda.DeviceBuffer, salt: usize) void {
    for (rec.bytes(buf.ptr, buf.len), 0..) |*b, i| b.* = @truncate(i *% 29 +% i / 509 +% salt);
}

test "GLM KV split carry: the home's latent rows land in the owner's places, in plan order, for any room size" {
    var d = rec.init();
    var f: [2]Fixture = undefined;
    for (&f, 0..) |*x, r| try x.init(&d, r, 2, true);
    defer for (&f) |*x| x.deinit();
    var pg: [2]Pages = undefined;
    for (&pg) |*p| p.* = try Pages.init(&d, f[0].e.c);
    defer for (&pg) |*p| p.kv.free();
    var odd_at: [cap]u32 = undefined;
    var even_at: [cap]u32 = undefined;
    for (&odd_at, &even_at, odd, even) |*o, *e, x, y| {
        o.* = x / 2;
        e.* = y / 2;
    }
    var kept = try pg[1].seq(&d, &f[1], &odd, 1, &odd_at); // rank 1, the home
    defer kept.st.deinit();
    var away = try pg[1].seq(&d, &f[1], &even, 0, &.{}); // the resumed stream on rank 1: rank 0 owns it
    defer away.st.deinit();
    var owned = try pg[0].seq(&d, &f[0], &even, 0, &even_at); // and on rank 0, its owner
    defer owned.st.deinit();
    fill(pg[1].kv, 3);
    fill(kept.st.arena.buf, 7);
    try kept.st.restoreHost(f[1].e.stream, at, at, &.{1}, .{ 0, 0 });
    try away.st.restoreHost(f[1].e.stream, at, at, &.{1}, .{ 0, 0 });
    try owned.st.restoreHost(f[0].e.stream, at, at, &.{1}, .{ 0, 0 });
    try f[1].e.prepareSnapshots(4 * f[1].e.stateBytes(at));
    f[1].e.cur = &kept;
    const id = try f[1].e.keepState();
    const s = f[1].e.snaps.get(id).?;
    const base = f[1].e.snap_pages.?.arena.?.ptr();
    const want = try parts(&try Plan.init(&kept.st, &f[0].e.c, at, at, 0));
    defer free(want);
    var room = try cuda.DeviceBuffer.alloc(&d, 5 << 20);
    defer room.free();
    for ([_]usize{ 1000, 4096, 777_777, 5 << 20 }) |n| {
        Wire.reset();
        defer Wire.reset();
        @memset(rec.bytes(pg[0].kv.ptr, pg[0].kv.len), 0xE9);
        const send_plan = try Plan.init(&away.st, &f[1].e.c, at, at, 0);
        const recv_plan = try Plan.init(&owned.st, &f[0].e.c, at, at, 0);
        const sent = try carry.run(Wire{ .d = &d }, &send_plan, .send, 0, s.lease.ids.items, base, room.ptr, n);
        const got = try carry.run(Wire{ .d = &d }, &recv_plan, .recv, 1, &.{}, 0, room.ptr, n);
        try t.expectEqual(sent, got);
        try t.expectEqual(carry.bytes(&recv_plan), got);
        try t.expectEqual(std.math.divCeil(usize, got, n) catch unreachable, Wire.sends); // the same pieces on both sides
        try t.expectEqual(Wire.sends, Wire.recvs);
        const have = try parts(&recv_plan);
        defer free(have);
        for (recv_plan.parts[0..recv_plan.count], want, have) |part, w, h| if (part.alt) try t.expectEqualSlices(u8, w.items, h.items);
        // nothing written past the stream's places: the trash page and the places of class 0's other pages stay
        const region = rec.bytes(pg[0].kv.ptr + 2 * cap * pg[0].pk, (cap + 1) * pg[0].lat);
        for (0..cap + 1) |place| if (std.mem.indexOfScalar(u32, &even_at, @intCast(place)) == null)
            for (region[place * pg[0].lat ..][0..pg[0].lat]) |b| try t.expectEqual(@as(u8, 0xE9), b);
    }
    try t.expectEqual(carry.Role.send, carry.role(1, 0, 1));
    try t.expectEqual(carry.Role.recv, carry.role(1, 0, 0));
    try t.expectEqual(carry.Role.none, carry.role(1, 0, 2)); // TP=3: a third rank takes no part
    try t.expectEqual(carry.Role.none, carry.role(1, 1, 1)); // at home: no carry
    try t.expectError(error.KvSplitCarry, carry.run(Wire{ .d = &d }, &try Plan.init(&away.st, &f[1].e.c, at, at, 0), .recv, 1, &.{}, 0, room.ptr, 4096));
}

test "GLM KV split carry: a restore off its home moves each rank's own parts and the home's latent rows to the owner" {
    var d = rec.init();
    var f: [2]Fixture = undefined;
    for (&f, 0..) |*x, r| try x.init(&d, r, 2, true);
    defer for (&f) |*x| x.deinit();
    var pg: [2]Pages = undefined;
    for (&pg) |*p| p.* = try Pages.init(&d, f[0].e.c);
    defer for (&pg) |*p| p.kv.free();
    var odd_at: [cap]u32 = undefined;
    var even_at: [cap]u32 = undefined;
    for (&odd_at, &even_at, odd, even) |*o, *e, x, y| {
        o.* = x / 2;
        e.* = y / 2;
    }
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclSend = Wire.ncclSend;
    lib.api.ncclRecv = Wire.ncclRecv;
    var rooms: [2]cuda.DeviceBuffer = undefined;
    for (&rooms) |*r| r.* = try cuda.DeviceBuffer.alloc(&d, 3000); // pieces smaller than one row's span
    defer for (&rooms) |*r| r.free();
    // the kept stream, class 1: rank 1 holds its latent rows; each rank keeps its own copy at the same mark
    var kept: [2]Seq = .{ try pg[0].seq(&d, &f[0], &odd, 1, &.{}), try pg[1].seq(&d, &f[1], &odd, 1, &odd_at) };
    defer for (&kept) |*k| k.st.deinit();
    var resumed: [2]Seq = .{ try pg[0].seq(&d, &f[0], &even, 0, &even_at), try pg[1].seq(&d, &f[1], &even, 0, &.{}) };
    defer for (&resumed) |*k| k.st.deinit();
    var want: [2][]std.ArrayList(u8) = undefined;
    var ids: [2]u64 = undefined;
    for (0..2) |r| {
        fill(pg[r].kv, 11 + r);
        fill(kept[r].st.arena.buf, 17 + r);
        try kept[r].st.restoreHost(f[r].e.stream, at, at, &.{1}, .{ 0, 0 });
        want[r] = try parts(&try Plan.init(&kept[r].st, &f[r].e.c, at, at, 0));
        try f[r].e.prepareSnapshots(4 * f[r].e.stateBytes(at));
        f[r].e.cur = &kept[r];
        ids[r] = try f[r].e.keepState();
        f[r].e.room = .{ .buf = rooms[r], .rank = r, .world = 2, .rows = 1, .lw = f[r].e.c.kv_lora, .heads = @splat(0), .lib = &lib, .comm = null };
    }
    defer for (want) |w| free(w);
    defer Wire.reset();
    for (0..2) |r| @memset(rec.bytes(pg[r].kv.ptr, pg[r].kv.len), 0xE9);
    for ([_]usize{ 1, 0 }) |r| { // the home sends first; the wire keeps its pieces for the owner's receives
        f[r].e.cur = &resumed[r];
        try f[r].e.restoreState(ids[r]);
    }
    try t.expect(Wire.sends > 1 and Wire.sends == Wire.recvs);
    for (0..2) |r| {
        const plan = try Plan.init(&resumed[r].st, &f[r].e.c, at, at, 0);
        const have = try parts(&plan);
        defer free(have);
        for (plan.parts[0..plan.count], want[r], want[1], have) |part, own, home, h| {
            if (!plan.holds(part)) continue; // rank 1 holds none of the resumed stream's latent rows
            try t.expectEqualSlices(u8, if (part.alt) home.items else own.items, h.items);
        }
    }
    // a split state on an unsplit stream, or an unsplit one on a split stream: refused by name
    resumed[0].st.owner = null;
    f[0].e.cur = &resumed[0];
    try t.expectError(error.KvSplitAway, f[0].e.restoreState(ids[0]));
    for (&f) |*x| x.e.cur = &x.seq;
}

/// The host copies a disk image takes (cuda.memory's download and upload), on host memory.
const Host = struct {
    fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], rec.bytes(src, n));
        return 0;
    }
    fn htod(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) cuda.abi.Result {
        @memcpy(rec.bytes(dst, n), @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn device(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0; // a discrete card: a download copies straight to the host
        return 0;
    }
};

test "GLM KV split disk: each rank's image holds its own parts, the home's the latent rows; read back, resumed equals kept" {
    const img = glm.disk_image;
    var d = rec.init();
    d.api.cuMemcpyDtoH_v2 = Host.dtoh;
    d.api.cuMemcpyHtoD_v2 = Host.htod;
    d.api.cuCtxGetDevice = Host.device;
    d.api.cuDeviceGetAttribute = Host.attribute;
    var f: [2]Fixture = undefined;
    for (&f, 0..) |*x, r| try x.init(&d, r, 2, true);
    defer for (&f) |*x| x.deinit();
    var pg: [2]Pages = undefined;
    for (&pg) |*p| p.* = try Pages.init(&d, f[0].e.c);
    defer for (&pg) |*p| p.kv.free();
    var odd_at: [cap]u32 = undefined;
    for (&odd_at, odd) |*o, x| o.* = x / 2;
    var kept: [2]Seq = .{ try pg[0].seq(&d, &f[0], &odd, 1, &.{}), try pg[1].seq(&d, &f[1], &odd, 1, &odd_at) };
    defer for (&kept) |*k| k.st.deinit();
    for (0..2) |r| {
        const e = &f[r].e;
        fill(pg[r].kv, 23 + r);
        fill(kept[r].st.arena.buf, 29 + r);
        try kept[r].st.restoreHost(e.stream, at, at, &.{1}, .{ 1024, 1024 }); // rings that still hold the state's rows
        const want = try parts(&try Plan.init(&kept[r].st, &e.c, at, at, 0));
        defer free(want);
        try e.prepareSnapshots(4 * e.stateBytes(at));
        e.cur = &kept[r];
        defer e.cur = &f[r].seq;
        const id = try e.keepState();
        // out: this rank's image, its latent runs absent off the home
        const sh = try img.Shape.of(&kept[r].st, &e.c, e.capacity, true);
        var segs: std.ArrayList(img.Seg) = .empty;
        defer segs.deinit(t.allocator);
        try img.segments(sh, at, at, &segs, t.allocator, r == 1);
        const last = segs.items[segs.items.len - 1];
        const bytes = try t.allocator.alloc(u8, @intCast(last.img + last.len));
        defer t.allocator.free(bytes);
        @memset(bytes, 0);
        const s = e.snaps.get(id).?;
        var words = (img.Words{ .at = at, .mtp_len = at, .base = s.bases[0], .mtp_base = s.bases[1], .cur = 1, .world = 2, .layout = .split, .owner = 1 }).encode();
        var absent: u64 = 0;
        for (segs.items) |g| {
            if (g.from == .absent) {
                absent += g.len;
                continue;
            }
            const cp: img.Copy = .{ .d = &d, .pool = &e.snap_pages.?, .lease = &s.lease, .segs = segs.items, .words = &words };
            try cp.run(g.img, bytes[@intCast(g.img)..][0..@intCast(g.len)], true);
        }
        try t.expectEqual(r == 0, absent > 0); // the home's image has every byte
        // in: a blank state at home 1, filled from the image, sealed by its words
        try glm.snapshots.blankAs(e, 99, at, 1);
        var back: [img.Words.count]u64 = undefined;
        const b = e.snaps.get(99).?;
        for (segs.items) |g| if (g.from != .absent) {
            const cp: img.Copy = .{ .d = &d, .pool = &e.snap_pages.?, .lease = &b.lease, .segs = segs.items, .words = &back };
            try cp.run(g.img, bytes[@intCast(g.img)..][0..@intCast(g.len)], false);
        };
        const w = try img.Words.decode(back, at, 2, .split);
        try t.expectEqual(@as(u64, 1), w.owner);
        try t.expectError(error.DiskLayout, img.Words.decode(back, at, 2, .unsplit)); // an unsplit server refuses it
        try glm.snapshots.seal(e, 99, @intCast(w.mtp_len), .{ w.base, w.mtp_base }, w.cur);
        try t.expectEqual(e.snaps.get(id).?.len, e.snaps.get(99).?.len); // the copy's own bytes, as the keep's
        try t.expectEqual(@as(?u32, 1), e.snaps.get(99).?.home);
        // resumed at home from the read: every byte this rank's plan holds is the kept one
        @memset(rec.bytes(pg[r].kv.ptr, pg[r].kv.len), 0xE9);
        @memset(rec.bytes(kept[r].st.arena.buf.ptr, kept[r].st.arena.buf.len), 0xE9);
        try e.restoreState(99);
        const have = try parts(&try Plan.init(&kept[r].st, &e.c, at, at, 0));
        defer free(have);
        for (want, have) |x, y| try t.expectEqualSlices(u8, x.items, y.items);
        try t.expectError(error.BadStep, glm.snapshots.blankAs(e, 100, at, 2)); // a home past the ranks
    }
}
