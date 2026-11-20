//! GLM disk steps across ranks on host memory and a fake link: a failed blank, byte order, refusals, shared bytes.
const std = @import("std");
const cuda = @import("cuda");
const ranks = @import("cuda_disk_ranks.zig");
const image = @import("cuda_disk_image.zig");
const follow = @import("cuda_follow.zig");
const snapshots = @import("cuda_snapshots.zig");
const pages = @import("cuda_snapshot_pages.zig");
const fixture = @import("cuda_disk_image_test.zig");
const State = @import("cuda_state.zig").State;
const Seq = @import("cuda_engine_types.zig").Seq;
const Engine = @import("cuda_engine.zig").Engine;
const t = std.testing;

/// The link's kernel and NCCL: the put kernel records each step's header; a broadcast fails, or brings a rank's bytes.
const Wire = struct {
    var fail = false;
    var odd: c_int = 0; // a rank whose bytes differ from rank 0's (0: none)
    var sent: [8]follow.Header = undefined;
    var steps: usize = 0;
    var casts: usize = 0;
    var seed: u64 = 0; // the last step's image offset: a take's bytes are the image's from there

    fn reset() void {
        fail = false;
        odd = 0;
        steps = 0;
        casts = 0;
    }
    fn byte(i: usize) u8 {
        return @truncate(i *% 13 +% 5);
    }
    fn put(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const w: *align(1) const follow.Words = @ptrCast(params.?[2].?);
        var h: follow.Header = undefined;
        @memcpy(std.mem.asBytes(&h), std.mem.sliceAsBytes(w.w[0..16]));
        seed = h.seed;
        if (steps < sent.len) sent[steps] = h;
        steps += 1;
        return 0;
    }
    fn broadcast(_: cuda.abi.DevicePtr, dst: cuda.abi.DevicePtr, n: usize, _: cuda.nccl.DataType, root: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        if (fail) return 1;
        casts += 1;
        if (root != 0) for (@as([*]u8, @ptrFromInt(dst))[0..n], 0..) |*b, i| {
            b.* = byte(@intCast(seed -% image.Words.bytes +% i)) ^ @intFromBool(root == odd); // repeats every 256 bytes
        };
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

/// A TP=2 engine's rank 0 over host memory (the fields the disk steps read), its snapshot pool and its link.
const Rig = struct {
    m: fixture.Model,
    d: cuda.Driver,
    lib: cuda.nccl.Library,
    seq: Seq,
    e: *Engine,
    buf: [64 + 64 * 4]u8 align(8), // the link's device buffer: header, then ids
    ids: [64]u32,
    link: follow.Link,

    fn init(x: *Rig) !void {
        const a = t.allocator;
        Wire.reset();
        try x.m.init();
        x.d = fixture.H.driver();
        x.d.api.cuLaunchKernel = Wire.put;
        x.d.api.cuCtxGetDevice = Wire.device;
        x.d.api.cuDeviceGetAttribute = Wire.attribute;
        x.lib = .{ .lib = undefined, .api = undefined };
        x.lib.api.ncclBroadcast = Wire.broadcast;
        x.seq = .{ .st = try State.initFormat(a, &x.d, &x.m.w, fixture.capacity, false) };
        errdefer x.seq.st.deinit();
        const e = try a.create(Engine);
        e.gpa = a;
        e.d = &x.d;
        e.c = x.m.w.config;
        e.share = x.m.w.share;
        e.cur = &x.seq;
        e.capacity = fixture.capacity;
        e.mtp = true;
        e.slot = 0;
        e.stream = .{ .d = &x.d, .handle = null };
        e.m.glue.put_words = .{ .d = &x.d, .handle = null };
        e.share_pages = false;
        e.next_snap = 1;
        e.announce = 0;
        e.snaps = .empty;
        e.dropped = .empty;
        e.snap_pages = null;
        x.e = e;
        errdefer x.free();
        try e.snaps.ensureTotalCapacity(a, 8);
        try e.dropped.ensureTotalCapacity(a, 9);
        e.snap_pages = try pages.Pool.init(a, &x.d, 256 << 20, false);
        x.link = .{ .d = &x.d, .lib = &x.lib, .comm = null, .stream = e.stream, .buf = .{ .d = &x.d, .ptr = @intFromPtr(&x.buf), .len = x.buf.len, .borrowed = true }, .ids = &x.ids, .gpa = a, .wait = false };
    }

    fn free(x: *Rig) void {
        const e = x.e;
        var it = e.snaps.valueIterator();
        while (it.next()) |s| snapshots.free(e, s);
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        if (e.snap_pages) |*p| p.deinit();
        e.gpa.destroy(e);
    }

    fn deinit(x: *Rig) void {
        ranks.link = null;
        x.free();
        x.seq.st.deinit();
    }
};

const disk = @backingInt(follow.Kind.disk);

fn op(o: ranks.Op) u32 {
    return @backingInt(o);
}

test "GLM disk ranks: a blank whose step fails on the link gives rank 0's lease back, and the next step drops it" {
    var x: Rig = undefined;
    try x.init();
    defer x.deinit();
    ranks.link = &x.link;
    const pool = &x.e.snap_pages.?;
    const all = pool.spare();
    Wire.fail = true;
    try t.expectError(error.NcclFailed, ranks.blank(x.e, 2048));
    try t.expectEqual(@as(u32, 0), x.e.snaps.count());
    try t.expectEqual(all, pool.spare());
    try t.expectEqualSlices(u32, &.{1}, x.e.dropped.items); // the step may have reached the others: they drop it
    Wire.reset();
    const id = try ranks.blank(x.e, 2048);
    try t.expectEqual(@as(u64, 2), id); // the dropped id is not used again
    try t.expectEqual(@as(usize, 2), Wire.steps);
    try t.expectEqual(@backingInt(follow.Kind.drop), Wire.sent[0].kind);
    try t.expectEqual(@as(u32, 1), Wire.sent[0].n);
    try t.expectEqual(disk, Wire.sent[1].kind);
    try t.expectEqual(op(.blank), Wire.sent[1].flags);
    try t.expectEqual(id, Wire.sent[1].a);
    try t.expect(pool.spare() < all);
}

test "GLM disk ranks: rank 0's link carries the disk steps at TP only, and its deinit leaves none behind" {
    var x: Rig = undefined;
    try x.init();
    defer x.deinit();
    var l = x.link;
    l.ids = try t.allocator.alloc(u32, 4);
    l.lead(true, 1); // one rank: every op stays local
    try t.expect(l.asleep and ranks.link == null);
    l.lead(false, 2);
    try t.expect(!l.asleep and ranks.link == &l);
    var other = x.link;
    other.ids = try t.allocator.alloc(u32, 4);
    other.deinit(); // another link's end leaves rank 0's
    try t.expect(ranks.link == &l);
    l.deinit();
    try t.expect(ranks.link == null);
}

test "GLM disk ranks: a blank state takes bytes past its words and front only once sealed" {
    var x: Rig = undefined;
    try x.init();
    defer x.deinit();
    const at: u32 = 2048;
    const id = try ranks.blank(x.e, at); // one rank: no link
    const front: usize = @intCast(image.Words.bytes + image.front(try ranks.shape(x.e), at));
    const buf = try t.allocator.alloc(u8, front + 64);
    defer t.allocator.free(buf);
    @memset(buf, 0);
    try ranks.copy(x.e, id, 0, buf[0..front], false);
    try t.expectError(error.DiskOrder, ranks.copy(x.e, id, front - 8, buf[0..16], false)); // across the front
    try t.expectError(error.DiskOrder, ranks.copy(x.e, id, front, buf[0..1], true)); // read out, too
    try snapshots.seal(x.e, id, at - 1, .{ 0, 0 }, 0);
    try ranks.copy(x.e, id, front - 8, buf[0..16], false);
    try ranks.copy(x.e, id, front, buf[0..1], true);
}

test "GLM disk ranks: a follower refuses a disk step rank 0 does not send, before its states or the link" {
    var x: Rig = undefined;
    try x.init();
    defer x.deinit();
    x.e.share.rank = 1;
    const room = x.link.ids.len * 4;
    const bad = [_]struct { h: follow.Header, n: usize }{
        .{ .h = .{ .kind = disk, .flags = 0 }, .n = 0 }, // no op
        .{ .h = .{ .kind = disk, .flags = op(.prefixes) + 1 }, .n = 0 }, // past the last op
        .{ .h = .{ .kind = disk, .flags = op(.seal), .a = 1 }, .n = 1 }, // words of another length
        .{ .h = .{ .kind = disk, .flags = op(.put), .a = 1, .depth = 4 * 4 + 1, .top_k = 1 }, .n = 4 }, // past its ids
        .{ .h = .{ .kind = disk, .flags = op(.take), .a = 1, .depth = @intCast(room + 1), .top_k = 0 }, .n = 0 }, // past the link
    };
    for (bad) |b| try t.expectError(error.BadStep, ranks.step(x.e, &x.link, b.h, x.ids[0..b.n]));
    x.e.share.world = 65; // more ranks than a gather holds
    try t.expectError(error.BadStep, ranks.step(x.e, &x.link, .{ .kind = disk, .flags = op(.prefixes) }, &.{}));
    try t.expectEqual(@as(u32, 0), x.e.snaps.count());
    try t.expectEqual(@as(usize, 0), Wire.casts);
}

test "GLM disk ranks: each other rank's shared bytes come from past its own prefix and must equal rank 0's" {
    var x: Rig = undefined;
    try x.init();
    defer x.deinit();
    ranks.link = &x.link;
    const prefixes = [_]u64{ 4096, 1024, 2048 }; // TP=3: each rank's own bytes first (256s: the wire's bytes line up)
    var shared: [100]u8 = undefined;
    for (&shared, 0..) |*b, i| b.* = Wire.byte(i);
    var theirs: [100]u8 = undefined;
    try ranks.same(x.e, 7, &prefixes, &shared, &.{}, &theirs);
    try t.expectEqual(@as(usize, 2), Wire.steps);
    for (Wire.sent[0..2], 1..) |h, r| {
        try t.expectEqual(op(.take), h.flags);
        try t.expectEqual(@as(u64, 7), h.a);
        try t.expectEqual(@as(u32, @intCast(r)), h.top_k);
        try t.expectEqual(image.Words.bytes + prefixes[r], h.seed);
        try t.expectEqual(@as(u32, theirs.len), h.depth);
    }
    // bytes a prompt pass does not write are left out: rank 0's as zeros, each other rank's whatever it holds there
    const loose = [_]image.Span{.{ .off = 10, .len = 20 }};
    image.clear(&loose, 0, &shared);
    try t.expectError(error.RanksDiffer, ranks.same(x.e, 7, &prefixes, &shared, &.{}, &theirs));
    Wire.steps = 0;
    try ranks.same(x.e, 7, &prefixes, &shared, &loose, &theirs);
    // only the bytes past `loose` are read: under TF_GLM_KV_SPLIT a rank off the home holds no latent rows there
    try t.expectEqual(@as(usize, 4), Wire.steps);
    for (Wire.sent[0..4], 0..) |h, k| {
        try t.expectEqual(image.Words.bytes + prefixes[1 + k / 2] + @as(u64, if (k % 2 == 0) 0 else 30), h.seed);
        try t.expectEqual(@as(u32, if (k % 2 == 0) 10 else 70), h.depth);
    }
    Wire.odd = 2;
    try t.expectError(error.RanksDiffer, ranks.same(x.e, 7, &prefixes, &shared, &loose, &theirs));
}
