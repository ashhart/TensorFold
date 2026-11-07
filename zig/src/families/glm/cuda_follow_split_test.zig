//! TF_GLM_KV_SPLIT on the wire: rank 0's pool and begin steps through two actual links' Headers to a follower's loop.
const std = @import("std");
const cuda = @import("cuda");
const follow = @import("cuda_follow.zig");
const pages_mod = @import("cuda_pages.zig");
const Fake = @import("cuda_pages_split_test.zig").Fake;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const gpa = std.testing.allocator;

/// NCCL and the put kernel on the host: rank 0's broadcasts kept in order, each handed to the follower's next.
const Wire = struct {
    var sent: std.ArrayList([]u8) = .empty;
    var taken: usize = 0;
    const leader: cuda.nccl.Comm = @ptrFromInt(0x100);
    const other: cuda.nccl.Comm = @ptrFromInt(0x200);

    fn bcast(send: u64, recv: u64, n: usize, _: cuda.nccl.DataType, root: c_int, comm: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        if (root != 0) return 1;
        if (comm == leader) {
            const src: [*]const u8 = @ptrFromInt(send);
            sent.append(gpa, gpa.dupe(u8, src[0..n]) catch return 1) catch return 1;
            return 0;
        }
        if (taken == sent.items.len or sent.items[taken].len != n) return 1; // the wire ran dry: the follower stops
        const dst: [*]u8 = @ptrFromInt(recv);
        @memcpy(dst[0..n], sent.items[taken]);
        taken += 1;
        return 0;
    }
    /// tf_glm_put_words(dst, n, words): the first `n` words to `dst`, as the kernel writes them.
    fn put(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const p = params orelse return 1;
        const dst: [*]u32 = @ptrFromInt(@as(*const u64, @ptrCast(@alignCast(p[0].?))).*);
        const n = @as(*const u32, @ptrCast(@alignCast(p[1].?))).*;
        const w: *const follow.Words = @ptrCast(@alignCast(p[2].?));
        @memcpy(dst[0..n], w.w[0..n]);
        return 0;
    }
    fn download(out: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        const from: [*]const u8 = @ptrFromInt(src);
        @memcpy(@as([*]u8, @ptrCast(out.?))[0..n], from[0..n]);
        return 0;
    }
    fn device(dev: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        dev.* = 0;
        return 0;
    }
    fn discrete(v: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        v.* = 0;
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn reset() void {
        for (sent.items) |b| gpa.free(b);
        sent.clearAndFree(gpa);
        taken = 0;
    }
    /// Recorded broadcast `i` as rank 0 sent it: a step's 64-byte header and, in the next, the same with its ids.
    fn header(i: usize) follow.Header {
        return std.mem.bytesToValue(follow.Header, sent.items[i][0..64]);
    }
};

/// A rank of two on the Fake engine, its link on host memory: what the leader puts there, the follower downloads.
const Rank = struct {
    f: Fake,
    lib: cuda.nccl.Library,
    mem: []align(256) u8,
    ids: []u32,
    link: follow.Link,

    fn init(r: *Rank, split: bool, comm: cuda.nccl.Comm) !void {
        try r.f.init(split);
        r.f.d.api.cuLaunchKernel = Wire.put;
        r.f.d.api.cuMemcpyDtoH_v2 = Wire.download;
        r.f.d.api.cuCtxGetDevice = Wire.device;
        r.f.d.api.cuDeviceGetAttribute = Wire.discrete;
        r.f.d.api.cuMemsetD8Async = Wire.clear;
        r.f.e.dropped = .empty;
        r.f.e.drafter = null;
        r.f.e.m.glue.put_words = .{ .d = &r.f.d, .handle = null };
        r.lib = .{ .lib = undefined, .api = undefined };
        r.lib.api.ncclBroadcast = Wire.bcast;
        r.mem = try gpa.alignedAlloc(u8, .fromByteUnits(256), 64 + 128 * 4);
        r.ids = try gpa.alloc(u32, 128);
        const d = &r.f.d;
        r.link = .{ .d = d, .lib = &r.lib, .comm = comm, .stream = .{ .d = d, .handle = null }, .buf = .{ .d = d, .ptr = @intFromPtr(r.mem.ptr), .len = r.mem.len }, .ids = r.ids, .gpa = gpa, .wait = false };
    }
    fn deinit(r: *Rank) void {
        r.f.deinit(r.f.e.pages);
        gpa.free(r.mem);
        gpa.free(r.ids);
    }
};

/// Rank 0's pool step, then a begin on slot 1 from owner class `owner`; the ids it sent.
fn lead(r: *Rank, owner: u32) ![]const u32 {
    const p = try r.f.pool(3, 16, true);
    r.f.e.pages = p;
    const x: follow.Leader = .{ .e = &r.f.e, .link = &r.link };
    try x.pool(3, 16, p.page);
    try x.bind(1);
    const ids = try p.openOn(1, 600, 100, owner);
    try x.beginOn(null, ids, 0);
    return ids;
}

test "GLM KV split: the pool step carries the switch and begin the owner, through the link's Header to a follower" {
    for ([_]bool{ true, false }) |split| {
        Wire.reset();
        defer Wire.reset();
        var r0: Rank = undefined;
        var r1: Rank = undefined;
        try r0.init(split, Wire.leader);
        defer r0.deinit();
        try r1.init(split, Wire.other);
        defer r1.deinit();
        const owner: u32 = @intFromBool(split); // off: one class, owner 0
        const ids = try lead(&r0, owner);
        try expectEqual(@as(usize, 2), ids.len);
        // the wire: the pool header, the begin header, then the begin header and its ids
        try expectEqual(@as(usize, 3), Wire.sent.items.len);
        const page = r0.f.e.pages.?.page;
        const pool: follow.Header = .{ .kind = @backingInt(follow.Kind.pool), .depth = 3, .a = 16, .seed = page, .flags = owner };
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&pool), Wire.sent.items[0]); // off: main's bytes
        const begin: follow.Header = .{ .kind = @backingInt(follow.Kind.begin), .n = 2, .depth = owner, .slot = 1 };
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&begin), Wire.sent.items[1]);
        try std.testing.expectEqualSlices(u8, std.mem.asBytes(&begin), Wire.sent.items[2][0..64]);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ids), Wire.sent.items[2][64..]);
        for (ids) |id| try expectEqual(owner, id % @as(u32, if (split) 2 else 1));
        // the follower's own loop decodes them: its pool books by class, and slot 1 maps the ids under their owner
        try pages_mod.ownSequence(&r1.f.e);
        try std.testing.expectError(error.NcclFailed, follow.follow(&r1.f.e, &r1.link));
        try expectEqual(@as(usize, 3), Wire.taken);
        const p1 = r1.f.e.pages.?;
        try expectEqual(@as(u32, if (split) 2 else 1), p1.pool.classes);
        try expect(p1.used[1]);
        try expectEqual(owner, p1.leases[1].owner);
        try std.testing.expectEqualSlices(u32, ids, p1.leases[1].ids.items);
        try std.testing.expectEqualSlices(u32, ids, r1.f.st(1).pages);
    }
}

test "GLM KV split: a begin whose owner on the wire is not its ids' class is refused by the follower, mapping nothing" {
    Wire.reset();
    defer Wire.reset();
    var r0: Rank = undefined;
    var r1: Rank = undefined;
    try r0.init(true, Wire.leader);
    defer r0.deinit();
    try r1.init(true, Wire.other);
    defer r1.deinit();
    _ = try lead(&r0, 1);
    var h = Wire.header(1);
    try expectEqual(@as(u32, 1), h.depth);
    h.depth = 0; // the owner the ids are not of
    @memcpy(Wire.sent.items[1], std.mem.asBytes(&h));
    try pages_mod.ownSequence(&r1.f.e);
    try std.testing.expectError(error.ResidueMismatch, follow.follow(&r1.f.e, &r1.link));
    try expectEqual(@as(usize, 3), Wire.taken); // the begin and its ids were received, then refused
    try expect(!r1.f.e.pages.?.used[1] and r1.f.st(1).capacity == 0);
}

test "GLM KV split: a follower started without the switch refuses rank 0's pool step, and makes no pool" {
    Wire.reset();
    defer Wire.reset();
    var r0: Rank = undefined;
    var r1: Rank = undefined;
    try r0.init(true, Wire.leader);
    defer r0.deinit();
    try r1.init(false, Wire.other);
    defer r1.deinit();
    _ = try lead(&r0, 1);
    try pages_mod.ownSequence(&r1.f.e);
    try std.testing.expectError(error.RankMismatch, follow.follow(&r1.f.e, &r1.link));
    try expectEqual(@as(usize, 1), Wire.taken);
    try expect(r1.f.e.pages == null);
}
