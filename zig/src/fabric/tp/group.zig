//! A `Collective` over a family's NCCL communicator and, once installed, its tp_group (RoCE or NCCL by size).
const std = @import("std");
const cuda = @import("cuda");
const coll = @import("collective.zig");
const nccl_backend = @import("nccl.zig");
const DevicePtr = coll.DevicePtr;
const DType = coll.DType;
const Op = coll.Op;
const Error = coll.Error;

pub const Backend = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: u32,
    world: u32,
    /// The family's group once it is up (serving): gathers then take its size rule. Null: plain NCCL calls.
    group: ?*cuda.tp_group.Group = null,
    /// The group that owns `comm` from the start (aborts go through it); null: the caller owns it and aborts it.
    owner: ?*cuda.tp_group.Group = null,
    aborted: std.atomic.Value(bool) = .init(false),
    /// The underlying error of the calling thread's last failed call (`Error` is a fixed set; the family reads this).
    last: ?anyerror = null,

    pub fn collective(b: *Backend) coll.Collective {
        return .{ .ptr = b, .vtable = &vtable };
    }

    const vtable: coll.Collective.VTable = .{
        .kind = .group,
        .rank = rankOf,
        .world = worldOf,
        .all_reduce = allReduce,
        .all_gather = allGather,
        .exchange = exchange,
        .send = send,
        .recv = recv,
        .barrier = barrier,
        .check = check,
        .abort = abort,
    };

    fn of(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn rankOf(ptr: *anyopaque) u32 {
        return of(ptr).rank;
    }

    fn worldOf(ptr: *anyopaque) u32 {
        return of(ptr).world;
    }

    fn live(b: *const Backend) Error!void {
        if (b.aborted.load(.acquire)) return error.Aborted;
        if (b.group) |g| if (g.aborted.load(.acquire)) return error.Aborted;
    }

    fn failed(b: *Backend, e: anyerror) Error {
        if (b.aborted.load(.acquire)) return error.Aborted;
        b.last = e;
        return error.BackendFailed;
    }

    /// A raw NCCL result: an abort from another thread ends the call as Aborted, before lib.check's fatal stop.
    fn nccl(b: *Backend, res: cuda.nccl.Result, what: []const u8) Error!void {
        if (res == 0) return;
        if (b.aborted.load(.acquire)) return error.Aborted;
        b.lib.check(res, what) catch |e| return b.failed(e);
    }

    fn stream(b: *const Backend, s: coll.Stream) cuda.Stream {
        return .{ .d = b.d, .handle = s };
    }

    /// Rank 0's partial first: under rank_fold the family's kernel adds, never NCCL's own order.
    fn allReduce(ptr: *anyopaque, _: DevicePtr, _: DevicePtr, _: usize, _: DType, _: Op, _: coll.Stream) Error!void {
        try of(ptr).live();
        return error.Unsupported;
    }

    fn allGather(ptr: *anyopaque, s: DevicePtr, r: DevicePtr, count: usize, t: DType, st: coll.Stream) Error!void {
        const b = of(ptr);
        try b.live();
        if (b.group) |g| return g.allGather(b.stream(st), s, r, count * t.size()) catch |e| b.failed(e);
        return b.nccl(b.lib.api.ncclAllGather(s, r, count, nccl_backend.dataType(t), b.comm, st), "ncclAllGather");
    }

    fn exchange(ptr: *anyopaque, _: DevicePtr, _: DevicePtr, _: usize, _: DType, _: u32, _: coll.Stream) Error!void {
        try of(ptr).live();
        return error.Unsupported;
    }

    fn send(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, st: coll.Stream) Error!void {
        const b = of(ptr);
        try b.live();
        try coll.checkPeer(b.rank, b.world, peer);
        if (b.group) |g| return g.send(b.stream(st), buf, count * t.size(), peer) catch |e| b.failed(e);
        return b.nccl(b.lib.api.ncclSend(buf, count, nccl_backend.dataType(t), @intCast(peer), b.comm, st), "ncclSend");
    }

    fn recv(ptr: *anyopaque, buf: DevicePtr, count: usize, t: DType, peer: u32, st: coll.Stream) Error!void {
        const b = of(ptr);
        try b.live();
        try coll.checkPeer(b.rank, b.world, peer);
        if (b.group) |g| return g.recv(b.stream(st), buf, count * t.size(), peer) catch |e| b.failed(e);
        return b.nccl(b.lib.api.ncclRecv(buf, count, nccl_backend.dataType(t), @intCast(peer), b.comm, st), "ncclRecv");
    }

    fn barrier(ptr: *anyopaque) Error!void {
        try of(ptr).live();
        return error.Unsupported;
    }

    /// A failed RoCE gather the group recorded (nothing to check without a group: NCCL reports through its calls).
    fn check(ptr: *anyopaque) Error!void {
        const b = of(ptr);
        try b.live();
        const g = b.group orelse return;
        g.check() catch |e| {
            b.last = e;
            return error.PeerFailed;
        };
    }

    /// Any thread; only the first acts. The owner (or the routing group) aborts; with neither, only later calls stop.
    fn abort(ptr: *anyopaque) void {
        const b = of(ptr);
        if (b.aborted.swap(true, .acq_rel)) return;
        if (b.owner orelse b.group) |g| g.abort();
    }
};

/// The error a family sees for a failed call: the group backend's own underlying error when it kept one, else `e`.
pub fn underlying(c: coll.Collective, e: Error) anyerror {
    if (c.vtable.kind != .group or (e != error.BackendFailed and e != error.PeerFailed)) return e;
    const b: *Backend = @ptrCast(@alignCast(c.ptr));
    return b.last orelse e;
}

const testing = std.testing;

const Fake = struct {
    var gathers: [8]struct { s: DevicePtr, r: DevicePtr, count: usize, t: cuda.nccl.DataType, st: coll.Stream } = undefined;
    var n: usize = 0;
    var multi: usize = 0;
    var wired: struct { s: cuda.abi.Stream, mine: DevicePtr, all: DevicePtr, bytes: usize } = undefined;
    var abort_inside: ?*Backend = null; // a fake NCCL call that another thread aborts while it is pending
    var aborts: usize = 0;
    var destroys: usize = 0;
    var fail = false;
    fn allGather(s: DevicePtr, r: DevicePtr, count: usize, t: cuda.nccl.DataType, _: cuda.nccl.Comm, st: coll.Stream) callconv(.c) cuda.nccl.Result {
        gathers[n] = .{ .s = s, .r = r, .count = count, .t = t, .st = st };
        n += 1;
        if (abort_inside) |b| {
            b.collective().abort();
            return 1;
        }
        return @intFromBool(fail);
    }
    fn commAbort(_: cuda.nccl.Comm) callconv(.c) cuda.nccl.Result {
        aborts += 1;
        return 0;
    }
    fn commDestroy(_: cuda.nccl.Comm) callconv(.c) cuda.nccl.Result {
        destroys += 1;
        return 0;
    }
    fn fits(_: *anyopaque, bytes: usize) bool {
        return bytes <= 4096;
    }
    fn wire(_: *anyopaque, s: cuda.abi.Stream, mine: cuda.abi.DevicePtr, all: cuda.abi.DevicePtr, bytes: usize) anyerror!void {
        wired = .{ .s = s, .mine = mine, .all = all, .bytes = bytes };
        multi += 1;
    }
    fn wireCheck(_: *anyopaque) anyerror!void {
        if (fail) return error.PeerFailed;
    }
    fn none(_: *anyopaque) void {}
    fn reset() void {
        n = 0;
        multi = 0;
        aborts = 0;
        destroys = 0;
        fail = false;
        abort_inside = null;
    }
};

fn fakeLib() cuda.nccl.Library {
    var lib: cuda.nccl.Library = undefined;
    lib.api.ncclAllGather = Fake.allGather;
    lib.api.ncclCommAbort = Fake.commAbort;
    return lib;
}

const test_stream: coll.Stream = @ptrFromInt(0x5);

test "without a group, a gather is NCCL's of the caller's type and count, on the caller's stream" {
    Fake.reset();
    const lib = fakeLib();
    var b: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 1, .world = 2 };
    const c = b.collective();
    try testing.expectEqual(coll.Kind.group, c.kind());
    try c.allGather(0x100, 0x200, 6, .f32, test_stream);
    try testing.expectEqual(@as(usize, 1), Fake.n);
    try testing.expectEqual(@as(usize, 6), Fake.gathers[0].count);
    try testing.expectEqual(cuda.nccl.DataType.f32, Fake.gathers[0].t);
    try testing.expectEqual(@as(DevicePtr, 0x100), Fake.gathers[0].s);
    try testing.expectEqual(@as(DevicePtr, 0x200), Fake.gathers[0].r);
    try testing.expectEqual(test_stream, Fake.gathers[0].st);
}

test "with a group, a gather takes its size rule: the wire when it fits, else NCCL of u8 bytes" {
    Fake.reset();
    var ctx: u8 = 0;
    var g: cuda.tp_group.Group = .{ .lib = fakeLib(), .comm = null, .rank = 0, .world = 3, .multi = .{ .ctx = &ctx, .fits = Fake.fits, .gather = Fake.wire, .check = Fake.wireCheck, .abort = Fake.none, .deinit = Fake.none } };
    const lib = fakeLib();
    var b: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 3, .group = &g };
    const c = b.collective();
    try c.allGather(0x100, 0x200, 1024, .f32, test_stream);
    try testing.expectEqual(@as(usize, 1), Fake.multi);
    try testing.expectEqual(@as(DevicePtr, 0x100), Fake.wired.mine);
    try testing.expectEqual(@as(DevicePtr, 0x200), Fake.wired.all);
    try testing.expectEqual(@as(usize, 1024 * 4), Fake.wired.bytes);
    try testing.expectEqual(test_stream, Fake.wired.s);
    try testing.expectEqual(@as(usize, 0), Fake.n);
    try c.allGather(0x100, 0x200, 1025, .f32, test_stream);
    try testing.expectEqual(@as(usize, 1), Fake.n);
    try testing.expectEqual(@as(usize, 1025 * 4), Fake.gathers[0].count);
    try testing.expectEqual(cuda.nccl.DataType.u8, Fake.gathers[0].t);
    try testing.expectEqual(@as(DevicePtr, 0x100), Fake.gathers[0].s);
    try testing.expectEqual(@as(DevicePtr, 0x200), Fake.gathers[0].r);
    try testing.expectEqual(test_stream, Fake.gathers[0].st);
    Fake.fail = true;
    try testing.expectError(error.PeerFailed, c.check());
    try testing.expectEqual(@as(?anyerror, error.PeerFailed), b.last);
}

test "a failing call keeps its own error; sums, exchanges and barriers are refused" {
    Fake.reset();
    const lib = fakeLib();
    var b: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2 };
    const c = b.collective();
    Fake.fail = true;
    try testing.expectError(error.BackendFailed, c.allGather(0x100, 0x200, 4, .f32, test_stream));
    try testing.expectEqual(@as(?anyerror, error.NcclFailed), b.last);
    try testing.expectError(error.Unsupported, c.allReduce(0x100, 0x200, 4, .f32, .sum, test_stream));
    try testing.expectError(error.Unsupported, c.exchange(0x100, 0x200, 4, .f32, 1, test_stream));
    try testing.expectError(error.Unsupported, c.barrier());
}

test "abort acts once, and every later call is refused without reaching NCCL" {
    Fake.reset();
    const lib = fakeLib();
    var b: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2 };
    const c = b.collective();
    c.abort();
    c.abort();
    try testing.expectEqual(@as(usize, 0), Fake.aborts); // the owner's communicator: the backend never frees it
    try testing.expectError(error.Aborted, c.allGather(0x100, 0x200, 4, .f32, test_stream));
    try testing.expectError(error.Aborted, c.check());
    try testing.expectError(error.Aborted, c.allReduce(0x100, 0x200, 4, .f32, .sum, test_stream));
    try testing.expectError(error.Aborted, c.exchange(0x100, 0x200, 4, .f32, 1, test_stream));
    try testing.expectError(error.Aborted, c.barrier());
    try testing.expectEqual(@as(usize, 0), Fake.n);
    // with a group: the group's abort runs once, and the backend refuses after it
    Fake.reset();
    var g: cuda.tp_group.Group = .{ .lib = fakeLib(), .comm = null, .rank = 0, .world = 2 };
    var bg: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2, .group = &g };
    const cg = bg.collective();
    cg.abort();
    cg.abort();
    try testing.expectEqual(@as(usize, 1), Fake.aborts);
    try testing.expectError(error.Aborted, cg.allGather(0x100, 0x200, 4, .f32, test_stream));
}

test "a call another thread aborts while it is pending ends as Aborted, not as a failure" {
    Fake.reset();
    const lib = fakeLib();
    var b: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2 };
    Fake.abort_inside = &b;
    try testing.expectError(error.Aborted, b.collective().allGather(0x100, 0x200, 4, .f32, test_stream));
    try testing.expectEqual(@as(?anyerror, null), b.last);
    try testing.expectEqual(@as(usize, 0), Fake.aborts);
    // with a group (NCCL's route, no wire): the group aborts its communicator once, and its own call ends as Aborted
    Fake.reset();
    var g: cuda.tp_group.Group = .{ .lib = fakeLib(), .comm = null, .rank = 0, .world = 2 };
    var bg: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2, .group = &g };
    Fake.abort_inside = &bg;
    try testing.expectError(error.Aborted, bg.collective().allGather(0x100, 0x200, 4, .f32, test_stream));
    try testing.expectEqual(@as(?anyerror, null), bg.last);
    try testing.expectEqual(@as(usize, 1), Fake.aborts);
    // the group's own failed call after its abort: Aborted, not NCCL's error (lib.check's fatal stop outside tests)
    try testing.expectError(error.Aborted, g.allGather(.{ .d = undefined, .handle = test_stream }, 0x100, 0x200, 16));
}

test "with an owner and no routing group, abort aborts the owner's communicator once and its destroy then does nothing" {
    Fake.reset();
    const lib = fakeLib();
    var owner: cuda.tp_group.Group = .{ .lib = fakeLib(), .comm = null, .rank = 0, .world = 2 };
    owner.lib.api.ncclCommDestroy = Fake.commDestroy;
    var b: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2, .owner = &owner };
    const c = b.collective();
    c.abort();
    c.abort();
    try testing.expectEqual(@as(usize, 1), Fake.aborts);
    owner.destroyComm();
    try testing.expectEqual(@as(usize, 0), Fake.destroys);
    try testing.expectError(error.Aborted, c.allGather(0x100, 0x200, 4, .f32, test_stream));
    // the routing group installed later is the same group: one abort still
    Fake.reset();
    var g: cuda.tp_group.Group = .{ .lib = fakeLib(), .comm = null, .rank = 0, .world = 2 };
    var bg: Backend = .{ .d = undefined, .lib = &lib, .comm = null, .rank = 0, .world = 2, .owner = &g, .group = &g };
    bg.collective().abort();
    try testing.expectEqual(@as(usize, 1), Fake.aborts);
}
