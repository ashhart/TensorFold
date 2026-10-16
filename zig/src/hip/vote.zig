//! A captured step's vote: each rank's word on its recording, gathered in buffers kept from startup.

const std = @import("std");
const abi = @import("abi.zig");
const collective = @import("collective.zig");
const graph_split = @import("graph_split.zig");
const memory = @import("memory.zig");
const raw = @import("raw.zig");
const runtime = @import("runtime.zig");
const Stream = @import("stream.zig").Stream;

/// Every rank's vote on a captured step, gathered in buffers made at startup: voting allocates nothing.
pub const Votes = struct {
    host: memory.HostBuffer,
    send: memory.DeviceBuffer,
    recv: memory.DeviceBuffer,
    world: usize,

    pub fn init(d: *const runtime.Runtime, world: usize) runtime.Error!Votes {
        var host = try memory.HostBuffer.alloc(d, 8 * (1 + world));
        errdefer host.free();
        var send = try memory.DeviceBuffer.alloc(d, 8);
        errdefer send.free();
        return .{ .host = host, .send = send, .recv = try memory.DeviceBuffer.alloc(d, 8 * world), .world = world };
    }

    pub fn deinit(v: *Votes) void {
        v.recv.free();
        v.send.free();
        v.host.free();
    }

    /// Device bytes the votes of a group of `world` take (none for one rank).
    pub fn bytes(world: usize) usize {
        return if (world > 1) 8 + 8 * world else 0;
    }

    /// Every rank's vote, `mine` among them: the step's outcome on every rank.
    pub fn agree(v: *Votes, comm: collective.Comm, mine: collective.Vote, stream: Stream) !collective.Vote {
        v.host.slice(u64)[0] = @backingInt(mine);
        try raw.upload(v.send, 0, v.host.bytes[0..8], stream.handle);
        try comm.allGather(v.send.base(), v.recv.base(), 1, .i64, stream.handle);
        try raw.download(v.recv, 0, v.host.bytes[8 .. 8 * (1 + v.world)], stream.handle);
        try stream.synchronize();
        return collective.tally(v.host.slice(u64)[1 .. 1 + v.world]);
    }
};

/// What recording a step left on this rank: its vote and, when it kept the step, the pieces and the step's outputs.
pub fn Recorded(comptime Out: type) type {
    return struct { vote: collective.Vote, split: ?graph_split.Split = null, out: Out = undefined };
}

/// `body(ctx)` recorded, nothing run, `tp.*` swapped meanwhile: kept, declined (a cut, `decline`) or failed (OOM).
pub fn record(
    comptime Out: type,
    gpa: std.mem.Allocator,
    stream: Stream,
    tp: *?collective.Comm,
    decline: bool,
    ctx: anytype,
    comptime body: fn (@TypeOf(ctx)) anyerror!Out,
) Recorded(Out) {
    var rec: graph_split.Recorder = .{ .gpa = gpa, .stream = stream, .inner = tp.* };
    rec.begin() catch return .{ .vote = .declined };
    const inner = tp.*;
    tp.* = rec.comm();
    const got = body(ctx);
    tp.* = inner;
    var split = (rec.finish() catch null) orelse return .{ .vote = .declined };
    const out = got catch |err| {
        split.deinit();
        if (err != error.OutOfDeviceMemory) return .{ .vote = .declined };
        std.log.warn("a round could not be recorded: {s}", .{@errorName(err)});
        return .{ .vote = .failed };
    };
    if (decline) {
        split.deinit();
        return .{ .vote = .declined };
    }
    return .{ .vote = .kept, .split = split, .out = out };
}

/// A runtime whose graph calls succeed unless told to fail, and which counts what it frees and allocates.
const Mock = struct {
    var fail_begin = false;
    var fail_instantiate = false;
    var freed: usize = 0;
    var allocs: usize = 0;

    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        return @intFromBool(fail_begin);
    }
    fn end(_: abi.Stream, g: *abi.Graph) callconv(.c) abi.Result {
        g.* = @ptrFromInt(0x1000);
        return 0;
    }
    fn instantiate(e: *abi.GraphExec, _: abi.Graph, _: c_ulonglong) callconv(.c) abi.Result {
        e.* = @ptrFromInt(0x2000);
        return @intFromBool(fail_instantiate);
    }
    fn upload(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn destroyGraph(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn destroyExec(_: abi.GraphExec) callconv(.c) abi.Result {
        freed += 1;
        return 0;
    }
    fn hostMalloc(p: *abi.DevicePtr, n: usize, _: c_uint) callconv(.c) abi.Result {
        allocs += 1;
        p.* = std.c.malloc(n) orelse return 1;
        return 0;
    }
    fn malloc(p: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
        return hostMalloc(p, n, 0);
    }
    fn free(p: abi.DevicePtr) callconv(.c) abi.Result {
        std.c.free(p);
        return 0;
    }
    fn htod(dst: abi.DevicePtr, src: [*]const u8, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], src[0..n]);
        return 0;
    }
    fn dtoh(dst: [*]u8, src: abi.DevicePtr, n: usize, _: abi.Stream) callconv(.c) abi.Result {
        @memcpy(dst[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }

    fn make() runtime.Runtime {
        var r: runtime.Runtime = undefined;
        r.api.hipStreamBeginCapture = begin;
        r.api.hipStreamEndCapture = end;
        r.api.hipGraphInstantiateWithFlags = instantiate;
        r.api.hipGraphUpload = upload;
        r.api.hipGraphDestroy = destroyGraph;
        r.api.hipGraphExecDestroy = destroyExec;
        r.api.hipHostMalloc = hostMalloc;
        r.api.hipHostFree = free;
        r.api.hipMalloc = malloc;
        r.api.hipFree = free;
        r.api.hipMemcpyHtoDAsync = htod;
        r.api.hipMemcpyDtoHAsync = dtoh;
        r.api.hipStreamSynchronize = sync;
        fail_begin = false;
        fail_instantiate = false;
        freed = 0;
        allocs = 0;
        return r;
    }
};

/// A collective whose every call goes through (one rank's view of the others: `others` are their votes).
const Peers = struct {
    others: []const u64 = &.{},

    const D = collective.Dtype;

    fn gather(p: *anyopaque, send: u64, recv: u64, _: usize, _: D, _: abi.Stream) collective.Error!void {
        const f: *Peers = @ptrCast(@alignCast(p));
        const out: [*]u64 = @ptrFromInt(recv);
        out[0] = @as(*const u64, @ptrFromInt(send)).*;
        for (f.others, 1..) |v, i| out[i] = v;
    }
    fn reduce(_: *anyopaque, _: u64, _: u64, _: usize, _: D, _: abi.Stream) collective.Error!void {}
    fn cast(_: *anyopaque, _: u64, _: u64, _: usize, _: D, _: usize, _: abi.Stream) collective.Error!void {}
    fn close(_: *anyopaque) void {}

    const vtable: collective.Comm.VTable = .{
        .all_reduce = reduce,
        .all_gather = gather,
        .broadcast = cast,
        .close = close,
    };

    fn comm(f: *Peers) collective.Comm {
        return .{ .ptr = f, .vtable = &vtable, .rank = 0, .world = 1 + f.others.len };
    }
};

/// A step's body for the tests: one collective through the communicator it is handed, then `result`.
const Step = struct {
    tp: *?collective.Comm,
    result: anyerror!u8,

    fn run(s: *const Step) anyerror!u8 {
        if (s.tp.*) |c| try c.allReduce(0, 0, 1, .f32, null);
        return s.result;
    }
};

test "a recorded step's vote: kept, declined by a cut, an error or on purpose, failed out of device memory" {
    var r = Mock.make();
    const stream: Stream = .{ .r = &r, .handle = null };
    var peers: Peers = .{};
    var tp: ?collective.Comm = peers.comm();
    const gpa = std.testing.allocator;
    var kept = record(u8, gpa, stream, &tp, false, &Step{ .tp = &tp, .result = 7 }, Step.run);
    try std.testing.expectEqual(collective.Vote.kept, kept.vote);
    try std.testing.expectEqual(@as(u8, 7), kept.out);
    try std.testing.expect(tp != null and tp.?.ptr == @as(*anyopaque, &peers));
    kept.split.?.deinit();
    const freed = Mock.freed;
    const oom = record(u8, gpa, stream, &tp, false, &Step{ .tp = &tp, .result = error.OutOfDeviceMemory }, Step.run);
    try std.testing.expectEqual(collective.Vote.failed, oom.vote);
    try std.testing.expect(oom.split == null and Mock.freed == freed + 2);
    const other = record(u8, gpa, stream, &tp, false, &Step{ .tp = &tp, .result = error.Invalid }, Step.run);
    try std.testing.expectEqual(collective.Vote.declined, other.vote);
    const asked = record(u8, gpa, stream, &tp, true, &Step{ .tp = &tp, .result = 7 }, Step.run);
    try std.testing.expectEqual(collective.Vote.declined, asked.vote);
    Mock.fail_instantiate = true;
    const cut = record(u8, gpa, stream, &tp, false, &Step{ .tp = &tp, .result = 7 }, Step.run);
    try std.testing.expectEqual(collective.Vote.declined, cut.vote);
    Mock.fail_instantiate = false;
    Mock.fail_begin = true;
    const unbegun = record(u8, gpa, stream, &tp, false, &Step{ .tp = &tp, .result = 7 }, Step.run);
    try std.testing.expectEqual(collective.Vote.declined, unbegun.vote);
    try std.testing.expect(tp.?.ptr == @as(*anyopaque, &peers));
}

test "the votes are gathered in buffers made at startup: one rank's failure is every rank's, and no allocation" {
    var r = Mock.make();
    const stream: Stream = .{ .r = &r, .handle = null };
    var v = try Votes.init(&r, 3);
    defer v.deinit();
    const made = Mock.allocs;
    var peers: Peers = .{ .others = &.{ 1, 1 } };
    try std.testing.expectEqual(collective.Vote.kept, try v.agree(peers.comm(), .kept, stream));
    try std.testing.expectEqual(collective.Vote.declined, try v.agree(peers.comm(), .declined, stream));
    peers.others = &.{ 0, 2 };
    try std.testing.expectEqual(collective.Vote.failed, try v.agree(peers.comm(), .kept, stream));
    try std.testing.expectEqual(made, Mock.allocs);
    try std.testing.expectEqual(@as(usize, 32), Votes.bytes(3));
    try std.testing.expectEqual(@as(usize, 0), Votes.bytes(1));
}
