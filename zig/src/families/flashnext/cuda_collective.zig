//! One two-rank NCCL owner: a nonblocking communicator bounds start-up; its blocking split carries the collectives.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const env = @import("sys").env;
const wire = @import("cuda_rank_wire.zig");
const nccl = cuda.nccl;

const Advanced = struct {
    ncclCommInitRankConfig: *const fn (*nccl.Comm, c_int, nccl.UniqueId, c_int, *nccl.Config) callconv(.c) nccl.Result,
    ncclCommSplit: *const fn (nccl.Comm, c_int, c_int, *nccl.Comm, *nccl.Config) callconv(.c) nccl.Result,
    ncclCommAbort: *const fn (nccl.Comm) callconv(.c) nccl.Result,
    ncclCommGetAsyncError: *const fn (nccl.Comm, *nccl.Result) callconv(.c) nccl.Result,
    ncclReduceScatter: @FieldType(nccl.Api, "ncclAllReduce"),
};

pub const Collective = struct {
    lib: nccl.Library,
    api: Advanced,
    comm: nccl.Comm = null, // the blocking split every collective uses once init returns
    parent: nccl.Comm = null, // the nonblocking start-up communicator, whose connections `comm` shares
    link: wire.Link,
    timeout_ms: u64,
    rank: usize = 0,
    failed: bool = false,

    /// `link` has completed the source/settings handshake; it stays owned by the caller.
    pub fn init(link: wire.Link, rank: usize, timeout_ms: u64) !Collective {
        if (rank > 1 or timeout_ms == 0) return error.BadRank;
        // every connection is made inside the bounded init and split, none by a collective; an operator's value stands
        _ = env.setenv("NCCL_RUNTIME_CONNECT", "0", 0);
        var c: Collective = .{ .lib = try nccl.Library.open(), .api = undefined, .link = link, .timeout_ms = timeout_ms, .rank = rank };
        errdefer c.lib.close();
        try c.lib.requireSupportedVersion();
        const info = @typeInfo(Advanced).@"struct";
        inline for (info.field_names, info.field_types) |name, T| @field(c.api, name) = c.lib.lib.lookup(T, name) orelse return error.MissingSymbol;
        var diag = cuda.p2p_report.begin(rank, 2);
        defer diag.end();
        var id: nccl.UniqueId = undefined;
        const until = wire.deadline(timeout_ms);
        if (rank == 0) {
            try c.lib.check(c.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id), until);
        } else try link.receive(std.mem.asBytes(&id), until);
        errdefer c.abort();
        try c.connect(id, until);
        cuda.p2p_report.afterInit(&diag, .{ .lib = &c.lib, .comm = c.comm }); // no driver here: no probe
        return c;
    }

    /// The parent shares its connections, so its split, which the collectives use, connects nothing of its own.
    fn connect(c: *Collective, id: nccl.UniqueId, until: u64) !void {
        var config: nccl.Config = .{ .split_share = 1 };
        try c.submitted(c.api.ncclCommInitRankConfig(&c.comm, 2, id, @intCast(c.rank), &config));
        try c.wait(null, until);
        var child: nccl.Comm = null;
        errdefer {
            c.abort(); // the parent first: it stops the split and joins it, after which NCCL has stored the child
            if (@atomicLoad(nccl.Comm, &child, .acquire)) |comm| _ = c.api.ncclCommAbort(comm);
        }
        var blocking: nccl.Config = .{ .blocking = 1 };
        try c.submitted(c.api.ncclCommSplit(c.comm, 0, @intCast(c.rank), &child, &blocking));
        try c.wait(null, wire.deadline(c.timeout_ms)); // the split runs in NCCL's thread and reports through its parent
        const made = @atomicLoad(nccl.Comm, &child, .acquire) orelse return error.NcclFailed;
        c.parent = c.comm;
        c.comm = made;
    }

    /// The parent first, which stops a split in flight, then the communicator the collectives use.
    pub fn abort(c: *Collective) void {
        c.failed = true;
        inline for (.{ "parent", "comm" }) |field| if (@field(c, field)) |comm| {
            _ = c.api.ncclCommAbort(comm);
            @field(c, field) = null;
        };
    }

    pub fn deinit(c: *Collective) void {
        // Abort is local and also releases an idle communicator if its peer exited first.
        c.abort();
        c.lib.close();
    }

    fn submitted(c: *Collective, result: nccl.Result) !void {
        if (result == 0 or result == 7) return;
        return c.lib.check(result, "NCCL nonblocking operation");
    }

    fn wait(c: *Collective, stream: ?cuda.Stream, until: u64) !void {
        errdefer c.abort();
        while (true) {
            try c.link.checkPeer();
            if (wire.now() >= until) return error.RankTimeout;
            var state: nccl.Result = 0;
            try c.lib.check(c.api.ncclCommGetAsyncError(c.comm, &state), "ncclCommGetAsyncError");
            if (state != 0 and state != 7) return c.lib.check(state, "NCCL asynchronous operation");
            if (state == 0) {
                if (stream) |s| {
                    if (try s.done()) return;
                } else return;
            }
            wire.pause();
        }
    }

    /// Wait at a host-visible boundary while retaining peer, asynchronous-error and timeout checks.
    pub fn finish(c: *Collective, stream: cuda.Stream) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        try c.wait(stream, wire.deadline(c.timeout_ms));
    }

    pub fn benchmarkShape(c: *Collective) !nccl.Shape {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        const actual = try c.lib.shape(c.comm);
        if (actual.world != 2) return error.BadRank;
        return actual;
    }

    pub fn benchmarkGather(c: *Collective, comptime T: type, d: *const cuda.Driver, stream: cuda.Stream, mine: T, out: []T) !void {
        const shape = try c.benchmarkShape();
        if (out.len != shape.world or mine.rank != shape.rank) return error.BadRank;
        const size = @sizeOf(T);
        var dev = try cuda.DeviceBuffer.alloc(d, size * out.len);
        defer dev.free();
        errdefer c.abort();
        try dev.uploadAsync(shape.rank * size, std.mem.asBytes(&mine), stream.handle);
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllGather(dev.ptr + shape.rank * size, dev.ptr, size, .u8, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
        try c.finish(stream);
        try dev.download(0, std.mem.sliceAsBytes(out));
    }

    /// BF16 rank blocks are concatenated without an arithmetic reduction.
    pub fn gather(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize) !void {
        return c.gatherType(stream, local, out, count, .bf16);
    }

    /// FP32 rank blocks are concatenated without rounding or reduction.
    pub fn gatherF32(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize) !void {
        return c.gatherType(stream, local, out, count, .f32);
    }

    fn gatherType(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize, dtype: nccl.DataType) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllGather(local, out, count, dtype, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    pub fn broadcast(c: *Collective, stream: cuda.Stream, values: u64, bytes: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclBroadcast(values, values, bytes, .u8, 0, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    /// FP32 sum is enqueued on the compute stream; later kernels on that stream observe its result.
    pub fn sum(c: *Collective, stream: cuda.Stream, values: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllReduce(values, values, count, .f32, .sum, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    /// sum's FP32 sum of block `rank` of `values` (blocks of `count`) alone, left in place there: a reduce-scatter.
    pub fn sumBlock(c: *Collective, stream: cuda.Stream, values: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.api.ncclReduceScatter(values, values + c.rank * count * 4, count, .f32, .sum, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }
};

// NCCL's jobs as the tests see them: the parent's init and split end at a scripted poll or when the parent is aborted.
const Fake = struct {
    const parent: nccl.Comm = @ptrFromInt(0x1000);
    const child: nccl.Comm = @ptrFromInt(0x2000);
    var states: []const nccl.Result = &.{};
    var polls: usize = 0;
    var split_result: nccl.Result = 0;
    var pending: ?*nccl.Comm = null;
    var aborted: [4]nccl.Comm = @splat(null);
    var aborts: usize = 0;
    var parent_config: nccl.Config = .{};
    var child_config: nccl.Config = .{};
    var color: c_int = -1;
    var key: c_int = -1;
    var leaves: c_int = -1; // a peer link that closes as the split starts

    fn initRank(comm: *nccl.Comm, _: c_int, _: nccl.UniqueId, _: c_int, config: *nccl.Config) callconv(.c) nccl.Result {
        parent_config = config.*;
        comm.* = parent;
        return nccl.in_progress;
    }
    fn split(_: nccl.Comm, c: c_int, k: c_int, out: *nccl.Comm, config: *nccl.Config) callconv(.c) nccl.Result {
        color = c;
        key = k;
        child_config = config.*;
        out.* = null;
        if (leaves >= 0) _ = std.c.shutdown(leaves, std.posix.SHUT.RDWR);
        if (split_result == 0 or split_result == nccl.in_progress) pending = out;
        return split_result;
    }
    /// The split's job stores the child as it exits, whether it succeeded, failed or was stopped.
    fn exit() void {
        if (pending) |out| out.* = child;
        pending = null;
    }
    fn asyncError(comm: nccl.Comm, state: *nccl.Result) callconv(.c) nccl.Result {
        state.* = 0;
        if (comm != parent) return 0;
        state.* = states[@min(polls, states.len - 1)];
        polls += 1;
        if (state.* != nccl.in_progress) exit();
        return 0;
    }
    fn abort(comm: nccl.Comm) callconv(.c) nccl.Result {
        if (comm == parent) exit();
        aborted[aborts] = comm;
        aborts += 1;
        return 0;
    }
    fn reset(script: []const nccl.Result, result: nccl.Result) void {
        states = script;
        polls = 0;
        split_result = result;
        pending = null;
        aborted = @splat(null);
        aborts = 0;
        color = -1;
        key = -1;
        leaves = -1;
    }
    fn expectAborts(want: []const nccl.Comm) !void {
        try std.testing.expectEqualSlices(nccl.Comm, want, aborted[0..aborts]);
    }
};

/// Rank 1 over one end of a socket pair; the other end is its peer's control link.
fn rig(pair: *[2]c_int, timeout_ms: u64) !Collective {
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
    const api: Advanced = .{ .ncclCommInitRankConfig = Fake.initRank, .ncclCommSplit = Fake.split, .ncclCommAbort = Fake.abort, .ncclCommGetAsyncError = Fake.asyncError, .ncclReduceScatter = undefined };
    return .{ .lib = undefined, .api = api, .link = .{ .fd = pair[0] }, .timeout_ms = timeout_ms, .rank = 1 };
}

fn closePair(pair: [2]c_int) void {
    for (pair) |fd| _ = std.c.close(fd);
}

test "a nonblocking parent that shares its connections hands the collectives to a blocking split" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    for ([_]nccl.Result{ 0, nccl.in_progress }) |returned| {
        var pair: [2]c_int = undefined;
        var c = try rig(&pair, 1000);
        defer closePair(pair);
        Fake.reset(&.{ nccl.in_progress, 0, nccl.in_progress, nccl.in_progress, 0 }, returned);
        try c.connect(undefined, wire.deadline(1000));
        try std.testing.expect(Fake.parent_config.blocking == 0 and Fake.parent_config.split_share == 1);
        try std.testing.expect(Fake.child_config.blocking == 1 and Fake.color == 0 and Fake.key == 1);
        try std.testing.expect(c.comm == Fake.child and c.parent == Fake.parent and !c.failed);
        try std.testing.expectEqual(@as(usize, 5), Fake.polls);
        c.abort();
        c.abort();
        try Fake.expectAborts(&.{ Fake.parent, Fake.child });
        try std.testing.expect(c.failed and c.comm == null and c.parent == null);
    }
}

test "a split that fails through its parent's asynchronous error aborts the parent, then the child NCCL returned" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var pair: [2]c_int = undefined;
    var c = try rig(&pair, 1000);
    defer closePair(pair);
    Fake.reset(&.{ 0, nccl.in_progress, 6 }, 0); // ncclRemoteError
    try std.testing.expectError(error.NcclFailed, c.connect(undefined, wire.deadline(1000)));
    try Fake.expectAborts(&.{ Fake.parent, Fake.child });
    try std.testing.expect(c.failed and c.comm == null and c.parent == null);
}

test "a split still running at its deadline is stopped through the parent within the bound" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var pair: [2]c_int = undefined;
    var c = try rig(&pair, 30);
    defer closePair(pair);
    Fake.reset(&.{ 0, nccl.in_progress }, 0);
    const start = wire.now();
    try std.testing.expectError(error.RankTimeout, c.connect(undefined, wire.deadline(1000)));
    try std.testing.expect(wire.now() - start < std.time.ns_per_s);
    try Fake.expectAborts(&.{ Fake.parent, Fake.child });
    try std.testing.expect(c.failed and c.comm == null and c.parent == null);
}

test "a peer that closes its link during the split surfaces as disconnected, both communicators aborted" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var pair: [2]c_int = undefined;
    var c = try rig(&pair, 1000);
    defer closePair(pair);
    Fake.reset(&.{ 0, nccl.in_progress }, 0);
    Fake.leaves = pair[1];
    try std.testing.expectError(error.RankDisconnected, c.connect(undefined, wire.deadline(1000)));
    try Fake.expectAborts(&.{ Fake.parent, Fake.child });
    try std.testing.expect(c.failed and c.comm == null and c.parent == null);
}

test "a split NCCL refuses outright aborts the parent and leaves no child" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    var pair: [2]c_int = undefined;
    var c = try rig(&pair, 1000);
    defer closePair(pair);
    Fake.reset(&.{0}, 4); // ncclInvalidArgument
    try std.testing.expectError(error.NcclFailed, c.connect(undefined, wire.deadline(1000)));
    try Fake.expectAborts(&.{Fake.parent});
    try std.testing.expect(c.failed and c.comm == null and c.parent == null);
}

test "this file calls NCCL only to start, poll and abort, and for collectives NCCL connected at init" {
    // a send or receive would connect lazily on the blocking split, where no deadline reaches it
    const source = @embedFile("cuda_collective.zig");
    try std.testing.expect(std.mem.indexOf(u8, source, "lib." ++ "group(") == null);
    const allowed = [_][]const u8{
        "ncclGetUniqueId",       "ncclCommInitRankConfig", "ncclCommSplit", "ncclCommAbort",
        "ncclCommGetAsyncError", "ncclAllGather",          "ncclBroadcast", "ncclAllReduce",
        "ncclReduceScatter",
    };
    inline for (@typeInfo(Advanced).@"struct".field_names) |name| {
        var known = false;
        for (allowed) |a| known = known or std.mem.eql(u8, a, name);
        try std.testing.expect(known);
    }
    const needle = "api" ++ ".";
    var at: usize = 0;
    var calls: usize = 0;
    while (std.mem.indexOfPos(u8, source, at, needle)) |i| : (calls += 1) {
        const name = source[i + needle.len ..][0 .. std.mem.indexOfScalar(u8, source[i + needle.len ..], '(') orelse 0];
        var known = false;
        for (allowed) |a| known = known or std.mem.eql(u8, a, name);
        if (!known) std.debug.print("unexpected NCCL entry point {s}\n", .{name});
        try std.testing.expect(known);
        at = i + needle.len;
    }
    try std.testing.expectEqual(@as(usize, 11), calls);
}
