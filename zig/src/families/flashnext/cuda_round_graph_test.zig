//! The decode round's graph on a counting driver and a faked communicator over a socket pair whose peer can leave.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const nccl = cuda.nccl;
const rg = @import("cuda_round_graph.zig");
const Collective = @import("cuda_collective.zig").Collective;

const Fake = struct {
    var capturing = false;
    var begins: usize = 0;
    var instantiates: usize = 0;
    var launches: usize = 0;
    var execs: usize = 0;
    var graphs: usize = 0;
    var captured_sums: usize = 0;
    var captured_gathers: usize = 0;
    /// What the communicator's abort found: a capture still open, executables and captured graphs still alive.
    var at_abort: ?[3]usize = null;

    fn reset() void {
        capturing = false;
        begins = 0;
        instantiates = 0;
        launches = 0;
        execs = 0;
        graphs = 0;
        captured_sums = 0;
        captured_gathers = 0;
        at_abort = null;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        capturing = true;
        begins += 1;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        graphs += 1;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn status(_: abi.Stream, s: *abi.CaptureStatus) callconv(.c) abi.Result {
        s.* = if (capturing) .active else .none;
        return 0;
    }
    fn instantiate(e: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        instantiates += 1;
        execs += 1;
        e.* = @ptrFromInt(instantiates);
        return 0;
    }
    fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
        graphs -= 1;
        return 0;
    }
    fn upload(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        launches += 1;
        return 0;
    }
    fn execDestroy(_: abi.GraphExec) callconv(.c) abi.Result {
        execs -= 1;
        return 0;
    }
    fn allReduce(_: abi.DevicePtr, _: abi.DevicePtr, _: usize, _: nccl.DataType, _: nccl.RedOp, _: nccl.Comm, _: abi.Stream) callconv(.c) c_int {
        if (capturing) captured_sums += 1;
        return 0;
    }
    fn allGather(_: abi.DevicePtr, _: abi.DevicePtr, _: usize, _: nccl.DataType, _: nccl.Comm, _: abi.Stream) callconv(.c) c_int {
        if (capturing) captured_gathers += 1;
        return 0;
    }
    fn asyncError(_: nccl.Comm, state: *c_int) callconv(.c) c_int {
        state.* = 0;
        return 0;
    }
    fn abort(_: nccl.Comm) callconv(.c) c_int {
        at_abort = .{ @intFromBool(capturing), execs, graphs };
        return 0;
    }
    fn driver() cuda.Driver {
        reset();
        var d: cuda.Driver = undefined;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphDestroy = graphDestroy;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphExecDestroy = execDestroy;
        return d;
    }
};

/// One rank's communicator over one end of a socket pair; closing `pair[1]` is its peer leaving.
fn collective(pair: *[2]c_int) !Collective {
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
    var c: Collective = undefined;
    c.failed = false;
    c.release = null;
    c.comm = @ptrFromInt(0x1000);
    c.parent = null;
    c.link = .{ .fd = pair[0] };
    c.timeout_ms = 1000;
    c.lib.api.ncclAllReduce = Fake.allReduce;
    c.lib.api.ncclAllGather = Fake.allGather;
    c.api.ncclCommGetAsyncError = Fake.asyncError;
    c.api.ncclCommAbort = Fake.abort;
    return c;
}

/// A staged round of one lane whose pass sums over the communicator, or uploads a host row as an unstaged pass would.
const Pass = struct {
    stream: cuda.Stream,
    staged: bool = true,
    uploads: bool = false,
    /// The dense split's layer: a BF16 head gather and an FP32 column gather beside the sum.
    split: bool = false,
    m: struct { collective: ?*Collective },

    pub fn chunk(p: *Pass, _: []const u32, _: usize, _: ?[]u8) !void {
        if (p.uploads) try cuda.copy_source.enqueue(p.stream.d, 0x5000, &.{ 1, 2, 3, 4 }, p.stream);
        if (p.split) {
            try p.m.collective.?.gather(p.stream, 0x7000, 0x8000, 4);
            try p.m.collective.?.gatherF32(p.stream, 0x9000, 0xa000, 4);
        }
        try p.m.collective.?.sum(p.stream, 0x6000, 4);
    }
};

fn key(owner: u64) !rg.Key {
    const Seq = struct { rank_id: u64, fed: usize };
    const q: Seq = .{ .rank_id = owner, .fed = 10 };
    return rg.Key.init(@as([]const *const Seq, &.{&q}), 1, 2, 3);
}

/// The same stream can be captured again: no capture is open and the copy guard was released.
fn expectReleased(stream: cuda.Stream) !void {
    try t.expectEqual(abi.CaptureStatus.none, try cuda.graph.captureStatus(stream));
    try cuda.graph.beginCapture(stream, .thread_local);
    var g = try cuda.graph.endCapture(stream);
    g.deinit();
}

/// Sets a copy-source context for the capture guard and returns the one to restore.
fn enter() abi.Context {
    const previous = cuda.copy_source.current();
    cuda.copy_source.setCurrent(@ptrFromInt(0x3254));
    return previous;
}

test "a decode round's graph: refused unstaged, captured once a key with its sums, replayed, recaptured on a new key" {
    var d = Fake.driver();
    const previous = enter();
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var pair: [2]c_int = undefined;
    var col = try collective(&pair);
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var c: rg.Cache = .{};
    var p: Pass = .{ .stream = stream, .staged = false, .m = .{ .collective = &col } };
    try t.expectError(error.UnexpectedTensor, c.launch(try key(1), &p, &.{7}));
    try t.expectEqual(@as(usize, 0), Fake.begins);
    p.staged = true;
    try c.launch(try key(1), &p, &.{7});
    try t.expectEqual([4]usize{ 1, 1, 1, 1 }, [4]usize{ Fake.begins, Fake.instantiates, Fake.launches, Fake.captured_sums });
    try t.expect(col.release != null);
    try c.launch(try key(1), &p, &.{8});
    try t.expectEqual([4]usize{ 1, 1, 2, 1 }, [4]usize{ Fake.begins, Fake.instantiates, Fake.launches, Fake.captured_sums });
    try c.launch(try key(2), &p, &.{9});
    try t.expectEqual([4]usize{ 2, 2, 3, 2 }, [4]usize{ Fake.begins, Fake.instantiates, Fake.launches, Fake.captured_sums });
    try t.expectEqual(@as(usize, 1), Fake.execs);
    c.clear();
    try t.expectEqual([2]usize{ 0, 0 }, [2]usize{ Fake.execs, Fake.graphs });
    try t.expect(col.release == null);
    try t.expectEqual([2]u64{ 2, 1 }, [2]u64{ c.captures, c.replays });
}

test "a dense-split round's gathers join its capture, and a clear between rounds recaptures each one" {
    var d = Fake.driver();
    const previous = enter();
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var pair: [2]c_int = undefined;
    var col = try collective(&pair);
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var c: rg.Cache = .{};
    defer c.clear();
    var p: Pass = .{ .stream = stream, .split = true, .m = .{ .collective = &col } };
    for (0..3) |_| try c.launch(try key(1), &p, &.{7});
    try t.expectEqual([3]usize{ 1, 2, 1 }, [3]usize{ Fake.begins, Fake.captured_gathers, Fake.captured_sums });
    try t.expectEqual([2]u64{ 1, 2 }, [2]u64{ c.captures, c.replays });
    // a drafted window's verify clears the graph ahead of the plain rounds in the same step
    for (0..3) |_| {
        c.clear();
        try c.launch(try key(1), &p, &.{7});
    }
    try t.expectEqual([3]usize{ 4, 8, 4 }, [3]usize{ Fake.begins, Fake.captured_gathers, Fake.captured_sums });
    try t.expectEqual([2]u64{ 4, 2 }, [2]u64{ c.captures, c.replays });
}

test "a decode round whose pass uploads inside its capture is refused; its stream and copy guard are released" {
    var d = Fake.driver();
    const previous = enter();
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var pair: [2]c_int = undefined;
    var col = try collective(&pair);
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var c: rg.Cache = .{};
    defer c.clear();
    var p: Pass = .{ .stream = stream, .uploads = true, .m = .{ .collective = &col } };
    try t.expectError(error.UnownedCaptureSource, c.launch(try key(1), &p, &.{7}));
    try t.expectEqual([2]usize{ 0, 0 }, [2]usize{ Fake.instantiates, Fake.graphs });
    try expectReleased(stream);
}

test "a peer lost after a replayed round: the communicator's abort finds the round's executable destroyed" {
    var d = Fake.driver();
    const previous = enter();
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var pair: [2]c_int = undefined;
    var col = try collective(&pair);
    defer _ = std.c.close(pair[0]);
    var c: rg.Cache = .{};
    defer c.clear();
    var p: Pass = .{ .stream = stream, .m = .{ .collective = &col } };
    try c.launch(try key(1), &p, &.{7});
    try c.launch(try key(1), &p, &.{8});
    _ = std.c.close(pair[1]);
    try t.expectError(error.RankDisconnected, col.finish(stream));
    try t.expectEqual([3]usize{ 0, 0, 0 }, Fake.at_abort.?);
    try t.expect(c.exec == null and col.release == null);
}

test "a peer lost during a round's capture: the capture is ended and its graph freed before the abort" {
    var d = Fake.driver();
    const previous = enter();
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var pair: [2]c_int = undefined;
    var col = try collective(&pair);
    defer _ = std.c.close(pair[0]);
    var c: rg.Cache = .{};
    defer c.clear();
    var p: Pass = .{ .stream = stream, .m = .{ .collective = &col } };
    _ = std.c.close(pair[1]);
    try t.expectError(error.RankDisconnected, c.launch(try key(1), &p, &.{7}));
    try t.expectEqual([3]usize{ 0, 0, 0 }, Fake.at_abort.?);
    try t.expectEqual(@as(usize, 0), Fake.instantiates);
    try expectReleased(stream);
}

test "control: a communicator without the release finds the round's executable alive at its abort" {
    var d = Fake.driver();
    const previous = enter();
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var pair: [2]c_int = undefined;
    var col = try collective(&pair);
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var c: rg.Cache = .{};
    var p: Pass = .{ .stream = stream, .m = .{ .collective = &col } };
    try c.launch(try key(1), &p, &.{7});
    col.release = null;
    col.abort();
    try t.expectEqual([3]usize{ 0, 1, 0 }, Fake.at_abort.?);
    c.clear();
}
