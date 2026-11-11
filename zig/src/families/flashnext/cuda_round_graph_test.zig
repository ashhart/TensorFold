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
    m: struct { collective: ?*Collective, pocket: ?*anyopaque = null },

    pub fn chunk(p: *Pass, _: []const u32, _: usize, _: ?[]u8) !void {
        if (p.uploads) try cuda.copy_source.enqueue(p.stream.d, 0x5000, &.{ 1, 2, 3, 4 }, p.stream);
        if (p.split) {
            try p.m.collective.?.gather(p.stream, 0x7000, 0x8000, 4);
            try p.m.collective.?.gatherF32(p.stream, 0x9000, 0xa000, 4);
        }
        try p.m.collective.?.sum(p.stream, 0x6000, 4);
    }
};

const Buf = struct { ptr: u64 };
const linear_n = @import("cuda_prompt_wt.zig").linear_n;
const attn_n = @import("cuda_prompt_wt.zig").attn_n;

/// A sequence's owner, length and the buffer addresses a round's lane takes from it.
const Seq = struct {
    rank_id: u64,
    fed: usize = 10,
    cap: usize = 1024,
    conv: [linear_n]Buf = undefined,
    state: [linear_n]Buf = undefined,
    kc: [attn_n]Buf = undefined,
    vc: [attn_n]Buf = undefined,
    ikc: [attn_n]Buf = undefined,
    pooled: [attn_n]Buf = undefined,
    tail: Buf = undefined,
    pg: @import("cuda_glue.zig").Pg = .{},

    /// `owner`'s sequence whose buffers start at `base`, each the next 64 KiB.
    fn at(owner: u64, base: u64) Seq {
        var q: Seq = .{ .rank_id = owner };
        var a = base;
        inline for (.{ "conv", "state", "kc", "vc", "ikc", "pooled" }) |f| {
            for (&@field(q, f)) |*b| {
                b.ptr = a;
                a += 0x10000;
            }
        }
        q.tail.ptr = a;
        return q;
    }

    /// A drafted window settled whole (Draft.settle): its states and the window's spares change places.
    fn settle(q: *Seq, spare: *[linear_n]Buf) void {
        for (&q.state, spare) |*a, *b| std.mem.swap(Buf, a, b);
    }
};

fn spares(base: u64) [linear_n]Buf {
    var s: [linear_n]Buf = undefined;
    for (&s, 0..) |*b, i| b.ptr = base + i * 0x10000;
    return s;
}

fn keyOf(seqs: []const *const Seq) !rg.Key {
    return rg.Key.init(seqs, 1, 2, 3);
}

fn key(owner: u64) !rg.Key {
    const q: Seq = .at(owner, owner << 32);
    return keyOf(&.{&q});
}

test "decode graph keys: owner order, attention and score tile seams, each lane's buffers and length" {
    var first: Seq = .at(1, 0x1000_0000);
    var second: Seq = .at(2, 0x2000_0000);
    const base = try keyOf(&.{ &first, &second });
    try t.expectEqual(base, try keyOf(&.{ &first, &second }));
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &second, &first })));
    const alone = try keyOf(&.{&first});
    var spare = spares(0x3000_0000);
    second.settle(&spare);
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &first, &second })));
    // another stream's settled window leaves a round without it on its graph
    try t.expectEqual(alone, try keyOf(&.{&first}));
    second.settle(&spare);
    try t.expectEqual(base, try keyOf(&.{ &first, &second }));
    second.tail.ptr += 0x100;
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &first, &second })));
    second.tail.ptr -= 0x100;
    second.cap = 2048;
    try t.expect(!std.meta.eql(base, try keyOf(&.{ &first, &second })));
    first.fed = 511;
    const before = try keyOf(&.{&first});
    first.fed = 512;
    try t.expect(!std.meta.eql(before, try keyOf(&.{&first})));
    first.fed = 2048;
    const dense = try keyOf(&.{&first});
    first.fed = 2051;
    try t.expect(!std.meta.eql(dense, try keyOf(&.{&first})));
    first.fed = 2303;
    const score = try keyOf(&.{&first});
    first.fed = 2304;
    try t.expect(!std.meta.eql(score, try keyOf(&.{&first})));
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
    // a sequence made or dropped clears the graph: the round after it is captured again with its gathers
    for (0..3) |_| {
        c.clear();
        try c.launch(try key(1), &p, &.{7});
    }
    try t.expectEqual([3]usize{ 4, 8, 4 }, [3]usize{ Fake.begins, Fake.captured_gathers, Fake.captured_sums });
    try t.expectEqual([2]u64{ 4, 2 }, [2]u64{ c.captures, c.replays });
}

test "mixed drafted and plain streams: verifies between plain rounds keep their graph; a settled lane is captured anew" {
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
    const plain: Seq = .at(1, 0x1000_0000);
    var drafted: Seq = .at(2, 0x2000_0000);
    var spare = spares(0x3000_0000);
    // each step: the plain stream's round, then the drafted stream's window verified and kept whole
    for (0..4) |_| {
        try c.launch(try keyOf(&.{&plain}), &p, &.{7});
        drafted.settle(&spare);
    }
    try t.expectEqual([2]u64{ 1, 3 }, [2]u64{ c.captures, c.replays });
    // a round that holds the drafted lane: its states moved after each settle, so the graph is recorded again
    for (0..3) |_| {
        try c.launch(try keyOf(&.{ &plain, &drafted }), &p, &.{ 7, 8 });
        try c.launch(try keyOf(&.{ &plain, &drafted }), &p, &.{ 7, 8 });
        drafted.settle(&spare);
    }
    try t.expectEqual([2]u64{ 4, 6 }, [2]u64{ c.captures, c.replays });
    try t.expectEqual([2]usize{ 4, 4 }, [2]usize{ Fake.begins, Fake.captured_sums });
    try t.expectEqual(@as(usize, 1), Fake.execs);
}

test "a decode round whose pass adds a pocket is refused before any capture" {
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
    var p: Pass = .{ .stream = stream, .m = .{ .collective = &col, .pocket = @ptrFromInt(0x10) } };
    try t.expectError(error.UnexpectedTensor, c.launch(try key(1), &p, &.{7}));
    try t.expectEqual([2]usize{ 0, 0 }, [2]usize{ Fake.begins, Fake.launches });
    try t.expect(c.exec == null);
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
