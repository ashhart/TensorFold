//! cuda.keyed_graph's cache on a driver that counts calls: replays, updates in place, new executables on new launches.
const std = @import("std");
const t = std.testing;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const copy_source = @import("copy_source.zig");
const kg = @import("keyed_graph.zig");

const Fake = struct {
    var capturing = false;
    var records: usize = 0;
    var instantiates: usize = 0;
    var updates: usize = 0;
    var launches: usize = 0;
    var eager: usize = 0;
    var destroyed: usize = 0;
    var verdict: abi.ExecUpdateResult = .success;
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        capturing = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        capturing = false;
        records += 1;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn status(_: abi.Stream, s: *abi.CaptureStatus) callconv(.c) abi.Result {
        s.* = if (capturing) .active else .none;
        return 0;
    }
    fn instantiate(e: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        instantiates += 1;
        e.* = @ptrFromInt(instantiates);
        return 0;
    }
    fn update(_: abi.GraphExec, _: abi.Graph, info: *abi.ExecUpdateResultInfo) callconv(.c) abi.Result {
        updates += 1;
        info.result = verdict;
        return if (verdict == .success) 0 else 910; // CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE
    }
    fn graphDestroy(_: abi.Graph) callconv(.c) abi.Result {
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
        destroyed += 1;
        return 0;
    }
};

/// A pass whose kernels the fake counts: run outside a capture, it ran eagerly.
const Pass = struct {
    pub fn run(_: Pass) !void {
        if (!Fake.capturing) Fake.eager += 1;
    }
};

const Key = struct { rows: u32, shape: u64, moving: u64 };
const Cache = kg.Cache(Key, 8);

/// A pass of `rows` rows whose kernels launch as `shape` says, at the host values `moving` hashes.
fn key(moving: u64, rows: u32, shape: u64) Key {
    return .{ .rows = rows, .shape = shape, .moving = moving };
}

/// After one more run: these counts of recordings, instantiations, updates, eager runs and graph launches.
fn expectAfter(c: *Cache, stream: Stream, k: Key, counts: [5]usize) !void {
    try c.run(t.allocator, stream, k, Pass{});
    try t.expectEqual(counts, [5]usize{ Fake.records, Fake.instantiates, Fake.updates, Fake.eager, Fake.launches });
}

test "keyed graphs: a held graph takes new values in place; new launches get a new executable" {
    var d: Driver = undefined;
    d.api.cuStreamBeginCapture_v2 = Fake.begin;
    d.api.cuStreamEndCapture = Fake.end;
    d.api.cuStreamIsCapturing = Fake.status;
    d.api.cuGraphInstantiateWithFlags = Fake.instantiate;
    d.api.cuGraphExecUpdate_v2 = Fake.update;
    d.api.cuGraphDestroy = Fake.graphDestroy;
    d.api.cuGraphUpload = Fake.upload;
    d.api.cuGraphLaunch = Fake.launch;
    d.api.cuGraphExecDestroy = Fake.execDestroy;
    const previous = copy_source.current();
    copy_source.setCurrent(@ptrFromInt(0x3254));
    defer copy_source.setCurrent(previous);
    const stream: Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var c: Cache = .{};

    try expectAfter(&c, stream, key(1, 3, 7), .{ 0, 0, 0, 1, 0 }); // a new slot runs eagerly once
    try expectAfter(&c, stream, key(1, 3, 7), .{ 1, 1, 0, 1, 1 }); // then is captured
    try expectAfter(&c, stream, key(1, 3, 7), .{ 1, 1, 0, 1, 2 }); // and replayed
    try expectAfter(&c, stream, key(2, 3, 7), .{ 2, 1, 1, 1, 3 }); // new values (a length, other buffers): updated
    try expectAfter(&c, stream, key(3, 3, 7), .{ 3, 1, 2, 1, 4 });
    try expectAfter(&c, stream, key(3, 3, 7), .{ 3, 1, 2, 1, 5 });
    Fake.verdict = .topology_changed;
    try expectAfter(&c, stream, key(4, 3, 7), .{ 4, 2, 3, 1, 6 }); // other launches: a new executable, the old freed
    try t.expectEqual(@as(usize, 1), Fake.destroyed);
    Fake.verdict = .success;
    try expectAfter(&c, stream, key(4, 3, 7), .{ 4, 2, 3, 1, 7 });
    try expectAfter(&c, stream, key(4, 4, 7), .{ 4, 2, 3, 2, 7 }); // another row count is another slot
    try expectAfter(&c, stream, key(4, 3, 8), .{ 4, 2, 3, 3, 7 }); // and another shape
    try t.expectEqual([3]u64{ 1, 3, 3 }, [3]u64{ c.captures, c.updates, c.replays });
    c.clear(t.allocator);
    try t.expectEqual(@as(usize, 2), Fake.destroyed);
}
