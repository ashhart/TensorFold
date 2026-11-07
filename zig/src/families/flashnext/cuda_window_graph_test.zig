//! The window graphs' cache on a driver that counts calls: replays, updates in place, new executables on new launches.
const std = @import("std");
const t = std.testing;
const cuda = @import("cuda");
const abi = cuda.abi;
const wg = @import("cuda_window_graph.zig");
const parts = @import("cuda_batch_state.zig");

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

/// A window of `rows` rows whose kernels launch as `shape` says, at the host values `moving` hashes.
fn key(moving: u64, rows: u32, shape: u64) wg.Key {
    return .{ .rows = rows, .shape = shape, .moving = moving };
}

/// After one more run: these counts of recordings, instantiations, updates, eager runs and graph launches.
fn expectAfter(c: *wg.Cache(wg.Key), stream: cuda.Stream, k: wg.Key, counts: [5]usize) !void {
    try c.run(t.allocator, stream, k, Pass{});
    try t.expectEqual(counts, [5]usize{ Fake.records, Fake.instantiates, Fake.updates, Fake.eager, Fake.launches });
}

test "window graphs: a held graph takes a new length or sequence in place; new launches get a new executable" {
    var d: cuda.Driver = undefined;
    d.api.cuStreamBeginCapture_v2 = Fake.begin;
    d.api.cuStreamEndCapture = Fake.end;
    d.api.cuStreamIsCapturing = Fake.status;
    d.api.cuGraphInstantiateWithFlags = Fake.instantiate;
    d.api.cuGraphExecUpdate_v2 = Fake.update;
    d.api.cuGraphDestroy = Fake.graphDestroy;
    d.api.cuGraphUpload = Fake.upload;
    d.api.cuGraphLaunch = Fake.launch;
    d.api.cuGraphExecDestroy = Fake.execDestroy;
    const previous = cuda.copy_source.current();
    cuda.copy_source.setCurrent(@ptrFromInt(0x3254));
    defer cuda.copy_source.setCurrent(previous);
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    var c: wg.Cache(wg.Key) = .{};

    try expectAfter(&c, stream, key(1, 3, 7), .{ 0, 0, 0, 1, 0 }); // a new slot runs eagerly once
    try expectAfter(&c, stream, key(1, 3, 7), .{ 1, 1, 0, 1, 1 }); // then is captured
    try expectAfter(&c, stream, key(1, 3, 7), .{ 1, 1, 0, 1, 2 }); // and replayed
    try expectAfter(&c, stream, key(2, 3, 7), .{ 2, 1, 1, 1, 3 }); // new values (a length, a request, a lane): updated
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

const Buf = struct { ptr: u64 };
const Seq = struct { gen: u64, cap: usize = 262144, kc: [1]Buf, state: [1]Buf };

fn windowKey(seq: Seq, pos0: usize, rows: usize, hooks: parts.Hooks) wg.Key {
    var s: parts.Scratch = undefined;
    s.h = 0x3000;
    s.cap = 64;
    return .init(seq, pos0, rows, &s, 0x5000, 0x4000, hooks);
}

test "window graph keys: a length, a request or a lane moves in place; the score grid and outputs are slots" {
    const lane0: Seq = .{ .gen = 1, .kc = .{.{ .ptr = 0x10000 }}, .state = .{.{ .ptr = 0x20000 }} };
    const lane1: Seq = .{ .gen = 2, .kc = .{.{ .ptr = 0x30000 }}, .state = .{.{ .ptr = 0x40000 }} };
    const hooks: parts.Hooks = .{ .logits = .all, .logits_out = 0x6000, .host_logits = false };
    const base = windowKey(lane0, 9000, 3, hooks);
    const same = [_]wg.Key{
        windowKey(lane0, 9000 + 256, 3, hooks), // 256 tokens on: another score grid size
        windowKey(lane1, 9000, 3, hooks), // the next request, on another lane's buffers
        windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x7000, .host_logits = false }),
    };
    for (same) |k| {
        try t.expect(!std.meta.eql(k, base));
        try t.expectEqual(wg.slotOf(base), wg.slotOf(k));
    }
    try t.expectEqual(base, windowKey(lane0, 9000, 3, hooks));
    const other = [_]wg.Key{
        windowKey(lane0, 100, 3, hooks), // no score grid yet: other kernels
        windowKey(lane0, 9000, 4, hooks),
        windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x6000, .streams = 0x8000, .host_logits = false }),
    };
    for (other) |k| try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(k)));
    var kept: parts.Keep = .{};
    kept.layers[0].proj = 0x9000;
    const with_keep = windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x6000, .keep = &kept, .host_logits = false });
    try t.expect(!std.meta.eql(wg.slotOf(base), wg.slotOf(with_keep)));
    kept.layers[0].proj = 0xa000;
    try t.expectEqual(wg.slotOf(with_keep), wg.slotOf(windowKey(lane0, 9000, 3, .{ .logits = .all, .logits_out = 0x6000, .keep = &kept, .host_logits = false })));
}
