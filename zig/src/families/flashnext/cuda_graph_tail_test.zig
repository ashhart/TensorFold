//! TF_FLASHNEXT_GRAPH_TAIL on the recording device: the tail's calls in today's order, the same token, graph replays.
const std = @import("std");
const cuda = @import("cuda");
const abi = cuda.abi;
const gt = @import("cuda_graph_tail.zig");
const wg = @import("cuda_window_graph.zig");
const parts = @import("cuda_batch_state.zig");
const mtp = @import("cuda_mtp.zig");
const draw = @import("cuda_mtp_draw.zig");
const f = @import("cuda_mtp_nonfinite_fixture.zig");
const vocab = @import("cuda_prompt_wt.zig").vocab;
const D = f.Device;
const t = std.testing;
const a = t.allocator;

/// Everything the recording device saw: copies (dst, src, bytes), launches (function, grid), events, best sources.
const Seen = struct {
    copies: [8][3]u64 = @splat(@splat(0)),
    launches: [8][2]u64 = @splat(@splat(0)),
    events: [8]D.Event = undefined,
    n: [3]usize = @splat(0),
    sources: [2]u64 = @splat(0),

    fn now() Seen {
        var s: Seen = .{ .n = .{ D.copies, D.launches, D.events_n } };
        for (0..D.copies) |i| s.copies[i] = .{ D.copy_dst[i], D.copy_src[i], D.copy_bytes[i] };
        for (0..D.launches) |i| s.launches[i] = .{ D.functions[i], D.grids[i] };
        @memcpy(s.events[0..D.events_n], D.events[0..D.events_n]);
        for (0..@min(D.best_calls, 2)) |i| s.sources[i] = D.sources[i];
        return s;
    }

    fn expectSame(want: Seen, got: Seen) !void {
        try t.expectEqual(want.n, got.n);
        try t.expectEqualSlices([3]u64, want.copies[0..want.n[0]], got.copies[0..got.n[0]]);
        try t.expectEqualSlices([2]u64, want.launches[0..want.n[1]], got.launches[0..got.n[1]]);
        try t.expectEqualSlices(D.Event, want.events[0..want.n[2]], got.events[0..got.n[2]]);
        try t.expectEqual(want.sources, got.sources);
    }
};

const before: gt.Copy = .{ .dst = 0x5000, .src = 0x6000, .n = 640 };
const after: gt.Copy = .{ .dst = 0x7000, .src = 0x8800, .n = 320 };

fn copyNow(st: cuda.Stream, c: gt.Copy) !void {
    try st.d.check(st.d.api.cuMemcpyDtoDAsync_v2(c.dst, c.src, c.n, st.handle), "cuMemcpyDtoDAsync");
}

test "graph tail: a lone greedy verify's copies and row best keep today's order, and its draw the same token" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const host = try a.alloc(u16, vocab);
    defer a.free(host);
    var d = f.draft(&driver, host);
    D.best = f.reduction(3.5, 4, 7);
    const st: cuda.Stream = .{ .d = &driver, .handle = null };
    // switch off: the copy before the pass, the copy after it, then draws' own row best and wait
    try copyNow(st, before);
    try copyNow(st, after);
    var want: [1]u32 = undefined;
    try draw.draws(&h, st, a, &d, null, &.{}, &want);
    const eager = Seen.now();
    try t.expectEqual(@as(usize, 1), D.best_calls);
    // switch on: the verify's tail queues the same calls, its draws only waits and reads
    D.reset();
    try t.expect(draw.bestFirst(null, &.{}));
    const tail: gt.Tail = .{ .before = before, .after = .{ .dst = after.dst, .n = after.n }, .head = &h, .scores = d.logits.ptr, .vocab = vocab, .rows = d.rows, .best = d.best.ptr };
    try tail.first(st);
    try tail.last(st, after.src);
    d.best_queued = true;
    var got: [1]u32 = undefined;
    try draw.draws(&h, st, a, &d, null, &.{}, &got);
    try Seen.expectSame(eager, Seen.now());
    try t.expectEqual(want, got);
    try t.expect(!d.best_queued);
    // a masked or sampled draw reduces otherwise, so no verify queues its best
    try t.expect(!draw.bestFirst(null, &.{1}));
    try t.expect(!draw.bestFirst(.{ .seed = 3, .temperature = 0.7, .top_k = 20, .top_p = 0.95 }, &.{}));
}

const Graph = struct {
    var capturing = false;
    var records: usize = 0;
    var replays: usize = 0;
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
        e.* = @ptrFromInt(2);
        return 0;
    }
    fn ok(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn upload(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        replays += 1;
        return 0;
    }
    fn destroy(_: abi.GraphExec) callconv(.c) abi.Result {
        return 0;
    }
};

const Scratch = struct { ple_nrow: u64 };

/// A verify pass on the recording device: one kernel (function 9) between the tail's two halves.
const Pass = struct {
    hooks: parts.Hooks,
    stream: cuda.Stream,
    s: *const Scratch,

    pub fn chunk(p: *Pass, _: []const u32, _: usize, _: anytype) !void {
        var args: cuda.Args = .{}; // four ints: the fixture reads any other launch's fourth as top-k
        for (0..4) |_| args.add(@as(c_int, 0));
        try cuda.launch.launch(.{ .d = p.stream.d, .handle = @ptrFromInt(9) }, .{ .grid = .{ .x = 3 }, .block = .{ .x = 32 } }, p.stream, &args);
    }
};

test "graph tail: the window graph records the eager pass's calls once, then replays with none" {
    var driver = D.driver();
    defer cuda.download_stage.closeDriver(&driver);
    driver.api.cuStreamBeginCapture_v2 = Graph.begin;
    driver.api.cuStreamEndCapture = Graph.end;
    driver.api.cuStreamIsCapturing = Graph.status;
    driver.api.cuGraphInstantiateWithFlags = Graph.instantiate;
    driver.api.cuGraphDestroy = Graph.ok;
    driver.api.cuGraphUpload = Graph.upload;
    driver.api.cuGraphLaunch = Graph.launch;
    driver.api.cuGraphExecDestroy = Graph.destroy;
    const previous = cuda.copy_source.current();
    cuda.copy_source.setCurrent(@ptrFromInt(0x3254));
    defer cuda.copy_source.setCurrent(previous);
    var h = try f.head(&driver);
    defer h.artifact.deinit();
    const st: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(1) };
    const tail: gt.Tail = .{ .before = before, .after = .{ .dst = after.dst, .n = after.n }, .head = &h, .scores = 0x4000, .vocab = vocab, .rows = 1, .best = 0x1000 };
    const scratch: Scratch = .{ .ple_nrow = after.src };
    var p: Pass = .{ .hooks = .{ .logits = .all, .tail = &tail }, .stream = st, .s = &scratch };
    var c: wg.Cache(wg.Key) = .{};
    defer c.clear(a);
    const k: wg.Key = .{ .rows = 1, .shape = 7, .moving = 1 };
    const chunk = wg.Chunk(Pass){ .p = &p, .ids = &.{5}, .pos0 = 0 };
    // first run: eager, the old order (copy, pass, copy, row best)
    try c.run(a, st, k, chunk);
    const eager = Seen.now();
    try t.expectEqualSlices([2]u64, &.{ .{ 9, 3 }, .{ 1, 1 } }, eager.launches[0..2]);
    try t.expectEqualSlices([3]u64, &.{ .{ before.dst, before.src, before.n }, .{ after.dst, after.src, after.n } }, eager.copies[0..2]);
    // second: captured, recording the same calls in the same order
    D.reset();
    try c.run(a, st, k, chunk);
    try Seen.expectSame(eager, Seen.now());
    try t.expectEqual(@as(usize, 1), Graph.records);
    // steady state: replays only, no host copies or launches
    D.reset();
    for (0..5) |_| try c.run(a, st, k, chunk);
    try t.expectEqual([2]usize{ 0, 0 }, [2]usize{ D.copies, D.launches });
    try t.expectEqual([3]u64{ 1, 0, 5 }, [3]u64{ c.captures, c.updates, c.replays });
    try t.expectEqual(@as(usize, 6), Graph.replays);
}
