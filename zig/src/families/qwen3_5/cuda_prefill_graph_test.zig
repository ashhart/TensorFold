//! Actual graph wrapper ABI under deterministic host-only failure and completion controls.
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("cuda_prefill_graph.zig");
const abi = cuda.abi;
const t = std.testing;

const Stage = enum { none, begin, body, end, instantiate, upload, launch };
const Fake = struct {
    fail: Stage = .none,
    active: bool = false,
    pending: bool = false,
    captures: usize = 0,
    bodies: usize = 0,
    eager: usize = 0,
    state_updates: usize = 0,
    uploads: usize = 0,
    launches: usize = 0,
    waits: usize = 0,
    graphs_freed: usize = 0,
    execs_live: usize = 0,
    execs_freed: usize = 0,
    rows: u32 = 7680,
    prefix: u32 = 0,
    captured_rows: u32 = 0,
    captured_prefix: u32 = 0,
    var current: *Fake = undefined;

    fn driver(f: *Fake) cuda.Driver {
        current = f;
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuStreamIsCapturing = status;
        d.api.cuGraphDestroy = destroyGraph;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphExecDestroy = destroyExec;
        d.api.cuStreamSynchronize = wait;
        return d;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        const f = current;
        f.captures += 1;
        if (f.fail == .begin) return abi.error_not_ready;
        f.active = true;
        return 0;
    }
    fn end(_: abi.Stream, out: *abi.Graph) callconv(.c) abi.Result {
        const f = current;
        f.active = false;
        if (f.fail == .end) {
            out.* = null;
            return abi.error_not_ready;
        }
        out.* = @ptrFromInt(0x1000);
        return 0;
    }
    fn status(_: abi.Stream, out: *abi.CaptureStatus) callconv(.c) abi.Result {
        out.* = if (current.active) .active else .none;
        return 0;
    }
    fn instantiate(out: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
        if (current.fail == .instantiate) return 800;
        out.* = @ptrFromInt(0x2000);
        current.execs_live += 1;
        return 0;
    }
    fn upload(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        current.uploads += 1;
        return if (current.fail == .upload) abi.error_not_ready else 0;
    }
    fn launch(_: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
        current.launches += 1;
        current.state_updates += 1;
        current.pending = true;
        return if (current.fail == .launch) abi.error_not_ready else 0;
    }
    fn wait(_: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(!current.active);
        current.waits += 1;
        current.pending = false;
        return 0;
    }
    fn destroyGraph(_: abi.Graph) callconv(.c) abi.Result {
        current.graphs_freed += 1;
        return 0;
    }
    fn destroyExec(_: abi.GraphExec) callconv(.c) abi.Result {
        std.debug.assert(!current.pending and current.execs_live > 0);
        current.execs_live -= 1;
        current.execs_freed += 1;
        return 0;
    }
    fn body(p: *anyopaque, _: cuda.Stream) !void {
        const f: *Fake = @ptrCast(@alignCast(p));
        f.bodies += 1;
        if (f.active) {
            f.captured_rows = f.rows;
            f.captured_prefix = f.prefix;
        } else {
            f.eager += 1;
            f.state_updates += 1;
        }
        if (f.fail == .body) return error.BodyFailed;
    }
    fn call(f: *Fake) graph.Body {
        return .{ .ctx = f, .run = body };
    }
};

test "prompt graph: strict opt-in and every incompatible mode is refused before capture" {
    try t.expect(!try graph.selected(null));
    try t.expect(!try graph.selected("0"));
    try t.expect(try graph.selected("1"));
    for ([_][]const u8{ "", "01", "yes", "true", "2", " 1" }) |bad|
        try t.expectError(error.BadPrefillGraph, graph.selected(bad));
    try t.expect((graph.Facts{}).admits());
    inline for (.{ "dense27", "one_segment", "prompt", "prepared" }) |field| {
        var f: graph.Facts = .{};
        @field(f, field) = false;
        try t.expect(!f.admits());
    }
    inline for (.{ "learning", "pockets", "images", "int8", "vendor", "profiling", "hooks" }) |field| {
        var f: graph.Facts = .{};
        @field(f, field) = true;
        try t.expect(!f.admits());
    }
}

test "prompt graph: disabled and refused calls execute exactly once without graph calls" {
    for ([_]bool{ false, true }) |on| {
        var f: Fake = .{};
        var d = f.driver();
        const s: cuda.Stream = .{ .d = &d, .handle = null };
        try t.expect((try graph.execute(on, .{ .profiling = on }, s, f.call())) == null);
        try t.expectEqual(@as(usize, 1), f.eager);
        try t.expectEqual(@as(usize, 1), f.state_updates);
        try t.expectEqual(@as(usize, 0), f.captures);
    }
}

test "prompt graph: per-call bindings remain full-width and executable survives until completion" {
    var f: Fake = .{};
    var d = f.driver();
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    for ([_]u32{ 0, 8192 }) |prefix| {
        f.prefix = prefix;
        var exec = (try graph.execute(true, .{}, s, f.call())).?;
        try t.expectEqual(@as(u32, 7680), f.captured_rows);
        try t.expectEqual(prefix, f.captured_prefix);
        try t.expect(f.pending and f.execs_live == 1);
        try t.expectEqual(@as(usize, 0), f.eager);
        try s.synchronize();
        exec.deinit();
        try t.expectEqual(@as(usize, 0), f.execs_live);
    }
    try t.expectEqual(@as(usize, 2), f.captures);
    try t.expectEqual(@as(usize, 2), f.state_updates);
    try t.expectEqual(@as(usize, 2), f.execs_freed);
}

test "prompt graph: each failure drains before releasing and never retries a partly executed state" {
    for ([_]Stage{ .begin, .body, .end, .instantiate, .upload, .launch }) |stage| {
        var f: Fake = .{ .fail = stage };
        var d = f.driver();
        const s: cuda.Stream = .{ .d = &d, .handle = null };
        const wanted = if (stage == .body) error.BodyFailed else if (stage == .instantiate) error.GraphNotPermitted else error.NotReady;
        try t.expectError(wanted, graph.execute(true, .{}, s, f.call()));
        try t.expect(!f.active and !f.pending);
        try t.expectEqual(@as(usize, 1), f.waits);
        try t.expectEqual(@as(usize, 0), f.execs_live);
        try t.expectEqual(@as(usize, 0), f.eager);
        try t.expectEqual(@as(usize, if (stage == .launch) 1 else 0), f.state_updates);
    }
}
