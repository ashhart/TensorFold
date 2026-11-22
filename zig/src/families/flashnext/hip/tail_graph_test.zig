//! The real HIP graph boundary on a recording driver: argument and copy bytes, ordering, refresh and refusal.
const std = @import("std");
const t = std.testing;
const hip = @import("hip");
const Tail = @import("tail_graph.zig").Tail;
const A = hip.abi;
const Op = struct {
    kind: enum { upload, kernel },
    addresses: [3]u64 = @splat(0),
    geometry: [7]u32 = @splat(0),
    scalars: [2]u32 = @splat(0),
    bytes: [16]u8 = @splat(0),
};
const Plan = struct { ops: [16]Op = undefined, n: usize = 0 };
const Mock = struct {
    var capture = false;
    var pending: Plan = .{};
    var graphs: [64]Plan = undefined;
    var execs: [64]Plan = undefined;
    var graph_n: usize = 0;
    var exec_n: usize = 0;
    var actual: Plan = .{};
    var begin_fail = false;
    var end_fail = false;
    var install_fail = false;
    var launch_fail = false;
    var refuse_update = false;
    var updates: usize = 0;
    var full_waits: usize = 0;
    var event_waits: usize = 0;
    var records: usize = 0;
    var record_fail = false;
    fn reset() void {
        capture = false;
        pending = .{};
        actual = .{};
        graph_n = 0;
        exec_n = 0;
        begin_fail = false;
        end_fail = false;
        install_fail = false;
        launch_fail = false;
        refuse_update = false;
        updates = 0;
        full_waits = 0;
        event_waits = 0;
        records = 0;
        record_fail = false;
    }
    fn execute(op: Op) void {
        var got = op;
        if (op.kind == .upload) {
            const source: [*]const u8 = @ptrFromInt(op.addresses[1]);
            @memcpy(got.bytes[0..op.addresses[2]], source[0..op.addresses[2]]);
        }
        actual.ops[actual.n] = got;
        actual.n += 1;
    }
    fn issue(op: Op) void {
        if (capture) {
            pending.ops[pending.n] = op;
            pending.n += 1;
        } else execute(op);
    }
    fn upload(dst: A.DevicePtr, src: [*]const u8, n: usize, _: A.Stream) callconv(.c) A.Result {
        issue(.{ .kind = .upload, .addresses = .{ @intFromPtr(dst), @intFromPtr(src), n } });
        return 0;
    }
    fn kernel(f: A.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: A.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) A.Result {
        const p: *const u64 = @ptrCast(@alignCast(args.?[0].?));
        const pos: *const i32 = @ptrCast(@alignCast(args.?[1].?));
        const scale: *const f32 = @ptrCast(@alignCast(args.?[2].?));
        issue(.{ .kind = .kernel, .addresses = .{ @intFromPtr(f), p.*, @intFromPtr(s) }, .geometry = .{ gx, gy, gz, bx, by, bz, shared }, .scalars = .{ @bitCast(pos.*), @bitCast(scale.*) } });
        return 0;
    }
    fn begin(_: A.Stream, _: A.CaptureMode) callconv(.c) A.Result {
        if (begin_fail) return 1;
        capture = true;
        pending = .{};
        return 0;
    }
    fn end(_: A.Stream, out: *A.Graph) callconv(.c) A.Result {
        capture = false;
        if (end_fail) {
            out.* = null;
            return 1;
        }
        graphs[graph_n] = pending;
        out.* = @ptrFromInt(100 + graph_n);
        graph_n += 1;
        return 0;
    }
    fn instantiate(out: *A.GraphExec, g: A.Graph, _: c_ulonglong) callconv(.c) A.Result {
        if (install_fail) return 1;
        execs[exec_n] = graphs[@intFromPtr(g) - 100];
        out.* = @ptrFromInt(300 + exec_n);
        exec_n += 1;
        return 0;
    }
    fn graphUpload(_: A.GraphExec, _: A.Stream) callconv(.c) A.Result {
        return 0;
    }
    fn launch(e: A.GraphExec, _: A.Stream) callconv(.c) A.Result {
        if (launch_fail) return 1;
        const plan = execs[@intFromPtr(e) - 300];
        for (plan.ops[0..plan.n]) |op| execute(op);
        return 0;
    }
    fn update(e: A.GraphExec, g: A.Graph, _: *A.GraphNode, result: *A.ExecUpdateResult) callconv(.c) A.Result {
        updates += 1;
        result.* = if (refuse_update) .topology_changed else .success;
        if (!refuse_update) execs[@intFromPtr(e) - 300] = graphs[@intFromPtr(g) - 100];
        return 0;
    }
    fn dropGraph(_: A.Graph) callconv(.c) A.Result {
        return 0;
    }
    fn dropExec(_: A.GraphExec) callconv(.c) A.Result {
        return 0;
    }
    fn event(out: *A.Event, flags: c_uint) callconv(.c) A.Result {
        std.debug.assert(flags == A.event_disable_timing);
        out.* = @ptrFromInt(900);
        return 0;
    }
    fn dropEvent(_: A.Event) callconv(.c) A.Result {
        return 0;
    }
    fn eventRecord(_: A.Event, _: A.Stream) callconv(.c) A.Result {
        records += 1;
        return @intFromBool(record_fail);
    }
    fn fullWait(_: A.Stream) callconv(.c) A.Result {
        full_waits += 1;
        return 0;
    }
    fn eventWait(_: A.Event) callconv(.c) A.Result {
        event_waits += 1;
        return 0;
    }
    fn runtime() hip.Runtime {
        var r: hip.Runtime = undefined;
        r.api.hipMemcpyHtoDAsync = upload;
        r.api.hipModuleLaunchKernel = kernel;
        r.api.hipStreamBeginCapture = begin;
        r.api.hipStreamEndCapture = end;
        r.api.hipGraphInstantiateWithFlags = instantiate;
        r.api.hipGraphUpload = graphUpload;
        r.api.hipGraphLaunch = launch;
        r.api.hipGraphExecUpdate = update;
        r.api.hipGraphDestroy = dropGraph;
        r.api.hipGraphExecDestroy = dropExec;
        r.api.hipEventCreateWithFlags = event;
        r.api.hipEventDestroy = dropEvent;
        r.api.hipEventRecord = eventRecord;
        r.api.hipEventSynchronize = eventWait;
        r.api.hipStreamSynchronize = fullWait;
        return r;
    }
};
const Ctx = struct {
    s: hip.Stream,
    payload: []const u8,
    position: i32,
    target: u64,
    state: *usize,
    fail: bool = false,
    pub fn save(c: @This()) usize {
        return c.state.*;
    }
    pub fn restore(c: @This(), value: usize) void {
        c.state.* = value;
    }
    fn body(c: @This()) anyerror!u32 {
        c.state.* += 1;
        try hip.runtime.check(c.s.r.api.hipMemcpyHtoDAsync(@ptrFromInt(c.target), c.payload.ptr, c.payload.len, c.s.handle));
        if (c.fail) return error.PlantedBodyFailure;
        var args: hip.launch.Args = .{};
        try args.add(c.target);
        try args.add(c.position);
        try args.add(@as(f32, 0.75));
        const f: hip.Function = .{ .r = c.s.r, .handle = @ptrFromInt(7) };
        try hip.launch.launch(f, .{ .grid = .{ .x = 3, .y = 2 }, .block = .{ .x = 64 }, .shared = 96 }, c.s, &args);
        return @intCast(c.position);
    }
};

test "HIP tail: exact eager sequence and bytes equal first capture and refreshed graph" {
    Mock.reset();
    const r = Mock.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = @ptrFromInt(8) };
    var tail = try Tail.init(t.allocator, &r);
    defer tail.deinit();
    var payload: [8]u8 = .{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var state: usize = 0;
    var ctx: Ctx = .{ .s = s, .payload = &payload, .position = 4, .target = 4096, .state = &state };
    for (0..3) |i| {
        ctx.position += 1;
        ctx.target += 256;
        payload[0] = @intCast(10 + i);
        Mock.actual.n = 0;
        state = 0;
        const want = try Ctx.body(ctx);
        const expected = Mock.actual;
        Mock.actual.n = 0;
        state = 0;
        try t.expectEqual(want, try tail.submit(u32, .{ .phase = .head, .rows = 1 }, s, ctx, Ctx.body));
        try t.expectEqual(@as(usize, 1), state);
        try t.expectEqual(expected.n, Mock.actual.n);
        try t.expectEqualDeep(expected.ops[0..expected.n], Mock.actual.ops[0..Mock.actual.n]);
    }
    try t.expectEqual(@as(usize, 2), Mock.updates);
    try t.expectEqual(@as(usize, 1), Mock.exec_n);
}

test "HIP tail: refused topology installs current arguments without launching stale nodes" {
    Mock.reset();
    const r = Mock.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = null };
    var tail = try Tail.init(t.allocator, &r);
    defer tail.deinit();
    var state: usize = 0;
    var ctx: Ctx = .{ .s = s, .payload = &.{3}, .position = 1, .target = 4096, .state = &state };
    _ = try tail.submit(u32, .{ .phase = .head, .rows = 1 }, s, ctx, Ctx.body);
    Mock.refuse_update = true;
    Mock.actual.n = 0;
    ctx.position = 99;
    ctx.target = 8192;
    _ = try tail.submit(u32, .{ .phase = .head, .rows = 1 }, s, ctx, Ctx.body);
    try t.expectEqual(@as(usize, 2), Mock.exec_n);
    try t.expectEqual(@as(u32, 99), Mock.actual.ops[1].scalars[0]);
    try t.expectEqual(@as(u64, 8192), Mock.actual.ops[1].addresses[1]);
}

test "HIP tail: capture failures restore host state before one eager fallback" {
    for (0..3) |which| {
        Mock.reset();
        const r = Mock.runtime();
        const s: hip.Stream = .{ .r = &r, .handle = null };
        var tail = try Tail.init(t.allocator, &r);
        defer tail.deinit();
        Mock.begin_fail = which == 0;
        Mock.end_fail = which == 1;
        Mock.install_fail = which == 2;
        var state: usize = 0;
        const ctx: Ctx = .{ .s = s, .payload = &.{4}, .position = 7, .target = 4096, .state = &state };
        _ = try tail.submit(u32, .{ .phase = .head, .rows = 1 }, s, ctx, Ctx.body);
        try t.expectEqual(@as(usize, 1), state);
        try t.expectEqual(@as(usize, 2), Mock.actual.n);
        try t.expectEqual(hip.graph_cache.State.failed, tail.graphs.entries.items[0].state);
    }
}

test "HIP tail: body error runs no captured work and launch failure is never retried" {
    Mock.reset();
    const r = Mock.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = null };
    var tail = try Tail.init(t.allocator, &r);
    defer tail.deinit();
    var state: usize = 0;
    var ctx: Ctx = .{ .s = s, .payload = &.{4}, .position = 7, .target = 4096, .state = &state, .fail = true };
    try t.expectError(error.PlantedBodyFailure, tail.submit(u32, .{ .phase = .head, .rows = 1 }, s, ctx, Ctx.body));
    try t.expectEqual(@as(usize, 0), state);
    try t.expectEqual(@as(usize, 0), Mock.actual.n);
    ctx.fail = false;
    Mock.launch_fail = true;
    try t.expectError(error.HipFailed, tail.submit(u32, .{ .phase = .draw, .rows = 1 }, s, ctx, Ctx.body));
    try t.expectEqual(@as(usize, 1), state);
    try t.expectEqual(@as(usize, 0), Mock.actual.n);
    try t.expectEqual(hip.graph_cache.State.failed, tail.graphs.entries.items[1].state);
}

test "HIP tail: completed reads remove only redundant waits and keep fences protect plan sources" {
    Mock.reset();
    const r = Mock.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = null };
    var tail = try Tail.init(t.allocator, &r);
    defer tail.deinit();
    try tail.beforeRound(s);
    try t.expectEqual(@as(usize, 1), Mock.full_waits);
    tail.afterRead();
    try tail.beforeRound(s);
    try t.expectEqual(@as(usize, 1), Mock.full_waits);
    tail.afterRead();
    try tail.kept(s);
    try tail.beforeDraft(s);
    try t.expectEqual(@as(usize, 0), Mock.event_waits);
    tail.afterRead();
    try tail.beforeRound(s);
    try t.expectEqual(@as(usize, 1), Mock.full_waits);
    tail.afterRead();
    try tail.kept(s);
    try tail.beforeRound(s);
    try t.expectEqual(@as(usize, 1), Mock.event_waits);
    tail.clear();
    try t.expect(!tail.host_done and !tail.plan_pending);
}

test "HIP tail: opt-in admits one stream and rank with graphs enabled only" {
    try t.expect(Tail.enabled(true, true, 1, 1, false));
    try t.expect(!Tail.enabled(false, true, 1, 1, false));
    try t.expect(!Tail.enabled(true, false, 1, 1, false));
    try t.expect(!Tail.enabled(true, true, 2, 1, false));
    try t.expect(!Tail.enabled(true, true, 1, 2, false));
    try t.expect(!Tail.enabled(true, true, 1, 1, true));
}

test "HIP tail: a failed plan fence invalidates the completion witness before host reuse" {
    Mock.reset();
    const r = Mock.runtime();
    const s: hip.Stream = .{ .r = &r, .handle = null };
    var tail = try Tail.init(t.allocator, &r);
    defer tail.deinit();
    tail.afterRead();
    Mock.record_fail = true;
    try t.expectError(error.HipFailed, tail.kept(s));
    try t.expect(!tail.host_done and tail.plan_pending);
    try tail.beforeRound(s);
    try t.expectEqual(@as(usize, 1), Mock.full_waits);
    try t.expectEqual(@as(usize, 0), Mock.event_waits);
}
