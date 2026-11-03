//! Exercise the actual upload, runtime launches and resource teardown through a recording CUDA driver.
const std = @import("std");
const cuda = @import("root.zig");
const abi = cuda.abi;
const pf = @import("l2_prefetch.zig");
const src = @import("l2_sources.zig");
const t = std.testing;
const Fake = struct {
    var calls: usize = 0;
    var fail_at: usize = 0;
    var buffers: usize = 0;
    var modules: usize = 0;
    var streams: usize = 0;
    var events: usize = 0;
    var next: usize = 80;
    var order: [512]u8 = undefined;
    var at: usize = 0;
    var table: [12][2]i64 = undefined;
    var grid: u32 = 0;
    var block: u32 = 0;
    var args: [5]u64 = undefined;
    fn note(c: u8) void {
        order[at] = c;
        at += 1;
    }
    fn attempt() bool {
        calls += 1;
        return fail_at != 0 and calls == fail_at;
    }
    fn reset(fail: usize) void {
        calls = 0;
        fail_at = fail;
        buffers = 0;
        modules = 0;
        streams = 0;
        events = 0;
        next = 80;
        at = 0;
    }
    fn alloc(p: *u64, _: usize) callconv(.c) abi.Result {
        if (attempt()) return abi.error_not_ready;
        buffers += 1;
        p.* = 0x10000;
        return 0;
    }
    fn free(_: u64) callconv(.c) abi.Result {
        buffers -= 1;
        note('B');
        return 0;
    }
    fn copy(_: u64, p: ?*const anyopaque, n: usize) callconv(.c) abi.Result {
        if (attempt()) return abi.error_not_ready;
        @memcpy(std.mem.sliceAsBytes(&table)[0..n], @as([*]const u8, @ptrCast(p.?))[0..n]);
        return 0;
    }
    fn sync(_: abi.Stream) callconv(.c) abi.Result {
        note('d');
        return 0;
    }
    fn load(m: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
        if (attempt()) return abi.error_not_ready;
        modules += 1;
        next += 1;
        m.* = @ptrFromInt(next);
        return 0;
    }
    fn unload(_: abi.Module) callconv(.c) abi.Result {
        modules -= 1;
        note('M');
        return 0;
    }
    fn function(f: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
        f.* = @ptrFromInt(if (std.mem.indexOf(u8, std.mem.span(name), "segments") != null) @as(usize, 1) else 2);
        return 0;
    }
    fn stream(s: *abi.Stream, flags: c_uint) callconv(.c) abi.Result {
        if (attempt()) return abi.error_not_ready;
        std.debug.assert(flags == abi.stream_non_blocking);
        streams += 1;
        s.* = @ptrFromInt(50);
        return 0;
    }
    fn destroyStream(_: abi.Stream) callconv(.c) abi.Result {
        streams -= 1;
        note('S');
        return 0;
    }
    fn event(e: *abi.Event, flags: c_uint) callconv(.c) abi.Result {
        if (attempt()) return abi.error_not_ready;
        std.debug.assert(flags == abi.event_disable_timing);
        events += 1;
        next += 1;
        e.* = @ptrFromInt(next);
        return 0;
    }
    fn destroyEvent(_: abi.Event) callconv(.c) abi.Result {
        events -= 1;
        note('E');
        return 0;
    }
    fn record(_: abi.Event, s: abi.Stream) callconv(.c) abi.Result {
        note(if (@intFromPtr(s) == 5) 'f' else 'j');
        return 0;
    }
    fn wait(s: abi.Stream, _: abi.Event, _: c_uint) callconv(.c) abi.Result {
        note(if (@intFromPtr(s) == 5) 'm' else 's');
        return 0;
    }
    fn launch(f: abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, shared: c_uint, s: abi.Stream, p: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        std.debug.assert(shared == 0 and @intFromPtr(s) == 50);
        note('l');
        grid = gx;
        block = bx;
        const params = p.?;
        args[0] = @as(*const u64, @ptrCast(@alignCast(params[0].?))).*;
        args[1] = @as(*const u32, @ptrCast(@alignCast(params[1].?))).*;
        args[2] = @as(*const u32, @ptrCast(@alignCast(params[2].?))).*;
        if (@intFromPtr(f) == 2) {
            args[3] = @as(*const u64, @ptrCast(@alignCast(params[3].?))).*;
            args[4] = @as(*const u64, @ptrCast(@alignCast(params[4].?))).*;
        }
        return 0;
    }
    fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result {
        note('[');
        return 0;
    }
    fn end(_: abi.Stream, g: *abi.Graph) callconv(.c) abi.Result {
        note(']');
        g.* = @ptrFromInt(1);
        return 0;
    }
    fn destroyGraph(_: abi.Graph) callconv(.c) abi.Result {
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        d.api.cuModuleLoadData = load;
        d.api.cuModuleUnload = unload;
        d.api.cuModuleGetFunction = function;
        d.api.cuStreamCreate = stream;
        d.api.cuStreamDestroy_v2 = destroyStream;
        d.api.cuEventCreate = event;
        d.api.cuEventDestroy_v2 = destroyEvent;
        d.api.cuEventRecord = record;
        d.api.cuStreamWaitEvent = wait;
        d.api.cuLaunchKernel = launch;
        d.api.cuStreamBeginCapture_v2 = begin;
        d.api.cuStreamEndCapture = end;
        d.api.cuGraphDestroy = destroyGraph;
        return d;
    }
    fn balanced() !void {
        try t.expect(buffers == 0 and modules == 0 and streams == 0 and events == 0);
    }
};
const image: [16]u8 align(16) = @splat(1);
fn runtime(d: *const cuda.Driver, knobs: pf.Knobs) !pf.Runtime {
    const spans = try t.allocator.alloc(pf.Span, 3);
    errdefer t.allocator.free(spans);
    spans[0] = .{ .at = 0, .rows = 1, .bytes = 12 << 20 };
    spans[1] = .{ .at = 1, .rows = 1, .bytes = 12 << 20 };
    spans[2] = .{ .at = 2, .rows = 1, .bytes = 12 << 20 };
    return pf.Runtime.initImages(t.allocator, d, knobs, &.{ .{ 0x20000, 12 << 20 }, .{ 0x30000, 12 << 20 }, .{ 0x40000, 12 << 20 } }, spans, .{ &image, &image });
}
test "L2 prefetch: off touches no allocator, view or driver" {
    Fake.reset(0);
    try t.expect(try src.start(undefined, undefined, false, .{}, undefined, undefined) == null);
    try t.expectEqual(@as(usize, 0), Fake.calls);
}
test "L2 prefetch: exact paced and unpaced argument bytes, captured forks and final join" {
    Fake.reset(0);
    var d = Fake.driver();
    var p = try runtime(&d, .{});
    const main: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(5) };
    Fake.at = 0;
    try cuda.graph.beginCapture(main, .thread_local);
    try p.fork(main, 0, .n);
    try t.expectEqual(@as(u32, 2), Fake.grid);
    try t.expectEqual(@as(u32, 32), Fake.block);
    try t.expectEqual([5]u64{ 0x10000, 1, 32768, 218, 0 }, Fake.args);
    try p.fork(main, 0, .f);
    try t.expectEqual(@as(u64, 0x10010), Fake.args[0]);
    try p.fork(main, 0, .o);
    try t.expectEqual(@as(u64, 0x10020), Fake.args[0]);
    try t.expectEqual(@as(u32, 2), Fake.grid);
    try t.expectEqual(@as(u32, 128), Fake.block);
    try p.fork(main, 9, .n); // no site or driver call beyond the recorded layers
    try p.join(main);
    var graph = try cuda.graph.endCapture(main);
    defer graph.deinit();
    try t.expectEqualStrings("[fslfslfsljm]", Fake.order[0..Fake.at]);
    Fake.at = 0;
    p.deinit();
    try Fake.balanced();
    try t.expect(std.mem.startsWith(u8, Fake.order[0..Fake.at], "dSEEBMM"));
    try t.expectEqual(@as(i64, 12 << 20), Fake.table[0][1]);
}
test "L2 prefetch: every partial resource failure unwinds, and pace zero takes segments" {
    for (1..8) |failure| {
        Fake.reset(failure);
        var d = Fake.driver();
        try t.expectError(error.NotReady, runtime(&d, .{}));
        try Fake.balanced();
    }
    Fake.reset(0);
    var d = Fake.driver();
    var p = try runtime(&d, .{ .pace_gbps = 0, .ctas = 4 });
    try p.fork(.{ .d = &d, .handle = @ptrFromInt(5) }, 0, .n);
    try t.expectEqual(@as(u32, 3), Fake.grid);
    try t.expectEqual(@as(u32, 128), Fake.block);
    p.deinit();
    try Fake.balanced();
}
fn Env(comptime pairs: anytype) type {
    return struct {
        fn get(k: [:0]const u8) ?[]const u8 {
            inline for (pairs) |p| if (std.mem.eql(u8, k, p[0])) return p[1];
            return null;
        }
    };
}
test "L2 prefetch: both families default off, knobs ignored off, bad settings name the family" {
    inline for (.{ "TF_QWEN_L2PF", "TF_NEMOTRON_L2PF" }) |prefix| {
        const Options = src.Settings(prefix);
        var problem: []const u8 = "";
        try t.expect(!(try Options.read(Env(.{}).get, &problem)).on);
        try t.expect(!(try Options.read(Env(.{.{ prefix ++ "_CTAS", "bad" }}).get, &problem)).on);
        try t.expect(!(try Options.read(Env(.{ .{ prefix, "0" }, .{ prefix ++ "_MB", "0" } }).get, &problem)).on);
        const on = try Options.read(Env(.{.{ prefix, "1" }}).get, &problem);
        try t.expect(on.on);
        try t.expectEqual(pf.Knobs{}, on.knobs);
        try t.expectError(error.BadSetting, Options.read(Env(.{.{ prefix, "true" }}).get, &problem));
        try t.expectEqualStrings(prefix ++ ": 0 or 1", problem);
        inline for (.{ "_MB", "_CTAS", "_PACE_GBPS" }) |knob| {
            try t.expectError(error.BadSetting, Options.read(Env(.{ .{ prefix, "1" }, .{ prefix ++ knob, "bad" } }).get, &problem));
            try t.expectEqualStrings(prefix ++ knob, problem);
        }
    }
}
test "L2 prefetch: real allocation bounds, interior views, source order and per-site budget" {
    var s: src.Sources = .{};
    s.add(0x1000);
    s.add(0x1040);
    s.add(0x1000);
    s.add(0x2000);
    var rows: std.ArrayList([2]i64) = .empty;
    defer rows.deinit(t.allocator);
    var sp: pf.Span = undefined;
    try src.append(t.allocator, s, &.{ .{ .ptr = 0x1000, .len = 128 }, .{ .ptr = 0x2000, .len = 100 } }, 160, &rows, &sp);
    try t.expectEqualSlices([2]i64, &.{ .{ 0x1000, 64 }, .{ 0x1040, 64 }, .{ 0x2000, 32 } }, rows.items);
    try t.expectEqual(@as(u64, 160), sp.bytes);
    rows.clearRetainingCapacity();
    try t.expectError(error.PrefetchWeightRange, src.append(t.allocator, s, &.{}, 160, &rows, &sp));
}
