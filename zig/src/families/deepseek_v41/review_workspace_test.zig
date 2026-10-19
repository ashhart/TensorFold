const std = @import("std");
const cuda = @import("cuda");
const mock = @import("load_mock.zig");
const ws = @import("forward_prefill_k4.zig");
const fwd = @import("forward.zig");
const pf = @import("forward_prefill.zig");
const run = @import("run.zig");
const t = std.testing;
test "Completion ownership: workspace completion failure retains its published owner and retries before use" {
    var d = mock.init();
    var runner: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var plan: @import("buffers.zig").Plan = .{ .a = arena.allocator() };
    try plan.sizes.put(arena.allocator(), "w.k4~one", 64);
    var state: pf.State = .{};
    state.k4ws = .{ .plan = plan, .allocs = 4, .releases = 4 };
    var f: fwd.Forward = undefined;
    f.prefill_state = &state;
    f.runner = &runner;
    defer {
        mock.fail_sync = false;
        runner.stream.synchronize() catch unreachable;
        if (state.k4ws.buf) |*b| {
            if (mock.live[b.ptr / 256]) b.free();
        }
    }
    mock.fail_sync = true;
    try t.expectError(error.NotReady, ws.k4Acquire(&f));
    try t.expect(state.k4ws.buf != null);
    try t.expect(mock.live[state.k4ws.buf.?.ptr / 256]);
    try t.expectEqual(@as(usize, 0), mock.early_free);
    try t.expectError(error.NotReady, ws.k4Acquire(&f));
    mock.fail_sync = false;
    try ws.k4Acquire(&f);
    try ws.k4Release(&f);
    try t.expect(runner.addressOf("w.k4~one") == null);
    try t.expectEqual(mock.allocated, mock.freed);
    try t.expectEqual(@as(usize, 0), mock.double_free);
}
test "Completion ownership: device growth preserves the old owner on replacement failure" {
    var d = mock.init();
    var b: ?cuda.DeviceBuffer = try cuda.DeviceBuffer.alloc(&d, 256);
    const old = b.?.ptr;
    defer if (b) |*x| {
        if (mock.live[x.ptr / 256]) x.free();
    };
    mock.fail_alloc = true;
    try t.expectError(error.NotReady, @import("forward_prefill_glue.zig").grow(&b, &d, 512));
    try t.expectEqual(old, b.?.ptr);
    try t.expect(mock.live[old / 256]);
    mock.fail_alloc = false;
    _ = try @import("forward_prefill_glue.zig").grow(&b, &d, 512);
    b.?.free();
    b = null;
    try t.expectEqual(mock.allocated, mock.freed);
    try t.expectEqual(@as(usize, 0), mock.double_free);
}

test "Completion ownership: borrowed upload caller can unwind after failed completion while its copy remains retained" {
    var d = mock.init();
    var runner: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    var cfg: @import("config.zig").Config = .{};
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, &runner, undefined);
    var dst = try cuda.DeviceBuffer.alloc(&d, 16);
    defer {
        mock.fail_sync = false;
        runner.stream.synchronize() catch unreachable;
        f.uploads.close() catch unreachable;
        dst.free();
        f.arena.deinit();
    }
    mock.fail_sync = true;
    {
        const source = try t.allocator.dupe(u8, &@as([16]u8, @splat(0x7b)));
        defer t.allocator.free(source);
        try t.expectError(error.NotReady, f.upload(dst.ptr, source));
    }
    try t.expect(f.uploads.pending);
    try t.expect(mock.pinned != null);
    try t.expect(std.mem.allEqual(u8, mock.pinned.?[0..16], 0x7b));
    try t.expectEqual(@as(usize, 0), mock.early_free);
    mock.fail_sync = false;
    try f.uploads.close();
    try t.expect(mock.pinned == null);
}

fn mappingCase(a: std.mem.Allocator) !void {
    var d = mock.init();
    var runner: run.Runner = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var plan: @import("buffers.zig").Plan = .{ .a = arena.allocator() };
    try plan.sizes.put(arena.allocator(), "w.k4~one", 16);
    try plan.sizes.put(arena.allocator(), "w.k4~two", 16);
    var state: pf.State = .{};
    state.k4ws = .{ .plan = plan, .allocs = 4, .releases = 4 };
    var f: fwd.Forward = undefined;
    f.prefill_state = &state;
    f.runner = &runner;
    defer {
        runner.stream.synchronize() catch unreachable;
        ws.k4Release(&f) catch unreachable;
    }
    ws.k4Acquire(&f) catch |e| {
        try t.expect(runner.addressOf("w.k4~one") == null);
        try t.expect(runner.addressOf("w.k4~two") == null);
        try t.expect(state.k4ws.buf != null);
        try t.expect(mock.live[state.k4ws.buf.?.ptr / 256]);
        return e;
    };
}
test "Completion ownership: failed workspace mapping publication unwinds every partial external binding" {
    try t.checkAllAllocationFailures(t.allocator, mappingCase, .{});
}
