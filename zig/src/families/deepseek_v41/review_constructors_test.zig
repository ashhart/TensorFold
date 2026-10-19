const std = @import("std");
const cuda = @import("cuda");
const mock = @import("load_mock.zig");
const fwd = @import("forward.zig");
const run = @import("run.zig");
const t = std.testing;
fn cleanup(d: *const cuda.Driver) void {
    mock.fail_sync = false;
    d.synchronize() catch unreachable;
    for (mock.live, 0..) |live, i| if (live) {
        var b: cuda.DeviceBuffer = .{ .d = d, .ptr = i * 256, .len = mock.lengths[i] };
        b.free();
    };
    if (mock.pinned != null) _ = d.api.cuMemFreeHost(mock.pinned.?.ptr);
}
test "Completion ownership: failed Batch pending allocation unwinds the already-created row table" {
    var d = mock.init();
    defer cleanup(&d);
    var runner: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    var f: fwd.Forward = undefined;
    f.runner = &runner;
    f.graphs_default = false;
    var ss: @import("slots.zig").SlotSet = undefined;
    ss.n = 1;
    var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 1 });
    try t.expectError(error.OutOfMemory, @import("batch.zig").Batch.init(fail.allocator(), &f, &ss, 1));
    try t.expectEqual(mock.allocated, mock.freed);
}
test "Completion ownership: failed swapped-list append unwinds its new aside buffer" {
    var d = mock.init();
    defer cleanup(&d);
    var runner: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    defer runner.deinit();
    try runner.external("s.ced.x", 256);
    var f: fwd.Forward = undefined;
    f.runner = &runner;
    var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    var ss: @import("slots.zig").SlotSet = .{ .gpa = fail.allocator(), .f = &f, .n = 1, .states = &.{}, .transactions = &.{} };
    defer ss.swapped.deinit(ss.gpa);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var plan: @import("buffers.zig").Plan = .{ .a = arena.allocator() };
    try plan.sizes.put(arena.allocator(), "s.ced.x", 16);
    try t.expectError(error.OutOfMemory, ss.swap(&plan, &.{"s.ced.x"}));
    try t.expectEqual(mock.allocated, mock.freed);
}
fn world(_: *anyopaque) u32 {
    return 1;
}
test "Completion ownership: failed DSpark event construction unwinds its preceding device and host owners" {
    var d = mock.init();
    defer cleanup(&d);
    var vt: @import("tp").collective.Collective.VTable = undefined;
    vt.world = world;
    const comm: @import("tp").collective.Collective = .{ .ptr = &d, .vtable = &vt };
    mock.fail_event = true;
    try t.expectError(error.NotReady, @import("dspark_dev.zig").Dev.init(t.allocator, &d, comm, .{ .d = &d, .handle = null }, .device, 1, 8, 1, false, 16, 1));
    try t.expectEqual(mock.allocated, mock.freed);
    try t.expect(mock.pinned == null);
}

fn gpuInitCase(a: std.mem.Allocator) !void {
    var d = mock.init();
    var vt: @import("tp").collective.Collective.VTable = undefined;
    vt.world = world;
    var cfg: @import("config.zig").Config = .{ .vocab = 8, .dspark_markov_rank = 2, .hidden = 16, .window = 2, .dspark_block = 1 };
    var widths: @import("block.zig").Widths = .{};
    var weights: @import("cuda_weights.zig").Weights = .{ .gpa = a };
    var runner: run.Runner = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = &weights };
    var f = fwd.Forward.init(a, &cfg, &widths, .{}, &runner, .{ .ptr = &d, .vtable = &vt });
    defer {
        f.arena.deinit();
        f.init_owners.deinit(a);
    }
    _ = @import("dspark_gpu.zig").GpuPass.init(a, &f, 2) catch |e| {
        try t.expect(f.drafter == null);
        try t.expectEqual(@as(usize, 0), f.init_owners.items.len);
        if (e == error.NoWeight) return;
        return e;
    };
    return error.UnexpectedSuccess;
}
test "Completion ownership: GpuPass host-array and fetch failures release every partial constructor owner" {
    try t.checkAllAllocationFailures(t.allocator, gpuInitCase, .{});
}
test "Completion ownership: SlotPass host-array allocation sweep leaves callbacks and owners untouched" {
    for (0..13) |fail_at| {
        var d = mock.init();
        var fail = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = fail_at });
        const a = fail.allocator();
        var vt: @import("tp").collective.Collective.VTable = undefined;
        vt.world = world;
        var cfg: @import("config.zig").Config = .{ .vocab = 8, .hidden = 16, .window = 2, .dspark_block = 1 };
        var widths: @import("block.zig").Widths = .{};
        var runner: run.Runner = .{ .gpa = a, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
        var f = fwd.Forward.init(a, &cfg, &widths, .{}, &runner, .{ .ptr = &d, .vtable = &vt });
        defer {
            f.arena.deinit();
            f.init_owners.deinit(a);
        }
        var one: @import("dspark_gpu.zig").GpuPass = undefined;
        one.k = 1;
        var b: @import("batch.zig").Batch = undefined;
        b.taps_stash = null;
        try t.expectError(error.OutOfMemory, @import("dspark_slots.zig").SlotPass.init(a, &f, &one, &b, 1, 1));
        try t.expect(f.ext == null);
        try t.expectEqual(@as(usize, 0), f.init_owners.items.len);
    }
}
test "Completion ownership: deferred target release retains failed rollback until the retry succeeds" {
    var d = mock.init();
    var runner: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = undefined };
    var cfg: @import("config.zig").Config = .{};
    var widths: @import("block.zig").Widths = .{};
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, &runner, undefined);
    defer {
        mock.fail_copy = false;
        mock.fail_sync = false;
        runner.stream.synchronize() catch unreachable;
        f.transaction.deinit(t.allocator);
        f.arena.deinit();
    }
    f.slot = .{ .pos = 5, .pending = .{ .start = 5, .n = 2 } };
    f.transaction.runner = &runner;
    f.transaction.state = .{ .position = 5, .pending = 2 };
    try f.transaction.regions.append(t.allocator, .{ .live = 256, .backup = 512, .row_bytes = 4, .rows = 1, .kind = .carry });
    var target: @import("target.zig").GpuTarget = .{ .f = &f, .defer_release = true };
    mock.fail_copy = true;
    try target.target().releaseChecked(0);
    target.settleReleases();
    try t.expect(!target.target().available(0));
    try t.expect(target.releasing[0]);
    try t.expectEqual(@as(?u32, 2), f.transaction.state.pending);
    mock.fail_copy = false;
    target.settleReleases();
    try t.expect(target.target().available(0));
    try t.expect(!target.releasing[0]);
    try t.expectEqual(@as(u64, 0), f.slot.pos);
}

test "Completion ownership: constructor unwind after a read retains the partial GpuPass when its drain fails" {
    var d = mock.init();
    var vt: @import("tp").collective.Collective.VTable = undefined;
    vt.world = world;
    var cfg: @import("config.zig").Config = .{ .vocab = 8, .dspark_markov_rank = 2, .hidden = 16, .window = 2, .dspark_block = 1 };
    var widths: @import("block.zig").Widths = .{};
    var weights: @import("cuda_weights.zig").Weights = .{ .gpa = t.allocator };
    defer weights.map.deinit(t.allocator);
    try weights.map.put(t.allocator, "dspark.markov.w1", .{ .ptr = 256, .len = 32, .kind = .bf16, .sha256 = @splat(0) });
    var runner: run.Runner = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null }, .kernels = undefined, .triton = undefined, .weights = &weights };
    var f = fwd.Forward.init(t.allocator, &cfg, &widths, .{}, &runner, .{ .ptr = &d, .vtable = &vt });
    defer {
        mock.fail_sync = false;
        @import("construction.zig").deinit(&f);
        f.arena.deinit();
    }
    mock.fail_sync = true;
    try t.expectError(error.NoWeight, @import("dspark_gpu.zig").GpuPass.init(t.allocator, &f, 2));
    try t.expect(f.drafter == null);
    try t.expectEqual(@as(usize, 1), f.init_owners.items.len);
    const partial: *@import("dspark_gpu.zig").GpuPass = @ptrCast(@alignCast(f.init_owners.items[0].ptr));
    try t.expectEqual(@as(usize, 16), partial.w1.len);
    mock.fail_sync = false;
    try f.drain();
}
