const std = @import("std");
const t = std.testing;
const abi = @import("abi.zig");
const source = @import("copy_source.zig");
const budget = @import("copy_source_budget.zig");
const streams = @import("stream.zig");
const Driver = @import("driver.zig").Driver;
const Context = @import("context.zig").Context;

/// A driver whose copies and events finish only when the test says the GPU got there (`complete`).
const Fake = struct {
    var allocs: usize = 0;
    var frees: usize = 0;
    var early: usize = 0; // pinned bytes freed, or copied from again, while a copy from them was still in flight
    var copies: usize = 0;
    var query_result: abi.Result = 0;
    var sync_result: abi.Result = 0;
    var next_event: usize = 1;
    var pending: std.ArrayList(usize) = .empty;
    var flight: std.ArrayList(usize) = .empty;
    var blocks: std.ArrayList([]align(16) u8) = .empty;
    var seen: [64]u8 = undefined;
    fn has(list: *std.ArrayList(usize), v: usize) bool {
        return std.mem.indexOfScalar(usize, list.items, v) != null;
    }
    fn complete() void {
        pending.clearRetainingCapacity();
        flight.clearRetainingCapacity();
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        const b = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        blocks.append(t.allocator, b) catch return abi.error_not_ready;
        allocs += 1;
        out.* = b.ptr;
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) abi.Result {
        if (has(&flight, @intFromPtr(ptr))) early += 1;
        for (blocks.items, 0..) |b, i| if (@intFromPtr(b.ptr) == @intFromPtr(ptr)) {
            t.allocator.free(b);
            _ = blocks.swapRemove(i);
            frees += 1;
            return 0;
        };
        return abi.error_not_ready;
    }
    fn copy(_: u64, ptr: ?*const anyopaque, _: usize, _: abi.Stream) callconv(.c) abi.Result {
        if (has(&flight, @intFromPtr(ptr))) early += 1;
        flight.append(t.allocator, @intFromPtr(ptr)) catch return abi.error_not_ready;
        seen[copies] = @as([*]const u8, @ptrCast(ptr.?))[0];
        copies += 1;
        return 0;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(next_event);
        next_event += 1;
        return 0;
    }
    fn record(e: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        pending.append(t.allocator, @intFromPtr(e)) catch return abi.error_not_ready;
        return 0;
    }
    fn query(e: abi.Event) callconv(.c) abi.Result {
        if (query_result != 0) return query_result;
        return if (has(&pending, @intFromPtr(e))) abi.error_not_ready else 0;
    }
    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
        return contextSync();
    }
    fn contextSync() callconv(.c) abi.Result {
        if (sync_result == 0) complete();
        return sync_result;
    }
    fn release(_: abi.Device) callconv(.c) abi.Result {
        return 0;
    }
    fn init() Driver {
        allocs = 0;
        frees = 0;
        early = 0;
        copies = 0;
        query_result = 0;
        sync_result = 0;
        complete();
        var d: Driver = undefined;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = record;
        d.api.cuEventQuery = query;
        d.api.cuEventDestroy_v2 = eventDestroy;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = contextSync;
        d.api.cuDevicePrimaryCtxRelease_v2 = release;
        return d;
    }
    fn deinit() void {
        pending.deinit(t.allocator);
        flight.deinit(t.allocator);
        blocks.deinit(t.allocator);
        pending = .empty;
        flight = .empty;
        blocks = .empty;
    }
};

fn spares() void {
    source.setSparesForTest(.{ .count = 4, .bytes = 1 << 20 });
}
fn settle(d: *const Driver) void {
    Fake.query_result = 0;
    Fake.sync_result = 0;
    source.synchronizeContext(d) catch unreachable;
    _ = source.keepDriver(d);
    source.setSparesForTest(.{ .count = 0, .bytes = 0 });
    Fake.deinit();
}

test "temporary uploads take a completed stage again instead of allocating another" {
    var d = Fake.init();
    spares();
    defer settle(&d);
    source.setCurrent(@ptrFromInt(9));
    const stream: streams.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    for (0..10) |i| {
        try source.enqueue(&d, 256, &@as([16]u8, @splat(@intCast(i))), stream);
        Fake.complete();
    }
    try source.synchronizeStream(&d, stream.handle);
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expectEqual(@as(usize, 0), Fake.early);
    for (0..10) |i| try t.expectEqual(@as(u8, @intCast(i)), Fake.seen[i]);
    try t.expectEqual(@as(usize, 1), source.spareCount());
    try t.expectEqual(@as(usize, 0), source.retained());
    source.dropSpares(&d, @ptrFromInt(9));
    try t.expectEqual(@as(usize, 0), source.spareCount());
    try t.expectEqual(Fake.allocs, Fake.frees);
}

test "a stage whose copy is still in flight is never handed to another upload" {
    var d = Fake.init();
    spares();
    defer settle(&d);
    source.setCurrent(@ptrFromInt(9));
    const stream: streams.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    for (0..6) |i| try source.enqueue(&d, 256, &@as([16]u8, @splat(@intCast(i))), stream);
    try t.expectEqual(@as(usize, 6), Fake.allocs);
    try t.expectEqual(@as(usize, 6), source.retained());
    try t.expectEqual(@as(usize, 0), Fake.early);
    try source.synchronizeStream(&d, stream.handle);
    try t.expectEqual(@as(usize, 4), source.spareCount()); // the bound keeps four and frees the rest
    try t.expectEqual(@as(usize, 2), Fake.frees);
    source.dropSpares(&d, @ptrFromInt(9));
    try t.expectEqual(Fake.allocs, Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.early);
}

test "a stage whose completion failed is quarantined, never kept for reuse" {
    var d = Fake.init();
    spares();
    defer settle(&d);
    source.setCurrent(@ptrFromInt(9));
    const stream: streams.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    try source.enqueue(&d, 256, &@as([16]u8, @splat(1)), stream);
    Fake.query_result = 700; // the copy's event reports a fault
    Fake.sync_result = abi.error_not_ready;
    try t.expectError(error.NotReady, source.enqueue(&d, 512, &@as([16]u8, @splat(2)), stream));
    try t.expectEqual(@as(usize, 1), Fake.allocs);
    try t.expectEqual(@as(usize, 1), source.retained());
    try t.expectEqual(@as(usize, 0), source.spareCount());
    Fake.query_result = 0;
    Fake.sync_result = 0;
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 0), source.retained());
    try t.expectEqual(@as(usize, 0), source.spareCount());
    try t.expectEqual(@as(usize, 1), Fake.frees);
    try t.expectEqual(@as(usize, 0), Fake.early);
}

test "a context's end and the driver's close free the spares" {
    var d = Fake.init();
    spares();
    defer settle(&d);
    const stream: streams.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    for ([_]usize{ 9, 10 }) |ctx| {
        source.setCurrent(@ptrFromInt(ctx));
        try source.enqueue(&d, 256, &@as([16]u8, @splat(3)), stream);
        try source.synchronizeStream(&d, stream.handle);
    }
    try t.expectEqual(@as(usize, 2), source.spareCount());
    var c: Context = .{ .d = &d, .device = 0, .handle = @ptrFromInt(10) };
    c.deinit();
    try t.expectEqual(@as(usize, 1), source.spareCount());
    try t.expect(!source.keepDriver(&d));
    try t.expectEqual(@as(usize, 0), source.spareCount());
    try t.expectEqual(Fake.allocs, Fake.frees);
}

test "spares count against the copy-source bound and give way to a new stage" {
    budget.setForTest(.{ .bytes = 1 << 20, .nodes = 1 });
    defer budget.setForTest(.{});
    var d = Fake.init();
    spares();
    defer settle(&d);
    source.setCurrent(@ptrFromInt(9));
    const stream: streams.Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    try source.enqueue(&d, 256, &@as([16]u8, @splat(4)), stream);
    try source.synchronizeStream(&d, stream.handle);
    try t.expectEqual(@as(usize, 1), source.spareCount());
    try source.enqueue(&d, 256, &@as([8192]u8, @splat(5)), stream); // too big for the spare; the bound holds one
    try t.expectEqual(@as(usize, 2), Fake.allocs);
    try t.expectEqual(@as(usize, 1), Fake.frees);
    try t.expectEqual(@as(u8, 5), Fake.seen[1]);
    try source.synchronizeStream(&d, stream.handle);
    source.dropSpares(&d, @ptrFromInt(9));
    try t.expectEqual(Fake.allocs, Fake.frees);
}
