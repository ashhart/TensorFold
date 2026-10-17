const std = @import("std");
const source = @import("copy_source.zig");
const abi = @import("abi.zig");
const t = std.testing;
const Fake = struct {
    var fail = true;
    var copies: usize = 0;
    var fail_event = false;
    var fail_record = false;
    var interleave = false;
    var driver: ?*@import("driver.zig").Driver = null;
    var early: usize = 0;
    var destroyed: usize = 0;
    var host: ?[]align(16) u8 = null;
    var pending: ?[]const u8 = null;
    var result: [16]u8 = undefined;
    fn copy(_: u64, ptr: ?*const anyopaque, len: usize) callconv(.c) abi.Result {
        copies += 1;
        pending = @as([*]const u8, @ptrCast(ptr.?))[0..len];
        return 0;
    }
    fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
        return sync();
    }
    fn sync() callconv(.c) abi.Result {
        if (interleave) {
            interleave = false;
            fail = true;
            source.synchronousCopy(driver.?, 512, &@as([16]u8, @splat(0x72)), t.allocator) catch {};
            fail = false;
            return 0;
        }
        if (fail) return abi.error_not_ready;
        if (pending) |bytes| @memcpy(result[0..bytes.len], bytes);
        pending = null;
        return 0;
    }
    fn asyncCopy(dst: u64, ptr: ?*const anyopaque, len: usize, _: abi.Stream) callconv(.c) abi.Result {
        return copy(dst, ptr, len);
    }
    fn hostAlloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        host = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        out.* = host.?.ptr;
        return 0;
    }
    fn hostFree(_: ?*anyopaque) callconv(.c) abi.Result {
        if (pending != null) early += 1;
        t.allocator.free(host.?);
        host = null;
        return 0;
    }
    fn eventCreate(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        if (fail_event) return abi.error_not_ready;
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return if (fail_record) abi.error_not_ready else 0;
    }
    fn eventSync(_: abi.Event) callconv(.c) abi.Result {
        return sync();
    }
    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn streamDestroy(_: abi.Stream) callconv(.c) abi.Result {
        destroyed += 1;
        return 0;
    }
    fn init() @import("driver.zig").Driver {
        fail = true;
        fail_event = false;
        fail_record = false;
        interleave = false;
        pending = null;
        early = 0;
        destroyed = 0;
        var d: @import("driver.zig").Driver = undefined;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuMemcpyHtoDAsync_v2 = asyncCopy;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuCtxSynchronize = sync;
        d.api.cuStreamDestroy_v2 = streamDestroy;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = eventSync;
        d.api.cuEventQuery = eventSync;
        d.api.cuEventDestroy_v2 = eventDestroy;
        return d;
    }
};
test "copy source survives caller unwind after accepted transfer and failed completion" {
    var d: @import("driver.zig").Driver = undefined;
    d.api.cuMemcpyHtoD_v2 = Fake.copy;
    d.api.cuStreamSynchronize = Fake.streamSync;
    d.api.cuCtxSynchronize = Fake.sync;
    source.setCurrent(@ptrFromInt(9));
    Fake.fail = true;
    Fake.pending = null;
    defer {
        Fake.pending = null;
        Fake.fail = false;
        source.synchronizeContext(&d) catch unreachable;
    }
    {
        const bytes = try t.allocator.dupe(u8, &@as([16]u8, @splat(0x69)));
        defer t.allocator.free(bytes);
        try t.expectError(error.NotReady, @import("memory.zig").DeviceBuffer.upload(.{ .d = &d, .ptr = 256, .len = 16, .borrowed = true }, 0, bytes));
    }
    try t.expectEqual(@as(usize, 1), source.retained());
    try t.expect(std.mem.allEqual(u8, Fake.pending.?, 0x69));
    Fake.fail = false;
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 0), source.retained());
    try t.expect(std.mem.allEqual(u8, &Fake.result, 0x69));
}

test "failed pinned-source disposal remains quarantined past unsuccessful stream teardown" {
    for (0..3) |kind| {
        var d = Fake.init();
        source.setCurrent(@ptrFromInt(9));
        var stage: source.Stage = .{ .gpa = t.allocator };
        var stream: @import("stream.zig").Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
        const before = @import("memory.zig").usage(false).host;
        defer {
            Fake.fail = false;
            source.synchronizeContext(&d) catch unreachable;
            stage.deinit();
        }
        const bytes = try stage.take(&d, 16);
        @memset(bytes, 0x53);
        Fake.fail_event = kind == 0;
        Fake.fail_record = kind == 1;
        const dst: @import("memory.zig").DeviceBuffer = .{ .d = &d, .ptr = 256, .len = 16, .borrowed = true };
        if (kind < 2) try t.expectError(error.NotReady, stage.copyTo(dst, 0, stream)) else try stage.copyTo(dst, 0, stream);
        stage.deinit();
        try t.expectEqual(@as(usize, 1), source.retained());
        try t.expectEqual(@as(usize, 0), Fake.early);
        stream.deinit();
        try t.expectEqual(@as(usize, 1), Fake.destroyed);
        try t.expectEqual(@as(usize, 1), source.retained());
        try t.expect(std.mem.allEqual(u8, Fake.pending.?, 0x53));
        Fake.fail = false;
        try source.synchronizeContext(&d);
        try t.expectEqual(@as(usize, 0), source.retained());
        try t.expect(Fake.host == null);
        try t.expectEqual(before, @import("memory.zig").usage(false).host);
        try t.expectEqual(@as(usize, 0), Fake.early);
    }
}
test "a failing caller allocator cannot own or invalidate custody storage" {
    var d = Fake.init();
    Fake.fail = false;
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try source.synchronousCopy(&d, 256, &@as([16]u8, @splat(1)), fail.allocator());
    try t.expectEqual(@as(usize, 0), fail.allocations);
    try t.expectEqual(@as(usize, 0), source.retained());
}
test "completion cannot retire a copy quarantined after that completion began" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(9));
    Fake.driver = &d;
    defer {
        Fake.interleave = false;
        Fake.fail = false;
        source.synchronizeContext(&d) catch unreachable;
    }
    Fake.interleave = true;
    Fake.fail = false;
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 1), source.retained());
    try t.expect(std.mem.allEqual(u8, Fake.pending.?, 0x72));
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 0), source.retained());
    try t.expect(std.mem.allEqual(u8, &Fake.result, 0x72));
}

test "a quarantined source closes copy admission until the context recovers" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(9));
    defer {
        Fake.fail = false;
        source.synchronizeContext(&d) catch unreachable;
    }
    const bytes: [16]u8 = @splat(4);
    try t.expectError(error.NotReady, source.synchronousCopy(&d, 256, &bytes, t.allocator));
    const copies = Fake.copies;
    try t.expectError(error.NotReady, source.synchronousCopy(&d, 512, &bytes, t.allocator));
    try t.expectEqual(copies, Fake.copies);
    try t.expectEqual(@as(usize, 1), source.retained());
    source.setCurrent(@ptrFromInt(10));
    Fake.fail = false;
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 1), source.retained());
    source.setCurrent(@ptrFromInt(9));
    try source.synchronizeContext(&d);
    try t.expectEqual(@as(usize, 0), source.retained());
}
test "portable pinned source custody follows its copy stream driver" {
    var pin_driver = Fake.init();
    var copy_driver = Fake.init();
    source.setCurrent(@ptrFromInt(9));
    var stage: source.Stage = .{ .gpa = t.allocator };
    defer {
        Fake.fail = false;
        source.synchronizeContext(&copy_driver) catch unreachable;
        stage.deinit();
    }
    @memset(try stage.take(&pin_driver, 16), 0x65);
    try stage.copyTo(.{ .d = &copy_driver, .ptr = 256, .len = 16, .borrowed = true }, 0, .{ .d = &copy_driver, .handle = @ptrFromInt(2) });
    stage.deinit();
    Fake.fail = false;
    try source.synchronizeContext(&pin_driver);
    try t.expectEqual(@as(usize, 1), source.retained());
    try source.synchronizeContext(&copy_driver);
    try t.expectEqual(@as(usize, 0), source.retained());
}

test "public async uploader owns the caller input on failed sync and teardown" {
    var d = Fake.init();
    source.setCurrent(@ptrFromInt(9));
    const b: @import("memory.zig").DeviceBuffer = .{ .d = &d, .ptr = 256, .len = 16 };
    {
        var bytes: [16]u8 = @splat(0x73);
        try b.uploadAsync(0, &bytes, @ptrFromInt(5));
        @memset(&bytes, 0);
    }
    try t.expectEqual(@as(usize, 1), source.retained());
    try t.expectError(error.NotReady, source.synchronizeStream(&d, @ptrFromInt(5)));
    try t.expect(std.mem.allEqual(u8, Fake.pending.?, 0x73));
    // No library close is allowed while quarantine holds its release callbacks.
    d.close();
    try t.expectEqual(@as(usize, 0), Fake.early);
    Fake.fail = false;
    try source.synchronizeStream(&d, @ptrFromInt(5));
    try t.expectEqual(@as(usize, 0), source.retained());
    try t.expect(std.mem.allEqual(u8, &Fake.result, 0x73));
}
