const std = @import("std");
const t = std.testing;
const abi = @import("abi.zig");
const source = @import("copy_source.zig");
const Driver = @import("driver.zig").Driver;
const Context = @import("context.zig").Context;
const Fake = struct {
    var actual_context: abi.Context = null;
    var bytes: ?[]align(16) u8 = null;
    var pending = false;
    var early = false;
    var fail_record = false;
    fn set(next_context: abi.Context) callconv(.c) abi.Result {
        actual_context = next_context;
        return 0;
    }
    fn alloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        bytes = t.allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        out.* = bytes.?.ptr;
        return 0;
    }
    fn free(_: ?*anyopaque) callconv(.c) abi.Result {
        early = early or pending;
        t.allocator.free(bytes.?);
        bytes = null;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize, _: abi.Stream) callconv(.c) abi.Result {
        pending = true;
        return 0;
    }
    fn create(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        if (!fail_record) return abi.error_not_ready;
        out.* = @ptrFromInt(3);
        return 0;
    }
    fn record(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return abi.error_not_ready;
    }
    fn destroy(_: abi.Event) callconv(.c) abi.Result {
        return 0;
    }
    fn sync() callconv(.c) abi.Result {
        if (actual_context == @as(abi.Context, @ptrFromInt(1))) pending = false;
        return 0;
    }
    fn streamSync(_: abi.Stream) callconv(.c) abi.Result {
        return sync();
    }
    fn init() Driver {
        actual_context = null;
        pending = false;
        early = false;
        var d: Driver = undefined;
        d.api.cuCtxSetCurrent = set;
        d.api.cuCtxSynchronize = sync;
        d.api.cuMemHostAlloc = alloc;
        d.api.cuMemFreeHost = free;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuEventCreate = create;
        d.api.cuEventRecord = record;
        d.api.cuEventDestroy_v2 = destroy;
        d.api.cuStreamSynchronize = streamSync;
        return d;
    }
};
fn context(d: *Driver, id: usize) Context {
    return .{ .d = d, .device = @intCast(id), .handle = @ptrFromInt(id) };
}
fn closeCase(record: bool) !void {
    var d = Fake.init();
    const a = context(&d, 1);
    const b = context(&d, 2);
    var s: source.Stage = .{ .gpa = t.allocator };
    defer {
        a.makeCurrent() catch unreachable;
        s.deinit();
    }
    try a.makeCurrent();
    @memset(try s.take(&d, 16), 0x61);
    Fake.fail_record = record;
    try t.expectError(error.NotReady, s.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, .{ .d = &d, .handle = null }));
    try b.makeCurrent();
    try t.expectError(error.StageBusy, s.close());
    try t.expect(Fake.pending and Fake.bytes != null and !Fake.early);
    try t.expectEqual(b.handle, source.current());
    try a.makeCurrent();
    try s.close();
    try t.expect(!Fake.pending and Fake.bytes == null and !Fake.early);
}
fn deinitCase(record: bool) !void {
    var d = Fake.init();
    const a = context(&d, 1);
    const b = context(&d, 2);
    var s: source.Stage = .{ .gpa = t.allocator };
    try a.makeCurrent();
    @memset(try s.take(&d, 16), 0x62);
    Fake.fail_record = record;
    try t.expectError(error.NotReady, s.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, .{ .d = &d, .handle = null }));
    try b.makeCurrent();
    s.deinit();
    // Always drain A before reporting a failed assertion, including on the unfixed implementation.
    const held = Fake.pending and Fake.bytes != null and !Fake.early and source.retained() == 1;
    try b.synchronize();
    const isolated = Fake.pending and Fake.bytes != null and source.retained() == 1;
    try a.makeCurrent();
    try a.synchronize();
    try t.expect(held and isolated and !Fake.early);
    try t.expect(Fake.bytes == null);
    try t.expectEqual(@as(usize, 0), source.retained());
}
test "stage close refuses another context after default-stream event creation failure" {
    try closeCase(false);
}
test "stage close refuses another context after default-stream event recording failure" {
    try closeCase(true);
}
test "stage deinit quarantines the saved default-stream context after event creation failure" {
    try deinitCase(false);
}
test "stage deinit quarantines the saved default-stream context after event recording failure" {
    try deinitCase(true);
}
test "same default-stream handle cannot rebind a pending stage to another context" {
    var d = Fake.init();
    const a = context(&d, 1);
    const b = context(&d, 2);
    var s: source.Stage = .{ .gpa = t.allocator };
    defer {
        a.makeCurrent() catch unreachable;
        s.deinit();
    }
    try a.makeCurrent();
    _ = try s.take(&d, 16);
    Fake.fail_record = false;
    try t.expectError(error.NotReady, s.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, .{ .d = &d, .handle = null }));
    try b.makeCurrent();
    try t.expectError(error.StageBusy, s.begin(.{ .d = &d, .handle = null }));
    try t.expectEqual(a.handle, s.context);
}
