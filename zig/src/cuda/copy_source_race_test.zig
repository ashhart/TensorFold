const custody_sync = @import("copy_source_sync.zig");
const std = @import("std");
const builtin = @import("builtin");
const t = std.testing;
const source = @import("copy_source.zig");
const budget = @import("copy_source_budget.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Fake = struct {
    var mutex: custody_sync.Mutex = .{};
    var cond: custody_sync.Condition = .{};
    var phase: u8 = 0;
    var allowed: u8 = 0;
    var close_started = false;
    var close_returned = false;
    var close_waiting = false;
    var premature = false;
    var bytes: []align(16) u8 = undefined;
    var reaper: *const fn (*const Driver) void = undefined;
    var completing = false;
    fn pause(at: u8) void {
        mutex.lock();
        defer mutex.unlock();
        phase = at;
        cond.broadcast();
        while (allowed < at) cond.wait(&mutex);
        premature = premature or close_returned or budget.usage().nodes == 0;
        if (budget.acquire(0, 1)) |_| {
            budget.release(0, 1);
            premature = true;
        } else |_| {}
    }
    fn alloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        bytes = std.heap.page_allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        out.* = bytes.ptr;
        return 0;
    }
    fn free(_: ?*anyopaque) callconv(.c) abi.Result {
        pause(3);
        std.heap.page_allocator.free(bytes);
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn create(out: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn record(_: abi.Event, _: abi.Stream) callconv(.c) abi.Result {
        return 0;
    }
    fn query(_: abi.Event) callconv(.c) abi.Result {
        pause(1);
        return 0;
    }
    fn destroy(_: abi.Event) callconv(.c) abi.Result {
        pause(2);
        return 0;
    }
    fn sync() callconv(.c) abi.Result {
        return 0;
    }
    fn run(d: *Driver) void {
        source.setCurrent(@ptrFromInt(41));
        if (completing) source.synchronizeContext(d) catch unreachable else reaper(d);
    }
    fn waiting() void {
        mutex.lock();
        close_waiting = true;
        cond.broadcast();
        mutex.unlock();
    }
    fn close(d: *Driver) void {
        mutex.lock();
        close_started = true;
        cond.broadcast();
        mutex.unlock();
        d.close();
        mutex.lock();
        close_returned = true;
        cond.broadcast();
        mutex.unlock();
    }
};
pub fn run(reaper: *const fn (*const Driver) void, completing: bool) !void {
    budget.setForTest(.{ .bytes = 8192, .nodes = 1 });
    defer budget.setForTest(.{});
    Fake.phase = 0;
    Fake.allowed = 0;
    Fake.close_started = false;
    Fake.close_returned = false;
    Fake.close_waiting = false;
    source.on_close_wait = Fake.waiting;
    defer source.on_close_wait = null;
    Fake.premature = false;
    Fake.reaper = reaper;
    Fake.completing = completing;
    var d: Driver = undefined;
    d.lib = try @import("sys").DynLib.open(if (builtin.os.tag == .windows) "kernel32.dll" else if (builtin.os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
    d.api.cuMemHostAlloc = Fake.alloc;
    d.api.cuMemFreeHost = Fake.free;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copy;
    d.api.cuEventCreate = Fake.create;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventQuery = Fake.query;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    d.api.cuCtxSynchronize = Fake.sync;
    source.setCurrent(@ptrFromInt(41));
    var s: source.Stage = .{};
    @memset(try s.take(&d, 16), 7);
    try s.copyTo(.{ .d = &d, .ptr = 256, .len = 16 }, 0, .{ .d = &d, .handle = null });
    s.abandon();
    const worker = try std.Thread.spawn(.{}, Fake.run, .{&d});
    Fake.mutex.lock();
    const first: u8 = if (completing) 2 else 1;
    while (Fake.phase < first) Fake.cond.wait(&Fake.mutex);
    Fake.mutex.unlock();
    const closer = try std.Thread.spawn(.{}, Fake.close, .{&d});
    Fake.mutex.lock();
    while (!Fake.close_started) Fake.cond.wait(&Fake.mutex);
    while (!Fake.close_waiting and !Fake.close_returned) Fake.cond.wait(&Fake.mutex);
    var counted = true;
    for (first..4) |phase| {
        while (Fake.phase < phase) Fake.cond.wait(&Fake.mutex);
        Fake.premature = Fake.premature or Fake.close_returned;
        Fake.mutex.unlock();
        counted = counted and source.retained() == 1;
        Fake.mutex.lock();
        Fake.allowed = @intCast(phase);
        Fake.cond.broadcast();
    }
    Fake.mutex.unlock();
    worker.join();
    closer.join();
    try t.expect(!Fake.premature);
    try t.expect(counted);
    try t.expect(Fake.close_returned);
    try t.expectEqual(@as(usize, 0), source.retained());
}
