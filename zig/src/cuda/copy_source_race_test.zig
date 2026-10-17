const std = @import("std");
const builtin = @import("builtin");
const t = std.testing;
const source = @import("copy_source.zig");
const budget = @import("copy_source_budget.zig");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Fake = struct {
    var mutex: std.c.pthread_mutex_t = .{};
    var cond: std.c.pthread_cond_t = .{};
    var phase: u8 = 0;
    var allowed: u8 = 0;
    var close_started = false;
    var close_returned = false;
    var premature = false;
    var bytes: []align(16) u8 = undefined;
    var reaper: *const fn (*const Driver) void = undefined;
    var completing = false;
    fn pause(at: u8) void {
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        phase = at;
        _ = std.c.pthread_cond_broadcast(&cond);
        while (allowed < at) _ = std.c.pthread_cond_wait(&cond, &mutex);
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
    fn run(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
        const d: *Driver = @ptrCast(@alignCast(arg.?));
        source.setCurrent(@ptrFromInt(41));
        if (completing) source.synchronizeContext(d) catch unreachable else reaper(d);
        return null;
    }
    fn close(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
        const d: *Driver = @ptrCast(@alignCast(arg.?));
        _ = std.c.pthread_mutex_lock(&mutex);
        close_started = true;
        _ = std.c.pthread_cond_broadcast(&cond);
        _ = std.c.pthread_mutex_unlock(&mutex);
        d.close();
        _ = std.c.pthread_mutex_lock(&mutex);
        close_returned = true;
        _ = std.c.pthread_cond_broadcast(&cond);
        _ = std.c.pthread_mutex_unlock(&mutex);
        return null;
    }
};
pub fn run(reaper: *const fn (*const Driver) void, completing: bool) !void {
    budget.setForTest(.{ .bytes = 8192, .nodes = 1 });
    defer budget.setForTest(.{});
    Fake.phase = 0;
    Fake.allowed = 0;
    Fake.close_started = false;
    Fake.close_returned = false;
    Fake.premature = false;
    Fake.reaper = reaper;
    Fake.completing = completing;
    var d: Driver = undefined;
    d.lib = try std.DynLib.open(if (builtin.os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
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
    var worker: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&worker, null, Fake.run, &d));
    _ = std.c.pthread_mutex_lock(&Fake.mutex);
    const first: u8 = if (completing) 2 else 1;
    while (Fake.phase < first) _ = std.c.pthread_cond_wait(&Fake.cond, &Fake.mutex);
    _ = std.c.pthread_mutex_unlock(&Fake.mutex);
    var closer: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&closer, null, Fake.close, &d));
    _ = std.c.pthread_mutex_lock(&Fake.mutex);
    while (!Fake.close_started) _ = std.c.pthread_cond_wait(&Fake.cond, &Fake.mutex);
    var deadline: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &deadline);
    deadline.sec += 1;
    while (!Fake.close_returned) {
        if (std.c.pthread_cond_timedwait(&Fake.cond, &Fake.mutex, &deadline) == .TIMEDOUT) break;
    }
    var counted = true;
    for (first..4) |phase| {
        while (Fake.phase < phase) _ = std.c.pthread_cond_wait(&Fake.cond, &Fake.mutex);
        Fake.premature = Fake.premature or Fake.close_returned;
        _ = std.c.pthread_mutex_unlock(&Fake.mutex);
        counted = counted and source.retained() == 1;
        _ = std.c.pthread_mutex_lock(&Fake.mutex);
        Fake.allowed = @intCast(phase);
        _ = std.c.pthread_cond_broadcast(&Fake.cond);
    }
    _ = std.c.pthread_mutex_unlock(&Fake.mutex);
    try t.expectEqual(.SUCCESS, std.c.pthread_join(worker, null));
    try t.expectEqual(.SUCCESS, std.c.pthread_join(closer, null));
    try t.expect(!Fake.premature);
    try t.expect(counted);
    try t.expect(Fake.close_returned);
    try t.expectEqual(@as(usize, 0), source.retained());
}
