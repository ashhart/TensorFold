const std = @import("std");
const builtin = @import("builtin");
const t = std.testing;
const abi = @import("abi.zig");
const source = @import("copy_source.zig");
const Driver = @import("driver.zig").Driver;

/// A reaper held inside its event query while the driver closes; the stage it then parks must go before the library.
const Fake = struct {
    var mutex: std.c.pthread_mutex_t = .{};
    var cond: std.c.pthread_cond_t = .{};
    const State = struct {
        in_query: bool = false,
        go: bool = false,
        close_started: bool = false,
        close_returned: bool = false,
        freed: bool = false,
        destroyed: bool = false,
        late: bool = false, // the stage's bytes or event went after close had returned
    };
    var s: State = .{};
    var bytes: []align(16) u8 = undefined;
    var reap: *const fn (*const Driver) void = undefined;
    fn set(flag: *bool) void {
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        flag.* = true;
        s.late = s.late or ((flag == &s.freed or flag == &s.destroyed) and s.close_returned);
        _ = std.c.pthread_cond_broadcast(&cond);
    }
    fn until(flag: *const bool) void {
        _ = std.c.pthread_mutex_lock(&mutex);
        defer _ = std.c.pthread_mutex_unlock(&mutex);
        while (!flag.*) _ = std.c.pthread_cond_wait(&cond, &mutex);
    }
    fn alloc(out: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        bytes = std.heap.page_allocator.alignedAlloc(u8, .@"16", len) catch return abi.error_not_ready;
        out.* = bytes.ptr;
        return 0;
    }
    fn free(_: ?*anyopaque) callconv(.c) abi.Result {
        std.heap.page_allocator.free(bytes);
        set(&s.freed);
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
        set(&s.in_query);
        until(&s.go);
        return 0;
    }
    fn destroy(_: abi.Event) callconv(.c) abi.Result {
        set(&s.destroyed);
        return 0;
    }
    fn reaper(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
        const d: *Driver = @ptrCast(@alignCast(arg.?));
        source.setCurrent(@ptrFromInt(41));
        reap(d);
        return null;
    }
    fn close(arg: ?*anyopaque) callconv(.c) ?*anyopaque {
        const d: *Driver = @ptrCast(@alignCast(arg.?));
        set(&s.close_started);
        d.close();
        set(&s.close_returned);
        return null;
    }
};

pub fn run(reap: *const fn (*const Driver) void) !void {
    source.setSparesForTest(.{ .count = 4, .bytes = 1 << 20 });
    defer source.setSparesForTest(.{ .count = 0, .bytes = 0 });
    Fake.s = .{};
    Fake.reap = reap;
    var d: Driver = undefined;
    d.lib = try std.DynLib.open(if (builtin.os.tag == .macos) "/usr/lib/libSystem.B.dylib" else "libc.so.6");
    d.api.cuMemHostAlloc = Fake.alloc;
    d.api.cuMemFreeHost = Fake.free;
    d.api.cuMemcpyHtoDAsync_v2 = Fake.copy;
    d.api.cuEventCreate = Fake.create;
    d.api.cuEventRecord = Fake.record;
    d.api.cuEventQuery = Fake.query;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    source.setCurrent(@ptrFromInt(41));
    defer source.dropSpares(&d, @ptrFromInt(41)); // a spare close left behind, so the next test starts empty
    try source.enqueue(&d, 256, &@as([16]u8, @splat(7)), .{ .d = &d, .handle = null });
    try t.expectEqual(@as(usize, 1), source.retained());
    var worker: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&worker, null, Fake.reaper, &d));
    Fake.until(&Fake.s.in_query);
    var closer: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&closer, null, Fake.close, &d));
    Fake.until(&Fake.s.close_started);
    _ = std.c.pthread_mutex_lock(&Fake.mutex);
    var deadline: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.REALTIME, &deadline);
    deadline.nsec += 300 * std.time.ns_per_ms; // close gets well into its wait on the busy stage
    if (deadline.nsec >= std.time.ns_per_s) {
        deadline.sec += 1;
        deadline.nsec -= std.time.ns_per_s;
    }
    while (!Fake.s.close_returned) {
        if (std.c.pthread_cond_timedwait(&Fake.cond, &Fake.mutex, &deadline) == .TIMEDOUT) break;
    }
    const early = Fake.s.close_returned;
    Fake.s.go = true;
    _ = std.c.pthread_cond_broadcast(&Fake.cond);
    _ = std.c.pthread_mutex_unlock(&Fake.mutex);
    try t.expectEqual(.SUCCESS, std.c.pthread_join(worker, null));
    try t.expectEqual(.SUCCESS, std.c.pthread_join(closer, null));
    try t.expect(!early);
    try t.expectEqual(@as(usize, 0), source.spareCount());
    try t.expect(Fake.s.freed and Fake.s.destroyed);
    try t.expect(!Fake.s.late);
    try t.expectEqual(@as(usize, 0), source.retained());
}
