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
        close_waiting: bool = false, // keepDriver reached its wait while the reaper held the stage
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
    fn waiting() void {
        set(&s.close_waiting);
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
    source.on_close_wait = Fake.waiting;
    defer source.on_close_wait = null;
    var worker: ?std.c.pthread_t = null;
    var closer: ?std.c.pthread_t = null;
    defer { // a failed check still lets the query return and both threads finish before the test does
        Fake.set(&Fake.s.go);
        if (closer) |c| _ = std.c.pthread_join(c, null);
        if (worker) |w| _ = std.c.pthread_join(w, null);
    }
    var w: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&w, null, Fake.reaper, &d));
    worker = w;
    Fake.until(&Fake.s.in_query);
    var c: std.c.pthread_t = undefined;
    try t.expectEqual(.SUCCESS, std.c.pthread_create(&c, null, Fake.close, &d));
    closer = c;
    Fake.until(&Fake.s.close_waiting);
    _ = std.c.pthread_mutex_lock(&Fake.mutex);
    const early = Fake.s.close_returned;
    _ = std.c.pthread_mutex_unlock(&Fake.mutex);
    Fake.set(&Fake.s.go);
    Fake.until(&Fake.s.close_returned);
    try t.expect(!early);
    try t.expectEqual(@as(usize, 0), source.spareCount());
    try t.expect(Fake.s.freed and Fake.s.destroyed);
    try t.expect(!Fake.s.late);
    try t.expectEqual(@as(usize, 0), source.retained());
}
