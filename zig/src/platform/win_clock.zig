//! Process, environment and clock calls of the Windows libc surface, re-exported by win_c.zig: UCRT and kernel timers.

const std = @import("std");
const c = @import("win_c.zig");

const HANDLE = *anyopaque;
const BOOL = c_int;
const FILETIME = extern struct { low: u32, high: u32 };

extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.winapi) BOOL;
extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.winapi) BOOL;
extern "kernel32" fn GetSystemTimePreciseAsFileTime(t: *FILETIME) callconv(.winapi) void;
extern "kernel32" fn CreateWaitableTimerExW(security: ?*anyopaque, name: ?[*:0]const u16, flags: u32, access: u32) callconv(.winapi) ?HANDLE;
extern "kernel32" fn SetWaitableTimer(h: HANDLE, due: *const i64, period: i32, routine: ?*anyopaque, arg: ?*anyopaque, resume_: BOOL) callconv(.winapi) BOOL;
extern "kernel32" fn WaitForSingleObject(h: HANDLE, ms: u32) callconv(.winapi) u32;
extern "kernel32" fn Sleep(ms: u32) callconv(.winapi) void;
extern "c" fn _getpid() c_int;
extern "c" fn _putenv_s(name: [*:0]const u8, value: [*:0]const u8) c_int;

pub fn getpid() c.pid_t {
    return _getpid();
}

pub fn getuid() u32 {
    return 0;
}

pub fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int {
    if (overwrite == 0 and c.getenv(name) != null) return 0;
    if (_putenv_s(name, value) != 0) return c.fail(.INVAL);
    return 0;
}

pub fn unsetenv(name: [*:0]const u8) c_int {
    if (_putenv_s(name, "") != 0) return c.fail(.INVAL);
    return 0;
}

threadlocal var sleep_timer: ?HANDLE = null;

/// High-resolution waitable timer per thread, so microsecond waits are not rounded to the 15.6 ms tick.
pub fn nanosleep(req: *const c.timespec, rem: ?*c.timespec) c_int {
    if (rem) |r| r.* = .{ .sec = 0, .nsec = 0 };
    const ns: i64 = req.sec * std.time.ns_per_s + req.nsec;
    if (ns <= 0) return 0;
    if (sleep_timer == null) sleep_timer = CreateWaitableTimerExW(null, null, 0x2, 0x1F0003); // HIGH_RESOLUTION, TIMER_ALL_ACCESS
    if (sleep_timer) |t| {
        const due: i64 = -@divFloor(ns + 99, 100); // relative, in 100 ns units
        if (SetWaitableTimer(t, &due, 0, null, null, 0) != 0) {
            _ = WaitForSingleObject(t, 0xFFFF_FFFF);
            return 0;
        }
    }
    Sleep(@intCast(@max(1, @divFloor(ns + std.time.ns_per_ms - 1, std.time.ns_per_ms))));
    return 0;
}

pub fn clock_gettime(clock: c_int, out: *c.timespec) c_int {
    if (clock == c.CLOCK.MONOTONIC) {
        var count: i64 = 0;
        var freq: i64 = 0;
        _ = QueryPerformanceCounter(&count);
        _ = QueryPerformanceFrequency(&freq);
        out.* = .{ .sec = @divFloor(count, freq), .nsec = @intCast(@divFloor(@mod(count, freq) * std.time.ns_per_s, freq)) };
        return 0;
    }
    var ft: FILETIME = undefined;
    GetSystemTimePreciseAsFileTime(&ft);
    const since_1601: i64 = @intCast((@as(u64, ft.high) << 32) | ft.low);
    const unix = since_1601 - 116_444_736_000_000_000;
    out.* = .{ .sec = @divFloor(unix, 10_000_000), .nsec = @intCast(@mod(unix, 10_000_000) * 100) };
    return 0;
}

pub fn gettimeofday(tv: *c.timeval, tz: ?*anyopaque) c_int {
    _ = tz;
    var ts: c.timespec = undefined;
    _ = clock_gettime(c.CLOCK.REALTIME, &ts);
    tv.* = .{ .sec = ts.sec, .usec = @intCast(@divFloor(ts.nsec, 1000)) };
    return 0;
}
