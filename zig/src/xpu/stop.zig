//! Graceful stop: SIGINT/SIGTERM or /tmp/arc_stop ($ARC_STOP_FILE) make `requested()` true; callers drain and exit.

const std = @import("std");

extern "c" fn signal(sig: c_int, handler: *const fn (c_int) callconv(.c) void) ?*anyopaque;
extern "c" fn access(path: [*:0]const u8, mode: c_int) c_int;

var flag = std.atomic.Value(bool).init(false);
var installed = false;

fn onSignal(_: c_int) callconv(.c) void {
    flag.store(true, .release);
}

/// Installs the SIGINT and SIGTERM handlers (once).
pub fn install() void {
    if (installed) return;
    installed = true;
    _ = signal(2, onSignal);
    _ = signal(15, onSignal);
}

/// True once a stop was asked for by signal or stop file.
pub fn requested() bool {
    if (flag.load(.acquire)) return true;
    const path: [*:0]const u8 = if (std.c.getenv("ARC_STOP_FILE")) |p| p else "/tmp/arc_stop";
    if (access(path, 0) == 0) {
        flag.store(true, .release);
        return true;
    }
    return false;
}
