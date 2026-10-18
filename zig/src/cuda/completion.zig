//! A destination or parent CUDA object is freed only once its stream, else its context, proves completion; else exit.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Stream = @import("stream.zig").Stream;
const tp_watch = @import("tp_watch.zig");

/// Waits for `s`, then the context, as the copy-source owner counts completion; else the ranks hear `why` and it exits.
pub fn drain(d: *const Driver, s: Stream, why: []const u8) void {
    if (s.synchronize()) |_| return else |_| {}
    if (d.synchronize()) |_| return else |_| {}
    tp_watch.fail(why);
    std.c._exit(1);
}
