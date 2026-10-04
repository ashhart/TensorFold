//! Monotonic time for a reply's timings, in nanoseconds and seconds.
const std = @import("std");

pub fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}
