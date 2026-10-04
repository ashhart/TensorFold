//! Host time in seconds on the clock Metal's GPUStartTime and GPUEndTime use (mach_absolute_time).
const std = @import("std");

var scale: f64 = 0;

pub fn seconds() f64 {
    if (scale == 0) {
        var info: std.c.mach_timebase_info_data = undefined;
        _ = std.c.mach_timebase_info(&info);
        scale = @as(f64, @floatFromInt(info.numer)) / @as(f64, @floatFromInt(info.denom)) / 1e9;
    }
    return @as(f64, @floatFromInt(std.c.mach_absolute_time())) * scale;
}
