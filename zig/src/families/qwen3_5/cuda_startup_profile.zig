//! Disjoint awake-clock startup phases; calibration completes during open before the lane thread starts.
const std = @import("std");

pub const Times = struct {
    load: i96 = 0,
    reserve: i96 = 0,
    warmups: i96 = 0,
    calibration: i96 = 0,

    pub fn add(t: *Times, other: Times) void {
        inline for (@typeInfo(Times).@"struct".field_names) |name| @field(t, name) += @field(other, name);
    }
    pub fn line(t: Times, buf: []u8) ![]u8 {
        return std.fmt.bufPrint(buf, "CUDA family startup phases: load={d:.3}s reserve={d:.3}s warmups={d:.3}s calibration={d:.3}s", .{ seconds(t.load), seconds(t.reserve), seconds(t.warmups), seconds(t.calibration) });
    }
    pub fn print(t: Times) void {
        var buf: [256]u8 = undefined;
        std.debug.print("{s}\n", .{t.line(&buf) catch unreachable});
    }
};

/// The base target's temporary cache and its warm forwards, with cleanup on a failed warm.
pub fn withCache(gpa: std.mem.Allocator, io: std.Io, e: anytype, comptime Cache: type) !Times {
    const started = now(io);
    var cache = try Cache.init(gpa, e, 64);
    errdefer cache.deinit(gpa);
    var phases: Times = .{ .reserve = elapsed(io, started) };
    const warming = now(io);
    try e.warmCache(&cache);
    phases.warmups = elapsed(io, warming);
    const freeing = now(io);
    cache.deinit(gpa);
    phases.reserve += elapsed(io, freeing);
    return phases;
}

pub fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}
pub fn elapsed(io: std.Io, start: i96) i96 {
    return @max(0, now(io) - start);
}
fn seconds(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

test "startup phases: one line keeps separately accumulated seconds with three decimals" {
    var times: Times = .{ .load = 1_234_000_000, .reserve = 2_345_000_000 };
    times.add(.{ .load = 1_000_000_000, .warmups = 3_456_000_000, .calibration = 4_567_000_000 });
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("CUDA family startup phases: load=2.234s reserve=2.345s warmups=3.456s calibration=4.567s", try times.line(&buf));
}

pub const Test = if (@import("builtin").is_test) @import("cuda_startup_profile_test.zig") else struct {};
