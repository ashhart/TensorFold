//! Checked host conversion of supplied OS counters; this module never calls an OS probe.
const std = @import("std");
const builtin = @import("builtin");
const fit = @import("load_fit.zig");

/// The sysctl adapter must retain the returned width; failed and partial reads never become a counter.
pub fn counter(bytes: []const u8) !u64 {
    return switch (bytes.len) {
        4 => std.mem.readInt(u32, bytes[0..4], builtin.cpu.arch.endian()),
        8 => std.mem.readInt(u64, bytes[0..8], builtin.cpu.arch.endian()),
        else => error.BadMemoryReading,
    };
}

pub const Input = struct {
    ram: ?u64,
    working_set: ?u64,
    footprint: ?u64,
    free_pages: ?u64,
    page_size: ?u64,
    load_limit: u64,
    read_floor: u64,
    started_ns: u64,
    finished_ns: u64,
};

/// Both the probe duration and the age at use are bounded by the same caller-selected monotonic interval.
pub fn snapshot(input: Input, now_ns: u64, max_age_ns: u64) !fit.Snapshot {
    if (max_age_ns == 0 or input.finished_ns < input.started_ns or now_ns < input.finished_ns)
        return error.BadMemoryReading;
    if (input.finished_ns - input.started_ns > max_age_ns or now_ns - input.started_ns > max_age_ns)
        return error.StaleMemoryReading;
    const pages = input.free_pages orelse return error.NoMemoryReading;
    const page = input.page_size orelse return error.NoMemoryReading;
    if (page == 0 or !std.math.isPowerOfTwo(page)) return error.BadMemoryReading;
    const free = try std.math.mul(u64, pages, page);
    const value = fit.Snapshot{ .ram = input.ram, .working_set = input.working_set, .footprint = input.footprint, .available = free, .load_limit = input.load_limit, .read_floor = input.read_floor };
    _ = try fit.Room.from(value);
    return value;
}

fn sample() Input {
    return .{ .ram = 128 << 30, .working_set = 100 << 30, .footprint = 10 << 30, .free_pages = 1 << 20, .page_size = 16384, .load_limit = 90 << 30, .read_floor = 8 << 30, .started_ns = 10, .finished_ns = 12 };
}

test "fit OS counters accept only complete four or eight byte native-endian values" {
    const four: u32 = 12345;
    const eight: u64 = 0x100000001;
    try std.testing.expectEqual(@as(u64, four), try counter(std.mem.asBytes(&four)));
    try std.testing.expectEqual(eight, try counter(std.mem.asBytes(&eight)));
    try std.testing.expectError(error.BadMemoryReading, counter(&.{}));
    try std.testing.expectError(error.BadMemoryReading, counter(&.{ 1, 2, 3 }));
    try std.testing.expectError(error.BadMemoryReading, counter(&.{ 1, 2, 3, 4, 5 }));
}

test "fit reading converts pages once and keeps the family limit and footprint" {
    const s = try snapshot(sample(), 15, 5);
    try std.testing.expectEqual(@as(?u64, 16 << 30), s.available);
    try std.testing.expectEqual(@as(u64, 90 << 30), s.load_limit);
    try std.testing.expectEqual(@as(u64, 8 << 30), (try fit.Room.from(s)).additional);
}

test "fit reading rejects failed probes, invalid units and multiplication overflow" {
    var input = sample();
    input.free_pages = null;
    try std.testing.expectError(error.NoMemoryReading, snapshot(input, 15, 5));
    input = sample();
    input.page_size = 3;
    try std.testing.expectError(error.BadMemoryReading, snapshot(input, 15, 5));
    input.page_size = 0;
    try std.testing.expectError(error.BadMemoryReading, snapshot(input, 15, 5));
    input.page_size = 16384;
    input.free_pages = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, snapshot(input, 15, 5));
    input = sample();
    input.ram = null;
    try std.testing.expectError(error.NoMemoryReading, snapshot(input, 15, 5));
}

test "fit reading refuses stale, slow and reversed clocks rather than reusing free memory" {
    const input = sample();
    try std.testing.expectError(error.StaleMemoryReading, snapshot(input, 16, 5));
    try std.testing.expectError(error.StaleMemoryReading, snapshot(input, 15, 1));
    try std.testing.expectError(error.BadMemoryReading, snapshot(input, 11, 5));
    try std.testing.expectError(error.BadMemoryReading, snapshot(input, 15, 0));
    var reversed = input;
    reversed.finished_ns = 9;
    try std.testing.expectError(error.BadMemoryReading, snapshot(reversed, 15, 5));
}
