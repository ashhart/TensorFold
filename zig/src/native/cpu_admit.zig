//! The CPU server's memory readings and stream admission: RAM and the process footprint on each OS, failing closed.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");

/// The share of RAM a served model may hold; the rest stays with the OS and other programs (the Metal share).
pub const SHARE_PERCENT: u64 = 70;
/// Kept back for buffers that grow with long prompts.
pub const MARGIN: u64 = 2 << 30;

/// Physical RAM in bytes, or null when the OS does not say.
pub fn ram() ?u64 {
    return switch (builtin.os.tag) {
        .macos => sysctl("hw.memsize"),
        .linux => meminfo("MemTotal:"),
        .windows => windowsRam(),
        else => null,
    };
}

/// The process's resident footprint in bytes, or null when the OS does not say.
pub fn footprint() ?u64 {
    return switch (builtin.os.tag) {
        .macos => @import("sys").memory_policy.footprint(),
        .linux => statm(),
        .windows => windowsFootprint(),
        else => null,
    };
}

var peak: std.atomic.Value(u64) = .init(0);

/// The counts `/health.memory` shows: the footprint now and its peak since the last reset (read from HTTP threads).
pub fn memory(reset_peak: bool) ?api.Memory {
    const active = footprint() orelse return null;
    if (reset_peak) peak.store(active, .monotonic) else _ = peak.fetchMax(active, .monotonic);
    return .{ .active = active, .cache = 0, .peak = peak.load(.monotonic), .pinned = 0, .limit = null };
}

/// Both readings, or null when either is missing: admission then fails closed to one stream and no prompt cache.
pub const Readings = struct { total: u64, used: u64 };
pub fn readings() ?Readings {
    return .{ .total = ram() orelse return null, .used = footprint() orelse return null };
}

/// Streams of `stream_bytes` each that fit beside the loaded model: the share of RAM less the footprint and margin.
pub fn fits(total: u64, used: u64, stream_bytes: u64) u32 {
    const limit = total / 100 * SHARE_PERCENT;
    const room = limit -| used -| MARGIN;
    return @intCast(@min(room / @max(stream_bytes, 1), std.math.maxInt(u32)));
}

/// The streams to serve: what fits, or the named --parallel when it fits (null with `problem`); no reading: one stream.
pub fn streams(a: std.mem.Allocator, family: []const u8, stream_bytes: u64, o: api.Open, window: u64, problem: *[]const u8) !?u32 {
    return streamsFrom(a, family, stream_bytes, o, window, readings(), problem);
}

pub fn streamsFrom(a: std.mem.Allocator, family: []const u8, stream_bytes: u64, o: api.Open, window: u64, read: ?Readings, problem: *[]const u8) !?u32 {
    const r = read orelse {
        if (o.lanes_fixed and o.lanes > 1) {
            problem.* = try std.fmt.allocPrint(a, "{s}: this host's RAM or this process's footprint could not be read, so --parallel {d} cannot be admitted; serve with --parallel 1", .{ family, o.lanes });
            return null;
        }
        std.log.warn("{s}: this host's RAM or this process's footprint could not be read; serving one stream", .{family});
        return 1;
    };
    const total = r.total;
    const used = r.used;
    const fit = fits(total, used, stream_bytes);
    if (fit == 0) {
        problem.* = try std.fmt.allocPrint(a, "{s}: one stream's {d}-token window ({d:.2} GiB) does not fit the {d:.2} GiB left of {d}% of this host's RAM; lower --context", .{ family, window, gib(stream_bytes), gib((total / 100 * SHARE_PERCENT) -| used -| MARGIN), SHARE_PERCENT });
        return null;
    }
    if (o.lanes_fixed and o.lanes > fit) {
        problem.* = try std.fmt.allocPrint(a, "{s}: --parallel {d} does not fit beside the model; {d} fit: serve fewer, or lower --context", .{ family, o.lanes, fit });
        return null;
    }
    return @min(o.lanes, fit);
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn sysctl(name: [*:0]const u8) ?u64 {
    if (builtin.os.tag != .macos) return null;
    var v: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname(name, &v, &len, null, 0) != 0 or len != @sizeOf(u64)) return null;
    return v;
}

/// A /proc/meminfo line's kB value in bytes.
fn meminfo(key: []const u8) ?u64 {
    if (builtin.os.tag != .linux) return null;
    var buf: [8192]u8 = undefined;
    const text = readFile("/proc/meminfo", &buf) orelse return null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        var it = std.mem.tokenizeScalar(u8, line[key.len..], ' ');
        const digits = it.next() orelse return null;
        return (std.fmt.parseInt(u64, digits, 10) catch return null) * 1024;
    }
    return null;
}

/// /proc/self/statm's resident pages times the page size.
fn statm() ?u64 {
    if (builtin.os.tag != .linux) return null;
    var buf: [256]u8 = undefined;
    const text = readFile("/proc/self/statm", &buf) orelse return null;
    var it = std.mem.tokenizeScalar(u8, text, ' ');
    _ = it.next() orelse return null;
    const pages = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
    return pages * std.heap.pageSize();
}

/// The first `buf.len` bytes of a file through libc, for the /proc readers (null when it cannot be opened or read).
pub fn readFile(path: [*:0]const u8, buf: []u8) ?[]const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n <= 0) return null;
    return buf[0..@intCast(n)];
}

const MemoryStatusEx = extern struct { length: u32, load: u32, total_phys: u64, avail_phys: u64, total_page_file: u64, avail_page_file: u64, total_virtual: u64, avail_virtual: u64, avail_extended_virtual: u64 };
const ProcessMemoryCounters = extern struct { cb: u32, page_fault_count: u32, peak_working_set: usize, working_set: usize, quota_peak_paged_pool: usize, quota_paged_pool: usize, quota_peak_non_paged_pool: usize, quota_non_paged_pool: usize, pagefile_usage: usize, peak_pagefile_usage: usize };
extern "kernel32" fn GlobalMemoryStatusEx(buffer: *MemoryStatusEx) callconv(.winapi) c_int;
extern "kernel32" fn GetCurrentProcess() callconv(.winapi) ?*anyopaque;
extern "kernel32" fn K32GetProcessMemoryInfo(process: ?*anyopaque, counters: *ProcessMemoryCounters, cb: u32) callconv(.winapi) c_int;

fn windowsRam() ?u64 {
    if (builtin.os.tag != .windows) return null;
    var status: MemoryStatusEx = std.mem.zeroes(MemoryStatusEx);
    status.length = @sizeOf(MemoryStatusEx);
    if (GlobalMemoryStatusEx(&status) == 0) return null;
    return status.total_phys;
}

fn windowsFootprint() ?u64 {
    if (builtin.os.tag != .windows) return null;
    var counters: ProcessMemoryCounters = std.mem.zeroes(ProcessMemoryCounters);
    counters.cb = @sizeOf(ProcessMemoryCounters);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, counters.cb) == 0) return null;
    return counters.working_set;
}

test "what fits: the share less the footprint and margin, in whole streams" {
    try std.testing.expectEqual(@as(u32, 0), fits(16 << 30, 10 << 30, 1 << 30));
    try std.testing.expectEqual(@as(u32, 7), fits(16 << 30, 2 << 30, 1 << 30));
    try std.testing.expectEqual(@as(u32, 27), fits(48 << 30, 4 << 30, 1 << 30));
}

test "admission fails closed without a reading: one stream, an explicit --parallel above it refused" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expectEqual(@as(?u32, 1), try streamsFrom(a, "qwen3.5", 1 << 20, .{ .dir = "", .model_type = "qwen3_5", .lanes = 8 }, 8, null, &problem));
    const refused = try streamsFrom(a, "qwen3.5", 1 << 20, .{ .dir = "", .model_type = "qwen3_5", .lanes = 2, .lanes_fixed = true }, 8, null, &problem);
    defer a.free(problem);
    try std.testing.expectEqual(@as(?u32, null), refused);
    try std.testing.expect(std.mem.indexOf(u8, problem, "could not be read") != null);
    try std.testing.expectEqual(@as(?u32, 7), try streamsFrom(a, "qwen3.5", 1 << 30, .{ .dir = "", .model_type = "qwen3_5", .lanes = 8 }, 8, .{ .total = 16 << 30, .used = 2 << 30 }, &problem));
}

test "admission: the named --parallel is refused when it does not fit, served when it does" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    const total = ram() orelse return error.SkipZigTest;
    const used = footprint() orelse 0;
    const fit = fits(total, used, 1 << 20);
    if (fit == 0) return error.SkipZigTest;
    const served = try streams(a, "qwen3.5", 1 << 20, .{ .dir = "", .model_type = "qwen3_5", .lanes = 2, .lanes_fixed = true }, 8, &problem);
    try std.testing.expectEqual(@as(?u32, 2), served);
    const refused = try streams(a, "qwen3.5", 1 << 20, .{ .dir = "", .model_type = "qwen3_5", .lanes = fit + 1, .lanes_fixed = true }, 8, &problem);
    defer a.free(problem);
    try std.testing.expectEqual(@as(?u32, null), refused);
    try std.testing.expect(std.mem.indexOf(u8, problem, "--parallel") != null);
}

test "an unreadable window refusal names the window and the share" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    const total = ram() orelse return error.SkipZigTest;
    const refused = try streams(a, "qwen3.5", total * 2, .{ .dir = "", .model_type = "qwen3_5" }, 8192, &problem);
    defer a.free(problem);
    try std.testing.expectEqual(@as(?u32, null), refused);
    try std.testing.expect(std.mem.indexOf(u8, problem, "lower --context") != null);
}
