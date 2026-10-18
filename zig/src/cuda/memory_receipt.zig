//! Opt-in process-local allocator boundary receipts; no CUDA calls and no global peak reset unless requested.
const std = @import("std");
const platform = @import("sys");
const memory = @import("memory.zig");

pub const Phase = enum { load_begin, load_end, cleanup_begin, cleanup_end, reset };

pub fn enabled() bool {
    if (@import("builtin").is_test) return false;
    const env = platform.getenv("TF_CUDA_MEMORY_RECEIPTS") orelse return false;
    return std.mem.eql(u8, env, "1");
}

pub fn emit(rank: u32, phase: Phase, reset: bool) void {
    if (!enabled()) return;
    emitUsage(rank, phase, reset, memory.usage(reset));
}

pub fn emitUsage(rank: u32, phase: Phase, reset: bool, usage: memory.Usage) void {
    if (!enabled()) return;
    var ts: platform.c.timespec = undefined;
    const ns: ?u64 = if (platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts) == 0)
        @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec))
    else
        null;
    const receipt: Receipt = .{ .pid = platform.c.getpid(), .rank = rank, .phase = phase, .reset = reset, .monotonic_ns = ns, .usage = usage };
    std.debug.print("TF_MEMORY {f}\n", .{std.json.fmt(receipt, .{})});
}

pub fn snapshot(rank: u32, phase: Phase, reset: bool, monotonic_ns: ?u64) Receipt {
    return .{ .pid = platform.c.getpid(), .rank = rank, .phase = phase, .reset = reset, .monotonic_ns = monotonic_ns, .usage = memory.usage(reset) };
}

pub const Receipt = struct {
    schema: u32 = 1,
    pid: platform.c.pid_t,
    rank: u32,
    phase: Phase,
    reset: bool,
    monotonic_ns: ?u64,
    scope: []const u8 = "process_owned_cuda_buffers",
    usage: memory.Usage,
};

test "rank and reset boundaries survive JSON without physical footprint claims" {
    const a = std.testing.allocator;
    for ([_]u32{ 0, 1 }) |rank| {
        const receipt = snapshot(rank, .load_begin, true, null);
        const text = try std.json.Stringify.valueAlloc(a, receipt, .{});
        defer a.free(text);
        var parsed = try std.json.parseFromSlice(Receipt, a, text, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(rank, parsed.value.rank);
        try std.testing.expect(parsed.value.reset);
        try std.testing.expect(parsed.value.monotonic_ns == null);
        try std.testing.expectEqual(receipt.usage.host, parsed.value.usage.host_peak);
        try std.testing.expectEqual(receipt.usage.device, parsed.value.usage.peak);
    }
}
