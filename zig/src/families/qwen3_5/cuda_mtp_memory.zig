//! Peak device-memory admission for loading and calibrating Qwen's CUDA MTP drafter.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const weights = @import("cuda_mtp_weights.zig");
const types = @import("cuda_mtp_types.zig");

pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub fn check(gpa: std.mem.Allocator, io: std.Io, g: geometry.Geometry, dir: []const u8, scratch_bytes: usize, ring_bytes: usize, calibration_cache_bytes: usize, room: ?Room) !void {
    const plan = try weights.allocationPlan(gpa, io, g, dir);
    const peak = try peakBytes(plan.resident, plan.staging, scratch_bytes, ring_bytes, calibration_cache_bytes, g.hidden, types.max_lanes);
    if (room) |r| {
        const available = r.read(r.ctx) orelse return error.MtpBudgetUnavailable;
        const need = std.math.cast(u64, peak) orelse return error.MtpMemorySizeOverflow;
        if (need > available) {
            std.log.err("qwen MTP startup needs {d:.2} GiB at its peak; {d:.2} GiB remains after the CUDA reserve", .{ @as(f64, @floatFromInt(need)) / (1 << 30), @as(f64, @floatFromInt(available)) / (1 << 30) });
            return error.MtpDoesNotFit;
        }
    }
}

fn peakBytes(resident_weights: usize, staging_weights: usize, scratch: usize, ring: usize, calibration_cache: usize, hidden: usize, lanes: usize) !usize {
    const hidden_bytes = std.math.mul(usize, hidden, 4) catch return error.MtpMemorySizeOverflow;
    const held_bytes = std.math.mul(usize, lanes, 4) catch return error.MtpMemorySizeOverflow;
    const ctx_hidden = std.math.add(usize, ring, hidden_bytes) catch return error.MtpMemorySizeOverflow;
    const ctx_bytes = std.math.add(usize, ctx_hidden, held_bytes) catch return error.MtpMemorySizeOverflow;
    const resident_state = std.math.add(usize, scratch, ctx_bytes) catch return error.MtpMemorySizeOverflow;
    const resident_peak = std.math.add(usize, resident_weights, resident_state) catch return error.MtpMemorySizeOverflow;
    const loading_peak = std.math.add(usize, resident_weights, staging_weights) catch return error.MtpMemorySizeOverflow;
    // Calibration follows warm-up release: target cache coexists with MTP weights and scratch, not warm-up context.
    const calibration_state = std.math.add(usize, scratch, calibration_cache) catch return error.MtpMemorySizeOverflow;
    const calibration_peak = std.math.add(usize, resident_weights, calibration_state) catch return error.MtpMemorySizeOverflow;
    return @max(@max(resident_peak, loading_peak), calibration_peak);
}

test "MTP startup peak counts resident state, conversion staging, and target calibration cache" {
    try std.testing.expectEqual(@as(usize, 158), try peakBytes(100, 40, 20, 10, 30, 5, 2));
    try std.testing.expectEqual(@as(usize, 300), try peakBytes(100, 200, 10, 10, 20, 1, 1));
    try std.testing.expectEqual(@as(usize, 230), try peakBytes(100, 20, 10, 10, 120, 1, 1));
}

test "MTP startup peak rejects allocation-size overflow" {
    try std.testing.expectError(error.MtpMemorySizeOverflow, peakBytes(std.math.maxInt(usize), 1, 0, 0, 0, 0, 0));
    try std.testing.expectError(error.MtpMemorySizeOverflow, peakBytes(0, 0, 0, 0, 0, std.math.maxInt(usize), 1));
    try std.testing.expectError(error.MtpMemorySizeOverflow, peakBytes(0, 0, 1, 0, std.math.maxInt(usize), 0, 0));
}
