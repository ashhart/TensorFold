//! DSpark command encoding and successful GPU receipts, without overlapping time spans.
const std = @import("std");
const mtl = @import("metal");
const Drafter = @import("dspark.zig").Drafter;

fn failed(msg: [*:0]const u8) anyerror {
    std.log.err("lfm2 dspark: command buffer failed: {s}", .{msg});
    return error.GpuFailed;
}

/// Native monotonic time, independent of the caller's Io implementation and without timer allocation.
pub fn encodeNow() !i128 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(.MONOTONIC, &ts) != 0 or ts.sec < 0 or ts.nsec < 0 or ts.nsec >= std.time.ns_per_s) return error.DraftTimingUnavailable;
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

/// Ends a precommit span exactly once; it never includes commit, GPU execution, waits or pick readback.
pub fn encoded(d: *Drafter, start_ns: i128) !f64 {
    const end_ns = try encodeNow();
    if (end_ns <= start_ns) return error.DraftTimingUnavailable;
    const ms = @as(f64, @floatFromInt(end_ns - start_ns)) / std.time.ns_per_ms;
    d.encode_cpu_ms += ms;
    return ms;
}

/// Commits and waits; CPU is recorded before commit, successful GPU time after the wait, in disjoint spans.
pub fn finish(d: *Drafter, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, start_ns: i128) !f64 {
    e.end();
    const encode_ms = try encoded(d, start_ns);
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| return failed(msg);
    const gpu_ms = cb.gpuSeconds() * 1000;
    if (!(gpu_ms > 0) or !std.math.isFinite(gpu_ms)) return error.DraftTimingUnavailable;
    d.hold_spent_ms += gpu_ms;
    return encode_ms + gpu_ms;
}
