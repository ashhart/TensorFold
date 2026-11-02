//! DSpark native timing receipts and calibrated clean-hold repricing.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
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

/// Reprices the clean grid with CPU-encode + GPU ms minus the matching calibrated context estimate.
pub fn priceHold(d: *Drafter, streams: usize, depth: u32, context_rows: u32, ms: f64) void {
    const timed = d.holds[0..d.holds_len];
    if (timed.len == 0 or streams == 0 or !(ms > 0) or !std.math.isFinite(ms)) return;
    const clean_ms = ms - lanes.hold_price.contextMs(d.contexts[0..d.contexts_len], streams, context_rows);
    if (!(clean_ms > 0) or !std.math.isFinite(clean_ms)) return;
    const priced = lanes.hold_price.holdMs(timed, streams, depth);
    if (!(priced > 0) or !std.math.isFinite(priced)) return;
    const step = 1 + 0.1 * (clean_ms / priced - 1);
    if (!std.math.isFinite(step)) return;
    var lo: u32 = 0;
    var hi: u32 = std.math.maxInt(u32);
    for (timed) |h| {
        if (h.streams <= streams) lo = @max(lo, h.streams);
        if (h.streams >= streams) hi = @min(hi, h.streams);
    }
    for (timed) |*h| if (h.streams >= lo and h.streams <= hi) {
        h.ms *= step;
    };
}
