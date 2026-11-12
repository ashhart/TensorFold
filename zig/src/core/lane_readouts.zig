//! Complete a filled readout prompt: the backend's logits into the request, then its state released once.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

/// The BF16 quiet NaN the logits hold until the backend writes them: an unwritten logit fails the finite check.
pub const unwritten: u16 = 0x7fc0;

/// `ok` (the trace's enclosing fill line) is set false when the completion fails, like the completion's own line.
pub fn complete(h: *LaneHost, job: *LaneHost.Job, ok: ?*bool) bool {
    const spec = job.request.readout.?;
    const t0 = if (h.trace) |t| t.now() else 0;
    const backend = h.core.backend;
    var failure: ?anyerror = null;
    if (job.stream.isCancelled()) {
        failure = error.Cancelled;
    } else {
        @memset(spec.logits, unwritten);
        if (backend.vtable.readout_finish) |run| {
            run(backend.ptr, &job.stream, spec.logits) catch |err| {
                failure = err;
            };
        } else failure = error.NoReadoutBackend;
        if (failure == null) for (spec.logits) |bits| if (!std.math.isFinite(widen(bits))) {
            failure = error.NonFiniteReadout;
            break;
        };
        if (failure == null and job.stream.isCancelled()) failure = error.Cancelled;
    }
    job.stream.finished = true;
    job.stream.reason = if (failure == null) .length else if (failure.? == error.Cancelled) .cancelled else .@"error";
    backend.release(&job.stream);
    if (ok) |o| o.* = o.* and failure == null;
    if (h.trace) |t| t.line(.{ .t0_ns = t0, .t1_ns = t.now(), .kind = .complete, .id = job.id, .stream = job.seq, .rows_from = job.stream.prompt_len, .rows_to = job.stream.prompt_len, .ok = failure == null });
    if (failure) |err| return if (err == error.Cancelled) h.cancel(job) else @import("lane_host_end.zig").dropFailed(h, job, err);
    h.remove(job);
    h.finish(job, .length, "");
    return true;
}

/// A BF16 value's bits as the FP32 it widens to exactly.
pub fn widen(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
