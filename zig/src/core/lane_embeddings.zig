//! Complete a stepped embedding prompt before releasing its backend state and the caller's request lifetime.
const std = @import("std");
const LaneHost = @import("lane_host.zig").LaneHost;

pub fn complete(h: *LaneHost, job: *LaneHost.Job) bool {
    const batch = job.request.embedding.?;
    const backend = h.core.backend;
    var failure: ?anyerror = null;
    if (job.stream.isCancelled()) {
        failure = error.Cancelled;
    } else {
        @memset(batch.output, std.math.nan(f32));
        if (backend.vtable.embedding) |run| {
            run(backend.ptr, &job.stream, batch.output) catch |err| {
                failure = err;
            };
        } else failure = error.NoEmbeddingBackend;
        if (failure == null) for (batch.output) |value| if (!std.math.isFinite(value)) {
            failure = error.NonFiniteEmbedding;
            break;
        };
        if (failure == null and job.stream.isCancelled()) failure = error.Cancelled;
    }
    job.stream.finished = true;
    job.stream.reason = if (failure == null) .length else if (failure.? == error.Cancelled) .cancelled else .@"error";
    backend.release(&job.stream);
    if (failure) |err| return if (err == error.Cancelled) h.cancel(job) else h.drop(job, h.words(err));
    h.remove(job);
    h.finish(job, .length, "");
    return true;
}
