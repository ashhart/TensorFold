//! Prompt progress for ``return_progress`` requests: a fill's rows after each call that leaves it open, then its end.
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

/// After a prompt call that left the pass open: its rows so far, sent when they pass the last count sent.
pub fn chunk(h: *const LaneHost, job: *Job) void {
    if (!job.request.return_progress or !h.core.backend.vtable.reports_fill) return; // no `fill_at`: no honest count
    const s = &job.stream;
    const total: u32 = @intCast(s.prompt_len);
    const done = @min(@max(s.cached, s.fill_at.load(.monotonic)), total);
    if (done <= job.progress_rows) return; // a call that moved nothing, or a requeued pass: never a count back
    job.progress_rows = done;
    LaneHost.emit(job, .{ .progress = .{ .done = done, .total = total, .cached = s.cached } });
}

/// The pass reported: a fill that sent progress ends at its whole prompt, before the prefilled event and any token.
pub fn filled(job: *Job) void {
    const total: u32 = @intCast(job.stream.prompt_len);
    if (job.progress_rows == 0 or job.progress_rows >= total) return; // filled in one call, or its end already sent
    job.progress_rows = total;
    LaneHost.emit(job, .{ .progress = .{ .done = total, .total = total, .cached = job.stream.cached } });
}
