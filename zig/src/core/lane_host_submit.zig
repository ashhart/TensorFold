//! Submit a borrowed request and release superseded untouched warm requests outside the queue lock.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const pressure = @import("lane_host_pressure.zig");
const warm = @import("lane_warm_queue.zig");

pub fn submit(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) api.SubmitError!void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    if (@import("lane_prompt_only.zig").refuses(h, request)) return error.InvalidRequest;
    const job = h.gpa.create(Job) catch return error.Busy;
    job.* = .{ .host = h, .id = id, .request = request, .sink = sink, .submitted = std.Io.Clock.awake.now(h.io).toNanoseconds() };
    // written before the lock: a trace line never holds the host's mutex (a refused request's line stays alone)
    if (h.trace) |t| t.line(.{ .t0_ns = job.submitted, .t1_ns = job.submitted, .kind = .submit, .id = id, .rows_to = request.prompt.len });
    var discarded: warm.Discarded = .{};
    {
        h.mutex.lockUncancelable(h.io);
        defer h.unlock();
        const unavailable = h.closing or (if (h.health) |source| !source.read(source.ctx) else false);
        if (unavailable or h.stale or pressure.refuses(h)) {
            h.gpa.destroy(job);
            return if (unavailable) error.Closed else if (h.stale) error.Stale else error.MemoryPressure;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        while (!request.background and at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.load.started += 1;
        warm.prune(h, job, &discarded);
        h.wake.signal(h.io);
    }
    discarded.finish(h);
}
