//! Request submission to the lane host: the same queue order and refusal conditions, with bounded reason counts.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const pressure = @import("lane_host_pressure.zig");

pub fn submit(h: *LaneHost, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
    if (@import("lane_prompt_only.zig").refuses(h, request)) {
        h.refused(.invalid_request);
        return error.InvalidRequest;
    }
    const job = h.gpa.create(Job) catch {
        h.refused(.engine_busy);
        return error.Busy;
    };
    job.* = .{ .host = h, .id = id, .request = request, .sink = sink, .submitted = std.Io.Clock.awake.now(h.io).toNanoseconds() };
    h.mutex.lockUncancelable(h.io);
    defer h.unlock();
    const unavailable = h.closing or (if (h.health) |source| !source.read(source.ctx) else false);
    if (unavailable or h.stale or pressure.refuses(h)) {
        h.refusals.note(if (unavailable or h.stale) .engine_unavailable else .memory_pressure);
        h.gpa.destroy(job);
        return if (unavailable) error.Closed else if (h.stale) error.Stale else error.MemoryPressure;
    }
    // foreground before background, each in arrival order (the Python job queue's priority)
    var at = h.queued.items.len;
    while (!request.background and at > 0 and h.queued.items[at - 1].request.background) at -= 1;
    h.queued.insert(h.gpa, at, job) catch {
        h.refusals.note(.engine_busy);
        h.gpa.destroy(job);
        return error.Busy;
    };
    h.load.started += 1;
    h.wake.signal(h.io);
}
