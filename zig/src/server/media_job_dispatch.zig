//! Queued job dispatch and independent retry after transient executor saturation.
const std = @import("std");
const media = @import("media_job.zig");
const Registry = media.Registry;

pub fn kick(r: *Registry) void {
    dispatch(r) catch |e| {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        if (r.last_error == null) r.last_error = e;
    };
}

/// Retries a transient Busy independently of later HTTP traffic or completions.
fn scheduleRetry(r: *Registry) !void {
    if (r.closed or r.retry_scheduled) return;
    r.retry_scheduled = true;
    r.retry_threads += 1;
    const thread = std.Thread.spawn(.{}, retryDispatch, .{r}) catch |e| {
        r.retry_scheduled = false;
        r.retry_threads -= 1;
        return e;
    };
    thread.detach();
}

fn retryDispatch(r: *Registry) void {
    std.Io.sleep(r.io, .fromMilliseconds(20), .awake) catch {};
    r.mutex.lockUncancelable(r.io);
    r.retry_scheduled = false;
    const closed = r.closed;
    r.mutex.unlock(r.io);
    if (!closed) kick(r);
    r.mutex.lockUncancelable(r.io);
    r.retry_threads -= 1;
    r.ended.broadcast(r.io);
    r.mutex.unlock(r.io);
}

pub fn dispatch(r: *Registry) !void {
    r.mutex.lockUncancelable(r.io);
    if (r.dispatching or r.closed or r.runner == null) {
        r.mutex.unlock(r.io);
        return;
    }
    r.dispatching = true;
    while (!r.closed and r.running < r.limits.running) {
        var selected: ?*@import("media_job_store.zig").Record = null;
        for (r.jobs.items) |record| if (record.state == .queued) {
            selected = record;
            break;
        };
        const record = selected orelse break;
        const runner = r.runner.?;
        var next = record.*;
        next.state = .running;
        r.store.save(&next) catch |e| {
            r.dispatching = false;
            r.mutex.unlock(r.io);
            return e;
        };
        record.state = .running;
        r.running += 1;
        const invocation: media.Invocation = .{ .registry = r, .serial = record.serial, .id = record.id, .kind = record.kind, .owner = record.owner, .request = record.request };
        r.mutex.unlock(r.io);
        runner.start(runner.ctx, invocation) catch |e| {
            r.mutex.lockUncancelable(r.io);
            r.dispatching = false;
            var current: ?*@import("media_job_store.zig").Record = null;
            for (r.jobs.items) |candidate| if (candidate.serial == invocation.serial) {
                current = candidate;
                break;
            };
            if (current == null or current.?.state != .running) {
                r.mutex.unlock(r.io);
                return;
            }
            switch (e) {
                error.Busy => {
                    if (current.?.cancel_requested) {
                        r.mutex.unlock(r.io);
                        return r.finish(invocation.serial, .cancelled);
                    }
                    var queued = current.?.*;
                    queued.state = .queued;
                    r.store.save(&queued) catch |save_error| {
                        r.mutex.unlock(r.io);
                        return save_error;
                    };
                    current.?.state = .queued;
                    r.running -= 1;
                    scheduleRetry(r) catch |retry_error| {
                        r.mutex.unlock(r.io);
                        return retry_error;
                    };
                    r.mutex.unlock(r.io);
                    return;
                },
                error.Closed => {
                    r.mutex.unlock(r.io);
                    return r.finish(invocation.serial, .{ .failed = "the media executor is closed" });
                },
            }
        };
        r.mutex.lockUncancelable(r.io);
    }
    r.dispatching = false;
    r.mutex.unlock(r.io);
}
