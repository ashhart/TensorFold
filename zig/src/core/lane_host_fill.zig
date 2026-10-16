//! Which waiting prompt a lane host fills next: foreground before background, then the fewest prompt tokens to run.
const LaneHost = @import("lane_host.zig").LaneHost;

/// The filling job to give the next chunk (the oldest of equals), so a short prompt never waits on a long one.
pub fn next(h: *const LaneHost) usize {
    var at: usize = 0;
    for (h.filling.items, 0..) |j, i| {
        const best = h.filling.items[at];
        if (j.request.background != best.request.background) {
            if (!j.request.background) at = i;
        } else if (j.stream.prompt_len - j.stream.cached < best.stream.prompt_len - best.stream.cached) at = i;
    }
    return at;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}
