//! Prefer foreground and least remaining prefill work, with bounded waiting for every admitted fill.
const LaneHost = @import("lane_host.zig").LaneHost;

pub const preference_rounds = 8;

/// After eight skipped choices, oldest wait wins; each fill runs within eight plus the admitted lane count.
pub fn next(h: *const LaneHost) usize {
    var at: usize = 0;
    for (h.filling.items, 0..) |j, i| {
        const best = h.filling.items[at];
        const aged = j.fill_wait >= preference_rounds;
        const best_aged = best.fill_wait >= preference_rounds;
        if (aged != best_aged) {
            if (aged) at = i;
        } else if (aged) {
            if (j.fill_wait > best.fill_wait) at = i;
        } else if (j.request.background != best.request.background) {
            if (!j.request.background) at = i;
        } else if (remaining(h, j) < remaining(h, best)) at = i;
    }
    for (h.filling.items, 0..) |j, i| j.fill_wait = if (i == at) 0 else j.fill_wait +| 1;
    return at;
}

fn remaining(h: *const LaneHost, j: *const LaneHost.Job) u64 {
    const completed = if (j.fill_began and h.core.backend.vtable.reports_fill)
        @max(j.stream.cached, j.stream.fill_at.load(.monotonic))
    else
        j.stream.cached;
    return j.stream.prompt_len -| completed;
}

/// Takes `job` off the filling list, wherever it is.
pub fn unfill(h: *LaneHost, job: anytype) void {
    for (h.filling.items, 0..) |j, i| if (j == job) {
        _ = h.filling.orderedRemove(i);
        return;
    };
}
