//! A backend observes owned engine work without holding an idle wait open as an operation.
pub const Activity = struct {
    ctx: *anyopaque,
    begin: *const fn (*anyopaque) void,
    end: *const fn (*anyopaque) void,

    pub const Scope = struct {
        activity: ?Activity,
        pub fn end(s: *Scope) void {
            if (s.activity) |a| a.end(a.ctx);
            s.activity = null;
        }
    };
    pub fn enter(activity: ?Activity) Scope {
        if (activity) |a| a.begin(a.ctx);
        return .{ .activity = activity };
    }
};

/// A family's optional weight owner: submission restores before queueing, and the idle engine may release.
pub const WeightRelease = struct {
    ctx: *anyopaque,
    restore: *const fn (*anyopaque) anyerror!void,
    idle: *const fn (*anyopaque) void,
};

/// Restore before a submission joins the queue; destroy its unused job and refuse if exact bytes cannot return.
pub fn restore(h: anytype, job: anytype) error{Busy}!void {
    if (h.weight_release) |r| r.restore(r.ctx) catch |err| {
        @import("std").log.warn("lane host: weight restoration refused admission ({s})", .{@errorName(err)});
        h.gpa.destroy(job);
        return error.Busy;
    };
}

/// Finish an owned work activity, release only at true idle, then give learned rows one idle turn.
pub fn idleRows(h: anytype, activity: *Activity.Scope) bool {
    activity.end();
    idle(h);
    return @import("lane_host_rows.zig").idle(h);
}

/// Release only an entirely idle host; callers hold the host lock so submission cannot slip into this episode.
pub fn idle(h: anytype) void {
    if (h.closing.load(.acquire)) return;
    const work = h.queued.items.len != 0 or h.admitted.items.len != 0 or h.filling.items.len != 0 or h.cancels.items.len != 0 or h.core.activeCount() != 0;
    if (work) return;
    if (h.weight_release) |r| r.idle(r.ctx);
}

/// A backend hears the error that failed a round or request; the message each stream receives does not depend on it.
pub const Failure = struct {
    ctx: *anyopaque,
    call: *const fn (*anyopaque, anyerror) void,
};
