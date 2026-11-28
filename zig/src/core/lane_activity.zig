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
    restoreWork(h) catch {
        h.gpa.destroy(job);
        return error.Busy;
    };
}

/// Restore before any queued work that reads weights, including scoring and folded-memory compilation.
pub fn restoreWork(h: anytype) error{Busy}!void {
    if (h.weight_release) |r| r.restore(r.ctx) catch |err| {
        @import("std").log.warn("lane host: weight restoration refused admission ({s})", .{@errorName(err)});
        return error.Busy;
    };
}

/// Finish an owned work activity, release only at true idle, then give learned rows one idle turn.
pub fn idleRows(h: anytype, activity: anytype) bool {
    activity.end();
    idle(h);
    return @import("lane_host_rows.zig").idle(h);
}

/// Release only an entirely idle host; callers hold the host lock so submission cannot slip into this episode.
pub fn idle(h: anytype) void {
    const release = h.weight_release orelse return;
    if (h.closing.load(.acquire)) return;
    const work = h.queued.items.len != 0 or h.admitted.items.len != 0 or h.filling.items.len != 0 or h.cancels.items.len != 0 or h.core.activeCount() != 0;
    const extra = h.scores.asks.items.len != 0 or h.mem_ops.asks.items.len != 0 or h.lesson_open or h.lessons.items.len != 0;
    if (work or extra) return;
    release.idle(release.ctx);
}

/// A backend hears the error that failed a round or request; the message each stream receives does not depend on it.
pub const Failure = struct {
    ctx: *anyopaque,
    call: *const fn (*anyopaque, anyerror) void,
};

/// Both observers follow the same work lifetime; a family's own progress/failure policy is retained.
pub const Work = struct {
    backend: Activity.Scope,
    sentinel: Activity.Scope,
    pub fn enter(h: anytype) Work {
        const sentinel = Activity.enter(h.stall.activity());
        return .{ .backend = Activity.enter(h.activity), .sentinel = sentinel };
    }
    pub fn end(w: *Work) void {
        w.backend.end();
        w.sentinel.end();
    }
};
