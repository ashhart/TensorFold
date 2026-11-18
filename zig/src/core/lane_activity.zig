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
