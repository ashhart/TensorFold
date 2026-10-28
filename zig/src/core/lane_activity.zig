//! A backend observes owned engine work without holding an idle wait open as an operation.
pub const Activity = struct {
    ctx: *anyopaque,
    begin: *const fn (*anyopaque) void,
    end: *const fn (*anyopaque) void,
    failed: ?*const fn (*anyopaque, anyerror) void = null,

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
