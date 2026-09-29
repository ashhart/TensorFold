pub const Cancellation = struct {
    context: ?*anyopaque = null,
    callback: ?*const fn (?*anyopaque) anyerror!void = null,

    pub fn check(c: Cancellation) !void {
        if (c.callback) |call| try call(c.context);
    }
};
