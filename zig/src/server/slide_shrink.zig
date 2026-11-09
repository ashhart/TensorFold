//! Shrinking reuses the complete lesson gate and prior-fact recheck; only a passing candidate is kept.
const std = @import("std");
const api = @import("engine_api");
const slide = @import("slide.zig");
const graph = @import("slide_graph.zig");
const errors = @import("errors.zig");
const teach = @import("slide_lesson.zig");
const check = @import("slide_verify.zig");
const recheck = @import("slide_recheck.zig");
const Server = @import("server.zig").Server;

/// Stage, judge and resolve every candidate. Any interrupted judgment restores before its error escapes.
pub fn drive(ctx: anytype) !void {
    while (try ctx.next()) {
        var resolved = false;
        errdefer if (!resolved) ctx.restore() catch {};
        if (try ctx.passes()) try ctx.keep() else try ctx.restore();
        resolved = true;
    }
}

pub fn run(srv: *Server, a: std.mem.Allocator, cx: *errors.Cx, plan: teach.Plan, gone: anytype, id: u32, out: *slide.Out) !void {
    const Context = struct {
        srv: *Server,
        a: std.mem.Allocator,
        cx: *errors.Cx,
        plan: teach.Plan,
        gone: @TypeOf(gone),
        id: u32,
        out: *slide.Out,
        before: []const graph.Earlier = &.{},
        pub fn next(c: *@This()) !bool {
            c.before = try c.srv.slide.earlier(c.a, c.id);
            return switch (slide.runReported(c.srv, c.a, &.{ .shrink = .next }, 0, null)) {
                .learned => |l| l.candidate,
                .none => false, // a host without the pocket has no shrink candidates
                else => error.ShrinkFailed,
            };
        }
        pub fn passes(c: *@This()) !bool {
            const v = try check.verify(c.srv, c.cx, c.plan, true, c.gone);
            if (v.damage != null or v.took != null or !std.mem.allEqual(bool, v.recalled, true)) return false;
            const results = try recheck.check(c.srv, c.cx, c.id, .all, c.before, c.gone);
            return (try recheck.forgotten(c.a, results)).len == 0;
        }
        fn resolve(c: *@This(), action: @FieldType(api.LearnRequest, "shrink")) !void {
            switch (slide.runReported(c.srv, c.a, &.{ .shrink = action }, if (action == .keep) c.id else 0, c.out)) {
                .learned => {},
                else => return error.ShrinkFailed,
            }
        }
        pub fn keep(c: *@This()) !void {
            try c.resolve(.keep);
            c.out.event("shrink", .{ .id = c.id, .message = "one neuron removed; every lesson and earlier-fact check still passes" });
        }
        pub fn restore(c: *@This()) !void {
            try c.resolve(.restore);
            c.srv.slide.restoreChecks(c.before);
        }
    };
    var c: Context = .{ .srv = srv, .a = a, .cx = cx, .plan = plan, .gone = gone, .id = id, .out = out };
    try drive(&c);
}
