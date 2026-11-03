//! Full-pocket policy: a least-used unpinned fact may leave only through the normal checked delete transaction.
const std = @import("std");
const api = @import("engine_api");
const slide = @import("slide.zig");
const usage = @import("slide_usage.zig");
const errors = @import("errors.zig");
const Server = @import("server.zig").Server;

/// Called after rounds returned, so its failed lesson's exact drop has finished before the cut starts.
pub fn makeRoom(srv: *Server, cx: *errors.Cx, out: *slide.Out, id: u32, why: ?[]const u8, gone: anytype) !bool {
    if (!srv.config.slide_forget or why == null or !std.mem.eql(u8, why.?, "PocketFull") or gone.check()) return false;
    if (!srv.info.pocket_usage) return error.UsageAuditUnavailable;
    try srv.slide.flushUsage();
    const facts = try srv.slide.snapshot(cx.a);
    const chosen = usage.least(facts, id) orelse return false;
    const at = for (facts, 0..) |f, i| {
        if (f.id == chosen) break i;
    } else unreachable;
    const reason = try std.fmt.allocPrint(cx.a, "least-used unpinned fact retired for fact {d} (uses {d})", .{ id, facts[at].uses });
    try @import("slide_forget.zig").delete(srv, cx, out, facts, at, reason, gone);
    const after = try srv.slide.snapshot(cx.a);
    for (after) |f| if (f.id == chosen) return f.state == .deleted;
    return false;
}
