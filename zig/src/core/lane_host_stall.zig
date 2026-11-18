//! Engine.configure_stall for a lane host: the server calls it once after each open, an idle reload's too.
const host = @import("lane_host.zig");
const stall = @import("lane_stall.zig");
pub fn configure(ctx: *anyopaque, config: stall.Config) !void {
    const h: *host.LaneHost = @ptrCast(@alignCast(ctx));
    try h.stall.start(h.gpa, h.io, config);
}
