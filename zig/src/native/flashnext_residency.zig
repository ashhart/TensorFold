//! Flash Next residency follows both model buffers and the optional vision tower.
const std = @import("std");
const mtl = @import("metal");
const Resident = @import("resident.zig").Resident;

/// The engine's buffers in one residency set the idle keepalive's ticks use; buffers made later join it or stay out.
pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    try list.appendSlice(h.gpa, h.eng.r.held.items);
    if (h.vis) |v| try list.appendSlice(h.gpa, &v.buffers());
    h.resident = try Resident.holdUnder(h.gpa, h.io, h.eng.r.device, h.eng.r.queue, list.items, "flash next", if (h.load_limit > 0) h.load_limit else null) orelse return;
    const r = &h.resident.?;
    h.warm = r.target();
    h.eng.r.follow = r.follow();
    h.eng.snap_pool.follow = r.follow(); // kept states' buffers join it, and leave it before they are freed
}

/// The set released; the engine's allocators stop reporting to it first.
pub fn drop(h: anytype) void {
    const r = if (h.resident) |*x| x else return;
    h.eng.r.follow = null;
    h.eng.snap_pool.follow = null;
    r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.eng.r.queue };
}
