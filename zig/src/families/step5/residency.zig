//! One initial inventory and a stable follow hook for every served Step5 cache and saved state.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

/// Weights, routers, scratches and tower buffers in one residency set for the keepalive, if the 70% limit holds them.
pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    for (m.weights.layers) |l| if (l.ffn == .moe) try list.append(h.gpa, l.ffn.moe.router);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.prompt);
    if (m.vision) |v| try list.appendSlice(h.gpa, &.{ v.owned, v.scratch, v.patches.buf });
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "step5");
    if (h.resident) |*r| h.metal.setResident(r.follow());
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = m.queue };
}

pub fn drop(h: anytype) void {
    h.metal.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
}
