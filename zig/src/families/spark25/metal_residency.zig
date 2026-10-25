//! One initial inventory and a stable follow hook for every served Spark2.5 cache and saved state on Metal.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

/// Weights, tables and scratches in one residency set for the keepalive, if the 70% limit holds them.
pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    try list.appendSlice(h.gpa, m.weights.stacked.items);
    try list.appendSlice(h.gpa, &m.inv);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.prompt);
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "spark2_5");
    if (h.resident) |*r| h.metal.setResident(r.follow());
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = m.queue };
}

pub fn drop(h: anytype) void {
    h.metal.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
}
