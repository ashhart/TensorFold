//! One initial inventory and a stable follow hook for every served Llama cache and saved state on Metal.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

/// Weights, tables and scratches, the drafter's too, in one residency set for the keepalive, under the 70% limit.
pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    try list.append(h.gpa, m.inv);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.prompt);
    if (h.drafter) |d| {
        for (d.model.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
        try list.append(h.gpa, d.model.inv);
        try list.append(h.gpa, d.tokens);
        try mtl.resident.fields.append(h.gpa, &list, &d.scratch);
        try mtl.resident.fields.append(h.gpa, &list, &d.prompt);
    }
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "llama");
    if (h.resident) |*r| {
        h.metal.setResident(r.follow());
        if (h.drafter) |d| d.setResident(r.follow());
    }
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = m.queue };
}

pub fn drop(h: anytype) void {
    h.metal.setResident(null);
    if (h.drafter) |d| d.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
}
