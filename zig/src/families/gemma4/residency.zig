//! The served Gemma owners share one initial inventory and clear every callback before their set is destroyed.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

/// The buffers that outlive a request, in one set the idle keepalive holds wired (a stream's caches come and go).
pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer); // the weights, the tower's among them
    try list.appendSlice(h.gpa, m.weights.owned.items);
    try list.appendSlice(h.gpa, &.{ m.inv_sliding, m.inv_global });
    try m.projections.appendBuffers(h.gpa, &list);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.prompt_scratch);
    if (h.draft) |d| {
        try list.appendSlice(h.gpa, d.w.owned.items);
        try mtl.resident.fields.append(h.gpa, &list, d);
        try list.append(h.gpa, d.attention.part);
        try d.projections.appendBuffers(h.gpa, &list);
    }
    if (h.assist) |d| {
        try list.appendSlice(h.gpa, d.w.owned.items);
        try mtl.resident.fields.append(h.gpa, &list, d);
        try d.projections.appendBuffers(h.gpa, &list);
    }
    if (m.vision) |v| try v.appendBuffers(h.gpa, &list);
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "gemma4") orelse return;
    const follow = (&h.resident.?).follow();
    m.projections.setResident(follow);
    h.metal.setResident(follow);
    if (h.draft) |d| d.setResident(follow);
    if (h.assist) |d| d.setResident(follow);
    h.warm = (&h.resident.?).target();
}

pub fn drop(h: anytype) void {
    h.metal.setResident(null);
    h.model.projections.setResident(null);
    if (h.draft) |d| d.setResident(null);
    if (h.assist) |d| d.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
}
