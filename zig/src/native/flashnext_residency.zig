//! Flash Next residency includes the runner, transport, snapshot pool, shared sessions and optional image tower.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

fn initial(h: anytype, list: *std.ArrayList(mtl.Buffer)) !void {
    try list.appendSlice(h.gpa, h.eng.r.held.items);
    if (h.eng.r.tp) |tp| try mtl.resident.fields.append(h.gpa, list, tp);
    for (h.eng.snap_pool.bufs) |b| if (b) |buffer| try list.append(h.gpa, buffer);
    if (h.vis) |v| try list.appendSlice(h.gpa, &v.buffers());
}

fn attach(h: anytype, io: std.Io, buffers: []const mtl.Buffer, label: []const u8) !void {
    h.resident = try Resident.holdUnder(h.gpa, io, h.eng.r.device, h.eng.r.queue, buffers, label, if (h.load_limit > 0) h.load_limit else null);
    if (h.resident) |*r| {
        h.warm = r.target();
        h.eng.r.follow = r.follow();
        h.eng.snap_pool.follow = r.follow();
    }
}

pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    try initial(h, &list);
    try attach(h, h.io, list.items, "flash next");
}

pub fn drop(h: anytype) void {
    h.eng.r.follow = null;
    h.eng.snap_pool.follow = null;
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.eng.r.queue };
}

pub fn holdShared(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    try initial(h, &list);
    for (h.back.sessions) |s| try list.appendSlice(h.gpa, s.owned.items);
    try list.append(h.gpa, h.back.hidden.b);
    try attach(h, h.host.io, list.items, "flash next shared");
    if (h.resident) |*r| for (h.back.sessions) |*session| session.setResident(r.follow());
}

pub fn dropShared(h: anytype) void {
    for (h.back.sessions) |*session| session.setResident(null);
    drop(h);
}
