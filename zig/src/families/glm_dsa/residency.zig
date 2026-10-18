//! A paired GLM host keeps its Metal allocations on the shared bounded residency path.
const std = @import("std");
const mtl = @import("metal");
fn layer(a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer), l: anytype) !void {
    try mtl.resident.fields.append(a, list, l);
    try mtl.resident.fields.append(a, list, &l.k);
}
pub fn hold(e: anytype, io: std.Io, limit: ?u64) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(e.gpa);
    try mtl.resident.fields.append(e.gpa, &list, e);
    try mtl.resident.fields.append(e.gpa, &list, &e.m);
    try mtl.resident.fields.append(e.gpa, &list, &e.m.K);
    try mtl.resident.fields.append(e.gpa, &list, &e.m.head);
    try mtl.resident.fields.append(e.gpa, &list, &e.m.vocab.k);
    for (e.m.layers) |*l| try layer(e.gpa, &list, l);
    if (e.m.mtp) |*l| try layer(e.gpa, &list, l);
    try mtl.resident.fields.append(e.gpa, &list, &e.s);
    try mtl.resident.fields.append(e.gpa, &list, &e.q);
    for (&e.pass) |*p| try mtl.resident.fields.append(e.gpa, &list, p);
    try list.append(e.gpa, e.pair.wbuf);
    var snapshots = e.snaps.valueIterator();
    while (snapshots.next()) |snap| try list.append(e.gpa, snap.*.buf);
    e.resident = mtl.resident.Resident.holdUnder(e.gpa, io, e.device, e.queue, list.items, "glm dsa", limit) catch |err| blk: {
        std.log.warn("glm dsa: residency skipped ({s})", .{@errorName(err)});
        break :blk null;
    };
}
pub fn target(e: anytype) mtl.keepalive.Target {
    return if (e.resident) |*r| r.target() else .{ .queue = e.queue };
}
pub fn drop(e: anytype) void {
    if (e.resident) |*r| r.deinit();
    e.resident = null;
}
