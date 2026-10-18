//! GLM's bounded shared residency set, with real keepalive work and later arena ownership.
const std = @import("std");
const mtl = @import("metal");
const Engine = @import("engine.zig").Engine;
pub const policy = mtl.residency_policy;
pub const Resident = mtl.resident.Resident;

pub fn wire(e: *Engine, io: std.Io) void {
    e.keepalive_target = .{ .queue = e.queue };
    if (!e.residency_enabled) {
        std.log.info("glm: residency skipped by GLM_RESIDENCY=0", .{});
        return;
    }
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(e.gpa);
    append(e, &list) catch |err| return skipped(err);
    e.residency = Resident.holdUnder(e.gpa, io, e.device, e.queue, list.items, "glm", if (e.limit.asked) e.limit.bytes else null) catch |err| {
        skipped(err);
        return;
    };
    if (e.residency) |*r| {
        e.arena.follow = r.follow();
        e.keepalive_target = r.target();
    }
}

pub fn append(e: anytype, list: *std.ArrayList(mtl.Buffer)) !void {
    try list.appendSlice(e.gpa, e.w.buffers.items);
    try list.appendSlice(e.gpa, e.arena.buffers.items);
    if (e.vision) |v| try list.appendSlice(e.gpa, &v.buffers());
}

fn skipped(err: anyerror) void {
    std.log.warn("glm: residency skipped ({s}); buffers can be unwired during idle", .{@errorName(err)});
}

pub fn drop(e: *Engine) void {
    e.arena.follow = null;
    if (e.residency) |*r| r.deinit();
    e.residency = null;
    e.keepalive_target = .{ .queue = e.queue };
}
