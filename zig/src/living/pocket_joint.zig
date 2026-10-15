//! Earlier facts on the open fact's subject fitted with it: MEMIT-Merge's one value a subject, a fact at a time.
const std = @import("std");

/// The earlier subject neurons the open fact shares its subject with, their replayed answers, their outputs as now.
pub fn find(p: anytype, d: usize) !void {
    p.joint.clearRetainingCapacity();
    p.joint_down.clearRetainingCapacity();
    p.joint_answers.clearRetainingCapacity();
    if (!p.joining or p.split[1] == 0 or p.subjects.items.len == 0) return;
    var group: std.ArrayList(u32) = .empty;
    defer group.deinit(p.gpa);
    for (p.learned.items[0..p.learned_before]) |e| {
        if (e.key != .subject or std.mem.indexOfScalar(u32, p.subjects.items, e.subject) == null) continue;
        if (std.mem.indexOfScalar(u32, p.retired_lessons.items, e.lesson) != null) continue;
        if (std.mem.indexOfScalar(u32, group.items, e.lesson) == null) try group.append(p.gpa, e.lesson);
    }
    const replay: []const @TypeOf(p.answered[0]) = if (p.steady.len > 2) p.steady[2] else &.{};
    for (replay, 0..) |ex, i| {
        if (i >= p.replay_lessons.len or std.mem.indexOfScalar(u32, group.items, p.replay_lessons[i]) == null) continue;
        if (p.keyRow(ex, .subject) != null) try p.joint_answers.append(p.gpa, ex);
    }
    // without their answers to hold them, earlier outputs would only bend to the new fact
    if (p.joint_answers.items.len == 0) return;
    for (p.learned.items[0..p.learned_before]) |e| if (e.key == .subject and std.mem.indexOfScalar(u32, group.items, e.lesson) != null) {
        try p.joint.append(p.gpa, .{ .slot = e.slot, .neuron = e.neuron, .key = .subject });
        try p.joint_down.appendSlice(p.gpa, p.neuron(p.down.items, e.slot, e.neuron, d));
    };
    std.log.info("slide pocket: {d} neurons of {d} earlier facts on its subject fitted with it, on {d} of their answers", .{ p.joint.items.len, group.items.len, p.joint_answers.items.len });
}

/// Their outputs as the fit starts, after the open fact's (which start at zero).
pub fn seed(p: anytype, d: usize) void {
    const k = p.taken.items.len;
    @memcpy(p.v.adam.items[k * d ..][0 .. p.joint.items.len * d], p.joint_down.items);
}

/// The earlier neurons as they were before the open fact fitted them (its lesson undone).
pub fn putBack(p: anytype, t: anytype, host: anytype, d: usize) !void {
    for (p.joint.items, 0..) |x, j| {
        @memcpy(p.neuron(p.down.items, x.slot, x.neuron, d), p.joint_down.items[j * d ..][0..d]);
        try p.upload(t, host, x.slot);
    }
    p.joint.clearRetainingCapacity();
    p.joint_down.clearRetainingCapacity();
    p.joint_answers.clearRetainingCapacity();
}
