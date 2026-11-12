//! A save lesson's one step: the whole model's lessons into its own shards, routed blocks a file each beside them.
const std = @import("std");
const learner = @import("learner.zig");
const adapters = @import("learn_shape.zig");
const whole_lessons = @import("whole.zig");

/// Every kept lesson not yet written goes beside the model's shards, a file each.
pub fn once(l: anytype, comptime has_sites: bool, comptime has_whole: bool) !learner.Step {
    const H = @TypeOf(l.*).Host;
    l.phase = .idle;
    const t = l.trainer orelse return .{ .done = true };
    if (l.whole) { // a lesson of the model itself goes into its own shards: the pocket, or an edit's codes
        if (has_whole and !try whole_lessons.save(l)) try whole_lessons.foldOpen(l);
        return .{ .done = true, .changed = true };
    }
    if (comptime !has_sites) return .{ .done = true };
    const ranks = t.sites.rank - @as(usize, if (l.opened and l.kept_rounds == 0) adapters.block else 0);
    if (ranks == 0) return .{ .done = true };
    const t0 = H.seconds(l.io);
    const bytes = try H.write(l.b, l.gpa, l.io, t, ranks);
    std.log.info("slide: {d} lessons beside the model's shards, a file each with its router: {d} MB written in {d:.1} s", .{ ranks / adapters.block, bytes >> 20, H.seconds(l.io) - t0 });
    return .{ .done = true };
}
