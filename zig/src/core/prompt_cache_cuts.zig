//! RAM-only snapshots whose arithmetic depends on every prompt cut before their saved row.
const std = @import("std");
const pc = @import("prompt_cache.zig");

fn through(starts: []const u32, at: u32) []const u32 {
    for (starts, 0..) |s, i| if (s > at) return starts[0..i];
    return starts;
}

pub fn capture(s: *const pc.Store, starts: []const u32, at: u32) ![]u32 {
    return if (s.rules.cut_sensitive) s.gpa.dupe(u32, through(starts, at)) else &.{};
}

pub fn matches(s: *const pc.Store, e: *const pc.Entry, starts: []const u32) bool {
    return !s.rules.cut_sensitive or std.mem.eql(u32, e.cuts, through(starts, e.at));
}
