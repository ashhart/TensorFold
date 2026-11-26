//! Host storage for attention metadata; the planner still rebuilds every row and path.
const std = @import("std");
const planning = @import("attention_plan.zig");

pub const Scratch = struct {
    arena: ?std.heap.ArenaAllocator = null,

    /// A runner owns one plan at a time; deinit that plan before rebuilding with the same allocator.
    pub fn plan(s: *Scratch, gpa: std.mem.Allocator, windows: []const planning.Window, geometry: planning.Geometry) !planning.Plan {
        if (s.arena == null) s.arena = .init(gpa);
        if (!s.arena.?.reset(.{ .retain_with_limit = 2 << 20 })) _ = s.arena.?.reset(.free_all);
        return planning.Plan.init(s.arena.?.allocator(), windows, geometry);
    }
    pub fn deinit(s: *Scratch) void {
        if (s.arena) |*arena| arena.deinit();
        s.* = .{};
    }
};

/// The runner's original geometry and pairing, with either independent or caller-owned host storage.
pub fn build(r: anytype, round: anytype, scratch: ?*Scratch) !planning.Plan {
    var windows: [planning.max_streams]planning.Window = undefined;
    if (round.windows.len > windows.len) return error.AttentionStreams;
    for (round.windows, round.slots, 0..) |window, slot, i| windows[i] = .{ .parents = window.parents, .start = window.start, .capacity = if (r.kv_step != null) r.heldRows(slot) else r.capacity };
    var geometry = r.geometry;
    geometry.solo_prompt = geometry.solo_prompt and r.model.kernels.prompt;
    var p = if (scratch) |s| try s.plan(r.allocator, windows[0..round.windows.len], geometry) else try planning.Plan.init(r.allocator, windows[0..round.windows.len], geometry);
    if (r.attention.join != null) p.pair(&.{});
    return p;
}
