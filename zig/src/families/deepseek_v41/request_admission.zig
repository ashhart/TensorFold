//! Direct target admission uses the same full-request pool plan as the round planner, before restore or prefill.

const std = @import("std");

pub const Resumed = struct { at: u64 = 0, damaged: bool = false };

/// A failed cache restore loses its shared-page credit; re-price the empty slot before any prompt row runs.
pub fn resumeRequest(s: anytype, prompt: []const u32, max_new: u64) !Resumed {
    const hit = try s.find(prompt);
    const r = try apply(s, prompt, max_new, hit);
    if (r.damaged) {
        try afterMiss(s, prompt, max_new);
    }
    return r;
}

/// Called after an agreed restore miss, also by the round planner's pre-planned admission path.
pub fn afterMiss(s: anytype, prompt: []const u32, max_new: u64) !void {
    if (s.kx.slot.len != 0) return error.SlotNotEmpty;
    const kv = @import("kv");
    const covered = @as(u64, s.kx.slot.quota orelse 0) * s.kx.slot.pool.page;
    const reply = covered -| (@as(u64, prompt.len) + kv.sess.slack);
    s.kx.slot.reserve(null);
    _ = try apply(s, prompt, @max(max_new, reply), @as(?kv.sessions.store.Hit, null));
}

fn apply(s: anytype, prompt: []const u32, max_new: u64, hit: anytype) !Resumed {
    var spills: std.ArrayList(u32) = .empty;
    defer spills.deinit(s.gpa);
    const held: []const u32 = if (hit) |h| &.{h.id} else &.{};
    const plan = try s.poolPlan(prompt.len, max_new, hit, held, 0, &spills);
    // Refuse before sending operations, evicting entries or mutating the slot; the lane host queues a LaneMemory job.
    if (plan.wait) return error.LaneMemory;
    const r = try s.resumePlanned(prompt, plan.need, hit, spills.items);
    return .{ .at = r.at, .damaged = r.damaged };
}
