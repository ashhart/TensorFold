//! GLM's opt-in copy facts; acceptance estimates and cost allocation stay in the shared lane core.
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

/// With DFlash2: a window's ms by rows on two GB10s (EXL3, TF_GLM_PROF: 86 to 107 ms at 5 to 8), shared as glm_cuda's.
pub const window_costs = costTable(@import("cuda_forward.zig").window_rows, 52.0, 6.8);
pub const shared_costs = costTable(engine.shared_rows, 76.0, 4.1);

fn costTable(comptime n: usize, comptime one: f64, comptime row: f64) [n]lanes.config.Cost {
    var c: [n]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = one + row * @as(f64, @floatFromInt(i)) };
    return c;
}

/// The serving drafts switch wins; unset keeps the family's facts. `measured`: timed prices, so no seed tables.
pub fn apply(base: lanes.Model, enabled: bool, measured: bool) lanes.Model {
    if (!enabled or !base.mtp) return base;
    var m = base;
    m.root_copies = true;
    m.context_copies = true;
    m.copy_match = 4;
    m.own_copy_rates = true;
    m.plain_guard = true;
    if (!measured) m.window_costs = &window_costs;
    if (!measured) m.shared_costs = &shared_costs;
    m.window_costs_lone = false; // every copy pays for its rows, in shared rounds too
    m.shared_drafts = 0;
    // Keep head confidence in the backend's probabilities so every copy still pays its target-window cost.
    m.draft_confidence = null;
    return m;
}

/// The same shared confidence prefix, encoded as zero tail probabilities for the ordinary cost allocator.
pub fn confidence(floor: ?f64, chances: []f64) void {
    if (floor == null) return;
    const kept = (lanes.mtp_policy.Controls{ .floor = floor }).prefix(chances);
    @memset(chances[kept..], 0);
}

/// Publish the explicit head prefix as the held count, so even a cheaper wider tile cannot cross the cap.
pub fn held(ptr: *anyopaque, s: *lanes.Stream, _: @import("std").mem.Allocator) !?lanes.stream.Held {
    const self: *@import("cuda_lanes.zig").Lanes = @ptrCast(@alignCast(ptr));
    if (self.mtp_confidence == null or !self.copy_confidence or self.x.e.drafter != null) return null;
    const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
    if (l.held_n > engine.max_drafts) return error.NoHeldDrafts;
    var chances: [engine.max_drafts]f64 = undefined;
    const out = chances[0..l.held_n];
    const b = self.backend();
    if (!try b.vtable.probabilities.?(ptr, s, out)) return error.NoDraftProbabilities;
    return .{ .count = @intCast((lanes.mtp_policy.Controls{ .floor = self.mtp_confidence }).prefix(out)) };
}
