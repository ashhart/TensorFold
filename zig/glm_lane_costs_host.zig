//! The production load-time measurement controller over recording timers, without devices or a checkpoint.
const std = @import("std");
const costs = @import("glm").lane_costs;
const t = std.testing;
const Recording = struct {
    windows: [costs.widths.len]usize = @splat(0),
    drafts: usize = 0,
    forms: [3]bool = undefined,
    changes: usize = 0,
    fact: bool = false,
    pub fn form(r: *@This(), fact: bool) !void {
        if (r.changes >= r.forms.len) return error.TooManyFormChanges;
        r.forms[r.changes] = fact;
        r.changes += 1;
        r.fact = fact;
    }
    pub fn window(r: *@This(), i: usize) !f64 {
        r.windows[i] += 1;
        return (20 + @as(f64, @floatFromInt(costs.widths[i]))) * @as(f64, if (r.fact) 0.8 else 1);
    }
    pub fn draft(r: *@This(), depth: usize) !f64 {
        r.drafts += 1;
        return 2 + 0.5 * @as(f64, @floatFromInt(depth));
    }
};

test "GLM load prices: DFlash2 races expert forms without timing unused windows or head steps" {
    var r: Recording = .{};
    const kept = try costs.measurements.measure(&r, true);
    try t.expectEqual([3]bool{ false, true, true }, r.forms);
    try t.expectEqual(@as(usize, 0), r.drafts);
    for (r.windows, 0..) |count, i| {
        const expected: usize = if (i == 1 or i == 3 or i == 5) 16 else if (i == costs.widths.len - 1) 8 else 0;
        try t.expectEqual(expected, count);
    }
    try t.expect(costs.measurements.reusable(kept, true));
    try t.expect(!costs.measurements.reusable(kept, false));
    try t.expect(costs.tableFor(kept, true) == null);
}

test "GLM load prices: MTP retains every timed width and the two-level head slope" {
    var r: Recording = .{};
    const kept = try costs.measurements.measure(&r, false);
    try t.expectEqual([3]bool{ false, true, true }, r.forms);
    try t.expectEqual(@as(usize, 32), r.drafts);
    for (r.windows, 0..) |count, i| {
        const expected: usize = 8 + @as(usize, if (i == 1 or i == 3 or i == 5) 16 else if (i == costs.widths.len - 1) 8 else 0);
        try t.expectEqual(expected, count);
    }
    try t.expectApproxEqAbs(@as(f64, 0.5), kept.step_ms, 1e-12);
    try t.expect(costs.measurements.reusable(kept, false));
    try t.expect(!costs.measurements.reusable(kept, true));
    try t.expect(costs.tableFor(kept, false) != null);
    try t.expect(costs.tableFor(kept, true) == null);
}
