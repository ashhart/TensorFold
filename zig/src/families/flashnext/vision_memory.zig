//! Vision shares the text owner's ceiling; source weights are charged once and request outputs are transient.
const std = @import("std");
pub const Plan = struct {
    source: u64,
    weights: u64,
    scratch: u64,
    rows: u64,
    temporary: u64,

    pub fn extra(p: Plan) !u64 {
        if (p.weights < p.source) return error.BadVisionMemoryPlan;
        return std.math.add(u64, p.weights, try std.math.add(u64, p.scratch, try std.math.add(u64, p.rows, p.temporary)));
    }
    pub fn tower(p: Plan) !u64 {
        return std.math.add(u64, p.weights, p.scratch);
    }
};

pub fn withPlan(settings: anytype, plan: Plan) !@TypeOf(settings) {
    var out = settings;
    out.startup_extra = try std.math.add(u64, out.startup_extra, try plan.extra());
    return out;
}

pub fn loadTower(comptime T: type, run: anytype, bytes: u64, context: anytype, comptime construct: anytype) !T {
    try run.checkGrowth(std.math.cast(usize, bytes) orelse return error.Overflow);
    return construct(context);
}

pub fn outputs(comptime B: type, run: anytype, total: usize, most: usize, row: usize, options: usize) !struct { rows: B, temporary: B } {
    const rows = try run.allocateOptions(try std.math.mul(usize, total, try std.math.mul(usize, 4, row)), options);
    errdefer rows.deinit();
    const temporary = try run.allocateOptions(try std.math.mul(usize, most, row), options);
    return .{ .rows = rows, .temporary = temporary };
}
