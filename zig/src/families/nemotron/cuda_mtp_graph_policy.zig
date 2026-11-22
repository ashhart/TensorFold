//! Device DepthRule inputs: the host computes the same rate and row costs; the device observes each confidence.
const std = @import("std");
const Rule = @import("core").draft_depth.DepthRule;
pub const Policy = extern struct {
    row: [33]f64,
    rate: f64,
    level: f64,
    floor: f64,
    most: u32,
    has_floor: u32,
    pub fn init(r: Rule) ?Policy {
        if (r.power != 2 or r.most < 1 or r.most > 31 or r.costs.rows > 32) return null;
        var p: Policy = .{ .row = @splat(0), .rate = r.rate(), .level = r.costs.level, .floor = r.floor orelse 0, .most = @intCast(r.most), .has_floor = @intFromBool(r.floor != null) };
        for (1..r.costs.rows) |k| p.row[k] = r.costs.row(k);
        return p;
    }
};
pub const State = extern struct {
    policy: Policy,
    run: f64 = 1,
    j: u32 = 1,
    n: u32 = 0,
    levels: u32 = 0,
    pad: u32 = 0,
    meta: [2]u32 = .{ 0, 0 },
    token: u32 = 0,
    probability: f32 = 0,
    ids: [31]u32 = @splat(0),
    probs: [31]f32 = @splat(0),
};
comptime {
    std.debug.assert(@sizeOf(Policy) == 296 and @sizeOf(State) == 584);
}
