//! Physical weights and independent state slots in execution order.
const config = @import("loop_config.zig");

pub const max_steps = 512;
pub const Step = struct {
    layer: usize,
    delta: ?usize = null,
    attention: ?usize = null,
    inside: bool = false,
    enter: bool = false,
    leave: bool = false,
};

/// Whether the schedule executes physical layer `layer` exactly once.
pub fn once(steps: []const Step, layer: usize) bool {
    var count: usize = 0;
    for (steps) |s| count += @intFromBool(s.layer == layer);
    return count == 1;
}

pub const Schedule = struct {
    steps: [max_steps]Step = undefined,
    len: usize = 0,
    deltas: usize = 0,
    attentions: usize = 0,

    pub fn slice(s: *const Schedule) []const Step {
        return s.steps[0..s.len];
    }

    fn add(s: *Schedule, linear: []const bool, layer: usize, mhc: ?config.Mhc) error{ScheduleTooLarge}!void {
        if (s.len == max_steps) return error.ScheduleTooLarge;
        var step: Step = .{ .layer = layer };
        if (linear[layer]) {
            step.delta = s.deltas;
            s.deltas += 1;
        } else {
            step.attention = s.attentions;
            s.attentions += 1;
        }
        if (mhc) |m| {
            step.inside = layer >= m.start and layer < m.end;
            step.enter = layer == m.start;
            step.leave = layer + 1 == m.end;
        }
        s.steps[s.len] = step;
        s.len += 1;
    }
};

pub fn build(linear: []const bool, loop: ?config.Config) error{ ScheduleTooLarge, BadLoopConfig }!Schedule {
    var s: Schedule = .{};
    if (loop) |c| {
        if (c.start >= c.end or c.end > linear.len or c.count == 0) return error.BadLoopConfig;
        for (0..c.start) |i| try s.add(linear, i, c.mhc);
        for (0..c.count) |_| for (c.start..c.end) |i| try s.add(linear, i, c.mhc);
        for (c.end..linear.len) |i| try s.add(linear, i, c.mhc);
    } else for (0..linear.len) |i| try s.add(linear, i, null);
    return s;
}

pub const Decision = struct {
    enter: bool,
    route: bool,
    first_add: bool,
    second_add: bool,
    merge: bool,
    leave: bool,
    pending: bool,

    pub fn actions(d: Decision) Actions {
        var out: Actions = .{};
        if (d.enter) out.add(.enter);
        if (d.route) out.add(.route);
        out.add(if (d.first_add) .norm_add else .norm_plain);
        out.add(.mixer);
        if (d.merge) out.add(.merge);
        if (d.route) out.add(.route);
        out.add(if (d.second_add) .norm_add else .norm_plain);
        out.add(.mlp);
        if (d.merge) out.add(.merge);
        if (d.leave) out.add(.leave);
        return out;
    }
};

pub const Action = enum { enter, route, norm_add, norm_plain, mixer, merge, mlp, leave };
pub const Actions = struct {
    items: [10]Action = undefined,
    len: usize = 0,
    fn add(a: *Actions, item: Action) void {
        a.items[a.len] = item;
        a.len += 1;
    }
    pub fn slice(a: *const Actions) []const Action {
        return a.items[0..a.len];
    }
};

pub fn decide(step: Step, pending: bool) Decision {
    return .{
        .enter = step.enter,
        .route = step.inside,
        .first_add = !step.inside and pending,
        .second_add = !step.inside,
        .merge = step.inside,
        .leave = step.leave,
        .pending = !step.inside,
    };
}
