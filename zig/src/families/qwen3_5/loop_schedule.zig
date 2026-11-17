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

pub fn enterMhc(step: Step, pending: bool) bool {
    _ = step;
    return pending;
}

pub fn leaveMhc(step: Step, pending: bool) bool {
    _ = step;
    return pending;
}

pub fn attentionMhc(step: Step, pending: bool) bool {
    _ = step;
    return pending;
}

pub fn ffnMhc(step: Step, pending: bool) bool {
    _ = step;
    return pending;
}
