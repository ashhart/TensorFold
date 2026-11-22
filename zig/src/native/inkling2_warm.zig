//! Verify-width timing for Inkling 2's Metal backend so depth pricing uses measured costs.
const std = @import("std");
const tf = @import("tensorfold");
const lanes = tf.lanes;
const fl = tf.inkling2_metal;

/// Time verify widths 1..costs.len on a clean lane and store median milliseconds per width.
pub fn warm(model: *fl.Model, state: *fl.state.State, capacity: u32, drafts: bool, costs: []lanes.config.Cost) !usize {
    defer model.reset(state);
    const max_w: u32 = @min(@as(u32, @intCast(costs.len)), capacity);
    if (max_w == 0) return 0;
    const prefix_len: u32 = @min(@as(u32, 16), capacity - max_w);
    const v: u32 = @intCast(@max(model.config.text.unpadded_vocab, 2));
    var ids: [16]u32 = undefined;
    std.debug.assert(max_w <= ids.len);
    for (ids[0..@max(max_w, prefix_len)], 0..) |*id, i| id.* = @intCast((1 + i) % v);
    if (prefix_len > 0) {
        try model.forward(&.{.{ .state = state, .ids = ids[0..prefix_len], .logit_rows = 0, .taps = false }});
    }
    for (1..max_w + 1) |width| {
        var times: [3]f64 = undefined;
        for (&times) |*tm| {
            const t0 = seconds();
            try model.forward(&.{.{ .state = state, .ids = ids[0..width], .logit_rows = @intCast(width), .taps = drafts }});
            tm.* = seconds() - t0;
            try model.keep(state, prefix_len);
        }
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        costs[width - 1] = .{ .width = @intCast(width), .ms = 1000.0 * times[1] };
    }
    return max_w;
}

fn seconds() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) + @as(f64, @floatFromInt(ts.nsec)) * 1e-9;
}
