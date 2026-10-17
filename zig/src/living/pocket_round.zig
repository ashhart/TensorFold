//! The landed shard encoder plus finite-value checks: the open fact's layers rounded once, as a save writes them.
const std = @import("std");
const math = @import("neuron.zig");
pub fn layer(comptime F: @import("family.zig").Family, gpa: std.mem.Allocator, d: usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !void {
    if (d == 0 or d % F.quant.group != 0 or neurons % F.quant.group != 0 or up.len != neurons * d or down.len != up.len)
        return error.BadPocket;
    if (F.neuron.act == .swiglu and gate.len != up.len) return error.BadPocket;
    for ([_][]f32{ gate, up, down }) |xs| for (xs) |x| if (!std.math.isFinite(x)) return error.NonFinitePocket;
    try @import("shards.zig").round(F, gpa, d, if (F.neuron.act == .swiglu) &.{ gate, up } else &.{up}, down);
    for ([_][]f32{ gate, up, down }) |xs| for (xs) |x| if (!std.math.isFinite(x)) return error.NonFinitePocket;
}

/// The open fact's layers (its neurons, the earlier ones it refitted, the joint ones) rounded once and uploaded.
pub fn asWritten(p: anytype, t: anytype, host: anytype) !void {
    var done: std.ArrayList(usize) = .empty;
    defer done.deinit(p.gpa);
    for (p.taken.items) |x| try once(p.gpa, &done, x.slot);
    for (p.refits.items) |x| try once(p.gpa, &done, x.slot);
    for (p.joint.items) |x| try once(p.gpa, &done, x.slot);
    for (done.items) |j| {
        try slot(p, t, host, j);
        try p.upload(t, host, j);
    }
}

fn once(gpa: std.mem.Allocator, list: *std.ArrayList(usize), j: usize) !void {
    if (std.mem.indexOfScalar(usize, list.items, j) == null) try list.append(gpa, j);
}

/// One slot rounded once by the shard encoder, SwiGLU's u through the up rows it is written as.
pub fn slot(p: anytype, t: anytype, host: anytype, j: usize) !void {
    _ = t;
    _ = host;
    const P = @TypeOf(p.*);
    const n = P.neurons * p.hidden;
    const up = p.up.items[j * n ..][0..n];
    const gate: []f32 = if (P.swiglu) p.gate.items[j * n ..][0..n] else &.{};
    if (P.swiglu) math.upRow(P.of.neuron, up, p.scratch.items);
    try layer(P.of, p.gpa, p.hidden, P.neurons, gate, if (P.swiglu) p.scratch.items else up, p.down.items[j * n ..][0..n]);
    if (P.swiglu) math.fromUp(P.of.neuron, p.scratch.items, up);
}
