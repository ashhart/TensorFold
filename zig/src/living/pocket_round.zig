//! The landed shard encoder plus finite-value and second-round stability checks before judgement.
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
pub fn asWritten(p: anytype, t: anytype, host: anytype) !void {
    for (0..p.used.items.len) |j| {
        try slot(p, t, host, j);
        try p.upload(t, host, j);
    }
}
pub fn slot(p: anytype, t: anytype, host: anytype, j: usize) !void {
    _ = t;
    _ = host;
    const P = @TypeOf(p.*);
    const n = P.neurons * p.hidden;
    const up = p.up.items[j * n ..][0..n];
    const gate: []f32 = if (P.swiglu) p.gate.items[j * n ..][0..n] else &.{};
    if (P.swiglu) math.upRow(P.of.neuron, up, p.scratch.items);
    try layer(P.of, p.gpa, p.hidden, P.neurons, gate, if (P.swiglu) p.scratch.items else up, p.down.items[j * n ..][0..n]);
    const written_up = if (P.swiglu) p.scratch.items else up;
    const down = p.down.items[j * n ..][0..n];
    const copy = try p.gpa.alloc(f32, gate.len + 2 * n);
    defer p.gpa.free(copy);
    const cg = copy[0..gate.len];
    const cu = copy[gate.len..][0..n];
    const cv = copy[gate.len + n ..];
    @memcpy(cg, gate);
    @memcpy(cu, written_up);
    @memcpy(cv, down);
    try layer(P.of, p.gpa, p.hidden, P.neurons, cg, cu, cv);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(cg), std.mem.sliceAsBytes(gate)) or
        !std.mem.eql(u8, std.mem.sliceAsBytes(cu), std.mem.sliceAsBytes(written_up)) or
        !std.mem.eql(u8, std.mem.sliceAsBytes(cv), std.mem.sliceAsBytes(down))) return error.LayerRoundsAgain;
    if (P.swiglu) math.fromUp(P.of.neuron, p.scratch.items, up);
}
