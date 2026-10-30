//! A shared HC/GDN/QSA/PLE layer graph; dense and routed feed-forwards differ at one owned boundary.
const std = @import("std");
pub const Kind = enum { linear_attention, indexed_attention };
pub const FeedForward = enum { dense, routed };
pub const Layer = struct { kind: Kind, ple: bool = false, feed_forward: FeedForward };
pub const Geometry = struct {
    hidden: usize,
    streams: usize,
    lowrank: usize,
    vocab: usize,
    intermediate: usize,
    eps: f32,
    layers: []const Layer,
};

/// Backend operations own tensors, queued work and per-stream state; the graph never synchronizes or schedules.
pub fn forward(o: anytype, g: Geometry, state: anytype, tokens: []const u32) !@TypeOf(o.*).Tensor {
    if (tokens.len == 0) return error.EmptyForward;
    if (g.hidden == 0 or g.streams < 2 or g.lowrank == 0 or g.layers.len == 0) return error.InvalidGraph;
    if (state.layers() != g.layers.len) return error.WrongStateLayout;
    for (tokens) |token| if (token >= g.vocab) return error.TokenOutOfRange;
    // Admission is complete before the first operation can queue a write.
    try state.admit(tokens.len);
    errdefer state.poison();
    var h = try o.embed(tokens, g.hidden, g.streams);
    for (g.layers, 0..) |layer, i| {
        if (layer.ple) h = try o.ple(i, state, tokens, h);
        const attention = try hyper(o, g, i, .attention, h, true);
        const branch = switch (layer.kind) {
            .linear_attention => try o.delta(i, state, attention.mixed),
            .indexed_attention => try o.indexed(i, state, attention.mixed),
        };
        h = try o.writeback(h, branch, attention.injection.?);
        const mlp = try hyper(o, g, i, .mlp, h, true);
        const y = switch (layer.feed_forward) {
            .dense => try dense(o, g, i, mlp.mixed),
            .routed => try o.routed(i, mlp.mixed),
        };
        h = try o.writeback(h, y, mlp.injection.?);
        if (@hasDecl(@TypeOf(o.*), "finishLayer")) try o.finishLayer(h);
    }
    const readout = try hyper(o, g, 0, .readout, h, false);
    const logits = try o.head(readout.mixed);
    try state.advanced(tokens);
    return logits;
}

pub const Boundary = enum { attention, mlp, readout };
pub const Projection = enum { down, up, inject, gate, value, output };
fn Mixed(comptime O: type) type {
    return struct { mixed: O.Tensor, injection: ?O.Tensor };
}
fn hyper(o: anytype, g: Geometry, layer: usize, boundary: Boundary, h: anytype, inject: bool) !Mixed(@TypeOf(o.*)) {
    const normed = try o.hcNorm(layer, boundary, h, g.hidden, g.eps);
    const down = try o.hcProject(layer, boundary, .down, normed);
    const low = try o.hcActivate(down, g.streams);
    const up = try o.hcProject(layer, boundary, .up, low);
    const mixed = try o.hcMix(up, normed, g.streams);
    const injection = if (inject) try o.hcInjection(try o.hcProject(layer, boundary, .inject, normed), g.streams) else null;
    return .{ .mixed = mixed, .injection = injection };
}

/// Every dense target/draft row uses the same projections and BF16 activation seam at every prompt cut.
pub fn dense(o: anytype, g: Geometry, layer: usize, x: anytype) !@TypeOf(o.*).Tensor {
    if (g.intermediate == 0) return error.InvalidDenseWidth;
    const gate = try o.denseProject(layer, .gate, x);
    const up = try o.denseProject(layer, .value, x);
    const act = try o.swiglu(gate, up);
    return o.denseProject(layer, .output, act);
}
