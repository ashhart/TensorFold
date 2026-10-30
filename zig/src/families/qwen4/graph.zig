//! Dense Qwen4 adapts to the shared Flash Next graph without a private scheduler.
const cfg = @import("config.zig");
const shared = @import("hybrid_graph");
pub const Layer = shared.Layer;

pub fn layers(c: *const cfg.Config, out: []Layer) ![]const Layer {
    if (out.len < c.layers) return error.LayerStorageTooSmall;
    for (out[0..c.layers], 0..) |*l, i| l.* = .{
        .kind = switch (c.kinds[i]) {
            .linear_attention => .linear_attention,
            .indexed_attention => .indexed_attention,
        },
        .ple = c.ple[i],
        .feed_forward = .dense,
    };
    return out[0..c.layers];
}
pub fn geometry(c: *const cfg.Config, ls: []const Layer) !shared.Geometry {
    if (ls.len != c.layers) return error.WrongLayerLayout;
    for (ls, 0..) |l, i| {
        if (l.feed_forward != .dense) return error.RoutedQwen4Layer;
        const kind: shared.Kind = if (c.kinds[i] == .linear_attention) .linear_attention else .indexed_attention;
        if (l.kind != kind or l.ple != c.ple[i]) return error.WrongLayerLayout;
    }
    return .{ .hidden = c.hidden, .streams = c.streams, .lowrank = c.lowrank, .vocab = c.vocab, .intermediate = c.intermediate, .eps = c.eps, .layers = ls };
}
pub const forward = shared.forward;
