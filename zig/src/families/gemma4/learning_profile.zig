//! Writable dense Gemma admission is narrower than serving and uses one shared GeGLU layout on both backends.
const std = @import("std");
const living = @import("living");
pub const Shape = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    heads: usize,
    sliding_kv: usize,
    global_kv: usize,
    vocab: usize = 262144,
    pub fn dim(_: Shape, l: usize) usize {
        return if (l % 6 == 5) 512 else 256;
    }
    pub fn kv(s: Shape, l: usize) usize {
        return if (l % 6 == 5) s.global_kv else s.sliding_kv;
    }
    pub fn from(c: anytype) Shape {
        return .{ .hidden = c.geo.hidden, .intermediate = c.geo.intermediate, .layers = c.geo.layers, .heads = c.geo.query_heads, .sliding_kv = c.geo.sliding_kv_heads, .global_kv = c.geo.global_kv_heads };
    }
};
pub fn check(c: anytype) !void {
    try c.requireDense();
    if (c.widths.embed.bits != 4 or c.widths.embed.group != 64) return error.GemmaLearningEmbeddingFormat;
    for (c.widths.layers[0..c.geo.layers]) |layer| for (layer) |w| if (w.bits != 4 or w.group != 64) return error.GemmaLearningProjectionFormat;
    if (c.geo.layers < 2 or c.geo.layers > 60 or c.geo.hidden % 64 != 0 or c.geo.intermediate % 64 != 0) return error.GemmaLearningGeometry;
}
pub fn layout(s: Shape) living.Layout {
    var all: living.Layers = .empty;
    for (0..s.layers) |l| all.set(l);
    return .fromExperts(s.hidden, s.intermediate, s.layers, all);
}
pub fn family(unified: bool) living.Family {
    return if (unified) living.families.gemma4_unified else living.families.gemma4;
}
