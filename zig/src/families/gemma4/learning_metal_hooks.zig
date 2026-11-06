//! Concrete Metal graph sites describe borrowed views; only an installed observer copies them to a native tape.
const metal = @import("metal");
const hook = @import("learning_hook.zig");
pub fn begin(m: anytype, segments: anytype, rows: usize) !void {
    const h = m.learning orelse return;
    var runs: [128]hook.Run = undefined;
    var at: usize = 0;
    if (segments.len > runs.len) return error.GemmaLearningRows;
    for (segments, 0..) |seg, i| {
        runs[i] = .{ .first = at, .rows = seg.rows, .position = seg.cache.len, .untaught = seg.cache.untaught };
        at += seg.rows;
    }
    try h.begin(h.ctx, runs[0..segments.len], rows);
}
pub fn one(m: anytype, cache: anytype, rows: usize) !void {
    if (m.learning) |h| try h.begin(h.ctx, &.{.{ .first = 0, .rows = rows, .position = cache.len, .untaught = cache.untaught }}, rows);
}
pub fn active(m: anytype) bool {
    return if (m.learning) |h| h.active(h.ctx) else false;
}
pub fn view(b: metal.Buffer, rows: usize, width: usize) hook.View(metal.Buffer) {
    return .{ .buffer = b, .len = rows * width * 2, .rows = rows, .width = width };
}
pub fn note(m: anytype, e: metal.ComputeEncoder, l: ?usize, node: hook.Node, b: metal.Buffer, rows: usize, width: usize) !void {
    if (m.learning) |h| try h.note(e, l, node, view(b, rows, width));
}
pub fn columns(m: anytype, e: metal.ComputeEncoder, l: usize, node: hook.Node, b: metal.Buffer, rows: usize, width: usize, stride: usize, offset: usize, interleaved: bool) !void {
    if (m.learning) |h| {
        var v = view(b, rows, width);
        v.offset = offset * 2;
        v.len = rows * stride * 2 - v.offset;
        v.stride = stride;
        v.layout = if (interleaved) .interleaved else .rows;
        try h.note(e, l, node, v);
    }
}
pub fn heads(m: anytype, e: metal.ComputeEncoder, l: usize, node: hook.Node, b: metal.Buffer, rows: usize, count: usize, dim: usize, offset: usize) !void {
    if (m.learning) |h| {
        var v = view(b, rows, count * dim);
        v.offset = offset;
        v.heads = count;
        v.dim = dim;
        v.layout = .heads;
        try h.note(e, l, node, v);
    }
}
pub fn dense(m: anytype, e: metal.ComputeEncoder, l: usize, input: metal.Buffer, output: metal.Buffer, rows: usize) !void {
    if (m.learning) |h| try h.dense(h.ctx, e, l, view(input, rows, m.config.geo.hidden), view(output, rows, m.config.geo.hidden));
}
