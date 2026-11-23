//! FP64 host primitives: causal bounds and a learned sink logit with zero value. No GPU execution path.
const std = @import("std");
const config = @import("config.zig");
pub const Bounds = struct { begin: usize, end: usize };

/// Reference convention: a sliding window includes the current token; end is exclusive. Checkpoint fidelity is open.
pub fn bounds(c: *const config.Config, layer: usize, position: usize) !Bounds {
    if (position >= c.context) return error.BadPosition;
    const end = position + 1;
    return .{ .begin = if (try c.kind(layer) == .sliding) end -| c.window else 0, .end = end };
}

/// Explicit denominator-only sink convention; scores are already scaled/rotated, values are [keys,width].
pub fn combine(scores: []const f64, values: []const f64, width: usize, sink: f64, out: []f64) !void {
    const n = std.math.mul(usize, scores.len, width) catch return error.BadAttentionShape;
    if (width == 0 or out.len != width or values.len != n) return error.BadAttentionShape;
    if (!std.math.isFinite(sink) and sink != -std.math.inf(f64)) return error.NonfiniteAttention;
    if (scores.len == 0 and sink == -std.math.inf(f64)) return error.EmptyAttention;
    var maximum = sink;
    for (scores) |s| {
        if (!std.math.isFinite(s)) return error.NonfiniteAttention;
        maximum = @max(maximum, s);
    }
    for (values) |v| if (!std.math.isFinite(v)) return error.NonfiniteAttention;
    var denominator: f64 = @exp(sink - maximum);
    for (scores) |s| denominator += @exp(s - maximum);
    for (out, 0..) |*value, column| {
        var sum: f64 = 0;
        for (scores, 0..) |s, row_index| sum += @exp(s - maximum) * values[row_index * width + column];
        value.* = sum / denominator;
    }
}

/// Remove future and out-of-window keys before reading their logits or values, including nonfinite masked inputs.
pub fn row(c: *const config.Config, layer: usize, position: usize, scores: []const f64, values: []const f64, width: usize, sink: f64, out: []f64) !void {
    const visible = try bounds(c, layer, position);
    const n = std.math.mul(usize, scores.len, width) catch return error.BadAttentionShape;
    if (visible.end > scores.len or values.len != n) return error.BadAttentionShape;
    return combine(scores[visible.begin..visible.end], values[visible.begin * width .. visible.end * width], width, sink, out);
}

/// One already-rotated query and its selected KV head, using FP64 dots and 1/sqrt(head_dim); no RoPE policy implied.
pub fn reference(a: std.mem.Allocator, c: *const config.Config, layer: usize, position: usize, query: []const f64, keys: []const f64, values: []const f64, sink: f64, out: []f64) !void {
    const width = c.head_dim;
    if (width == 0 or query.len != width or out.len != width or keys.len % width != 0 or values.len != keys.len) return error.BadAttentionShape;
    const visible = try bounds(c, layer, position);
    if (visible.end > keys.len / width) return error.BadAttentionShape;
    for (query) |v| if (!std.math.isFinite(v)) return error.NonfiniteAttention;
    const scores = try a.alloc(f64, visible.end - visible.begin);
    defer a.free(scores);
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(width)));
    for (scores, visible.begin..) |*score, index| {
        var sum: f64 = 0;
        for (query, keys[index * width ..][0..width]) |q, k| sum += q * k;
        score.* = sum * scale;
    }
    return combine(scores, values[visible.begin * width .. visible.end * width], width, sink, out);
}
