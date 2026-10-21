//! MRL response normalization acts on completed float32 vectors, independently for each input.
const std = @import("std");

pub fn normalize(values: []f32) !void {
    var square: f64 = 0;
    for (values) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteEmbedding;
        const x: f64 = value;
        square += x * x;
    }
    if (square == 0) return error.ZeroEmbeddingNorm;
    const norm = @sqrt(square);
    for (values) |*value| value.* = @floatCast(@as(f64, value.*) / norm);
}
