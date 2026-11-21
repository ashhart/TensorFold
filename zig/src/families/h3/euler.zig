//! The reference's Euler step over latent rows, shared by tf-h3-dit (Metal) and tf-h3-dit-cuda.

/// One Euler step as the reference takes it: x0 from the sigma the model saw, then the grid's ratio.
pub fn advance(x: []f32, velocity: []const f32, seen: f32, ratio: f32) void {
    for (x, velocity) |*o, v| o.* = ratio * o.* + (1.0 - ratio) * (o.* + seen * v);
}
