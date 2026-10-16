//! What every HIP family's lane backend shares beside the kernels: tokens drawn from logits rows, greedy or keyed.
pub const draw = @import("draw.zig");
pub const sample = @import("sample.zig");

test {
    _ = draw;
    _ = sample;
}
