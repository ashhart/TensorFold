/// quant/mlx_tiles.hpp's StreamSides: up to four products that share x, K, width and group in one launch.
pub const StreamSides = extern struct {
    words: [4]u64 = @splat(0),
    scale: [4]u64 = @splat(0),
    bias: [4]u64 = @splat(0),
    out: [4]u64 = @splat(0),
    n: [4]c_int = @splat(0),
    first: [5]c_int = @splat(0),
    count: c_int = 0,
    out_half: c_int = 0,
    pair_cols: c_int = 0,
    limit: f32 = 0,
};
