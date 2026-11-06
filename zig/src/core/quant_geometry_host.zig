//! Standalone host entry for the actual affine reader, dense stored dtypes, expert stack and bit-exact slicing.
test {
    _ = @import("quant/mlx.zig");
    _ = @import("quant/experts.zig");
    _ = @import("quant/exact_test.zig");
    _ = @import("quant/dense.zig");
}
