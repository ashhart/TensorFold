//! Standalone host entry for the actual affine reader, expert stack and bit-preserving slicing regressions.
test {
    _ = @import("quant/mlx.zig");
    _ = @import("quant/experts.zig");
    _ = @import("quant/exact_test.zig");
}
