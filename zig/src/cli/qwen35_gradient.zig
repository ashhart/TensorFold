//! The attention sink needs local steps; other rows retain the original three scales and the same error bound.
pub fn fractions(row: usize) [3]f64 {
    return if (row == 0) .{ 0.00025, 0.0005, 0.001 } else .{ 0.002, 0.004, 0.008 };
}
