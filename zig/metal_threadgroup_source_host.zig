//! Complete Metal source assembly and reservation checks; no device or Apple SDK.
test {
    _ = @import("kernels/metal/threadgroup_source_test.zig");
}
