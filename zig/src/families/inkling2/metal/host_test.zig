//! Host-only Inkling 2 Mini Metal launch plans and scalar arithmetic references.

test {
    _ = @import("attention_test.zig");
    _ = @import("glue_test.zig");
    _ = @import("experts_test.zig");
    _ = @import("loop_test.zig");
    _ = @import("runtime_test.zig");
    _ = @import("runtime_kernels_test.zig");
    _ = @import("dspark_kernels_test.zig");
    _ = @import("dspark_fold.zig");
    _ = @import("dspark_weights.zig");
    _ = @import("dspark_state.zig");
    _ = @import("dspark.zig");
    _ = @import("backend.zig");
    _ = @import("serving_test.zig");
}
