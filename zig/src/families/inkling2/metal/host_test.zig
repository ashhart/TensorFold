//! Host-only Inkling 2 Mini Metal launch plans and scalar arithmetic references.

test {
    _ = @import("attention_test.zig");
    _ = @import("dense_tail_test.zig");
    _ = @import("glue_test.zig");
    _ = @import("experts_test.zig");
    _ = @import("loop_test.zig");
    _ = @import("runtime_test.zig");
    _ = @import("runtime_kernels_test.zig");
    _ = @import("steer_test.zig");
    _ = @import("dspark_kernels_test.zig");
    _ = @import("draft_head_q4.zig");
    _ = @import("dspark_fold.zig");
    _ = @import("dspark_weights.zig");
    _ = @import("dspark_state.zig");
    _ = @import("dspark.zig");
    _ = @import("draw.zig");
    _ = @import("flight.zig");
    _ = @import("sample.zig");
    _ = @import("gpu_tokens_test.zig");
    _ = @import("dense_test.zig");
    _ = @import("dflash2_weights.zig");
    _ = @import("dflash2_kernels.zig");
    _ = @import("dflash2.zig");
    _ = @import("backend.zig");
    _ = @import("images.zig");
    _ = @import("vision_image_test.zig");
    _ = @import("tree_test.zig");
    _ = @import("tfq3.zig");
    _ = @import("tfq3_test.zig");
}
