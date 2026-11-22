//! Every Inkling 2 Mini source control must apply once without a Metal compiler.

test {
    _ = @import("metal_inkling2_attention_controls.zig");
    _ = @import("metal_inkling2_glue_controls.zig");
    _ = @import("metal_inkling2_experts_controls.zig");
    _ = @import("metal_inkling2_mma_controls.zig");
    _ = @import("metal_inkling2_dense_mma_controls.zig");
    _ = @import("metal_inkling2_gates_mma_controls.zig");
    _ = @import("metal_inkling2_loop_controls.zig");
    _ = @import("metal_inkling2_runtime_controls.zig");
    _ = @import("metal_inkling2_dspark_controls.zig");
    _ = @import("metal_inkling2_dspark_chain_controls.zig");
    _ = @import("metal_inkling2_split_controls.zig");
    _ = @import("metal_inkling2_vision_controls.zig");
}
