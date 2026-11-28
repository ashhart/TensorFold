//! Every Inkling 2 Mini source control must apply once without a Metal compiler.

test {
    _ = @import("metal_inkling2_roles_controls.zig");
    _ = @import("metal_inkling2_attention_controls.zig");
    _ = @import("metal_inkling2_attention_gqa_controls.zig");
    _ = @import("metal_inkling2_glue_controls.zig");
    _ = @import("metal_inkling2_logits_controls.zig");
    _ = @import("metal_inkling2_experts_controls.zig");
    _ = @import("metal_inkling2_mma_controls.zig");
    _ = @import("metal_inkling2_dense_mma_controls.zig");
    _ = @import("metal_inkling2_dense_tiled_controls.zig");
    _ = @import("metal_inkling2_gates_mma_controls.zig");
    _ = @import("metal_inkling2_loop_controls.zig");
    _ = @import("metal_inkling2_runtime_controls.zig");
    _ = @import("metal_inkling2_steer_controls.zig");
    _ = @import("metal_inkling2_steer_fused_controls.zig");
    _ = @import("metal_inkling2_steer_norm_controls.zig");
    _ = @import("metal_inkling2_dspark_controls.zig");
    _ = @import("metal_inkling2_split_controls.zig");
    _ = @import("metal_inkling2_vision_controls.zig");
    _ = @import("metal_inkling2_tree_controls.zig");
    _ = @import("metal_inkling2_dflash2_controls.zig");
}
