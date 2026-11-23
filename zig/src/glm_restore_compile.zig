//! Compile the actual GLM restoration entry, including its default drained pool mover; never run this object.
pub const safetensors = @import("core/safetensors.zig");
pub const checkpoint_host = @import("core/checkpoint.zig");
pub const Checkpoint = checkpoint_host.Checkpoint;
pub const checkpoint_metal = @import("core/checkpoint_metal.zig");
pub const quant = @import("core/quant/quant.zig");
pub const row_projection = @import("core/row_projection.zig");
pub const lane_projection = @import("core/lane_projection.zig");
pub const affine4 = @import("core/affine4.zig");
pub const affine4_lane = @import("core/affine4_lane.zig");
pub const gpu_profile = @import("core/gpu_profile.zig");
pub const shared_attention = @import("core/shared_attention.zig");
pub const bf16_topk = @import("bf16_topk");
pub const bf16_topk_gpu = @import("bf16_topk_gpu");
pub const draft_ops = @import("draft_ops");

const Backend = @import("families/glm/backend.zig").Backend;
const Stream = @import("lanes").Stream;

export fn glmRestoreCompileOnly(b: *Backend, s: *Stream) u8 {
    b.backend().prefill(s) catch return 1;
    return 0;
}
