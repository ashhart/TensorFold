//! Minimal shader-free public core surface for the Qwen4 host fixtures.
pub const safetensors = @import("src/core/safetensors.zig");
pub const checkpoint = @import("src/core/checkpoint.zig");
pub const Checkpoint = checkpoint.Checkpoint;
pub const quant = @import("src/core/quant/quant.zig");
pub const replay_glue = @import("src/families/flashnext/replay_glue_source.zig").text;
pub const api = @import("engine_api");
pub const lanes = @import("lanes");
pub const lanes_sampling = lanes.sampling;
pub const lanes_gpu_rule = lanes.gpu_rule;
pub const weight_only = @import("src/core/weight_only.zig");
pub const contract = @import("src/core/contract.zig");
pub const weight_only_contract = @import("src/core/contract/weight_only.zig");
pub const weight_only_cuda_adapter = @import("src/core/weight_only_cuda_impl.zig").Adapter;
pub const weight_only_metal_adapter = @import("src/core/weight_only_metal_impl.zig").Adapter;
pub const hc_activation = @import("src/core/hc_activation.zig");
pub const hc_activation_contract = @import("src/core/contract/hc_activation.zig");
pub const hc_activation_cuda_adapter = @import("src/core/hc_activation_cuda_impl.zig").Adapter;
pub const hc_activation_metal_adapter = @import("src/core/hc_activation_metal_impl.zig").Adapter;
pub const hc_pointwise = @import("src/core/hc_pointwise.zig");
pub const hc_pointwise_cuda_adapter = @import("src/core/hc_pointwise_cuda_impl.zig").Adapter;
pub const hc_pointwise_metal_adapter = @import("src/core/hc_pointwise_metal_impl.zig").Adapter;
pub const hc_pointwise_contract = @import("src/core/contract/hc_pointwise.zig");

pub const attention_ref = @import("src/core/attention_ref.zig");
