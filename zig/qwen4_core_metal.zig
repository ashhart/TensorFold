//! The Metal family's imports alias the enclosing engine's single physical core owner.
pub const replay_glue = @import("owner").flashnext_replay.glue_source;
const owner = @import("owner");
pub const checkpoint = owner.checkpoint_host;
pub const Checkpoint = checkpoint.Checkpoint;
pub const safetensors = owner.safetensors;
pub const quant = owner.quant;
pub const weight_only = owner.weight_only;
pub const weight_only_metal_adapter = owner.weight_only_metal_adapter;
pub const hc_activation = owner.hc_activation;
pub const hc_activation_metal_adapter = owner.hc_activation_metal_adapter;
pub const hc_pointwise = owner.hc_pointwise;
pub const hc_pointwise_metal_adapter = owner.hc_pointwise_metal_adapter;

pub const contract = @import("contract");
pub const attention_ref = owner.attention_ref;
