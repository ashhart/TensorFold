//! The Metal family's imports alias the enclosing engine's single physical core owner.
pub const replay_glue = @import("owner").flashnext_replay.glue_source;
const owner = @import("owner");
pub const checkpoint = owner.checkpoint_host;
pub const Checkpoint = checkpoint.Checkpoint;
pub const safetensors = owner.safetensors;
pub const shard_edit = owner.shard_edit;
pub const quant = owner.quant;
pub const weight_only = owner.weight_only;
pub const living_rows = owner.living_rows;
pub const living_gdn = owner.living_gdn;
pub const living_qsa = owner.living_qsa;
pub const weight_only_metal_adapter = owner.weight_only_metal_adapter;
pub const hc_activation = owner.hc_activation;
pub const hc_activation_metal_adapter = owner.hc_activation_metal_adapter;
pub const hc_pointwise = owner.hc_pointwise;
pub const hc_pointwise_metal_adapter = owner.hc_pointwise_metal_adapter;

pub const contract = @import("contract");
pub const attention_ref = owner.attention_ref;

pub const living_rows_cuda = owner.living_rows_cuda;
pub const living_rows_metal = owner.living_rows_metal;

pub const living_gemma = owner.living_gemma;

pub const dense_storage_cuda = owner.dense_storage_cuda;
pub const dense_storage_metal = owner.dense_storage_metal;
