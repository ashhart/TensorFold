//! GLM-5.3-Flash on CUDA: the family's parts, ported from the Python family's CUDA path.

pub const experts = @import("cuda_experts.zig");
pub const exl3 = @import("cuda_exl3.zig");
pub const vision = @import("cuda_vision.zig");
pub const mm = @import("cuda_mm.zig");
pub const vision_config = @import("vision_config.zig");
pub const image_core = @import("image_core.zig");
/// An image's bytes to the tower's patches, as the Metal family prepares them (core's decoder).
pub const image = image_core.For(@import("core").picture);
pub const glue = @import("cuda_glue.zig");
pub const kda = @import("cuda_kda.zig");
pub const b16 = @import("cuda_b16.zig");
pub const latent = @import("cuda_latent.zig");
pub const sparse = @import("cuda_sparse.zig");
pub const moe = @import("cuda_moe.zig");
pub const kda_block = @import("cuda_kda_block.zig");
pub const dsa_block = @import("cuda_dsa_block.zig");
pub const mlp = @import("cuda_mlp.zig");
pub const shard = @import("cuda_shard.zig");
pub const weights = @import("cuda_weights.zig");
pub const forward = @import("cuda_forward.zig");
pub const relay = @import("cuda_relay.zig");
pub const config = @import("config.zig");
pub const quant = @import("cuda_quant.zig");
pub const q4 = @import("cuda_q4.zig");
pub const ranks = @import("cuda_ranks.zig");
pub const sampler = @import("cuda_sampler.zig");
pub const engine = @import("cuda_engine.zig");
pub const follow = @import("cuda_follow.zig");
pub const lanes = @import("cuda_lanes.zig");
pub const native = @import("cuda_native.zig");

test {
    _ = @import("cuda_load_owner.zig");
    _ = @import("cuda_images_interleave_test.zig");
    _ = experts;
    _ = exl3;
    _ = vision;
    _ = mm;
    _ = image_core;
    _ = glue;
    _ = kda;
    _ = b16;
    _ = latent;
    _ = sparse;
    _ = moe;
    _ = kda_block;
    _ = dsa_block;
    _ = mlp;
    _ = shard;
    _ = weights;
    _ = forward;
    _ = relay;
    _ = quant;
    _ = q4;
    _ = ranks;
    _ = sampler;
    _ = engine;
    _ = follow;
    _ = lanes;
    _ = native;
    _ = @import("cuda_qpack.zig");
    _ = @import("cuda_tile.zig");
    _ = @import("heat"); // upstream's heat bands, run where the GLM ranks use them
}

pub const dflash = @import("dflash.zig");
pub const dflash_candidates = @import("dflash_candidates.zig");
pub const dflash_load_owner = @import("dflash_load_owner.zig");
