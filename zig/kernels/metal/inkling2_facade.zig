//! Family exports share the checked shader texts owned by kernel_sources.
const owner = @import("kernel_sources").inkling2;

pub const attention = owner.attention;
pub const prompt_attention = owner.prompt_attention;
pub const prompt_attention_pre_m5 = owner.prompt_attention_pre_m5;
pub const glue = owner.glue;
pub const experts = owner.experts;
pub const experts_mma = owner.experts_mma;
pub const dense_mma = owner.dense_mma;
pub const dense_tiled = owner.dense_tiled;
pub const loop = owner.loop;
pub const loop_mma = owner.loop_mma;
pub const runtime = owner.runtime;
pub const steer = owner.steer;
pub const tree = owner.tree;
pub const inkling2_dspark = owner.inkling2_dspark;
pub const inkling2_dspark_pre_m5 = owner.inkling2_dspark_pre_m5;
pub const vision = owner.vision;
pub const inkling2_dflash2 = owner.inkling2_dflash2;
pub const sample = owner.sample;
