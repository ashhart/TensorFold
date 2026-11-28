//! Mini Metal shaders for the family and kernel contract, rooted beside nax.h for prompt-header package scope.

pub const attention = @import("inkling2/attention_sources.zig").attention;
pub const prompt_attention = @import("inkling2/attention_sources.zig").prompt_attention;
pub const prompt_attention_pre_m5 = @import("inkling2/attention_sources.zig").prompt_attention_pre_m5;
pub const glue = @import("inkling2/glue_sources.zig").glue;
pub const affine4 = @import("inkling2/affine4_sources.zig").affine4;
pub const experts = @import("inkling2/experts_sources.zig").experts;
pub const experts_mma = @import("inkling2/experts_sources.zig").experts_mma;
pub const dense_mma = @import("inkling2/dense_sources.zig").dense_mma;
pub const dense_tiled = @import("inkling2/dense_sources.zig").dense_tiled;
pub const loop = @import("inkling2/loop_sources.zig").loop;
pub const loop_mma = @import("inkling2/loop_sources.zig").loop_mma;
pub const runtime = @import("inkling2/runtime_sources.zig").runtime;
pub const tree = @import("inkling2/tree_sources.zig").tree;
pub const inkling2_dspark = @import("inkling2/dspark_sources.zig").dspark;
pub const inkling2_dspark_pre_m5 = @import("inkling2/dspark_sources.zig").dspark_pre_m5;
pub const vision = @import("inkling2/vision_sources.zig").vision;
pub const inkling2_dflash2 = @import("inkling2/dflash2_sources.zig").dflash2;
pub const sample = @import("inkling2/sample_sources.zig").sample;
