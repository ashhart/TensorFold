//! Mini Metal shaders for the family and kernel contract, rooted beside nax.h for prompt-header package scope.

pub const attention = @import("inkling2/attention_sources.zig").attention;
pub const prompt_attention = @import("inkling2/attention_sources.zig").prompt_attention;
pub const prompt_attention_pre_m5 = @import("inkling2/attention_sources.zig").prompt_attention_pre_m5;
pub const glue = @import("inkling2/glue_sources.zig").glue;
pub const experts = @import("inkling2/experts_sources.zig").experts;
pub const experts_units = @import("inkling2/experts_sources.zig").experts_units;
pub const loop = @import("inkling2/loop_sources.zig").loop;
pub const runtime = @import("inkling2/runtime_sources.zig").runtime;
pub const inkling2_dspark = @import("inkling2/dspark_sources.zig").dspark;
pub const inkling2_dspark_units = @import("inkling2/dspark_sources.zig").dspark_units;
