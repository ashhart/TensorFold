//! Nemotron-H on CUDA in Zig: the Python 0.6.5 engine's layouts, its glue captured (GB10) or ours (every other chip).

pub const Config = @import("config.zig").Config;
pub const Kind = @import("config.zig").Kind;
pub const weights = @import("cuda_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const state = @import("cuda_state.zig");
pub const Forward = @import("cuda_forward.zig").Forward;
pub const Walk = @import("cuda_forward.zig").Walk;
pub const Marks = @import("cuda_forward.zig").Marks;
pub const Class = @import("cuda_forward.zig").Class;
pub const Dump = @import("cuda_dump.zig").Dump;
pub const engine = @import("cuda_engine.zig");
pub const Engine = engine.Engine;
pub const decode = @import("cuda_decode.zig");
pub const Drafter = @import("cuda_drafts.zig").Drafter;
pub const Head = @import("cuda_mtp.zig").Head;
pub const Lanes = @import("cuda_lanes.zig").Cuda;
pub const native = @import("cuda_native.zig");
pub const glue = @import("cuda_glue.zig");
pub const glue_ref = @import("glue_ref.zig");
pub const glue_math = @import("glue_math.zig");
pub const reuse = @import("cuda_reuse.zig");
pub const slide = @import("cuda_slide.zig");
pub const train = @import("cuda_train.zig");
pub const train_ops = @import("cuda_train_ops.zig");
pub const sites = @import("cuda_sites.zig");
pub const slide_dims = @import("slide_dims.zig");

test {
    _ = @import("config.zig");
    _ = @import("draft_ids.zig");
    _ = kernels;
    _ = @import("cuda_triton.zig");
    _ = @import("cuda_sampler.zig");
    _ = @import("cuda_torch_ops.zig");
    _ = decode;
    _ = glue;
    _ = glue_ref;
    _ = glue_math;
    _ = @import("cuda_prompt_grid.zig");
    _ = @import("learner.zig");
    _ = @import("subspace.zig");
    _ = @import("cuda_train_back.zig");
    _ = @import("cuda_learned.zig");
}
