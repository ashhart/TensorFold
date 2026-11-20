//! Inkling 2's CUDA kernels: their compile records, one file per kernel group.
const attn = @import("inkling2_attn_records.zig");

pub const cuda = attn.cuda;
pub const cuda_132 = attn.cuda_132;
pub const cuda_133 = attn.cuda_133;
