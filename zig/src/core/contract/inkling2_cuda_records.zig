//! Inkling 2's CUDA kernels: their compile records, one file per kernel group.
const attn = @import("inkling2_attn_records.zig");
const ops = @import("inkling2_ops_records.zig");
const experts = @import("inkling2_experts_records.zig");

pub const cuda = attn.cuda ++ ops.cuda ++ experts.cuda;
pub const cuda_132 = attn.cuda_132 ++ ops.cuda_132 ++ experts.cuda_132;
pub const cuda_133 = attn.cuda_133 ++ ops.cuda_133 ++ experts.cuda_133;
