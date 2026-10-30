//! The Qwen family uses the shared CUDA BF16 row operator.
pub const Kernels = @import("cuda").qwen35_bf16_rows.Kernels;
