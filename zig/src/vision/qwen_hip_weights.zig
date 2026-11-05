//! The backend-neutral Qwen tower tensor map also feeds the HIP sink, which widens BF16 tensors without loss.
pub const load = @import("qwen_cuda_weights.zig").load;
