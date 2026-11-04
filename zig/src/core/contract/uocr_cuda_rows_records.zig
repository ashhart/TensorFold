//! Unlimited-OCR-2's CUDA rows launches (uocr.cu).
const registry = @import("registry.zig");

/// nvcc 13.3's usage on sm_80, sm_86, sm_89, sm_120 and sm_121.
pub const cuda = [_]registry.CudaRecord{};
