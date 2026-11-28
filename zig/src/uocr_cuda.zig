//! Unlimited-OCR-2 on CUDA: the CUDA files beside the Metal family's shared host files.
pub const config = @import("families/uocr/config.zig");
pub const preprocess = @import("families/uocr/preprocess.zig");
pub const prompt = @import("families/uocr/prompt.zig");
pub const resize = @import("families/uocr/resize.zig");
pub const kernels = @import("families/uocr/cuda_kernels.zig");
pub const weights = @import("families/uocr/cuda_weights.zig");
pub const vision = @import("families/uocr/cuda_vision.zig");
pub const decoder = @import("families/uocr/cuda_decoder.zig");
pub const gemm = @import("families/uocr/cuda_gemm.zig");
pub const serve = @import("families/uocr/cuda_serve.zig");
pub const guard = @import("families/uocr/guard.zig");
pub const gate = @import("families/uocr/gate.zig");

test {
    _ = config;
    _ = prompt;
    _ = resize;
    _ = kernels;
    _ = decoder;
    _ = gemm;
    _ = guard;
    _ = gate;
    _ = serve; // its cancellation tests
}
