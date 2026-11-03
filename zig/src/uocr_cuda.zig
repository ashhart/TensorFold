//! Unlimited-OCR-2 (baidu/Unlimited-OCR-2) on CUDA: document pages to text, the release's bf16 weights as published.
//! Its host files (geometry, page preprocessing, prompt positions, the page guard) are the Metal family's.
pub const config = @import("families/uocr/config.zig");
pub const preprocess = @import("families/uocr/preprocess.zig");
pub const prompt = @import("families/uocr/prompt.zig");
pub const resize = @import("families/uocr/resize.zig");
pub const kernels = @import("families/uocr/cuda_kernels.zig");
pub const weights = @import("families/uocr/cuda_weights.zig");
pub const vision = @import("families/uocr/cuda_vision.zig");
pub const decoder = @import("families/uocr/cuda_decoder.zig");
pub const serve = @import("families/uocr/cuda_serve.zig");

test {
    _ = config;
    _ = prompt;
    _ = resize;
    _ = kernels;
    _ = @import("families/uocr/guard.zig");
}
