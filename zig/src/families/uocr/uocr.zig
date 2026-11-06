//! Unlimited-OCR-2 (baidu/Unlimited-OCR-2) on Metal: document pages to text, the release's bf16 weights as published.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const vision = @import("vision.zig");
pub const resize = @import("resize.zig");
pub const gate = @import("gate.zig");
pub const preprocess = @import("preprocess.zig");
/// The page steps over core/picture (decode, letterbox, tiles).
pub const pages = preprocess.With(@import("../../core/picture/picture.zig"));
pub const prompt = @import("prompt.zig");
pub const decoder = @import("decoder.zig");
pub const generate = @import("generate.zig");
pub const serve = @import("serve.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
