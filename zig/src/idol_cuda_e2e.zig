//! tf-idol-generate's one module (the family, BiRefNet, core's files by path), the family's `core` too: no file twice.
const core = @import("core/root.zig");

pub const Checkpoint = core.Checkpoint;
pub const checkpoint = core.checkpoint;
pub const safetensors = core.safetensors;
/// The denoiser, both encoders and the decoders (families/idol/cuda.zig).
pub const idol = @import("families/idol/cuda.zig");
/// BiRefNet, the photo's preprocessing and the PNG writer (idol_birefnet_cuda.zig).
pub const birefnet = @import("idol_birefnet_cuda.zig");
