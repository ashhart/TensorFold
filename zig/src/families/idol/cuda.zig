//! Idol's denoiser and decoders on CUDA: driven by tf-idol and tf-idol-decode beside the lane engine, as on Metal.
pub const config = @import("config.zig");
pub const host = @import("host.zig");
pub const reference = @import("reference.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const Model = @import("cuda_model.zig").Model;
pub const Condition = @import("cuda_model.zig").Condition;
pub const Timing = @import("cuda_model.zig").Timing;
pub const seconds = @import("cuda_model.zig").seconds;
pub const admitted = @import("cuda_model.zig").admitted;
pub const decoder_config = @import("decoder_config.zig");
pub const decoder_kernels = @import("cuda_decoder_kernels.zig");
pub const octree = @import("octree.zig");
pub const gaussian = @import("gaussian.zig");
pub const decode = @import("decode.zig");
pub const decode_reference = @import("decode_reference.zig");
pub const Decoder = @import("cuda_decoder.zig").Decoder;

test {
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_gemm.zig");
    _ = @import("cuda_model.zig");
    _ = @import("cuda_weights.zig");
    _ = @import("cuda_decoder.zig");
    _ = @import("cuda_decoder_kernels.zig");
    _ = @import("cuda_decoder_weights.zig");
    _ = decoder_config;
    _ = octree;
    _ = gaussian;
    _ = decode;
    _ = decode_reference;
    _ = config;
    _ = host;
    _ = reference;
}
