//! Idol's background removal on CUDA; rooted here so preprocess.zig reaches core/picture as on Metal.
pub const config = @import("families/idol/birefnet_config.zig");
pub const preprocess = @import("families/idol/preprocess.zig");
pub const reference = @import("families/idol/birefnet_reference.zig");
pub const safetensors = @import("core/safetensors.zig");
pub const kernels = @import("families/idol/cuda_birefnet_kernels.zig");
pub const weights = @import("families/idol/cuda_birefnet_weights.zig");
pub const decoder = @import("families/idol/cuda_birefnet_decoder.zig");
pub const net = @import("families/idol/cuda_birefnet.zig");
pub const Net = net.Net;
pub const Timing = net.Timing;
pub const seconds = net.seconds;
pub const admitted = net.admitted;

test {
    _ = config;
    _ = preprocess;
    _ = reference;
    _ = kernels;
    _ = weights;
    _ = decoder;
    _ = net;
}
