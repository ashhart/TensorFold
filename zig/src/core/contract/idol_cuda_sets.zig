//! Every Idol component's CUDA records (nvcc 13.0, each SM), joined in the pipeline's order for registry.zig.
pub const cuda = @import("idol_cuda_records.zig").cuda ++ @import("idol_dino_cuda_records.zig").cuda ++ @import("idol_vae_cuda_records.zig").cuda ++ @import("idol_decoder_cuda_records.zig").cuda ++ @import("idol_birefnet_cuda_records.zig").cuda;

test {
    _ = @import("idol_cuda_records.zig");
    _ = @import("idol_dino_cuda_records.zig");
    _ = @import("idol_vae_cuda_records.zig");
    _ = @import("idol_decoder_cuda_records.zig");
    _ = @import("idol_birefnet_cuda_records.zig");
}
