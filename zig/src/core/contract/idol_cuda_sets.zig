//! Every Idol component's CUDA records (nvcc 13.0, each SM), joined in the pipeline's order for registry.zig.
pub const cuda = @import("idol_cuda_records.zig").cuda;

test {
    _ = @import("idol_cuda_records.zig");
}
