//! GLM's MTP draft KV capacity, from the CUDA latent cache's loaded dimensions.

pub fn mtpKv(capacity: usize, width: usize, kv8: bool) @import("lanes").config.DraftKv {
    return .{
        .bytes = @import("cuda_latent.zig").cacheBytes(capacity, width, kv8) catch unreachable,
        .dtype = if (kv8) .fp8 else .bf16,
        .scope = .stream_limit,
    };
}
