//! GPU check of Ling MLA around k3_mla: RoPE, head-gate expand, mixer vs mla.zig.
test {
    _ = @import("ling_mla_gpu_kernels.zig");
    _ = @import("ling_mla_gpu_mixer.zig");
    _ = @import("ling_mla_gpu_chunks.zig");
}
