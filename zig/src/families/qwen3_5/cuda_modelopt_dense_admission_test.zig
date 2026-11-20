//! The published dense ModelOpt NVFP4 config (nvidia/Qwen3.8-27B-NVFP4) parses for CUDA admission.
const std = @import("std");
const geometry = @import("cuda_geometry.zig");
const a = std.testing.allocator;

test "the published dense ModelOpt NVFP4 config parses with the modelopt kinds flag" {
    const c = @embedFile("cuda_modelopt_nvfp4_config.json");
    const g = try geometry.parse(a, c);
    try std.testing.expect(g.kinds.modelopt);
    try std.testing.expect(!g.bf16);
    try std.testing.expect(!g.exl3);
}
