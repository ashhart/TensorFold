//! DeepSeek's complete libraries reuse the nested expert include assembler.
const std = @import("std");
const ks = @import("kernel_sources").dsv41;
const Set = @import("set.zig").Set;
pub fn add(s: *Set) !void {
    inline for (.{ "fp8_mma", "fp8_rows", "fp8_tile", "csa2_attn", "topk512", "glue", "glue_tmp", "fused", "fold", "compress", "mv", "cand", "index", "engram", "hc" }) |name|
        try s.raw("deepseek-v41", name, @field(ks, name));
    try s.raw("deepseek-v41", "experts", try @import("../src/families/deepseek_v41/metal/experts.zig").source(s.a, "", &.{}));
    try s.raw("deepseek-v41", "expert-mma", try @import("../src/families/deepseek_v41/metal/experts_mma.zig").source(s.a, "", &.{}));
    for ([_][2]u32{ .{ 256, 8 }, .{ 128, 4 } }) |shape|
        try s.raw("deepseek-v41", "route", try @import("../src/families/deepseek_v41/metal/pipelines.zig").routeSource(s.a, shape[0], shape[1]));
}
